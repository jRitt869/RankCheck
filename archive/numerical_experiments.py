import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, jacrev, vmap
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings(
    "ignore", message="Attempting to run cuBLAS, but there was no current CUDA context"
)

class NeuralVectorField(nn.Module):
    """Simple fully connected neural network used as a vector field."""
    def __init__(self, input_dim, depth, latent_dim, activation=True):
        super().__init__()
        layers = [nn.Linear(input_dim, latent_dim)]
        for _ in range(depth - 1):
            layers.append(nn.Linear(latent_dim, latent_dim))
        layers.append(nn.Linear(latent_dim, input_dim))
        self.net = nn.ModuleList(layers)
        self.relu = activation

    def forward(self, x):
        for i, layer in enumerate(self.net):
            if i < len(self.net) - 1:
                x = F.relu(layer(x)) if self.relu else F.gelu(layer(x))
            else:
                x = layer(x)
        return x

def solve_ode_fixed_step(func, p, x0, t_end=1.0, steps=20):
    """Solve an ODE with explicit Euler integration on a fixed time grid."""
    dt = torch.tensor(t_end / steps, device=x0.device, dtype=torch.float64)
    x = x0
    pts = []
    for _ in range(steps):
        pts.append(x)   # Store point where the vector field is evaluated
        x = x + dt * func(p, x)
    return x, torch.stack(pts)

def sample_uniform_ball(num_samples, n, rho, device="cpu", dtype=torch.float64):
    """Sample points uniformly from the n-dimensional ball of radius `rho`."""
    u = torch.randn(num_samples, n, device=device, dtype=dtype)
    u = u / torch.norm(u, dim=1, keepdim=True)
    r = torch.rand(num_samples, 1, device=device, dtype=dtype).pow(1 / n) * rho
    return u * r

def compute_jac(compiled_var_step, model, params, x_0, t_end=1.0, steps=4000, N=2):
    """Compute a projected parameter Jacobian for the flow map segment by segment."""
    batch_size, state_dim = x_0.shape
    dt = torch.tensor(t_end / steps, device=x_0.device)
    steps_per_segment = steps // N
    num_params = sum(p.numel() for p in params.values())

    # -----------------------------------------------------
    # Step A: Compute the forward trajectory
    # -----------------------------------------------------
    # We only need to save the states at the start of each of the N segments
    segment_start_states = torch.zeros((N, batch_size, state_dim), device=x_0.device)
    x = x_0.clone()

    with torch.no_grad():
        for seg in range(N):
            segment_start_states[seg] = x
            for _ in range(steps_per_segment):
                x = x + dt * functional_call(model, params, x)

    # -----------------------------------------------------
    # Step B: Integrate the variational equations in parallel
    # -----------------------------------------------------
    # Initialize Jacobians for each segment
    # dφ/dx is Identity, dφ/dp is Zero
    J_x = torch.eye(state_dim, device=x_0.device).expand(N, batch_size, state_dim, state_dim).clone()
    J_p = torch.zeros((N, batch_size, state_dim, num_params), device=x_0.device)
    
    x_var = segment_start_states.clone()    # We must re-simulate the states while simultaneously pushing the Jacobians

    for _ in range(steps_per_segment):
        with torch.no_grad():
            torch.compiler.cudagraph_mark_step_begin()
            J_x, J_p = compiled_var_step(x_var.clone(), J_x.clone(), J_p.clone())

        # Advance x_var for the next step calculation
        batched_f = vmap(vmap(lambda x_: functional_call(model, params, x_)))
        x_var = x_var + dt * batched_f(x_var)

    # -----------------------------------------------------
    # Step D: Composition of the segment Jacobians
    # -----------------------------------------------------
    # J_x[seg] is now dφ(T/N)/dx evaluated at segment_start_states[seg]
    # J_p[seg] is now dφ(T/N)/dp evaluated at segment_start_states[seg]
    
    # Apply the chain rule iteratively from left to right (t=0 to t=T)
    # Total Jacobian up to segment i
    Total_J_p = J_p[0]
    for seg in range(1, N):
        # Chain rule: dφ/dp(T_n) = dφ/dp(T_current) + dφ/dx(T_current) * Total_J_p(previous)
        Total_J_p = J_p[seg] + torch.bmm(J_x[seg], Total_J_p)

    return torch.flatten(Total_J_p, start_dim=0, end_dim=1)

def compute_n_ball_volume(rho_val, n_dim):
    """Compute the volume of an n-dimensional ball of radius `rho_val`."""
    if n_dim % 2 == 0:
        k = n_dim / 2
        return torch.pi**k / torch.prod(torch.arange(1, k + 1)) * rho_val**n_dim
    else:
        k = (n_dim - 1) / 2
        return (2 ** (k + 1) * torch.pi**k / torch.prod(2 * torch.arange(1, k + 1) + 1) * rho_val**n_dim)

def eval_funcs(model):
    """Build compiled helper functions for variational dynamics and parameter Jacobians."""
    # -----------------------------------------------------
    # Define compiled Variational Equation steps
    # -----------------------------------------------------
    # Define how to push the sensitivity matrices (Jacobians) forward by ONE Euler step
    def euler_variational_step(x_curr, J_x, J_p, params, dt):
        df_dx = jacrev(lambda x_: functional_call(model, params, x_))(x_curr)
        df_dp_dict = jacrev(lambda p: functional_call(model, p, x_curr))(params)
        df_dp = torch.cat([v.flatten(1) for v in df_dp_dict.values()], dim=-1)

        next_J_x = J_x + dt * df_dx @ J_x
        next_J_p = J_p + dt * (df_dx @ J_p + df_dp)
        return next_J_x, next_J_p

    batched_var_step = vmap(
        vmap(euler_variational_step, in_dims=(0, 0, 0, None, None)),
        in_dims=(0, 0, 0, None, None),
    )
    compiled_var_step = torch.compile(batched_var_step, backend="cudagraphs", fullgraph=True)

    def ff_func(x, params):
        df_dp_dict = jacrev(lambda p: functional_call(model, p, x))(params)
        return torch.cat([v.flatten(1) for v in df_dp_dict.values()], dim=-1)

    return compiled_var_step, vmap(ff_func, in_dims=(0, None))

def numerical_rank_from_svals(svals, rtol=1e-12, atol=1e-10):
    """Estimate numerical rank from singular values using mixed tolerance."""
    thr = max(atol, float(rtol * svals.max().item()))
    return int((svals > thr).sum().item())

def compute_percentile_stats(line_plots):
    """
    Computes the mean and standard deviation of the 10th-100th percentiles 
    across multiple line plots.
    """
    percentiles = np.arange(0, 110, 10)
    all_px, all_py = [], []

    for plot in line_plots:
        all_px.append(np.percentile(plot["x"], percentiles))
        all_py.append(np.flip(np.percentile(np.log(plot["y"]), percentiles)))

    all_px, all_py = np.array(all_px), np.array(all_py)
    return np.mean(all_px, axis=0), np.std(all_px, axis=0), np.mean(all_py, axis=0), np.std(all_py, axis=0)

def get_parameter_initialization_scale(key, params_ref):
    """Compute the sampling half-width used for a given parameter tensor."""
    parts = key.split(".")
    p = params_ref[key] if parts[2] == "weight" else params_ref[f"net.{parts[1]}.weight"]
    return torch.sqrt(1 / torch.tensor(p.shape[1], dtype=torch.float64))

def generate_random_parameters(model, iterations, device):
    """Generate batched random parameter samples for Monte Carlo iterations."""
    params0 = {k: p.detach().clone() for k, p in model.named_parameters()}
    return {
        k: (torch.rand((iterations, *p.shape), device=device, dtype=torch.float64) - 0.5) 
           * (2 * get_parameter_initialization_scale(k, params0))
        for k, p in params0.items()
    }

def run_monte_carlo_experiment():
    """Execute the main sampling loop to estimate Jacobian singular values."""
    torch.set_default_dtype(torch.float64)

    compute_aug = True          # augmented Neural ODE
    compute_gelu = False        # GELU activation function

    # Settings
    n_dim = 2                   # input dimension
    rho_val = 50                # radius of input domain
    depth_val = 3               # depth of Neural Network
    latent_val = 3              # latent dimensions
    num_samples_ff = 800000     # Monte Carlo samples for Feedforward Nerual Network
    num_samples = 40000         # Monte Carlo samples for Neural ODE

    t_end = 1.0                 # terminal time
    steps = 500                 # number of integration steps
    iterations = 600            # number of parameter samples

    vol = compute_n_ball_volume(rho_val, n_dim)
    vol_aug = compute_n_ball_volume(rho_val, n_dim - 1) if compute_aug else None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    dt = torch.tensor(t_end / steps, device=device, dtype=torch.float64)
    model = NeuralVectorField(n_dim, depth_val, latent_val, activation=True).to(device)
    if compute_gelu:
        modelGELU = NeuralVectorField(n_dim, depth_val, latent_val, activation=False).to(device)

    # Choose Mote Carlo samples
    samples = sample_uniform_ball(num_samples, n_dim, rho_val, device=device)
    samples_ff = sample_uniform_ball(num_samples_ff, n_dim, rho_val * 2, device=device) # larger input space for computation of rank bound 

    if compute_aug:
        samples_aug = sample_uniform_ball(num_samples, n_dim - 1, rho_val, device=device)
        # embedding the samples into R^n_dim
        samples_aug = torch.cat([samples_aug, torch.zeros([samples_aug.shape[0], 1], device=device, dtype=torch.float64)], dim=1)

    compiled_var_step, ff_func = eval_funcs(model)
    if compute_gelu:
        compiled_var_step_gelu, ff_func_gelu = eval_funcs(modelGELU)
    sVals_flow, sVals_aug, sRanks_ff = [], [], []
    sVals_flow_gelu, sVals_aug_gelu, sRanks_ff_gelu = [], [], []

    params_batch = generate_random_parameters(model, iterations, device)

    for i in range(iterations):
        params = {k: p[i] for k, p in params_batch.items()}     # sample random parameters

        # compute singular values of Jacobian instead of its Gram matrix for higher precision
        J_flow = compute_jac(lambda x, jx, jp: compiled_var_step(x, jx, jp, params, dt), 
                             model, params, samples, t_end=t_end, steps=steps, N=10)
        s_flow = torch.linalg.svdvals(J_flow / torch.sqrt(torch.tensor(num_samples) / vol))
        
        # remove numerical noise
        thr = max(1e-10, float(1e-12 * s_flow.max().item()))
        sVals_flow.append((s_flow * (s_flow > thr)).detach().cpu().numpy())

        if compute_aug:
            J_aug = compute_jac(lambda x, jx, jp: compiled_var_step(x, jx, jp, params, dt), 
                                model, params, samples_aug, t_end=t_end, steps=steps, N=10)
            s_aug = torch.linalg.svdvals(J_aug / torch.sqrt(torch.tensor(num_samples) / vol_aug))
            thr = max(1e-10, float(1e-12 * s_aug.max().item()))
            sVals_aug.append((s_aug * (s_aug > thr)).detach().cpu().numpy())

        if compute_gelu:
            J_flow_gelu = compute_jac(lambda x_curr, J_x, J_p: compiled_var_step_gelu(x_curr,J_x,J_p,params,dt), modelGELU, params, samples, t_end=t_end, steps=steps, N=10)
            s_flow_gelu = torch.linalg.svdvals(J_flow_gelu / torch.sqrt(torch.tensor(num_samples) / vol))
            sVals_flow_gelu.append(s_flow_gelu.detach().cpu().numpy())
            
            if compute_aug: 
                J_aug_gelu = compute_jac(lambda x_curr, J_x, J_p: compiled_var_step_gelu(x_curr,J_x,J_p,params,dt), modelGELU, params, samples_aug, t_end=t_end, steps=steps, N=10)
                s_aug_gelu = torch.linalg.svdvals(J_aug_gelu / torch.sqrt(torch.tensor(num_samples) / vol_aug))
                sVals_aug_gelu.append(s_aug_gelu.detach().cpu().numpy())

        torch.compiler.cudagraph_mark_step_begin()

        # computation of the rank bound using the rank of the jacobian J_ff
        J_ff = torch.flatten(ff_func(samples_ff, params), start_dim=0, end_dim=1)
        s_ff = torch.linalg.svdvals(J_ff / torch.sqrt(torch.tensor(num_samples_ff) / (2**n_dim * vol)))
        sRanks_ff.append(numerical_rank_from_svals(s_ff))

        if compute_gelu:
            J_ff_gelu = ff_func_gelu(samples_ff, params)
            J_ff_gelu = torch.flatten(J_ff_gelu, start_dim=0, end_dim=1)
            s_ff_gelu = torch.linalg.svdvals(J_ff_gelu / torch.sqrt(torch.tensor(num_samples_ff) / (2**n_dim * vol)))
            sRanks_ff_gelu.append(numerical_rank_from_svals(s_ff_gelu))

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        torch._dynamo.reset()
        print(f"step : {i+1} / {iterations}")

    return sVals_flow, sVals_aug, sRanks_ff, sVals_flow_gelu, sVals_aug_gelu, sRanks_ff_gelu

def save_results(sVals_flow, sVals_aug, sRanks_ff):
    """Aggregate collected iterations for the ReLU case and export summary metrics to CSV files."""
    flow_data = np.array(sVals_flow)
    ff_data = np.array(sRanks_ff)
    aug_data = np.array(sVals_aug) if sVals_aug else None

    unique_ranks = np.unique(ff_data)
    data_ranks = {r: [] for r in unique_ranks}
    data_ranks_aug = {r: [] for r in unique_ranks}

    all_flow, all_aug, ranks_aug, ranks_ff = [], [], [], []

    for i in range(len(ff_data)):
        d = flow_data[i]
        rank_flow = sum(d > 0)
        data_ranks[ff_data[i]].append({"x": np.arange(rank_flow) + 1, "y": d[d > 0]})
        ranks_ff.append(ff_data[i])
        all_flow.append({"x": np.arange(rank_flow) + 1, "y": d[d > 0]})

        if aug_data is not None:
            d_aug = aug_data[i]
            rank_aug = sum(d_aug > 0)
            data_ranks_aug[ff_data[i]].append({"x": np.arange(rank_aug) + 1, "y": d_aug[d_aug > 0]})
            ranks_aug.append(rank_aug)
            all_aug.append({"x": np.arange(rank_aug) + 1, "y": d_aug[d_aug > 0]})

    if ranks_aug:
        pd.DataFrame(np.transpose([ranks_aug, ranks_ff]), columns=["rank_aug", "rank_ff"]).to_csv("Ranks.csv", index=False) # 2d Histogram data 

    out = []
    mean_x, _, mean_y, _ = compute_percentile_stats(all_flow)
    out.extend([mean_x, mean_y])
    columns = ["x_flow", "y_flow"]

    if all_aug:
        mean_x_aug, _, mean_y_aug, _ = compute_percentile_stats(all_aug)
        out.extend([mean_x_aug, mean_y_aug])
        columns.extend(["x_aug", "y_aug"])

    pd.DataFrame(np.transpose(out), columns=columns).to_csv("plot.csv", index=False)

if __name__ == "__main__":
    flow, aug, ff, _, _, _ = run_monte_carlo_experiment()
    save_results(flow, aug, ff)
