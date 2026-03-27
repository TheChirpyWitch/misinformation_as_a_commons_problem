# Trust-as-a-Commons ABM

An agent-based model (ABM) of misinformation spread and trust dynamics on a social network, implemented in Python using the [Mesa](https://mesa.readthedocs.io/) framework.

Agents allocate attention between credible and misinformation sources. Their credibility beliefs evolve via bounded-confidence Deffuant dynamics, while global trust evolves as a common-pool resource depleted by misinformation and replenished by credible engagement. The network can rewire adaptively (homophily) or randomly.

---

## Installation

```bash
pip install -r requirements.txt
```

**Dependencies:** `mesa==1.2.1`, `networkx`, `numpy`, `pandas`, `matplotlib`, `tqdm`, `SALib`

Python 3.9+ recommended.

---

## Project Structure

```
.
├── params.py          # TrustCommonsParams dataclass + price/attention utility functions
├── agent.py           # TrustCommonsAgent — credibility, preference, attention
├── model.py           # TrustCommonsABM  — full model with cascade propagation
├── experiment.py      # run_experiment(), run_experiment_with_spatial_metrics()
├── visualization.py   # all plotting functions + data normalization helpers
├── regimes.py         # regime initializers (init_*) and convenience runners (run_*)
├── analysis.py        # shock experiments, stress tests, sensitivity sweeps
├── metrics.py         # Moran's I, assortativity, correlation series
├── tests.py           # unit tests
└── main.py            # runs all experiments end-to-end (mirrors notebook execution)
```

---

## Quick Start

### Minimal run

```python
from params import TrustCommonsParams
from experiment import run_experiment
from visualization import plot_global_trajectories

params = TrustCommonsParams(N=200, random_seed=42)
model, model_df, agent_df = run_experiment(steps=500, params=params)
plot_global_trajectories(model_df)
```

`model_df` is a pandas DataFrame with one row per step and columns:
`T`, `mean_c`, `var_T_local`, `mean_eta`, `g_sum`, `m_sum`.

`agent_df` is a MultiIndex DataFrame `(Step, AgentID)` with columns:
`c`, `eta`, `T_local`, `g`, `m`.

### Run a named regime

```python
from regimes import run_baseline, run_misinfo_fast, run_credible_dominant, run_polarized_T
from visualization import plot_global_trajectories

model, model_df, agent_df = run_baseline(steps=500, random_seed=42)
plot_global_trajectories(model_df)
```

Available regime runners:

| Function | Description |
|---|---|
| `run_baseline()` | Balanced preferences, moderate trust dynamics |
| `run_misinfo_fast()` | Low η agents, fast trust erosion |
| `run_credible_dominant()` | High η agents, faster trust recovery |
| `run_polarized_T()` | Bimodal η distribution, strong homophily |
| `run_homogeneous_T_baseline()` | No local trust heterogeneity (xi=0) |
| `run_fixed_T_baseline()` | Trust frozen at 0.5 — isolates Deffuant dynamics |

### Custom initialization

```python
from params import TrustCommonsParams
from model import TrustCommonsABM
from experiment import run_experiment

def my_init(model):
    for a in model.agent_lookup.values():
        a.eta = 0.9          # all agents strongly prefer credible info
        a.c   = 0.7          # start with high credibility belief
        a.update_attention()

params = TrustCommonsParams(N=300, random_seed=1, beta_homophily=6.0)
model, model_df, agent_df = run_experiment(steps=300, params=params, init_fn=my_init)
```

### Misinformation shock experiment

```python
from params import TrustCommonsParams
from regimes import init_baseline_regime
from analysis import run_misinfo_shock_experiment, plot_shock_recovery

params = TrustCommonsParams(N=300, random_seed=42, xi_local_heterogeneity=0.3)
model, model_df, agent_df = run_misinfo_shock_experiment(
    steps=500,
    params=params,
    shock_start=100,
    shock_end=150,
    init_fn=init_baseline_regime,
    misinfo_level=1.0,       # full misinfo flood during shock window
)
plot_shock_recovery(model_df, shock_start=100, shock_end=150)
```

### Adaptive vs random rewiring stress test

```python
from params import TrustCommonsParams
from regimes import init_baseline_regime
from experiment import run_experiment_with_spatial_metrics
from visualization import plot_stress_test

params_ad = TrustCommonsParams(N=300, random_seed=42, rewiring_mode="adaptive", p_rewire=0.5)
params_rn = TrustCommonsParams(N=300, random_seed=42, rewiring_mode="random",   p_rewire=0.5)

_, _, _, spatial_ad = run_experiment_with_spatial_metrics(500, params_ad, init_fn=init_baseline_regime)
_, _, _, spatial_rn = run_experiment_with_spatial_metrics(500, params_rn, init_fn=init_baseline_regime)

plot_stress_test(spatial_ad, spatial_rn)   # Moran's I + assortativity comparison
```

### Sensitivity analysis

```python
from analysis import run_T_star_sweeps, run_varT_sweeps, run_var_eta_sweeps

run_T_star_sweeps()     # one-at-a-time sweeps for equilibrium trust T*
run_varT_sweeps()       # sweeps for Var(T_local)
run_var_eta_sweeps()    # sweeps for preference polarization Var(η)
```

### Regime map and phase diagram

```python
from analysis import run_regime_map_sampling, run_polarization_phase_diagram

# Sample 200 random parameter vectors, classify regimes, plot scatter
T_stars, var_etas, labels = run_regime_map_sampling(n_samples=200, steps=500)

# 20×20 grid sweep over (beta_homophily, p_rewire)
grid1, grid2 = run_polarization_phase_diagram(steps=500)
```

---

## Model Description

### State variables

| Variable | Level | Description |
|---|---|---|
| `T` | Model | Global trust stock ∈ [0, 1] |
| `c_i` | Agent | Credibility belief ∈ [0, 1] |
| `η_i` | Agent | Preference for credible sources ∈ [0, 1] |
| `T_local_i` | Agent | Perceived local trust ∈ [0, 1] |
| `g_i`, `m_i` | Agent | Attention to credible / misinformation |

### One model step

1. **Cascade propagation** — if no cascade is active, seed a random node; otherwise advance one BFS node. Each activated node exchanges credibility beliefs with neighbors via bounded-confidence (Deffuant) updates.
2. **Optional rewiring** — the activated node may drop a dissimilar neighbor and rewire to a more similar one (adaptive) or a random node (random).
3. **Attention allocation** — all agents recompute `g_i`, `m_i` based on η, T, and T_local.
4. **Global trust update** — `ΔT = dt·(α·ĪG·(1−T) − β·ĪM·T)`.
5. **Local trust update** — each agent's T_local mixes global T with local credibility signal.
6. **Preference imitation** — agents stochastically copy a neighbor's η with noise.

### Key equations

**Trust dynamics (CPR):**
$$\Delta T = dt \cdot \bigl(\alpha \cdot \bar{I}_g \cdot (1 - T) - \beta \cdot \bar{I}_m \cdot T\bigr)$$

**Credibility update (Deffuant):**
$$c_i \leftarrow c_i + \mu \cdot (c_j - c_i) \quad \text{if } |c_i - c_j| < \varepsilon$$

**Attention allocation (share-of-wallet):**
$$g_i = A \cdot \frac{\eta_i^\kappa / p_g}{(\eta_i^\kappa / p_g) + ((1-\eta_i)^\kappa / p_m)}, \qquad m_i = A - g_i$$

**Price functions:**
$$p_g = 1 - \phi_T(T - 0.5) - \phi_\ell(T^\ell_i - 0.5), \qquad p_m = 1 + \phi_T(T - 0.5) + \phi_\ell(T^\ell_i - 0.5)$$

---

## Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `N` | 20 | Population size |
| `avg_degree` | 8 | Mean degree in Watts–Strogatz network |
| `small_world_rewire_p` | 0.1 | WS rewiring probability |
| `alpha_trust_up` | 1.0 | Trust repair rate (α) |
| `beta_trust_down` | 2.0 | Trust erosion rate (β) |
| `dt_trust` | 0.001 | Trust update timescale |
| `mu_learning` | 0.05 | Deffuant learning rate (μ) |
| `epsilon_conf` | 0.2 | Bounded-confidence threshold (ε) |
| `phi_forgetting` | 0.01 | Credibility decay toward 0.5 |
| `xi_local_heterogeneity` | 0.3 | Local vs global trust mix |
| `kappa_attention` | 2.0 | Sharpness of η→attention mapping (κ) |
| `beta_homophily` | 4.0 | Homophily strength in adaptive rewiring |
| `p_rewire` | 0.02 | Per-step rewiring attempt probability |
| `rewiring_mode` | `"adaptive"` | `"adaptive"`, `"random"`, or `"none"` |
| `p_imitation` | 0.3 | Probability of imitating a neighbor's η |
| `rho_eta_update` | 0.3 | Imitation step size (ρ) |
| `sigma_eta_mutation` | 0.05 | Noise added during imitation |
| `fixed_T` | `False` | If True, freeze trust (isolates Deffuant) |
| `random_seed` | 42 | RNG seed for reproducibility |

All parameters are fields of `TrustCommonsParams` (a Python dataclass) and can be set at construction time or overridden in place.

---

## Visualization Reference

All functions are in `visualization.py` and call `plt.show()` at the end.

| Function | Input | What it shows |
|---|---|---|
| `plot_global_trajectories(model_df)` | model_df | T, mean_c, mean_η, Var(T_local) over time |
| `plot_credibility_distribution(agent_df)` | agent_df | Histogram of c at final step |
| `plot_c_eta_scatter(agent_df)` | agent_df | c vs η scatter at final step |
| `plot_sample_credibility_trajectories(agent_df, n)` | agent_df | c trajectories for n sampled agents |
| `plot_network_credibility(model, agent_df)` | both | Network colored by c |
| `plot_network_attribute(model, agent_df, attr)` | both | Network colored by any agent attribute |
| `plot_attention_shares(model_df)` | model_df | g/(g+m) and m/(g+m) over time |
| `plot_T_vs_c_scatter(agent_df)` | agent_df | T_local vs c scatter at a given step |
| `plot_space_time_heatmap(agent_df, attr)` | agent_df | Agent × step heatmap of attr |
| `plot_T_c_correlation_over_time(agent_df)` | agent_df | corr(c, T_local) per step |
| `plot_T_and_mean_c_over_time(model_df)` | model_df | Time series + phase plot of T vs c̄ |
| `plot_stress_test(spatial_ad, spatial_rn)` | spatial_dfs | Moran's I and assortativity comparison |
| `plot_variances_over_time(adf_ad, adf_rn)` | agent_dfs | Var(c), Var(T_local) for two runs |

Spatial metrics DataFrames (`spatial_df`) are produced by `run_experiment_with_spatial_metrics()` and have columns: `step`, `I_Tlocal`, `I_c`, `r_Tlocal`, `r_c`.

---

## Running Tests

```bash
python -m unittest tests -v
```

14 of 16 tests pass. The 2 known failures (`test_bounded_confidence_*`) reproduce a pre-existing issue documented in the original notebook: `TrustCommonsParams(N=2, avg_degree=1)` produces a degenerate Watts–Strogatz graph. All substantive model behavior tests pass.

---

## Extending the Model

### Adding a new regime

Define an init function and optionally a convenience runner:

```python
# in regimes.py (or your own file)
from params import TrustCommonsParams
from experiment import run_experiment

def init_high_trust_regime(model):
    model.T = 0.9
    for a in model.agent_lookup.values():
        a.eta = 0.8
        a.c   = 0.8
        a.update_attention()

def run_high_trust(steps=500, random_seed=42):
    params = TrustCommonsParams(random_seed=random_seed)
    return run_experiment(steps=steps, params=params, init_fn=init_high_trust_regime)
```

### Adding a new agent attribute

1. Add the attribute in `TrustCommonsAgent.__init__` (`agent.py`).
2. Add it to the `agent_reporters` dict in `TrustCommonsABM.__init__` (`model.py`).
3. Update it inside the relevant model step method.

### Adding a new model-level metric

Add a lambda to the `model_reporters` dict in `TrustCommonsABM.__init__`:

```python
"my_metric": lambda m: some_function(m),
```

It will appear as a column in `model_df` automatically.

### Plugging in a different network topology

Override `_init_network()` in a subclass:

```python
import networkx as nx
from model import TrustCommonsABM

class BarabasiAlbertABM(TrustCommonsABM):
    def _init_network(self):
        return nx.barabasi_albert_graph(self.params.N, m=3, seed=self.params.random_seed)
```

---

## Reproducibility

Set `random_seed` in `TrustCommonsParams` to get deterministic runs. The model seeds both Python's `random` module and NumPy's RNG at initialization.

```python
params = TrustCommonsParams(random_seed=123)
```

For parameter sweeps, pass an explicit `seed` argument to `run_for_var_eta_grid()` or use the `base_seed` argument in `sweep_grid()`.
