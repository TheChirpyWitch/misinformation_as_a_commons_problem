import random

import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from tqdm import tqdm

from params import TrustCommonsParams
from model import TrustCommonsABM
from experiment import run_experiment, run_experiment_with_spatial_metrics
from regimes import (
    init_baseline_regime, run_baseline, run_misinfo_fast, run_credible_dominant,
)
from visualization import (
    plot_global_trajectories, plot_credibility_distribution, plot_c_eta_scatter,
    plot_sample_credibility_trajectories, plot_network_credibility,
    plot_attention_shares, plot_attention_totals, plot_network_attribute,
    plot_T_vs_c_scatter, plot_T_c_correlation_over_time, plot_T_and_mean_c_over_time,
    plot_stress_test, plot_variances_over_time,
)
from metrics import compute_corr_series


# ---------- Regime comparison ----------

def compare_regimes_T(steps=300):
    # run all three
    base_model, base_df, _ = run_baseline(steps=steps)
    mis_model,  mis_df,  _ = run_misinfo_fast(steps=steps)
    cred_model, cred_df, _ = run_credible_dominant(steps=steps)

    plt.figure(figsize=(8, 5))
    base_df["T"].plot(label="Baseline")
    mis_df["T"].plot(label="Misinfo-dominant")
    cred_df["T"].plot(label="Credible-dominant")

    plt.xlabel("Step")
    plt.ylabel("Global trust T")
    plt.title("Global trust under different regimes")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return {
        "baseline": base_df,
        "misinfo": mis_df,
        "credible": cred_df,
    }


def run_and_plot_regime(regime_name: str, run_fn, steps: int = 300, n_agents_plot: int = 5):
    """
    Run a regime and produce:
      - global trajectories
      - credibility distribution
      - c–η scatter
      - sample credibility trajectories
      - network snapshot colored by c
    """
    print(f"\n{'='*60}\nRegime: {regime_name}\n{'='*60}")

    model, model_df, agent_df = run_fn(steps=steps)

    # 1. Global trajectories
    plot_global_trajectories(model_df)

    # 2. Credibility distribution at final step
    plot_credibility_distribution(agent_df)

    # 3. c–η scatter at final step
    plot_c_eta_scatter(agent_df)

    # 4. Sample agents' credibility trajectories
    plot_sample_credibility_trajectories(agent_df, n_agents=n_agents_plot)

    # 5. Network snapshot colored by credibility
    plot_network_credibility(model, agent_df)

    return model, model_df, agent_df


def run_and_plot_all_regimes(steps: int = 300, n_agents_plot: int = 5):
    from regimes import run_baseline, run_misinfo_fast, run_credible_dominant
    results = {}

    # Baseline
    results["baseline"] = run_and_plot_regime(
        "Baseline",
        run_baseline,
        steps=steps,
        n_agents_plot=n_agents_plot,
    )

    # Misinfo-dominant (trust should fall)
    results["misinfo"] = run_and_plot_regime(
        "Misinfo-dominant",
        run_misinfo_fast,
        steps=steps,
        n_agents_plot=n_agents_plot,
    )

    # Credible-dominant (trust should rise)
    results["credible"] = run_and_plot_regime(
        "Credible-dominant",
        run_credible_dominant,
        steps=steps,
        n_agents_plot=n_agents_plot,
    )

    return results


# ---------- Shock experiment ----------

def run_misinfo_shock_experiment(
    steps: int,
    params: TrustCommonsParams,
    shock_start: int = 100,
    shock_end: int = 150,
    init_fn=None,
    misinfo_level: float = 1.0,
):
    """
    Run the TrustCommonsABM with a temporary misinformation shock.

    For steps in [shock_start, shock_end):
      - We override attention so agents allocate `misinfo_level` of their
        attention budget to misinfo and (1 - misinfo_level) to credible info,
        then update trust based on that.

    Outside the shock window:
      - We run the normal model.step().

    Returns
    -------
    model, model_df, agent_df
    """
    model = TrustCommonsABM(params)

    # Optional regime initializer (baseline / misinfo-dominant / credible-dominant)
    if init_fn is not None:
        init_fn(model)

    for t in range(steps):
        if shock_start <= t < shock_end:
            # --- SHOCK STEP: override attention to misinfo ---
            A = model.params.A_bar

            # You can keep cascades / c / eta evolving if you want,
            # but here we just impose the attention shock.
            for a in model.agent_lookup.values():
                a.g = A * (1.0 - misinfo_level)
                a.m = A * misinfo_level

            # Trust responds to this forced attention
            model._update_global_trust()
            model._update_local_trust()
            model._imitate_preferences()  # still let preferences evolve, optional

            # Keep scheduler time consistent
            model.schedule.step()

            # Manually record this shock step
            model.datacollector.collect(model)
        else:
            # --- NORMAL STEP ---
            model.step()

    model_df = model.datacollector.get_model_vars_dataframe()
    agent_df = model.datacollector.get_agent_vars_dataframe()
    return model, model_df, agent_df


def plot_shock_recovery(model_df: pd.DataFrame,
                        shock_start: int,
                        shock_end: int,
                        title: str = "Misinformation shock and trust recovery"):
    """
    Plot global trust T over time with the shock window highlighted.
    """
    # ensure we have a step axis
    if "Step" in model_df.columns:
        df = model_df.reset_index(drop=True)
        steps = df["Step"]
    else:
        df = model_df.reset_index().rename(columns={"index": "Step"})
        steps = df["Step"]

    T_series = df["T"]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, T_series, lw=2, label="Global trust T")

    # Shock window shading
    ax.axvspan(shock_start, shock_end, color="red", alpha=0.15, label="Shock window")

    ax.set_xlabel("Step")
    ax.set_ylabel("Global trust T")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ---------- Network stress test helpers ----------

def summarize_rewiring(model: TrustCommonsABM):
    G = model.G
    final_edges = {tuple(sorted(e)) for e in G.edges()}
    changed_edges = final_edges.symmetric_difference(model.initial_edges)

    print("=== Rewiring summary ===")
    print(f"Attempts:       {model.rewire_attempts}")
    print(f"Successes:      {model.rewire_successes}")
    if model.rewire_attempts > 0:
        print(f"Success rate:   {model.rewire_successes / model.rewire_attempts:.3f}")
    print(f"Edges changed (vs initial): {len(changed_edges)} of {len(model.initial_edges)}")

    if model.cascade_sizes:
        sizes = np.array(model.cascade_sizes)
        print("\n=== Cascade coverage ===")
        print(f"Number of cascades: {len(sizes)}")
        print(f"Mean size:          {sizes.mean():.1f}")
        print(f"Median size:        {np.median(sizes):.1f}")
        print(f"Min / Max size:     {sizes.min()} / {sizes.max()}")
        print(f"Coverage fraction (mean): {sizes.mean() / model.params.N:.2f}")


def build_merged_timeseries(model_df: pd.DataFrame, spatial_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge global quantities (T, mean_c, g_sum, m_sum) with spatial metrics.
    Assumes model_df index is step or has a 'Step' column equivalent.
    """
    # if your model_df index is not step, reset:
    if "Step" in model_df.columns:
        mdf = model_df.reset_index(drop=True).rename(columns={"Step": "step"})
    else:
        mdf = model_df.reset_index().rename(columns={"index": "step"})

    merged = mdf.merge(spatial_df, on="step", how="left")

    # add attention shares if they exist
    if {"g_sum", "m_sum"}.issubset(merged.columns):
        total = merged["g_sum"] + merged["m_sum"]
        total = total.replace(0, 1.0)
        merged["share_g"] = merged["g_sum"] / total
        merged["share_m"] = merged["m_sum"] / total

    return merged


# ---------- Sensitivity analysis ----------

def init_baseline_regime_sa(model: TrustCommonsABM,
                            eta_mean: float = 0.5,
                            eta_noise: float = 0.15):
    """
    Baseline for sensitivity analysis:
      - Only initialize eta.
      - DO NOT overwrite alpha_trust_up, beta_trust_down, etc.
    """
    p = model.params  # not actually used here, but fine to keep

    for a in model.agent_lookup.values():
        eta_raw = model.random.gauss(eta_mean, eta_noise)
        a.eta = max(0.0, min(1.0, eta_raw))
        a.update_attention()


def run_for_T_star(param_overrides, steps=500):
    # start from your usual baseline params
    base = TrustCommonsParams(
        lambda_trust_price=2.0,
        alpha_trust_up=1.0,
        beta_trust_down=2.0,
        rho_eta_update=0.3,
        beta_homophily=4.0,
        dt_trust=0.001,
        # + any other defaults you care about
    )

    # apply overrides for the sensitivity sweep
    for k, v in param_overrides.items():
        setattr(base, k, v)

    model, model_df, agent_df, spatial_df = run_experiment_with_spatial_metrics(
        steps=steps,
        params=base,
        init_fn=init_baseline_regime_sa,  # <- use SA init, not the old one
    )

    tail = int(0.2 * len(model_df))
    return float(model_df["T"].iloc[-tail:].mean())


def run_for_varT(param_overrides, steps=500):
    base = TrustCommonsParams(
        lambda_trust_price=2.0,
        alpha_trust_up=1.0,
        beta_trust_down=2.0,
        rho_eta_update=0.3,
        beta_homophily=4.0,
        dt_trust=0.001,
        p_rewire=0.3,
    )

    for k, v in param_overrides.items():
        setattr(base, k, v)

    model, model_df, agent_df, spatial_df = run_experiment_with_spatial_metrics(
        steps=steps,
        params=base,
        init_fn=init_baseline_regime_sa,
    )

    tail = int(0.2 * len(model_df))
    tail_df = model_df.iloc[-tail:]
    return float(tail_df["var_T_local"].mean())


def run_for_var_eta(param_overrides, steps=500):
    base = TrustCommonsParams(
        lambda_trust_price=2.0,
        alpha_trust_up=1.0,
        beta_trust_down=2.0,
        rho_eta_update=0.3,
        beta_homophily=4.0,
        dt_trust=0.001,
        p_rewire=0.3,
    )

    for k, v in param_overrides.items():
        setattr(base, k, v)

    model, model_df, agent_df, spatial_df = run_experiment_with_spatial_metrics(
        steps=steps,
        params=base,
        init_fn=init_baseline_regime_sa,
    )

    # agent_df is already the final snapshot
    eta_vals = agent_df["eta"].to_numpy()
    var_eta = float(np.var(eta_vals))

    return var_eta


def run_model_metrics(theta, steps=500):
    """
    theta = [lambda_trust_price,
             alpha_trust_up,
             beta_trust_down,
             rho_eta_update,
             beta_homophily,
             p_rewire]
    returns: T_star (tail mean T), var_eta (final preference variance)
    """

    (lambda_trust_price,
     alpha_trust_up,
     beta_trust_down,
     rho_eta_update,
     beta_homophily,
     p_rewire) = theta

    params = TrustCommonsParams(
        lambda_trust_price = float(lambda_trust_price),
        alpha_trust_up     = float(alpha_trust_up),
        beta_trust_down    = float(beta_trust_down),
        rho_eta_update     = float(rho_eta_update),
        beta_homophily     = float(beta_homophily),
        dt_trust           = 0.001,
        p_rewire           = float(p_rewire),
        # set any other defaults you normally use here
    )

    model, model_df, agent_df, spatial_df = run_experiment_with_spatial_metrics(
        steps=steps,
        params=params,
        init_fn=init_baseline_regime_sa,   # <-- the SA init that doesn't overwrite α/β/...
    )

    # T* = tail-averaged global trust
    tail = int(0.2 * len(model_df))
    T_star = float(model_df["T"].iloc[-tail:].mean())

    # Var(η) = variance of preferences in final snapshot
    eta_vals = agent_df["eta"].to_numpy(dtype=float)
    var_eta = float(np.var(eta_vals))

    return T_star, var_eta


def run_for_var_eta_grid(beta_h, p_rewire, steps=500, seed=0, ddof=0):
    seed = int(seed)
    np.random.seed(seed)
    random.seed(seed)

    params = TrustCommonsParams(
        lambda_trust_price = 2.0,
        alpha_trust_up     = 1.0,
        beta_trust_down    = 2.0,
        rho_eta_update     = 0.3,
        beta_homophily     = float(beta_h),
        dt_trust           = 0.001,
        p_rewire           = float(p_rewire),
        random_seed        = seed,
    )

    model, model_df, agent_df, spatial_df = run_experiment_with_spatial_metrics(
        steps=steps,
        params=params,
        init_fn=init_baseline_regime_sa,
    )

    if "step" in agent_df.columns:
        final_agents = agent_df.loc[agent_df["step"] == agent_df["step"].max()]
    else:
        final_agents = agent_df

    eta_vals = final_agents["eta"].to_numpy(dtype=float)
    return float(np.var(eta_vals, ddof=ddof))


def sweep_grid(run_fn, beta_h_vals, p_rewire_vals, steps=500, base_seed=123):
    grid = np.zeros((len(p_rewire_vals), len(beta_h_vals)))
    for i, p_r in enumerate(tqdm(p_rewire_vals, desc="p_rewire")):
        for j, bh in enumerate(beta_h_vals):
            seed = base_seed + 10_000*i + 100*j
            grid[i, j] = run_fn(bh, p_r, steps=steps, seed=seed)
    return grid


def plot_grid(grid, title, beta_h_vals, p_rewire_vals, vmin, vmax):
    fig, ax = plt.subplots(figsize=(7,5))
    im = ax.imshow(
        grid, origin="lower", aspect="auto",
        extent=[beta_h_vals[0], beta_h_vals[-1], p_rewire_vals[0], p_rewire_vals[-1]],
        vmin=vmin, vmax=vmax
    )
    ax.set_xlabel(r"$\beta_{\mathrm{homophily}}$")
    ax.set_ylabel(r"$p_{\mathrm{rewire}}$")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$\mathrm{Var}(\eta)$")
    plt.tight_layout()
    plt.show()


# ---------- Sensitivity sweep runners ----------

def run_T_star_sweeps():
    param_ranges_Tstar_core = {
        # trust repair strength
        "alpha_trust_up":  np.linspace(0.2, 3.0, 10),

        # misinfo harm strength
        "beta_trust_down": np.linspace(0.5, 5.0, 10),

        # imitation strength (strong effect, non-linear)
        "rho_eta_update":  np.linspace(0.0, 0.9, 10),

        # trust timescale (higher = faster change; affects apparent T* over finite horizon)
        "dt_trust":        np.linspace(0.0002, 0.005, 10),

        "mu_learning":        np.linspace(0.0002, 0.005, 10),
    }

    results = {}

    for pname, values in param_ranges_Tstar_core.items():
        T_stars = []
        pbar = tqdm(values, desc=f"Sweeping {pname}", leave=False)

        for v in pbar:
            overrides = {pname: v}
            T_star = run_for_T_star(overrides)
            T_stars.append(T_star)
            pbar.set_postfix({"T*": f"{T_star:.3f}"})

        results[pname] = (values, np.array(T_stars))

    # Plot results
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()
    axes[5].set_visible(False)

    for ax, (pname, (vals, T_stars)) in zip(axes, results.items()):
        ax.plot(vals, T_stars, marker="o")
        ax.set_xlabel(pname)
        ax.set_ylabel("T* (Global trust)")
        ax.set_title(f"T* vs {pname}")
        ax.grid(True)

    plt.tight_layout()
    plt.show()

    return results


def run_varT_sweeps():
    param_ranges_VarT = {
        # how different local trust can be from global T
        "xi_local_heterogeneity": np.linspace(0.01, 0.8, 10),

        # how strongly preferences follow neighbors
        "rho_eta_update":         np.linspace(0.0, 0.9, 10),

        # propensity to connect to similar agents
        "beta_homophily":         np.linspace(0.0, 12.0, 10),

        # how often rewiring happens
        "p_rewire":               np.linspace(0.0, 0.5, 10),

        # trust repair and harm (set somewhat narrower; they modulate level, not so much variance)
        "alpha_trust_up":         np.linspace(0.5, 2.0, 8),
        "beta_trust_down":        np.linspace(1.0, 4.0, 8),
    }

    results_var = {}

    for pname, values in param_ranges_VarT.items():
        var_vals = []

        pbar = tqdm(values, desc=f"Sweeping {pname} (Var T_local)", leave=False)
        for v in pbar:
            overrides = {pname: v}
            var_T = run_for_varT(overrides)
            var_vals.append(var_T)
            pbar.set_postfix({"Var(T_local)": f"{var_T:.4f}"})

        results_var[pname] = (values, np.array(var_vals))

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()
    axes[5].set_visible(False)

    for ax, (pname, (vals, var_vals)) in zip(axes, results_var.items()):
        ax.plot(vals, var_vals, marker="o")
        ax.set_xlabel(pname)
        ax.set_ylabel("Var(T_local) (tail avg)")
        ax.set_title(f"Var(T_local) vs {pname}")
        ax.grid(True)

    plt.tight_layout()
    plt.show()

    return results_var


def run_var_eta_sweeps():
    param_ranges_var_eta = {
        # imitation & noise
        "rho_eta_update":     np.linspace(0.0, 0.9, 10),
        "p_imitation":        np.linspace(0.0, 1.0, 10),
        "sigma_eta_mutation": np.linspace(0.0, 0.4, 10),

        # network structure
        "beta_homophily":     np.linspace(0.0, 12.0, 10),
        "p_rewire":           np.linspace(0.0, 0.5, 10),
    }

    results_eta = {}

    for pname, values in param_ranges_var_eta.items():
        var_vals = []
        pbar = tqdm(values, desc=f"Sweeping {pname} (Var eta)", leave=False)
        for v in pbar:
            overrides = {pname: v}
            var_eta = run_for_var_eta(overrides)
            var_vals.append(var_eta)
            pbar.set_postfix({"Var(eta)": f"{var_eta:.4f}"})
        results_eta[pname] = (values, np.array(var_vals))

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()
    axes[5].set_visible(False)

    for ax, (pname, (vals, var_vals)) in zip(axes, results_eta.items()):
        ax.plot(vals, var_vals, marker="o")
        ax.set_xlabel(pname)
        ax.set_ylabel("Var(η) at final step")
        ax.set_title(f"Var(η) vs {pname}")
        ax.grid(True)

    plt.tight_layout()
    plt.show()

    return results_eta


def run_regime_map_sampling(n_samples=200, steps=500):
    param_bounds = {
        "lambda_trust_price": (0.5, 4.0),
        "alpha_trust_up":     (0.2, 3.0),
        "beta_trust_down":    (0.5, 4.0),
        "rho_eta_update":     (0.0, 0.9),
        "beta_homophily":     (0.0, 12.0),
        "p_rewire":           (0.0, 0.4),
    }

    param_names = list(param_bounds.keys())

    def sample_params(n_samples=200):
        thetas = []
        for _ in range(n_samples):
            theta = []
            for name in param_names:
                lo, hi = param_bounds[name]
                theta.append(np.random.uniform(lo, hi))
            thetas.append(theta)
        return np.array(thetas)

    param_values = sample_params(n_samples=n_samples)

    T_stars = []
    var_etas = []

    for theta in tqdm(param_values, desc="Running regime samples"):
        T_star, var_eta = run_model_metrics(theta, steps=steps)
        T_stars.append(T_star)
        var_etas.append(var_eta)

    T_stars = np.array(T_stars)
    var_etas = np.array(var_etas)

    # Polarization threshold = upper tercile of Var(eta)
    var_eta_thresh = np.quantile(var_etas, 0.66)

    regime_labels = []
    for T_star, v_eta in zip(T_stars, var_etas):
        if v_eta >= var_eta_thresh:
            regime_labels.append("Polarized")
        else:
            if T_star >= 0.6:
                regime_labels.append("Credible")
            elif T_star <= 0.4:
                regime_labels.append("Misinfo")
            else:
                regime_labels.append("Mixed")
    regime_labels = np.array(regime_labels)

    # Map regimes to colors
    regime_to_color = {
        "Credible":  "#88c0d0",  # blue-ish
        "Misinfo":   "#d08770",  # orange
        "Polarized": "#a3be8c",  # green
        "Mixed":     "#e5e9f0",  # light grey
    }

    colors = [regime_to_color[r] for r in regime_labels]

    fig, ax = plt.subplots(figsize=(7, 5))

    sc = ax.scatter(T_stars, var_etas, c=colors, alpha=0.8, edgecolor="k", linewidth=0.3)

    ax.set_xlabel(r"$T^*$ (equilibrium global trust)")
    ax.set_ylabel(r"$\mathrm{Var}(\eta)$ (preference polarization)")
    ax.set_title("Regime Map in $(T^*,\\, \\mathrm{Var}(\\eta))$ Space")

    ax.axvline(0.4, color="k", linestyle="--", linewidth=0.8)
    ax.axvline(0.6, color="k", linestyle="--", linewidth=0.8)
    ax.axhline(var_eta_thresh, color="k", linestyle="--", linewidth=0.8)

    handles = [mpatches.Patch(color=regime_to_color[name], label=name)
               for name in ["Credible", "Misinfo", "Polarized", "Mixed"]]
    ax.legend(handles=handles, title="Regime", loc="best")

    plt.tight_layout()
    plt.show()

    return T_stars, var_etas, regime_labels


def run_polarization_phase_diagram(steps=500):
    beta_h_vals   = np.linspace(0.0, 12.0, 20)
    p_rewire_vals = np.linspace(0.0, 0.5, 20)

    grid1 = sweep_grid(run_for_var_eta_grid, beta_h_vals, p_rewire_vals, steps=steps, base_seed=111)
    grid2 = sweep_grid(run_for_var_eta_grid, beta_h_vals, p_rewire_vals, steps=steps, base_seed=222)

    global_vmin = min(grid1.min(), grid2.min())
    global_vmax = max(grid1.max(), grid2.max())
    plot_grid(grid1, "Grid 1", beta_h_vals, p_rewire_vals, global_vmin, global_vmax)
    plot_grid(grid2, "Grid 2", beta_h_vals, p_rewire_vals, global_vmin, global_vmax)

    return grid1, grid2
