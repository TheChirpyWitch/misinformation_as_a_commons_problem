"""
main.py — Runs all experiments from the notebook in sequence.

Sections:
  1. Regime tests
  2. Misinfo shock experiment
  3. Network stress tests (adaptive vs random rewiring)
  4. Calibration checks
  5. Sensitivity analysis
  6. Polarization phase diagram
"""

import numpy as np

from params import TrustCommonsParams
from experiment import run_experiment, run_experiment_with_spatial_metrics
from visualization import (
    plot_global_trajectories, plot_credibility_distribution, plot_c_eta_scatter,
    plot_sample_credibility_trajectories, plot_network_credibility,
    plot_attention_shares, plot_attention_totals, plot_network_attribute,
    plot_T_vs_c_scatter, plot_T_c_correlation_over_time, plot_T_and_mean_c_over_time,
    plot_stress_test, plot_variances_over_time,
)
from regimes import (
    run_baseline, run_misinfo_fast, run_credible_dominant,
    run_fixed_T_baseline, run_homogeneous_T_baseline, run_polarized_T,
    init_baseline_regime,
)
from analysis import (
    compare_regimes_T, run_and_plot_all_regimes,
    run_misinfo_shock_experiment, plot_shock_recovery,
    summarize_rewiring, build_merged_timeseries,
    run_T_star_sweeps, run_varT_sweeps, run_var_eta_sweeps,
    run_regime_map_sampling, run_polarization_phase_diagram,
)
from metrics import compute_corr_series


def run_regime_tests():
    print("\n=== REGIME TESTS ===")

    # Polarized T
    model, model_df, agent_df = run_polarized_T(steps=1000)
    plot_global_trajectories(model_df)
    plot_credibility_distribution(agent_df)
    plot_c_eta_scatter(agent_df)
    plot_sample_credibility_trajectories(agent_df, n_agents=5)
    plot_network_credibility(model, agent_df)
    plot_attention_shares(model_df)
    plot_attention_totals(model_df)
    plot_network_attribute(model, agent_df, attr="c")
    plot_network_attribute(model, agent_df, attr="T_local", cmap="plasma")
    plot_network_attribute(model, agent_df, attr="eta", cmap="coolwarm")
    plot_T_vs_c_scatter(agent_df)
    plot_T_c_correlation_over_time(agent_df)
    plot_T_and_mean_c_over_time(model_df)

    # Baseline
    model, model_df, agent_df = run_baseline(steps=1000)
    plot_global_trajectories(model_df)
    plot_credibility_distribution(agent_df)
    plot_c_eta_scatter(agent_df)
    plot_sample_credibility_trajectories(agent_df, n_agents=5)
    plot_network_credibility(model, agent_df)
    plot_attention_shares(model_df)
    plot_attention_totals(model_df)
    plot_network_attribute(model, agent_df, attr="c")
    plot_network_attribute(model, agent_df, attr="T_local", cmap="plasma")
    plot_network_attribute(model, agent_df, attr="eta", cmap="coolwarm")
    plot_T_vs_c_scatter(agent_df)
    plot_T_c_correlation_over_time(agent_df)
    plot_T_and_mean_c_over_time(model_df)

    # Misinfo fast
    model, model_df, agent_df = run_misinfo_fast(steps=1000)
    plot_global_trajectories(model_df)
    plot_credibility_distribution(agent_df)
    plot_c_eta_scatter(agent_df)
    plot_sample_credibility_trajectories(agent_df, n_agents=5)
    plot_network_credibility(model, agent_df)
    plot_attention_shares(model_df)
    plot_attention_totals(model_df)
    plot_network_attribute(model, agent_df, attr="c")
    plot_network_attribute(model, agent_df, attr="T_local", cmap="plasma")
    plot_network_attribute(model, agent_df, attr="eta", cmap="coolwarm")
    plot_T_vs_c_scatter(agent_df)
    plot_T_c_correlation_over_time(agent_df)
    plot_T_and_mean_c_over_time(model_df)

    # Credible dominant
    model, model_df, agent_df = run_credible_dominant(steps=1000)
    plot_global_trajectories(model_df)
    plot_credibility_distribution(agent_df)
    plot_c_eta_scatter(agent_df)
    plot_sample_credibility_trajectories(agent_df, n_agents=5)
    plot_network_credibility(model, agent_df)
    plot_attention_shares(model_df)
    plot_attention_totals(model_df)
    plot_network_attribute(model, agent_df, attr="c")
    plot_network_attribute(model, agent_df, attr="T_local", cmap="plasma")
    plot_network_attribute(model, agent_df, attr="eta", cmap="coolwarm")
    plot_T_vs_c_scatter(agent_df)
    plot_T_c_correlation_over_time(agent_df)
    plot_T_and_mean_c_over_time(model_df)

    # Homogeneous T verification
    model, model_df, agent_df = run_homogeneous_T_baseline(steps=1000)
    plot_global_trajectories(model_df)
    plot_credibility_distribution(agent_df)
    plot_c_eta_scatter(agent_df)
    plot_sample_credibility_trajectories(agent_df, n_agents=5)
    plot_network_credibility(model, agent_df)
    plot_attention_shares(model_df)
    plot_attention_totals(model_df)
    plot_network_attribute(model, agent_df, attr="c")
    plot_network_attribute(model, agent_df, attr="T_local", cmap="plasma")
    plot_network_attribute(model, agent_df, attr="eta", cmap="coolwarm")
    plot_T_vs_c_scatter(agent_df)
    plot_T_c_correlation_over_time(agent_df)
    plot_T_and_mean_c_over_time(model_df)

    # Fixed T verification
    model, model_df, agent_df = run_fixed_T_baseline(steps=500)
    plot_global_trajectories(model_df)
    plot_credibility_distribution(agent_df)
    plot_c_eta_scatter(agent_df)
    plot_sample_credibility_trajectories(agent_df, n_agents=5)
    plot_network_credibility(model, agent_df)
    plot_attention_shares(model_df)
    plot_attention_totals(model_df)
    plot_network_attribute(model, agent_df, attr="c")
    plot_network_attribute(model, agent_df, attr="T_local", cmap="plasma")
    plot_network_attribute(model, agent_df, attr="eta", cmap="coolwarm")
    plot_T_vs_c_scatter(agent_df)
    plot_T_c_correlation_over_time(agent_df)
    plot_T_and_mean_c_over_time(model_df)

    # Regime comparison
    regime_results = compare_regimes_T(steps=300)
    run_and_plot_all_regimes(steps=300, n_agents_plot=5)


def run_shock_tests():
    print("\n=== MISINFO SHOCK TEST ===")

    params = TrustCommonsParams(
        random_seed=42,
        N=300,
        xi_local_heterogeneity=0.3,
    )

    shock_start = 100
    shock_end   = 150

    model, model_df, agent_df = run_misinfo_shock_experiment(
        steps=500,
        params=params,
        shock_start=shock_start,
        shock_end=shock_end,
        init_fn=init_baseline_regime,
        misinfo_level=1.0,
    )

    plot_shock_recovery(model_df, shock_start=shock_start, shock_end=shock_end)
    plot_global_trajectories(model_df)
    plot_credibility_distribution(agent_df)
    plot_c_eta_scatter(agent_df)
    plot_sample_credibility_trajectories(agent_df, n_agents=5)
    plot_network_credibility(model, agent_df)
    plot_attention_shares(model_df)
    plot_attention_totals(model_df)
    plot_network_attribute(model, agent_df, attr="c")
    plot_network_attribute(model, agent_df, attr="T_local", cmap="plasma")
    plot_network_attribute(model, agent_df, attr="eta", cmap="coolwarm")
    plot_T_vs_c_scatter(agent_df)
    plot_T_c_correlation_over_time(agent_df)
    plot_T_and_mean_c_over_time(model_df)

    # Spatial check at step 500
    from visualization import normalize_agent_df
    step_to_check = 500
    df_long, _ = normalize_agent_df(agent_df)
    if "Step" in df_long.index.names:
        df_step = df_long.xs(step_to_check, level="Step")
    else:
        df_step = df_long[df_long["Step"] == step_to_check]
    c_bar = df_step["c"].mean()
    c_vals = dict(zip(df_step["AgentID"], df_step["c"]))
    diffs = []
    for i in model.G.nodes():
        neighbors = list(model.G.neighbors(i))
        if not neighbors:
            continue
        neighbor_c = np.mean([c_vals[j] for j in neighbors])
        diffs.append(neighbor_c - c_bar)
    diffs = np.array(diffs)
    print("Mean(neighbor_c - c_bar):", diffs.mean())
    print("Std (neighbor_c - c_bar):", diffs.std())
    print("Min/Max:", diffs.min(), diffs.max())


def run_network_stress_tests():
    print("\n=== NETWORK STRESS TESTS ===")

    steps = 500

    # 1. Adaptive rewiring
    params_adaptive = TrustCommonsParams(
        N=300,
        random_seed=42,
        rewiring_mode="adaptive",
        enable_rewiring=True,
        p_rewire=0.5,
        trust_update_interval=10,
        xi_local_heterogeneity=0.3,
    )

    model_ad, mdf_ad, adf_ad, spatial_ad = run_experiment_with_spatial_metrics(
        steps, params_adaptive, init_fn=init_baseline_regime
    )

    model_df  = mdf_ad
    agent_df  = adf_ad
    model     = model_ad
    spatial_df = spatial_ad
    plot_global_trajectories(model_df)
    plot_credibility_distribution(agent_df)
    plot_c_eta_scatter(agent_df)
    plot_sample_credibility_trajectories(agent_df, n_agents=5)
    plot_network_credibility(model, agent_df)
    plot_attention_shares(model_df)
    plot_attention_totals(model_df)
    plot_network_attribute(model, agent_df, attr="c")
    plot_network_attribute(model, agent_df, attr="T_local", cmap="plasma")
    plot_network_attribute(model, agent_df, attr="eta", cmap="coolwarm")
    plot_T_vs_c_scatter(agent_df)
    plot_T_c_correlation_over_time(agent_df)
    plot_T_and_mean_c_over_time(model_df)

    # 2. Random rewiring (stress test)
    params_random = TrustCommonsParams(
        N=300,
        random_seed=42,
        rewiring_mode="random",
        enable_rewiring=True,
        p_rewire=0.5,
        trust_update_interval=10,
        xi_local_heterogeneity=0.3,
    )

    model_rn, mdf_rn, adf_rn, spatial_rn = run_experiment_with_spatial_metrics(
        steps, params_random, init_fn=init_baseline_regime
    )

    model_df  = mdf_rn
    agent_df  = adf_rn
    model     = model_rn
    spatial_df = spatial_rn
    plot_global_trajectories(model_df)
    plot_credibility_distribution(agent_df)
    plot_c_eta_scatter(agent_df)
    plot_sample_credibility_trajectories(agent_df, n_agents=5)
    plot_network_credibility(model, agent_df)
    plot_attention_shares(model_df)
    plot_attention_totals(model_df)
    plot_network_attribute(model, agent_df, attr="c")
    plot_network_attribute(model, agent_df, attr="T_local", cmap="plasma")
    plot_network_attribute(model, agent_df, attr="eta", cmap="coolwarm")
    plot_T_vs_c_scatter(agent_df)
    plot_T_c_correlation_over_time(agent_df)
    plot_T_and_mean_c_over_time(model_df)

    # Correlation comparison
    corr_adaptive = compute_corr_series(adf_ad)
    corr_random   = compute_corr_series(adf_rn)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 4))
    steps_arr = np.arange(len(corr_adaptive))
    plt.plot(steps_arr, corr_adaptive, label="Adaptive Rewiring", linewidth=2)
    plt.plot(steps_arr, corr_random,  label="Random Rewiring", linewidth=2, linestyle="--")
    plt.xlabel("Step")
    plt.ylabel("corr(c, T_local)")
    plt.title("Correlation Between Credibility and Local Trust Over Time")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    plot_stress_test(spatial_ad, spatial_rn)
    plot_variances_over_time(adf_ad, adf_rn)

    # Rewiring stats
    merged = build_merged_timeseries(mdf_rn, spatial_rn)
    summarize_rewiring(model_ad)
    summarize_rewiring(model_rn)


def run_calibration():
    print("\n=== CALIBRATION ===")

    params = TrustCommonsParams(
        N=300,
        xi_local_heterogeneity=0.3,
        enable_rewiring=True,
        mu_learning=0.05,
        epsilon_conf=0.2,
        k_T=0.01,
        r_misinfo_harm=1.5,
    )
    model, model_df, agent_df = run_experiment(steps=500, params=params)
    plot_global_trajectories(model_df)
    plot_sample_credibility_trajectories(agent_df, n_agents=5)

    params = TrustCommonsParams(
        N=300,
        xi_local_heterogeneity=0.3,
        enable_rewiring=True,
        mu_learning=0.05,
        epsilon_conf=0.2,
        k_T=0.01,
        r_misinfo_harm=3.0,
    )
    model, model_df, agent_df = run_experiment(steps=500, params=params)
    print("unique T:", model_df["T"].nunique())
    print(model_df["T"].head(), model_df["T"].tail())
    plot_global_trajectories(model_df)
    plot_attention_shares(model_df)
    plot_attention_totals(model_df)

    params = TrustCommonsParams(
        N=1000,
        xi_local_heterogeneity=0.3,
        mu_learning=0.05,
        epsilon_conf=0.2,
        k_T=0.01,
        r_misinfo_harm=1.5,
    )
    model, model_df, agent_df = run_experiment(steps=500, params=params)
    plot_global_trajectories(model_df)
    plot_sample_credibility_trajectories(agent_df, n_agents=5)
    plot_attention_shares(model_df)
    plot_attention_totals(model_df)


def run_sensitivity_analysis():
    print("\n=== SENSITIVITY ANALYSIS ===")
    run_T_star_sweeps()
    run_varT_sweeps()
    run_var_eta_sweeps()
    run_regime_map_sampling(n_samples=200, steps=500)
    run_polarization_phase_diagram(steps=500)


if __name__ == "__main__":
    run_regime_tests()
    run_shock_tests()
    run_network_stress_tests()
    run_calibration()
    run_sensitivity_analysis()
