import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


# ---------- Data utilities ----------

def normalize_agent_df(agent_df: pd.DataFrame):
    """
    For our new format (MultiIndex Step, AgentID),
    just flatten it into columns.
    """
    if agent_df.index.nlevels == 2:
        step_index = agent_df.index.get_level_values(0)
        agent_index = agent_df.index.get_level_values(1)
        df_long = agent_df.copy()
        df_long = df_long.assign(Step=step_index, AgentID=agent_index)
        return df_long, True

    # Fallbacks if needed
    df_long = agent_df.reset_index()
    has_step = "Step" in df_long.columns
    return df_long, has_step


def get_last_step_snapshot(agent_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a dataframe of agent variables at the last available step.
    If no explicit step dimension exists, just returns agent_df.
    """
    if agent_df.index.nlevels > 1:
        # MultiIndex: (Step, AgentID)
        last_step = agent_df.index.get_level_values(0).max()
        return agent_df.xs(last_step, level=0)
    if "Step" in agent_df.columns:
        last_step = agent_df["Step"].max()
        return agent_df[agent_df["Step"] == last_step].set_index("AgentID")
    # No step info – assume snapshot
    return agent_df


def _flatten_agent_df(agent_df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten Mesa agent_df so that:
      - 'Step' is a normal column
      - 'AgentID' is a normal column
      - no duplicate 'Step' columns
    """
    df = agent_df.copy()

    # If it's a MultiIndex (Step, AgentID), flatten it
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()  # creates 'Step' and 'AgentID' columns
    else:
        # If index is named 'Step' but not in columns yet, flatten it
        if df.index.name == "Step" and "Step" not in df.columns:
            df = df.reset_index()

    # If somehow multiple 'Step' columns exist, drop extras
    cols = list(df.columns)
    while cols.count("Step") > 1:
        first = cols.index("Step")
        for i, c in enumerate(cols):
            if c == "Step" and i != first:
                df = df.drop(columns=[c])
                break
        cols = list(df.columns)

    return df


# ---------- Plot functions ----------

def plot_global_trajectories(model_df: pd.DataFrame):
    """
    Plot time series of:
      - global trust T
      - mean credibility
      - mean eta (if available)
      - variance of local trust (if available)
    """
    fig, axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True)

    model_df["T"].plot(ax=axes[0])
    axes[0].set_ylabel("Global trust T")
    axes[0].set_title("Global trust over time")

    model_df["mean_c"].plot(ax=axes[1])
    axes[1].set_ylabel("Mean credibility c̄")

    if "mean_eta" in model_df.columns:
        model_df["mean_eta"].plot(ax=axes[2])
        axes[2].set_ylabel("Mean η (pref for credible)")
    else:
        axes[2].set_visible(False)

    if "var_T_local" in model_df.columns:
        model_df["var_T_local"].plot(ax=axes[3])
        axes[3].set_ylabel("Var(T_local)")
    else:
        axes[3].set_visible(False)

    axes[-1].set_xlabel("Step")
    plt.tight_layout()
    plt.show()


def plot_credibility_distribution(agent_df: pd.DataFrame, bins: int = 20):
    """
    Histogram of credibility at final step (or snapshot).
    """
    agents_last = get_last_step_snapshot(agent_df)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(agents_last["c"], bins=bins, edgecolor="black")
    ax.set_xlabel("Credibility c")
    ax.set_ylabel("Count of agents")
    ax.set_title("Credibility distribution at final step")
    plt.tight_layout()
    plt.show()


def plot_c_eta_scatter(agent_df: pd.DataFrame):
    """
    Scatter plot of credibility vs. preference η at final step (if η exists).
    """
    agents_last = get_last_step_snapshot(agent_df)

    if "eta" not in agents_last.columns:
        print("No 'eta' column in agent_df; cannot make c–η scatter.")
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(agents_last["c"], agents_last["eta"], alpha=0.5)
    ax.set_xlabel("Credibility c")
    ax.set_ylabel("Preference η (share attention to credible)")
    ax.set_title("c vs η at final step")
    plt.tight_layout()
    plt.show()


def plot_sample_credibility_trajectories(agent_df: pd.DataFrame, n_agents: int = 5):
    df_long, has_step = normalize_agent_df(agent_df)
    if not has_step:
        print("No explicit step dimension in agent_df; skipping trajectory plot.")
        return

    sample_ids = df_long["AgentID"].unique()[:n_agents]

    fig, ax = plt.subplots(figsize=(8, 5))
    for aid in sample_ids:
        sub = df_long[df_long["AgentID"] == aid]
        ax.plot(sub["Step"], sub["c"], label=f"Agent {aid}")

    ax.set_xlabel("Step")
    ax.set_ylabel("Credibility c")
    ax.set_title("Sample agents' credibility trajectories")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_network_credibility(model, agent_df: pd.DataFrame, layout_seed: int = 0):
    """
    Draw the network with nodes colored by credibility at final step.
    Falls back to agents' current c if DataCollector isn't aligned.
    """
    agents_last = get_last_step_snapshot(agent_df)

    # map node -> c
    c_map = {}
    for node_id, agent in model.agent_lookup.items():
        try:
            c_val = agents_last.loc[node_id]["c"]
        except Exception:
            c_val = agent.c
        c_map[node_id] = c_val

    pos = nx.spring_layout(model.G, seed=layout_seed)

    fig, ax = plt.subplots(figsize=(6, 6))
    nodes = model.G.nodes()
    node_colors = [c_map[n] for n in nodes]

    nodes_coll = nx.draw_networkx_nodes(
        model.G, pos, node_color=node_colors, cmap="viridis", ax=ax
    )
    nx.draw_networkx_edges(model.G, pos, alpha=0.2, ax=ax)
    plt.colorbar(nodes_coll, ax=ax, label="Credibility c")
    ax.set_axis_off()
    ax.set_title("Network snapshot colored by credibility")
    plt.tight_layout()
    plt.show()


def plot_network_attribute(
    model,
    agent_df: pd.DataFrame,
    attr: str = "c",
    step: int = None,
    layout_seed: int = 0,
    cmap: str = "viridis",
    title: str = None,
):
    """
    Draw the network with nodes colored by a given agent attribute at a chosen step.
    """
    # normalize to a long form with Step & AgentID columns
    df_long, _ = normalize_agent_df(agent_df)

    # pick step
    if step is None:
        step = df_long["Step"].max()

    df_step = df_long[df_long["Step"] == step]

    if attr not in df_step.columns:
        raise ValueError(f"{attr} not found in agent_df columns: {df_step.columns.tolist()}")

    # map node -> attr value, fall back to live agent if missing
    val_map = {}
    step_vals = dict(zip(df_step["AgentID"], df_step[attr]))

    for node_id, agent in model.agent_lookup.items():
        v = step_vals.get(node_id, getattr(agent, attr, np.nan))
        val_map[node_id] = v

    # layout and draw
    pos = nx.spring_layout(model.G, seed=layout_seed)

    fig, ax = plt.subplots(figsize=(6, 6))
    nodes = list(model.G.nodes())
    node_colors = [val_map[n] for n in nodes]

    nodes_coll = nx.draw_networkx_nodes(
        model.G, pos, node_color=node_colors, cmap=cmap, ax=ax
    )
    nx.draw_networkx_edges(model.G, pos, alpha=0.2, ax=ax)

    cbar = plt.colorbar(nodes_coll, ax=ax, label=attr)
    cbar.formatter.set_useOffset(False)
    cbar.formatter.set_scientific(False)
    cbar.update_ticks()

    ax.set_axis_off()
    if title is None:
        ax.set_title(f"Network snapshot colored by {attr} (step {step})")
    else:
        ax.set_title(title)

    plt.tight_layout()
    plt.show()


def plot_attention_shares(model_df):
    """
    Plot fraction of attention going to credible vs misinfo over time.
    """
    if not {"g_sum", "m_sum"}.issubset(model_df.columns):
        raise ValueError("model_df must contain 'g_sum' and 'm_sum' columns.")

    att = model_df[["g_sum", "m_sum"]].copy()
    total = att["g_sum"] + att["m_sum"]
    # avoid division by zero
    total = total.replace(0, 1.0)

    att["share_g"] = att["g_sum"] / total
    att["share_m"] = att["m_sum"] / total

    fig, ax = plt.subplots(figsize=(8, 4))
    att["share_g"].plot(ax=ax, label="Share to credible (g / (g+m))")
    att["share_m"].plot(ax=ax, label="Share to misinfo (m / (g+m))")
    ax.set_xlabel("Step")
    ax.set_ylabel("Attention share")
    ax.set_ylim(0, 1)
    ax.set_title("Share of attention to credible vs misinfo over time")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_attention_totals(model_df):
    """
    Plot total attention to credible (g_sum) and misinfo (m_sum) over time.
    """
    if not {"g_sum", "m_sum"}.issubset(model_df.columns):
        raise ValueError("model_df must contain 'g_sum' and 'm_sum' columns.")

    fig, ax = plt.subplots(figsize=(8, 4))
    model_df["g_sum"].plot(ax=ax, label="Total credible attention (Σg)")
    model_df["m_sum"].plot(ax=ax, label="Total misinfo attention (Σm)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Total attention")
    ax.set_title("Total attention to credible vs misinfo over time")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_trust_and_spatial_metrics(merged: pd.DataFrame):
    steps = merged["step"]

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)

    # 1) Global trust + mean credibility
    ax = axes[0]
    if "T" in merged.columns:
        ax.plot(steps, merged["T"], label="Global trust T")
    if "mean_c" in merged.columns:
        ax.plot(steps, merged["mean_c"], label="Mean credibility c̄", linestyle="--")
    ax.set_ylabel("T, c̄")
    ax.set_title("Global trust and mean credibility over time")
    ax.legend()

    # 2) Moran's I for local trust and credibility
    ax = axes[1]
    ax.plot(steps, merged["I_Tlocal"], label="Moran's I (T_local)")
    ax.plot(steps, merged["I_c"],      label="Moran's I (c)", linestyle="--")
    ax.set_ylabel("Moran's I")
    ax.set_title("Spatial autocorrelation over time")
    ax.legend()

    # 3) Assortativity for local trust and credibility
    ax = axes[2]
    ax.plot(steps, merged["r_Tlocal"], label="Assortativity (T_local)")
    ax.plot(steps, merged["r_c"],      label="Assortativity (c)", linestyle="--")
    ax.set_ylabel("Assortativity")
    ax.set_xlabel("Step")
    ax.set_title("Network assortativity over time")
    ax.legend()

    plt.tight_layout()
    plt.show()


def plot_attention_and_spatial(merged: pd.DataFrame):
    steps = merged["step"]
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    ax = axes[0]
    ax.plot(steps, merged["share_g"], label="Share to credible")
    ax.plot(steps, merged["share_m"], label="Share to misinfo", linestyle="--")
    ax.set_ylabel("Attention share")
    ax.set_title("Attention allocation over time")
    ax.legend()

    ax = axes[1]
    ax.plot(steps, merged["I_Tlocal"], label="Moran's I (T_local)")
    ax.plot(steps, merged["I_c"],      label="Moran's I (c)", linestyle="--")
    ax.set_ylabel("Moran's I")
    ax.set_xlabel("Step")
    ax.set_title("Spatial autocorrelation vs attention")
    ax.legend()

    plt.tight_layout()
    plt.show()


def plot_T_vs_c_scatter(agent_df, step=None, alpha=0.6):
    df_long, _ = normalize_agent_df(agent_df)

    if step is None:
        step = df_long["Step"].max()

    df = df_long[df_long["Step"] == step]

    plt.figure(figsize=(6,5))
    plt.scatter(df["c"], df["T_local"], alpha=alpha, s=40)
    plt.xlabel("Credibility c")
    plt.ylabel("Local trust T_local")
    plt.title(f"T_local vs c (Step {step})")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()


def make_space_time_matrix(agent_df, attr: str = "T_local"):
    """
    Build a matrix M[agent_idx, step_idx] of an attribute (T_local or c) over time.

    Works with:
      - MultiIndex index: (Step, AgentID)
      - or flat index + 'Step' and 'AgentID' columns
    """
    df_long, _ = normalize_agent_df(agent_df)

    # Case 1: MultiIndex with Step & AgentID as index levels
    if isinstance(df_long.index, pd.MultiIndex) and \
       "Step" in df_long.index.names and "AgentID" in df_long.index.names:

        idx_names = list(df_long.index.names)
        step_pos = idx_names.index("Step")
        agent_pos = idx_names.index("AgentID")

        step_values = df_long.index.get_level_values("Step")
        agent_values = df_long.index.get_level_values("AgentID")

        steps = np.array(sorted(step_values.unique()))
        agents = np.array(sorted(agent_values.unique()))

        step_index  = {s: i for i, s in enumerate(steps)}
        agent_index = {a: i for i, a in enumerate(agents)}

        M = np.full((len(agents), len(steps)), np.nan)

        for idx, row in df_long.iterrows():
            # idx is a tuple (Step, AgentID, ...) or exactly (Step, AgentID)
            step = idx[step_pos]
            aid  = idx[agent_pos]
            i = agent_index[aid]
            t = step_index[step]
            M[i, t] = row[attr]

    # Case 2: Step & AgentID are columns
    else:
        if "Step" not in df_long.columns or "AgentID" not in df_long.columns:
            raise ValueError("make_space_time_matrix: need Step and AgentID either as index levels or columns.")

        steps = np.array(sorted(df_long["Step"].unique()))
        agents = np.array(sorted(df_long["AgentID"].unique()))
        step_index  = {s: i for i, s in enumerate(steps)}
        agent_index = {a: i for i, a in enumerate(agents)}

        M = np.full((len(agents), len(steps)), np.nan)

        for _, row in df_long.iterrows():
            i = agent_index[row["AgentID"]]
            t = step_index[row["Step"]]
            M[i, t] = row[attr]

    return M, agents, steps


def plot_space_time_heatmap(agent_df, attr="T_local", cmap="viridis"):
    """
    Plot a heatmap of attr (T_local or c) over time for each agent.
    """
    M, agents, steps = make_space_time_matrix(agent_df, attr=attr)

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(M, aspect="auto", origin="lower",
                   extent=[steps.min(), steps.max(), 0, len(agents)],
                   cmap=cmap)

    ax.set_xlabel("Step")
    ax.set_ylabel("Agent (index)")
    ax.set_title(f"{attr} over time (space–time heatmap)")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(attr)

    plt.tight_layout()
    plt.show()


def plot_T_c_correlation_over_time(agent_df):
    df_long, _ = normalize_agent_df(agent_df)

    corrs = []
    steps = sorted(df_long["Step"].unique())

    for step in steps:
        df = df_long[df_long["Step"] == step]
        corr = df["c"].corr(df["T_local"])
        corrs.append(corr)

    plt.figure(figsize=(8,4))
    plt.plot(steps, corrs, lw=2)
    plt.xlabel("Step")
    plt.ylabel("corr(c, T_local)")
    plt.title("Correlation between credibility and local trust over time")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_T_and_mean_c_over_time(model_df: pd.DataFrame):
    """
    Plot global trust T(t) and mean credibility c̄(t) over time,
    plus a phase plot T vs c̄.

    Expects model_df to have columns:
        - "T"      : global trust
        - "mean_c" : mean credibility across agents
    """
    # make sure we have a step axis
    if "Step" in model_df.columns:
        df = model_df.reset_index(drop=True)
        steps = df["Step"]
    else:
        df = model_df.reset_index().rename(columns={"index": "Step"})
        steps = df["Step"]

    T_series = df["T"]
    c_bar    = df["mean_c"]

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=False)

    # --- 1) Time series of T and c̄ ---
    ax = axes[0]
    ax.plot(steps, T_series, label="Global trust T", lw=2)
    ax.plot(steps, c_bar, label="Mean credibility c̄", lw=2, linestyle="--")
    ax.set_xlabel("Step")
    ax.set_ylabel("Value")
    ax.set_title("Global trust T and mean credibility c̄ over time")
    ax.legend()
    ax.grid(alpha=0.3)

    # --- 2) Phase plot: T vs c̄ (colored by time) ---
    ax = axes[1]
    sc = ax.scatter(c_bar, T_series, c=steps, cmap="viridis", s=30)
    ax.set_xlabel("Mean credibility c̄")
    ax.set_ylabel("Global trust T")
    ax.set_title("Phase plot: T vs c̄ (color = time)")
    plt.colorbar(sc, ax=ax, label="Step")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_variances_over_time(agent_df_ad, agent_df_rn,
                             label_ad="Adaptive", label_rn="Random"):
    """
    Compare Var(c) and Var(T_local) over time between two runs
    (e.g., adaptive vs random rewiring).
    """
    df_ad = _flatten_agent_df(agent_df_ad)
    df_rn = _flatten_agent_df(agent_df_rn)

    # group by Step
    var_c_ad = df_ad.groupby("Step")["c"].var()
    var_c_rn = df_rn.groupby("Step")["c"].var()
    var_T_ad = df_ad.groupby("Step")["T_local"].var()
    var_T_rn = df_rn.groupby("Step")["T_local"].var()

    steps = var_c_ad.index

    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    # --- Var(c) ---
    ax[0].plot(steps, var_c_ad, label=f"{label_ad} var(c)")
    ax[0].plot(steps, var_c_rn, label=f"{label_rn} var(c)", linestyle="--")
    ax[0].set_ylabel("Var(c)")
    ax[0].grid(alpha=0.3)
    ax[0].legend()

    # --- Var(T_local) ---
    ax[1].plot(steps, var_T_ad, label=f"{label_ad} var(T_local)")
    ax[1].plot(steps, var_T_rn, label=f"{label_rn} var(T_local)", linestyle="--")
    ax[1].set_ylabel("Var(T_local)")
    ax[1].set_xlabel("Step")
    ax[1].grid(alpha=0.3)
    ax[1].legend()

    plt.tight_layout()
    plt.show()


def plot_stress_test(spatial_ad, spatial_rn):
    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    # ---- Moran's I ----
    ax[0].plot(spatial_ad["step"], spatial_ad["I_Tlocal"], lw=2, label="Adaptive – I(T_local)")
    ax[0].plot(spatial_rn["step"], spatial_rn["I_Tlocal"], lw=2, linestyle="--", label="Random – I(T_local)")
    ax[0].plot(spatial_ad["step"], spatial_ad["I_c"], lw=1.6, alpha=0.7, label="Adaptive – I(c)")
    ax[0].plot(spatial_rn["step"], spatial_rn["I_c"], lw=1.6, alpha=0.7, linestyle="--", label="Random – I(c)")
    ax[0].set_ylabel("Moran's I")
    ax[0].set_title("Spatial Autocorrelation")
    ax[0].grid(alpha=0.3)
    ax[0].legend()

    # ---- Assortativity ----
    ax[1].plot(spatial_ad["step"], spatial_ad["r_Tlocal"], lw=2, label="Adaptive – r(T_local)")
    ax[1].plot(spatial_rn["step"], spatial_rn["r_Tlocal"], lw=2, linestyle="--", label="Random – r(T_local)")
    ax[1].plot(spatial_ad["step"], spatial_ad["r_c"], lw=1.6, alpha=0.7, label="Adaptive – r(c)")
    ax[1].plot(spatial_rn["step"], spatial_rn["r_c"], lw=1.6, alpha=0.7, linestyle="--", label="Random – r(c)")
    ax[1].set_xlabel("Step")
    ax[1].set_ylabel("Assortativity")
    ax[1].set_title("Node-level Assortativity")
    ax[1].grid(alpha=0.3)
    ax[1].legend()

    plt.tight_layout()
    plt.show()
