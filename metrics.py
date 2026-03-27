import numpy as np
import networkx as nx
import pandas as pd

from visualization import normalize_agent_df


def morans_I_on_graph(G: nx.Graph, values: dict) -> float:
    """
    Moran's I for a scalar node attribute on an unweighted graph.

    Parameters
    ----------
    G : networkx.Graph
        The graph.
    values : dict
        Mapping node_id -> x_i (float).

    Returns
    -------
    float
        Moran's I (≈ 0: no spatial structure, >0: positive clustering).
    """
    nodes = list(G.nodes())
    if len(nodes) == 0:
        return 0.0

    x = np.array([values[i] for i in nodes], dtype=float)
    x_mean = x.mean()
    denom = np.sum((x - x_mean) ** 2)
    if denom == 0:
        return 0.0

    num = 0.0
    W = 0
    for u, v in G.edges():
        num += (values[u] - x_mean) * (values[v] - x_mean)
        W += 1

    if W == 0:
        return 0.0

    I = (len(nodes) / W) * (num / denom)
    return float(I)


def spatial_trust_correlation_metrics(model, agent_df, step: int = None):
    """
    Compute Moran's I and assortativity for T_local and c at a given step.

    Parameters
    ----------
    model : TrustCommonsABM
    agent_df : pd.DataFrame
        Agent-level datacollector output.
    step : int or None
        If None, use the last step in agent_df.

    Returns
    -------
    dict with keys:
        "I_Tlocal", "I_c", "r_Tlocal", "r_c"
    """
    df_long, _ = normalize_agent_df(agent_df)
    if step is None:
        step = df_long["Step"].max()

    df_step = df_long[df_long["Step"] == step]
    T_local_vals = dict(zip(df_step["AgentID"], df_step["T_local"]))
    c_vals       = dict(zip(df_step["AgentID"], df_step["c"]))

    G = model.G

    # Moran's I
    I_Tlocal = morans_I_on_graph(G, T_local_vals)
    I_c      = morans_I_on_graph(G, c_vals)

    # Assortativity: attach attributes and use networkx.numeric_assortativity_coefficient
    nx.set_node_attributes(G, T_local_vals, "T_local_attr")
    nx.set_node_attributes(G, c_vals,       "c_attr")

    r_Tlocal = nx.numeric_assortativity_coefficient(G, "T_local_attr")
    r_c      = nx.numeric_assortativity_coefficient(G, "c_attr")

    return {
        "step": step,
        "I_Tlocal": I_Tlocal,
        "I_c": I_c,
        "r_Tlocal": r_Tlocal,
        "r_c": r_c,
    }


def get_node_values_at_step(agent_df, step: int):
    """
    From the long agent_df, get dicts of T_local and c at a given step.
    """
    df_long, _ = normalize_agent_df(agent_df)
    df_step = df_long[df_long["Step"] == step]

    T_local_vals = dict(zip(df_step["AgentID"], df_step["T_local"]))
    c_vals       = dict(zip(df_step["AgentID"], df_step["c"]))

    return T_local_vals, c_vals


def compute_corr_series(agent_df):
    df_long, _ = normalize_agent_df(agent_df)

    corrs = []
    steps = sorted(df_long["Step"].unique())

    for step in steps:
        df = df_long[df_long["Step"] == step]
        corr = df["c"].corr(df["T_local"])
        corrs.append(corr)

    return np.array(corrs)
