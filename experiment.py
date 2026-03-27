import numpy as np
import networkx as nx
import pandas as pd

from params import TrustCommonsParams
from model import TrustCommonsABM
from metrics import morans_I_on_graph


def run_experiment(steps: int = 200,
                   params: TrustCommonsParams = None,
                   init_fn=None):
    """
    Run the model for `steps` steps.
    Optionally apply `init_fn(model)` after initialization.
    Returns:
      - model
      - model_df: per-step model stats
      - agent_df: per-step, per-agent stats (MultiIndex: Step, AgentID)
    """
    if params is None:
        params = TrustCommonsParams()

    model = TrustCommonsABM(params)

    # optional regime-specific initialization
    if init_fn is not None:
        init_fn(model)
        for a in model.agent_lookup.values():
            a.update_attention()

    # Removed manual history collection; DataCollector will handle it.

    for t in range(steps):
        model.step()

    model_df = model.datacollector.get_model_vars_dataframe()
    agent_df = model.datacollector.get_agent_vars_dataframe()

    return model, model_df, agent_df


def run_experiment_with_spatial_metrics(
    steps: int,
    params: TrustCommonsParams,
    init_fn=None,
) -> tuple:
    """
    Run the ABM and collect:
      - model_df: existing model-level datacollector output
      - agent_df: existing agent-level datacollector output
      - spatial_df: time series of Moran's I and assortativity for T_local and c

    spatial_df columns:
      step, I_Tlocal, I_c, r_Tlocal, r_c
    """
    model = TrustCommonsABM(params)

    # optional regime initializer
    if init_fn is not None:
        init_fn(model)

    spatial_rows = []

    for t in range(steps):
        # 1) advance model one step
        model.step()

        # 2) compute spatial metrics at this step
        G = model.G

        # get current node-level values directly from agents
        T_local_vals = {i: a.T_local for i, a in model.agent_lookup.items()}
        c_vals       = {i: a.c       for i, a in model.agent_lookup.items()}

        # Moran's I
        I_Tlocal = morans_I_on_graph(G, T_local_vals)
        I_c      = morans_I_on_graph(G, c_vals)

        # Assortativity: attach attributes, then use networkx
        nx.set_node_attributes(G, T_local_vals, "T_local_attr")
        nx.set_node_attributes(G, c_vals,       "c_attr")

        try:
            r_Tlocal = nx.numeric_assortativity_coefficient(G, "T_local_attr")
        except Exception:
            r_Tlocal = np.nan

        try:
            r_c = nx.numeric_assortativity_coefficient(G, "c_attr")
        except Exception:
            r_c = np.nan

        spatial_rows.append({
            "step": t,
            "I_Tlocal": I_Tlocal,
            "I_c": I_c,
            "r_Tlocal": r_Tlocal,
            "r_c": r_c,
        })

    # pull out the usual mesa DataCollector outputs
    model_df = model.datacollector.get_model_vars_dataframe()
    agent_df = model.datacollector.get_agent_vars_dataframe()
    spatial_df = pd.DataFrame(spatial_rows)

    return model, model_df, agent_df, spatial_df
