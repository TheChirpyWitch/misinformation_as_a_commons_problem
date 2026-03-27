from params import TrustCommonsParams
from model import TrustCommonsABM
from experiment import run_experiment


# ---------- Regime initialization functions ----------

def init_baseline_regime(model: TrustCommonsABM,
                         eta_mean: float = 0.5,
                         eta_noise: float = 0.15):
    """
    Baseline:
      - η roughly centered near 0.5 (balanced preference)
      - misinfo harm moderate
      - attention mapping unchanged (kappa_attention moderate)
    """
    p = model.params

    # trust dynamics: moderate up/down balance
    p.r_misinfo_harm = 2.0
    p.alpha_trust_up = 1.0
    p.beta_trust_down = 2.0
    p.kappa_attention = 2.0  # how sharply η tilts attention

    for a in model.agent_lookup.values():
        # draw from N(eta_mean, eta_noise) using Mesa RNG
        eta_raw = model.random.gauss(eta_mean, eta_noise)
        # clip to [0, 1]
        a.eta = max(0.0, min(1.0, eta_raw))
        a.update_attention()


def init_fixed_T_baseline(model: TrustCommonsABM):
    """
    Verification baseline 2a:
      - Global trust T is fixed (no CPR feedback).
      - Prices do NOT depend on trust.
      - Resulting dynamics reduce to Deffuant + imitation (+ rewiring).
    """
    p = model.params

    # toggle fixed-T behavior
    p.fixed_T = True
    model.T = p.fixed_T_value   # usually 0.5
    # no trust evolution
    p.dt_trust = 0.0
    p.alpha_trust_up = 0.0
    p.beta_trust_down = 0.0

    # prices independent of trust
    p.phi_price_T = 0.0
    p.phi_price_local = 0.0

    # recompute attention once with these settings
    for a in model.agent_lookup.values():
        a.update_attention()


def init_misinfo_regime(model: TrustCommonsABM,
                        eta_mean: float = 0.2,
                        eta_noise: float = 0.10,
                        r_misinfo_harm: float = 3.0):
    """
    Misinfo-dominant:
      - agents are biased toward misinfo (low η)
      - misinfo does more damage to trust (higher r_misinfo_harm, beta_trust_down)
      - keep kappa_attention relatively high so η strongly affects attention.
    """
    p = model.params

    p.r_misinfo_harm = r_misinfo_harm
    p.alpha_trust_up = 1.0          # credible still helps a bit
    p.beta_trust_down = 3.0         # misinfo erodes trust faster
    p.kappa_attention = 2.5         # η differences show up strongly in attention

    for a in model.agent_lookup.values():
        eta_raw = model.random.gauss(eta_mean, eta_noise)
        # bias strongly toward misinfo: cap upper bound < 0.5
        eta_clipped = max(0.0, min(0.4, eta_raw))
        a.eta = eta_clipped
        a.update_attention()


def init_credible_regime(model: TrustCommonsABM,
                         eta_mean: float = 0.8,
                         eta_noise: float = 0.10,
                         r_misinfo_harm: float = 1.5):
    """
    Credible-dominant:
      - agents favor credible sources (high η)
      - misinfo still harmful but less extreme
      - trust recovers more easily (higher alpha_trust_up).
    """
    p = model.params

    p.r_misinfo_harm = r_misinfo_harm
    p.alpha_trust_up = 1.3         # credible attention rebuilds trust faster
    p.beta_trust_down = 1.7        # misinfo still erodes trust
    p.kappa_attention = 2.0

    for a in model.agent_lookup.values():
        eta_raw = model.random.gauss(eta_mean, eta_noise)
        # bias toward credible: lower bound > 0.5
        eta_clipped = max(0.6, min(1.0, eta_raw))
        a.eta = eta_clipped
        a.update_attention()


def init_homogeneous_T_baseline(model: TrustCommonsABM):
    """
    Verification baseline 2b:
      - Global trust T evolves normally.
      - Local heterogeneity is disabled (xi = 0):
            T_i^ℓ = T^g for all i.
      - Prices do not explicitly depend on local trust (only global, or not at all).
    """
    p = model.params

    # no local heterogeneity
    p.xi_local_heterogeneity = 0.0

    # optional: make prices ignore local trust entirely in this baseline
    p.phi_price_local = 0.0

    # recompute T_local and attention under these settings
    model._update_local_trust()
    for a in model.agent_lookup.values():
        a.update_attention()


def init_polarized_regime(model: TrustCommonsABM,
                          eta_low_mean: float = 0.2,
                          eta_high_mean: float = 0.8,
                          eta_noise: float = 0.10,
                          frac_high: float = 0.5,
                          r_misinfo_harm: float = 2.0):
    """
    Polarized regime:
      - population splits into two camps:
          * credible-leaning (low η)
          * misinfo-leaning (high η)
      - strong homophily + rewiring amplify clustering
      - trust dynamics are more fragile (similar up/down rates).
    """
    p = model.params

    # Trust dynamics: more fragile balance between repair and harm
    p.r_misinfo_harm   = r_misinfo_harm
    p.alpha_trust_up   = 1.0    # credible attention repairs trust
    p.beta_trust_down  = 2.0    # misinfo erodes trust somewhat more strongly
    p.kappa_attention  = 2.0

    # Network / local heterogeneity knobs to encourage polarization
    p.beta_homophily          = 8.0   # strong preference for similar neighbors
    p.p_rewire                = 0.3   # frequent adaptive rewiring
    p.xi_local_heterogeneity  = 0.5   # allow local trust to diverge from global T (if used in your model)
    p.N = 1000

    rng = model.random

    for a in model.agent_lookup.values():
        # Randomly assign each agent to one of two camps
        if rng.random() < frac_high:
            # misinfo-leaning camp: high η
            eta_raw = rng.gauss(eta_high_mean, eta_noise)
            # keep them clearly on the "high" side
            eta_clipped = max(0.6, min(1.0, eta_raw))
        else:
            # credible-leaning camp: low η
            eta_raw = rng.gauss(eta_low_mean, eta_noise)
            # keep them clearly on the "low" side
            eta_clipped = max(0.0, min(0.4, eta_raw))

        a.eta = eta_clipped
        a.update_attention()


# ---------- Regime runner functions ----------

def run_baseline(steps=500, random_seed=42):
    params = TrustCommonsParams(random_seed=random_seed)
    return run_experiment(steps=steps,
                          params=params,
                          init_fn=init_baseline_regime)

def run_misinfo_fast(steps=500, random_seed=123):
    params = TrustCommonsParams(random_seed=random_seed)
    return run_experiment(steps=steps,
                          params=params,
                          init_fn=init_misinfo_regime)

def run_credible_dominant(steps=500, random_seed=7):
    params = TrustCommonsParams(random_seed=random_seed)
    return run_experiment(steps=steps,
                          params=params,
                          init_fn=init_credible_regime)

def run_fixed_T_baseline(steps: int = 500, random_seed: int = 42):
    params = TrustCommonsParams(random_seed=random_seed, fixed_T=True)
    model, model_df, agent_df = run_experiment(
        steps=steps,
        params=params,
        init_fn=init_fixed_T_baseline,
    )
    return model, model_df, agent_df

def run_homogeneous_T_baseline(steps: int = 500, random_seed: int = 42):
    params = TrustCommonsParams(random_seed=random_seed, xi_local_heterogeneity=0.0)
    model, model_df, agent_df = run_experiment(
        steps=steps,
        params=params,
        init_fn=init_homogeneous_T_baseline,
    )
    return model, model_df, agent_df

def run_polarized_T(steps: int = 500, random_seed: int = 42):
    params = TrustCommonsParams(random_seed=random_seed)
    model, model_df, agent_df = run_experiment(
        steps=steps,
        params=params,
        init_fn=init_polarized_regime,
    )
    return model, model_df, agent_df
