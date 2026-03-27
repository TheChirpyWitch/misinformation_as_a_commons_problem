import math
from dataclasses import dataclass
from typing import Tuple


@dataclass
class TrustCommonsParams:
    # Population + network
    N: int = 20                  # population size
    avg_degree: int = 8           # neighbors in Watts–Strogatz
    small_world_rewire_p: float = 0.1

    #Stress test
    rewiring_mode: str = "adaptive"
    p_rewire: float = 0.02        # options: "adaptive", "random", "none"

    # Attention budget
    A_bar: float = 1.0            # fixed attention per agent

    # Global trust dynamics (Eq. 1)
    k_T: float = 0.05             # trust responsiveness
    r_misinfo_harm: float = 1.5   # harm multiplier for misinfo

    # Credibility update (Eq. 4)
    mu_learning: float = 0.05      # learning rate μ
    epsilon_conf: float = 0.2    # bounded confidence ε

    # Attention signals in credibility update (λ_g, λ_m)
    lambda_g_signal: float = 0.3
    lambda_m_signal: float = 0.3

    # Trust → price mapping (Eq. 8)
    lambda_trust_price: float = 2.0

    # Preference imitation (Eq. 9)
    omega_eta: float = 0.3        # imitation weight
    sigma_eta: float = 0.0        # imitation noise

    # Local heterogeneity (Eq. 2)
    xi_local_heterogeneity: float = 0.3   # how strong local vs global
    gamma_local_trust: float = 0.1        # inertia of T_local update
    w_self_local_signal: float = 0.4      # weight on own c vs neighbors

    # Adaptive rewiring parameters
    p_reconsider_edge: float = 0.02  # probability to reconsider edges
    beta_homophily: float = 4.0      # controls homophily strength
    enable_rewiring: bool = True

    phi_forgetting: float = 0.01
    eta_mean_init: float = 0.5
    eta_std_init: float = 0.2   # quite a bit of spread

    p_imitation: float = 0.3
    rho_eta_update: float = 0.3
    sigma_eta_mutation: float = 0.05

    alpha_trust_up: float = 1.0
    beta_trust_down: float = 2.0
    dt_trust: float = 0.001   # small for slow changes
    trust_update_interval: int = 1   # how many node activations between trust updates

    phi_price_T: float = 0.5
    phi_price_local: float = 0.5
    kappa_attention: float = 2.0

    fixed_T: bool = False        # if True, trust never changes
    fixed_T_value: float = 0.5   # optional: starting T for baseline

    # Random seed
    random_seed: int = 42


def clip01(x: float) -> float:
    """Clamp x to [0, 1]."""
    return min(1.0, max(0.0, x))


def price_g(T_i: float, lam: float) -> float:
    """
    Price of credible info:
        pg(T_i) = exp(-λ T_i)
    """
    return math.exp(-lam * T_i)


def price_m(T_i: float, lam: float) -> float:
    """
    Price of misinfo:
        pm(T_i) = exp(λ (1 - T_i))
    """
    return math.exp(lam * (1.0 - T_i))


def allocate_attention(
    eta_i: float,
    T_i: float,
    A_bar: float,
    lambda_trust_price: float
) -> Tuple[float, float]:
    """
    Cobb–Douglas demand given prices pg, pm and budget Ā.

    We use the closed-form allocation:
        g_i = (η_i / pg(T_i)) * Ā
        m_i = ((1 - η_i) / pm(T_i)) * Ā

    And *define* A_i = Ā, so "fractions" are relative to this fixed budget.
    """
    T_i = clip01(T_i)
    eta_i = clip01(eta_i)

    pg_i = price_g(T_i, lambda_trust_price)
    pm_i = price_m(T_i, lambda_trust_price)

    g_i = (eta_i / pg_i) * A_bar
    m_i = ((1.0 - eta_i) / pm_i) * A_bar

    # numerical safety
    g_i = max(0.0, g_i)
    m_i = max(0.0, m_i)

    return g_i, m_i
