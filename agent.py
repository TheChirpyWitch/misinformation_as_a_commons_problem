from typing import TYPE_CHECKING, List

from params import clip01

if TYPE_CHECKING:
    from model import TrustCommonsABM


class TrustCommonsAgent:
    """
    Agent with:
      - unique_id : node id in the network
      - c         : credibility belief in [0, 1]
      - eta       : preference for credible vs misinfo attention in [0, 1]
      - T_local   : perceived local trust [0, 1]
      - g, m      : attention to credible/misinfo (per step, in budget units)
    """

    def __init__(self, unique_id: int, model: "TrustCommonsABM"):
        # Minimal interface for Mesa schedulers:
        #   - unique_id
        #   - model
        #   - (optionally) .step(), but we do everything in the model.step()
        self.unique_id = unique_id
        self.model = model
        self.random = model.random  # reuse model RNG for reproducibility

        # state variables
        self.c: float = self.random.uniform(0.3, 0.7)
        #self.eta: float = self.random.uniform(0.4, 0.6)
        p = model.params

        # draw from a wider normal around 0.5 and clip
        mean_eta = getattr(p, "eta_mean_init", 0.5)
        std_eta  = getattr(p, "eta_std_init", 0.2)

        eta_raw = self.random.gauss(mean_eta, std_eta)
        self.eta = clip01(eta_raw)

        self.T_local: float = 0.5

        self.g: float = 0.0
        self.m: float = 0.0

    @property
    def neighbors(self) -> List[int]:
        return list(self.model.G.neighbors(self.unique_id))

    def update_attention(self):
        """
        Map preference η and trust into attention g (credible) and m (misinfo).

        - η in [0,1]: preference for credible
        - T, T_local shape prices p_g, p_m
        - κ controls how sharply η tilts attention
        """
        p = self.model.params
        eta = self.eta
        T_global = self.model.T
        T_loc = self.T_local

        # --- price functions ---
        # lower p_g when trust is high -> credible cheaper
        # higher p_m when trust is high -> misinfo more costly
        base_price = 1.0

        phi_T = getattr(p, "phi_price_T", 0.5)       # sensitivity to global trust
        phi_loc = getattr(p, "phi_price_local", 0.5) # sensitivity to local trust

        if getattr(p, "fixed_T", False):
            phi_T = 0.0
            phi_loc = 0.0

        # keep prices bounded away from 0
        p_g = base_price - phi_T * (T_global - 0.5) - phi_loc * (T_loc - 0.5)
        p_m = base_price + phi_T * (T_global - 0.5) + phi_loc * (T_loc - 0.5)

        p_g = max(p_g, 0.1)
        p_m = max(p_m, 0.1)

        # --- preference weighting ---
        kappa = getattr(p, "kappa_attention", 2.0)   # >1 makes η effects stronger

        # weights before normalization
        w_g = (eta ** kappa) / p_g
        w_m = ((1.0 - eta) ** kappa) / p_m

        if w_g + w_m <= 0:
            share_g = 0.5
        else:
            share_g = w_g / (w_g + w_m)

        share_m = 1.0 - share_g

        A = p.A_bar
        self.g = A * share_g
        self.m = A * share_m

    def bounded_confidence_update(
        self,
        other: "TrustCommonsAgent",
        I_g_self_from_other: float,
        I_m_self_from_other: float,
        I_g_other_from_self: float,
        I_m_other_from_self: float,
    ):
        p = self.model.params
        ci, cj = self.c, other.c

        if abs(ci - cj) >= p.epsilon_conf:
            return

        # plain Deffuant: move toward each other, no built-in drift
        self.c  = clip01(ci + p.mu_learning * (cj - ci))
        other.c = clip01(cj + p.mu_learning * (ci - cj))

        # optional: gentle forgetting toward 0.5
        phi = getattr(p, "phi_forgetting", 0.0)
        if phi > 0.0:
            self.c  = clip01((1 - phi) * self.c  + phi * 0.5)
            other.c = clip01((1 - phi) * other.c + phi * 0.5)

    def step(self):
        """
        No-op step.

        Mesa's RandomActivation expects agents to have a .step() method.
        We do all dynamics in the model.step() via cascades, so this is empty.
        """
        pass
