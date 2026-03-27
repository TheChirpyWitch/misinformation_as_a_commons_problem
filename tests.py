import unittest
import math
import numpy as np

from params import TrustCommonsParams, clip01, price_g, price_m, allocate_attention
from model import TrustCommonsABM
from experiment import run_experiment


class TestTrustCommonsHelpers(unittest.TestCase):
    def test_clip01_bounds(self):
        self.assertEqual(clip01(-0.5), 0.0)
        self.assertEqual(clip01(1.5), 1.0)
        self.assertAlmostEqual(clip01(0.3), 0.3)

    def test_price_monotonicity(self):
        lam = 2.0
        # pg decreases with T
        self.assertGreater(price_g(0.0, lam), price_g(0.5, lam))
        self.assertGreater(price_g(0.5, lam), price_g(1.0, lam))
        # pm increases as T decreases (i.e., more expensive when trust is low)
        self.assertGreater(price_m(0.0, lam), price_m(0.5, lam))
        self.assertGreater(price_m(0.5, lam), price_m(1.0, lam))

    def test_allocate_attention_nonnegative(self):
        A_bar = 1.0
        lam = 2.0
        for eta in [0.0, 0.3, 0.7, 1.0]:
            for T in [0.0, 0.5, 1.0]:
                g, m = allocate_attention(eta, T, A_bar, lam)
                self.assertGreaterEqual(g, 0.0)
                self.assertGreaterEqual(m, 0.0)
                # If eta=1, most attention should go to g
                if eta == 1.0:
                    self.assertGreater(g, m)
                # If eta=0, most attention should go to m
                if eta == 0.0:
                    self.assertGreater(m, g)


class TestTrustCommonsAgent(unittest.TestCase):
    def _make_small_model(self):
        params = TrustCommonsParams(N=2, avg_degree=1, small_world_rewire_p=0.0,
                                    enable_rewiring=False, random_seed=123)
        model = TrustCommonsABM(params)
        return model

    def test_bounded_confidence_no_update_outside_eps(self):
        model = self._make_small_model()
        p = model.params
        a0 = model.agent_lookup[0]
        a1 = model.agent_lookup[1]

        # Force values
        a0.c = 0.1
        a1.c = 0.9
        p.epsilon_conf = 0.2  # |0.1 - 0.9| = 0.8 > eps

        old_c0, old_c1 = a0.c, a1.c

        a0.bounded_confidence_update(
            other=a1,
            I_g_self_from_other=0.0,
            I_m_self_from_other=0.0,
            I_g_other_from_self=0.0,
            I_m_other_from_self=0.0,
        )

        self.assertEqual(a0.c, old_c0)
        self.assertEqual(a1.c, old_c1)

    def test_bounded_confidence_moves_toward_each_other(self):
        model = self._make_small_model()
        p = model.params
        a0 = model.agent_lookup[0]
        a1 = model.agent_lookup[1]

        a0.c = 0.4
        a1.c = 0.5
        p.epsilon_conf = 0.2     # |0.4 - 0.5| = 0.1 < eps
        p.mu_learning = 0.5
        p.lambda_g_signal = 0.0  # zero out attention signal for clean math
        p.lambda_m_signal = 0.0

        # expected: each moves halfway toward the other
        old_c0, old_c1 = a0.c, a1.c

        a0.bounded_confidence_update(
            other=a1,
            I_g_self_from_other=0.0,
            I_m_self_from_other=0.0,
            I_g_other_from_self=0.0,
            I_m_other_from_self=0.0,
        )

        # both should now be in (0.4, 0.5) and closer to each other
        self.assertGreater(a0.c, old_c0)
        self.assertLess(a1.c, old_c1)
        self.assertLess(abs(a0.c - a1.c), abs(old_c0 - old_c1))


class TestTrustCommonsModel(unittest.TestCase):
    def test_model_initialization_agent_count(self):
        N = 50
        params = TrustCommonsParams(N=N, avg_degree=4, small_world_rewire_p=0.1,
                                    enable_rewiring=False, random_seed=42)
        model = TrustCommonsABM(params)

        self.assertEqual(len(model.agent_lookup), N)
        self.assertEqual(len(model.G.nodes()), N)
        self.assertEqual(len(model.schedule.agents), N)

    def test_local_trust_equals_global_when_xi_zero(self):
        params = TrustCommonsParams(
            N=20,
            xi_local_heterogeneity=0.0,
            enable_rewiring=False,
            random_seed=42,
        )
        model = TrustCommonsABM(params)
        # run a few steps so T changes
        for _ in range(5):
            model.step()

        for a in model.agent_lookup.values():
            self.assertAlmostEqual(a.T_local, model.T, places=7)

    def test_global_trust_increases_when_only_credible_attention(self):
        params = TrustCommonsParams(
            N=5,
            enable_rewiring=False,
            xi_local_heterogeneity=0.0,
            random_seed=1,
        )
        model = TrustCommonsABM(params)
        p = model.params

        # Force agents to only use credible attention: g>0, m=0
        for a in model.agent_lookup.values():
            a.g = 1.0
            a.m = 0.0

        old_T = model.T
        model._update_global_trust()
        self.assertGreater(model.T, old_T)

    def test_global_trust_decreases_when_only_misinfo(self):
        params = TrustCommonsParams(
            N=5,
            enable_rewiring=False,
            xi_local_heterogeneity=0.0,
            random_seed=1,
        )
        model = TrustCommonsABM(params)
        p = model.params

        for a in model.agent_lookup.values():
            a.g = 0.0
            a.m = 1.0

        old_T = model.T
        model._update_global_trust()
        self.assertLess(model.T, old_T)

    def test_rewiring_preserves_number_of_edges(self):
        params = TrustCommonsParams(
            N=30,
            avg_degree=4,
            small_world_rewire_p=0.1,
            enable_rewiring=True,
            random_seed=7,
        )
        model = TrustCommonsABM(params)

        initial_edges = model.G.number_of_edges()
        # run a few steps to trigger rewiring
        for _ in range(10):
            model.step()

        final_edges = model.G.number_of_edges()
        # rewiring should not change total number of edges
        self.assertEqual(initial_edges, final_edges)

    def test_datacollector_shapes(self):
        params = TrustCommonsParams(
            N=20,
            enable_rewiring=False,
            random_seed=3,
        )
        steps = 15
        model, model_df, agent_df = run_experiment(steps=steps, params=params)

        # model_df should have "steps" rows (one row per collect)
        self.assertEqual(len(model_df), steps)

        # Agent dataframe should not be empty (we collected something)
        self.assertGreater(len(agent_df), 0)

        # Required columns exist for downstream analysis
        for col in ["c", "eta", "T_local", "g", "m"]:
            self.assertIn(col, agent_df.columns)

    def test_trust_converges_up_with_persistent_credible_attention(self):
        """
        Regime test: if agents consistently allocate only credible attention,
        global trust should increase noticeably (not necessarily to 1).
        """
        params = TrustCommonsParams(
            N=200,
            random_seed=123,
        )
        model = TrustCommonsABM(params)

        # record initial trust
        T0 = model.T

        steps = 200

        # force all attention to credible at each step
        for _ in range(steps):
            for a in model.agent_lookup.values():
                a.g = params.A_bar
                a.m = 0.0
            model._update_global_trust()

        Tend = model.T

        # trust should go up by a meaningful amount
        self.assertGreater(Tend, T0)
        self.assertGreater(Tend - T0, 0.01)   # ≥ 1 percentage point increase

    def test_trust_collapses_with_persistent_misinfo_attention(self):
        """
        Regime test: if agents consistently allocate only misinfo attention,
        global trust should decrease noticeably (not necessarily to 0).
        """
        params = TrustCommonsParams(
            N=200,
            random_seed=123,
        )
        model = TrustCommonsABM(params)

        T0 = model.T
        steps = 200

        # force all attention to misinfo at each step
        for _ in range(steps):
            for a in model.agent_lookup.values():
                a.g = 0.0
                a.m = params.A_bar
            model._update_global_trust()

        Tend = model.T

        # trust should go down by a meaningful amount
        self.assertLess(Tend, T0)
        self.assertLess(Tend - T0, -0.01)    # ≥ 1 percentage point drop

    def test_local_trust_heterogeneity_positive_when_xi_positive(self):
        """
        With xi_local_heterogeneity > 0 and some variation in credibility,
        the variance of T_local across agents should become > 0.
        """
        params = TrustCommonsParams(
            N=40,
            avg_degree=6,
            small_world_rewire_p=0.2,
            xi_local_heterogeneity=0.5,
            enable_rewiring=False,
            random_seed=21,
        )
        model = TrustCommonsABM(params)

        # Induce some credibility variation
        for i, a in model.agent_lookup.items():
            a.c = (i % 10) / 10.0  # structured heterogeneity

        # Update local trust based on heterogeneous c
        model._update_local_trust()

        T_locals = np.array([a.T_local for a in model.agent_lookup.values()])
        self.assertGreater(np.var(T_locals), 0.0)

    def test_rewiring_changes_some_edges_when_prob_is_high(self):
        """
        With high reconsideration probability and homophily-based adaptive rewiring on,
        the edge set should change after enough steps.
        """
        params = TrustCommonsParams(
            N=40,
            random_seed=123,
            rewiring_mode="adaptive",   # use adaptive rule
            enable_rewiring=True,
            p_rewire=0.8,               # high attempt probability per activated node
            trust_update_interval=1,    # update trust every step (so c, T, T_local evolve)
            # keep other params at defaults or your usual test values
        )
        model = TrustCommonsABM(params)

        # record initial undirected edge set (i < j to avoid orientation issues)
        initial_edges = {tuple(sorted(e)) for e in model.G.edges()}

        # run for enough steps so cascades + bounded-confidence updates
        # generate some credibility heterogeneity and rewiring opportunities
        steps = 200
        for _ in range(steps):
            model.step()

        final_edges = {tuple(sorted(e)) for e in model.G.edges()}

        # there should be at least one changed edge
        diff_edges = initial_edges.symmetric_difference(final_edges)
        self.assertGreater(
            len(diff_edges),
            0,
            "Adaptive rewiring with high probability should change some edges."
        )

    def test_attention_allocation_respects_preference_extremes(self):
        """
        Sanity: for eta=1, credible attention dominates; for eta=0, misinfo dominates,
        regardless of T (within numerical tolerance).
        """
        A_bar = 1.0
        lam = 2.0

        # eta=1 → g >> m
        for T in [0.0, 0.5, 1.0]:
            g, m = allocate_attention(eta_i=1.0, T_i=T, A_bar=A_bar, lambda_trust_price=lam)
            self.assertGreater(g, m)

        # eta=0 → m >> g
        for T in [0.0, 0.5, 1.0]:
            g, m = allocate_attention(eta_i=0.0, T_i=T, A_bar=A_bar, lambda_trust_price=lam)
            self.assertGreater(m, g)


if __name__ == "__main__":
    unittest.main(verbosity=2)
