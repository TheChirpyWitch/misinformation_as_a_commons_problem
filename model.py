import random
from typing import Dict

import networkx as nx
import numpy as np
from mesa import Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector

from params import TrustCommonsParams, clip01
from agent import TrustCommonsAgent


class TrustCommonsABM(Model):
    """
    Trust-as-a-commons ABM.

    One step:
      1. Credibility updates over edges (bounded confidence + signals)
      2. Adaptive rewiring (optional)
      3. Attention allocation for all agents
      4. Global trust update
      5. Local trust update (with or without heterogeneity)
      6. Preference imitation
    """

    def __init__(self, params: TrustCommonsParams):
        super().__init__()
        self.params = params

        # Seed Python and NumPy randomness
        self.random = random.Random(params.random_seed)
        np.random.seed(params.random_seed)

        # Global trust stock
        self.T: float = 0.5

        self.time_step = 0  # counts node-activation steps

        # Network
        self.G = self._init_network()

        # Scheduler + agents
        self.schedule = RandomActivation(self)
        self.agent_lookup: Dict[int, TrustCommonsAgent] = {}

        for node_id in self.G.nodes():
            agent = TrustCommonsAgent(unique_id=node_id, model=self)
            self.schedule.add(agent)
            self.agent_lookup[node_id] = agent

        # Initialize local trust + attention
        self._update_local_trust()
        for agent in self.agent_lookup.values():
            agent.update_attention()

        # Data collector
        self.datacollector = DataCollector(
            model_reporters={
                "T":       lambda m: m.T,
                "mean_c":  lambda m: np.mean([a.c for a in m.agent_lookup.values()]),
                "var_T_local": lambda m: np.var([a.T_local for a in m.agent_lookup.values()]),
                "mean_eta":    lambda m: np.mean([a.eta for a in m.agent_lookup.values()]),
                "g_sum":   lambda m: sum(a.g for a in m.agent_lookup.values()),
                "m_sum":   lambda m: sum(a.m for a in m.agent_lookup.values()),
            },
            agent_reporters={
                "c":       "c",
                "eta":     "eta",
                "T_local": "T_local",
                "g":       "g",
                "m":       "m",
            },
        )

        # --- cascade state (for node-by-node propagation) ---
        self.cascade_active = False
        self.cascade_frontier = []   # queue of node ids to process
        self.cascade_visited = set() # nodes already hit in current cascade

        # --- instrumentation state ---
        # rewiring
        self.rewire_attempts = 0
        self.rewire_successes = 0
        self.rewire_log = []  # list of dicts, one per successful rewiring event

        # track initial edge set for "edges changed" stats
        self.initial_edges = {tuple(sorted(e)) for e in self.G.edges()}

        # cascades
        self.cascade_id = 0
        self.cascade_sizes = []   # how many nodes each cascade touched

        # you probably already have these, but make sure they exist:
        self.cascade_active = False
        self.cascade_frontier = []
        self.cascade_visited = set()


    # ---------- Network ----------
    def _init_network(self) -> nx.Graph:
        p = self.params

        # Ensure k is valid for Watts–Strogatz: 0 < k < N and k even
        k = min(p.avg_degree, p.N - 1)
        if k <= 0:
            k = 2 if p.N >= 3 else max(1, p.N - 1)
        if k % 2 == 1 and p.N > 2:
            k -= 1  # make it even

        G = nx.watts_strogatz_graph(
            n=p.N,
            k=k,
            p=p.small_world_rewire_p,
            seed=p.random_seed,
        )
        return G

    # ---------- Cascade logic: pick seed, then propagate ----------

    def _start_new_cascade(self):
        """Pick a random seed node and start a new propagation cascade."""
        nodes = list(self.G.nodes())
        if not nodes:
            self.cascade_active = False
            return

        seed = self.random.choice(nodes)
        self.cascade_active = True
        self.cascade_frontier = [seed]
        self.cascade_visited = {seed}

    def _cascade_step_one_node(self):
        """
        One cascade micro-step:
          - take ONE node from the frontier
          - update its credibility via interactions with its neighbors
          - add unseen neighbors to the frontier
        This is the only place where c changes in a given time step.
        """
        if not self.cascade_active or not self.cascade_frontier:
            # cascade exhausted
            if self.cascade_active:
                # log coverage for the finished cascade
                self.cascade_sizes.append(len(self.cascade_visited))
            self.cascade_active = False
            return

        i = self.cascade_frontier.pop(0)  # BFS-style queue
        ai = self.agent_lookup[i]
        p = self.params
        A_bar = p.A_bar

        # interact with each neighbor of i
        for j in self.G.neighbors(i):
            aj = self.agent_lookup[j]

            # attention fractions relative to Ā
            I_g_ij = ai.g / A_bar if A_bar > 0 else 0.0
            I_m_ij = ai.m / A_bar if A_bar > 0 else 0.0
            I_g_ji = aj.g / A_bar if A_bar > 0 else 0.0
            I_m_ji = aj.m / A_bar if A_bar > 0 else 0.0

            # bounded-confidence credibility update for this pair
            ai.bounded_confidence_update(
                other=aj,
                I_g_self_from_other=I_g_ji,
                I_m_self_from_other=I_m_ji,
                I_g_other_from_self=I_g_ij,
                I_m_other_from_self=I_m_ij,
            )

            # grow cascade frontier
            if j not in self.cascade_visited:
                self.cascade_visited.add(j)
                self.cascade_frontier.append(j)

        # rewiring from activated node i
        mode = getattr(self.params, "rewiring_mode", "adaptive")
        if mode != "none":
            self._maybe_rewire_from_node(i)

    def _maybe_rewire_from_node(self, i: int) -> bool:
        """
        Attempt rewiring from node i, depending on rewiring_mode and p_rewire.
        Returns True if a rewiring actually happened.
        """
        p = self.params

        # global attempt probability
        if self.random.random() > getattr(p, "p_rewire", 0.1):
            return False

        mode = getattr(p, "rewiring_mode", "adaptive")
        self.rewire_attempts += 1

        if mode == "adaptive":
            success = self._adaptive_rewiring_from_node(i, mode="adaptive")
        elif mode == "random":
            success = self._random_rewiring_from_node(i, mode="random")
        else:
            return False

        if success:
            self.rewire_successes += 1

        return success

    def _random_rewiring_from_node(self, i: int, mode: str = "random") -> bool:
        """
        Degree-preserving random rewiring for node i.
        Returns True if a rewiring happened.
        """
        G = self.G
        neighbors = list(G.neighbors(i))
        if not neighbors:
            return False

        j = self.random.choice(neighbors)
        G.remove_edge(i, j)

        candidates = [k for k in G.nodes() if k != i and not G.has_edge(i, k)]
        if not candidates:
            G.add_edge(i, j)
            return False

        k_new = self.random.choice(candidates)
        G.add_edge(i, k_new)

        ai = self.agent_lookup[i]
        aj = self.agent_lookup[j]
        ak = self.agent_lookup[k_new]

        self.rewire_log.append({
            "step": self.time_step,
            "mode": mode,
            "i": i,
            "j_old": j,
            "k_new": k_new,
            "c_i_before": ai.c,
            "c_j": aj.c,
            "c_k": ak.c,
        })
        return True

    def _adaptive_rewiring_from_node(self, i: int, mode: str = "adaptive") -> bool:
        """
        At most ONE homophily-based rewiring event for node i.
        Returns True if a rewiring happened.
        """
        p = self.params
        G = self.G

        neighbors = list(G.neighbors(i))
        if not neighbors:
            return False

        j = self.random.choice(neighbors)
        ai = self.agent_lookup[i]
        aj = self.agent_lookup[j]

        drop_prob = min(1.0, abs(ai.c - aj.c))
        if self.random.random() > drop_prob:
            return False

        # remove old tie
        G.remove_edge(i, j)

        # choose new partner for i based on homophily in c
        candidates = [k for k in G.nodes() if k != i and not G.has_edge(i, k)]
        if not candidates:
            # restore old tie if no candidates
            G.add_edge(i, j)
            return False

        c_i = ai.c
        beta = p.beta_homophily
        dists = np.array([abs(c_i - self.agent_lookup[k].c) for k in candidates])
        weights = np.exp(-beta * dists)
        if weights.sum() <= 0:
            k_new = self.random.choice(candidates)
        else:
            weights = weights / weights.sum()
            k_new = np.random.choice(candidates, p=weights)

        G.add_edge(i, k_new)

        # log event
        self.rewire_log.append({
            "step": self.time_step,
            "mode": mode,
            "i": i,
            "j_old": j,
            "k_new": k_new,
            "c_i_before": c_i,
            "c_j": aj.c,
            "c_k": self.agent_lookup[k_new].c,
        })
        return True



    # ---------- One step ----------
    def step(self):
        """
        One model time step:
          - if no active cascade, start a new one from a random seed
          - otherwise, propagate cascade by activating ONE node
          - then update attention, trust, local trust, imitation
        """
        # start new cascade if needed
        if not self.cascade_active:
            self._start_new_cascade()

        # one node activation along the current cascade
        self._cascade_step_one_node()

        # recompute attention everywhere (given updated c & T_local)
        self._allocate_attention_all_agents()

        # global + local trust updates
        self._update_global_trust()
        self._update_local_trust()

        # preference imitation (could also be local if you prefer)
        self._imitate_preferences()

        # advance Mesa scheduler time (agents' .step() is no-op)
        self.schedule.step()

        # record data
        self.datacollector.collect(self)


    # ---------- Step 1: Credibility ----------
    def _credibility_updates(self):
        p = self.params
        A_bar = p.A_bar

        edges = list(self.G.edges())
        self.random.shuffle(edges)

        for i, j in edges:
            ai = self.agent_lookup[i]
            aj = self.agent_lookup[j]

            # "Fractions" relative to Ā
            I_g_ij = ai.g / A_bar if A_bar > 0 else 0.0
            I_m_ij = ai.m / A_bar if A_bar > 0 else 0.0

            I_g_ji = aj.g / A_bar if A_bar > 0 else 0.0
            I_m_ji = aj.m / A_bar if A_bar > 0 else 0.0

            ai.bounded_confidence_update(
                other=aj,
                I_g_self_from_other=I_g_ji,
                I_m_self_from_other=I_m_ji,
                I_g_other_from_self=I_g_ij,
                I_m_other_from_self=I_m_ij,
            )

    def _adaptive_rewiring_step(self):
        """
        Adaptive rewiring pass on a subset of nodes.
        Number of nodes considered is proportional to p_rewire.
        """
        p = self.params
        G = self.G

        if G.number_of_edges() == 0 or p.p_rewire <= 0.0:
            return

        nodes = list(G.nodes())
        # number of nodes to consider this slow step
        k = max(1, int(p.p_rewire * len(nodes)))
        to_consider = self.random.sample(nodes, k)

        for i in to_consider:
            self._adaptive_rewiring_from_node(i)

    # ---------- Step 3: Attention allocation ----------
    def _allocate_attention_all_agents(self):
        for agent in self.agent_lookup.values():
            agent.update_attention()

    # ---------- Step 4: Global trust ----------
    def _update_global_trust(self):
        """
        ΔT = dt * ( alpha * avg_I_g * (1 - T) - beta * avg_I_m * T )
        evaluated only every `trust_update_interval` node activations.
        """
        if getattr(self.params, "fixed_T", False):
            return
        p = self.params
        agents = list(self.agent_lookup.values())
        N = len(agents)
        if N == 0:
            return

        sum_I_g = 0.0
        sum_I_m = 0.0
        for a in agents:
            total = a.g + a.m
            if total > 0:
                I_g_i = a.g / total
                I_m_i = a.m / total
            else:
                I_g_i = 0.0
                I_m_i = 0.0
            sum_I_g += I_g_i
            sum_I_m += I_m_i

        avg_I_g = sum_I_g / N
        avg_I_m = sum_I_m / N

        alpha = getattr(p, "alpha_trust_up", 1.0)
        beta  = getattr(p, "beta_trust_down", 2.0)
        dt    = getattr(p, "dt_trust", 0.005)   # note: much smaller now

        dT = dt * (alpha * avg_I_g * (1.0 - self.T) - beta * avg_I_m * self.T)
        self.T = clip01(self.T + dT)

    # ---------- Step 5: Local trust ----------
    def _update_local_trust(self):
        """
        Update local trust T_local_i based on global trust T and local credibility.

        - xi_local_heterogeneity (xi) controls mix of global T vs local credibility.
        - w_self_local_signal (w_self) mixes self c_i and neighbor-average c̄_N(i).
        - gamma_local_trust (gamma) is an inertia parameter:
            T_local_i(t+1) = (1-gamma)*T_local_i(t) + gamma*T_target_i
        """

        p = self.params
        xi    = getattr(p, "xi_local_heterogeneity", 0.0)
        gamma = getattr(p, "gamma_local_trust", 0.1)
        w_self = getattr(p, "w_self_local_signal", 0.4)

        # Homogeneous baseline: no local structure, T_i^ℓ = T^g
        if xi <= 0.0:
            for a in self.agent_lookup.values():
                a.T_local = self.T
            return

        G = self.G

        # global mean credibility at current step
        c_bar = np.mean([a.c for a in self.agent_lookup.values()])

        for i, a in self.agent_lookup.items():
            neighbors = list(G.neighbors(i))
            if neighbors:
                neigh_mean_c = np.mean([self.agent_lookup[j].c for j in neighbors])
            else:
                neigh_mean_c = a.c  # fallback: only self

            # local credibility signal: mix of self and neighbors
            s_i = w_self * a.c + (1.0 - w_self) * neigh_mean_c

            # target local trust: mix of global T and local signal
            T_target = (1.0 - xi) * self.T + xi * s_i
            T_target = clip01(T_target)

            # inertia: smooth update toward target
            a.T_local = clip01((1.0 - gamma) * a.T_local + gamma * T_target)

    # ---------- Step 6: Preference imitation ----------
    def _imitate_preferences(self):
        """
        Local imitation with mutation:
          - each agent i samples one neighbor j
          - with prob p_imitation, move η_i partly toward η_j plus noise
          - keeps heterogeneity alive instead of collapsing to consensus
        """
        p = self.params
        if getattr(p, "p_imitation", 0.0) <= 0.0:
            return

        rho = getattr(p, "rho_eta_update", 0.3)      # strength of update toward neighbor
        sigma = getattr(p, "sigma_eta_mutation", 0.05)  # mutation noise

        # we'll compute new etas in a buffer to avoid order effects
        new_eta = {}

        for i, a in self.agent_lookup.items():
            neighbors = list(self.G.neighbors(i))
            if not neighbors or self.random.random() > p.p_imitation:
                new_eta[i] = a.eta
                continue

            j = self.random.choice(neighbors)
            aj = self.agent_lookup[j]

            # target = neighbor's eta plus small noise
            target = aj.eta + self.random.gauss(0, sigma)

            eta_updated = (1.0 - rho) * a.eta + rho * target
            new_eta[i] = clip01(eta_updated)

        # apply updates
        for i, eta_val in new_eta.items():
            self.agent_lookup[i].eta = eta_val
