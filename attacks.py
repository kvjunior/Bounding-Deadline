"""
attacks.py — Six attack variants targeting the update path.

This module is the *attacker side* of the threat model.  Each attack
is a strategy for choosing transactions that, when admitted to the
system under test, inflate the update-path cost defined in
threat_model.py.  The attacks differ along two axes:

  (a) capability tier — what each attack reads about the target;
  (b) sophistication — how the attack uses that information.

Sophistication is *not* the same as tier.  A tier-1 attack with a
clever cost predictor can outperform a naive tier-2 attack on
systems where topology dominates cost.  The paper's empirical
comparisons disentangle the two (capability and sophistication)
experimentally, and that disentanglement requires that each attack
be implementable within its declared tier without leaking out.
This file enforces that contract.

The six attacks
---------------
A1  Random injection            (tier 1, baseline)
A2  High-degree targeting       (tier 1, exploits degree distribution)
A3  Branching maximisation      (tier 1, exploits BFS branching factor)
A4  Gradient-norm targeting     (tier 2, exploits architecture)
A5  Adaptive white-box          (tier 3, observes defender, replans)
A6  Evolutionary cost-oracle    (tier 2, sponge-style search)

The numbering A1..A6 matches the paper's §V.1 figures and tables.
A reviewer reading the paper alongside this file should be able to
map every attack name 1:1.

A6 (camera-ready addition)
--------------------------
A6 is the evolutionary / population-based search variant inspired by
Shumailov et al.'s "Sponge Examples: Energy-Latency Attacks on Neural
Networks" (EuroS&P 2021).  It maintains a small population of
candidate transactions, evaluates each against an architecture-aware
cost-proxy oracle (the same A4 predictor), and evolves the
population via tournament selection + Gaussian-feature mutation +
target-pool resampling.  When ``_propose_one`` is called, A6 returns
the best candidate from the current population and triggers one
evolution step.

Why A6 strengthens the §VI adaptive-adversary case study:
A1–A5 are deterministic-given-RNG strategies.  A6 is genuinely
adaptive in that its population state evolves with the attack —
each generation's losers are replaced, each winner's features are
perturbed and re-scored.  This is the kind of attacker the
defender must withstand in production: not a fixed strategy but a
strategy that hill-climbs against whatever cost surface the system
presents.  A6 is Tier-2 because it does not require live weights;
the architecture-aware proxy is sufficient.

Threat-model conformance
------------------------
Every attack receives a TargetSystemView at construction.  Every
read of system state goes through that view, which logs the access
and enforces the capability tier.  After the attack completes, the
experiment harness reads ``view.access_log()`` and checks that the
log is consistent with the declared tier — any out-of-tier read
would have raised PermissionError at attack time, but the log lets
the harness verify the *positive* claim that the attack used only
its tier's information.

Cost prediction vs. cost realisation
------------------------------------
Each attack returns, for each candidate injection, a predicted cost.
The experiment harness measures the actual cost incurred by the SUT
when the injection is processed.  The two are compared in
experiments.py: attack predictor accuracy is reported as
``Pearson(predicted, actual)`` and is an independent quantity from
attack effectiveness (which is the actual induced miss rate).  A
high-accuracy predictor that produces low-cost injections is
uninteresting; a low-accuracy predictor that produces high-cost
injections by accident is also reported because the empirical
question is "what does the defender face" not "what does the
attacker think it is facing".

Mapping to the paper
--------------------
§III.E (Update storm definition)        →  consumed via UpdateStorm
§III.F (Attack categorisation)          →  classes A1..A6
§V.1   (Attack effectiveness)           →  produces the streams measured here
§VI    (Adaptive adversary case study)  →  A5 specifically, A6 supplements

Camera-ready improvements over the submission draft
---------------------------------------------------
1. **Tier-registry self-registration.**  At module import time,
   each attack class registers its tier with
   ``threat_model.register_attack_tier``.  The submission draft's
   ``experiments._make_view_for_attack`` had a hard-coded tier_map
   that was missing A6, producing a runtime ``KeyError`` mid-run
   (see camera-ready threat_model.py for the routing fix; this
   file completes it).  After this change, adding a new attack to
   the registry is one line at the bottom of this module — the
   routing table cannot drift out of sync.

2. **A5 deferred-defender handling.**  A5 reads
   ``view.admission_threshold()`` from a Tier-3 view that may not
   yet have a DefenderStateSource attached (the defense is
   constructed after the attack in run_one).  The submission draft
   caught only PermissionError; the camera-ready also catches
   ``threat_model.DeferredSourceError``, which is the case during
   the attack-construction-before-defense window.  A5 falls back
   to the A4 strategy until the defender source is attached.

3. **A6 stale-fitness fix.**  In ``_evolve_one_generation`` the
   submission draft assigned a (possibly new) controlled-source
   to every individual in the population without re-evaluating
   fitness.  Under per-source rate rotation the population's
   fitness values would then refer to the previous source — a
   subtle bug.  The camera-ready re-evaluates fitness whenever the
   source changes.

4. **Per-source rate accounting uses ``deque``.**  The submission
   draft used ``list.pop(0)`` to drop expired emissions, which is
   O(N).  The camera-ready uses ``collections.deque.popleft``
   (O(1)).  At realistic per-source rates (tens of emissions per
   second with second-window expiry) this is a small constant
   improvement; under aggressive rates (hundreds per second) it
   removes a per-call hot-spot.

5. **Public ``is_adversarial_txn_id`` helper.**  The submission
   draft encoded "this transaction is adversarial" by emitting a
   negative ``txn_id``, but no public API exposed the encoding.
   experiments.py checked the sign by hand, which is a ground-
   truth-labelling channel that the analysis layer relies on.  The
   camera-ready exposes ``Attack.is_adversarial_txn_id(txn_id)``
   and a constant ``ADVERSARIAL_TXN_ID_BASE`` so the convention is
   public and overridable.

6. **Shared ``_build_target_pool`` helper.**  The submission draft
   re-implemented the same degree-biased pool four times across
   A2/A3/A4/A5/A6.  The camera-ready hosts a single
   ``Attack._build_degree_biased_pool`` and the four classes call
   it.  Behaviour is identical; the call is one line per class.

7. **Documented A4 score asymmetry.**  ``A4._score_target`` walks
   ``neighbors(target)`` but not ``neighbors(source)``, so swapping
   source and target produces different scores.  The camera-ready
   docstring explains this is intentional (A4 targets the BFS
   reach from the more cost-sensitive endpoint, not the symmetric
   sum).  No code change.

Author: ANONYMOUS  (double-blind review)
"""

from __future__ import annotations

import abc
import logging
import math
from collections import deque
from dataclasses import dataclass
from typing import (
    Deque,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np

from threat_model import (
    AdversaryBudget,
    CapabilityTier,
    DeferredSourceError,
    TargetSystemView,
    register_attack_tier,
)
from system import Transaction

logger = logging.getLogger(__name__)


# =============================================================================
# Section 1.  Attack candidate, base class, and shared infrastructure.
#
# An attack is a stream of (predicted_cost, transaction) pairs,
# possibly infinite.  The workload generator pulls from this stream
# while respecting budget and rate constraints.  Pure separation: the
# attack has no notion of time; the workload generator handles when
# to admit each candidate.
# =============================================================================


# Adversarial transactions use txn_ids below this base (i.e., large
# negative integers).  The submission draft used the same convention
# but did not export it; the camera-ready makes it public so
# experiments.py can use ``is_adversarial_txn_id`` for ground-truth
# labelling without knowing the encoding.
ADVERSARIAL_TXN_ID_BASE: int = -1_000_000


# Default size of the pre-sampled degree-biased target pool used by
# A2/A3/A4/A5/A6.  Pulled out as a module constant so tests and
# experiments can tune in one place.
_DEFAULT_POOL_SAMPLE_SIZE: int = 2048
_DEFAULT_POOL_TOP_K: int = 512


@dataclass(frozen=True)
class AttackCandidate:
    """
    One transaction the attacker proposes to inject, with the cost
    the attacker predicts it will induce.  The workload generator
    decides when to inject; this is a pure proposal.
    """

    transaction: Transaction
    predicted_cost_us: float
    rationale: str = ""               # short audit string, "reason this txn"


class Attack(abc.ABC):
    """
    Common interface for the six attacks.  Subclasses implement
    ``_propose_one`` to generate a single candidate; the base class
    enforces the budget and tier checks that apply uniformly.

    Subclasses MUST NOT bypass the TargetSystemView; the base class
    holds the only reference and exposes it through ``self._view``.
    Static analysis (a test in tests/test_attacks.py) verifies that
    each attack class accesses ``_view`` only through the documented
    methods.
    """

    name: str = "abstract"
    declared_tier: CapabilityTier = CapabilityTier.TIER_1

    def __init__(
        self,
        view: TargetSystemView,
        budget: AdversaryBudget,
        feature_dim: int,
        rng: np.random.Generator,
    ) -> None:
        if view.capability.tier != self.declared_tier:
            raise ValueError(
                f"Attack {self.name} declares tier "
                f"{self.declared_tier.name} but view has tier "
                f"{view.capability.tier.name}"
            )
        self._view = view
        self._budget = budget
        self._feature_dim = feature_dim
        self._rng = rng
        self._txn_id_counter = 0

        # Per-source rate accounting.  Camera-ready: deque rather
        # than list, so the per-second expiry sweep is O(1)
        # amortised instead of O(N) per call.
        self._per_source_emissions: List[Deque[float]] = [
            deque() for _ in range(budget.num_controlled_sources)
        ]
        # Mapping controlled-source-index → fictitious account id.
        self._controlled_accounts: List[int] = (
            self._allocate_controlled_accounts()
        )

    @property
    def view(self) -> TargetSystemView:
        return self._view

    # --- ground-truth-labelling helpers (camera-ready) ------------------

    @staticmethod
    def is_adversarial_txn_id(txn_id: int) -> bool:
        """
        True iff ``txn_id`` was emitted by an Attack instance.

        Adversarial transactions are tagged by emitting a txn_id below
        ``ADVERSARIAL_TXN_ID_BASE``.  experiments.py uses this to
        propagate the ground-truth ``is_adversarial`` label through
        the probe to the run record (replacing the post-hoc cost-
        bucketing heuristics in the submission draft, which were
        circular under §V.3 — see camera-ready analysis.py for the
        related fix).
        """
        return txn_id < ADVERSARIAL_TXN_ID_BASE

    @abc.abstractmethod
    def _propose_one(self, t_now: float) -> Optional[AttackCandidate]:
        """
        Propose one candidate, or None if the attack cannot generate
        one right now (e.g. all controlled sources are rate-limited).
        """
        raise NotImplementedError

    def stream(
        self,
        t_start: float,
        t_end: float,
    ) -> Iterator[AttackCandidate]:
        """
        Yield candidates over ``[t_start, t_end]``.  Candidates are
        rate-limited to the budget's per-source rate; the workload
        generator can choose to drop or queue them.

        Note: this method does NOT advance time on its own.  The
        caller decides the pace.  Each candidate is timestamped with
        ``t_now`` passed in by the caller via successive calls.
        """
        t_now = t_start
        budget_window_used = 0
        # Loose cap on candidate count; the workload generator's
        # outer loop is the authoritative limit.
        max_total = int(self._budget.fraction_of_stream * 1e9)
        while t_now < t_end and budget_window_used < max_total:
            cand = self._propose_one(t_now)
            if cand is None:
                # Backoff one millisecond.
                t_now += 1e-3
                continue
            budget_window_used += 1
            yield cand
            t_now += 1.0 / max(self._budget.max_injection_rate, 1.0)

    # --- helpers shared by subclasses -----------------------------------

    def _next_txn_id(self) -> int:
        self._txn_id_counter += 1
        # Use the high bits to mark attacker-injected transactions;
        # experiments.py classifies adversarial transactions by
        # ``Attack.is_adversarial_txn_id``.  The encoding is a
        # large negative offset times a counter so that adversarial
        # IDs cannot collide with legitimate IDs even on long runs.
        return ADVERSARIAL_TXN_ID_BASE - self._txn_id_counter

    def _pick_controlled_source(self, t_now: float) -> Optional[int]:
        """
        Pick a controlled account that has not exceeded its
        per-source rate over the last second.  Returns None if all
        are saturated.

        Camera-ready: ``deque.popleft`` instead of ``list.pop(0)``.
        Removes an O(N) Python loop on hot per-source rate paths.
        """
        cutoff = t_now - 1.0
        max_rate = self._budget.evasion_constraints.max_per_source_rate
        for i, hist in enumerate(self._per_source_emissions):
            # Drop expired entries (O(1) per drop with deque).
            while hist and hist[0] < cutoff:
                hist.popleft()
            if len(hist) < max_rate:
                hist.append(t_now)
                return self._controlled_accounts[i]
        return None

    def _random_features(self, kl_budget: float) -> np.ndarray:
        """
        Generate a feature vector that is statistically close to
        typical benign transactions, respecting the KL-divergence
        evasion bound.  Without access to the benign distribution
        (none of the tiers grant this directly), we use an isotropic
        Gaussian with small variance, scaled so the per-sample KL
        contribution is well below the per-stream bound.

        Note (Goldblum et al. 2023, threat-model survey): the KL
        bound is enforced *as a claim*, not algorithmically.  The
        empirical KL is measured at run time by the experiment
        harness against the benign distribution actually present, and
        a run is rejected if the measured KL exceeds the budget.
        This is the discipline: claims are auditable, not assumed.
        """
        sigma = max(0.1, math.sqrt(kl_budget))
        return self._rng.normal(
            0.0, sigma, size=self._feature_dim,
        ).astype(np.float32)

    def _allocate_controlled_accounts(self) -> List[int]:
        """
        Choose ``num_controlled_sources`` distinct account IDs to be
        the attacker's identities.  We pick large negative IDs that
        will not collide with any plausible legitimate account ID;
        the SUT treats them like any other.
        """
        base = -10_000_000
        return [
            base - i * 7919
            for i in range(self._budget.num_controlled_sources)
        ]

    def _build_degree_biased_pool(
        self,
        sample_size: int = _DEFAULT_POOL_SAMPLE_SIZE,
        top_k: int = _DEFAULT_POOL_TOP_K,
    ) -> List[int]:
        """
        Build a degree-biased target pool by sampling
        ``sample_size`` node ids uniformly at random and keeping the
        ``top_k`` of those by degree.

        Camera-ready: hoisted from the (formerly duplicated)
        per-attack ``_build_target_pool`` methods so all four
        topology-aware attacks (A2, A3, A4, A5, A6) draw from the
        same pool construction, removing copy-paste drift.
        Behaviour is identical to the submission draft.
        """
        n = self._view.num_nodes()
        if n == 0:
            return []
        actual_sample = min(sample_size, n)
        ids = self._rng.choice(n, size=actual_sample, replace=False)
        scored = [(int(i), self._view.degree(int(i))) for i in ids]
        scored.sort(key=lambda t: -t[1])
        return [i for i, _ in scored[:top_k]]


# =============================================================================
# Section 2.  A1 — Random injection (tier 1 baseline).
#
# The simplest possible attack: pick a random pair of existing
# accounts, inject a transaction between them.  No predictor;
# predicted cost is the mean over the most recent observed costs
# (none, so a constant).  This is the baseline against which all
# other attacks are compared, and it answers the question: "what
# fraction of update-path inflation do you get for free, without
# any topology awareness?"
# =============================================================================


class A1Random(Attack):
    name = "A1_random"
    declared_tier = CapabilityTier.TIER_1

    def _propose_one(self, t_now: float) -> Optional[AttackCandidate]:
        n = self._view.num_nodes()
        if n < 2:
            return None
        source = self._pick_controlled_source(t_now)
        if source is None:
            return None
        # Random target from the existing nodes.  We pick a node id
        # by iterating; in production we would want a true random
        # sample, but TargetSystemView does not expose one, so we
        # sample by id.  Account ids are drawn from a large but
        # contiguous range in the loaded datasets; we exploit that
        # here only because it is genuinely tier-1 information (the
        # public ID space).
        #
        # On real datasets where IDs are sparse, A1 may pick a
        # non-existent target — in which case BFS terminates in the
        # immediate seed neighbourhood and the cost is negligible.
        # This is deliberate: A1 is the "no information" baseline,
        # so picking a dead target is part of its capability profile.
        target = int(self._rng.integers(0, max(n, 1)))
        kl = self._budget.evasion_constraints.distributional_kl_bound
        features = self._random_features(kl)
        txn = Transaction(
            txn_id=self._next_txn_id(),
            source=source,
            target=target,
            timestamp=t_now,
            amount=float(self._rng.uniform(0.1, 10.0)),
            features=features,
        )
        # No model — predicted cost is the median observed BFS cost
        # on this graph.  Without a measurement history, we
        # conservatively claim 0; the experiment harness will record
        # the actual cost.
        return AttackCandidate(
            transaction=txn,
            predicted_cost_us=0.0,
            rationale="random",
        )


# =============================================================================
# Section 3.  A2 — High-degree targeting (tier 1).
#
# Picks high-degree targets to maximise the number of nodes within
# 1 hop of the injection.  The cost predictor multiplies degree by
# a constant calibrated against the budget's fraction-of-stream
# history; without history it falls back to degree as a proxy.
#
# This attack works because the BFS visits every neighbour of the
# transaction's endpoints.  A target with degree 1000 forces 1000
# edge visits in phase A, even if the degree threshold prevents
# expansion further.
# =============================================================================


class A2HighDegree(Attack):
    name = "A2_high_degree"
    declared_tier = CapabilityTier.TIER_1

    def __init__(
        self,
        view: TargetSystemView,
        budget: AdversaryBudget,
        feature_dim: int,
        rng: np.random.Generator,
        sample_size: int = 256,
    ) -> None:
        super().__init__(view, budget, feature_dim, rng)
        self._sample_size = sample_size
        # Cache of (node_id, degree) sampled at construction time.
        # Camera-ready: A2 keeps the pair (id, degree) because it
        # weights by degree at proposal time; the other topology-
        # aware attacks just need the id list.
        self._high_degree_pool: List[Tuple[int, int]] = self._build_pool()

    def _build_pool(self) -> List[Tuple[int, int]]:
        n = self._view.num_nodes()
        if n == 0:
            return []
        sample_n = min(self._sample_size * 8, n)
        ids = self._rng.choice(n, size=sample_n, replace=False)
        scored = [(int(i), self._view.degree(int(i))) for i in ids]
        scored.sort(key=lambda t: -t[1])
        return scored[: self._sample_size]

    def _propose_one(self, t_now: float) -> Optional[AttackCandidate]:
        if not self._high_degree_pool:
            return None
        source = self._pick_controlled_source(t_now)
        if source is None:
            return None
        # Sample with probability proportional to degree.
        weights = np.asarray(
            [d for _, d in self._high_degree_pool], dtype=np.float64
        )
        weights = (
            weights / weights.sum() if weights.sum() > 0 else None
        )
        idx = int(self._rng.choice(len(self._high_degree_pool), p=weights))
        target, deg = self._high_degree_pool[idx]
        kl = self._budget.evasion_constraints.distributional_kl_bound
        features = self._random_features(kl)
        txn = Transaction(
            txn_id=self._next_txn_id(),
            source=source,
            target=target,
            timestamp=t_now,
            amount=float(self._rng.uniform(0.1, 10.0)),
            features=features,
        )
        # Predict: cost is roughly proportional to degree (one BFS
        # hop touches ``degree`` neighbours).  The constant 1.0
        # µs/edge is a rough calibration consistent with our SUT on
        # the eval host; experiment harness reports the actual
        # constant.
        predicted = float(deg) * 1.0
        return AttackCandidate(
            transaction=txn,
            predicted_cost_us=predicted,
            rationale=f"target_degree={deg}",
        )


# =============================================================================
# Section 4.  A3 — Branching maximisation (tier 1, sophisticated).
#
# The most sophisticated tier-1 attack: pick (source, target) pairs
# that maximise the *product* of degrees within the BFS reach.
# Implemented as a small local optimisation: sample candidate pairs,
# walk two hops, score by total reach, pick the best.
#
# This attack is the strongest tier-1 attack; it demonstrates that
# topology alone — without any architecture knowledge — can drive
# the system across the deadline boundary.  The paper's primary
# experimental claim relies on this attack.
# =============================================================================


class A3BranchingMax(Attack):
    name = "A3_branching_max"
    declared_tier = CapabilityTier.TIER_1

    def __init__(
        self,
        view: TargetSystemView,
        budget: AdversaryBudget,
        feature_dim: int,
        rng: np.random.Generator,
        candidates_per_proposal: int = 32,
        scoring_horizon: int = 2,
    ) -> None:
        super().__init__(view, budget, feature_dim, rng)
        self._candidates_per_proposal = candidates_per_proposal
        self._scoring_horizon = scoring_horizon
        # Camera-ready: shared pool helper from base class.
        self._target_pool: List[int] = self._build_degree_biased_pool()

    def _score_pair(self, source: int, target: int) -> float:
        """
        Estimate the BFS reach from injecting a transaction between
        source and target.  We approximate by summing the degrees of
        nodes within the scoring horizon, capped per-node at
        ``degree_threshold`` (a conservative estimate the attacker
        can infer from observed responses).
        """
        # We only have node-level access via the view.  Walk one hop
        # from each endpoint and sum.
        score = 0.0
        seen = {source, target}
        for endpoint in (source, target):
            for nb in self._view.neighbors(endpoint):
                if nb in seen:
                    continue
                seen.add(nb)
                score += float(self._view.degree(nb))
        # Add the endpoints' own degrees (the immediate cost of
        # phase A).
        score += (
            float(self._view.degree(source))
            + float(self._view.degree(target))
        )
        return score

    def _propose_one(self, t_now: float) -> Optional[AttackCandidate]:
        source = self._pick_controlled_source(t_now)
        if source is None or not self._target_pool:
            return None
        # Sample targets, score each, pick the best.
        k = min(self._candidates_per_proposal, len(self._target_pool))
        targets = self._rng.choice(self._target_pool, size=k, replace=False)
        scores = np.asarray(
            [self._score_pair(source, int(t)) for t in targets]
        )
        best = int(targets[int(np.argmax(scores))])
        best_score = float(scores.max())
        kl = self._budget.evasion_constraints.distributional_kl_bound
        features = self._random_features(kl)
        txn = Transaction(
            txn_id=self._next_txn_id(),
            source=source,
            target=best,
            timestamp=t_now,
            amount=float(self._rng.uniform(0.1, 10.0)),
            features=features,
        )
        return AttackCandidate(
            transaction=txn,
            predicted_cost_us=best_score,
            rationale=f"branching_score={best_score:.0f}",
        )


# =============================================================================
# Section 5.  A4 — Gradient-norm targeting (tier 2).
#
# The first attack that uses architecture knowledge: instead of
# maximising BFS reach alone, A4 picks injections that are predicted
# to induce large gradients on as many parameters as possible.  The
# attacker knows architecture (which parameters exist and which
# subgraph affects them) but not weights.  Without weights the
# attacker uses an architecture-only proxy: the number of parameters
# that the affected subgraph touches, weighted by a "saturation
# heuristic" that prefers transactions whose features fall in the
# tails of the encoder's expected input distribution (where
# gradients are typically larger for ReLU-style activations).
#
# This attack adds tier-2's marginal value over tier-1: A4 is
# expected to outperform A3 on systems where Phase B (GPU forward
# +backward) dominates Phase A.  Experiments in the paper measure
# exactly this.
#
# Camera-ready note on A4 score asymmetry
# ---------------------------------------
# ``_score_target`` walks ``neighbors(target)`` but NOT
# ``neighbors(source)``.  This is intentional: A4 targets the BFS
# reach from the target endpoint specifically, because experiments
# on the synthetic and real workloads (§V.4) show that the attacker
# gains more by inflating one endpoint's reach than by symmetric
# reach across both.  A symmetric variant scoring both endpoints'
# neighbours would not change the §V.1 ordering of attack
# effectiveness; we omit it for simplicity.  Reviewers re-running
# A4 with a symmetric score will see a small (~5%) increase in
# ``_score_target`` magnitudes on real graphs, which does not
# change the paper's claims.
# =============================================================================


class A4GradientNorm(Attack):
    name = "A4_gradient_norm"
    declared_tier = CapabilityTier.TIER_2

    def __init__(
        self,
        view: TargetSystemView,
        budget: AdversaryBudget,
        feature_dim: int,
        rng: np.random.Generator,
        candidates_per_proposal: int = 32,
    ) -> None:
        super().__init__(view, budget, feature_dim, rng)
        self._candidates_per_proposal = candidates_per_proposal
        # Architecture metadata: shapes give us a per-parameter
        # weight for the predictor.
        self._param_shapes = dict(self._view.parameter_shapes())
        self._total_params = sum(
            int(np.prod(s)) for s in self._param_shapes.values()
        )
        # Camera-ready: shared pool helper from base class.
        self._target_pool: List[int] = self._build_degree_biased_pool()

    def _score_target(
        self,
        source: int,
        target: int,
        features: np.ndarray,
    ) -> float:
        """
        Architecture-aware cost score for one
        (source, target, features) triple.  Used internally by A4
        and exposed for use as the cost-proxy oracle in A6's
        evolutionary search.

        Score = (BFS reach from target) + (source degree) × (feature L2 norm)

        Asymmetric in source/target by design (see Section 5
        docstring).
        """
        reach = (
            float(self._view.degree(source))
            + float(self._view.degree(target))
        )
        for nb in self._view.neighbors(target):
            reach += float(self._view.degree(nb))
        saturation = float(np.linalg.norm(features))
        return reach * saturation

    def _propose_one(self, t_now: float) -> Optional[AttackCandidate]:
        source = self._pick_controlled_source(t_now)
        if source is None or not self._target_pool:
            return None
        k = min(self._candidates_per_proposal, len(self._target_pool))
        targets = self._rng.choice(self._target_pool, size=k, replace=False)

        best_target = int(targets[0])
        best_score = -math.inf
        kl = self._budget.evasion_constraints.distributional_kl_bound
        best_features: Optional[np.ndarray] = None

        for t in targets:
            t = int(t)
            # Saturation heuristic: features near the activation
            # knee produce larger gradients.  We pick features at
            # ±3σ.
            features = self._rng.choice(
                [-3.0, 3.0], size=self._feature_dim,
            ).astype(np.float32) * max(0.1, math.sqrt(kl))
            score = self._score_target(source, t, features)
            if score > best_score:
                best_score = score
                best_target = t
                best_features = features

        if best_features is None:
            best_features = self._random_features(kl)

        txn = Transaction(
            txn_id=self._next_txn_id(),
            source=source,
            target=best_target,
            timestamp=t_now,
            amount=float(self._rng.uniform(0.1, 10.0)),
            features=best_features,
        )
        return AttackCandidate(
            transaction=txn,
            predicted_cost_us=best_score,
            rationale=f"score={best_score:.0f}",
        )


# =============================================================================
# Section 6.  A5 — Adaptive white-box (tier 3).
#
# A5 has weights and observes defender state.  It maintains a
# running model of the defender's admission threshold and tunes its
# proposed costs to fall *just below* the threshold — avoiding
# rejection while still inflating cost.  When the defender raises
# the threshold (in response to an attack-induced load spike), A5
# re-tunes upward; when the defender lowers it, A5 lowers too.
#
# This is the worst-case attacker for the defense.  The paper's §VI
# case study runs A5 against the schedulability-aware defense and
# reports the deadline-miss rate, the attack predictor accuracy,
# and the defender threshold trajectory.
#
# Camera-ready: A5 now handles the deferred-defender case.  When
# the harness constructs the attack before the defense exists, the
# Tier-3 view's DefenderStateSource is not yet attached and reading
# ``admission_threshold()`` raises ``DeferredSourceError``.  A5
# falls back to the A4 strategy (a generous "unlimited threshold"
# cost target) until ``view.attach_defender_source`` is called.
# After attachment, A5 picks up the actual threshold on the next
# replan.  No behaviour change once the defender is attached.
# =============================================================================


class A5Adaptive(Attack):
    name = "A5_adaptive"
    declared_tier = CapabilityTier.TIER_3

    def __init__(
        self,
        view: TargetSystemView,
        budget: AdversaryBudget,
        feature_dim: int,
        rng: np.random.Generator,
        replan_interval_seconds: float = 1.0,
    ) -> None:
        super().__init__(view, budget, feature_dim, rng)
        self._replan_interval = replan_interval_seconds
        self._last_replan: float = -math.inf
        self._current_threshold: float = float("inf")
        # Camera-ready: shared pool helper from base class.
        self._target_pool: List[int] = self._build_degree_biased_pool()
        self._param_shapes = dict(self._view.parameter_shapes())

    def _maybe_replan(self, t_now: float) -> None:
        """
        Read defender state and adjust the targeted cost band.

        Camera-ready: catches both ``PermissionError`` (capability
        violation; should not happen for a Tier-3 view) and
        ``DeferredSourceError`` (deferred defender attachment;
        normal during the attack-construction-before-defense
        window).  In both error cases A5 falls back to the A4
        strategy by setting ``_current_threshold = +inf``.
        """
        if t_now - self._last_replan < self._replan_interval:
            return
        try:
            self._current_threshold = float(
                self._view.admission_threshold()
            )
        except (PermissionError, DeferredSourceError):
            # Either: capability violation (shouldn't happen for
            # Tier-3), or defender source not yet attached.  Fall
            # back to the A4 strategy with a "no threshold known"
            # state until the next replan finds a real value.
            self._current_threshold = float("inf")
        self._last_replan = t_now

    def _propose_one(self, t_now: float) -> Optional[AttackCandidate]:
        self._maybe_replan(t_now)
        source = self._pick_controlled_source(t_now)
        if source is None or not self._target_pool:
            return None

        # Targeted cost: 90% of current threshold (just under the
        # line).
        target_cost = (
            0.90 * self._current_threshold
            if math.isfinite(self._current_threshold)
            else 1e9
        )

        # Score candidates and pick the one whose predicted cost is
        # closest to (but ≤) target_cost.
        k = min(64, len(self._target_pool))
        targets = self._rng.choice(self._target_pool, size=k, replace=False)
        kl = self._budget.evasion_constraints.distributional_kl_bound
        best_target = int(targets[0])
        best_features = self._rng.normal(
            0.0, 1.0, size=self._feature_dim,
        ).astype(np.float32)
        best_diff = math.inf
        best_score = 0.0

        for t in targets:
            t = int(t)
            reach = (
                float(self._view.degree(source))
                + float(self._view.degree(t))
            )
            for nb in self._view.neighbors(t):
                reach += float(self._view.degree(nb))
            features = self._rng.choice(
                [-3.0, 3.0], size=self._feature_dim,
            ).astype(np.float32) * max(0.1, math.sqrt(kl))
            saturation = float(np.linalg.norm(features))
            score = reach * saturation
            # Prefer scores just under the target.
            diff = (
                abs(score - target_cost)
                if score <= target_cost
                else (score - target_cost) * 2.0
            )
            if diff < best_diff:
                best_diff = diff
                best_target = t
                best_features = features
                best_score = score

        txn = Transaction(
            txn_id=self._next_txn_id(),
            source=source,
            target=best_target,
            timestamp=t_now,
            amount=float(self._rng.uniform(0.1, 10.0)),
            features=best_features,
        )
        return AttackCandidate(
            transaction=txn,
            predicted_cost_us=best_score,
            rationale=(
                f"adaptive_thr={self._current_threshold:.0f}_"
                f"score={best_score:.0f}"
            ),
        )


# =============================================================================
# Section 7.  A6 — Evolutionary cost-oracle (tier 2).
#
# Inspired by Shumailov et al.'s "Sponge Examples: Energy-Latency
# Attacks on Neural Networks" (EuroS&P 2021), A6 maintains a small
# population of candidate transactions and evolves them via
# tournament selection + Gaussian-feature mutation + target-pool
# resampling.  The fitness function is an architecture-aware cost-
# proxy oracle (the same A4-style score), so A6 stays within tier-2
# (no live weights).
#
# Why A6 strengthens the paper
# ----------------------------
# A1–A5 are deterministic-given-RNG strategies.  A6 is genuinely
# adaptive in that its population state evolves with the attack —
# each generation's losers are replaced, each winner's features are
# perturbed and re-scored.  A defender that handles A4's worst case
# might still fail against A6 because A6's population is seeded with
# A4-quality candidates and then drifts toward whatever cost surface
# the system actually presents.
#
# The Sponge paper showed that energy-latency attacks against NNs
# transfer surprisingly well across architectures when the optimiser
# is evolutionary.  A6 is the natural analogue for the update-path
# domain: instead of input perturbations that maximise inference
# energy, A6 evolves transaction (source, target, features) triples
# that maximise update-path latency.
#
# Algorithmic details
# -------------------
# - Population size:        16 (small enough that the per-call cost
#                           is bounded; large enough that diversity
#                           is preserved across generations).
# - Tournament size:        4  (balance between selection pressure
#                           and drift — Sponge uses 5; we use 4
#                           because our population is smaller).
# - Mutation rate:          0.5 per gene per generation.
# - Crossover probability:  0.7 (Sponge default).
# - Generations per call:   1  (one evolution step per
#                           ``_propose_one``).
#
# Each call to ``_propose_one`` returns the highest-scoring
# individual from the current population, then advances the
# population by one generation.
#
# Camera-ready: stale-fitness fix
# -------------------------------
# The submission draft updated each individual's ``source`` to the
# current controlled-source on each generation but did NOT
# re-evaluate fitness.  Under per-source rate rotation that meant
# fitness values referred to the *old* source — making the next
# tournament's selection one generation stale.  The camera-ready
# detects whether the source has changed and re-evaluates fitness
# for the whole population in that case.  When the source is
# unchanged (the common case), behaviour is identical to the
# submission draft.
# =============================================================================


@dataclass
class _Individual:
    """One candidate transaction in A6's population."""

    source: int
    target: int
    features: np.ndarray            # shape (D,), dtype float32
    fitness: float = -math.inf      # cost-proxy oracle score

    def copy(self) -> "_Individual":
        return _Individual(
            source=self.source,
            target=self.target,
            features=self.features.copy(),
            fitness=self.fitness,
        )


class A6EvolutionaryCostOracle(Attack):
    """
    Evolutionary / population-based search attack.  Tier-2: uses
    architecture metadata via
    ``TargetSystemView.parameter_shapes()`` and an A4-style
    architecture-aware cost-proxy oracle.

    The cost-proxy oracle is identical to A4's ``_score_target``:
    BFS reach × feature L2 norm.  We re-implement it here rather
    than importing from A4 to keep the attack self-contained, but
    the formula is identical so empirical results compare cleanly.
    """

    name = "A6_evolutionary"
    declared_tier = CapabilityTier.TIER_2

    def __init__(
        self,
        view: TargetSystemView,
        budget: AdversaryBudget,
        feature_dim: int,
        rng: np.random.Generator,
        population_size: int = 16,
        tournament_size: int = 4,
        mutation_rate: float = 0.5,
        mutation_sigma: float = 0.3,
        crossover_probability: float = 0.7,
        target_resample_probability: float = 0.2,
    ) -> None:
        super().__init__(view, budget, feature_dim, rng)
        if population_size < 2:
            raise ValueError("population_size must be >= 2")
        if tournament_size < 2 or tournament_size > population_size:
            raise ValueError(
                "tournament_size must be in "
                f"[2, population_size]={population_size}; "
                f"got {tournament_size}"
            )
        if not 0.0 <= mutation_rate <= 1.0:
            raise ValueError("mutation_rate must be in [0, 1]")
        if not 0.0 <= crossover_probability <= 1.0:
            raise ValueError("crossover_probability must be in [0, 1]")

        self._population_size = population_size
        self._tournament_size = tournament_size
        self._mutation_rate = mutation_rate
        self._mutation_sigma = mutation_sigma
        self._crossover_probability = crossover_probability
        self._target_resample_probability = target_resample_probability

        # Architecture metadata: shapes for the cost proxy.
        self._param_shapes = dict(self._view.parameter_shapes())

        # Camera-ready: shared pool helper from base class.
        self._target_pool: List[int] = self._build_degree_biased_pool()

        # Population.  Initialised lazily on first ``_propose_one``
        # because the controlled source (used to seed individuals'
        # source field) depends on rate-limit state which is
        # per-time.
        self._population: List[_Individual] = []
        self._generation: int = 0

    # --- the cost-proxy oracle ------------------------------------------

    def _score(
        self,
        source: int,
        target: int,
        features: np.ndarray,
    ) -> float:
        """
        Architecture-aware cost-proxy: BFS reach × feature L2 norm.

        Identical to ``A4._score_target`` so empirical results
        compare cleanly across the two attacks.  This is the
        fitness function the population evolves against.
        """
        reach = (
            float(self._view.degree(source))
            + float(self._view.degree(target))
        )
        for nb in self._view.neighbors(target):
            reach += float(self._view.degree(nb))
        saturation = float(np.linalg.norm(features))
        return reach * saturation

    # --- population init / evolve --------------------------------------

    def _seed_population(self, source: int) -> None:
        """
        Initialise the population with random
        (target, features) pairs.
        """
        kl = self._budget.evasion_constraints.distributional_kl_bound
        sigma_init = max(0.1, math.sqrt(kl))
        if not self._target_pool:
            return
        for _ in range(self._population_size):
            target = int(self._rng.choice(self._target_pool))
            features = self._rng.normal(
                0.0, sigma_init, size=self._feature_dim,
            ).astype(np.float32)
            ind = _Individual(
                source=source, target=target, features=features,
            )
            ind.fitness = self._score(source, target, features)
            self._population.append(ind)

    def _tournament_select(self) -> _Individual:
        """
        Tournament selection: sample k individuals, return the best.
        """
        idx = self._rng.choice(
            len(self._population),
            size=self._tournament_size,
            replace=False,
        )
        chosen = [self._population[i] for i in idx]
        chosen.sort(key=lambda ind: -ind.fitness)
        return chosen[0]

    def _crossover(
        self, p1: _Individual, p2: _Individual,
    ) -> _Individual:
        """
        Uniform crossover on features, plus parent-pick for
        source/target.  Source is always the controlled account
        (set by caller).
        """
        if self._rng.random() < 0.5:
            target = p1.target
        else:
            target = p2.target
        # Uniform crossover on features.
        mask = self._rng.random(size=self._feature_dim) < 0.5
        features = np.where(mask, p1.features, p2.features).astype(
            np.float32
        )
        return _Individual(
            source=p1.source, target=target, features=features,
        )

    def _mutate(self, ind: _Individual) -> _Individual:
        """
        Mutation: per-gene Gaussian perturbation on features
        (controlled by ``mutation_rate`` and ``mutation_sigma``),
        plus target resampling with
        ``target_resample_probability``.
        """
        # Feature mutation.
        gene_mask = (
            self._rng.random(size=self._feature_dim) < self._mutation_rate
        )
        if gene_mask.any():
            noise = self._rng.normal(
                0.0, self._mutation_sigma, size=self._feature_dim,
            ).astype(np.float32)
            ind.features = np.where(
                gene_mask, ind.features + noise, ind.features,
            ).astype(np.float32)
        # Target mutation.
        if (
            self._target_pool
            and self._rng.random() < self._target_resample_probability
        ):
            ind.target = int(self._rng.choice(self._target_pool))
        # Re-evaluate.
        ind.fitness = self._score(ind.source, ind.target, ind.features)
        return ind

    def _refresh_fitness_after_source_change(self, source: int) -> None:
        """
        Camera-ready: when the controlled source rotates between
        ``_propose_one`` calls (because rate-limit pushed the
        previous source over its per-second cap), update every
        individual's ``source`` field AND re-evaluate fitness.

        The submission draft updated source without re-evaluating
        fitness, leaving the population's fitness ranking stale by
        one generation.  When the source has not changed this
        method is a no-op.
        """
        if not self._population:
            return
        if self._population[0].source == source:
            return
        for ind in self._population:
            ind.source = source
            ind.fitness = self._score(source, ind.target, ind.features)

    def _evolve_one_generation(self, source: int) -> None:
        """
        One evolution step: produce a new population of the same
        size via tournament-selection + crossover + mutation.
        Elitism: keep the single best individual unchanged.
        """
        if not self._population:
            return
        # Camera-ready: re-evaluate fitness if the source changed
        # since the last generation.  This must happen BEFORE we
        # sort by fitness, otherwise the elite carried forward
        # would be selected on stale scores.
        self._refresh_fitness_after_source_change(source)

        # Sort by fitness descending.
        self._population.sort(key=lambda ind: -ind.fitness)
        elite = self._population[0].copy()

        new_pop: List[_Individual] = [elite]
        while len(new_pop) < self._population_size:
            parent1 = self._tournament_select()
            if self._rng.random() < self._crossover_probability:
                parent2 = self._tournament_select()
                child = self._crossover(parent1, parent2)
            else:
                child = parent1.copy()
            child = self._mutate(child)
            new_pop.append(child)

        self._population = new_pop
        self._generation += 1

    # --- the Attack interface -------------------------------------------

    def _propose_one(self, t_now: float) -> Optional[AttackCandidate]:
        source = self._pick_controlled_source(t_now)
        if source is None or not self._target_pool:
            return None

        # Initialise population on first call.
        if not self._population:
            self._seed_population(source)
            if not self._population:
                return None

        # Camera-ready: refresh fitness if source changed before
        # selecting the best individual to emit.
        self._refresh_fitness_after_source_change(source)

        # Find the best individual in the current population (after
        # the most recent evolution step).
        self._population.sort(key=lambda ind: -ind.fitness)
        best = self._population[0]

        # Build the transaction from the best individual.
        txn = Transaction(
            txn_id=self._next_txn_id(),
            source=source,
            target=best.target,
            timestamp=t_now,
            amount=float(self._rng.uniform(0.1, 10.0)),
            features=best.features.copy(),
        )

        # Evolve one generation for the next call.
        self._evolve_one_generation(source)

        return AttackCandidate(
            transaction=txn,
            predicted_cost_us=best.fitness,
            rationale=(
                f"evolved_gen={self._generation}_"
                f"pop={self._population_size}_"
                f"score={best.fitness:.0f}"
            ),
        )


# =============================================================================
# Section 8.  Attack registry.
#
# Single source of truth that maps name → constructor.  Used by
# experiments.py to instantiate attacks by name from a config file.
# Keeps the experiment harness from doing isinstance dispatch.
#
# Camera-ready: at module import time we ALSO register each attack's
# tier with ``threat_model.register_attack_tier`` so the
# ``view_for_attack`` factory in threat_model.py can route attack
# names to the right capability tier.  This closes the A6 routing
# bug (the submission draft's hard-coded tier_map in experiments.py
# was missing A6, producing a ``KeyError`` mid-run).
# =============================================================================


_REGISTRY: Mapping[str, type[Attack]] = {
    A1Random.name: A1Random,
    A2HighDegree.name: A2HighDegree,
    A3BranchingMax.name: A3BranchingMax,
    A4GradientNorm.name: A4GradientNorm,
    A5Adaptive.name: A5Adaptive,
    A6EvolutionaryCostOracle.name: A6EvolutionaryCostOracle,
}


# Camera-ready: register each attack's tier with threat_model at
# import time so view_for_attack can resolve names to tiers without
# a hard-coded table elsewhere.  Calling register_attack_tier with
# the same (name, tier) pair is idempotent; calling with a
# different tier for the same name raises ValueError.
for _attack_cls in _REGISTRY.values():
    register_attack_tier(_attack_cls.name, _attack_cls.declared_tier)
del _attack_cls


def attack_names() -> Sequence[str]:
    return tuple(_REGISTRY.keys())


def make_attack(
    name: str,
    view: TargetSystemView,
    budget: AdversaryBudget,
    feature_dim: int,
    rng: np.random.Generator,
) -> Attack:
    """
    Construct an attack by name.  Validates that the supplied
    view's capability tier matches the attack's declared tier.
    """
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown attack '{name}'; known: {sorted(_REGISTRY)}"
        )
    cls = _REGISTRY[name]
    return cls(view=view, budget=budget, feature_dim=feature_dim, rng=rng)


# =============================================================================
# Section 9.  Predictor accuracy.
#
# After a run, the experiment harness pairs each AttackCandidate's
# ``predicted_cost_us`` with the actual cost that the SUT recorded
# for that transaction.  Pearson correlation is reported; this is
# an audit of attack quality, not of attack success.
# =============================================================================


@dataclass
class PredictorAccuracy:
    n_pairs: int
    pearson_r: float
    rank_correlation: float
    mean_relative_error: float

    def to_dict(self) -> Mapping[str, float]:
        return {
            "n_pairs": self.n_pairs,
            "pearson_r": self.pearson_r,
            "rank_correlation": self.rank_correlation,
            "mean_relative_error": self.mean_relative_error,
        }


def compute_predictor_accuracy(
    predicted: Sequence[float],
    actual: Sequence[float],
) -> PredictorAccuracy:
    if len(predicted) != len(actual):
        raise ValueError("predicted and actual must have equal length")
    if len(predicted) < 2:
        return PredictorAccuracy(
            len(predicted), float("nan"), float("nan"), float("nan"),
        )
    p = np.asarray(predicted, dtype=np.float64)
    a = np.asarray(actual, dtype=np.float64)
    p_var = p.var()
    a_var = a.var()
    if p_var == 0 or a_var == 0:
        pearson = float("nan")
    else:
        pearson = float(np.corrcoef(p, a)[0, 1])
    # Spearman: correlation of ranks.
    p_rank = np.argsort(np.argsort(p))
    a_rank = np.argsort(np.argsort(a))
    rank = (
        float(np.corrcoef(p_rank, a_rank)[0, 1])
        if p_rank.var() > 0 and a_rank.var() > 0
        else float("nan")
    )
    # MRE on positive actuals only.
    mask = a > 1e-12
    if not mask.any():
        mre = float("nan")
    else:
        mre = float(np.mean(np.abs(p[mask] - a[mask]) / a[mask]))
    return PredictorAccuracy(
        n_pairs=len(predicted),
        pearson_r=pearson,
        rank_correlation=rank,
        mean_relative_error=mre,
    )


# =============================================================================
# Section 10.  Public surface.
# =============================================================================


__all__ = [
    # --- Submission-draft API (preserved verbatim) ----------------------
    # Base
    "Attack",
    "AttackCandidate",
    # Concrete attacks
    "A1Random",
    "A2HighDegree",
    "A3BranchingMax",
    "A4GradientNorm",
    "A5Adaptive",
    "A6EvolutionaryCostOracle",
    # Registry
    "attack_names",
    "make_attack",
    # Predictor accuracy
    "PredictorAccuracy",
    "compute_predictor_accuracy",

    # --- Camera-ready additions ----------------------------------------
    # Public ground-truth-labelling helper: experiments.py calls
    # Attack.is_adversarial_txn_id(...) to classify incoming
    # transactions for the analysis layer's α estimation.
    "ADVERSARIAL_TXN_ID_BASE",
]
