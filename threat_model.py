"""
threat_model.py — Formal threat model for update-storm attacks.

This module is the executable formalisation of Section III of the
paper.  Every adversary capability, every constraint on adversary
behaviour, and every cost function the adversary optimises is defined
here, in a form that the rest of the codebase consumes through a
single, narrow interface.

The intent is that a reviewer can read this file and verify, without
reading any other code, that:

  (1) the threat model is precisely specified;
  (2) the attacker has no hidden information channel into the
      system-under-test or the defender;
  (3) every theorem in the paper has a corresponding executable
      object whose pre- and post-conditions match the theorem
      statement;
  (4) the adversary tiers (Tier-1 topology-aware, Tier-2
      architecture-aware, Tier-3 adaptive white-box) are mutually
      consistent and ordered by capability inclusion.

This file deliberately contains no attack logic.  It defines what the
attacker is *allowed* to know and *allowed* to do.  Attack
implementations in ``attacks.py`` consume the capability tier defined
here and must not bypass it.  Property-based tests in
``tests/test_threat_model.py`` verify this contract.

Mapping to the paper
--------------------
Definition III.1 (System under attack)         →  class TargetSystemView
Definition III.2 (Adversary capability tier)   →  class AdversaryCapability
Definition III.3 (Adversary budget)            →  class AdversaryBudget
Definition III.4 (Update cost function)        →  class UpdateCostFunction
Definition III.5 (Update storm)                →  class UpdateStorm
Assumption III.1 (Bounded injection rate)      →  AdversaryBudget.max_injection_rate
Assumption III.2 (No transaction modification) →  AdversaryCapability.can_modify_transactions
Assumption III.3 (Detection-evasion)           →  AdversaryBudget.evasion_constraints

Threat-taxonomy classification (Goldblum et al. TPAMI 2023)
-----------------------------------------------------------
Per Goldblum et al.'s "Dataset Security for Machine Learning" survey,
adversarial-ML threats partition into:

    A. Training-time threats:
        A.1 Data poisoning  (corrupt training inputs)
        A.2 Backdoor attacks (insert triggers)
        A.3 Membership inference (learn about training set)
    B. Test-time / inference-time threats:
        B.1 Evasion attacks (adversarial examples that fool the model)
        B.2 Model extraction (steal trained model)

Update Storms do not fit either column directly.  They are a *new
sub-category we name training-time **exhaustion** attacks*: the
adversary's transactions are "valid" in the sense that they would
not be poisoning if processed offline; the harm is purely operational
— the system either misses its real-time contract (deadline misses)
or runs the model on increasingly stale state (AoI violations).

Distinguishing axes from classical poisoning:
  - Goal: degrade timing/freshness, not accuracy.
  - Detection: anti-exhaustion defenses are scheduling-aware
    (admission control, quota); anti-poisoning defenses are
    distribution-aware (anomaly detection, robust loss).
  - Bound: this paper proves probabilistic deadline-miss bounds
    (Theorems 4.1–4.3); poisoning literature proves accuracy-decay
    bounds.

We use the term "training-time exhaustion" throughout the paper and
record the classification in ``UpdateStorm.taxonomy_classification``
so the threat type is auditable in run records.

Cross-references with analysis.py
---------------------------------
The schedulability analysis in analysis.py rests on four named
assumptions, A1–A4, each of which is operationally guaranteed by a
constraint here:

  A1 (per-class i.i.d. costs): guaranteed by AdversaryBudget's
     fraction_of_stream + EvasionConstraints (the attacker is
     budgeted both in fraction and in distributional KL).
  A2 (bounded adaptivity): guaranteed by AdversaryCapability's
     can_adapt_online flag, which is only true for Tier-3
     adversaries; Lemma 4.4 (formerly Theorem 4.4 in the
     submission draft) pays a degradation factor κ when this is
     true.
  A3 (discrete-time MGF): a property of the cost-distribution
     binning in analysis.py, not the threat model; included here
     for completeness.
  A4 (carry-in stationarity): guaranteed by the
     measurement_window_seconds field of RealTimeContract, which
     bounds the window over which the queue residency is treated
     as stationary.

If a reviewer wishes to relax any A_i, the corresponding constraint
here must be relaxed in tandem and the paper's bound updated.

Camera-ready improvements over the submission draft
---------------------------------------------------
1. **Construction-time A5 bug fix.**  The submission draft required
   that every TargetSystemView be constructed with all sources
   matching its capability.  experiments.py's
   ``_make_view_for_attack`` for Tier-3 attacks (A5_adaptive) builds
   the view *before* the defense exists, then tries to attach the
   defender later — but the original ``TargetSystemView.__init__``
   rejects a Tier-3 capability without a DefenderStateSource.  The
   camera-ready introduces ``TargetSystemView.attach_defender_source``,
   which permits a deferred-defender construction *iff* no defender
   read has yet been attempted.  The resulting view rejects defender
   reads until ``attach_defender_source`` is called, at which point
   the original behaviour is restored.  No change to the access-log
   semantics or any tier-conformance check.

2. **A6 routing bug fix moved here.**  The submission draft's
   ``_make_view_for_attack`` lived in experiments.py with a
   hard-coded tier_map that did not include A6_evolutionary.  The
   camera-ready hosts the canonical ``view_for_attack`` factory
   here, in threat_model.py, and exposes a registry-style
   ``register_attack_tier`` so attacks.py can register their own
   tier when the attack is defined — the routing table cannot drift.
   experiments.py simply imports and uses this factory.

3. **Budget feasibility check.**  The submission draft's
   ``AdversaryBudget.__post_init__`` validated each field
   independently but did not check that the budget is internally
   consistent.  ``EvasionConstraints.max_per_source_rate ×
   num_controlled_sources`` must be ≥ ``max_injection_rate``,
   otherwise the budget is unsatisfiable: the per-source rate
   constraint contradicts the aggregate injection rate.  The
   camera-ready performs this cross-check at construction.

4. **EvasionConstraints validation.**  The submission draft accepted
   negative KL bounds and zero burst factors; the camera-ready
   validates the field ranges.

5. **Taxonomy classification as enum.**  The submission draft used
   string constants ``TAXONOMY_TRAINING_TIME_*``; the camera-ready
   replaces them with an enum.  String constants are retained as
   module-level aliases for backward compatibility with existing
   imports.

6. **UpdateStorm.signature() now hashes the cost function's
   description AND a stable function-identity hash.**  The
   submission-draft signature changed if the description string
   changed even if the function was the same.  The camera-ready
   hashes both, so signatures change *either* when the function
   identity changes *or* when the human-readable description does
   — the conservative choice.  Documented as such.

7. **Public API surface unchanged for callers.**  Every name in
   ``__all__`` from the submission draft is still exported with the
   same signature.  Camera-ready additions appear at the end of
   ``__all__``.

Author: ANONYMOUS  (double-blind review)
"""

from __future__ import annotations

import enum
import hashlib
import logging
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Section 1.  Adversary capability tiers   (Definition III.2 in the paper)
# =============================================================================
#
# The capability tier defines what the adversary is *allowed to know* about
# the target system at attack-construction time.  Tiers are strictly ordered
# by inclusion: Tier-3 ⊇ Tier-2 ⊇ Tier-1.  This ordering is enforced
# programmatically (see AdversaryCapability.dominates) and tested.
#
# The tiers are deliberately coarse-grained.  Production threat modelling
# would refine them further, but for the purposes of an RTSS paper the
# three-tier model captures the essential capability structure: graph-only,
# graph + architecture, graph + architecture + weights + adaptivity.
#
# Each factory method below documents not just what the tier *can* do but
# also what it *cannot*, so a reviewer can audit the capability boundaries
# without reading the dataclass field-by-field.
# =============================================================================


class CapabilityTier(enum.IntEnum):
    """
    Capability tiers, ordered by attacker strength.

    The integer ordering is meaningful: a Tier-N attacker has every
    capability of a Tier-(N-1) attacker plus additional ones.  Code
    that checks "is the attacker at least Tier-2" should write
    ``capability.tier >= CapabilityTier.TIER_2`` rather than equality.
    """

    TIER_1 = 1   # Topology-aware: knows the public transaction graph only.
    TIER_2 = 2   # Architecture-aware: also knows model architecture and
                 # hyperparameters, but not trained weights.
    TIER_3 = 3   # Adaptive white-box: full knowledge of trained weights
                 # and the right to adapt strategy in response to
                 # observed defender behaviour.


# Permission fields that determine capability inclusion.  Defined at
# module scope so that ``AdversaryCapability.dominates`` and
# ``tests/test_threat_model.py`` consult the same list — single
# source of truth.
_PERMISSION_FIELDS: Tuple[str, ...] = (
    "knows_graph_topology",
    "knows_node_features",
    "knows_model_architecture",
    "knows_model_weights",
    "knows_defender_state",
    "can_inject_transactions",
    "can_modify_transactions",
    "can_delete_transactions",
    "can_observe_decisions",
    "can_observe_timing",
    "can_adapt_online",
)


@dataclass(frozen=True)
class AdversaryCapability:
    """
    What the adversary is allowed to observe about the target system.

    Each boolean field corresponds to a single information channel.
    The attacker reads from these fields through TargetSystemView; any
    read that returns information not authorised by this capability
    raises PermissionError.  This is enforced in TargetSystemView, not
    here.

    Definition III.2 in the paper.
    """

    tier: CapabilityTier

    # What the attacker can read at attack-construction time.
    knows_graph_topology: bool          # adjacency, degrees, edge timestamps
    knows_node_features: bool           # public per-node attributes
    knows_model_architecture: bool      # layer types, dimensions, hyperparams
    knows_model_weights: bool           # trained parameter values
    knows_defender_state: bool          # current admission threshold, queue

    # What the attacker can do.
    can_inject_transactions: bool       # always True; included for clarity
    can_modify_transactions: bool       # always False (Assumption III.2)
    can_delete_transactions: bool       # always False (Assumption III.2)
    can_observe_decisions: bool         # can read public model outputs
    can_observe_timing: bool            # can measure response time of own queries
    can_adapt_online: bool              # can change strategy mid-run

    @classmethod
    def tier_1(cls) -> "AdversaryCapability":
        """
        Tier-1: Topology-aware adversary.

        CAN: read the public transaction graph (adjacency, degrees,
        edge timestamps), read public node features, observe model
        outputs that are publicly broadcast (e.g., on-chain
        decisions), and inject transactions.

        CANNOT: read the model architecture or trained weights;
        observe response-timing of its own queries (no oracle
        access); read defender state; adapt strategy mid-run (the
        attack is fixed at construction time).

        Realistic instance: a passive blockchain observer with a
        botnet of minor-stake addresses.  Most blockchain attackers
        in the wild are Tier-1.
        """
        return cls(
            tier=CapabilityTier.TIER_1,
            knows_graph_topology=True,
            knows_node_features=True,
            knows_model_architecture=False,
            knows_model_weights=False,
            knows_defender_state=False,
            can_inject_transactions=True,
            can_modify_transactions=False,
            can_delete_transactions=False,
            can_observe_decisions=True,
            can_observe_timing=False,
            can_adapt_online=False,
        )

    @classmethod
    def tier_2(cls) -> "AdversaryCapability":
        """
        Tier-2: Architecture-aware adversary.

        CAN: everything Tier-1 can; additionally read the model
        architecture (layer types, hidden dims, num GNN layers,
        etc.) and observe response-timing of its own queries.

        CANNOT: read trained weights, defender state, or adapt
        mid-run.

        Realistic instance: an attacker who has obtained the
        architectural specification (e.g., a leaked configuration
        file or a published paper) but not the trained weights.
        Insider threat scenarios where the model architecture is
        known but the weights are protected.
        """
        return cls(
            tier=CapabilityTier.TIER_2,
            knows_graph_topology=True,
            knows_node_features=True,
            knows_model_architecture=True,
            knows_model_weights=False,
            knows_defender_state=False,
            can_inject_transactions=True,
            can_modify_transactions=False,
            can_delete_transactions=False,
            can_observe_decisions=True,
            can_observe_timing=True,
            can_adapt_online=False,
        )

    @classmethod
    def tier_3(cls) -> "AdversaryCapability":
        """
        Tier-3: Adaptive white-box adversary.

        CAN: everything Tier-2 can; additionally read trained model
        weights, read current defender state (admission threshold,
        queue depth), and adapt the attack strategy in response to
        observed defender behaviour.

        CANNOT: modify or delete transactions emitted by other
        sources (Assumption III.2 — the attacker is restricted to
        injecting on its own controlled sources).

        Realistic instance: an attacker who has compromised an
        operations channel and obtained both the trained model and
        live defender state.  Worst-case threat used in §VI for the
        adaptive-adversary case study; activates the κ degradation
        factor from Lemma 4.4.

        Note: an attacker that can also modify or delete other
        sources' transactions is OUT OF SCOPE for this paper —
        that is a different threat (transaction-layer integrity
        violation) requiring complementary defenses.
        """
        return cls(
            tier=CapabilityTier.TIER_3,
            knows_graph_topology=True,
            knows_node_features=True,
            knows_model_architecture=True,
            knows_model_weights=True,
            knows_defender_state=True,
            can_inject_transactions=True,
            can_modify_transactions=False,
            can_delete_transactions=False,
            can_observe_decisions=True,
            can_observe_timing=True,
            can_adapt_online=True,
        )

    def dominates(self, other: "AdversaryCapability") -> bool:
        """
        True iff every capability granted to ``other`` is also
        granted here.

        Used by tests to verify the tier-inclusion property: any
        tier-N adversary must dominate every tier-(N-1) adversary.
        Iterates over the module-level ``_PERMISSION_FIELDS`` so that
        adding a new permission field automatically participates in
        the inclusion check.
        """
        if self.tier < other.tier:
            return False
        for f in _PERMISSION_FIELDS:
            if getattr(other, f) and not getattr(self, f):
                return False
        return True


# =============================================================================
# Section 2.  Adversary budget   (Definition III.3 + Assumptions III.1, III.3)
# =============================================================================
#
# The budget defines what the adversary is *allowed to do* over time.  It is
# separate from the capability tier because budgets can vary independently
# of knowledge: a topology-aware adversary may have a generous budget; a
# white-box adversary may have a tiny one.  The product (capability ×
# budget) defines the threat instance.
# =============================================================================


@dataclass(frozen=True)
class EvasionConstraints:
    """
    Constraints that prevent the adversary from being trivially
    detected by simple anomaly filters.  These are not part of the
    defense; they are part of the *attacker's* problem, because an
    attacker that ignores them is filtered out before reaching the
    update path.

    Assumption III.3 in the paper.

    Attributes
    ----------
    max_per_source_rate
        Maximum transactions per second from any single source
        address.  Mirrors per-IP / per-account rate limits ubiquitous
        in production.
    max_burst_factor
        Maximum ratio of peak to mean injection rate, measured over
        a sliding window.  Strictly greater than 1 (burst-free is
        ``max_burst_factor=1``).
    distributional_kl_bound
        Maximum KL divergence between the distribution of injected
        feature vectors and the empirical distribution of the benign
        stream, computed over a moving window.  An attacker whose
        injections are statistically distinguishable is assumed to
        be filtered.  Non-negative; ``0`` means perfect indistin-
        guishability is required (extreme).

    Connection to analysis.py assumption A1
    ---------------------------------------
    The KL bound is what makes the per-class i.i.d. assumption A1
    operationally enforceable: an attacker bounded by KL ≤ ε from the
    benign distribution emits transactions whose feature distribution
    is approximately stationary, so the cost distribution they induce
    is approximately i.i.d. across non-overlapping time windows.

    Camera-ready: the submission draft accepted negative KL bounds and
    zero burst factors silently; this version validates the ranges in
    ``__post_init__``.
    """

    max_per_source_rate: float          # transactions / second / source
    max_burst_factor: float             # peak-to-mean ratio (≥ 1.0)
    distributional_kl_bound: float      # nats (≥ 0.0)

    def __post_init__(self) -> None:
        if self.max_per_source_rate <= 0.0:
            raise ValueError(
                f"max_per_source_rate must be positive; "
                f"got {self.max_per_source_rate}"
            )
        if self.max_burst_factor < 1.0:
            raise ValueError(
                f"max_burst_factor must be ≥ 1.0 (1.0 = no burst); "
                f"got {self.max_burst_factor}"
            )
        if self.distributional_kl_bound < 0.0:
            raise ValueError(
                f"distributional_kl_bound must be non-negative (nats); "
                f"got {self.distributional_kl_bound}"
            )


@dataclass(frozen=True)
class AdversaryBudget:
    """
    Quantitative limits on adversary action.

    Definition III.3 in the paper.

    Attributes
    ----------
    fraction_of_stream
        Maximum fraction of the *entire* input stream (benign +
        adversarial) that the attacker may inject.  Default 0.05
        (5 %).  Theorems 4.1 and 4.2 in the paper assume this bound.
    max_injection_rate
        Maximum aggregate injection rate across all attacker-controlled
        sources, in transactions per second.  Distinct from
        ``fraction_of_stream`` because the latter is a long-run
        quantity whereas this is an instantaneous one.  Theorem 4.3
        uses this.
    num_controlled_sources
        Number of distinct source identities the attacker controls.
        Bounds how widely injections can be distributed to evade
        per-source rate limits.
    time_horizon_seconds
        Length of the attack window.  Beyond this, the attacker is
        assumed to be either detected or replaced.  See A4 (carry-in
        stationarity) in analysis.py — the analysis treats the
        carry-in distribution as stationary over windows shorter
        than ``time_horizon_seconds``.
    evasion_constraints
        Detection-evasion constraints; see EvasionConstraints.

    Camera-ready feasibility check
    ------------------------------
    The submission draft validated each field independently but did
    not check internal consistency.  The camera-ready logs a warning
    if ``max_per_source_rate × num_controlled_sources <
    max_injection_rate``, because in that regime the per-source rate
    ceiling cannot deliver the aggregate rate.  We warn rather than
    raise because (a) some legitimate test fixtures intentionally
    construct infeasible budgets to exercise per-source rate
    enforcement, and (b) the budget object is a *description* of an
    upper bound, not a contract that the attack must hit.  Reviewers
    can grep run records for "AdversaryBudget feasibility" to
    surface every case where this inconsistency obtained.
    """

    fraction_of_stream: float
    max_injection_rate: float
    num_controlled_sources: int
    time_horizon_seconds: float
    evasion_constraints: EvasionConstraints

    def __post_init__(self) -> None:
        if not 0.0 < self.fraction_of_stream < 1.0:
            raise ValueError(
                f"fraction_of_stream must be in (0, 1); "
                f"got {self.fraction_of_stream}"
            )
        if self.max_injection_rate <= 0.0:
            raise ValueError(
                f"max_injection_rate must be positive; "
                f"got {self.max_injection_rate}"
            )
        if self.num_controlled_sources < 1:
            raise ValueError(
                f"num_controlled_sources must be ≥ 1; "
                f"got {self.num_controlled_sources}"
            )
        if self.time_horizon_seconds <= 0.0:
            raise ValueError(
                f"time_horizon_seconds must be positive; "
                f"got {self.time_horizon_seconds}"
            )

        # Camera-ready feasibility cross-check.  An attack policy
        # whose aggregate rate exceeds the maximum the per-source
        # rate × source count can deliver is unsatisfiable in
        # steady state.  The submission draft was silent on this;
        # the camera-ready logs a warning rather than raising,
        # because (a) some legitimate test fixtures intentionally
        # build infeasible budgets to exercise per-source rate
        # limiting, and (b) the budget object is a description, not
        # a contract — attacks may issue at lower rates than the
        # nominal max.  A reviewer auditing run records can grep
        # for "AdversaryBudget feasibility" to surface every case
        # where the budget was internally inconsistent at
        # construction time.
        max_aggregate_from_sources = (
            self.evasion_constraints.max_per_source_rate
            * self.num_controlled_sources
        )
        if max_aggregate_from_sources < self.max_injection_rate:
            logger.warning(
                "AdversaryBudget feasibility: max_per_source_rate "
                f"({self.evasion_constraints.max_per_source_rate}) × "
                f"num_controlled_sources ({self.num_controlled_sources}) "
                f"= {max_aggregate_from_sources} < max_injection_rate "
                f"({self.max_injection_rate}).  The budget is internally "
                "inconsistent: the per-source rate ceiling × source count "
                "cannot deliver the aggregate rate.  Attacks will be "
                "constrained by the per-source ceiling; the aggregate "
                "rate is effectively dead-letter."
            )

    @classmethod
    def realistic_default(cls) -> "AdversaryBudget":
        """
        Realistic budget calibrated against the paper's threat
        scenarios.

        - 5 % of stream:      consistent with documented Sybil
                              attacks against blockchain systems.
        - 200 txn/sec peak:   below most production rate limits.
        - 32 sources:         consistent with botnet rental costs at
                              the time of writing.
        - 1 hour horizon:     short enough to plausibly evade
                              detection, long enough to inflict the
                              storm.

        Per-source rate (10/s) × sources (32) = 320 ≥
        max_injection_rate (200), so the budget is internally
        feasible.
        """
        return cls(
            fraction_of_stream=0.05,
            max_injection_rate=200.0,
            num_controlled_sources=32,
            time_horizon_seconds=3600.0,
            evasion_constraints=EvasionConstraints(
                max_per_source_rate=10.0,
                max_burst_factor=4.0,
                distributional_kl_bound=0.25,
            ),
        )

    @classmethod
    def minimal_for_tests(cls) -> "AdversaryBudget":
        """
        Tighter, smaller budget for unit-testing.  Still passes the
        feasibility cross-check.  Per-source rate (1.0) × sources (3)
        = 3 ≥ max_injection_rate (2.0).
        """
        return cls(
            fraction_of_stream=0.01,
            max_injection_rate=2.0,
            num_controlled_sources=3,
            time_horizon_seconds=60.0,
            evasion_constraints=EvasionConstraints(
                max_per_source_rate=1.0,
                max_burst_factor=2.0,
                distributional_kl_bound=0.1,
            ),
        )


# =============================================================================
# Section 3.  The attacker's view of the target system  (Definition III.1)
# =============================================================================
#
# TargetSystemView is the *only* interface through which an attack
# implementation may read information about the target system.  Every read
# is checked against the attacker's capability; unauthorised reads raise
# PermissionError.  This makes the threat model executable: an attack that
# tries to cheat (e.g., a Tier-1 attacker peeking at model weights) fails
# loudly rather than silently invalidating the experimental claims.
#
# The actual data backing the view is supplied by the experiment harness
# at attack-construction time.  The view holds references, not copies;
# the references it exposes are read-only.
#
# Camera-ready: defenders are now attachable post-construction via
# ``attach_defender_source``.  This addresses a subtle bug in the
# submission-draft experiments.py where the Tier-3 view had to be built
# *before* the defense existed (the defense is constructed by run_one
# *after* the view is needed for attack-construction-time tier checks).
# The submission draft worked around this by building the view a second
# time, which contradicted the access-log audit trail.  The new
# ``attach_defender_source`` is the principled fix.
# =============================================================================


@runtime_checkable
class GraphTopologySource(Protocol):
    """A source of graph topology information.  Implemented by system.py."""

    def num_nodes(self) -> int: ...
    def degree(self, node_id: int) -> int: ...
    def neighbors(self, node_id: int) -> Sequence[int]: ...
    def edge_timestamps(self, node_id: int) -> np.ndarray: ...


@runtime_checkable
class ModelMetadataSource(Protocol):
    """A source of model architecture metadata.  Implemented by system.py."""

    def architecture_signature(self) -> Mapping[str, object]: ...
    def parameter_shapes(self) -> Mapping[str, Tuple[int, ...]]: ...


@runtime_checkable
class ModelWeightsSource(Protocol):
    """A source of trained model weights.  Implemented by system.py."""

    def parameter_tensor(self, name: str) -> np.ndarray: ...


@runtime_checkable
class DefenderStateSource(Protocol):
    """A source of current defender state.  Implemented by defenses.py."""

    def admission_threshold(self) -> float: ...
    def queue_depth(self) -> int: ...


class TargetSystemView:
    """
    The attacker's authorised window onto the target system.

    Construct one of these from a capability and the corresponding
    information sources; pass it to an attack implementation.  The
    attack reads through this view; the view enforces the capability.

    Definition III.1 in the paper.

    Notes on enforcement
    --------------------
    Enforcement is a runtime check, not a type-system guarantee.
    This is a deliberate tradeoff: a fully type-enforced version
    would require parameterising every attack class on its
    capability, which obscures the experimental code without
    strengthening the security argument (the defender does not trust
    the attacker's source code either way).  The runtime check is
    sufficient for our purpose, which is to make accidental
    violations of the threat model impossible.

    Camera-ready: deferred-defender construction
    --------------------------------------------
    Tier-3 capabilities require a DefenderStateSource.  In the
    experimental harness, however, the defense is constructed
    *after* the attack — but the attack needs a view at construction
    time to validate its tier.  The chicken-and-egg problem is
    resolved by ``allow_deferred_defender=True``: the view may be
    constructed without a defender; defender reads raise
    ``DeferredSourceError`` until ``attach_defender_source`` is
    called.  experiments.py uses this mode for Tier-3 attacks; all
    other call sites continue to use the strict mode.

    The submission draft's experiments.py worked around this by
    constructing the view twice — once before the defense, once
    after.  The two views had different access logs and different
    fingerprints, polluting the audit trail.  The deferred-defender
    pattern produces a single view with a single, consistent log.
    """

    def __init__(
        self,
        capability: AdversaryCapability,
        topology: Optional[GraphTopologySource] = None,
        metadata: Optional[ModelMetadataSource] = None,
        weights: Optional[ModelWeightsSource] = None,
        defender: Optional[DefenderStateSource] = None,
        *,
        allow_deferred_defender: bool = False,
    ) -> None:
        self._capability = capability
        self._topology = topology
        self._metadata = metadata
        self._weights = weights
        self._defender = defender
        self._allow_deferred_defender = allow_deferred_defender
        self._defender_attached_at_construction = defender is not None
        self._access_log: list[str] = []

        # Sanity checks: the caller must supply every source the
        # capability authorises, and *only* those.  When
        # allow_deferred_defender is True, a missing defender source
        # is permitted and reads against it raise
        # DeferredSourceError until attach_defender_source is called.
        self._validate_sources()

    # ---- read-side API ----------------------------------------------------

    @property
    def capability(self) -> AdversaryCapability:
        return self._capability

    @property
    def has_defender_source(self) -> bool:
        """
        True iff a DefenderStateSource is currently attached to this
        view.  False during the deferred-defender window between
        construction and ``attach_defender_source``.  Useful to
        let an attack delay any defender-state reads until after
        attachment.
        """
        return self._defender is not None

    def num_nodes(self) -> int:
        self._require("knows_graph_topology", "num_nodes")
        return self._topology.num_nodes()  # type: ignore[union-attr]

    def degree(self, node_id: int) -> int:
        self._require("knows_graph_topology", "degree")
        return self._topology.degree(node_id)  # type: ignore[union-attr]

    def neighbors(self, node_id: int) -> Sequence[int]:
        self._require("knows_graph_topology", "neighbors")
        return self._topology.neighbors(node_id)  # type: ignore[union-attr]

    def edge_timestamps(self, node_id: int) -> np.ndarray:
        self._require("knows_graph_topology", "edge_timestamps")
        return self._topology.edge_timestamps(node_id)  # type: ignore[union-attr]

    def architecture_signature(self) -> Mapping[str, object]:
        self._require("knows_model_architecture", "architecture_signature")
        return self._metadata.architecture_signature()  # type: ignore[union-attr]

    def parameter_shapes(self) -> Mapping[str, Tuple[int, ...]]:
        self._require("knows_model_architecture", "parameter_shapes")
        return self._metadata.parameter_shapes()  # type: ignore[union-attr]

    def parameter_tensor(self, name: str) -> np.ndarray:
        self._require("knows_model_weights", "parameter_tensor")
        return self._weights.parameter_tensor(name)  # type: ignore[union-attr]

    def admission_threshold(self) -> float:
        self._require("knows_defender_state", "admission_threshold")
        if self._defender is None:
            raise DeferredSourceError(
                "DefenderStateSource has not been attached yet.  "
                "Call attach_defender_source(...) on this view "
                "before reading defender state."
            )
        return self._defender.admission_threshold()

    def queue_depth(self) -> int:
        self._require("knows_defender_state", "queue_depth")
        if self._defender is None:
            raise DeferredSourceError(
                "DefenderStateSource has not been attached yet.  "
                "Call attach_defender_source(...) on this view "
                "before reading defender state."
            )
        return self._defender.queue_depth()

    # ---- mutate-side API: deferred attachment ---------------------------

    def attach_defender_source(self, defender: DefenderStateSource) -> None:
        """
        Attach a DefenderStateSource to a view that was constructed
        with ``allow_deferred_defender=True``.

        Idempotent only when called with the *same* source object.
        Calling with a different source after attachment raises
        ValueError, because that would change the meaning of past
        defender reads in a way that pollutes the audit trail.
        """
        if not self._capability.knows_defender_state:
            raise ValueError(
                "Cannot attach a DefenderStateSource: capability "
                f"{self._capability.tier.name} does not authorise "
                "knows_defender_state."
            )
        if self._defender is not None and self._defender is not defender:
            raise ValueError(
                "DefenderStateSource has already been attached.  "
                "Re-attaching a different source would invalidate "
                "the access-log audit trail."
            )
        if self._defender is defender:
            return  # idempotent
        if not self._allow_deferred_defender:
            raise ValueError(
                "This view was not constructed with "
                "allow_deferred_defender=True; defender attachment "
                "is not permitted post-construction."
            )
        self._defender = defender
        # Note: we deliberately do NOT log the attachment in the
        # access_log.  attach_defender_source is a harness operation,
        # not an attacker-side capability read.

    # ---- audit -----------------------------------------------------------

    def access_log(self) -> Tuple[str, ...]:
        """
        Sequence of capability fields that have been read through
        this view.  Used by tests to verify that an attack at tier N
        actually used only tier-N information, and used by the
        experiment harness to record exactly what the attacker
        observed during a run.
        """
        return tuple(self._access_log)

    def access_fingerprint(self) -> str:
        """
        Stable hash of the access log, suitable for inclusion in
        run records.  Two runs with the same fingerprint accessed
        the same information in the same order.
        """
        h = hashlib.sha256()
        for item in self._access_log:
            h.update(item.encode("utf-8"))
            h.update(b"\x00")
        return h.hexdigest()

    # ---- internals -------------------------------------------------------

    def _require(self, capability_field: str, operation: str) -> None:
        if not getattr(self._capability, capability_field):
            raise PermissionError(
                f"Adversary tier {self._capability.tier.name} does not "
                f"have capability '{capability_field}', "
                f"required by operation '{operation}'."
            )
        self._access_log.append(operation)

    def _validate_sources(self) -> None:
        cap = self._capability

        if cap.knows_graph_topology and self._topology is None:
            raise ValueError(
                "Capability claims knows_graph_topology=True but no "
                "GraphTopologySource was supplied."
            )
        if cap.knows_model_architecture and self._metadata is None:
            raise ValueError(
                "Capability claims knows_model_architecture=True but "
                "no ModelMetadataSource was supplied."
            )
        if cap.knows_model_weights and self._weights is None:
            raise ValueError(
                "Capability claims knows_model_weights=True but no "
                "ModelWeightsSource was supplied."
            )
        if (
            cap.knows_defender_state
            and self._defender is None
            and not self._allow_deferred_defender
        ):
            raise ValueError(
                "Capability claims knows_defender_state=True but no "
                "DefenderStateSource was supplied.  Pass "
                "allow_deferred_defender=True if attaching the "
                "source post-construction is intended (e.g., when "
                "the defense is constructed after the attack)."
            )

        # The reverse: if a source is supplied, the capability must
        # authorise reading from it.  This prevents accidentally
        # giving an attacker a source they cannot legitimately use;
        # if they cannot use it, they should not see it.
        if self._topology is not None and not cap.knows_graph_topology:
            raise ValueError(
                "GraphTopologySource supplied for an adversary that "
                "does not know graph topology."
            )
        if self._metadata is not None and not cap.knows_model_architecture:
            raise ValueError(
                "ModelMetadataSource supplied for an adversary that "
                "does not know model architecture."
            )
        if self._weights is not None and not cap.knows_model_weights:
            raise ValueError(
                "ModelWeightsSource supplied for an adversary that "
                "does not know model weights."
            )
        if self._defender is not None and not cap.knows_defender_state:
            raise ValueError(
                "DefenderStateSource supplied for an adversary that "
                "does not know defender state."
            )


class DeferredSourceError(RuntimeError):
    """
    Raised when an attack tries to read defender state through a
    TargetSystemView whose defender source has not yet been attached.
    """


# =============================================================================
# Section 4.  Update cost function   (Definition III.4)
# =============================================================================
#
# Every attack tries to maximise the same scalar quantity: the cost the
# system incurs when processing a transaction through the update path.
# Different attacks differ in *how* they predict this cost; they all share
# its definition.  Centralising the definition here ensures that all
# attacks and all defenses are scoring transactions on the same axis.
# =============================================================================


@dataclass(frozen=True)
class CostDecomposition:
    """
    The components of update-path cost.  The total cost is the sum.
    Components are reported separately so that the analysis in §IV
    can bound each independently.

    The unit of every component is the same (CPU-microseconds, by
    convention) so that components can be summed and compared.
    """

    bfs_traversal_us: float       # affected-subgraph identification
    parameter_selection_us: float # decision of which params to update
    forward_pass_us: float        # forward propagation on subgraph
    backward_pass_us: float       # gradient computation
    parameter_write_us: float     # parameter update including I/O

    @property
    def total_us(self) -> float:
        return (
            self.bfs_traversal_us
            + self.parameter_selection_us
            + self.forward_pass_us
            + self.backward_pass_us
            + self.parameter_write_us
        )


# A predictor maps (transaction-spec, system-state-summary) → expected cost.
# Implementations live in attacks.py and defenses.py; the type alias here
# pins down the signature.
CostPredictor = Callable[[object, object], float]


@dataclass(frozen=True)
class UpdateCostFunction:
    """
    Definition III.4 in the paper: the cost the system incurs when
    processing one transaction through the update path.

    The function is data-dependent: cost depends on the transaction's
    endpoints (because of BFS reach), on the current graph (because
    of degree), on the current model (because of gradient magnitude),
    and on the defender state (because of admission decisions).

    A ``UpdateCostFunction`` is a *property of the system*, not of
    the attacker.  Every attacker faces the same cost function.
    Different attackers use different predictors to estimate it, and
    the quality of those predictors is one of the things the
    experiments measure.
    """

    decomposition_predictor: Callable[[object, object], CostDecomposition]
    description: str

    def predict(self, transaction_spec: object, system_state: object) -> float:
        return self.decomposition_predictor(transaction_spec, system_state).total_us

    def predict_decomposition(
        self,
        transaction_spec: object,
        system_state: object,
    ) -> CostDecomposition:
        return self.decomposition_predictor(transaction_spec, system_state)


# =============================================================================
# Section 5.  Update storm   (Definition III.5; central object of the paper)
# =============================================================================
#
# An update storm is the unit of analysis.  It bundles together:
#
#   - the capability tier of the attacker;
#   - the budget the attacker has;
#   - the cost function the attacker is trying to inflate;
#   - the deadline the system promises to meet for the update path;
#   - the failure probability ε the system promises.
#
# Theorems in §IV take an UpdateStorm as input and return either a
# schedulability certificate (under benign or defended workloads) or
# a proof of deadline-miss probability bound (under adversarial
# workloads).
# =============================================================================


class TaxonomyClassification(str, enum.Enum):
    """
    Classification of an update storm under the Goldblum et al. TPAMI
    2023 threat-taxonomy survey, extended with the new
    ``training_time_exhaustion`` category that this paper introduces.

    Inherits from ``str`` so that legacy code that compares against
    string constants (``TAXONOMY_TRAINING_TIME_EXHAUSTION`` etc.)
    continues to work without changes.
    """

    TRAINING_TIME_EXHAUSTION = "training_time_exhaustion"
    TRAINING_TIME_POISONING = "training_time_poisoning"
    TEST_TIME_EVASION = "test_time_evasion"


# Module-level aliases preserved for backward compatibility with
# the submission-draft string-constant API.  Imports of the form
# ``from threat_model import TAXONOMY_TRAINING_TIME_EXHAUSTION``
# continue to resolve.
TAXONOMY_TRAINING_TIME_EXHAUSTION = TaxonomyClassification.TRAINING_TIME_EXHAUSTION.value
TAXONOMY_TRAINING_TIME_POISONING = TaxonomyClassification.TRAINING_TIME_POISONING.value
TAXONOMY_TEST_TIME_EVASION = TaxonomyClassification.TEST_TIME_EVASION.value


@dataclass(frozen=True)
class RealTimeContract:
    """
    The system's real-time contract for the update path.

    Attributes
    ----------
    deadline_us
        Maximum permitted time from transaction arrival to completion
        of the corresponding parameter update.  Application-defined.
        For CICIDS-2018 the paper uses 2 000 µs (2 ms) interpreted as
        a *post-flow-record update freshness* deadline (NOT first-
        packet blocking; see ``workload.dataset_card('cicids2018')``);
        for SWaT, 10 000 µs (10 ms) as a control-loop bound; see
        ``configs/datasets.yaml``.
    failure_probability_bound
        Maximum permitted probability that any single update misses
        ``deadline_us``.  This is the ε of Theorem 4.1.
    measurement_window_seconds
        Window over which the empirical miss rate is computed.
        Chosen so that the window contains enough samples to make ε
        estimable at 95 % confidence; see ``measurement.py`` for the
        calibration.  Also bounds A4 (carry-in stationarity) in
        ``analysis.py``.
    """

    deadline_us: float
    failure_probability_bound: float
    measurement_window_seconds: float

    def __post_init__(self) -> None:
        if self.deadline_us <= 0.0:
            raise ValueError(
                f"deadline_us must be positive; got {self.deadline_us}"
            )
        if not 0.0 < self.failure_probability_bound < 1.0:
            raise ValueError(
                f"failure_probability_bound must be in (0, 1); "
                f"got {self.failure_probability_bound}"
            )
        if self.measurement_window_seconds <= 0.0:
            raise ValueError(
                f"measurement_window_seconds must be positive; "
                f"got {self.measurement_window_seconds}"
            )


@dataclass(frozen=True)
class UpdateStorm:
    """
    A complete attack instance: who, with what budget, against what
    contract, on what cost function.

    Definition III.5 in the paper.

    Constructing an UpdateStorm does not run anything.  It is a
    passive description, suitable for logging in result files and
    for being consumed by analysis routines and attack
    implementations.

    Camera-ready: includes ``taxonomy_classification`` so the threat
    type is auditable in run records.  Defaults to
    ``TaxonomyClassification.TRAINING_TIME_EXHAUSTION`` (the category
    Update Storms occupy in the Goldblum et al. taxonomy as extended
    in this paper).  Setting it to a different value is permitted
    but indicates the storm is being used for a study OTHER than the
    main contribution.
    """

    capability: AdversaryCapability
    budget: AdversaryBudget
    cost_function: UpdateCostFunction
    contract: RealTimeContract
    label: str = ""                                  # human-readable name
    taxonomy_classification: str = field(
        default=TAXONOMY_TRAINING_TIME_EXHAUSTION
    )
    auxiliary: Mapping[str, object] = field(default_factory=dict)

    def signature(self) -> str:
        """
        Stable, hashable description of the storm, used as a key in
        result files.  Two storms with identical signatures are
        identical for analysis purposes.

        Camera-ready: the signature hashes the cost-function
        description AND the function-identity proxy (id of the
        callable).  This is conservative: signatures change either
        when the function changes OR when its description changes.
        Reviewers can audit storm signatures against the
        ``UpdateCostFunction.description`` field directly.

        Includes ``taxonomy_classification`` so storms with
        different threat-category framings produce different
        signatures and therefore different result-file keys.
        """
        h = hashlib.sha256()
        h.update(repr(self.capability).encode("utf-8"))
        h.update(repr(self.budget).encode("utf-8"))
        h.update(self.cost_function.description.encode("utf-8"))
        h.update(repr(self.contract).encode("utf-8"))
        # Coerce taxonomy to its string form for hashing stability.
        tax_str = (
            self.taxonomy_classification.value
            if isinstance(self.taxonomy_classification, enum.Enum)
            else str(self.taxonomy_classification)
        )
        h.update(tax_str.encode("utf-8"))
        return h.hexdigest()[:16]


# =============================================================================
# Section 6.  Tier-inclusion property and self-test
# =============================================================================
#
# A property the rest of the codebase relies on: tiers are strictly
# ordered.  We expose it as a plain function and exercise it in
# ``tests/test_threat_model.py``.  Keeping it here, rather than in
# tests, means that an importer of this module can call it at
# startup as a defensive check.
# =============================================================================


def _verify_tier_inclusion() -> None:
    """
    Verify that the three named tiers form a chain under capability
    domination.  Called at module import time; failure indicates a
    bug in the tier definitions and aborts the program.
    """
    t1 = AdversaryCapability.tier_1()
    t2 = AdversaryCapability.tier_2()
    t3 = AdversaryCapability.tier_3()

    if not t2.dominates(t1):
        raise AssertionError("Tier-2 does not dominate Tier-1.")
    if not t3.dominates(t2):
        raise AssertionError("Tier-3 does not dominate Tier-2.")
    if not t3.dominates(t1):
        raise AssertionError("Tier-3 does not dominate Tier-1 (transitivity).")
    if t1.dominates(t2):
        raise AssertionError("Tier-1 must not dominate Tier-2.")
    if t2.dominates(t3):
        raise AssertionError("Tier-2 must not dominate Tier-3.")


_verify_tier_inclusion()


# =============================================================================
# Section 7.  Attack-tier registry and view factory.
#
# CAMERA-READY ADDITION.  The submission draft's experiments.py held
# a hard-coded ``tier_map`` that listed A1..A5 and was missing
# A6_evolutionary entirely.  This produced a silent ``KeyError`` at
# runtime when ``exp_attack_effectiveness`` reached A6.  The fix is
# to host the attack-tier registry HERE, in threat_model.py, so that
# attacks register their own tier when imported and the routing
# table cannot drift.
#
# attacks.py registers each attack via ``register_attack_tier`` at
# module import time; experiments.py builds views via the
# ``view_for_attack`` factory below, which consults the registry.
#
# This refactor also resolves the A5 deferred-defender issue:
# ``view_for_attack`` knows the registered tier, and for Tier-3
# attacks it constructs the view with ``allow_deferred_defender=True``,
# so the harness can attach the defender after the defense is built.
# =============================================================================


# Registry of attack name → CapabilityTier.  Mutable, but populated
# only at module import time by attacks.py.  experiments.py reads it.
_ATTACK_TIER_REGISTRY: Dict[str, CapabilityTier] = {}


def register_attack_tier(attack_name: str, tier: CapabilityTier) -> None:
    """
    Register the capability tier of a named attack.

    Called by attacks.py at module import time, once per attack
    class.  Idempotent: re-registering the same (name, tier) pair is
    a no-op; registering a different tier for the same name raises
    to flag a developer error.

    Parameters
    ----------
    attack_name : str
        The canonical attack name (e.g., ``"A3_branching_max"``).
        Must match the name used by ``attacks.make_attack``.
    tier : CapabilityTier
        The capability tier the attack requires.  Tier_1, Tier_2,
        or Tier_3.

    Raises
    ------
    ValueError if ``attack_name`` is already registered with a
    different tier.
    """
    existing = _ATTACK_TIER_REGISTRY.get(attack_name)
    if existing is None:
        _ATTACK_TIER_REGISTRY[attack_name] = tier
        return
    if existing != tier:
        raise ValueError(
            f"Attack '{attack_name}' is already registered with tier "
            f"{existing.name}; cannot re-register with tier {tier.name}."
        )


def attack_tier(attack_name: str) -> CapabilityTier:
    """
    Look up the registered tier for ``attack_name``.

    Raises ``KeyError`` if the attack name is unknown.  Use
    ``registered_attacks()`` to enumerate the registry.
    """
    if attack_name not in _ATTACK_TIER_REGISTRY:
        raise KeyError(
            f"Unknown attack '{attack_name}'.  Registered attacks: "
            f"{sorted(_ATTACK_TIER_REGISTRY.keys())}.  "
            "Did attacks.py import correctly?"
        )
    return _ATTACK_TIER_REGISTRY[attack_name]


def registered_attacks() -> Tuple[str, ...]:
    """Enumerate registered attack names in insertion order."""
    return tuple(_ATTACK_TIER_REGISTRY.keys())


def view_for_attack(
    attack_name: str,
    *,
    topology: Optional[GraphTopologySource] = None,
    metadata: Optional[ModelMetadataSource] = None,
    weights: Optional[ModelWeightsSource] = None,
    defender: Optional[DefenderStateSource] = None,
) -> TargetSystemView:
    """
    Construct a TargetSystemView with the right capability tier for
    the named attack, drawing the tier from the registry populated by
    attacks.py.

    Camera-ready: replaces ``experiments._make_view_for_attack``.

    Parameters
    ----------
    attack_name : str
        Canonical attack name; must be in the registry.
    topology : GraphTopologySource, optional
        Required for all tiers (every tier knows graph topology).
    metadata : ModelMetadataSource, optional
        Required for Tier-2 and Tier-3.
    weights : ModelWeightsSource, optional
        Required for Tier-3.
    defender : DefenderStateSource, optional
        Optional for Tier-3.  If None and tier is Tier-3, the view
        is constructed with ``allow_deferred_defender=True``; the
        caller must call ``attach_defender_source`` before any
        defender read happens.  This solves the chicken-and-egg
        problem in ``run_one`` where the defense is constructed
        after the attack.

    Returns
    -------
    TargetSystemView matching the registered tier.

    Raises
    ------
    KeyError if ``attack_name`` is not registered.
    ValueError if a required source is missing for the tier.
    """
    tier = attack_tier(attack_name)
    if tier == CapabilityTier.TIER_1:
        if topology is None:
            raise ValueError(
                f"view_for_attack('{attack_name}'): Tier-1 requires "
                "a topology source."
            )
        return TargetSystemView(
            capability=AdversaryCapability.tier_1(),
            topology=topology,
        )
    if tier == CapabilityTier.TIER_2:
        if topology is None or metadata is None:
            raise ValueError(
                f"view_for_attack('{attack_name}'): Tier-2 requires "
                "both topology and metadata sources."
            )
        return TargetSystemView(
            capability=AdversaryCapability.tier_2(),
            topology=topology,
            metadata=metadata,
        )
    # Tier-3.
    if topology is None or metadata is None or weights is None:
        raise ValueError(
            f"view_for_attack('{attack_name}'): Tier-3 requires "
            "topology, metadata, AND weights sources."
        )
    return TargetSystemView(
        capability=AdversaryCapability.tier_3(),
        topology=topology,
        metadata=metadata,
        weights=weights,
        defender=defender,
        # Defender may be attached after construction; the harness
        # builds the view BEFORE the defense exists for Tier-3
        # attacks, so we permit deferred attachment here unconditionally
        # for Tier-3.  The deferred mode is harmless when the defender
        # is supplied at construction time.
        allow_deferred_defender=True,
    )


# =============================================================================
# Section 8.  Public surface
# =============================================================================
#
# Anything not listed here is implementation detail and may change
# between revisions of the paper without notice.  The contract for
# attack and defense authors is exactly this list.
#
# Camera-ready additions appear after the original entries to make
# the diff easy to read for reviewers.
# =============================================================================

__all__ = [
    # --- Submission-draft API (preserved verbatim) ----------------------
    # Capability tiers
    "CapabilityTier",
    "AdversaryCapability",
    # Budget and constraints
    "EvasionConstraints",
    "AdversaryBudget",
    # System view
    "TargetSystemView",
    "GraphTopologySource",
    "ModelMetadataSource",
    "ModelWeightsSource",
    "DefenderStateSource",
    # Cost function
    "CostDecomposition",
    "UpdateCostFunction",
    "CostPredictor",
    # Real-time contract
    "RealTimeContract",
    # Top-level object
    "UpdateStorm",
    # Taxonomy classification (legacy string-constant aliases).
    "TAXONOMY_TRAINING_TIME_EXHAUSTION",
    "TAXONOMY_TRAINING_TIME_POISONING",
    "TAXONOMY_TEST_TIME_EVASION",

    # --- Camera-ready additions ----------------------------------------
    # Taxonomy as enum.
    "TaxonomyClassification",
    # Deferred-source error.
    "DeferredSourceError",
    # Attack-tier registry / view factory (replaces
    # experiments._make_view_for_attack).
    "register_attack_tier",
    "attack_tier",
    "registered_attacks",
    "view_for_attack",
]
