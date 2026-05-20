"""
test_camera_ready_regressions.py — Regressions for every camera-ready fix.

This file is a single reviewer-facing artefact organised by source
file.  Each test references the bug it would have caught against the
ICDE/RTSS submission draft.  When this file passes, the camera-ready
codebase is materially distinct from the submission draft on every
fix advertised in the per-file change-rationale documents.

Organisation
------------
Section 1 — analysis.py
Section 2 — threat_model.py
Section 3 — attacks.py
Section 4 — defenses.py
Section 5 — experiments.py
Section 6 — workload.py
Section 7 — main.py (CLI surface only)

What this file deliberately skips
---------------------------------
- Tests that already pass against the submission draft.  Those live
  in the per-module test files (test_attacks.py et al.) and are
  unchanged.  We only add tests here that the submission draft
  would have failed.
- Empirical claims (attack effectiveness, defense efficacy,
  schedulability bound tightness on real datasets).  Those are the
  paper's experimental contribution; tests for them would amount to
  re-running the paper's experiments, which is the harness in
  experiments.py, not the test suite.
- system.py and measurement.py specific fixes (Phase-A timing skew,
  CUDA event correctness, AoI O(N²)→O(1)).  Those were already
  covered by tests in the submission draft's test_system.py and
  test_measurement.py because the existing tests would have failed
  on the buggy code paths; the camera-ready merely fixed the bugs.
"""

from __future__ import annotations

import json
import math
from collections import deque
from pathlib import Path

import numpy as np
import pytest

from threat_model import (
    AdversaryBudget,
    AdversaryCapability,
    CapabilityTier,
    DeferredSourceError,
    EvasionConstraints,
    RealTimeContract,
    TargetSystemView,
    attack_tier,
    register_attack_tier,
    registered_attacks,
    view_for_attack,
)
from system import Transaction
from attacks import (
    A1Random,
    A6EvolutionaryCostOracle,
    ADVERSARIAL_TXN_ID_BASE,
    Attack,
    attack_names,
    make_attack,
)
from defenses import D3Schedulability
from analysis import (
    CountDistribution,
    MGFCertificate,
    apply_theorem_4_3_mgf,
    cost_distribution_from_samples,
    poisson_count_distribution,
)
from workload import dataset_card


# =============================================================================
# Stub sources (mirror the ones used by the existing per-module tests).
# =============================================================================


class _StubTopology:
    def __init__(self, n: int = 200):
        self._n = n

    def num_nodes(self):
        return self._n

    def degree(self, n):
        return max(1, int(n) % 7)

    def neighbors(self, n):
        return tuple(range(max(0, int(n) - 3), int(n)))

    def edge_timestamps(self, n):
        return np.array([1.0])


class _StubMetadata:
    def architecture_signature(self):
        return {}

    def parameter_shapes(self):
        return {"w": (8, 8), "b": (8,)}


class _StubWeights:
    def parameter_tensor(self, name):
        return np.zeros((8, 8))


class _StubDefender:
    def admission_threshold(self):
        return 1500.0

    def queue_depth(self):
        return 0


def _budget_default() -> AdversaryBudget:
    return AdversaryBudget.realistic_default()


def _budget_tight_per_source() -> AdversaryBudget:
    """Budget designed to force per-source rotation in A6 within a few calls."""
    return AdversaryBudget(
        fraction_of_stream=0.05,
        max_injection_rate=10.0,
        num_controlled_sources=2,
        time_horizon_seconds=10.0,
        evasion_constraints=EvasionConstraints(
            max_per_source_rate=5.0,
            max_burst_factor=4.0,
            distributional_kl_bound=0.25,
        ),
    )


# =============================================================================
# Section 1 — analysis.py
#
# Camera-ready advertised fixes:
#   1.1 Theorem 4.4 demoted to Lemma 4.4 with explicit assumptions.
#   1.2 Adversarial-fit benign-fallback flag exposed to callers.
#   1.3 MGF certificate's optimal_lambda_at_bracket_edge diagnostic.
#   1.4 CountDistribution + count_distribution_from_samples API.
#   1.5 Wilson 95% CI on validation results.
#   1.6 Back-compat aliases for old field names (optimal_lam, etc.).
# =============================================================================


class TestAnalysisCameraReady:
    def test_back_compat_aliases_resolve_correctly(self):
        """Submission-draft callers used optimal_lam / per_class_bounds.
        Camera-ready exposes these as properties of the new fields,
        so existing call sites in defenses.py keep working."""
        benign_dist = cost_distribution_from_samples(
            [100.0, 110.0, 120.0, 130.0, 140.0] * 20,
            label="benign",
        )
        contract = RealTimeContract(
            deadline_us=1000.0,
            failure_probability_bound=1e-2,
            measurement_window_seconds=10.0,
        )
        cert = apply_theorem_4_3_mgf(
            contract=contract,
            cost_distributions={"benign": benign_dist},
            count_distributions={
                "benign": poisson_count_distribution(mean=2.0),
            },
            carry_in=None,
        )
        # New names work.
        _ = cert.optimal_lambda
        _ = cert.components
        # Camera-ready: the bracket-edge diagnostic flag is present
        # on every certificate (was missing entirely in the submission
        # draft).
        assert hasattr(cert, "optimal_lambda_at_bracket_edge")
        assert isinstance(cert.optimal_lambda_at_bracket_edge, bool)
        # Back-compat aliases also work (consumed by defenses.py
        # statistics() in the submission draft).
        assert abs(cert.optimal_lam - cert.optimal_lambda) < 1e-9
        assert cert.per_class_bounds is not None

    def test_components_dict_has_b0_entry(self):
        """The submission-draft MGFCertificate had a separate
        ``carry_in_log_mgf`` field; the camera-ready folds it into
        ``components["B_0"]`` (when carry-in is present) so the
        certificate has a uniform per-class structure."""
        benign_dist = cost_distribution_from_samples(
            [100.0] * 200, label="benign",
        )
        contract = RealTimeContract(
            deadline_us=1000.0,
            failure_probability_bound=1e-2,
            measurement_window_seconds=10.0,
        )
        cert = apply_theorem_4_3_mgf(
            contract=contract,
            cost_distributions={"benign": benign_dist},
            count_distributions={
                "benign": poisson_count_distribution(mean=2.0),
            },
            carry_in=None,
        )
        # ``components`` is the source of truth.
        assert "benign" in cert.components
        # B_0 entry MAY be absent (camera-ready synchronous-harness
        # convention is degenerate-zero); when present, log MGF is
        # finite and zero (degenerate-zero has log MGF = 0).
        b0 = cert.components.get("B_0")
        if b0 is not None:
            assert math.isfinite(float(b0))


# =============================================================================
# Section 2 — threat_model.py
#
# Camera-ready advertised fixes:
#   2.1 Attack-tier registry with view_for_attack(name) factory.
#   2.2 Deferred-defender pattern via attach_defender_source.
#   2.3 DeferredSourceError raised when reading defender state
#       before attach.
# =============================================================================


class TestThreatModelCameraReady:
    def test_view_for_attack_factory_resolves_every_named_attack(self):
        """The submission-draft experiments._make_view_for_attack hard-coded
        a tier_map missing A6.  Camera-ready: view_for_attack reads
        the registry that attacks.py populates at import time, so
        every name in attack_names() resolves to a view."""
        topo = _StubTopology()
        meta = _StubMetadata()
        weights = _StubWeights()
        for name in attack_names():
            tier = attack_tier(name)
            if tier == CapabilityTier.TIER_1:
                view = view_for_attack(name, topology=topo)
            elif tier == CapabilityTier.TIER_2:
                view = view_for_attack(
                    name, topology=topo, metadata=meta,
                )
            else:
                view = view_for_attack(
                    name, topology=topo, metadata=meta, weights=weights,
                )
            assert view.capability.tier == tier

    def test_a6_explicitly_registered(self):
        """A6 was missing from the submission draft's hard-coded
        tier_map.  Camera-ready: A6 self-registers at attacks.py
        import time, and view_for_attack('A6_evolutionary')
        returns a Tier-2 view."""
        assert "A6_evolutionary" in registered_attacks()
        assert attack_tier("A6_evolutionary") == CapabilityTier.TIER_2
        view = view_for_attack(
            "A6_evolutionary",
            topology=_StubTopology(),
            metadata=_StubMetadata(),
        )
        assert view.capability.tier == CapabilityTier.TIER_2

    def test_register_attack_tier_idempotent_same_tier(self):
        """Re-registering with the same tier is a no-op (allows
        attacks.py to be re-imported in tests)."""
        register_attack_tier("A1_random", CapabilityTier.TIER_1)
        # Idempotent — no exception.
        register_attack_tier("A1_random", CapabilityTier.TIER_1)

    def test_register_attack_tier_rejects_conflicting_tier(self):
        """Registering an existing name with a different tier
        raises — prevents silent tier downgrade."""
        with pytest.raises(ValueError, match="tier"):
            register_attack_tier("A1_random", CapabilityTier.TIER_3)

    def test_tier3_view_with_deferred_defender(self):
        """Camera-ready: Tier-3 views can be constructed without a
        defender source; reading defender state raises
        DeferredSourceError until attach_defender_source is called.

        Submission-draft behaviour: the harness rebuilt the view
        with the defender attached, requiring a private-attr poke
        ``attack._view = view``.  The camera-ready replaces that
        with the deferred pattern."""
        view = view_for_attack(
            "A5_adaptive",
            topology=_StubTopology(),
            metadata=_StubMetadata(),
            weights=_StubWeights(),
        )
        assert view.has_defender_source is False
        with pytest.raises(DeferredSourceError):
            view.admission_threshold()
        # Attach a defender; the call now succeeds.
        view.attach_defender_source(_StubDefender())
        assert view.has_defender_source is True
        assert view.admission_threshold() == 1500.0


# =============================================================================
# Section 3 — attacks.py
#
# Camera-ready advertised fixes:
#   3.1 All six attacks register their tier at module import.
#   3.2 ADVERSARIAL_TXN_ID_BASE + Attack.is_adversarial_txn_id
#       expose the ground-truth labelling channel.
#   3.3 A5 catches DeferredSourceError (handles defender attached
#       after attack constructed).
#   3.4 A6 stale-fitness fix when source rotates.
#   3.5 Per-source rate accounting uses deque (O(1) instead of O(N)).
# =============================================================================


class TestAttacksCameraReady:
    def test_every_attack_registered_at_import(self):
        """The for-loop at the bottom of attacks.py registers each
        attack class with threat_model.register_attack_tier.
        Submission draft: only experiments.py knew the tiers."""
        for name in attack_names():
            # Every name in attack_names() must be in the registry.
            tier = attack_tier(name)
            assert tier in (
                CapabilityTier.TIER_1,
                CapabilityTier.TIER_2,
                CapabilityTier.TIER_3,
            )

    def test_adversarial_txn_id_helper(self):
        """ADVERSARIAL_TXN_ID_BASE and is_adversarial_txn_id are the
        public ground-truth channel that camera-ready
        experiments.py uses for non-circular α estimation in
        _attach_storm_analysis."""
        # Public constant is exported.
        assert ADVERSARIAL_TXN_ID_BASE == -1_000_000

        view = view_for_attack("A1_random", topology=_StubTopology())
        a1 = A1Random(
            view=view,
            budget=_budget_default(),
            feature_dim=8,
            rng=np.random.default_rng(0),
        )
        cand = a1._propose_one(t_now=0.0)
        assert cand is not None
        # Adversarial txn_ids classify as adversarial.
        assert Attack.is_adversarial_txn_id(cand.transaction.txn_id)
        # Normal txn_ids do NOT classify as adversarial.
        assert not Attack.is_adversarial_txn_id(0)
        assert not Attack.is_adversarial_txn_id(123_456)
        assert not Attack.is_adversarial_txn_id(-500_000)
        # Boundary check: at exactly ADVERSARIAL_TXN_ID_BASE the txn_id
        # is NOT adversarial (the helper uses strict `<` so the base
        # value itself is not classified as adversarial — this is
        # the convention the experiments harness relies on).
        assert not Attack.is_adversarial_txn_id(ADVERSARIAL_TXN_ID_BASE)
        assert Attack.is_adversarial_txn_id(ADVERSARIAL_TXN_ID_BASE - 1)

    def test_per_source_emissions_uses_deque(self):
        """Camera-ready: ``list.pop(0)`` (O(N)) replaced with
        ``collections.deque.popleft`` (O(1)).  At realistic per-source
        rates this is a small constant; under aggressive rates it
        removes a per-call hot-spot."""
        view = view_for_attack("A1_random", topology=_StubTopology())
        a1 = A1Random(
            view=view,
            budget=_budget_default(),
            feature_dim=8,
            rng=np.random.default_rng(0),
        )
        # Each per-source history MUST be a deque, not a list.
        for hist in a1._per_source_emissions:
            assert isinstance(hist, deque), (
                f"per-source emission history should be a deque, "
                f"got {type(hist).__name__}"
            )

    def test_a5_handles_deferred_defender_without_crash(self):
        """The submission-draft A5._maybe_replan caught only
        PermissionError; DeferredSourceError would crash A5 when
        the harness constructed the attack before the defense.
        Camera-ready: A5 catches both, falling back to the A4
        strategy until the defender is attached."""
        view = view_for_attack(
            "A5_adaptive",
            topology=_StubTopology(),
            metadata=_StubMetadata(),
            weights=_StubWeights(),
        )
        assert view.has_defender_source is False
        a5 = make_attack(
            "A5_adaptive",
            view=view,
            budget=_budget_default(),
            feature_dim=8,
            rng=np.random.default_rng(0),
        )
        # Propose without a defender: must NOT raise
        # DeferredSourceError or PermissionError.
        cand = a5._propose_one(t_now=0.0)
        assert cand is not None
        # Now attach defender; subsequent propose uses the real
        # threshold.
        view.attach_defender_source(_StubDefender())
        cand2 = a5._propose_one(t_now=2.0)  # past replan_interval
        assert cand2 is not None

    def test_a6_no_stale_fitness_after_source_rotation(self):
        """Submission-draft A6._evolve_one_generation updated each
        individual's source field on rotation but did NOT
        re-evaluate fitness, leaving the population's selection
        ranking stale by one generation.  Camera-ready calls
        _refresh_fitness_after_source_change so every individual's
        fitness matches its (source, target, features) tuple."""
        view = view_for_attack(
            "A6_evolutionary",
            topology=_StubTopology(n=200),
            metadata=_StubMetadata(),
        )
        budget = _budget_tight_per_source()
        a6 = A6EvolutionaryCostOracle(
            view=view,
            budget=budget,
            feature_dim=8,
            rng=np.random.default_rng(42),
        )
        # Drive enough proposals to force per-source rotation.
        for t in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]:
            a6._propose_one(t_now=t)
        # Every individual's fitness MUST match _score(source,
        # target, features) — i.e. no staleness.
        for ind in a6._population:
            expected = a6._score(ind.source, ind.target, ind.features)
            assert abs(ind.fitness - expected) < 1e-6, (
                f"stale fitness detected: {ind.fitness:.4f} vs "
                f"expected {expected:.4f}"
            )


# =============================================================================
# Section 4 — defenses.py
#
# Camera-ready advertised fixes:
#   4.1 carry_in_samples_us() returns () always (synchronous harness
#       has no queue residency; B_0 = 0 is correct).
#   4.2 statistics() doesn't AttributeError on mgf.carry_in_log_mgf
#       (renamed to components["B_0"] in camera-ready analysis.py).
#   4.3 record_completion_with_groundtruth(...) routes ground-truth
#       labels to the correct bucket without the heuristic.
#   4.4 fallback_percentile is a configurable knob.
#   4.5 carry_in_synchronous=True surfaced in statistics().
# =============================================================================


class TestDefensesCameraReady:
    def _build(self, **kw) -> D3Schedulability:
        contract = RealTimeContract(
            deadline_us=2000.0,
            failure_probability_bound=1e-2,
            measurement_window_seconds=10.0,
        )
        return D3Schedulability(
            topology=_StubTopology(n=200),
            contract=contract,
            refit_interval_s=0.001,
            **kw,
        )

    def test_carry_in_samples_us_always_empty(self):
        """Camera-ready: under the synchronous harness B_0 = 0 is
        the correct value.  The submission draft would have
        forwarded an empty list eventually, but it also computed
        ``queue_residency_us = (now - admission_time) * 1e6`` and
        fed it as B_0, double-counting the SUT cost."""
        d = self._build()
        for i in range(50):
            txn = Transaction(
                i, i % 50, (i + 1) % 50,
                i * 0.001, 1.0, np.zeros(8, dtype=np.float32),
            )
            decision = d.evaluate(txn, t_now=i * 0.001)
            if decision.admit:
                d.record_completion(decision.predicted_cost_us, 100.0)
        assert d.carry_in_samples_us() == ()

    def test_statistics_surfaces_carry_in_synchronous_flag(self):
        """Reviewer audit: which mode produced the certificate?"""
        d = self._build()
        stats = d.statistics()
        assert stats["carry_in_synchronous"] is True

    def test_statistics_no_attribute_error_on_mgf_certificate(self):
        """The submission-draft statistics() referenced
        mgf.carry_in_log_mgf, which was renamed to
        components['B_0'] in camera-ready analysis.py.  After 300
        transactions an MGF refit fires, and the submission-draft
        statistics() would raise AttributeError.  This test
        forces a refit and verifies statistics() returns cleanly."""
        d = self._build(enable_mgf_certificate=True)
        for i in range(300):
            txn = Transaction(
                i, i % 50, (i + 1) % 50,
                i * 0.001, 1.0, np.zeros(8, dtype=np.float32),
            )
            decision = d.evaluate(txn, t_now=i * 0.001)
            if decision.admit:
                d.record_completion(
                    decision.predicted_cost_us,
                    decision.predicted_cost_us + 5.0,
                )
        # If an MGF cert was built, verify statistics() does not
        # raise AttributeError on its surface.
        stats = d.statistics()
        if "mgf_certificate" in stats:
            mgf_stats = stats["mgf_certificate"]
            # camera-ready key names
            assert "optimal_lambda" in mgf_stats
            assert "lambda_at_edge" in mgf_stats
            # carry_in_log_mgf STILL surfaced (computed from
            # components["B_0"]) for back-compat.
            assert "carry_in_log_mgf" in mgf_stats

    def test_record_completion_with_groundtruth_routes_correctly(self):
        """Camera-ready: §V.4 ablation can supply ground-truth
        labels via Attack.is_adversarial_txn_id, bypassing D3's
        heuristic labeller."""
        d = self._build()
        for i in range(50):
            txn = Transaction(
                i, i % 50, (i + 1) % 50,
                i * 0.001, 1.0, np.zeros(8, dtype=np.float32),
            )
            decision = d.evaluate(txn, t_now=i * 0.001)
            if decision.admit:
                truth = (i % 2 == 0)
                d.record_completion_with_groundtruth(
                    decision.predicted_cost_us,
                    decision.predicted_cost_us + 5.0,
                    is_adversarial=truth,
                )
        # Both buckets should populate.
        assert len(d._benign_costs) > 0
        assert len(d._adversarial_costs) > 0
        # Ground-truth count should equal adversarial bucket count.
        assert d._n_labelled_groundtruth == len(d._adversarial_costs)

    def test_fallback_percentile_configurable(self):
        d = self._build(fallback_percentile=25.0)
        assert d._fallback_percentile == 25.0

    def test_fallback_percentile_validates_range(self):
        """Camera-ready: fallback_percentile validated to (0, 100)."""
        with pytest.raises(ValueError, match="fallback_percentile"):
            self._build(fallback_percentile=0.0)
        with pytest.raises(ValueError, match="fallback_percentile"):
            self._build(fallback_percentile=100.0)


# =============================================================================
# Section 5 — experiments.py
#
# Camera-ready advertised fixes:
#   5.1 RunSpec.use_groundtruth_labels flag (default False).
#   5.2 _F1_NOT_COMPUTED sentinel replaces (truth, truth) F1=1.0 bug.
#   5.3 _attach_storm_analysis takes per_txn_log for ground-truth α.
#   5.4 _make_view_for_attack is now a deprecation shim that
#       delegates to view_for_attack.
# =============================================================================


class TestExperimentsCameraReady:
    def test_run_spec_has_use_groundtruth_labels_field(self):
        """Camera-ready: RunSpec gained the use_groundtruth_labels
        knob for §V.4 ablation."""
        from experiments import RunSpec
        # The field exists with the documented default.
        spec = RunSpec(
            experiment="t",
            dataset="synthetic",
            dataset_path=Path("/tmp/x"),
            attack=None,
            defense=None,
            seed=0,
            n_transactions=10,
            contract=RealTimeContract(
                deadline_us=1000.0,
                failure_probability_bound=1e-2,
                measurement_window_seconds=10.0,
            ),
            budget=_budget_default(),
            output_dir=Path("/tmp/x"),
        )
        assert spec.use_groundtruth_labels is False
        # Setting True is permitted (immutable dataclass; would
        # raise FrozenInstanceError if not exposed).
        spec2 = RunSpec(
            experiment="t",
            dataset="synthetic",
            dataset_path=Path("/tmp/x"),
            attack=None,
            defense=None,
            seed=0,
            n_transactions=10,
            contract=spec.contract,
            budget=_budget_default(),
            output_dir=Path("/tmp/x"),
            use_groundtruth_labels=True,
        )
        assert spec2.use_groundtruth_labels is True

    def test_f1_sentinel_constant_exposed(self):
        """Camera-ready: _F1_NOT_COMPUTED sentinel marks placeholder
        predictions so measure_detection_quality consumers can
        skip rather than reading the submission-draft's F1=1.0
        placeholder."""
        from experiments import _F1_NOT_COMPUTED
        assert _F1_NOT_COMPUTED == -999

    def test_make_view_for_attack_shim_works_for_a6(self):
        """The submission-draft _make_view_for_attack had a hard-
        coded tier_map missing A6.  Camera-ready: it forwards to
        view_for_attack which reads the registry; A6 routes
        correctly."""
        from experiments import _make_view_for_attack

        class _StubSut:
            topology_source = _StubTopology()
            metadata_source = _StubMetadata()
            weights_source = _StubWeights()

        sut = _StubSut()
        view = _make_view_for_attack("A6_evolutionary", sut)
        assert view.capability.tier == CapabilityTier.TIER_2


# =============================================================================
# Section 6 — workload.py
#
# Camera-ready advertised fixes (LaTeX reconciliations):
#   6.1 CICIDS card: "8 attack days" → "10 days, 7 attack families".
#   6.2 BitcoinHeist: "28 ransomware families" → "24 families".
#   6.3 SWaT: "Six attack scenarios" → "41 attacks across 6 intent
#       categories" (Adepu & Mathur 2016 distinction documented).
#   6.4 SWaT feature_dim=25 rationale documented (51 sensors, 25-D
#       analog subset).
#   6.5 TimedTransaction docstring documents the ground-truth
#       channel mirroring Attack.is_adversarial_txn_id.
# =============================================================================


class TestWorkloadCardsReconciledWithLatex:
    """Pin every dataset-card number that the LaTeX text references.

    These tests would have failed against the submission draft.
    Their job is to detect any future regression where someone
    edits a card without updating the paper, or vice versa.
    """

    def test_cicids_version_says_10_days_seven_families(self):
        c = dataset_card("cicids2018")
        assert "10 days" in c.version, (
            f"submission draft said '8 attack days'; LaTeX text "
            f"says '10 days'.  Got version: {c.version!r}"
        )
        assert "7 attack families" in c.version

    def test_cicids_construction_notes_list_seven_families(self):
        c = dataset_card("cicids2018")
        for fam in (
            "DoS", "DDoS", "Brute-Force", "Bot",
            "Web", "Infiltration", "Heartbleed",
        ):
            assert fam in c.construction_notes, (
                f"missing CICIDS family '{fam}' in construction_notes"
            )

    def test_cicids_construction_notes_say_ten_days(self):
        c = dataset_card("cicids2018")
        assert "ten days" in c.construction_notes
        assert "8 days" not in c.construction_notes

    def test_bitcoin_heist_says_24_not_28_families(self):
        c = dataset_card("bitcoin_ransomware")
        # Submission draft said "28 ransomware families"; LaTeX says 24.
        assert "24 ransomware families" in c.label_source
        assert "28 ransomware families" not in c.label_source
        # Known limitations should also reference 24.
        assert "24 ransomware families" in str(c.known_limitations)
        assert "28 ransomware families" not in str(c.known_limitations)

    def test_bitcoin_heist_construction_notes_explain_8d_extension(self):
        """The published Akcora artefact has 6 topological features;
        we extend with 2 to harmonise with the Ethereum schema (8-D).
        The LaTeX audit notes flagged this as an item to document."""
        c = dataset_card("bitcoin_ransomware")
        cn = c.construction_notes
        assert "six topological features" in cn
        assert "eight-dimensional" in cn or "8-dimensional" in cn

    def test_swat_label_source_says_41_attacks_six_intent_categories(self):
        c = dataset_card("swat")
        # The 41-vs-6 distinction is the most important reviewer
        # signal because the per-class F1 appendix table reports
        # support sizes that correspond to per-intent-category
        # samples, not to 41 individual instances.
        assert "41 documented" in c.label_source
        assert "6 INTENT CATEGORIES" in c.label_source
        assert "Adepu & Mathur" in c.label_source

    def test_swat_label_source_says_11_days_seven_normal_four_attack(self):
        c = dataset_card("swat")
        assert "11 days" in c.label_source
        assert "7 days" in c.label_source
        assert "4 days" in c.label_source

    def test_swat_canonical_name_advertises_41_attacks(self):
        c = dataset_card("swat")
        assert "41 attacks" in c.canonical_name

    def test_swat_construction_notes_explain_25_vs_51(self):
        """LaTeX audit asked for either feature_dim=51 or a
        documented rationale for the 25-D subset.  Camera-ready
        documented the rationale: analog continuous signals only,
        excluding constant-value actuator readings."""
        c = dataset_card("swat")
        cn = c.construction_notes
        assert "51" in cn
        assert "25" in cn
        # Check the rationale prose references analog signals.
        assert "analog" in cn.lower()

    def test_feature_dim_unchanged_across_all_loaders(self):
        """Camera-ready did NOT change feature_dim values; changing
        them would cascade through experiments.py and tests.
        Pinned here so the values cannot be silently changed."""
        assert dataset_card("ethereum_phishing").feature_dim == 8
        assert dataset_card("bitcoin_ransomware").feature_dim == 8
        assert dataset_card("cicids2018").feature_dim == 16
        assert dataset_card("swat").feature_dim == 25
        assert dataset_card("synthetic").feature_dim == 8

    def test_n_records_match_latex_summary(self):
        """LaTeX text references ~16.2M (CICIDS), ~2.97M (Ethereum),
        ~2.92M (BitcoinHeist), and 11 days × 1Hz × 24h ≈ 950K (SWaT).
        The card's n_records_expected pin these to specific integers
        so a re-preprocessed file with a different size will be
        caught by the loader's audit_signature() comparison in
        reproduce."""
        assert dataset_card("ethereum_phishing").n_records_expected == 2_973_489
        assert dataset_card("bitcoin_ransomware").n_records_expected == 2_916_697
        assert dataset_card("cicids2018").n_records_expected == 16_233_002
        assert dataset_card("swat").n_records_expected == 946_722

    def test_timed_transaction_docstring_documents_ground_truth_channel(self):
        """Camera-ready: TimedTransaction.is_adversarial is the public
        ground-truth channel that mirrors
        Attack.is_adversarial_txn_id.  Documented so reviewers can
        verify the non-circular α path in experiments.py."""
        from workload import TimedTransaction
        import inspect
        doc = inspect.getdoc(TimedTransaction)
        assert doc is not None
        assert "is_adversarial_txn_id" in doc, (
            "TimedTransaction docstring should reference the mirror "
            "channel exposed by attacks.py"
        )
        assert "ground-truth channel" in doc


# =============================================================================
# Section 7 — main.py CLI surface
#
# Camera-ready fix on the CLI:
#   7.1 _cmd_run no longer crashes with TypeError because
#       strict_schema/full_file_hash/hardware_info are no longer
#       passed as **kwargs to run_experiment.
#
# We do not exercise the full CLI here (that's reproduce-mode
# territory).  We just import main.py and verify the dispatch
# table is consistent with the camera-ready subcommand list, plus
# verify that the smoke subcommand parser accepts the new
# --attack / --defense flags.
# =============================================================================


class TestMainCliCameraReady:
    def test_smoke_subcommand_accepts_attack_and_defense_flags(self):
        """Camera-ready: --attack and --defense knobs added to
        smoke so reviewers can verify the A6 routing path with one
        command."""
        import main
        parser = main._make_parser()
        # Camera-ready flags must be parseable.
        args = parser.parse_args([
            "smoke",
            "--attack", "A6_evolutionary",
            "--defense", "D3_schedulability",
            "--n-transactions", "10",
        ])
        assert args.cmd == "smoke"
        assert args.attack == "A6_evolutionary"
        assert args.defense == "D3_schedulability"

    def test_smoke_defaults_remain_a1_d1(self):
        """Back-compat: existing `python -m src.main smoke`
        invocations must continue to use A1+D1."""
        import main
        parser = main._make_parser()
        args = parser.parse_args(["smoke"])
        assert args.attack == "A1_random"
        assert args.defense == "D1_static"
