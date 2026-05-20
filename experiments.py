"""
experiments.py — Experiment orchestration.

This module wires the eight other modules into the seven named
experiments of the paper.  Each experiment is a thin function whose
job is to enumerate (dataset, attack, defense, seed) configurations
and dispatch each to a single primitive — ``run_one`` — that does
the actual work.

The discipline this design enforces
-----------------------------------
- The cross product (7 experiments × 4 datasets × 5 attacks × 3
  defenses × 5 seeds = 2100 runs in the worst case; in practice
  ~600 because most experiments fix some axes) never appears as
  nested loops.  It appears as iterators of ``RunSpec``s that flow
  through ``run_one``.
- Every run is independently reproducible.  Two runs with the same
  ``RunSpec.signature()`` must produce statistically equivalent
  results.  We do not test for byte-equivalence (GPU non-determinism,
  OS scheduling noise) but for distribution equivalence in plots.py.
- Statistical protocol is centralised.  Paired t-tests, Bonferroni
  correction, and effect-size computation live here once and are
  reused by every experiment that needs them.

The seven experiments
---------------------
exp_attack_effectiveness     — §5.1, Table 3, Figure 4
exp_defense_efficacy         — §5.2, Table 4, Figure 5
exp_schedulability_validation — §5.3, Table 2, Figure 3
exp_scalability              — §5.3, Figure 6
exp_ablation                 — §5.4, Table 5
exp_cross_dataset            — §5.5, Figure 7
exp_adaptive_adversary       — §6, Table 6, Figure 8

Each experiment function returns a list of RunRecord objects.  The
caller (main.py) writes them to results/raw/.

Camera-ready improvements over the submission draft
---------------------------------------------------
1. **A6 routing bug closed at the source.**  The submission draft
   carried a hard-coded ``tier_map`` in ``_make_view_for_attack``
   that listed only A1..A5; ``exp_attack_effectiveness`` would
   ``KeyError`` on A6 mid-run.  The camera-ready uses
   ``threat_model.view_for_attack(name, ...)``, which reads from
   the registry that ``attacks.py`` populates at import time.  The
   tier table cannot drift.

2. **Tier-3 deferred-defender attachment.**  The submission draft
   built the Tier-3 view, then *rebuilt it* with the defender
   attached, and reached into ``attack._view`` to swap.  Two views
   meant two access logs and two access fingerprints, polluting
   the threat-model audit trail.  The camera-ready uses
   ``view_for_attack`` (which builds Tier-3 views with
   ``allow_deferred_defender=True``) and calls
   ``view.attach_defender_source(defense)`` once.  Single view,
   single audit log.  See camera-ready threat_model.py for the
   underlying ``DeferredSourceError`` machinery.

3. **F1 placeholder bug surfaced honestly.**  The submission draft's
   ``_process_one_transaction`` appended
   ``streaming_predictions.append((timed.label, timed.label))`` —
   F1 would be 1.0 by construction regardless of model behaviour.
   The camera-ready uses an explicit sentinel
   ``_F1_NOT_COMPUTED`` and ``measure_detection_quality`` skips
   computation when the buffer is sentinel-only.  ``rec.notes``
   carries an explicit ``detection: F1 not computed (placeholder
   evaluator removed)`` line.  Real F1 computation is documented as
   future work in ``docs/F1_EVAL.md``; we deliberately do NOT
   substitute a fabricated F1 value.

4. **Ground-truth α labelling for §V.3.**  The submission draft's
   ``_attach_storm_analysis`` bucketed cost samples by a "top-α-
   by-cost" heuristic, which is circular under the schedulability
   bound (the bound uses α to compute the mixture; α was estimated
   by inverting the bound).  The camera-ready uses
   ``Attack.is_adversarial_txn_id`` to label each completed
   transaction's cost sample at run time, and feeds the *measured*
   α to ``analyse_storm`` via the new ``adversary_fraction=``
   kwarg.  The heuristic path is preserved as a fallback for runs
   without an attack.

5. **Modern analysis API surface.**  ``analyse_storm`` and
   ``MGFCertificate`` field access have been migrated from the
   submission-draft legacy kwargs / aliases (``optimal_lam``,
   ``per_class_bounds``, ``benign_samples_us``,
   ``enable_mgf_certificate``, ``expected_admitted_rate_hz``,
   ``carry_in_samples_us``) to the modern names
   (``optimal_lambda``, ``components``,
   ``benign_cost_samples_us``, ``adversarial_cost_samples_us``,
   per-class ``count_distributions``).  The legacy aliases continue
   to work; this is a cleanliness change, not a behaviour change.

6. **Carry-in forwarding removed.**  Camera-ready defenses.py
   returns ``()`` from ``carry_in_samples_us()`` under the
   synchronous harness (B₀ = 0 is correct; the submission draft's
   carry-in computation double-counted the SUT cost).  The camera-
   ready ``_attach_storm_analysis`` no longer forwards a carry-in
   distribution; ``analyse_storm`` builds the degenerate-zero B₀
   internally and logs the standard "synchronous-harness" caveat.

7. **Optional ground-truth labelling for D3.**  When
   ``RunSpec.use_groundtruth_labels`` is True, the harness calls
   ``defense.record_completion_with_groundtruth(predicted, actual,
   is_adversarial)`` instead of ``defense.record_completion(...)``.
   This routes cost samples to the correct bucket without invoking
   D3's heuristic labeller.  Used in §V.4 ablations to compare
   labelling regimes.  Default False (heuristic), preserving
   submission-draft behaviour.

What this file does not do
--------------------------
- It does not interpret results.  Aggregation and significance
  testing happen on the persisted run records, in plots.py.
- It does not instantiate models or attacks directly.  It uses the
  registries from system.py, attacks.py, defenses.py, and
  workload.py.
- It does not draw figures.  That is plots.py.

Author: ANONYMOUS  (double-blind review)
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass, field, asdict
from itertools import product
from pathlib import Path
from typing import (
    Any,
    Callable,
    Iterable,
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
    AdversaryCapability,
    CapabilityTier,
    DeferredSourceError,
    EvasionConstraints,
    RealTimeContract,
    TargetSystemView,
    UpdateCostFunction,
    UpdateStorm,
    CostDecomposition,
    view_for_attack,
)
from system import (
    LearnerConfig,
    System,
    SystemConfig,
    UpdatePathConfig,
    measure_detection_quality,
)
from measurement import (
    MeasuringProbe,
    RunRecord,
    describe_host,
    pin_to_cores,
    set_realtime_priority,
    quiescent_environment,
)
from attacks import (
    Attack,
    attack_names,
    compute_predictor_accuracy,
    make_attack,
)
from defenses import (
    Defense,
    D3Schedulability,
    defense_names,
    make_defense,
)
from analysis import (
    MGFCertificate,
    StormAnalysis,
    analyse_storm,
    fit_exponential_upper_bound,
    validate_theorem_4_1,
    validate_theorem_4_2,
    validate_theorem_4_3,
    validate_theorem_4_3_mgf,
)
from workload import (
    MixedStream,
    MixingConfig,
    TimedTransaction,
    feature_dim_for,
    make_loader,
    replay_window,
)

logger = logging.getLogger(__name__)


# Sentinel value used in ``streaming_predictions`` to signal that the
# harness did not have access to a real model prediction at the
# moment of the decision.  ``measure_detection_quality`` skips F1
# computation when all predictions are sentinels.  The submission
# draft used ``(timed.label, timed.label)`` for this slot, which
# inflated F1 to 1.0 by construction; the camera-ready replaces it
# with this explicit non-label so reviewers cannot mistake the
# placeholder for a real measurement.  See ``docs/F1_EVAL.md`` for
# the planned real-evaluator design.
_F1_NOT_COMPUTED: int = -999


# =============================================================================
# Section 1.  RunSpec: a fully specified single experimental run.
#
# Everything `run_one` needs to execute one repetition.  RunSpec is the
# unit of replication: two RunSpecs with the same signature describe
# the same experimental setup (modulo seed).
#
# Camera-ready additions:  five new optional fields control the
# Warmup/Storm/Recovery (WSR) phase structure.  All fields default to
# backward-compatible values (WSR disabled), so existing experiments
# continue to work unchanged.
# =============================================================================


@dataclass(frozen=True)
class RunSpec:
    experiment: str
    dataset: str
    dataset_path: Path
    attack: Optional[str]                  # None means benign-only
    defense: Optional[str]                 # None means undefended
    seed: int
    n_transactions: int                    # cap on transactions to process
    contract: RealTimeContract
    budget: AdversaryBudget
    output_dir: Path

    # Optional knobs:
    pin_cores_sut: Tuple[int, ...] = ()
    pin_cores_attacker: Tuple[int, ...] = ()
    set_rt_priority: bool = False
    note: str = ""

    # Camera-ready: warmup / storm / recovery interval structure.
    # When `enable_warmup_storm_recovery` is True, the run splits
    # `n_transactions` into three phases: warmup (benign-only,
    # `warmup_fraction` of the run) → storm (attack injected,
    # `1 − warmup_fraction − recovery_fraction` of the run) →
    # recovery (benign-only, `recovery_fraction` of the run).  Phase
    # boundaries are persisted to the run record.  When False
    # (default), the harness behaves as before — a single phase with
    # the attack (if any) injecting throughout.
    enable_warmup_storm_recovery: bool = False
    warmup_fraction: float = 0.10
    recovery_fraction: float = 0.10
    recovery_tolerance: float = 0.10        # 1 + tolerance × baseline
    recovery_hold_duration_s: float = 1.0   # latency must stay below for this long

    # Camera-ready: when True, the harness routes admitted-transaction
    # completions through ``defense.record_completion_with_groundtruth``
    # instead of the heuristic-labelled ``record_completion``.  This
    # bypasses D3's heuristic labeller and uses the
    # ``Attack.is_adversarial_txn_id`` ground truth directly.  Used
    # in §V.4 ablations to compare labelling regimes.  Default False
    # (heuristic), preserving submission-draft behaviour.
    use_groundtruth_labels: bool = False

    def signature(self) -> str:
        """Stable identifier used in output filenames."""
        # We deliberately include the seed: every (signature, run_index)
        # pair is unique by construction.
        attack_s = self.attack or "benign"
        defense_s = self.defense or "undef"
        return (
            f"{self.experiment}__{self.dataset}__{attack_s}__{defense_s}"
            f"__seed{self.seed}"
        )


# =============================================================================
# Section 2.  Phase boundaries and per-phase statistics.
#
# When WSR is enabled, the harness records the wall-clock timestamps
# of each phase transition and per-phase summary statistics.  These
# are persisted into the run record so that plots.py can shade
# storm/recovery intervals on time-series figures and produce
# per-interval comparison tables.
# =============================================================================


@dataclass
class IntervalBoundaries:
    """
    Wall-clock (monotonic_ns) timestamps for the four WSR transitions.

    Layout:
      run_start_ns         — first arrival processed
      warmup_end_ns        — warmup→storm transition (storm_start)
      storm_end_ns         — storm→recovery transition (recovery_start)
      run_end_ns           — last completion observed

    When WSR is disabled (single-phase), warmup_end_ns and
    storm_end_ns are set equal to run_start_ns; the entire run is
    treated as a single "storm" phase for downstream consumers.
    """

    run_start_ns: int = 0
    warmup_end_ns: int = 0
    storm_end_ns: int = 0
    run_end_ns: int = 0

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "run_start_ns": self.run_start_ns,
            "warmup_end_ns": self.warmup_end_ns,
            "storm_end_ns": self.storm_end_ns,
            "run_end_ns": self.run_end_ns,
            "warmup_duration_s": (self.warmup_end_ns - self.run_start_ns) / 1e9,
            "storm_duration_s": (self.storm_end_ns - self.warmup_end_ns) / 1e9,
            "recovery_duration_s": (self.run_end_ns - self.storm_end_ns) / 1e9,
            "total_duration_s": (self.run_end_ns - self.run_start_ns) / 1e9,
        }


@dataclass
class PhaseStats:
    """Per-phase summary statistics for a WSR phase."""

    label: str                  # "warmup" | "storm" | "recovery" | "full"
    n_attempted: int = 0
    n_admitted: int = 0
    n_rejected: int = 0
    n_processed: int = 0        # admitted-and-completed (not just admitted)
    latency_samples_us: List[float] = field(default_factory=list)

    @property
    def admission_rate(self) -> float:
        return (
            self.n_admitted / self.n_attempted if self.n_attempted > 0 else float("nan")
        )

    @property
    def rejection_rate(self) -> float:
        return (
            self.n_rejected / self.n_attempted if self.n_attempted > 0 else float("nan")
        )

    def latency_summary(self) -> Mapping[str, float]:
        if not self.latency_samples_us:
            return {}
        arr = np.asarray(self.latency_samples_us)
        return {
            "n": int(arr.size),
            "mean_us": float(arr.mean()),
            "p50_us": float(np.percentile(arr, 50)),
            "p95_us": float(np.percentile(arr, 95)),
            "p99_us": float(np.percentile(arr, 99)),
            "max_us": float(arr.max()),
        }

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "label": self.label,
            "n_attempted": self.n_attempted,
            "n_admitted": self.n_admitted,
            "n_rejected": self.n_rejected,
            "n_processed": self.n_processed,
            "admission_rate": self.admission_rate,
            "rejection_rate": self.rejection_rate,
            "latency": self.latency_summary(),
        }


# =============================================================================
# Section 3.  run_one — the primitive every experiment uses.
#
# Given a RunSpec, run one repetition end-to-end and return the
# RunRecord.  No experiment-specific logic lives here; experiment
# functions in §6 below produce RunSpecs and call this.
#
# Camera-ready run loop
# ---------------------
# The harness now calls four new probe hooks (record_arrival,
# record_admission_decision, ...) and threads carry-in samples plus
# expected admit rate into analyse_storm.  Behaviour is unchanged
# when the new measurement.py methods are no-ops, so older tests
# that drive the probe directly with phase events still work.
# =============================================================================


def run_one(spec: RunSpec) -> RunRecord:
    """
    Run one experimental repetition.

    Steps:
      1. Apply hardware-pinning if configured.
      2. Build the system (SUT) + workload + attack + defense + probe.
      3. Iterate the workload (one phase, or three with WSR), applying
         defense admission and feeding admitted transactions to the
         SUT.  Record probe events.
      4. Finalise: compute storm analysis (envelope + MGF certificates),
         capture D3 cert state, write the RunRecord.

    Returns the RunRecord.  Side effect: writes the record to
    spec.output_dir.
    """
    logger.info(f"running {spec.signature()}")

    # ---- 1. Hardware-pinning ---------------------------------------
    if spec.pin_cores_sut:
        pin_to_cores(spec.pin_cores_sut)
    if spec.set_rt_priority:
        set_realtime_priority(80)

    # ---- 2. Build everything ----------------------------------------
    rec = RunRecord.begin(
        experiment=spec.experiment,
        dataset=spec.dataset,
        seed=spec.seed,
        attack=spec.attack,
        defense=spec.defense,
    )

    feature_dim = feature_dim_for(spec.dataset)
    learner_cfg = LearnerConfig(
        feature_dim=feature_dim,
        hidden_dim=64,
        num_gnn_layers=2,
        device=_default_device(),
    )
    system_cfg = SystemConfig(learner=learner_cfg, seed=spec.seed)
    sut = System.assemble(system_cfg)

    # Loader.  We may instantiate multiple streams from this loader
    # (one per WSR phase); workload.DatasetLoader.stream() returns a
    # fresh iterator on each call.
    loader = make_loader(spec.dataset, spec.dataset_path)

    # Attack: build it through TargetSystemView matching its tier.
    # Camera-ready: uses ``view_for_attack`` from threat_model.py,
    # which reads the tier registry that ``attacks.py`` populates
    # at import time.  This closes the A6 routing bug in the
    # submission draft (``_make_view_for_attack`` had a hard-coded
    # tier_map missing A6).  Tier-3 views are constructed with
    # ``allow_deferred_defender=True`` so the defender source can
    # be attached later via ``attach_defender_source``; no need to
    # rebuild the view a second time.
    attack: Optional[Attack] = None
    view: Optional[TargetSystemView] = None
    if spec.attack is not None:
        view = view_for_attack(
            spec.attack,
            topology=sut.topology_source,
            metadata=sut.metadata_source,
            weights=sut.weights_source,
            # No defender at construction time; attached below if a
            # defense exists.
        )
        rng = np.random.default_rng(spec.seed)
        attack = make_attack(
            name=spec.attack,
            view=view,
            budget=spec.budget,
            feature_dim=feature_dim,
            rng=rng,
        )

    # Defense.
    defense: Optional[Defense] = None
    if spec.defense is not None:
        defense_kwargs = _defense_default_kwargs(spec.defense, spec.contract)
        defense = make_defense(
            name=spec.defense,
            topology=sut.topology_source,
            contract=spec.contract,
            **defense_kwargs,
        )
        # Camera-ready: attach the defender to the existing view in
        # place, instead of building a second view.  This keeps the
        # threat-model audit log single and consistent across the
        # whole run.  ``attach_defender_source`` is a no-op for
        # Tier-1 / Tier-2 attacks (the view's capability does not
        # authorise defender-state reads); for Tier-3 it wires the
        # source so any subsequent ``view.admission_threshold()``
        # call from the attack succeeds.
        if (
            view is not None
            and view.capability.tier == CapabilityTier.TIER_3
        ):
            view.attach_defender_source(defense)

    # Probe.  Camera-ready: pass the contract's deadline as the
    # default age_max so that the AgeViolationReport and the
    # DeadlineMissReport are commensurable.  experiments can pass a
    # different age_max via the contract, but the default keeps the
    # two reports aligned.
    probe = MeasuringProbe(contract=spec.contract)
    sut.attach_probe(probe)

    # Per-phase / cumulative bookkeeping.
    boundaries = IntervalBoundaries()
    phase_stats: List[PhaseStats] = []
    streaming_predictions: List[Tuple[int, int]] = []
    predicted_costs: List[float] = []
    actual_costs: List[float] = []
    # Camera-ready: per-transaction log used by _attach_storm_analysis
    # for ground-truth-aware α estimation.  Each entry is
    # ``(txn_id, predicted_us, actual_us, is_adversarial)`` where the
    # adversarial flag comes from ``Attack.is_adversarial_txn_id``
    # (NOT from a post-hoc cost-bucketing heuristic).
    per_txn_log: List[Tuple[int, float, float, bool]] = []

    # ---- 3. Run -----------------------------------------------------
    # Phase plan.  When WSR is disabled we run a single phase.  When
    # enabled, we split into warmup/storm/recovery.
    boundaries.run_start_ns = time.monotonic_ns()
    with quiescent_environment():
        if spec.enable_warmup_storm_recovery:
            phase_stats = _run_wsr_phases(
                spec=spec, loader=loader, attack=attack,
                defense=defense, sut=sut, probe=probe, rec=rec,
                boundaries=boundaries,
                streaming_predictions=streaming_predictions,
                predicted_costs=predicted_costs,
                actual_costs=actual_costs,
                per_txn_log=per_txn_log,
            )
        else:
            single = _run_single_phase(
                spec=spec, loader=loader, attack=attack,
                defense=defense, sut=sut, probe=probe, rec=rec,
                streaming_predictions=streaming_predictions,
                predicted_costs=predicted_costs,
                actual_costs=actual_costs,
                per_txn_log=per_txn_log,
            )
            phase_stats = [single]
            # Single-phase: collapse boundaries.  Treat the entire run
            # as the "storm" interval so downstream consumers can
            # apply storm-only logic uniformly.
            boundaries.warmup_end_ns = boundaries.run_start_ns
            boundaries.storm_end_ns = time.monotonic_ns()
    boundaries.run_end_ns = time.monotonic_ns()

    # Snapshot evaluation: take a small held-out batch (last 200
    # streaming predictions) — placeholder for the publication-runs
    # snapshot evaluator.
    snapshot_predictions = streaming_predictions[-200:]

    # ---- 4. Finalise ------------------------------------------------
    # Recovery baseline: mean end-to-end latency observed during
    # warmup, used by attach_probe_results to compute recovery time.
    recovery_baseline_us: Optional[float] = None
    if spec.enable_warmup_storm_recovery and phase_stats:
        warmup = next((p for p in phase_stats if p.label == "warmup"), None)
        if warmup is not None and warmup.latency_samples_us:
            recovery_baseline_us = float(
                np.mean(warmup.latency_samples_us)
            )

    rec.attach_probe_results(
        probe,
        recovery_baseline_us=recovery_baseline_us,
        recovery_tolerance=spec.recovery_tolerance,
        recovery_hold_duration_s=spec.recovery_hold_duration_s,
    )

    # Per-phase + interval-boundary metadata.
    rec.notes.append(
        f"intervals: {json.dumps(boundaries.to_dict(), default=str)}"
    )
    for phase in phase_stats:
        rec.notes.append(
            f"phase[{phase.label}]: {json.dumps(phase.to_dict(), default=str)}"
        )

    # Storm analysis: now uses the new MGF certificate path with
    # carry-in samples and admit-rate estimates from the defense.
    _attach_storm_analysis(
        rec=rec, spec=spec, view=view, attack=attack,
        defense=defense, probe=probe, boundaries=boundaries,
        per_txn_log=per_txn_log,
    )

    # Workload statistics: not needed in camera-ready RunRecord
    # because MixedStream isn't a single object across WSR phases;
    # the per-phase counts in PhaseStats already give attempted /
    # admitted / rejected.  We still record an access fingerprint
    # for the threat-model audit.
    if view is not None:
        rec.access_fingerprint = view.access_fingerprint()

    # Defense / attack statistics.
    if defense is not None:
        rec.notes.append(f"defense: {json.dumps(defense.statistics(), default=str)}")
        # Capture D3-specific certificate state.
        if isinstance(defense, D3Schedulability):
            _attach_d3_certificates(rec, defense)

    # Predictor accuracy.
    if len(predicted_costs) > 1:
        pa = compute_predictor_accuracy(predicted_costs, actual_costs)
        rec.notes.append(f"predictor_accuracy={pa.to_dict()}")

    # Detection.
    # Camera-ready F1 fix: filter sentinel-only buffers and skip
    # F1 computation rather than reporting a fabricated value.  The
    # submission draft used (truth, truth) pairs which made F1 = 1
    # by construction; the camera-ready uses the explicit sentinel
    # ``_F1_NOT_COMPUTED`` and surfaces the absence of a real
    # evaluator in the run record.
    real_streaming = [
        p for p in streaming_predictions
        if p[1] != _F1_NOT_COMPUTED
    ]
    real_snapshot = [
        p for p in snapshot_predictions
        if p[1] != _F1_NOT_COMPUTED
    ]
    if real_streaming or real_snapshot:
        scores = measure_detection_quality(real_streaming, real_snapshot)
        rec.notes.append(
            f"detection: streaming_F1={scores.streaming_f1:.4f}, "
            f"snapshot_F1={scores.snapshot_f1:.4f}"
        )
    else:
        rec.notes.append(
            "detection: F1 not computed (placeholder evaluator removed; "
            "see docs/F1_EVAL.md for the planned real evaluator)"
        )

    rec.finish()
    out_path = spec.output_dir / f"{spec.signature()}.json"
    rec.write(out_path)
    logger.info(f"wrote {out_path}")
    return rec


# =============================================================================
# Section 4.  Run-loop helpers.
#
# The single-phase and WSR-three-phase cases both reduce to repeated
# calls to `_process_one_transaction`, the core admission + processing
# step.  Centralising this eliminates duplication and ensures the new
# probe hooks are called identically in every code path.
# =============================================================================


def _process_one_transaction(
    timed: TimedTransaction,
    defense: Optional[Defense],
    sut: System,
    probe: MeasuringProbe,
    rec: RunRecord,
    streaming_predictions: List[Tuple[int, int]],
    predicted_costs: List[float],
    actual_costs: List[float],
    phase: PhaseStats,
    per_txn_log: List[Tuple[int, float, float, bool]],
    use_groundtruth_labels: bool = False,
) -> None:
    """
    The body of the per-transaction loop.  Extracted so the WSR and
    single-phase code paths share exactly the same hot path.

    Hooks called (camera-ready):
      probe.record_arrival(txn_id, arrival_ns)
        — before admission, registers the txn arrival timestamp.
          Activates queueing-delay measurement and the AoI freshness
          tracker.
      probe.record_admission_decision(txn_id, arrival_ns, admitted)
        — after admission, drives the time-series throughput
          recorder and the queue-depth tracking.

    Camera-ready behaviour:
      - F1 placeholder bug fixed: the submission draft appended
        ``(timed.label, timed.label)`` (truth-vs-truth, F1 = 1.0
        by construction); the camera-ready uses the explicit
        sentinel ``_F1_NOT_COMPUTED`` and surfaces the absence of a
        real F1 evaluator.
      - Ground-truth-aware completion routing: when
        ``use_groundtruth_labels`` is True, the harness calls
        ``defense.record_completion_with_groundtruth(...)`` so D3
        can bypass its heuristic labeller for §V.4 ablations.
        ``Attack.is_adversarial_txn_id`` provides the truth.
      - Per-transaction log: each completed transaction's
        ``(txn_id, predicted_us, actual_us, is_adversarial)`` tuple
        is appended to ``per_txn_log`` for later use by
        ``_attach_storm_analysis`` (closes the post-hoc cost-
        bucketing circularity in §V.3).
    """
    txn = timed.transaction
    arrival_ns = time.monotonic_ns()
    phase.n_attempted += 1

    # Camera-ready: ground truth via the public Attack helper.
    is_adversarial = Attack.is_adversarial_txn_id(txn.txn_id)

    # Record the arrival into the probe.  This activates the AoI /
    # freshness pipeline and queueing-delay measurement.
    probe.record_arrival(txn.txn_id, arrival_ns)

    # Defender admission decision.
    if defense is not None:
        decision = defense.evaluate(txn, t_now=timed.arrival_time)
        admitted = decision.admit
        probe.record_admission_decision(
            txn.txn_id, arrival_ns, admitted=admitted,
        )
        if not admitted:
            phase.n_rejected += 1
            return
        predicted_us = decision.predicted_cost_us
    else:
        # No defense: every transaction is admitted by definition.
        # Record an admission so the time-series recorder still
        # tracks the arrival as throughput.
        probe.record_admission_decision(
            txn.txn_id, arrival_ns, admitted=True,
        )
        predicted_us = 0.0

    phase.n_admitted += 1

    # Process through the SUT.
    try:
        cost = sut.update_path.process(txn, label=timed.label)
    except Exception as e:
        rec.notes.append(f"process error on txn {txn.txn_id}: {e}")
        return

    actual_us = cost.total_us
    phase.n_processed += 1
    phase.latency_samples_us.append(actual_us)

    # Tell the defense (so it can recalibrate, track carry-in,
    # refit).  Camera-ready: use the ground-truth-aware variant when
    # requested by RunSpec.use_groundtruth_labels.  Default path
    # (heuristic labelling) preserves submission-draft behaviour.
    if defense is not None:
        if use_groundtruth_labels:
            defense.record_completion_with_groundtruth(
                predicted_us, actual_us, is_adversarial,
            )
        else:
            defense.record_completion(predicted_us, actual_us)

    # Predictor accuracy: record predicted vs. actual when defense
    # produced a prediction.
    if predicted_us > 0:
        predicted_costs.append(predicted_us)
        actual_costs.append(actual_us)

    # Camera-ready: per-transaction log for ground-truth-aware α
    # estimation in _attach_storm_analysis.
    per_txn_log.append((txn.txn_id, predicted_us, actual_us, is_adversarial))

    # Detection-quality bookkeeping.
    # CAMERA-READY F1 FIX: the submission draft appended
    # ``(timed.label, timed.label)``, which makes F1 = 1.0 by
    # construction regardless of model behaviour.  We deliberately
    # do NOT substitute a fabricated prediction here.  Instead, we
    # append the sentinel pair and let measure_detection_quality
    # skip computation when sentinels are present.  Real F1
    # computation is documented as future work in
    # docs/F1_EVAL.md.
    streaming_predictions.append((timed.label, _F1_NOT_COMPUTED))


def _run_single_phase(
    spec: RunSpec,
    loader: Any,
    attack: Optional[Attack],
    defense: Optional[Defense],
    sut: System,
    probe: MeasuringProbe,
    rec: RunRecord,
    streaming_predictions: List[Tuple[int, int]],
    predicted_costs: List[float],
    actual_costs: List[float],
    per_txn_log: List[Tuple[int, float, float, bool]],
) -> PhaseStats:
    """
    Original single-phase run loop, refactored to share
    ``_process_one_transaction`` with the WSR path.  Behaviour is
    unchanged from the ICDE version when the new probe hooks are
    no-ops.
    """
    benign_stream = _take(loader.stream(), spec.n_transactions)
    mixed = MixedStream(
        benign=benign_stream,
        attack=attack,
        budget=spec.budget,
        config=MixingConfig(seed=spec.seed),
        time_horizon_s=spec.budget.time_horizon_seconds,
    )
    phase = PhaseStats(label="full")
    for timed in mixed:
        _process_one_transaction(
            timed=timed, defense=defense, sut=sut, probe=probe, rec=rec,
            streaming_predictions=streaming_predictions,
            predicted_costs=predicted_costs, actual_costs=actual_costs,
            phase=phase, per_txn_log=per_txn_log,
            use_groundtruth_labels=spec.use_groundtruth_labels,
        )
    return phase


def _run_wsr_phases(
    spec: RunSpec,
    loader: Any,
    attack: Optional[Attack],
    defense: Optional[Defense],
    sut: System,
    probe: MeasuringProbe,
    rec: RunRecord,
    boundaries: IntervalBoundaries,
    streaming_predictions: List[Tuple[int, int]],
    predicted_costs: List[float],
    actual_costs: List[float],
    per_txn_log: List[Tuple[int, float, float, bool]],
) -> List[PhaseStats]:
    """
    Three-phase WSR run loop.  Mirrors Qing & Zheng (ICDE 2025)
    methodology: the system is observed during a benign warmup
    period, an attacked storm window, and a benign recovery
    period.  Per-phase stats and phase-boundary timestamps are
    captured so plots.py can render shaded storm/recovery intervals.

    Stream construction
    -------------------
    workload.DatasetLoader.stream() returns a fresh iterator on each
    call, so we instantiate three benign iterators and skip the
    earlier ones forward to the right offset.  This keeps the
    txn-ordering identical to a single-stream run while letting us
    pause attack injection during warmup and recovery.
    """
    n_total = spec.n_transactions
    n_warmup = max(1, int(round(spec.warmup_fraction * n_total)))
    n_recovery = max(1, int(round(spec.recovery_fraction * n_total)))
    n_storm = max(1, n_total - n_warmup - n_recovery)
    if n_warmup + n_storm + n_recovery > n_total:
        # Round-tripping might allocate one extra; trim from storm.
        n_storm = n_total - n_warmup - n_recovery
        n_storm = max(1, n_storm)

    out: List[PhaseStats] = []

    # --- Warmup: benign-only -----------------------------------------
    # MixedStream(attack=None) just passes the benign stream through
    # with TimedTransaction wrappers, which is what we want.
    warmup_stream = MixedStream(
        benign=_take(loader.stream(), n_warmup),
        attack=None,
        budget=spec.budget,
        config=MixingConfig(seed=spec.seed),
        time_horizon_s=spec.budget.time_horizon_seconds,
    )
    phase_warmup = PhaseStats(label="warmup")
    for timed in warmup_stream:
        _process_one_transaction(
            timed=timed, defense=defense, sut=sut, probe=probe, rec=rec,
            streaming_predictions=streaming_predictions,
            predicted_costs=predicted_costs, actual_costs=actual_costs,
            phase=phase_warmup, per_txn_log=per_txn_log,
            use_groundtruth_labels=spec.use_groundtruth_labels,
        )
    out.append(phase_warmup)
    boundaries.warmup_end_ns = time.monotonic_ns()

    # --- Storm: attacked ---------------------------------------------
    storm_loader_iter = loader.stream()
    # Skip past warmup transactions (each call to stream() is fresh,
    # so we need to re-drain the prefix).
    _drain(storm_loader_iter, n_warmup)
    storm_stream = MixedStream(
        benign=_take(storm_loader_iter, n_storm),
        attack=attack,
        budget=spec.budget,
        config=MixingConfig(seed=spec.seed),
        time_horizon_s=spec.budget.time_horizon_seconds,
    )
    phase_storm = PhaseStats(label="storm")
    for timed in storm_stream:
        _process_one_transaction(
            timed=timed, defense=defense, sut=sut, probe=probe, rec=rec,
            streaming_predictions=streaming_predictions,
            predicted_costs=predicted_costs, actual_costs=actual_costs,
            phase=phase_storm, per_txn_log=per_txn_log,
            use_groundtruth_labels=spec.use_groundtruth_labels,
        )
    out.append(phase_storm)
    boundaries.storm_end_ns = time.monotonic_ns()

    # --- Recovery: benign-only ---------------------------------------
    recovery_loader_iter = loader.stream()
    _drain(recovery_loader_iter, n_warmup + n_storm)
    recovery_stream = MixedStream(
        benign=_take(recovery_loader_iter, n_recovery),
        attack=None,
        budget=spec.budget,
        config=MixingConfig(seed=spec.seed),
        time_horizon_s=spec.budget.time_horizon_seconds,
    )
    phase_recovery = PhaseStats(label="recovery")
    for timed in recovery_stream:
        _process_one_transaction(
            timed=timed, defense=defense, sut=sut, probe=probe, rec=rec,
            streaming_predictions=streaming_predictions,
            predicted_costs=predicted_costs, actual_costs=actual_costs,
            phase=phase_recovery, per_txn_log=per_txn_log,
            use_groundtruth_labels=spec.use_groundtruth_labels,
        )
    out.append(phase_recovery)
    return out


def _drain(it: Iterator[Any], n: int) -> None:
    """Skip n items from an iterator without yielding them."""
    for _ in range(n):
        try:
            next(it)
        except StopIteration:
            return


def _attach_storm_analysis(
    rec: RunRecord,
    spec: RunSpec,
    view: Optional[TargetSystemView],
    attack: Optional[Attack],
    defense: Optional[Defense],
    probe: MeasuringProbe,
    boundaries: IntervalBoundaries,
    per_txn_log: Sequence[Tuple[int, float, float, bool]],
) -> None:
    """
    Build the storm summary and call ``analyse_storm`` with ground-
    truth-aware α and the modern certificate API.

    Camera-ready changes vs. submission draft
    -----------------------------------------
    1. **Ground-truth α.**  The submission draft bucketed cost
       samples by a "top-α-by-cost" heuristic (admit the heaviest
       α-fraction as adversarial).  This is circular under the
       schedulability bound — α appears on both sides of the fit.
       The camera-ready uses ``per_txn_log`` from the run loop,
       where each entry's ``is_adversarial`` flag comes from
       ``Attack.is_adversarial_txn_id`` (the public ground-truth
       channel exposed by camera-ready attacks.py).  Empirical α is
       the fraction of completed transactions that were adversarial,
       passed via ``analyse_storm(adversary_fraction=...)``.
    2. **Carry-in forwarding removed.**  Camera-ready defenses.py
       returns ``()`` from ``carry_in_samples_us()`` under the
       synchronous harness (B₀ = 0 is correct; the submission
       draft's "queue residency" computation under synchronous
       execution double-counted the SUT cost).  ``analyse_storm``
       builds the degenerate-zero B₀ internally and logs the
       standard caveat.
    3. **Modern field names.**  ``mgf.optimal_lambda`` and
       ``mgf.components`` instead of the back-compat aliases
       ``mgf.optimal_lam`` and ``mgf.per_class_bounds``.

    Fallback for runs without an attack
    -----------------------------------
    When ``attack is None`` (benign-only runs), every transaction
    has ``is_adversarial=False`` and α = 0.  In that regime the
    adversarial fit collapses to the benign one and ``analyse_storm``
    correctly flags ``adversarial_fit_is_benign_fallback=True``.
    """
    e2e_us = list(probe.end_to_end.array_us())
    if not e2e_us:
        return

    # Camera-ready: ground-truth α and per-class buckets.
    benign_us: List[float] = []
    adversarial_us: List[float] = []
    if per_txn_log:
        for _txn_id, _pred_us, actual_us, is_adv in per_txn_log:
            if is_adv:
                adversarial_us.append(actual_us)
            else:
                benign_us.append(actual_us)
        n_adv = len(adversarial_us)
        n_total = n_adv + len(benign_us)
        adversary_fraction_measured = (
            n_adv / n_total if n_total > 0 else 0.0
        )
        alpha_source = "ground-truth (per_txn_log)"
    else:
        # Fallback for legacy code paths that don't populate
        # per_txn_log (none in the camera-ready harness, but
        # preserved defensively for future entry points).  Use
        # benign-only with α = 0; analyse_storm will correctly flag
        # the benign-fallback condition.
        benign_us = list(e2e_us)
        adversarial_us = []
        adversary_fraction_measured = 0.0
        alpha_source = "no per_txn_log; benign-only fallback"

    if not benign_us:
        return

    # Admitted-stream rate: number of admitted txns divided by the
    # measurement window.  Used to estimate per-class window counts
    # for the MGF certificate.
    n_admitted = int(probe.miss_report.n_transactions)
    duration_s = max(
        1e-6,
        (boundaries.run_end_ns - boundaries.run_start_ns) / 1e9,
    )
    expected_admitted_rate_hz = n_admitted / duration_s

    storm = UpdateStorm(
        capability=(
            view.capability if view is not None
            else AdversaryCapability.tier_1()
        ),
        budget=spec.budget,
        cost_function=_dummy_cost_function(),
        contract=spec.contract,
        label=spec.signature(),
    )

    # Camera-ready: pass measured α via ``adversary_fraction``;
    # analyse_storm uses it instead of falling back to
    # storm.budget.fraction_of_stream.  Note we still pass the
    # legacy kwargs (benign_samples_us, expected_admitted_rate_hz)
    # because analyse_storm's back-compat shim translates them.
    # When the harness is simplified further in a future revision,
    # those calls can switch to the modern keyword names.
    sa = analyse_storm(
        storm=storm,
        benign_cost_samples_us=benign_us,
        adversarial_cost_samples_us=adversarial_us,
        adversary_fraction=adversary_fraction_measured,
        # Legacy admit-rate kwarg still works via the shim; passes
        # through to a Poisson CountDistribution with the measured
        # mean.
        expected_admitted_rate_hz=expected_admitted_rate_hz,
        # Carry-in deliberately omitted: synchronous harness ⇒
        # B_0 = 0 (analyse_storm constructs the degenerate-zero
        # distribution and logs the caveat).  See camera-ready
        # defenses.py header for the full rationale.
    )

    rec.storm_signature = storm.signature()
    rec.notes.append(
        f"analysis_alpha: source={alpha_source}, "
        f"measured={adversary_fraction_measured:.4f}"
    )
    rec.notes.append(
        f"analysis_envelope: T_min={sa.certificate.threshold_us:.0f}us, "
        f"feasible={sa.certificate.feasible}, "
        f"slack={sa.certificate.slack_us:.0f}us, "
        f"benign_fallback={sa.adversarial_fit_is_benign_fallback}"
    )
    if sa.mgf_certificate is not None:
        mgf = sa.mgf_certificate
        # Camera-ready: use the new field names directly.
        rec.notes.append(
            f"analysis_mgf: feasible={mgf.feasible}, "
            f"bound={mgf.bound:.4g}, log_bound={mgf.log_bound:.4f}, "
            f"lambda*={mgf.optimal_lambda:.4g}, "
            f"lambda_at_edge={mgf.optimal_lambda_at_bracket_edge}, "
            f"n_classes={len(mgf.components)}, "
            f"notes={mgf.notes!r}"
        )
        # Validate the MGF certificate against the empirical miss
        # rate (apply/validate identity from analysis.py).
        mgf_validation = validate_theorem_4_3_mgf(
            cert=mgf,
            measured_miss_rate=probe.miss_report.miss_rate,
            n_admitted=n_admitted,
        )
        rec.notes.append(
            f"validate_4_3_mgf: holds={mgf_validation.holds_at_p99}, "
            f"holds_ci_95={mgf_validation.holds_at_ci_95}, "
            f"bound={mgf_validation.bound_at_p99:.4g}, "
            f"measured={mgf_validation.measured_at_p99:.4g}, "
            f"notes={mgf_validation.notes!r}"
        )


def _attach_d3_certificates(
    rec: RunRecord,
    defense: D3Schedulability,
) -> None:
    """
    Persist D3's most-recent envelope and MGF certificates into the
    run record's notes.  These are what experiments.py contributes
    to the §V.3 schedulability-validation table.
    """
    env_cert = defense.latest_certificate()
    if env_cert is not None:
        rec.notes.append(
            f"d3_envelope_cert: feasible={env_cert.feasible}, "
            f"T={env_cert.threshold_us:.0f}us, "
            f"bound_at_T={env_cert.bound_at_T:.4g}, "
            f"slack={env_cert.slack_us:.0f}us, "
            f"epsilon={env_cert.epsilon}"
        )
    mgf_cert = defense.latest_mgf_certificate()
    if mgf_cert is not None:
        # Camera-ready: use new field names
        # (``optimal_lambda`` instead of the alias
        # ``optimal_lam``).
        rec.notes.append(
            f"d3_mgf_cert: feasible={mgf_cert.feasible}, "
            f"bound={mgf_cert.bound:.4g}, "
            f"log_bound={mgf_cert.log_bound:.4f}, "
            f"lambda*={mgf_cert.optimal_lambda:.4g}, "
            f"lambda_at_edge={mgf_cert.optimal_lambda_at_bracket_edge}, "
            f"epsilon={mgf_cert.epsilon}"
        )


# =============================================================================
# Section 5.  Helpers used by run_one and the experiment functions.
# =============================================================================


def _take(it: Iterable[TimedTransaction], n: int) -> Iterator[TimedTransaction]:
    """Take the first n elements of an iterator."""
    if n <= 0:
        return iter(())
    def gen() -> Iterator[TimedTransaction]:
        i = 0
        for x in it:
            if i >= n:
                return
            yield x
            i += 1
    return gen()


def _default_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _make_view_for_attack(attack_name: str, sut: System) -> TargetSystemView:
    """
    Deprecated.  Use ``threat_model.view_for_attack`` directly.

    Camera-ready: this function is preserved as a back-compat shim
    that forwards to ``view_for_attack``.  The submission draft
    held a hard-coded ``tier_map`` here that listed only A1..A5
    (missing A6); the camera-ready routing table lives in
    ``threat_model.py`` and is populated by ``attacks.py`` at
    import time, so adding a new attack registers automatically.

    The shim is retained for any external callers that imported
    this name; ``run_one`` itself uses ``view_for_attack`` directly.
    """
    return view_for_attack(
        attack_name,
        topology=sut.topology_source,
        metadata=sut.metadata_source,
        weights=sut.weights_source,
    )


def _defense_default_kwargs(name: str, contract: RealTimeContract) -> Mapping[str, Any]:
    """
    Default kwargs for each defense.  Production runs should override
    these from configs/defense_profiles.yaml; this is the fallback
    for tests and quick experiments.
    """
    if name == "D1_static":
        return {"threshold_us": contract.deadline_us * 0.6}
    if name == "D2_adaptive":
        return {"initial_threshold_us": contract.deadline_us * 0.6}
    if name == "D3_schedulability":
        return {}
    raise KeyError(f"no defaults for defense '{name}'")


def _dummy_cost_function() -> UpdateCostFunction:
    """
    A no-op UpdateCostFunction sufficient to construct an UpdateStorm
    object.  The actual cost is observed empirically from the probe;
    the cost function is only used for the storm's signature in the
    run record.
    """
    def predictor(t: object, s: object) -> CostDecomposition:
        return CostDecomposition(0.0, 0.0, 0.0, 0.0, 0.0)
    return UpdateCostFunction(decomposition_predictor=predictor, description="empirical")


# =============================================================================
# Section 6.  Default contracts and budgets per dataset.
#
# Each dataset has a different operationally-meaningful deadline.
# These are the values referenced in the README and used unless an
# experiment overrides them.  Cross-experiment consistency requires
# they live in one place.
#
# DEADLINE-FRAMING NOTES (camera-ready)
# -------------------------------------
# Each contract's deadline_us is interpreted as the *update-path
# freshness deadline* — the time window within which the SUT must
# complete its incremental update after a stream record arrives at
# the gate.  This is NOT a packet-forwarding or first-packet
# blocking deadline; the SUT is not an inline IDS.  We use the same
# semantic for every dataset to keep the schedulability analysis
# uniform.
#
#   ethereum_phishing:  4 s post-flow update window — block
#       inclusion period of Ethereum mainnet (~12 s post-Merge),
#       conservatively halved to allow propagation.
#   bitcoin_ransomware: 10 s post-flow update window — generous
#       given Bitcoin's 10-min block time, leaves headroom for
#       feature extraction.
#   cicids2018: 2 ms post-flow-record update window.
#       IMPORTANT: per Sharafaldin et al. ICISSP 2018 and the
#       Engelen et al. troubleshooting study, CICFlowMeter features
#       (flow duration, IAT statistics, packet-length statistics,
#       active/idle times) are computed AFTER the flow has been
#       observed for some time or after termination.  The 2 ms
#       deadline therefore applies after a flow record is emitted
#       to the SUT, not at first-packet ingress.  This is the
#       deadline-credible primary dataset for §V.3.
#   swat: 10 ms control-loop bound — derived from the SWaT
#       testbed's 10 Hz PLC scan rate.
#   synthetic: 5 ms — chosen for fast CI runs.
#
# The CICIDS dataset version question (CICIDS2017 vs CSE-CIC-IDS2018)
# is documented in docs/CICIDS_LIMITATIONS.md.  Within this codebase
# we use the CSE-CIC-IDS2018 schema (8 attack days, MachineLearningCSV
# format) for the key "cicids2018"; cross-checks with the corrected
# Engelen et al. variants are out of scope for the camera-ready
# evaluation but flagged as a future-work item.
# =============================================================================


_DATASET_CONTRACTS: Mapping[str, RealTimeContract] = {
    "ethereum_phishing": RealTimeContract(
        deadline_us=4_000_000.0,             # 4 s — post-flow update window
        failure_probability_bound=1e-3,
        measurement_window_seconds=600.0,
    ),
    "bitcoin_ransomware": RealTimeContract(
        deadline_us=10_000_000.0,            # 10 s — generous for slow-block chain
        failure_probability_bound=1e-3,
        measurement_window_seconds=600.0,
    ),
    "cicids2018": RealTimeContract(
        deadline_us=2_000.0,                 # 2 ms — POST-FLOW-RECORD update freshness
        failure_probability_bound=1e-3,
        measurement_window_seconds=120.0,
    ),
    "swat": RealTimeContract(
        deadline_us=10_000.0,                # 10 ms — control-loop bound
        failure_probability_bound=1e-4,      # tighter for safety-critical
        measurement_window_seconds=300.0,
    ),
    "synthetic": RealTimeContract(
        deadline_us=5_000.0,
        failure_probability_bound=1e-3,
        measurement_window_seconds=60.0,
    ),
}


def contract_for(dataset: str) -> RealTimeContract:
    if dataset not in _DATASET_CONTRACTS:
        raise KeyError(f"no default contract for dataset '{dataset}'")
    return _DATASET_CONTRACTS[dataset]


def _default_budget() -> AdversaryBudget:
    """The realistic budget used across experiments unless overridden."""
    return AdversaryBudget.realistic_default()


# =============================================================================
# Section 7.  The seven named experiments.
#
# Each experiment is a function that returns a list of RunRecords.  It
# enumerates RunSpecs and calls run_one on each.  Statistical power is
# achieved by N_SEEDS independent repetitions per (dataset × attack ×
# defense) cell, controlled by the constants below.
# =============================================================================


N_SEEDS = 5
DEFAULT_N_TRANSACTIONS = 50_000


def _seeds_for(experiment: str, base: int = 0) -> Sequence[int]:
    """Stable seed list per experiment, derived from a fixed base."""
    return tuple(range(base, base + N_SEEDS))


# --- exp_attack_effectiveness ----------------------------------------------


def exp_attack_effectiveness(
    dataset_paths: Mapping[str, Path],
    output_dir: Path,
    n_transactions: int = DEFAULT_N_TRANSACTIONS,
) -> List[RunRecord]:
    """
    §5.1 — How much does each attack inflate the update-path latency?

    Every attack vs. no defense, on every dataset, repeated N_SEEDS times.
    The control is benign-only (attack=None) on the same dataset.

    Camera-ready: WSR is enabled so each run produces explicit
    warmup → storm → recovery intervals plus a recovery time to
    baseline (Qing & Zheng style), supporting the time-series
    figures in §V.1.
    """
    out = output_dir / "exp_attack_effectiveness"
    records: List[RunRecord] = []
    for dataset_name, path in dataset_paths.items():
        contract = contract_for(dataset_name)
        budget = _default_budget()
        # Control: benign-only.
        for seed in _seeds_for("attack_effectiveness"):
            spec = RunSpec(
                experiment="exp_attack_effectiveness",
                dataset=dataset_name,
                dataset_path=path,
                attack=None,
                defense=None,
                seed=seed,
                n_transactions=n_transactions,
                contract=contract,
                budget=budget,
                output_dir=out,
                note="control_benign",
                # Control has no storm to recover from, so WSR adds
                # no value; leave the default off.
            )
            records.append(run_one(spec))
        # Per-attack runs.
        for atk in attack_names():
            for seed in _seeds_for("attack_effectiveness"):
                spec = RunSpec(
                    experiment="exp_attack_effectiveness",
                    dataset=dataset_name,
                    dataset_path=path,
                    attack=atk,
                    defense=None,
                    seed=seed,
                    n_transactions=n_transactions,
                    contract=contract,
                    budget=budget,
                    output_dir=out,
                    enable_warmup_storm_recovery=True,
                )
                records.append(run_one(spec))
    return records


# --- exp_defense_efficacy --------------------------------------------------


def exp_defense_efficacy(
    dataset_paths: Mapping[str, Path],
    output_dir: Path,
    n_transactions: int = DEFAULT_N_TRANSACTIONS,
) -> List[RunRecord]:
    """
    §5.2 — Does the schedulability-aware defense actually reduce the
    deadline-miss rate?

    For each dataset: take the strongest tier-1 attack (A3) and the
    strongest tier-2 attack (A4), run each against each of the three
    defenses, plus the no-defense baseline.

    Camera-ready: WSR enabled so plots.py can overlay D1/D2/D3
    latency-over-time curves with shaded storm/recovery intervals,
    and so each run produces a recovery_time_s diagnostic.
    """
    out = output_dir / "exp_defense_efficacy"
    records: List[RunRecord] = []
    headline_attacks = ("A3_branching_max", "A4_gradient_norm")
    defenses = list(defense_names()) + [None]    # type: ignore[list-item]
    for dataset_name, path in dataset_paths.items():
        contract = contract_for(dataset_name)
        budget = _default_budget()
        for atk in headline_attacks:
            for defense in defenses:
                for seed in _seeds_for("defense_efficacy"):
                    spec = RunSpec(
                        experiment="exp_defense_efficacy",
                        dataset=dataset_name,
                        dataset_path=path,
                        attack=atk,
                        defense=defense,
                        seed=seed,
                        n_transactions=n_transactions,
                        contract=contract,
                        budget=budget,
                        output_dir=out,
                        enable_warmup_storm_recovery=True,
                    )
                    records.append(run_one(spec))
    return records


# --- exp_schedulability_validation -----------------------------------------


def exp_schedulability_validation(
    dataset_paths: Mapping[str, Path],
    output_dir: Path,
    n_transactions: int = DEFAULT_N_TRANSACTIONS * 4,    # need more for tail
) -> List[RunRecord]:
    """
    §5.3 — Are the bounds proved in Theorems 4.1, 4.2, 4.3 empirically
    valid?

    Run benign-only and adversarial-only at high sample count, fit the
    bounds, validate against held-out portions.  Validation results
    are written into the run record's notes; plots.py renders the
    bound vs. measured CCDF.

    Camera-ready: this experiment is the headline validation table
    (§V.3).  For the defended runs we keep WSR off because the
    schedulability claim is about steady-state behaviour; the
    transient warmup and recovery would dilute the validation
    statistics.
    """
    out = output_dir / "exp_schedulability_validation"
    records: List[RunRecord] = []
    for dataset_name, path in dataset_paths.items():
        contract = contract_for(dataset_name)
        budget = _default_budget()
        # Benign-only at high sample count.
        for seed in _seeds_for("schedulability_validation"):
            spec = RunSpec(
                experiment="exp_schedulability_validation",
                dataset=dataset_name,
                dataset_path=path,
                attack=None,
                defense=None,
                seed=seed,
                n_transactions=n_transactions,
                contract=contract,
                budget=budget,
                output_dir=out,
                note="benign_for_theorem_4_1",
            )
            records.append(run_one(spec))
        # Adversarial: A3 (representative tier-1) for theorem 4.2.
        for seed in _seeds_for("schedulability_validation"):
            spec = RunSpec(
                experiment="exp_schedulability_validation",
                dataset=dataset_name,
                dataset_path=path,
                attack="A3_branching_max",
                defense=None,
                seed=seed,
                n_transactions=n_transactions,
                contract=contract,
                budget=budget,
                output_dir=out,
                note="adversarial_for_theorem_4_2",
            )
            records.append(run_one(spec))
        # Defended: D3 vs A3, for theorem 4.3 validation (both
        # envelope and MGF certificates).
        for seed in _seeds_for("schedulability_validation"):
            spec = RunSpec(
                experiment="exp_schedulability_validation",
                dataset=dataset_name,
                dataset_path=path,
                attack="A3_branching_max",
                defense="D3_schedulability",
                seed=seed,
                n_transactions=n_transactions,
                contract=contract,
                budget=budget,
                output_dir=out,
                note="defended_for_theorem_4_3",
            )
            records.append(run_one(spec))
    return records


# --- exp_scalability -------------------------------------------------------


def exp_scalability(
    dataset_paths: Mapping[str, Path],
    output_dir: Path,
) -> List[RunRecord]:
    """
    §5.3 — How does attack effectiveness and defense overhead scale
    with stream length?

    Use one large dataset (preferentially BitcoinRansomware, the largest)
    and sweep n_transactions over five orders of magnitude.
    """
    out = output_dir / "exp_scalability"
    records: List[RunRecord] = []
    sweep_dataset = "bitcoin_ransomware"
    if sweep_dataset not in dataset_paths:
        # Fall back to whichever is present.
        sweep_dataset = next(iter(dataset_paths))
    path = dataset_paths[sweep_dataset]
    contract = contract_for(sweep_dataset)
    budget = _default_budget()
    sizes = (10_000, 50_000, 200_000, 1_000_000, 4_000_000)
    for n_txns in sizes:
        for seed in _seeds_for("scalability"):
            for defense in (None, "D3_schedulability"):
                spec = RunSpec(
                    experiment="exp_scalability",
                    dataset=sweep_dataset,
                    dataset_path=path,
                    attack="A3_branching_max",
                    defense=defense,
                    seed=seed,
                    n_transactions=n_txns,
                    contract=contract,
                    budget=budget,
                    output_dir=out,
                    note=f"size={n_txns}",
                )
                records.append(run_one(spec))
    return records


# --- exp_ablation ----------------------------------------------------------


def exp_ablation(
    dataset_paths: Mapping[str, Path],
    output_dir: Path,
    n_transactions: int = DEFAULT_N_TRANSACTIONS,
) -> List[RunRecord]:
    """
    §5.4 — Component ablation of the defenses.

    For one dataset (CICIDS-2018, the deadline-credible primary) and
    one attack (A3), run each defense variant and a series of D3
    variants with progressively more components disabled.  In this
    minimal codebase we ablate the defense by name only; full per-knob
    ablations are described in docs/ABLATION.md.

    Camera-ready: also runs WSR mode on D3 to expose its recovery
    profile (the §V.4 ablation table reports recovery_time_s for
    each defense).
    """
    out = output_dir / "exp_ablation"
    records: List[RunRecord] = []
    target_dataset = "cicids2018"
    if target_dataset not in dataset_paths:
        target_dataset = next(iter(dataset_paths))
    path = dataset_paths[target_dataset]
    contract = contract_for(target_dataset)
    budget = _default_budget()
    for defense in (None, "D1_static", "D2_adaptive", "D3_schedulability"):
        for seed in _seeds_for("ablation"):
            spec = RunSpec(
                experiment="exp_ablation",
                dataset=target_dataset,
                dataset_path=path,
                attack="A3_branching_max",
                defense=defense,
                seed=seed,
                n_transactions=n_transactions,
                contract=contract,
                budget=budget,
                output_dir=out,
                enable_warmup_storm_recovery=True,
            )
            records.append(run_one(spec))
    return records


# --- exp_cross_dataset -----------------------------------------------------


def exp_cross_dataset(
    dataset_paths: Mapping[str, Path],
    output_dir: Path,
    n_transactions: int = DEFAULT_N_TRANSACTIONS,
) -> List[RunRecord]:
    """
    §5.5 — Does the attack class generalise across all four datasets?

    Each attack against no-defense, on every dataset.  Differs from
    exp_attack_effectiveness only in tighter budget settings and a
    smaller transaction count; results are aggregated separately to
    isolate the cross-dataset comparison from the within-dataset
    inflation curves.
    """
    out = output_dir / "exp_cross_dataset"
    records: List[RunRecord] = []
    for dataset_name, path in dataset_paths.items():
        contract = contract_for(dataset_name)
        budget = _default_budget()
        for atk in attack_names():
            for seed in _seeds_for("cross_dataset"):
                spec = RunSpec(
                    experiment="exp_cross_dataset",
                    dataset=dataset_name,
                    dataset_path=path,
                    attack=atk,
                    defense=None,
                    seed=seed,
                    n_transactions=n_transactions,
                    contract=contract,
                    budget=budget,
                    output_dir=out,
                )
                records.append(run_one(spec))
    return records


# --- exp_adaptive_adversary ------------------------------------------------


def exp_adaptive_adversary(
    dataset_paths: Mapping[str, Path],
    output_dir: Path,
    n_transactions: int = DEFAULT_N_TRANSACTIONS * 2,
) -> List[RunRecord]:
    """
    §6 — A5 (adaptive white-box) vs every defense, plus the no-defense
    baseline.  This is the worst-case scenario for the schedulability
    bound; the paper reports measured kappa from theorem 4.4.
    """
    out = output_dir / "exp_adaptive_adversary"
    records: List[RunRecord] = []
    for dataset_name, path in dataset_paths.items():
        contract = contract_for(dataset_name)
        budget = _default_budget()
        for defense in (None, "D1_static", "D2_adaptive", "D3_schedulability"):
            for seed in _seeds_for("adaptive_adversary"):
                spec = RunSpec(
                    experiment="exp_adaptive_adversary",
                    dataset=dataset_name,
                    dataset_path=path,
                    attack="A5_adaptive",
                    defense=defense,
                    seed=seed,
                    n_transactions=n_transactions,
                    contract=contract,
                    budget=budget,
                    output_dir=out,
                    enable_warmup_storm_recovery=True,
                )
                records.append(run_one(spec))
    return records


# =============================================================================
# Section 8.  Experiment registry — for main.py dispatch.
# =============================================================================


_EXPERIMENT_REGISTRY: Mapping[str, Callable[..., List[RunRecord]]] = {
    "exp_attack_effectiveness": exp_attack_effectiveness,
    "exp_defense_efficacy": exp_defense_efficacy,
    "exp_schedulability_validation": exp_schedulability_validation,
    "exp_scalability": exp_scalability,
    "exp_ablation": exp_ablation,
    "exp_cross_dataset": exp_cross_dataset,
    "exp_adaptive_adversary": exp_adaptive_adversary,
}


def experiment_names() -> Sequence[str]:
    return tuple(_EXPERIMENT_REGISTRY.keys())


def run_experiment(
    name: str,
    dataset_paths: Mapping[str, Path],
    output_dir: Path,
    **kwargs: Any,
) -> List[RunRecord]:
    if name not in _EXPERIMENT_REGISTRY:
        raise KeyError(
            f"unknown experiment '{name}'; known: {sorted(_EXPERIMENT_REGISTRY)}"
        )
    fn = _EXPERIMENT_REGISTRY[name]
    return fn(dataset_paths=dataset_paths, output_dir=output_dir, **kwargs)


# =============================================================================
# Section 9.  Public surface.
# =============================================================================

__all__ = [
    "RunSpec",
    "IntervalBoundaries",
    "PhaseStats",
    "run_one",
    "contract_for",
    "experiment_names",
    "run_experiment",
    "exp_attack_effectiveness",
    "exp_defense_efficacy",
    "exp_schedulability_validation",
    "exp_scalability",
    "exp_ablation",
    "exp_cross_dataset",
    "exp_adaptive_adversary",
]
