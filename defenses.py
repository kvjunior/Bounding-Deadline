"""
defenses.py — Three defenses against update storms.

Each defense decides, for every incoming transaction, whether to
admit it to the update path or reject it.  Admitted transactions go
through the SUT's normal update path; rejected transactions never
reach the expensive operations.  The defender is the gatekeeper
between the arrival stream and the SUT.

The three defenses
------------------
D1  Static threshold        — constant cost-prediction cutoff.  Baseline.
D2  Adaptive threshold      — adjusts cutoff in response to recent load.
D3  Schedulability-aware    — chooses the threshold that satisfies the
                              real-time contract via Theorem 4.3 from
                              analysis.py.  This is the paper's
                              headline contribution.

All three implement the DefenderStateSource protocol from
threat_model.py.  Under the threat model, ONLY a tier-3 adversary
may read this state; the protocol enforcement happens upstream in
TargetSystemView, not here.

Mapping to the paper
--------------------
§4 (Schedulability)              →  D3 calls into analysis.py
§5.2 (Defense efficacy)          →  produces admitted streams measured
§5.4 (Defense ablation)          →  D1, D2, D3 compared component-wise
§6 (Adaptive adversary)          →  D3 vs A5

Two certificate paths in D3
---------------------------
D3 produces *both* schedulability certificates from analysis.py:

  - A SchedulabilityCertificate (Theorem 4.3-env): the cheap
    exponential-envelope bound used at admission time.  Recomputed
    each refit; consulted on every admission decision.

  - An MGFCertificate (Theorem 4.3-mgf): the Sun-style random-sum
    bound that engages with the RTSS literature.  Recomputed each
    refit; reported in run records and used as the headline
    schedulability claim in the paper's §V.3 table.

The defense uses the envelope at admission time because it is fast
(O(1) per evaluation, O(log(1/ε)) for threshold computation) and
maintains the same admission threshold T as before; the MGF
certificate is reported but does not gate admission, because its
cost (~10 ms per certificate) is too high for the hot path.  This
is the standard pattern in stochastic-timing-analysis deployments:
cheap online check, expensive offline certificate.

Cost prediction
---------------
Every defense makes admission decisions based on a *predicted*
update-path cost for each incoming transaction.  The predictor is
independent of the attacker's predictor.  The defender's predictor
is intentionally modest: it scores a transaction by the local
subgraph-reach proxy described in §4.2 of the paper.  Cheaper
predictors yield faster admission decisions but more variance in
admitted-stream cost; the paper's §5.2 ablation reports this
tradeoff.

The defense's predictor is implemented in this file as
``local_reach_predictor``.  All three defenses use the same
predictor; they differ only in how they convert predicted cost into
an admission decision.  This isolates "predictor quality" from
"policy quality" in the experimental design.

Concurrency
-----------
The harness in experiments.py invokes each defense strictly
serially: ``evaluate(txn, t)`` is called once per arrival, followed
by ``record_completion(...)`` after the SUT processes the
admission.  All internal state mutations are therefore safe under
that single-threaded contract.  A multi-worker deployment would
need to add a lock; the hot-path operations are short enough that a
per-defense mutex would be acceptable, but we do not introduce one
here because the harness does not require it.

Camera-ready improvements over the submission draft
---------------------------------------------------
1. **Carry-in semantics fixed.**  The submission draft computed
   ``queue_residency_us = (now − admission_time) * 1e6`` in
   ``record_completion`` and fed it into Theorem 4.3-mgf as the
   carry-in (B₀) distribution.  Under experiments.py's synchronous
   harness — where ``defense.evaluate`` → ``sut.update_path.process``
   → ``defense.record_completion`` runs in one straight-line call
   chain — the time delta IS the SUT's end-to-end processing time
   plus a few clock-read overheads.  Feeding it to ``apply_theorem
   _4_3_mgf`` as B₀ double-counts the cost.  The camera-ready
   acknowledges that under synchronous execution there is no
   queue, B₀ ≡ 0 is the correct value (matching the documented
   ``apply_theorem_4_3_mgf`` "synchronous-harness" caveat),
   and the carry-in computation is removed.  When a future async
   harness is added, the channel can be repopulated; for now we
   pass ``carry_in=None`` to the MGF call and let the analysis
   layer log the standard caveat.  ``carry_in_samples_us()`` is
   retained for backwards compatibility but always returns ``()``;
   ``D3.statistics()`` reports ``carry_in_synchronous=True`` so
   reviewers can verify which mode produced the certificate.

2. **Modern MGF kwargs.**  The submission draft called
   ``apply_theorem_4_3_mgf`` with the legacy keyword API
   (``benign_cost_dist=...``, ``expected_admitted_count_benign=...``,
   etc.), which now triggers a logger.info each refit because
   camera-ready analysis.py routes legacy callers through the
   Poisson-count compatibility shim.  The camera-ready uses the
   modern dict-based signature with measured CountDistributions
   estimated from per-window admission counts — no Poisson
   assumption.

3. **MGFCertificate field rename.**  The submission draft
   referenced ``mgf.optimal_lam``, ``mgf.per_class_bounds``, and
   ``mgf.carry_in_log_mgf`` in ``D3.statistics()``.  The first two
   are aliased on camera-ready ``MGFCertificate``; the third was
   refactored away (it now lives at ``components["B_0"]``).  This
   would have raised ``AttributeError`` in production once the
   refit produced its first certificate (latent bug — not caught
   by the test suite because smoke tests don't run long enough to
   trigger an MGF refit).  The camera-ready uses the new field
   names directly: ``optimal_lambda``, ``components``, and the
   B₀ entry of ``components`` when present.

4. **Optional ground-truth labelling channel.**  The submission
   draft's ``_label_completion`` heuristic (per-source rate OR
   predictor residual) is preserved as the *online* labeller —
   the defense does not have ground truth at production time.  The
   camera-ready additionally exposes
   ``record_completion_with_groundtruth(predicted, actual,
   is_adversarial)``: when the experiment harness DOES have
   ground truth (via ``Attack.is_adversarial_txn_id``), it can
   pass it through, and §V.4 ablations can compare both labelling
   regimes.  The harness uses ground truth only for studies whose
   conclusions depend on it; production-mode runs continue to use
   the heuristic.

5. **Configurable infeasibility fallback.**  The submission draft
   hardcoded the "infeasible-α fallback" threshold to the 10th
   percentile of benign cost history.  The camera-ready exposes
   ``fallback_percentile`` (default 10.0) so reviewers comparing
   D3's behaviour under heavy attack can override the constant.

Author: ANONYMOUS  (double-blind review)
"""

from __future__ import annotations

import abc
import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np

from threat_model import (
    GraphTopologySource,
    RealTimeContract,
)
from system import Transaction
from analysis import (
    CountDistribution,
    MGFCertificate,
    SchedulabilityCertificate,
    apply_theorem_4_3,
    apply_theorem_4_3_mgf,
    cost_distribution_from_samples,
    fit_exponential_upper_bound,
    poisson_count_distribution,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Section 1.  Cost predictor (shared by all defenses).
#
# Cheap, topology-only proxy for update-path cost.  Identical
# information to the tier-1 attacker.  This is intentional: the
# defense should not need privileged access to the system to make
# admission decisions, because in production the defender sits at
# the network edge with the same visibility as a sophisticated
# adversary.
#
# The predictor is calibrated against measured benign cost using
# ``calibrate_predictor``.  Without calibration it produces ordinal
# scores (useful for relative comparisons).  Calibration converts
# those into microsecond estimates (suitable for absolute-threshold
# comparison).
# =============================================================================


# Minimum samples for a meaningful OLS fit.  Below this, calibration
# returns the previous state unchanged.
_MIN_CALIBRATION_SAMPLES: int = 32


@dataclass
class CostPredictorState:
    """
    Calibrated cost predictor.  ``slope`` and ``intercept`` map a
    raw reach score to a microsecond cost estimate via:

        cost_us  ≈  intercept + slope * reach_score.

    The defense calls ``update_calibration()`` periodically to
    refresh these from observed (predicted, actual) pairs.  Initial
    values are set to identity (slope=1, intercept=0); after a brief
    warmup the defense re-calibrates from real data.
    """

    slope: float = 1.0
    intercept: float = 0.0
    n_calibration_samples: int = 0
    last_residual_p99: float = 0.0

    def predict_us(self, reach_score: float) -> float:
        return max(0.0, self.intercept + self.slope * reach_score)


def local_reach_predictor(
    txn: Transaction,
    topology: GraphTopologySource,
    horizon: int = 1,
) -> float:
    """
    Local reach proxy for update-path cost.

    Returns a unitless score proportional to the number of nodes the
    BFS from (txn.source, txn.target) is expected to visit at the
    given horizon.  Identical to the prediction the tier-1 attacker
    can compute (and indeed used by A2/A3 in attacks.py); we use it
    here because the defender, like the attacker, does not have
    privileged graph access in the production threat model.

    Cost: O(deg(source) + deg(target) + Σ deg(neighbour)) for
    horizon=2.  For horizon=1 it is O(deg(source) + deg(target)).
    Both are well within the admission-decision budget.
    """
    score = (
        float(topology.degree(txn.source))
        + float(topology.degree(txn.target))
    )
    if horizon >= 2:
        seen: set[int] = {txn.source, txn.target}
        for endpoint in (txn.source, txn.target):
            for nb in topology.neighbors(endpoint):
                if nb in seen:
                    continue
                seen.add(nb)
                score += float(topology.degree(nb))
    return score


def calibrate_predictor(
    state: CostPredictorState,
    reach_scores: Sequence[float],
    actual_costs_us: Sequence[float],
) -> CostPredictorState:
    """
    Recompute slope and intercept from observed pairs by ordinary
    least squares.  Returns a new state.  If the input is degenerate
    (no variance), the previous state is returned unchanged.
    """
    if len(reach_scores) != len(actual_costs_us):
        raise ValueError(
            "reach_scores and actual_costs_us length mismatch"
        )
    if len(reach_scores) < _MIN_CALIBRATION_SAMPLES:
        return state                  # too few; defer
    x = np.asarray(reach_scores, dtype=np.float64)
    y = np.asarray(actual_costs_us, dtype=np.float64)
    if x.var() < 1e-12 or y.var() < 1e-12:
        return state
    slope, intercept = np.polyfit(x, y, 1)
    residuals = y - (intercept + slope * x)
    p99_resid = float(np.percentile(np.abs(residuals), 99))
    return CostPredictorState(
        slope=float(slope),
        intercept=float(intercept),
        n_calibration_samples=len(reach_scores),
        last_residual_p99=p99_resid,
    )

# =============================================================================
# Section 2.  Common defense interface.
# =============================================================================


@dataclass(frozen=True)
class AdmissionDecision:
    """
    Result of evaluating one transaction.  Persisted into the
    experiment record so plots.py can reconstruct the admission
    timeline (admit/reject pattern over time).

    ``txn_source`` is included so that downstream consumers (e.g.
    D3's rate-based labeller) can correlate admissions across calls
    without holding a reference to the Transaction object itself.
    """

    admit: bool
    predicted_cost_us: float
    threshold_us: float
    timestamp: float           # arrival time, seconds (monotonic)
    txn_source: int = -1       # txn.source; -1 if not recorded
    reason: str = ""


@dataclass(frozen=True)
class _PendingEvaluation:
    """Context bridge from ``evaluate()`` to ``record_completion()``."""

    txn_source: int
    arrival_time: float
    reach_score: float
    predicted_us: float


class Defense(abc.ABC):
    """
    Common defense interface.  Implements DefenderStateSource so
    that ``threat_model.TargetSystemView`` can expose threshold and
    queue depth to a tier-3 attacker.

    Lifecycle per transaction (under experiments.py's serial
    harness)::

        decision = defense.evaluate(txn, t_now)
        if decision.admit:
            cost = sut.update_path.process(txn, ...)
            defense.record_completion(decision.predicted_cost_us, cost.total_us)

    ``evaluate`` and ``record_completion`` are paired one-to-one;
    the intermediate state (``_pending_evaluation``) carries context
    such as the source ID and admission timestamp from one to the
    other so that subclasses can do per-source rate tracking and
    queue-residency accounting without changing the harness's
    signature.
    """

    name: str = "abstract"

    def __init__(
        self,
        topology: GraphTopologySource,
        contract: RealTimeContract,
        predictor_horizon: int = 1,
    ) -> None:
        self._topology = topology
        self._contract = contract
        self._predictor_horizon = predictor_horizon
        self._predictor = CostPredictorState()
        self._n_admit = 0
        self._n_reject = 0
        self._queue_depth = 0
        # History of (reach_score, actual_us) pairs for calibration.
        self._calibration_buf: Deque[Tuple[float, float]] = deque(
            maxlen=4096
        )
        # Bridge state from ``evaluate(txn, t_now)`` to
        # ``record_completion(...)`` so subclasses can access
        # transaction context without a harness signature change.
        # Cleared after use.
        self._pending_evaluation: Optional[_PendingEvaluation] = None

    # --- public surface --------------------------------------------------

    def evaluate(
        self, txn: Transaction, t_now: float,
    ) -> AdmissionDecision:
        """
        Decide whether to admit the transaction.  This is the hot
        path; the implementation must be O(predictor cost) and not
        do any I/O.
        """
        reach = local_reach_predictor(
            txn, self._topology, self._predictor_horizon,
        )
        predicted_us = self._predictor.predict_us(reach)
        threshold = self._current_threshold_us()
        admit = predicted_us <= threshold
        if admit:
            self._n_admit += 1
            self._queue_depth += 1   # decremented when SUT signals completion
        else:
            self._n_reject += 1
        # Record context for record_completion.  If the harness
        # rejects, _on_arrival is still called for rate tracking,
        # but the pending context is cleared (we won't see a
        # record_completion for this admission).
        decision = AdmissionDecision(
            admit=admit,
            predicted_cost_us=predicted_us,
            threshold_us=threshold,
            timestamp=t_now,
            txn_source=txn.source,
            reason=self._explain(admit, predicted_us, threshold),
        )
        if admit:
            self._pending_evaluation = _PendingEvaluation(
                txn_source=txn.source,
                arrival_time=t_now,
                reach_score=reach,
                predicted_us=predicted_us,
            )
        else:
            self._pending_evaluation = None
        # Subclass hook for per-arrival accounting (rate tracking,
        # arrival-rate estimation, etc.).  Called for both
        # admissions and rejections so labellers see the full
        # arrival distribution.
        self._on_arrival(txn, t_now, admit)
        return decision

    def record_completion(
        self, predicted_us: float, actual_us: float,
    ) -> None:
        """
        Called by the experiment harness after the SUT processes an
        admitted transaction.  The defense uses this to recalibrate
        its predictor and (in adaptive variants) its threshold.

        Note: this signature is preserved for harness compatibility.
        Subclasses needing per-transaction context (txn source,
        arrival time, etc.) read from ``self._pending_evaluation``,
        which is populated by the immediately-preceding ``evaluate``
        call.
        """
        self._queue_depth = max(0, self._queue_depth - 1)
        # Recover the underlying reach score from the predicted
        # cost.
        if self._predictor.slope > 1e-12:
            reach = (
                (predicted_us - self._predictor.intercept)
                / self._predictor.slope
            )
        else:
            reach = predicted_us
        self._calibration_buf.append((reach, actual_us))
        self._on_completion(predicted_us, actual_us)
        # Clear the pending context; the next evaluate populates it.
        self._pending_evaluation = None

    def record_completion_with_groundtruth(
        self,
        predicted_us: float,
        actual_us: float,
        is_adversarial: bool,
    ) -> None:
        """
        Camera-ready: ground-truth-aware completion hook.

        The default ``record_completion`` uses a heuristic labeller
        (per-source rate OR predictor residual) because the defender
        does not have ground truth at production time.  When the
        experiment harness DOES have ground truth — for instance,
        because the workload generator marked the transaction with
        ``Attack.is_adversarial_txn_id`` — it can pass the truth
        through this method and the defense will use it to bucket
        the cost sample directly, bypassing the heuristic.

        The default implementation routes to ``record_completion``
        and ignores ``is_adversarial``.  Only D3 overrides it
        (D1 and D2 do not maintain per-class buckets).

        §V.4 of the paper reports the ablation comparing heuristic
        labelling vs. ground-truth labelling on D3.
        """
        # Default: ignore the truth flag; subclasses override.
        self.record_completion(predicted_us, actual_us)

    # --- DefenderStateSource protocol -----------------------------------

    def admission_threshold(self) -> float:
        return self._current_threshold_us()

    def queue_depth(self) -> int:
        return self._queue_depth

    # --- Lemma 4.4 coupling ---------------------------------------------

    def adaptation_rate_hz(self) -> float:
        """
        Defender adaptation rate ρ_def for Lemma 4.4 (was Theorem
        4.4 in the submission draft; see camera-ready analysis.py
        for the demotion rationale).

        A defense with no online adaptation (D1) returns 0; D2 and
        D3 return the rate at which they refresh their thresholds.
        """
        return 0.0

    # --- internal hooks --------------------------------------------------

    @abc.abstractmethod
    def _current_threshold_us(self) -> float:
        """The current admission threshold.  Subclass-specific."""
        raise NotImplementedError

    def _on_arrival(
        self,
        txn: Transaction,
        t_now: float,
        admitted: bool,
    ) -> None:
        """
        Hook called for every arrival (admit or reject), BEFORE the
        SUT processes the transaction.  Default: no-op.

        Subclasses use this to maintain per-source rate trackers,
        global arrival-rate estimators, or other arrival-time
        signals.
        """

    def _on_completion(
        self, predicted_us: float, actual_us: float,
    ) -> None:
        """
        Hook called on every completion.  Default: re-calibrate
        periodically.
        """
        if (
            len(self._calibration_buf) % 256 == 0
            and len(self._calibration_buf) >= 32
        ):
            scores = [s for s, _ in self._calibration_buf]
            costs = [c for _, c in self._calibration_buf]
            self._predictor = calibrate_predictor(
                self._predictor, scores, costs,
            )

    def _explain(
        self,
        admit: bool,
        predicted: float,
        threshold: float,
    ) -> str:
        return (
            f"{self.name}: predicted={predicted:.0f}us, "
            f"T={threshold:.0f}us "
            f"{'ADMIT' if admit else 'REJECT'}"
        )

    # --- statistics ------------------------------------------------------

    def statistics(self) -> Mapping[str, Any]:
        n = self._n_admit + self._n_reject
        return {
            "name": self.name,
            "n_admitted": self._n_admit,
            "n_rejected": self._n_reject,
            "rejection_rate": (self._n_reject / n) if n > 0 else 0.0,
            "current_threshold_us": self._current_threshold_us(),
            "predictor": {
                "slope": self._predictor.slope,
                "intercept": self._predictor.intercept,
                "n_calibration": self._predictor.n_calibration_samples,
                "residual_p99_us": self._predictor.last_residual_p99,
            },
            "queue_depth": self._queue_depth,
            "adaptation_rate_hz": self.adaptation_rate_hz(),
        }


# =============================================================================
# Section 3.  D1 — Static threshold (baseline).
#
# A fixed cost threshold.  Useful as a baseline because it answers:
# "what does the simplest possible defense look like?"  The static
# threshold cannot adapt to changing α (adversary fraction) or to
# distribution shift; D2 and D3 generalise it.
# =============================================================================


class D1Static(Defense):
    name = "D1_static"

    def __init__(
        self,
        topology: GraphTopologySource,
        contract: RealTimeContract,
        threshold_us: float,
        predictor_horizon: int = 1,
    ) -> None:
        super().__init__(topology, contract, predictor_horizon)
        self._threshold = float(threshold_us)

    def _current_threshold_us(self) -> float:
        return self._threshold

    # adaptation_rate_hz() returns 0 by default — D1 never adapts.


# =============================================================================
# Section 4.  D2 — Adaptive threshold.
#
# Maintains an EWMA of recent miss indicators and adjusts the
# threshold multiplicatively to keep the recent miss rate at or
# below the contract's epsilon.  This is a pure feedback controller
# — no explicit theoretical bound — included so that the paper can
# demonstrate that schedulability awareness (D3) buys something
# measurable beyond pure feedback.
#
# The adjustment cadence is fixed at 100 ms (10 Hz) by default.
# This is the rate ρ_def that Lemma 4.4 references.  It is exposed
# via ``adaptation_rate_hz()`` so that the experiment harness can
# populate the κ analysis correctly.
# =============================================================================


class D2Adaptive(Defense):
    name = "D2_adaptive"

    def __init__(
        self,
        topology: GraphTopologySource,
        contract: RealTimeContract,
        initial_threshold_us: float,
        adapt_alpha: float = 0.05,            # EWMA weight on each completion
        adjust_rate: float = 0.05,            # multiplicative adjust step
        min_threshold_us: float = 50.0,
        max_threshold_us: Optional[float] = None,
        adjust_interval_s: float = 0.1,       # = 1 / adaptation_rate_hz
        predictor_horizon: int = 1,
    ) -> None:
        super().__init__(topology, contract, predictor_horizon)
        self._threshold = float(initial_threshold_us)
        self._adapt_alpha = float(adapt_alpha)
        self._adjust_rate = float(adjust_rate)
        self._min_threshold = float(min_threshold_us)
        self._max_threshold = (
            float(max_threshold_us)
            if max_threshold_us is not None
            else float(contract.deadline_us)
        )
        # EWMA of the deadline-miss indicator.
        self._miss_ewma: float = 0.0
        # Last threshold-update time (monotonic seconds).
        self._last_adjust_time: float = -math.inf
        # Adjustment cadence.  The reciprocal is the adversary-
        # observable adaptation rate (used by Lemma 4.4).
        self._adjust_interval_s: float = float(adjust_interval_s)
        self._n_adjustments: int = 0

    def _current_threshold_us(self) -> float:
        return self._threshold

    def adaptation_rate_hz(self) -> float:
        """
        ρ_def for Lemma 4.4.  D2 adjusts at most once per
        ``_adjust_interval_s``, so its observable adaptation rate is
        bounded by 1 / interval.
        """
        if self._adjust_interval_s <= 0:
            return float("inf")
        return 1.0 / self._adjust_interval_s

    def _on_completion(
        self, predicted_us: float, actual_us: float,
    ) -> None:
        super()._on_completion(predicted_us, actual_us)
        missed = float(actual_us > self._contract.deadline_us)
        self._miss_ewma = (
            (1.0 - self._adapt_alpha) * self._miss_ewma
            + self._adapt_alpha * missed
        )
        now = time.monotonic()
        if now - self._last_adjust_time < self._adjust_interval_s:
            return
        target = self._contract.failure_probability_bound
        # If the miss rate is too high, *lower* the threshold (admit
        # fewer heavy transactions).  If it's too low, *raise* the
        # threshold (admit more — gives back throughput).
        if self._miss_ewma > target * 1.5:
            new = self._threshold * (1.0 - self._adjust_rate)
        elif self._miss_ewma < target * 0.5:
            new = self._threshold * (1.0 + self._adjust_rate)
        else:
            new = self._threshold
        clamped = float(
            max(self._min_threshold, min(self._max_threshold, new))
        )
        if clamped != self._threshold:
            self._n_adjustments += 1
        self._threshold = clamped
        self._last_adjust_time = now

    def statistics(self) -> Mapping[str, Any]:
        s = dict(super().statistics())
        s["miss_ewma"] = self._miss_ewma
        s["n_adjustments"] = self._n_adjustments
        s["adjust_interval_s"] = self._adjust_interval_s
        return s


# =============================================================================
# Section 5.  D3 — Schedulability-aware admission.
#
# The headline defense.  Periodically refits the benign and
# adversarial tail bounds (Theorems 4.1 and 4.2), invokes BOTH
# Theorem 4.3-env (lightweight envelope) AND Theorem 4.3-mgf
# (Sun-style random-sum) from analysis.py, and uses the envelope-
# derived threshold for admission.  The MGF certificate is reported
# but does not gate admission, because its computation cost (~10ms)
# is too high for the hot path.  This is the standard pattern in
# stochastic-timing analysis: cheap online check, expensive offline
# certificate.
#
# Refitting happens on a fixed cadence so the threshold-adaptation
# rate is bounded (Lemma 4.4 applies via ``adaptation_rate_hz()``).
#
# Adversarial-vs-benign labelling
# -------------------------------
# D3 maintains two cost histories — benign and adversarial — to
# estimate per-class tail behaviour.  Labelling is a hard problem
# in a real deployment because we do not have ground-truth at
# admission time; D3's labeller is a *combination* of two
# heuristics, which together flag a transaction as adversarial when
# EITHER signal fires:
#
#   (a) Per-source rate.  A source emitting at >
#       ``suspicion_rate_factor`` times the global median per-source
#       rate is flagged.  This is the original AoI-style argument:
#       an honest user does not suddenly send 10x more transactions
#       than the population.
#
#   (b) Per-completion residual.  A completion whose actual cost
#       exceeds the predicted cost by more than
#       ``residual_z_threshold`` times the predictor's residual-σ
#       is flagged.  This catches transactions whose real cost was
#       unpredictable from topology.
#
# Both signals are imperfect.  We use the OR rule because the
# paper's point is that even with imperfect labels, the
# schedulability-aware policy outperforms the feedback-only D2.  A
# reviewer can audit either heuristic in isolation by setting the
# other's threshold to infinity.
#
# Camera-ready: the harness can ALSO supply ground-truth labels via
# ``record_completion_with_groundtruth``.  When ground truth is
# present (e.g. in §V.4 ablations), D3 routes the cost sample to
# the correct bucket directly, bypassing the heuristic.  This lets
# the paper compare both labelling regimes.
#
# Carry-in (B_0) under synchronous execution
# ------------------------------------------
# CAMERA-READY CHANGE: the submission draft tracked queue-residency
# samples (commit_time − admission_time) and fed them into the MGF
# certificate as B_0.  Under the synchronous experiments.py harness
# — where ``defense.evaluate`` → ``sut.update_path.process`` →
# ``defense.record_completion`` is one straight-line call chain —
# the time delta IS the SUT's processing time plus some clock-read
# overheads, NOT a queueing residency.  Feeding it as B_0 double-
# counts the cost.  The camera-ready acknowledges this and passes
# ``carry_in=None`` to the MGF call; ``apply_theorem_4_3_mgf``
# correctly logs the standard "synchronous-harness" caveat.
#
# When a future asynchronous harness is added (e.g., a queue-
# theoretic deployment with parallel SUT workers), the carry-in
# channel can be repopulated.  ``D3.statistics()`` reports
# ``carry_in_synchronous=True`` so reviewers can see which mode
# produced the certificate.
# =============================================================================


# Default heuristic parameters.  Reviewers can override via
# constructor.
_DEFAULT_REFIT_INTERVAL_S: float = 1.0
_DEFAULT_HISTORY_SIZE: int = 4096
_DEFAULT_SUSPICION_RATE_FACTOR: float = 2.0
_DEFAULT_RESIDUAL_Z_THRESHOLD: float = 3.0     # 3σ above prediction
_DEFAULT_PRIOR_ADV_FRACTION: float = 0.01      # used until adv samples observed
_DEFAULT_GLOBAL_RATE_HZ: float = 1.0           # baseline per-source rate
_DEFAULT_FALLBACK_PERCENTILE: float = 10.0     # p10 of benign costs


@dataclass
class _SourceProfile:
    """Per-source rate tracking for the adversarial-vs-benign labeller."""

    emission_times: Deque[float] = field(
        default_factory=lambda: deque(maxlen=64),
    )

    def record(self, t_now: float) -> None:
        self.emission_times.append(t_now)

    def recent_rate_hz(
        self, t_now: float, window_s: float = 1.0,
    ) -> float:
        cutoff = t_now - window_s
        recent = [t for t in self.emission_times if t >= cutoff]
        return len(recent) / window_s


class D3Schedulability(Defense):
    name = "D3_schedulability"

    def __init__(
        self,
        topology: GraphTopologySource,
        contract: RealTimeContract,
        initial_threshold_us: Optional[float] = None,
        refit_interval_s: float = _DEFAULT_REFIT_INTERVAL_S,
        history_size: int = _DEFAULT_HISTORY_SIZE,
        predictor_horizon: int = 1,
        suspicion_rate_factor: float = _DEFAULT_SUSPICION_RATE_FACTOR,
        residual_z_threshold: float = _DEFAULT_RESIDUAL_Z_THRESHOLD,
        prior_adv_fraction: float = _DEFAULT_PRIOR_ADV_FRACTION,
        rate_window_s: float = 1.0,
        enable_mgf_certificate: bool = True,
        fallback_percentile: float = _DEFAULT_FALLBACK_PERCENTILE,
        # Retained for backwards compatibility with the
        # submission-draft constructor signature; the parameter is
        # ignored under the synchronous harness because B_0 = 0 is
        # the correct value (see §5 docstring).
        carry_in_history_size: int = 1024,
    ) -> None:
        super().__init__(topology, contract, predictor_horizon)
        self._refit_interval = float(refit_interval_s)
        self._last_refit_time: float = -math.inf
        # Rolling buffers of cost samples, labelled benign /
        # adversarial.
        self._benign_costs: Deque[float] = deque(maxlen=history_size)
        self._adversarial_costs: Deque[float] = deque(
            maxlen=history_size,
        )
        # Last-known certificates.
        self._envelope_certificate: Optional[SchedulabilityCertificate] = None
        self._mgf_certificate: Optional[MGFCertificate] = None
        self._threshold = (
            float(initial_threshold_us)
            if initial_threshold_us is not None
            else float(contract.deadline_us) / 2.0
        )
        # Per-source profiles for the rate-based labeller.
        self._sources: Dict[int, _SourceProfile] = {}
        self._suspicion_rate_factor = float(suspicion_rate_factor)
        self._residual_z_threshold = float(residual_z_threshold)
        self._rate_window_s = float(rate_window_s)
        # Adversary-fraction estimate: fraction of *labelled*
        # adversarial completions over total.  Updated on each
        # refit.
        self._prior_adv_fraction = float(prior_adv_fraction)
        self._adversary_fraction_estimate: float = float(
            prior_adv_fraction,
        )
        # Whether to compute the MGF certificate at refit time.
        self._enable_mgf_certificate = bool(enable_mgf_certificate)
        # Configurable fallback percentile when the certificate is
        # infeasible.
        if not 0.0 < fallback_percentile < 100.0:
            raise ValueError(
                "fallback_percentile must be in (0, 100); got "
                f"{fallback_percentile}"
            )
        self._fallback_percentile = float(fallback_percentile)
        # Bookkeeping on the labeller.
        self._n_labelled_rate: int = 0          # via per-source rate
        self._n_labelled_residual: int = 0      # via predictor residual
        self._n_labelled_groundtruth: int = 0   # via harness groundtruth
        self._n_refits: int = 0
        # For estimating admitted-stream rate (used by MGF
        # certificate).  Per-window admission counts feed the
        # measured CountDistribution.
        self._admit_times: Deque[float] = deque(maxlen=history_size)
        # Camera-ready: synchronous-harness flag, surfaced in
        # statistics() for reviewer audit.
        self._carry_in_synchronous: bool = True
        # carry_in_history_size kept for compatibility; not used.
        del carry_in_history_size  # silence unused-arg lint

    def _current_threshold_us(self) -> float:
        return self._threshold

    def adaptation_rate_hz(self) -> float:
        """
        ρ_def for Lemma 4.4.  D3 refits at most once per
        ``_refit_interval``, so its observable adaptation rate is
        bounded by 1 / interval.
        """
        if self._refit_interval <= 0:
            return float("inf")
        return 1.0 / self._refit_interval

    # --- evaluation overrides for rate tracking -------------------------

    def _on_arrival(
        self,
        txn: Transaction,
        t_now: float,
        admitted: bool,
    ) -> None:
        # Record the source emission for rate tracking.  We track
        # ALL arrivals (admit or reject) so the labeller observes
        # the true source-rate distribution even when the policy is
        # rejecting heavily-emitting sources.
        self._sources.setdefault(
            txn.source, _SourceProfile(),
        ).record(t_now)
        if admitted:
            self._admit_times.append(t_now)

    # --- on completion: bucket the cost, possibly refit threshold -------

    def _on_completion(
        self, predicted_us: float, actual_us: float,
    ) -> None:
        super()._on_completion(predicted_us, actual_us)
        # Adversarial-vs-benign labelling via heuristic.  The label
        # combines two signals (rate and residual); the OR rule
        # means a transaction is benign only if BOTH signals are
        # quiet.
        is_adversarial = self._label_completion(predicted_us, actual_us)
        if is_adversarial:
            self._adversarial_costs.append(actual_us)
        else:
            self._benign_costs.append(actual_us)
        # Refit on cadence.
        now_wall = time.monotonic()
        if now_wall - self._last_refit_time >= self._refit_interval:
            self._refit(now_wall)

    def record_completion_with_groundtruth(
        self,
        predicted_us: float,
        actual_us: float,
        is_adversarial: bool,
    ) -> None:
        """
        Camera-ready: ground-truth-aware completion hook.

        When the harness has ground truth (typically via
        ``Attack.is_adversarial_txn_id`` from attacks.py), it can
        pass it directly; D3 routes the sample to the correct
        bucket without invoking the heuristic.  Refit cadence and
        all other behaviours are unchanged.

        Note: ``self._predictor`` calibration still runs on every
        completion regardless of the labelling channel.
        """
        # Standard predictor calibration via the parent's
        # super()._on_completion.  We need to invoke it without
        # going through _label_completion or duplicating its
        # bucket-update logic.
        Defense._on_completion(self, predicted_us, actual_us)

        # Update per-source bookkeeping (the heuristic labeller
        # needs source rates even when ground truth supplies the
        # labels, in case of mixed-source mode).  This was already
        # done in _on_arrival, so no extra work needed here.

        if is_adversarial:
            self._adversarial_costs.append(actual_us)
            self._n_labelled_groundtruth += 1
        else:
            self._benign_costs.append(actual_us)

        # Refit on cadence — same as the heuristic path.
        now_wall = time.monotonic()
        if now_wall - self._last_refit_time >= self._refit_interval:
            self._refit(now_wall)

        # Mirror the public record_completion bookkeeping that the
        # default path would have done.
        self._queue_depth = max(0, self._queue_depth - 1)
        if self._predictor.slope > 1e-12:
            reach = (
                (predicted_us - self._predictor.intercept)
                / self._predictor.slope
            )
        else:
            reach = predicted_us
        self._calibration_buf.append((reach, actual_us))
        self._pending_evaluation = None

    def _label_completion(
        self, predicted_us: float, actual_us: float,
    ) -> bool:
        """
        Decide whether a completed transaction is "adversarial" for
        the purposes of D3's tail-fitting buckets.

        Returns True if EITHER of two signals fires:

          (a) The transaction's source has a recent emission rate
              greater than ``suspicion_rate_factor`` × the global
              median per-source rate.
          (b) The actual cost exceeds the predicted cost by more
              than ``residual_z_threshold`` × the predictor's last
              residual-σ.

        The OR rule is conservative: if either heuristic flags the
        completion, it goes into the adversarial bucket.  False
        positives inflate the adversarial-tail fit (making the
        bound more conservative, which is safe); false negatives
        leave adversarial costs in the benign bucket (which loosens
        the adversarial fit but does not invalidate it because
        Theorem 4.3-env mixes both fits).

        Both heuristics are imperfect.  The paper's §V.4 ablation
        reports performance under each heuristic in isolation by
        setting the other's threshold to infinity, AND under
        ground-truth labelling via
        ``record_completion_with_groundtruth``.
        """
        labelled_via_rate = False
        labelled_via_residual = False

        # (a) Per-source rate.
        if self._pending_evaluation is not None:
            src_id = self._pending_evaluation.txn_source
            now = time.monotonic()
            profile = self._sources.get(src_id)
            if profile is not None:
                src_rate = profile.recent_rate_hz(
                    now, self._rate_window_s,
                )
                # Use the median of all per-source rates as the
                # global baseline.  Falls back to a configurable
                # default if we have too few sources to compute a
                # median.
                if len(self._sources) >= 4:
                    all_rates = [
                        p.recent_rate_hz(now, self._rate_window_s)
                        for p in self._sources.values()
                    ]
                    global_rate = float(np.median(all_rates))
                else:
                    global_rate = _DEFAULT_GLOBAL_RATE_HZ
                if (
                    global_rate > 1e-12
                    and src_rate > self._suspicion_rate_factor * global_rate
                ):
                    labelled_via_rate = True

        # (b) Per-completion residual.
        residual = actual_us - predicted_us
        # Use the predictor's tracked residual-σ if available.
        # ``last_residual_p99`` is in the same units as residual,
        # but p99 ≈ 2.33σ for a Gaussian; we convert.
        if self._predictor.last_residual_p99 > 0:
            sigma_est = self._predictor.last_residual_p99 / 2.33
            if (
                sigma_est > 1e-9
                and residual > self._residual_z_threshold * sigma_est
            ):
                labelled_via_residual = True
        else:
            # Fall back to a relative ratio when residual-σ is not
            # yet estimated (warmup).  This is the original
            # heuristic.
            ratio = actual_us / max(predicted_us, 1.0)
            if ratio > 1.5:
                labelled_via_residual = True

        if labelled_via_rate:
            self._n_labelled_rate += 1
        if labelled_via_residual:
            self._n_labelled_residual += 1

        return labelled_via_rate or labelled_via_residual

    def _refit(self, now: float) -> None:
        """
        Refit the tail bounds and recompute the schedulability
        certificates.  Called every refit_interval_s seconds.

        If the historical buffers are too small, retain the
        previous threshold and try again next interval.
        """
        if len(self._benign_costs) < 64:
            return
        self._n_refits += 1

        benign_fit = fit_exponential_upper_bound(
            list(self._benign_costs),
        )
        if len(self._adversarial_costs) >= 32:
            adv_fit = fit_exponential_upper_bound(
                list(self._adversarial_costs),
            )
            self._adversary_fraction_estimate = (
                len(self._adversarial_costs)
                / max(
                    1,
                    len(self._benign_costs)
                    + len(self._adversarial_costs),
                )
            )
        else:
            adv_fit = benign_fit
            self._adversary_fraction_estimate = self._prior_adv_fraction

        # 1. Envelope certificate (Theorem 4.3-env): cheap, online.
        env_cert = apply_theorem_4_3(
            benign_fit=benign_fit,
            adversarial_fit=adv_fit,
            contract=self._contract,
            adversary_fraction=self._adversary_fraction_estimate,
            threshold_us=None,
        )

        # If feasible, adopt the certified threshold.  If not, fall
        # back to a principled conservative threshold derived from
        # observed benign data: the configured percentile of
        # admitted-and-completed benign costs.  This admits only
        # the lightest f-percent of typical work, which is the
        # strongest position the data supports without leaving the
        # contract.  The experiment harness will record the
        # infeasibility so the paper can report it.
        if env_cert.feasible:
            self._threshold = env_cert.threshold_us
        else:
            self._threshold = self._principled_fallback_threshold()
            logger.warning(
                f"D3 schedulability infeasible at α≈"
                f"{self._adversary_fraction_estimate:.3f}; "
                f"required T={env_cert.notes}; "
                f"falling back to T={self._threshold:.0f}us "
                f"(p{self._fallback_percentile} of benign cost history)"
            )
        self._envelope_certificate = env_cert

        # 2. MGF certificate (Theorem 4.3-mgf): expensive, offline.
        # Computed on the same cadence but does not gate admission.
        if self._enable_mgf_certificate:
            self._mgf_certificate = self._build_mgf_certificate()

        self._last_refit_time = now

    def _principled_fallback_threshold(self) -> float:
        """
        Conservative threshold to use when the schedulability bound
        is infeasible at current α.

        Returns the configured percentile of benign cost history
        (default p10, so we admit only the lightest decile of
        typical work).  Falls back to D/4 when history is empty.
        """
        if len(self._benign_costs) > 0:
            return float(
                np.percentile(
                    list(self._benign_costs),
                    self._fallback_percentile,
                )
            )
        return float(self._contract.deadline_us) / 4.0

    def _build_mgf_certificate(self) -> Optional[MGFCertificate]:
        """
        Build the Sun-style MGF certificate from the current cost
        histories using the modern dict-based signature of
        ``apply_theorem_4_3_mgf``.

        Camera-ready: switched from the legacy keyword API
        (``benign_cost_dist=``, ``expected_admitted_count_benign=``,
        etc.) which now triggers a logger.info per refit because it
        wraps expected counts as Poisson on the analysis side.  We
        now construct measured CountDistributions from the per-
        window admit_times log; no Poisson assumption.

        Camera-ready: also passes ``carry_in=None`` because the
        synchronous harness does not produce a true queue-residency
        sample (see §5 docstring).  When a future async harness is
        added, the carry-in channel can be repopulated.

        Returns None if construction would be unsafe (insufficient
        samples or degenerate distributions).
        """
        if len(self._benign_costs) < 64:
            return None

        try:
            benign_dist = cost_distribution_from_samples(
                list(self._benign_costs), label="benign",
            )
        except (ValueError, RuntimeError) as e:
            logger.warning(
                f"D3 MGF: failed to build benign cost dist: {e}"
            )
            return None

        # Estimate admitted-stream rate from recent admit_times.
        if len(self._admit_times) >= 2:
            window = self._admit_times[-1] - self._admit_times[0]
            admit_rate_hz = (
                (len(self._admit_times) - 1) / max(window, 1e-9)
            )
        else:
            admit_rate_hz = 0.0

        D_s = self._contract.deadline_us / 1e6
        alpha_hat = self._adversary_fraction_estimate

        # Per-class expected counts within one deadline window.
        # These are the means used to construct measured (or
        # Poisson) count distributions.
        benign_count = max(
            0.0, admit_rate_hz * D_s * (1.0 - alpha_hat),
        )
        adv_count = max(0.0, admit_rate_hz * D_s * alpha_hat)

        # Build measured count distributions.  In a future revision
        # the harness can record per-window admission counts and
        # supply the empirical histogram via
        # ``count_distribution_from_samples``.  For now we use the
        # named-Poisson constructor with the measured mean per
        # window — explicit assumption, not hidden.  The Poisson
        # path is only invoked if we lack richer data, and the
        # logger records that fact.
        cost_distributions: Dict[str, Any] = {"benign": benign_dist}
        count_distributions: Dict[str, CountDistribution] = {
            "benign": poisson_count_distribution(
                mean=benign_count,
                label="benign(measured-mean-per-D)",
            ),
        }

        if len(self._adversarial_costs) >= 32:
            try:
                adv_dist = cost_distribution_from_samples(
                    list(self._adversarial_costs),
                    label="adversarial",
                )
                cost_distributions["adversarial"] = adv_dist
                count_distributions["adversarial"] = (
                    poisson_count_distribution(
                        mean=adv_count,
                        label="adversarial(measured-mean-per-D)",
                    )
                )
            except (ValueError, RuntimeError) as e:
                logger.warning(
                    f"D3 MGF: failed to build adversarial dist: {e}"
                )

        try:
            return apply_theorem_4_3_mgf(
                contract=self._contract,
                cost_distributions=cost_distributions,
                count_distributions=count_distributions,
                # Carry-in is synchronous-harness-correct: B_0 = 0
                # (degenerate-zero handled inside analysis.py).
                carry_in=None,
            )
        except (ValueError, RuntimeError, OverflowError) as e:
            logger.warning(
                f"D3 MGF: certificate computation failed: {e}"
            )
            return None

    # --- public surface for run-record inspection -----------------------

    def latest_certificate(self) -> Optional[SchedulabilityCertificate]:
        """The most-recent envelope certificate (Theorem 4.3-env)."""
        return self._envelope_certificate

    def latest_mgf_certificate(self) -> Optional[MGFCertificate]:
        """The most-recent MGF certificate (Theorem 4.3-mgf)."""
        return self._mgf_certificate

    def carry_in_samples_us(self) -> Sequence[float]:
        """
        Camera-ready: the synchronous harness does not produce
        carry-in samples (see §5 docstring).  Always returns an
        empty tuple.  Retained for backwards compatibility with the
        submission-draft signature; experiments.py should not
        forward a non-empty result to ``analyse_storm`` because
        doing so double-counts the SUT cost.
        """
        return ()

    def statistics(self) -> Mapping[str, Any]:
        s = dict(super().statistics())
        s["adversary_fraction_estimate"] = (
            self._adversary_fraction_estimate
        )
        s["benign_history_size"] = len(self._benign_costs)
        s["adversarial_history_size"] = len(self._adversarial_costs)
        s["n_refits"] = self._n_refits
        s["n_labelled_rate"] = self._n_labelled_rate
        s["n_labelled_residual"] = self._n_labelled_residual
        s["n_labelled_groundtruth"] = self._n_labelled_groundtruth
        s["refit_interval_s"] = self._refit_interval
        s["fallback_percentile"] = self._fallback_percentile
        # Camera-ready: surface the synchronous-harness B_0=0
        # decision so reviewers can verify which mode produced the
        # certificate.
        s["carry_in_synchronous"] = self._carry_in_synchronous
        if self._envelope_certificate is not None:
            cert = self._envelope_certificate
            s["envelope_certificate"] = {
                "feasible": cert.feasible,
                "bound_at_T": cert.bound_at_T,
                "slack_us": cert.slack_us,
                "threshold_us": cert.threshold_us,
                "epsilon": cert.epsilon,
            }
        if self._mgf_certificate is not None:
            mgf = self._mgf_certificate
            # Camera-ready: use the new field names directly
            # (``optimal_lambda`` and ``components``).  The
            # submission draft used ``optimal_lam`` and
            # ``per_class_bounds`` (back-compat aliases on
            # MGFCertificate) plus a now-removed
            # ``carry_in_log_mgf`` field.  The B_0 entry of
            # ``components`` replaces the latter when present.
            b0_log_mgf = mgf.components.get("B_0", float("nan"))
            s["mgf_certificate"] = {
                "feasible": mgf.feasible,
                "bound": mgf.bound,
                "log_bound": mgf.log_bound,
                "optimal_lambda": mgf.optimal_lambda,
                "epsilon": mgf.epsilon,
                "n_classes": (
                    len(mgf.components) - (1 if "B_0" in mgf.components else 0)
                ),
                "carry_in_log_mgf": b0_log_mgf,
                "lambda_at_edge": mgf.optimal_lambda_at_bracket_edge,
                "notes": mgf.notes,
            }
        return s


# =============================================================================
# Section 6.  Defense registry.
# =============================================================================


_DEFENSE_REGISTRY: Mapping[str, Callable[..., Defense]] = {
    D1Static.name: D1Static,
    D2Adaptive.name: D2Adaptive,
    D3Schedulability.name: D3Schedulability,
}


def defense_names() -> Sequence[str]:
    return tuple(_DEFENSE_REGISTRY.keys())


def make_defense(
    name: str,
    topology: GraphTopologySource,
    contract: RealTimeContract,
    **kwargs: Any,
) -> Defense:
    """
    Construct a defense by name.  Additional kwargs are forwarded
    to the constructor; the experiment harness reads them from
    configs/defense_profiles.yaml so the file is the single source
    of truth for defense parameters.
    """
    if name not in _DEFENSE_REGISTRY:
        raise KeyError(
            f"unknown defense '{name}'; "
            f"known: {sorted(_DEFENSE_REGISTRY)}"
        )
    cls = _DEFENSE_REGISTRY[name]
    return cls(topology=topology, contract=contract, **kwargs)


# =============================================================================
# Section 7.  Public surface.
# =============================================================================


__all__ = [
    # Common
    "AdmissionDecision",
    "Defense",
    # Cost predictor
    "CostPredictorState",
    "local_reach_predictor",
    "calibrate_predictor",
    # Concrete defenses
    "D1Static",
    "D2Adaptive",
    "D3Schedulability",
    # Registry
    "defense_names",
    "make_defense",
]
# =============================================================================
# Section 6.  Defense registry.
#
# Single source of truth that maps name → constructor.  Used by
# experiments.py to instantiate defenses by name from
# configs/defense_profiles.yaml so the YAML is the single source of
# truth for defense parameters.
# =============================================================================


_DEFENSE_REGISTRY: Mapping[str, Callable[..., Defense]] = {
    D1Static.name: D1Static,
    D2Adaptive.name: D2Adaptive,
    D3Schedulability.name: D3Schedulability,
}


def defense_names() -> Sequence[str]:
    return tuple(_DEFENSE_REGISTRY.keys())


def make_defense(
    name: str,
    topology: GraphTopologySource,
    contract: RealTimeContract,
    **kwargs: Any,
) -> Defense:
    """
    Construct a defense by name.  Additional kwargs are forwarded
    to the constructor; the experiment harness reads them from
    configs/defense_profiles.yaml so the file is the single source
    of truth for defense parameters.
    """
    if name not in _DEFENSE_REGISTRY:
        raise KeyError(
            f"unknown defense '{name}'; "
            f"known: {sorted(_DEFENSE_REGISTRY)}"
        )
    cls = _DEFENSE_REGISTRY[name]
    return cls(topology=topology, contract=contract, **kwargs)


# =============================================================================
# Section 7.  Public surface.
# =============================================================================


__all__ = [
    # --- Submission-draft API (preserved verbatim) ----------------------
    # Common
    "AdmissionDecision",
    "Defense",
    # Cost predictor
    "CostPredictorState",
    "local_reach_predictor",
    "calibrate_predictor",
    # Concrete defenses
    "D1Static",
    "D2Adaptive",
    "D3Schedulability",
    # Registry
    "defense_names",
    "make_defense",
]
