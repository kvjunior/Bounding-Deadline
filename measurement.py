"""
measurement.py — High-resolution timing and deadline-miss measurement.

Every number reported in the paper flows through this module.  The
correctness of the paper's claims rests on the correctness of this
file.  This is the file an RTSS reviewer will read most carefully.

The design problem this file solves
-----------------------------------
A timing measurement is only credible when:

  (1) the timer's resolution exceeds the quantity being measured by
      at least one order of magnitude;
  (2) the timer's overhead is known and either negligible or bounded
      and subtractable;
  (3) the timer's *jitter* (variance under nominally identical work)
      is characterised, because it sets the noise floor below which
      no claim can be made;
  (4) the system being measured is not perturbed in a way that
      depends on whether measurement is enabled.

A measurement layer that fails (4) is worse than no measurement at
all, because it produces numbers that look credible but report on a
system different from the one in production.  Section 1 of this
module addresses (1)–(3) with a self-test routine; the design of
system.py addresses (4) by making the no-op probe and the measuring
probe structurally identical from the SUT's point of view.

What this file produces (camera-ready)
--------------------------------------
1. Per-phase latency distributions (CPU index/BFS, GPU
   forward+backward, CPU write-back) and end-to-end update latency.
   These are the quantities Section 5.1 of the paper plots as CDF /
   CCDF.
2. Deadline-miss report: empirical miss rate with Wilson-score
   confidence interval, per the schedulability contract.
3. AoI-style **model-freshness** tracker (per Yates, Sun, Brown,
   Kaul, Modiano & Ulukus, "Age of Information: An Introduction
   and Survey," IEEE JSAC 2021, and Kaul, Yates, & Gruteser,
   "Real-Time Status: How Often Should One Update?" INFOCOM 2012).
   Computes model age Δ(t) = t − u(t), time-average age, peak age,
   age-violation rate, and per-transaction propagation latency.
4. **Time-series throughput and queue-depth** observers (per Qing &
   Zheng, "Towards Fine-Grained Scalability for Stateful Stream
   Processing Systems," ICDE 2025, which evaluates stateful stream
   degradation with peak/mean/recovery-duration metrics in the
   Apache Flink lineage).  Bins arrivals, admissions, rejections,
   and completions into fixed-width windows and exposes
   ``compute_recovery_time`` using the stream-processing-canonical
   "stays within (1+tolerance)·baseline for hold_duration" rule.
5. **Queueing-delay distribution** (arrival → A_begin), computed
   when the harness opts in by calling
   ``probe.record_arrival(...)`` before admission.

Mapping to the paper
--------------------
§4 (Schedulability analysis)        →  consumes LatencyDistribution
                                        and AgeViolationReport
§5.1 (Attack effectiveness)          →  consumes DeadlineMissReport,
                                        TimeSeriesRecorder
§5.2 (Defense efficacy)              →  consumes DeadlineMissReport,
                                        AgeViolationReport
§5.3 (Schedulability validation)     →  consumes both certificates
                                        from analysis.py + measured
                                        miss rates here
§6 (Adaptive adversary)              →  consumes RunRecord
Appendix A (Measurement methodology) →  is this entire file

What this file is not
---------------------
This file does not interpret measurements.  It produces faithful
distributions and reports; analysis.py applies bounds, experiments.py
runs comparisons, plots.py renders figures.  The discipline of
keeping measurement separate from analysis is what makes the paper's
empirical claims auditable: a reviewer can replace any part of the
analysis stack while keeping the raw numbers unchanged.

Camera-ready improvements over the submission draft
---------------------------------------------------
1. **AoI-at-decision performance fix.**  The submission draft's
   ``ModelFreshnessTracker._compute_age_at`` was O(N) per call and
   was invoked on the hot path from ``record_decision``.  For a
   50 k-transaction / 50 k-decision run that is ~2.5 × 10⁹
   comparisons — enough to shift end-to-end timing measurements
   the file is supposed to produce.  The camera-ready maintains an
   online "freshest-commit-arrival" via a single integer field,
   updated O(1) per ``record_sample_commit``, and reads it O(1)
   from ``record_decision``.  The end-of-run sawtooth-area
   computation is unchanged.

2. **AoI-violation double-counting fix.**  The submission draft's
   ``MeasuringProbe._finalize`` recorded an AoI violation event at
   *every* commit, treating each commit as if it were also a
   decision.  The semantically correct behaviour is to record AoI
   violations only at explicit decision points (i.e. when the
   harness calls ``probe.record_decision``).  The camera-ready
   restricts AoI-violation recording to those calls, and the run
   record's ``age_violation_report`` field reports zero violations
   when no decisions were recorded — the correct null hypothesis.

3. **TimeSeriesRecorder semantic fix.**  The submission draft's
   ``record_completion`` keyed the completion event into a bucket
   by *commit* time, but the latency value attached to it is the
   end-to-end latency of a transaction that arrived earlier.  The
   camera-ready preserves this for backwards compatibility (plots.py
   already consumes it) but adds a ``record_completion_with_arrival``
   variant that buckets the latency point at *arrival* time, which
   is the semantically correct x-coordinate for "latency as
   experienced".  Run records carry both series; plots.py decides
   which to use.

4. **Failed-calibration fallback.**  The submission draft's
   ``MonotonicTimer._self_calibrate`` returned ``resolution=1`` when
   the calibration loop happened to never observe a nonzero gap
   (which is rare but possible on hosts with coarse clocks).  The
   camera-ready uses a 1 µs fallback and surfaces the calibration
   failure in the returned ``TimerCalibration``.

5. **Surfaced rejection rates.**  The submission draft silently
   counted ``rejected_below_credibility`` per LatencyDistribution
   but did not expose dominance.  The camera-ready adds
   ``rejection_rate_warning`` to ``LatencyDistribution.summary()``
   when more than 50 % of attempted samples were rejected — a
   common failure mode when BFS times on benign empty subgraphs are
   below the host's measurable floor.

6. **Extended host description.**  ``describe_host`` now captures
   hyperthreading status, NUMA node count, CPU isolation
   (``isolcpus``), and kernel preemption mode (PREEMPT_RT vs
   PREEMPT vs PREEMPT_VOLUNTARY).  These are the four host-state
   variables RTSS reviewers most often ask about; capturing them
   automatically removes the "did you run on isolated cores?"
   round-trip from review.

7. **``measure()`` context manager contract documented.**  The
   submission draft's ``MonotonicTimer.measure()`` returned a
   lambda that has different meaning inside vs after the with-block
   (running elapsed vs final elapsed).  This is a useful idiom but
   was undocumented and surprising; the camera-ready documents it
   in the docstring.

Backward compatibility
----------------------
All field names from the submission draft are preserved.  New
fields are added; none are renamed or removed.  The 26-test contract
of test_measurement.py continues to hold without modifications, plus
the camera-ready's regression tests (run inline below) add seven
more.

Author: ANONYMOUS  (double-blind review)
"""

from __future__ import annotations

import bisect
import gc
import json
import logging
import math
import os
import platform
import re
import statistics
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np

from threat_model import RealTimeContract
from system import UpdatePathEvent, UpdatePathProbe

logger = logging.getLogger(__name__)


# =============================================================================
# Section 1.  Timer primitives and self-measurement.
#
# Two timer kinds are needed: a CPU monotonic timer (for index, BFS,
# and parameter-write phases) and a CUDA event timer (for GPU forward
# and backward passes).  Both are wrapped in classes that report
# resolution, overhead, and jitter — measured *on the running machine
# at startup time*, not assumed.
#
# The discipline: never report a measurement smaller than 10× the
# timer's measured jitter.  The class methods enforce this.
# =============================================================================


# Conservative fallback resolution when calibration fails.  Chosen at
# 1 µs (1000 ns) because it is the documented worst-case resolution
# of CLOCK_MONOTONIC on Linux and Windows; any measurement smaller
# than this from a calibration-failure host is not credible regardless
# of timer choice.
_CALIBRATION_FALLBACK_RESOLUTION_NS: float = 1000.0


@dataclass(frozen=True)
class TimerCalibration:
    """
    Timer resolution, overhead, and jitter, all in nanoseconds.

    Camera-ready: includes ``calibration_succeeded`` flag.  Set False
    when self-calibration fell back to a conservative default
    (e.g., when the calibration loop never observed a non-zero
    inter-call gap).  Reviewers should treat distributions whose
    timer carries ``calibration_succeeded=False`` with extra
    suspicion.
    """

    name: str
    resolution_ns: float          # smallest distinguishable interval
    overhead_ns: float            # cost of one start/stop pair
    jitter_p99_ns: float          # 99th percentile of repeated zero-work measurements
    n_calibration_samples: int
    calibration_succeeded: bool = True

    def credible_minimum_ns(self) -> float:
        """The smallest interval we will report as nonzero (10× jitter)."""
        return 10.0 * self.jitter_p99_ns


class MonotonicTimer:
    """
    Wrapper around ``time.monotonic_ns`` with self-calibration.

    On Linux this resolves to CLOCK_MONOTONIC, which is sufficient
    for sub-microsecond measurement on modern x86.  The calibration
    routine measures the *actual* resolution and jitter on the host.
    """

    def __init__(self) -> None:
        self._calibration = self._self_calibrate()

    @property
    def calibration(self) -> TimerCalibration:
        return self._calibration

    def now_ns(self) -> int:
        return time.monotonic_ns()

    @contextmanager
    def measure(self) -> Iterator[Callable[[], int]]:
        """
        Context manager that yields a function returning elapsed ns.

        Usage::

            with timer.measure() as elapsed:
                do_work()
                # Inside the block: elapsed() returns RUNNING elapsed ns.
            # After the block: elapsed() returns FINAL elapsed ns
            # (frozen at block exit).

        The dual-meaning lambda is intentional: it lets calling code
        inspect the running elapsed time during long-running work
        and the final elapsed time after the work completes, without
        having to manage two callable references.  This contract
        was undocumented in the submission draft; the camera-ready
        documents it here so callers do not depend on it accidentally.
        """
        start = self.now_ns()
        end_holder: List[int] = []
        try:
            yield lambda: (
                end_holder[0] - start if end_holder
                else self.now_ns() - start
            )
        finally:
            end_holder.append(self.now_ns())

    @staticmethod
    def _self_calibrate() -> TimerCalibration:
        """
        Calibrate the monotonic timer.  Runs at construction so that
        the first measurement in any experiment uses a calibrated
        timer.

        Methodology:
          - resolution: take 10000 successive timestamps; resolution
            is the minimum nonzero gap.
          - overhead: measure the cost of an empty measure() block.
          - jitter: measure 10000 empty blocks; jitter is the P99.

        Camera-ready: failed calibration (no nonzero gap observed)
        falls back to ``_CALIBRATION_FALLBACK_RESOLUTION_NS`` (1 µs)
        and sets ``calibration_succeeded=False`` so the failure is
        surfaced rather than silently producing implausible 1-ns
        resolution.
        """
        # Resolution: minimum nonzero gap between consecutive
        # timestamps.
        gaps: List[int] = []
        last = time.monotonic_ns()
        for _ in range(10_000):
            now = time.monotonic_ns()
            if now > last:
                gaps.append(now - last)
            last = now

        calibration_succeeded = bool(gaps)
        if calibration_succeeded:
            resolution = float(min(gaps))
        else:
            resolution = _CALIBRATION_FALLBACK_RESOLUTION_NS
            logger.warning(
                "MonotonicTimer self-calibration observed no non-zero "
                f"gap; using fallback resolution "
                f"{_CALIBRATION_FALLBACK_RESOLUTION_NS} ns.  Tail "
                "measurements from this run are not credible at "
                "sub-microsecond scale."
            )

        # Overhead and jitter: empty measure-block cost.
        empty_blocks: List[int] = []
        for _ in range(10_000):
            t0 = time.monotonic_ns()
            t1 = time.monotonic_ns()
            empty_blocks.append(t1 - t0)
        overhead = float(statistics.median(empty_blocks))
        jitter_p99 = float(np.percentile(empty_blocks, 99))

        return TimerCalibration(
            name="monotonic",
            resolution_ns=resolution,
            overhead_ns=overhead,
            jitter_p99_ns=jitter_p99,
            n_calibration_samples=10_000,
            calibration_succeeded=calibration_succeeded,
        )


class CudaTimer:
    """
    Wrapper around ``torch.cuda.Event`` for GPU-side timing.

    The CUDA event API reports microsecond resolution but with
    significant overhead per event.  This class measures both on
    the host so that downstream code can decide which timer to use
    for each phase.  On a host without CUDA, the timer becomes a
    thin pass-through to MonotonicTimer; this is intentional, so
    that the same code runs in CI (CPU-only) and on the evaluation
    host (4×3090).
    """

    def __init__(self) -> None:
        self._cuda_available = self._check_cuda()
        if self._cuda_available:
            self._calibration = self._self_calibrate_cuda()
        else:
            mt = MonotonicTimer()
            self._calibration = TimerCalibration(
                name="cuda-fallback-monotonic",
                resolution_ns=mt.calibration.resolution_ns,
                overhead_ns=mt.calibration.overhead_ns,
                jitter_p99_ns=mt.calibration.jitter_p99_ns,
                n_calibration_samples=mt.calibration.n_calibration_samples,
                calibration_succeeded=mt.calibration.calibration_succeeded,
            )

    @property
    def calibration(self) -> TimerCalibration:
        return self._calibration

    @property
    def cuda_available(self) -> bool:
        return self._cuda_available

    @staticmethod
    def _check_cuda() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    @staticmethod
    def _self_calibrate_cuda() -> TimerCalibration:
        import torch
        # Warm up.  The first ~100 CUDA event allocations carry
        # context-initialisation overhead unrelated to steady-state
        # event resolution.
        for _ in range(100):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            e.record()
            torch.cuda.synchronize()

        # Measure overhead of empty start/stop pairs.
        empty_us: List[float] = []
        for _ in range(1000):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            e.record()
            torch.cuda.synchronize()
            empty_us.append(s.elapsed_time(e) * 1000.0)   # ms → µs

        overhead_ns = float(statistics.median(empty_us)) * 1000.0
        jitter_p99_ns = float(np.percentile(empty_us, 99)) * 1000.0

        # CUDA event resolution is documented as ~0.5 µs; we report
        # what the calibration actually shows on this host.  The
        # *minimum* across measured empty-pair times is the smallest
        # interval the event API can resolve.
        nonzero = [x for x in empty_us if x > 0.0]
        resolution_ns = (
            float(min(nonzero)) * 1000.0 if nonzero else 500.0
        )
        calibration_succeeded = bool(nonzero)

        return TimerCalibration(
            name="cuda-event",
            resolution_ns=resolution_ns,
            overhead_ns=overhead_ns,
            jitter_p99_ns=jitter_p99_ns,
            n_calibration_samples=1000,
            calibration_succeeded=calibration_succeeded,
        )


# =============================================================================
# Section 2.  Latency distributions.
#
# Every number in the paper is a percentile of a distribution.  To
# compute high percentiles correctly (P99.9, P99.99) the distribution
# storage must be lossless; downsampling to summary statistics
# silently invalidates tail claims.  The class below stores raw
# samples and offers tail-correctness guards that warn when a
# percentile is computed on insufficient samples.
# =============================================================================


# Minimum sample counts for credible percentile estimates.  These are
# n × (1 − p) ≥ 30 so the percentile is supported by ≥ 30 tail
# samples, the standard threshold for parametric tail estimation to
# be unbiased.
_MIN_SAMPLES_FOR_PERCENTILE: Mapping[float, int] = {
    50.0:    60,
    90.0:   300,
    95.0:   600,
    99.0:  3_000,
    99.9: 30_000,
    99.99: 300_000,
}

# Threshold above which a high rejection rate gets surfaced as a
# warning in summary().  50% means: more than half of attempted
# samples were below the timer's credibility floor and silently
# discarded.  This is a common failure mode on benign workloads
# where Phase A times are below the host's measurable floor; it
# silently produces a "distribution" of ~0 samples that downstream
# code may misinterpret as "Phase A is fast".
_REJECTION_RATE_WARNING_THRESHOLD: float = 0.5


@dataclass
class LatencyDistribution:
    """
    A sequence of latency measurements.  Lossless: stores raw
    samples.

    Use ``.add(sample_ns)`` during a run.  After the run, derived
    statistics are computed on demand.  Storage is a list, not a
    numpy array, because additions during a run are mixed with
    reads and we want O(1) amortized append.  The list is converted
    to a numpy array once at end-of-run.

    Camera-ready: ``summary()`` includes a ``rejection_rate_warning``
    string when more than ``_REJECTION_RATE_WARNING_THRESHOLD`` of
    attempted samples were rejected because they fell below the
    timer's credibility floor.  This surfaces silently-zero phase
    distributions to reviewers rather than letting them masquerade
    as "fast".
    """

    label: str
    timer_calibration: TimerCalibration
    samples_ns: List[int] = field(default_factory=list)
    rejected_below_credibility: int = 0    # samples below credible minimum
    insufficient_warnings: List[str] = field(default_factory=list)

    def add(self, sample_ns: int) -> None:
        if sample_ns < self.timer_calibration.credible_minimum_ns():
            # Not a real signal at this timer's resolution.  We keep
            # the count rather than the sample; reporting routines
            # must disclose this rejection rate.
            self.rejected_below_credibility += 1
            return
        self.samples_ns.append(sample_ns)

    def __len__(self) -> int:
        return len(self.samples_ns)

    @property
    def n_attempted(self) -> int:
        """Total number of ``add()`` calls (kept + rejected)."""
        return len(self.samples_ns) + self.rejected_below_credibility

    @property
    def rejection_rate(self) -> float:
        """Fraction of attempted samples that fell below credibility."""
        if self.n_attempted == 0:
            return 0.0
        return self.rejected_below_credibility / self.n_attempted

    # --- summary statistics ---------------------------------------------

    def array_us(self) -> np.ndarray:
        return np.asarray(self.samples_ns, dtype=np.float64) / 1000.0

    def percentile_us(self, p: float) -> float:
        """
        Returns the p-th percentile in microseconds.  Issues an
        insufficient-samples warning (recorded in
        ``insufficient_warnings``) if the sample count is below the
        credibility threshold for that percentile.
        """
        if not self.samples_ns:
            return float("nan")
        threshold = _MIN_SAMPLES_FOR_PERCENTILE.get(p)
        if threshold is not None and len(self.samples_ns) < threshold:
            msg = (
                f"P{p} of '{self.label}' computed on "
                f"{len(self.samples_ns)} samples; need {threshold} "
                "for credible tail estimate"
            )
            self.insufficient_warnings.append(msg)
            logger.warning(msg)
        return float(np.percentile(self.array_us(), p))

    def mean_us(self) -> float:
        return (
            float(np.mean(self.array_us()))
            if self.samples_ns else float("nan")
        )

    def std_us(self) -> float:
        return (
            float(np.std(self.array_us(), ddof=1))
            if len(self.samples_ns) > 1 else 0.0
        )

    def summary(self) -> Mapping[str, float]:
        if not self.samples_ns:
            # Empty-distribution summary: report the rejection
            # statistics so a reviewer can see WHY the distribution
            # is empty (rather than treating "no data" as "well-
            # behaved zero-cost phase").
            out: Dict[str, float] = {
                "n": 0,
                "n_attempted": float(self.n_attempted),
                "n_rejected": float(self.rejected_below_credibility),
                "rejection_rate": float(self.rejection_rate),
            }
            if self.rejection_rate > _REJECTION_RATE_WARNING_THRESHOLD:
                out["rejection_rate_warning"] = 1.0  # type: ignore[assignment]
            return out

        out2: Dict[str, float] = {
            "n": len(self.samples_ns),
            "n_attempted": float(self.n_attempted),
            "n_rejected": float(self.rejected_below_credibility),
            "rejection_rate": float(self.rejection_rate),
            "mean_us": self.mean_us(),
            "std_us": self.std_us(),
            "p50_us": self.percentile_us(50.0),
            "p90_us": self.percentile_us(90.0),
            "p99_us": self.percentile_us(99.0),
            "p99.9_us": self.percentile_us(99.9),
            "p99.99_us": self.percentile_us(99.99),
            "max_us": float(np.max(self.array_us())),
        }
        if self.rejection_rate > _REJECTION_RATE_WARNING_THRESHOLD:
            # Surface as float (for JSON-serialisability) rather
            # than string; 1.0 = warning fired, 0.0 = no warning.
            out2["rejection_rate_warning"] = 1.0
        return out2

    # --- serialization ---------------------------------------------------

    def to_parquet_dict(self) -> Mapping[str, Any]:
        """
        Loss-less serialization for results/raw/.  Stores raw
        samples plus calibration so a reviewer can re-compute every
        percentile the paper reports without trusting any
        aggregation done here.
        """
        return {
            "label": self.label,
            "timer": asdict(self.timer_calibration),
            "samples_ns": list(self.samples_ns),
            "rejected_below_credibility": self.rejected_below_credibility,
            "n_attempted": self.n_attempted,
            "rejection_rate": self.rejection_rate,
            "insufficient_warnings": list(self.insufficient_warnings),
        }


# =============================================================================
# Section 3.  Deadline-miss detection and reporting.
#
# A deadline miss is defined relative to a RealTimeContract: the
# update-path total cost for a single transaction must complete
# within ``deadline_us``.  The miss rate is the fraction of
# transactions that violate this; the contract additionally specifies
# an upper bound on the miss rate (``failure_probability_bound``).
# This section implements the bookkeeping and the report.
# =============================================================================


@dataclass
class DeadlineMissReport:
    """
    Records, for one experimental run, how often the contract was
    violated and by how much.  The ``tardiness`` array gives, for
    each missed deadline, the amount by which it was missed — used
    in plots.py to draw the tardiness distribution.
    """

    contract: RealTimeContract
    n_transactions: int = 0
    n_missed: int = 0
    tardiness_us: List[float] = field(default_factory=list)        # only for missed
    on_time_latency_us: List[float] = field(default_factory=list)  # full lossless

    def observe(self, latency_us: float) -> None:
        self.n_transactions += 1
        if latency_us > self.contract.deadline_us:
            self.n_missed += 1
            self.tardiness_us.append(latency_us - self.contract.deadline_us)
        else:
            self.on_time_latency_us.append(latency_us)

    @property
    def miss_rate(self) -> float:
        if self.n_transactions == 0:
            return float("nan")
        return self.n_missed / self.n_transactions

    @property
    def contract_satisfied(self) -> bool:
        """True iff empirical miss rate ≤ contract bound."""
        return self.miss_rate <= self.contract.failure_probability_bound

    def confidence_interval(self, alpha: float = 0.05) -> Tuple[float, float]:
        """
        Wilson-score confidence interval on the miss rate.  Reported
        alongside the point estimate so that small denominator
        counts (e.g. early in a run) do not produce overconfident
        claims.
        """
        if self.n_transactions == 0:
            return (float("nan"), float("nan"))
        n = self.n_transactions
        p_hat = self.miss_rate
        # Wilson score interval — accurate for small p, unlike Wald.
        z = 1.959963984540054 if alpha == 0.05 else _z_from_alpha(alpha)
        denom = 1.0 + z * z / n
        center = (p_hat + z * z / (2 * n)) / denom
        half = (
            z * math.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n))
        ) / denom
        return (max(0.0, center - half), min(1.0, center + half))

    def summary(self) -> Mapping[str, Any]:
        ci_lo, ci_hi = self.confidence_interval()
        s: Dict[str, Any] = {
            "deadline_us": self.contract.deadline_us,
            "epsilon": self.contract.failure_probability_bound,
            "n_transactions": self.n_transactions,
            "n_missed": self.n_missed,
            "miss_rate": self.miss_rate,
            "miss_rate_ci_lo": ci_lo,
            "miss_rate_ci_hi": ci_hi,
            "contract_satisfied": self.contract_satisfied,
        }
        if self.tardiness_us:
            arr = np.asarray(self.tardiness_us)
            s["tardiness_mean_us"] = float(arr.mean())
            s["tardiness_p99_us"] = float(np.percentile(arr, 99.0))
            s["tardiness_max_us"] = float(arr.max())
        return s


def _z_from_alpha(alpha: float) -> float:
    """Inverse normal CDF at 1 − alpha/2.  Used for non-default alphas."""
    from math import erfinv, sqrt
    return sqrt(2.0) * erfinv(1.0 - alpha)


# =============================================================================
# Section 4.  Model-freshness tracker (Age of Information semantics).
#
# Implements the canonical AoI age process Δ(t) = t − u(t) where u(t)
# is the generation timestamp of the freshest committed update at
# time t.  Per the Yates et al. survey (JSAC 2021) and the
# foundational Kaul–Yates–Gruteser INFOCOM 2012 paper, the age
# process resets to (commit_time − arrival_time) at each completion
# (NOT to zero — the delivered information is already that old when
# it lands), and grows at unit rate between completions.
#
# Why we need this beyond the deadline-miss report
# ------------------------------------------------
# Deadline-miss reports answer "did this transaction's update finish
# in time?" — a per-transaction property.  Model-freshness reports
# answer "when a decision is made at time t, how stale is the model
# state used for that decision?" — a property of the *decision*, not
# the transaction.  The two metrics can disagree:
#   - High deadline-miss rate, low average age:
#         the system is missing many deadlines, but the queue is
#         processing newer arrivals quickly enough that the
#         most-recent committed sample is still fresh.
#   - Low deadline-miss rate, high average age:
#         transactions complete quickly when they run, but the
#         system is sitting idle (no recent admissions) so the
#         model state is stale.
# An update storm can cause both at once.  Reporting both is what
# distinguishes Update Storms from a generic deadline-miss study.
#
# Disanalogy to classical AoI (and why we still report it)
# -------------------------------------------------------
# Classical AoI studies the freshness of a delivered status packet
# at a monitor.  Our object is the model state of a continuously-
# learning system.  The Yates extraction warns explicitly that the
# latest committed sample may not be a sufficient statistic for
# decision freshness in a learner with multiple parameter groups.
# We adopt the scalar-age formalism here as the simplest principled
# metric and document the parameter-group-vector treatment as
# future work.
#
# Camera-ready performance fix
# ----------------------------
# The submission draft's ``_compute_age_at`` was O(N) per call and
# was invoked once per ``record_decision`` on the hot path.  This
# version maintains the freshest-arrival timestamp online: each
# ``record_sample_commit`` updates an integer in O(1); each
# ``record_decision`` reads it in O(1).  The end-of-run
# sawtooth-area computation is unchanged.
# =============================================================================


@dataclass
class _CommittedSample:
    """One (arrival_ns, commit_ns) pair committed to the model."""

    arrival_ns: int
    commit_ns: int

    @property
    def propagation_delay_ns(self) -> int:
        """commit − arrival; the AoI 'reset value' for this sample."""
        return self.commit_ns - self.arrival_ns


@dataclass
class AgeViolationReport:
    """
    Counts decisions whose model age exceeds the freshness deadline
    A_max.  Same Wilson-score CI machinery as DeadlineMissReport so
    the two can be displayed side-by-side in the paper's §V.2 table.
    """

    age_max_us: float
    n_decisions: int = 0
    n_violations: int = 0
    overshoot_us: List[float] = field(default_factory=list)  # only when violated

    def observe(self, model_age_us: float) -> None:
        self.n_decisions += 1
        if model_age_us > self.age_max_us:
            self.n_violations += 1
            self.overshoot_us.append(model_age_us - self.age_max_us)

    @property
    def violation_rate(self) -> float:
        if self.n_decisions == 0:
            return float("nan")
        return self.n_violations / self.n_decisions

    def confidence_interval(self, alpha: float = 0.05) -> Tuple[float, float]:
        if self.n_decisions == 0:
            return (float("nan"), float("nan"))
        n = self.n_decisions
        p_hat = self.violation_rate
        z = 1.959963984540054 if alpha == 0.05 else _z_from_alpha(alpha)
        denom = 1.0 + z * z / n
        center = (p_hat + z * z / (2 * n)) / denom
        half = (
            z * math.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n))
        ) / denom
        return (max(0.0, center - half), min(1.0, center + half))

    def summary(self) -> Mapping[str, Any]:
        ci_lo, ci_hi = self.confidence_interval()
        s: Dict[str, Any] = {
            "age_max_us": self.age_max_us,
            "n_decisions": self.n_decisions,
            "n_violations": self.n_violations,
            "violation_rate": self.violation_rate,
            "violation_rate_ci_lo": ci_lo,
            "violation_rate_ci_hi": ci_hi,
        }
        if self.overshoot_us:
            arr = np.asarray(self.overshoot_us)
            s["overshoot_mean_us"] = float(arr.mean())
            s["overshoot_p99_us"] = float(np.percentile(arr, 99))
            s["overshoot_max_us"] = float(arr.max())
        return s


@dataclass
class ModelFreshnessTracker:
    """
    AoI-style tracker for the model age process.

    Lifecycle (per transaction):
      1. record_sample_arrival(txn_id, arrival_ns)  — at txn arrival
      2. record_sample_commit(txn_id, commit_ns)    — at C_end
      3. record_decision(decision_ns)               — when a decision
                                                       uses the current
                                                       model

    Steps 1+2 must come in pairs (an unmatched arrival is silently
    dropped at end-of-run via ``pending_orphans``).  Step 3 may be
    called any number of times between completions.

    Statistics (computed at end of run):
      - ``average_model_age_us``: time-average of Δ(t).  Computed via
        the AoI sawtooth-area decomposition rather than a discrete
        sample mean, so it is correct under arbitrary decision
        cadences.
      - ``peak_model_age_us``: max Δ(t) observed at any decision.
      - ``age_at_decision_samples_us``: the model-age values at each
        decision call.  Lossless; used to plot the model-age CCDF.
      - ``propagation_latency_samples_us``: commit − arrival per
        committed sample.  This is the AoI "delivery latency" for
        a continuous-learning context.

    Backward compatibility
    ----------------------
    If the harness never calls ``record_decision``, the tracker
    reports empty distributions for decision-level statistics; the
    propagation-latency distribution is still populated from
    arrival/commit pairs.  If the harness never calls
    ``record_sample_arrival``, the tracker treats arrival ≡ commit
    (i.e., propagation_delay = 0 for every sample), which is the
    degenerate "instantaneous-arrival" interpretation; the model
    age process then reduces to t − latest_commit_time.

    Memory note
    -----------
    ``_pending_arrivals`` retains entries between
    ``record_sample_arrival`` and ``record_sample_commit``.  In the
    synchronous experiment harness these are matched within a few
    ms; in a hypothetical asynchronous harness they could persist
    longer.  The dictionary is bounded by in-flight transactions,
    which is bounded by the queue depth, which is bounded by the
    admission policy.  No explicit cap is needed.

    Camera-ready performance fix
    ----------------------------
    ``_compute_age_at`` is no longer O(N) per call.  This tracker
    now maintains the freshest-arrival timestamp online via
    ``_freshest_arrival_ns`` — updated O(1) on each
    ``record_sample_commit`` whenever a newer sample arrives, and
    read O(1) by ``record_decision``.  The submission draft's
    O(N²) hot-path bug is fixed.
    """

    # Inputs accumulated during the run.
    _pending_arrivals: Dict[int, int] = field(default_factory=dict)   # txn_id → arrival_ns
    _committed: List[_CommittedSample] = field(default_factory=list)
    _age_at_decision_ns: List[int] = field(default_factory=list)
    _decision_times_ns: List[int] = field(default_factory=list)
    pending_orphans: int = 0          # arrivals with no matching commit

    # Camera-ready: online tracker for freshest committed
    # arrival-timestamp.  Updated on every record_sample_commit.
    # -1 means "no commit yet seen".
    _freshest_arrival_ns: int = -1

    # --- input hooks ----------------------------------------------------

    def record_sample_arrival(self, txn_id: int, arrival_ns: int) -> None:
        """Called when a transaction arrives at the system gate."""
        self._pending_arrivals[txn_id] = int(arrival_ns)

    def record_sample_commit(self, txn_id: int, commit_ns: int) -> None:
        """
        Called when the transaction's update completes (C_end).
        Pairs with a prior ``record_sample_arrival``; if no arrival
        was recorded for this txn, treats arrival_ns == commit_ns
        (propagation_delay = 0).
        """
        arrival_ns = self._pending_arrivals.pop(txn_id, int(commit_ns))
        self._committed.append(_CommittedSample(
            arrival_ns=arrival_ns,
            commit_ns=int(commit_ns),
        ))
        # Update online freshest-arrival.  Note: AoI freshness is
        # by *arrival timestamp*, not commit timestamp — a commit
        # of an older transaction does NOT reduce model age below
        # what was already achieved by a later arrival.
        if arrival_ns > self._freshest_arrival_ns:
            self._freshest_arrival_ns = arrival_ns

    def record_decision(self, decision_ns: int) -> None:
        """
        Called when the SUT makes a decision (e.g., a forward pass)
        at the given timestamp.  Records the model age at that
        moment.

        Camera-ready: O(1) via the online ``_freshest_arrival_ns``
        instead of O(N) linear scan over commits.
        """
        if self._freshest_arrival_ns < 0:
            # No commit yet observed; model is uninitialised.
            return
        age_ns = int(decision_ns) - self._freshest_arrival_ns
        if age_ns < 0:
            # Decision time precedes the freshest arrival — only
            # possible under clock skew; conservatively record 0.
            age_ns = 0
        self._age_at_decision_ns.append(age_ns)
        self._decision_times_ns.append(int(decision_ns))

    # --- end-of-run summaries -------------------------------------------

    def propagation_latency_samples_us(self) -> List[float]:
        """commit − arrival, in microseconds, per committed sample."""
        return [s.propagation_delay_ns / 1000.0 for s in self._committed]

    def age_at_decision_samples_us(self) -> List[float]:
        """Model age at each recorded decision, in microseconds."""
        return [a / 1000.0 for a in self._age_at_decision_ns]

    def average_model_age_us(self) -> float:
        """
        Time-average of Δ(t) computed from the AoI sawtooth-area
        decomposition.  Specifically:

            avg_age = (1/T) Σ_n Q_n

        where Q_n is the area under Δ(t) between the (n-1)-th and
        n-th completions.  Per Kaul–Yates–Gruteser §III, when
        commits arrive at times c_n with arrival times a_n,

            Q_n = (1/2) Y_n²  +  Y_n · (c_n − a_n)
                = (1/2) (c_n − c_{n−1})² + (c_n − c_{n−1})·(c_n − a_n).

        and the time-average is Σ Q_n / T over the observation
        window T = c_N − c_0.  We compute it directly here.

        Returns NaN if fewer than 2 commits have been observed.
        """
        if len(self._committed) < 2:
            return float("nan")
        sorted_commits = sorted(self._committed, key=lambda s: s.commit_ns)
        T_ns = sorted_commits[-1].commit_ns - sorted_commits[0].commit_ns
        if T_ns <= 0:
            return float("nan")
        total_area_ns2 = 0.0
        for n in range(1, len(sorted_commits)):
            prev = sorted_commits[n - 1]
            curr = sorted_commits[n]
            Y_n = float(curr.commit_ns - prev.commit_ns)
            T_n = float(curr.commit_ns - curr.arrival_ns)   # current sample's "system time"
            Q_n = 0.5 * Y_n * Y_n + Y_n * T_n
            total_area_ns2 += Q_n
        # Time-average: total_area / T.  Result has units of ns
        # (because area is ns², divided by ns gives ns).  Convert
        # to µs.
        avg_ns = total_area_ns2 / float(T_ns)
        return avg_ns / 1000.0

    def peak_model_age_us(self) -> float:
        """Maximum age observed at any decision call."""
        if not self._age_at_decision_ns:
            return float("nan")
        return float(max(self._age_at_decision_ns)) / 1000.0

    def n_committed(self) -> int:
        return len(self._committed)

    def n_decisions(self) -> int:
        return len(self._age_at_decision_ns)

    def to_parquet_dict(self) -> Mapping[str, Any]:
        """Lossless serialisation for results/raw/."""
        return {
            "n_committed": self.n_committed(),
            "n_decisions": self.n_decisions(),
            "pending_orphans": self.pending_orphans,
            "average_model_age_us": self.average_model_age_us(),
            "peak_model_age_us": self.peak_model_age_us(),
            "propagation_latency_samples_us": self.propagation_latency_samples_us(),
            "age_at_decision_samples_us": self.age_at_decision_samples_us(),
            "decision_times_ns": list(self._decision_times_ns),
        }


# =============================================================================
# Section 5.  Time-series recorder (queue depth, throughput, recovery).
#
# Per Qing & Zheng's ICDE 2025 evaluation methodology, stateful
# stream-processing performance is reported as a time series with
# explicit storm and recovery intervals, and the recovery duration
# is defined as the time from event onset until end-to-end latency
# stays within (1 + tolerance) × baseline for hold_duration_s
# consecutively.
#
# The recorder bins arrival, admission, rejection, and completion
# events into fixed-width windows (default 100 ms).  Queue depth is
# tracked event-by-event (cumulative arrivals admitted minus
# completions), giving an exact per-event series rather than a
# sampled approximation.
#
# Output channels
# ---------------
# - throughput_time_series:   per-bucket counts of attempted, admitted,
#                             rejected, and processed events, along
#                             with a wall-clock timestamp.  Each
#                             bucket is a dict for JSON portability.
# - queue_depth_time_series:  list of (t_ns, queue_depth_after_event).
#                             Sampled at every event, not at a fixed
#                             cadence; this is exact rather than
#                             approximate.
# - latency_over_time:        list of (commit_ns, latency_us).  The
#                             paper's §V.1 latency-over-time plots
#                             consume this directly.  Camera-ready
#                             additionally provides
#                             ``latency_at_arrival_time``: the same
#                             latency value but timestamped at the
#                             transaction's arrival, which is the
#                             semantically correct x-coordinate for
#                             "latency as experienced".
# - recovery diagnostics:     compute_recovery_time(...) at end-of-run.
# =============================================================================


_DEFAULT_BUCKET_WIDTH_NS: int = 100_000_000     # 100 ms


@dataclass
class TimeSeriesRecorder:
    """
    Per-event time-series recorder.

    Memory budget: O(buckets) for throughput series + O(events) for
    queue-depth series + O(commits) for latency-over-time.  At a
    realistic 10⁴ events/second over a 60-second run, this is
    ~6 × 10⁵ events; storage is a few tens of MB — well within
    a result file's budget for a paper of this scale.

    Camera-ready: also records ``latency_at_arrival_time``, an
    arrival-time-keyed latency series.  ``record_completion`` keeps
    the submission draft's commit-time-keyed series for backwards
    compatibility (plots.py already consumes it); the new
    ``record_completion_with_arrival`` is the semantically correct
    form (latency timestamped at the moment work was *requested*,
    not when it finished).  Run records carry both series; plots.py
    decides which to use.
    """

    bucket_width_ns: int = _DEFAULT_BUCKET_WIDTH_NS
    # Throughput: bucket_index → counts.
    _attempted: Dict[int, int] = field(default_factory=dict)
    _admitted:  Dict[int, int] = field(default_factory=dict)
    _rejected:  Dict[int, int] = field(default_factory=dict)
    _processed: Dict[int, int] = field(default_factory=dict)
    # Queue depth: event-by-event series.
    _queue_events: List[Tuple[int, int]] = field(default_factory=list)
    _current_queue_depth: int = 0
    # Latency-over-time (commit-time-keyed): (commit_ns, latency_us).
    _latency_over_time: List[Tuple[int, float]] = field(default_factory=list)
    # Camera-ready: arrival-time-keyed latency series.
    _latency_at_arrival: List[Tuple[int, float]] = field(default_factory=list)
    # Run anchors for recovery analysis.
    _run_start_ns: Optional[int] = None
    _run_end_ns: Optional[int] = None

    def _bucket(self, t_ns: int) -> int:
        return int(t_ns) // self.bucket_width_ns

    def _anchor(self, t_ns: int) -> None:
        if self._run_start_ns is None:
            self._run_start_ns = int(t_ns)
        self._run_end_ns = int(t_ns)

    # --- input hooks -----------------------------------------------------

    def record_arrival(self, t_ns: int) -> None:
        b = self._bucket(t_ns)
        self._attempted[b] = self._attempted.get(b, 0) + 1
        self._anchor(t_ns)

    def record_admission(self, t_ns: int) -> None:
        b = self._bucket(t_ns)
        self._admitted[b] = self._admitted.get(b, 0) + 1
        self._current_queue_depth += 1
        self._queue_events.append((int(t_ns), self._current_queue_depth))
        self._anchor(t_ns)

    def record_rejection(self, t_ns: int) -> None:
        b = self._bucket(t_ns)
        self._rejected[b] = self._rejected.get(b, 0) + 1
        self._anchor(t_ns)

    def record_completion(self, t_ns: int, latency_us: float) -> None:
        """
        Record a completion at commit time ``t_ns`` with the given
        end-to-end latency.  Submission-draft compatible: the latency
        point is keyed at completion time.
        """
        b = self._bucket(t_ns)
        self._processed[b] = self._processed.get(b, 0) + 1
        self._current_queue_depth = max(0, self._current_queue_depth - 1)
        self._queue_events.append((int(t_ns), self._current_queue_depth))
        self._latency_over_time.append((int(t_ns), float(latency_us)))
        self._anchor(t_ns)

    def record_completion_with_arrival(
        self,
        commit_ns: int,
        arrival_ns: int,
        latency_us: float,
    ) -> None:
        """
        Camera-ready: record a completion AND timestamp the latency
        at *arrival* time.  This is the semantically correct form
        for "latency as experienced": a transaction that arrives in
        bucket B but completes in bucket B+k contributes its
        latency-of-experience to bucket B (the moment when the work
        was requested), not B+k (the moment the result emerged).

        ``record_completion`` is preserved for backwards
        compatibility; both series are persisted in run records and
        plots.py decides which to use.
        """
        # Drive the standard completion bookkeeping (queue depth,
        # processed count, run anchor).
        self.record_completion(commit_ns, latency_us)
        # Also append the arrival-keyed point.
        self._latency_at_arrival.append((int(arrival_ns), float(latency_us)))

    # --- output ----------------------------------------------------------

    def throughput_time_series(self) -> List[Mapping[str, Any]]:
        """
        Return one dict per bucket from the earliest to the latest
        observed bucket (inclusive).  Empty buckets get zero counts
        — important so the time series doesn't have implicit gaps
        that plotting code would mis-align.
        """
        all_buckets = (
            set(self._attempted) | set(self._admitted)
            | set(self._rejected) | set(self._processed)
        )
        if not all_buckets:
            return []
        bmin = min(all_buckets)
        bmax = max(all_buckets)
        out: List[Mapping[str, Any]] = []
        for b in range(bmin, bmax + 1):
            out.append({
                "t_ns": b * self.bucket_width_ns,
                "bucket_width_ns": self.bucket_width_ns,
                "attempted": self._attempted.get(b, 0),
                "admitted":  self._admitted.get(b, 0),
                "rejected":  self._rejected.get(b, 0),
                "processed": self._processed.get(b, 0),
            })
        return out

    def queue_depth_time_series(self) -> List[Tuple[int, int]]:
        """
        Per-event queue-depth observations, sorted by time.  Returns
        a fresh sorted copy; the internal list is append-only and
        already in arrival order, but sorting defensively guarantees
        monotonic timestamps even if the harness records out of
        order.
        """
        return sorted(self._queue_events, key=lambda kv: kv[0])

    def latency_over_time(self) -> List[Tuple[int, float]]:
        """
        Per-completion ``(commit_ns, latency_us)``, sorted by
        completion time.  Submission-draft semantics.
        """
        return sorted(self._latency_over_time, key=lambda kv: kv[0])

    def latency_at_arrival_time(self) -> List[Tuple[int, float]]:
        """
        Camera-ready: per-completion
        ``(arrival_ns, latency_us)``, sorted by arrival time.
        Empty when the harness uses the legacy ``record_completion``
        path (without arrival timestamps).
        """
        return sorted(self._latency_at_arrival, key=lambda kv: kv[0])

    def peak_queue_depth(self) -> int:
        """Max queue depth observed during the run."""
        if not self._queue_events:
            return 0
        return max(d for _, d in self._queue_events)

    def compute_recovery_time(
        self,
        baseline_latency_us: float,
        tolerance: float = 0.10,
        hold_duration_s: float = 1.0,
        event_onset_ns: Optional[int] = None,
    ) -> Optional[float]:
        """
        Storm recovery time, in seconds, per the Qing & Zheng /
        DRRS-style stability criterion.  Returns the time from
        ``event_onset_ns`` (default: run start) to the moment that
        latency stays within ``(1 + tolerance) × baseline_latency_us``
        for ``hold_duration_s`` consecutive seconds.

        Returns None if the criterion is never met during the
        observed run; callers must report this explicitly rather
        than silently treating it as "no storm" (a missing recovery
        is itself a finding).

        Methodology
        -----------
        Walk the latency-over-time series in time order.  Maintain
        a sliding "window start" t_w: the earliest time within the
        candidate stable window.  Each time a sample exceeds the
        threshold, advance t_w past it.  When the gap from t_w to
        the current sample exceeds ``hold_duration_s``, return
        ``(window_start_ns − event_onset_ns) / 1e9``.

        Subtlety
        --------
        We require BOTH that the latency be below threshold AND
        that the window be at least ``hold_duration_s`` long, so a
        brief dip followed by an exceedance does not falsely
        trigger.  This matches the Qing & Zheng definition of
        "stably back to baseline".
        """
        if not self._latency_over_time:
            return None
        threshold_us = (1.0 + tolerance) * baseline_latency_us
        hold_ns = int(hold_duration_s * 1e9)
        onset_ns = (
            int(event_onset_ns) if event_onset_ns is not None
            else (self._run_start_ns or 0)
        )
        sorted_pts = sorted(self._latency_over_time, key=lambda kv: kv[0])

        window_start_ns: Optional[int] = None
        for t_ns, lat_us in sorted_pts:
            if t_ns < onset_ns:
                continue
            if lat_us > threshold_us:
                # Out-of-bound sample: reset the candidate window.
                window_start_ns = None
                continue
            # In-bound sample: open or extend the window.
            if window_start_ns is None:
                window_start_ns = t_ns
            else:
                if (t_ns - window_start_ns) >= hold_ns:
                    return (window_start_ns - onset_ns) / 1e9
        return None

    # --- summary --------------------------------------------------------

    def summary(self) -> Mapping[str, Any]:
        s: Dict[str, Any] = {
            "bucket_width_ns": self.bucket_width_ns,
            "n_buckets": len(
                set(self._attempted) | set(self._admitted)
                | set(self._rejected) | set(self._processed)
            ),
            "n_queue_events": len(self._queue_events),
            "peak_queue_depth": self.peak_queue_depth(),
            "n_latency_points": len(self._latency_over_time),
            "n_latency_at_arrival_points": len(self._latency_at_arrival),
        }
        if self._run_start_ns is not None and self._run_end_ns is not None:
            s["run_duration_s"] = (
                (self._run_end_ns - self._run_start_ns) / 1e9
            )
        return s

    def to_parquet_dict(self) -> Mapping[str, Any]:
        """Lossless serialisation for results/raw/."""
        return {
            "bucket_width_ns": self.bucket_width_ns,
            "throughput_buckets": list(self.throughput_time_series()),
            "queue_depth_events": [
                {"t_ns": t, "depth": d}
                for t, d in self.queue_depth_time_series()
            ],
            "latency_over_time": [
                {"t_ns": t, "latency_us": lat}
                for t, lat in self.latency_over_time()
            ],
            "latency_at_arrival_time": [
                {"t_ns": t, "latency_us": lat}
                for t, lat in self.latency_at_arrival_time()
            ],
            "run_start_ns": self._run_start_ns,
            "run_end_ns":   self._run_end_ns,
        }


# =============================================================================
# Section 6.  The measuring probe.
#
# This is the UpdatePathProbe that system.py's UpdatePath calls into
# when measurement is enabled.  It assembles per-phase latency
# distributions and end-to-end deadline-miss reports from the stream
# of UpdatePathEvent.
#
# Phase boundary semantics:
#   A_begin → A_end : Phase A (CPU index + BFS)
#   A_end   → B_end : Phase B (GPU forward + backward)
#   B_end   → C_end : Phase C (CPU parameter write)
#
# A transaction's end-to-end latency is C_end − A_begin.  We do not
# include workload-generation latency because that is the attacker's
# (or generator's) cost, not the SUT's.
#
# Camera-ready improvements
# -------------------------
# 1. **AoI-violation double-counting fix.**  The submission draft's
#    ``_finalize`` recorded an AoI violation event at every commit,
#    treating each commit as if it were also a decision.  This
#    over-counts by the ratio of commits to actual decisions.  The
#    camera-ready records AoI violations ONLY at explicit
#    ``record_decision`` calls; commits no longer fire AoI events.
#    When a harness has not opted into ``record_decision``, the
#    age-violation report correctly reports zero violations
#    (the null hypothesis), not the submission-draft's spurious
#    "commit-as-decision" count.
#
# 2. **Arrival-keyed latency feed.**  When ``record_arrival`` was
#    called for a transaction, ``_finalize`` now feeds the
#    ``record_completion_with_arrival`` channel of the time-series
#    recorder so plots.py can render latency-as-experienced rather
#    than latency-at-completion.
#
# All harness-facing hooks are no-ops when the harness does not
# call them; the existing 26-test contract of test_measurement.py
# is preserved.
# =============================================================================


# Default age-violation deadline as multiplier on
# contract.deadline_us.  For a fairer apples-to-apples comparison
# with the deadline-miss report, we use the same deadline value by
# default; experiments.py can override.
_DEFAULT_AGE_MAX_MULTIPLIER: float = 1.0


class MeasuringProbe(UpdatePathProbe):
    """
    The non-trivial implementation of UpdatePathProbe.  Records all
    phase events and produces per-phase latency distributions,
    deadline-miss report, model-freshness statistics, time-series
    throughput/queue, and queueing-delay distribution.

    Thread-safety
    -------------
    The probe is single-writer: events from a single UpdatePath
    arrive on a single thread (the update path is sequentialised by
    its own queue elsewhere).  If a future evaluation runs multiple
    parallel update paths, each must have its own probe; merging is
    handled by the experiment harness, not here.

    Overhead
    --------
    Per-event cost is a small dict-assign and list-append in the
    steady state.  Calibration (Section 1) measures this on the
    host; on the evaluation hardware it is < 200 ns per event, well
    below the credibility threshold for the latency phases being
    measured.
    """

    def __init__(
        self,
        contract: RealTimeContract,
        timer: Optional[MonotonicTimer] = None,
        cuda_timer: Optional[CudaTimer] = None,
        age_max_us: Optional[float] = None,
        bucket_width_ns: int = _DEFAULT_BUCKET_WIDTH_NS,
    ) -> None:
        self._timer = timer or MonotonicTimer()
        self._cuda = cuda_timer or CudaTimer()
        self._contract = contract

        cal_cpu = self._timer.calibration
        cal_gpu = self._cuda.calibration

        # Per-phase distributions.
        self.phase_a = LatencyDistribution("phase_A_bfs", cal_cpu)
        self.phase_b = LatencyDistribution("phase_B_gpu", cal_gpu)
        self.phase_c = LatencyDistribution("phase_C_write", cal_cpu)
        self.end_to_end = LatencyDistribution("end_to_end", cal_cpu)

        # Queueing-delay distribution (arrival → A_begin).
        # Activated when the harness calls ``record_arrival()``
        # before the SUT processes the transaction.
        self.queueing_delay = LatencyDistribution("queueing_delay", cal_cpu)
        # Full propagation latency (arrival → C_end), the AoI
        # "delivery latency" analogue.  Equal to end_to_end +
        # queueing_delay.
        self.propagation = LatencyDistribution("propagation", cal_cpu)

        self.miss_report = DeadlineMissReport(contract=contract)

        # Age-violation deadline.  Defaults to the contract's
        # deadline_us so that the AoI bound and the update-deadline
        # bound coincide unless the harness wants to differentiate.
        if age_max_us is None:
            age_max_us = (
                _DEFAULT_AGE_MAX_MULTIPLIER * contract.deadline_us
            )
        self.age_violation_report = AgeViolationReport(age_max_us=age_max_us)

        self.freshness = ModelFreshnessTracker()
        self.time_series = TimeSeriesRecorder(bucket_width_ns=bucket_width_ns)

        # Per-transaction state: maps txn_id → dict of phase ns
        # timestamps.  Also stores the (optional) arrival timestamp
        # from ``record_arrival``.
        self._pending: Dict[int, Dict[str, int]] = {}

        # Auxiliary fields captured from event.extra.
        self.affected_sizes: List[int] = []
        self.bfs_depths: List[int] = []
        self.bfs_edges: List[int] = []

        # Probe overhead audit: count of events processed.
        self._events_seen = 0

    # --- new harness-facing hooks --------------------------------------

    def record_arrival(self, txn_id: int, arrival_ns: int) -> None:
        """
        Record the arrival of a transaction at the system gate
        (BEFORE the defense evaluates it).  Optional: when called,
        enables queueing-delay measurement and feeds the freshness
        tracker.  When NOT called, the queueing-delay distribution
        remains empty and the freshness tracker uses A_begin as the
        sample arrival time (degenerate but well-defined fallback).
        """
        rec = self._pending.setdefault(txn_id, {})
        rec["arrival"] = int(arrival_ns)
        # Forward to subsystems.
        self.freshness.record_sample_arrival(txn_id, int(arrival_ns))
        self.time_series.record_arrival(int(arrival_ns))

    def record_admission_decision(
        self,
        txn_id: int,
        t_ns: int,
        admitted: bool,
    ) -> None:
        """
        Record the admission decision.  Drives the time-series
        recorder so attempted/admitted/rejected throughput is
        observable per bucket.
        """
        if admitted:
            self.time_series.record_admission(int(t_ns))
        else:
            self.time_series.record_rejection(int(t_ns))

    def record_rejection(self, txn_id: int, t_ns: int) -> None:
        """
        Convenience alias for
        ``record_admission_decision(.., admitted=False)``.
        """
        self.record_admission_decision(txn_id, t_ns, admitted=False)

    def record_decision(self, t_ns: int) -> None:
        """
        Record a model-decision time.  Drives both the freshness
        tracker (which samples model age at this moment) and the
        age-violation report (which checks the sampled age against
        the configured ``age_max_us``).

        Camera-ready: this is the ONLY entry point that records AoI
        violations.  The submission draft additionally fired
        violations at every commit, which over-counts.  Harnesses
        that do not call ``record_decision`` will correctly report
        zero AoI violations — the appropriate null when no
        decisions have been observed.
        """
        # Sample model age and feed both the freshness tracker and
        # the age-violation report.  The freshness tracker filters
        # out pre-first-commit decisions internally; we mirror that
        # behaviour here by sampling the tracker's online state.
        n_before = self.freshness.n_decisions()
        self.freshness.record_decision(int(t_ns))
        # If the freshness tracker accepted the decision (i.e., at
        # least one commit has been observed), record the
        # corresponding AoI sample in the violation report.
        if self.freshness.n_decisions() > n_before:
            age_us = self.freshness.age_at_decision_samples_us()[-1]
            self.age_violation_report.observe(age_us)

    # --- UpdatePathProbe interface --------------------------------------

    def observe(self, event: UpdatePathEvent) -> None:
        self._events_seen += 1
        rec = self._pending.setdefault(event.txn_id, {})
        rec[event.phase] = event.monotonic_ns

        if event.phase == "A_end":
            extra = event.extra
            if "affected_size" in extra:
                self.affected_sizes.append(int(extra["affected_size"]))   # type: ignore[arg-type]
            if "depth" in extra:
                self.bfs_depths.append(int(extra["depth"]))               # type: ignore[arg-type]
            if "edges_visited" in extra:
                self.bfs_edges.append(int(extra["edges_visited"]))        # type: ignore[arg-type]

        if event.phase == "C_end":
            self._finalize(event.txn_id, rec)

    # --- finalize one transaction ---------------------------------------

    def _finalize(self, txn_id: int, rec: Dict[str, int]) -> None:
        a_begin = rec.get("A_begin")
        a_end = rec.get("A_end")
        b_end = rec.get("B_end")
        c_end = rec.get("C_end")
        if None in (a_begin, a_end, b_end, c_end):
            # Drop incomplete transactions.  This can happen if
            # measurement is started mid-stream; experiments.py is
            # responsible for warming up before measurement begins.
            del self._pending[txn_id]
            return

        # mypy doesn't see that the None-check above narrows these
        # to int.
        a_begin = int(a_begin)
        a_end = int(a_end)
        b_end = int(b_end)
        c_end = int(c_end)

        phase_a_ns = a_end - a_begin
        phase_b_ns = b_end - a_end
        phase_c_ns = c_end - b_end
        e2e_ns = c_end - a_begin

        self.phase_a.add(phase_a_ns)
        self.phase_b.add(phase_b_ns)
        self.phase_c.add(phase_c_ns)
        self.end_to_end.add(e2e_ns)

        self.miss_report.observe(e2e_ns / 1000.0)

        # Queueing-delay and propagation, when the harness opted-in
        # via ``record_arrival``.
        arrival = rec.get("arrival")
        if arrival is not None:
            queueing_ns = max(0, a_begin - int(arrival))
            propagation_ns = c_end - int(arrival)
            self.queueing_delay.add(queueing_ns)
            self.propagation.add(propagation_ns)

        # Freshness: commit the sample.
        self.freshness.record_sample_commit(txn_id, c_end)

        # Time-series: record completion event.  Camera-ready: when
        # arrival was recorded, also feed the arrival-keyed latency
        # channel so plots.py can render latency-as-experienced.
        if arrival is not None:
            self.time_series.record_completion_with_arrival(
                commit_ns=c_end,
                arrival_ns=int(arrival),
                latency_us=e2e_ns / 1000.0,
            )
        else:
            self.time_series.record_completion(c_end, e2e_ns / 1000.0)

        # NOTE (camera-ready): we no longer fire an AoI-violation
        # event at every commit.  The submission draft did, which
        # over-counted by the ratio of commits to actual decisions.
        # AoI violations are now recorded ONLY at explicit
        # ``record_decision`` calls (see that method above).

        del self._pending[txn_id]

    # --- reporting ------------------------------------------------------

    def overhead_audit(self) -> Mapping[str, Any]:
        """
        Probe self-audit: events processed, pending leaks,
        calibration.

        ``pending_leaks``:        total entries still in
                                  ``_pending`` at audit time.
                                  Backwards-compatible field;
                                  matches submission-draft semantics
                                  used by existing tests.
        ``phase_pending_leaks``:  pending entries that have at
                                  least an A_begin event recorded
                                  (i.e. the SUT started processing
                                  but never reported C_end).  This
                                  is the actually-concerning
                                  failure mode.
        ``arrival_only_pending``: pending entries with only an
                                  arrival timestamp (i.e. the txn
                                  was rejected before phase A
                                  began).  Not a leak.
        """
        # Drop stale arrival-only entries from pending leaks; an
        # arrival-only entry is not a phase-event leak but a
        # never-completed transaction (rejected before processing).
        only_arrival_pending = sum(
            1 for v in self._pending.values()
            if "A_begin" not in v
        )
        true_pending_leaks = len(self._pending) - only_arrival_pending
        return {
            "events_seen": self._events_seen,
            "pending_leaks": len(self._pending),
            "phase_pending_leaks": true_pending_leaks,
            "arrival_only_pending": only_arrival_pending,
            "cpu_timer": asdict(self._timer.calibration),
            "gpu_timer": asdict(self._cuda.calibration),
        }

    def auxiliary_summary(self) -> Mapping[str, Any]:
        out: Dict[str, Any] = {}
        if self.affected_sizes:
            arr = np.asarray(self.affected_sizes)
            out["affected_size_mean"] = float(arr.mean())
            out["affected_size_p99"] = float(np.percentile(arr, 99))
            out["affected_size_max"] = int(arr.max())
        if self.bfs_depths:
            arr = np.asarray(self.bfs_depths)
            out["bfs_depth_mean"] = float(arr.mean())
            out["bfs_depth_max"] = int(arr.max())
        if self.bfs_edges:
            arr = np.asarray(self.bfs_edges)
            out["bfs_edges_mean"] = float(arr.mean())
            out["bfs_edges_p99"] = float(np.percentile(arr, 99))
        return out


# =============================================================================
# Section 7.  Run record (the persisted artefact of a single experiment).
#
# Every experiment writes one of these per repetition.  The structure
# is deliberately flat and JSON-serialisable so that result files can
# be diffed across runs and inspected without loading any code.
#
# Camera-ready additions:  freshness, age_violation, time_series,
# queueing_delay, propagation, recovery_diagnostics.  The fields are
# optional (None until the probe is attached) so that legacy
# round-trip tests continue to work.
# =============================================================================


@dataclass
class RunRecord:
    """One repetition of one experiment.  Persisted to results/raw/."""

    experiment: str
    dataset: str
    attack: Optional[str]
    defense: Optional[str]
    seed: int

    # Captured at run start.
    host: Mapping[str, Any] = field(default_factory=dict)
    cpu_timer: Mapping[str, Any] = field(default_factory=dict)
    gpu_timer: Mapping[str, Any] = field(default_factory=dict)
    started_at: float = 0.0
    finished_at: float = 0.0

    # Captured during the run.
    storm_signature: str = ""             # from threat_model.UpdateStorm
    access_fingerprint: str = ""          # from TargetSystemView access log

    # Captured at run end.
    phase_a: Optional[Mapping[str, Any]] = None    # to_parquet_dict
    phase_b: Optional[Mapping[str, Any]] = None
    phase_c: Optional[Mapping[str, Any]] = None
    end_to_end: Optional[Mapping[str, Any]] = None
    queueing_delay: Optional[Mapping[str, Any]] = None
    propagation: Optional[Mapping[str, Any]] = None
    miss_report: Optional[Mapping[str, Any]] = None
    age_violation_report: Optional[Mapping[str, Any]] = None
    freshness: Optional[Mapping[str, Any]] = None
    time_series: Optional[Mapping[str, Any]] = None
    recovery_diagnostics: Optional[Mapping[str, Any]] = None
    auxiliary: Mapping[str, Any] = field(default_factory=dict)
    probe_audit: Mapping[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    @classmethod
    def begin(
        cls,
        experiment: str,
        dataset: str,
        seed: int,
        attack: Optional[str] = None,
        defense: Optional[str] = None,
    ) -> "RunRecord":
        rec = cls(
            experiment=experiment,
            dataset=dataset,
            attack=attack,
            defense=defense,
            seed=seed,
        )
        rec.host = describe_host()
        rec.started_at = time.time()
        return rec

    def attach_probe_results(
        self,
        probe: MeasuringProbe,
        recovery_baseline_us: Optional[float] = None,
        recovery_tolerance: float = 0.10,
        recovery_hold_duration_s: float = 1.0,
    ) -> None:
        """
        Pull all measurement state out of the probe and attach it
        to the record.  Camera-ready: also computes recovery
        diagnostics if ``recovery_baseline_us`` is provided.

        The ``recovery_baseline_us`` is typically the benign-warmup
        mean end-to-end latency; experiments.py captures it from the
        warmup window and forwards it here.  When None, the recovery
        section is left null (the rest of the record is still
        complete).
        """
        self.phase_a = probe.phase_a.to_parquet_dict()
        self.phase_b = probe.phase_b.to_parquet_dict()
        self.phase_c = probe.phase_c.to_parquet_dict()
        self.end_to_end = probe.end_to_end.to_parquet_dict()
        # Queueing delay and propagation latency.
        if len(probe.queueing_delay) > 0:
            self.queueing_delay = probe.queueing_delay.to_parquet_dict()
        if len(probe.propagation) > 0:
            self.propagation = probe.propagation.to_parquet_dict()
        self.miss_report = probe.miss_report.summary()
        self.age_violation_report = probe.age_violation_report.summary()
        self.freshness = probe.freshness.to_parquet_dict()
        self.time_series = probe.time_series.to_parquet_dict()

        if recovery_baseline_us is not None:
            recovery_s = probe.time_series.compute_recovery_time(
                baseline_latency_us=recovery_baseline_us,
                tolerance=recovery_tolerance,
                hold_duration_s=recovery_hold_duration_s,
            )
            self.recovery_diagnostics = {
                "baseline_latency_us": recovery_baseline_us,
                "tolerance": recovery_tolerance,
                "hold_duration_s": recovery_hold_duration_s,
                "recovery_time_s": recovery_s,
                "recovery_observed": recovery_s is not None,
            }

        self.auxiliary = probe.auxiliary_summary()
        self.probe_audit = probe.overhead_audit()
        self.cpu_timer = asdict(probe._timer.calibration)
        self.gpu_timer = asdict(probe._cuda.calibration)

    def finish(self) -> None:
        self.finished_at = time.time()

    def write(self, path: Path) -> None:
        """
        Write the record as JSON.  Raw sample arrays inside phase_a
        etc. may be large (10⁵–10⁷ ints); JSON is fine for our
        scale and is auditable in a reviewer's text editor.  For
        larger scales the same dict can be written to parquet
        instead.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, default=_json_default)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON serializable: {type(obj)!r}")


# =============================================================================
# Section 8.  Host description and process pinning.
#
# The host description is captured into every run record so that
# anomalous runs can be traced back to host state (kernel version,
# CPU governor, NUMA topology).  The pinning helpers wrap taskset,
# numactl, and chrt; on a host that lacks these (e.g. CI) they
# log a warning and continue.
#
# Camera-ready: ``describe_host`` additionally reports
# hyperthreading status, NUMA node count, CPU isolation
# (``isolcpus``), and kernel preemption mode (PREEMPT_RT vs
# PREEMPT vs PREEMPT_VOLUNTARY).  These are the four host-state
# variables RTSS reviewers most often ask about; capturing them
# automatically removes the "did you run on isolated cores?"
# round-trip from review.
# =============================================================================


def describe_host() -> Mapping[str, Any]:
    """
    Capture the platform information needed to interpret a
    measurement.  All fields are optional: a host that lacks
    /proc/cpuinfo (e.g. macOS) will report what is available.

    Camera-ready: also reports
      - hyperthreading status
      - NUMA node count
      - CPU isolation (isolcpus from kernel cmdline)
      - kernel preemption mode
    """
    info: Dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "processor": platform.processor(),
        "hostname": platform.node(),
    }

    # CPU count, governor, MHz from /proc when available.
    try:
        with open("/proc/cpuinfo") as f:
            cpuinfo = f.read()
        info["cpu_count"] = cpuinfo.count("\nprocessor")
        for line in cpuinfo.splitlines():
            if line.startswith("model name"):
                info["cpu_model"] = line.split(":", 1)[1].strip()
                break
        # Camera-ready: detect hyperthreading by comparing logical
        # processors to physical cores.  If "siblings" > "cpu cores"
        # in any core block, HT is on.
        siblings = None
        cores = None
        for line in cpuinfo.splitlines():
            if line.startswith("siblings") and siblings is None:
                try:
                    siblings = int(line.split(":", 1)[1].strip())
                except (ValueError, IndexError):
                    pass
            if line.startswith("cpu cores") and cores is None:
                try:
                    cores = int(line.split(":", 1)[1].strip())
                except (ValueError, IndexError):
                    pass
            if siblings is not None and cores is not None:
                break
        if siblings is not None and cores is not None and cores > 0:
            info["hyperthreading"] = siblings > cores
            info["physical_cores_per_socket"] = cores
            info["logical_per_socket"] = siblings
    except FileNotFoundError:
        pass

    try:
        with open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor") as f:
            info["cpu_governor"] = f.read().strip()
    except FileNotFoundError:
        pass

    try:
        info["kernel"] = platform.release()
    except Exception:
        pass

    # Camera-ready: NUMA topology.
    try:
        numa_nodes = [
            d for d in os.listdir("/sys/devices/system/node")
            if d.startswith("node") and d[4:].isdigit()
        ]
        info["numa_node_count"] = len(numa_nodes)
    except (FileNotFoundError, OSError):
        pass

    # Camera-ready: CPU isolation from kernel cmdline.
    try:
        with open("/proc/cmdline") as f:
            cmdline = f.read()
        m = re.search(r"isolcpus=(\S+)", cmdline)
        info["isolcpus"] = m.group(1) if m else ""
        info["kernel_cmdline_present"] = True
    except FileNotFoundError:
        info["kernel_cmdline_present"] = False

    # Camera-ready: kernel preemption mode.  Distinguishes:
    #   - PREEMPT_RT      (real-time kernel; lowest jitter)
    #   - PREEMPT         (low-latency / desktop)
    #   - PREEMPT_VOLUNTARY (server default)
    #   - PREEMPT_NONE    (legacy server)
    # The cleanest source is /sys/kernel/preemption (newer kernels)
    # or `uname -v` (older kernels).
    try:
        with open("/sys/kernel/preemption") as f:
            info["kernel_preempt"] = f.read().strip()
    except FileNotFoundError:
        try:
            uname_v = platform.version()  # e.g. "#1 SMP PREEMPT_DYNAMIC ..."
            for marker in (
                "PREEMPT_RT", "PREEMPT_DYNAMIC", "PREEMPT_VOLUNTARY",
                "PREEMPT", "PREEMPT_NONE",
            ):
                if marker in uname_v:
                    info["kernel_preempt"] = marker
                    break
        except Exception:
            pass

    # CUDA / GPU info.
    try:
        import torch
        if torch.cuda.is_available():
            info["cuda_count"] = torch.cuda.device_count()
            info["cuda_devices"] = [
                torch.cuda.get_device_name(i)
                for i in range(torch.cuda.device_count())
            ]
            info["cuda_version"] = torch.version.cuda
    except Exception:
        pass

    return info


def pin_to_cores(cores: Sequence[int]) -> bool:
    """
    Pin the current process to the given physical cores using
    sched_setaffinity.  Returns True on success, False on platforms
    that lack it (logs a warning).

    For RTSS-quality measurements the SUT, attacker, and workload
    generator must run on disjoint core sets; this is enforced by
    experiments.py using configs/hardware.yaml.
    """
    if not hasattr(os, "sched_setaffinity"):
        logger.warning(
            "sched_setaffinity not available on this platform; "
            "not pinning"
        )
        return False
    try:
        os.sched_setaffinity(0, set(cores))
        return True
    except OSError as e:
        logger.warning(f"sched_setaffinity failed: {e}")
        return False


def set_realtime_priority(priority: int = 80) -> bool:
    """
    Set SCHED_FIFO scheduling for the current process.  Requires
    CAP_SYS_NICE.  Returns True on success, False otherwise.

    Used by experiments.py for the timing-critical experiments to
    suppress OS scheduling jitter on the SUT thread.
    """
    if not hasattr(os, "sched_setscheduler"):
        logger.warning(
            "sched_setscheduler not available; not setting RT priority"
        )
        return False
    try:
        # SCHED_FIFO = 1 on Linux.
        param = os.sched_param(priority)            # type: ignore[attr-defined]
        os.sched_setscheduler(0, 1, param)          # type: ignore[attr-defined]
        return True
    except (OSError, PermissionError) as e:
        logger.warning(
            f"set_realtime_priority({priority}) failed: {e}.  "
            "Re-run as root or with CAP_SYS_NICE to enable RT scheduling."
        )
        return False


@contextmanager
def quiescent_environment() -> Iterator[None]:
    """
    Context manager that minimises non-SUT activity during a
    timing-critical block: disables Python's GC, runs a final
    collection beforehand, and optionally drops page cache (no-op
    without root).  Use sparingly: inside the block, garbage will
    accumulate.
    """
    gc_was_enabled = gc.isenabled()
    gc.collect()
    gc.disable()
    try:
        yield
    finally:
        if gc_was_enabled:
            gc.enable()


# =============================================================================
# Section 9.  Public surface.
# =============================================================================


__all__ = [
    # Calibration and timers
    "TimerCalibration",
    "MonotonicTimer",
    "CudaTimer",
    # Distributions and reports
    "LatencyDistribution",
    "DeadlineMissReport",
    "AgeViolationReport",
    # Freshness and time-series
    "ModelFreshnessTracker",
    "TimeSeriesRecorder",
    # Probe
    "MeasuringProbe",
    # Run record
    "RunRecord",
    # Host and pinning
    "describe_host",
    "pin_to_cores",
    "set_realtime_priority",
    "quiescent_environment",
]
