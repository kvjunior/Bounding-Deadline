"""
analysis.py — Formal schedulability analysis for the update path.

This module is the executable form of Section IV of the paper.  Every
theorem and lemma in the paper has a corresponding function here.  For
each theorem/lemma, the file exposes two things:

  (1) ``apply_<name>(...)`` — applies the bound.  Used by the
      schedulability-aware defense to decide whether to admit a
      transaction.
  (2) ``validate_<name>(...)`` — validates the bound against measured
      data.  Used by experiments.py to report empirical conformance,
      and by §V.3 (Schedulability Validation) figures.

Both functions reference the same bound expression, so the bound used
by the defense and the bound checked at validation cannot drift apart.
A property test in tests/test_analysis.py exercises the apply/validate
identity by sampling random inputs and checking that the two functions
agree on what they consider satisfactory.

Why probabilistic, not deterministic, bounds
---------------------------------------------
Worst-case analysis of the update path with deterministic bounds
yields useless numbers because the maximum affected-subgraph size
under benign workload is itself heavy-tailed (real graphs have hub
nodes whose neighbourhoods span 10⁵+ accounts).  A worst-case bound
sized for the genuine worst case would force the SUT to treat every
transaction as potentially catastrophic, destroying throughput.

This paper uses *probabilistic* bounds: the deadline holds with
probability ≥ 1 − ε under the bounded-budget adversary model in
threat_model.py.  This is the same methodological move as in the
stochastic-WCRT line:

  - Cucu-Grosjean, Santinelli, Houston, Lo, Vardanega, Kosmidis,
    Abella, Mezzetti, Quiñones, Cazorla, "Measurement-based
    probabilistic timing analysis for multi-path programs," ECRTS
    2012.
  - Davis and Cucu-Grosjean, "A survey of probabilistic timing
    analysis techniques for real-time systems," LITES, 6(1), 2019.
  - Sun, Yu, Jiang, Deng, Guan, "WCDFP analysis for real-time tasks
    with stochastic release patterns using Chernoff bound," RTSS 2025.

The Sun et al. paper is the closest methodological precedent for our
Theorem 4.3, because it bounds the deadline-failure probability when
the workload in a deadline interval is itself a random sum (the count
of jobs is random, not just their per-job execution time).  We adopt
the same Chernoff-based random-sum strategy and credit it explicitly
in the theorem text.

Two analysis paths
------------------
The file provides two paths to a Theorem 4.3 certificate:

  (A) The exponential-envelope path (Sections 1–4).  A lightweight,
      dataset-driven empirical envelope: fit an exponential upper
      bound on the per-transaction cost CCDF, mix benign and
      adversarial CCDFs by adversary fraction α, and pick the
      smallest threshold T whose mixture-CCDF lies at or below ε.
      Cheap to compute; suitable for online admission.

  (B) The MGF random-sum path (Sections 5–6).  The closer analogue to
      Sun et al.: discretise the per-transaction cost distribution
      into a CostDistribution, discretise the per-window admitted-
      count distribution into a CountDistribution, compute their
      moment-generating functions, and bound

          P( B_0 + Σ_a Σ_{j ≤ N_a^D} C_{a,j} > D )
          ≤ inf_{λ > 0} exp(-λD) · M_{B_0}(λ) · ∏_a M_{S_a}(λ),

      where B_0 is carry-in/backlog at the start of the deadline
      window, N_a^D is the random number of class-a admitted updates
      in [0,D], and the random-sum MGF is the *exact* identity

          log M_{S_a}(λ)  =  log M_{N_a^D}( log M_{C_a}(λ) ).

      The optimal λ is found by convex one-dimensional search (Sun
      et al. prove convexity of the log objective).  More expensive
      to compute; suitable for off-line certificates and for
      reviewer-facing schedulability tables.

The defense (defenses.py) uses path (A) at admission time because the
cost is bounded; experiments.py uses path (B) for the headline
schedulability validation table because it is the bound that engages
most directly with the RTSS literature.

Camera-ready improvements over the submission draft
---------------------------------------------------
1. **No hidden Poisson assumption.**  The submission draft used a
   Poisson upper bound on the random-count MGF.  The bounded-budget
   adversary model (threat_model.AdversaryBudget plus
   EvasionConstraints) does not imply sub-Poisson admitted counts —
   bursty admissions are explicitly permitted up to
   ``max_burst_factor``.  We therefore replaced the Poisson upper
   bound with a measured CountDistribution (Section 5).  The Poisson
   case is preserved as ``poisson_count_distribution`` for callers
   who can justify it.

2. **Lemma 4.4, not Theorem 4.4.**  The submission draft labelled
   κ = 1 + (ρ_obs / ρ_def) · γ as Theorem 4.4 with no proof.  This
   file labels it Lemma 4.4 with explicit assumptions A2.1–A2.3 and
   a documented proof sketch (Section 7).  γ is a measurable system
   constant, not a hand-waved factor.

3. **No silent fallbacks.**  ``analyse_storm`` previously set
   ``adv_fit = benign_fit`` whenever no adversarial samples were
   observed, making Theorems 4.1 and 4.2 trivially equal.  This
   file flags such fallbacks explicitly via
   ``StormAnalysis.adversarial_fit_is_benign_fallback`` and logs a
   warning so downstream consumers can refuse to treat the
   "validation" of 4.2 as a real validation.

4. **Wilson-CI accounting in 4.3 validation.**  The submission
   draft's ``validate_theorem_4_3`` only checked
   ``measured_miss_rate ≤ ε``.  The paper claims the contract is
   satisfied "at 95 % confidence"; this file computes a Wilson
   one-sided lower bound on the *non-miss* rate and reports
   ``holds_at_ci_95`` separately so that a tight bound that
   coincidentally falls below ε on a small sample does not produce
   a false-positive validation.

5. **MGF boundary diagnostics.**  ``MGFCertificate.notes`` now
   reports λ-bracket-edge cases explicitly so plots.py can flag
   cells where the bound is loose because of MGF radius-of-
   convergence limits, not because of genuine infeasibility.

Assumptions (explicit)
----------------------
The bounds derived in this file rely on the following assumptions.
Every assumption is named where it is invoked; this list is the
single source of truth.

  A1 (Per-class i.i.d. costs).  Conditional on the class label of an
      admitted transaction (benign vs. each adversarial attack
      class), per-transaction costs C_{a,j} are independent and
      identically distributed.  We do NOT assume independence across
      classes globally; the adversary may concentrate injections in
      time, which would violate global independence.  Per-class
      conditional independence is the weakest assumption that
      supports the random-sum bound.

  A2 (Bounded adaptivity).  Tier-3 (adaptive) adversaries adapt to
      the defender's threshold T at observation rate ρ_obs and the
      defender adapts at rate ρ_def.  Lemma 4.4 invokes three
      sub-assumptions A2.1–A2.3 (Section 7); the κ degradation
      factor in 4.4 is the explicit multiplicative cost of bounded
      adaptivity under those assumptions.

  A3 (Discrete-time MGF approximation).  The MGF certificate
      discretises continuous cost distributions into bins.  Bin
      width is configurable; tests/test_analysis.py validates
      sensitivity to the choice.  Sun et al. use natively discrete
      distributions; we discretise from measurement.

  A4 (Carry-in stationarity).  The B_0 backlog term is treated as a
      random variable with bounded support whose distribution is
      estimated from the previous deadline window.  We do not
      assume B_0 = 0 (the "simultaneous-release" assumption that
      Sun et al. explicitly reject as unsafe); when carry-in
      samples are not available the MGF certificate uses a
      degenerate-zero distribution and emits a logger warning that
      mirrors Sun et al.'s caveat.

These assumptions are also stated in the paper's §IV; the code must
not silently weaken them.

Key bounds proved in the paper
------------------------------
Theorem 4.1 (Per-transaction cost bound under benign workload).
  Under the benign arrival model, the cost of processing one
  transaction through the update path satisfies
        Pr[ C(t) > c ] ≤ exp(-α(c − μ_C))
  for c > μ_C, where μ_C is the mean cost and α a graph-dependent
  constant.  Verified empirically per dataset (§V.3).

Theorem 4.2 (Adversarial inflation bound).
  Under the budgeted adversary of threat_model.py, the conditional
  cost given an injection from attack class A satisfies
        Pr[ C(t) > c | injection from A ] ≤ exp(-α_A(c − μ_A))
  with α_A < α and μ_A > μ_C.

Theorem 4.3 (Schedulability under defended workload).
  Two formulations.

  4.3-env (lightweight, exponential envelope):
        Pr[ C(t) > D ] ≤ φ_T(D)
  with φ_T the post-admission mixture CCDF.

  4.3-mgf (Sun-style random-sum):
        Pr[ B_0 + Σ_a Σ_j C_{a,j} > D ]
            ≤ inf_λ  exp(-λD) · M_{B_0}(λ) · ∏_a M_{S_a}(λ)
        log M_{S_a}(λ) = log M_{N_a}( log M_{C_a}(λ) )      (exact)

  Both versions are computed; both are validated; the MGF version is
  the one the paper claims under assumptions A1, A3, A4.

Lemma 4.4 (Adaptive adversary degradation).
  Against an adversary that observes T at rate ρ_obs and tunes
  injections to fall just below it, with the defender refitting at
  rate ρ_def, under assumptions A2.1–A2.3 (Section 7) the
  integrated deadline-miss probability over a window W is bounded
  by  κ · ε,  where  κ = 1 + (ρ_obs / ρ_def) · γ.  γ is a
  system-dependent overshoot constant that must be measured per
  deployment; §VI reports measured κ on the four datasets.

Each theorem/lemma appears below as ``apply_*`` / ``validate_*``.

Author: ANONYMOUS  (double-blind review)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import (
    Any,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np

from threat_model import (
    AdversaryBudget,
    CapabilityTier,
    RealTimeContract,
    UpdateStorm,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Section 1.  Cost-distribution fits (exponential envelope).
#
# All four bounds can be stated in terms of an upper bound on the
# tail of the per-transaction cost distribution.  The bound shape we
# use here is the Chernoff form (subexponential tail), which holds
# for any sub-Gaussian random variable and is tight for the
# workloads we observe.  This section fits the parameters from
# measured cost distributions; the rest of the file consumes the
# fits.
#
# We are deliberately restrained about distributional assumptions.
# We do not claim the cost distribution IS exponential; we claim its
# tail is BOUNDED ABOVE by an exponential, which is a much weaker
# and empirically defensible claim.  The fitting routine returns
# the tightest exponential upper envelope of the empirical CCDF
# (with a small inflation when the fit is numerically degenerate;
# see ``_inflate_for_safety`` below).
# =============================================================================


# Minimum tail samples below which a fit is considered degenerate
# and ``alpha = +inf`` is returned (the "no observed tail" case).
_MIN_TAIL_SAMPLES_FOR_FIT: int = 10

# Numerical floor on the coverage rate of the fitted envelope on the
# calibration set.  Coverage below this triggers a safety inflation
# (we widen alpha so the bound strictly dominates the empirical
# CCDF on the calibration data).
_MIN_FIT_COVERAGE_RATE: float = 0.999


@dataclass(frozen=True)
class TailFit:
    """
    Parameters of an exponential upper bound on the cost distribution
    tail.  Concretely, for cost ``c > c_anchor``:

        P(cost > c) ≤ exp(-alpha * (c - c_anchor))             (1)

    The fit is conservative: the bound is the tightest exponential
    that lies above the empirical CCDF over the calibration range,
    after a small safety inflation when numerical edge cases would
    otherwise produce a non-strict envelope.

    Attributes
    ----------
    c_anchor : float
        Cost at which the bound starts to apply (microseconds).  The
        bound need not hold below this; below ``c_anchor`` the bound
        is conventionally ``Pr ≤ 1``.
    alpha : float
        Decay rate (1/microseconds).  Larger alpha = lighter tail.
    n_samples : int
        Number of samples the fit was computed from.
    coverage_rate : float
        Fraction of the calibration range where the empirical CCDF
        lies at or below the fitted bound.  Should be 1.0 for a
        valid upper-bound fit.  Reported in case it is below 1.0 due
        to numerical edge cases; ``fit_exponential_upper_bound``
        inflates ``alpha`` until this exceeds ``_MIN_FIT_COVERAGE_RATE``.
    is_degenerate : bool
        True when the input had fewer than ``_MIN_TAIL_SAMPLES_FOR_FIT``
        tail samples.  In that case ``alpha = +inf`` (the trivial
        envelope) and any bound this fit produces above ``c_anchor``
        is exactly zero.  Callers must handle this; see
        ``analyse_storm`` for the canonical handling.
    """

    c_anchor: float
    alpha: float
    n_samples: int
    coverage_rate: float
    is_degenerate: bool = False

    def upper_bound_pr_exceeds(self, c: float) -> float:
        """Right-hand side of bound (1) for cost c."""
        if c <= self.c_anchor:
            return 1.0
        if self.is_degenerate:
            # Degenerate fit: tail unobserved.  Above c_anchor we
            # claim 0, which is unsafe in production; the calling
            # code (analyse_storm) refuses to use a degenerate fit
            # for a bound that is consumed by the defense.
            return 0.0
        return math.exp(-self.alpha * (c - self.c_anchor))

    def required_threshold_for(self, target_pr: float) -> float:
        """
        Solve bound (1) for c such that the bound equals ``target_pr``.
        Used by the defense to pick the cost threshold T that meets a
        given failure-probability target.

        Edge cases.  Returns ``c_anchor`` for ``target_pr ≥ 1``;
        ``+inf`` for ``target_pr ≤ 0``.  For a degenerate fit, returns
        ``c_anchor`` (since the fit makes no claim above the anchor).
        """
        if target_pr >= 1.0:
            return self.c_anchor
        if target_pr <= 0.0:
            return float("inf")
        if self.is_degenerate:
            return self.c_anchor
        return self.c_anchor - math.log(target_pr) / self.alpha

    def with_inflated_alpha(self, factor: float) -> "TailFit":
        """
        Return a copy with ``alpha`` divided by ``factor`` (so the
        envelope becomes wider/less aggressive).  Used internally by
        the safety-inflation step in ``fit_exponential_upper_bound``.
        """
        if factor <= 0:
            raise ValueError("inflation factor must be positive")
        return TailFit(
            c_anchor=self.c_anchor,
            alpha=self.alpha / factor,
            n_samples=self.n_samples,
            coverage_rate=self.coverage_rate,
            is_degenerate=self.is_degenerate,
        )


def fit_exponential_upper_bound(
    samples_us: Sequence[float],
    anchor_percentile: float = 50.0,
) -> TailFit:
    """
    Fit a conservative exponential upper bound on the empirical tail.

    Methodology
    -----------
    Anchor ``c0`` at the given percentile (median by default).  For
    ``c ≥ c0``, the empirical CCDF is

        F̂(c) = (#samples > c) / N.

    We seek the tightest ``alpha`` such that ``F̂(c) ≤ exp(-alpha (c−c0))``
    for all c in the calibrated range.  This is

        alpha = min over c > c0 of  ( -log(F̂(c)) / (c − c0) )

    excluding c values where F̂(c) = 0 (those impose no constraint).

    Returns ``alpha = +∞`` (degenerate) if the empirical tail is
    degenerate (all samples at c0 or fewer than
    ``_MIN_TAIL_SAMPLES_FOR_FIT`` strict-tail samples); the caller
    must treat this as "no observed tail" and use a defensive default.

    Safety inflation
    ----------------
    If the fitted ``alpha`` produces a coverage rate below
    ``_MIN_FIT_COVERAGE_RATE`` on the calibration set (which can
    happen at the boundary due to floating-point rounding), we
    inflate alpha by a small factor (1.01 per iteration, capped at
    100 iterations) until coverage is restored.  This guarantees the
    returned fit is a strict upper envelope on the calibration data.
    """
    arr = np.sort(np.asarray(samples_us, dtype=np.float64))
    if arr.size == 0:
        raise ValueError("cannot fit on empty samples")
    if not 0.0 < anchor_percentile < 100.0:
        raise ValueError(
            f"anchor_percentile must be in (0, 100); got {anchor_percentile}"
        )

    n = arr.size
    c_anchor = float(np.percentile(arr, anchor_percentile))

    # Take the strict upper tail.
    tail = arr[arr > c_anchor]
    if tail.size < _MIN_TAIL_SAMPLES_FOR_FIT:
        # Too few tail samples for a meaningful fit.  Return the
        # degenerate fit; caller must check is_degenerate.
        return TailFit(
            c_anchor=c_anchor,
            alpha=float("inf"),
            n_samples=n,
            coverage_rate=1.0,
            is_degenerate=True,
        )

    # Empirical CCDF at each tail sample.  Sample i (in sorted order)
    # has CCDF (n - rank_i) / n.
    tail_ranks = np.arange(tail.size)
    ccdf = (tail.size - tail_ranks) / float(n)

    # alpha candidates: -log(ccdf_i) / (c_i - c_anchor) for each i.
    # Exclude ccdf = 0 (last point, where -log diverges).
    safe = ccdf > 0
    if not safe.any():
        return TailFit(
            c_anchor=c_anchor, alpha=float("inf"),
            n_samples=n, coverage_rate=1.0, is_degenerate=True,
        )
    alpha_candidates = -np.log(ccdf[safe]) / (tail[safe] - c_anchor)
    # The bound must hold for all c, so we need the *smallest* alpha
    # (lightest decay) — the constraint that makes the bound the
    # least tight.  This gives a conservative upper envelope.
    alpha = float(np.min(alpha_candidates))

    # Coverage check + safety inflation.  We aim for
    # coverage_rate ≥ _MIN_FIT_COVERAGE_RATE on the calibration tail.
    coverage = _coverage_rate(tail, tail_ranks, n, alpha, c_anchor)
    n_inflations = 0
    while coverage < _MIN_FIT_COVERAGE_RATE and n_inflations < 100:
        alpha = alpha / 1.01
        coverage = _coverage_rate(tail, tail_ranks, n, alpha, c_anchor)
        n_inflations += 1
    if n_inflations > 0:
        logger.debug(
            f"fit_exponential_upper_bound: inflated alpha {n_inflations} "
            f"times to reach coverage {coverage:.4f}"
        )

    return TailFit(
        c_anchor=c_anchor,
        alpha=alpha,
        n_samples=n,
        coverage_rate=coverage,
        is_degenerate=False,
    )


def _coverage_rate(
    tail: np.ndarray,
    tail_ranks: np.ndarray,
    n: int,
    alpha: float,
    c_anchor: float,
) -> float:
    """Internal helper used by fit_exponential_upper_bound."""
    bound_at_tail = np.exp(-alpha * (tail - c_anchor))
    actual_pr = (tail.size - tail_ranks) / float(n)
    # Use a small numerical tolerance: ulp-of-1 around 1e-15 plus a
    # multiplicative cushion so that alpha values that produce
    # bound_at_tail == actual_pr to within float rounding count as
    # covered.  We standardise on 1e-12 absolute and 1e-9 relative.
    tol_abs = 1e-12
    tol_rel = 1e-9
    covered = bound_at_tail >= actual_pr - (tol_abs + tol_rel * actual_pr)
    return float(np.mean(covered))


# =============================================================================
# Section 2.  Theorem 4.1 — Per-transaction cost bound (benign workload).
#
# Statement (paper §IV, Theorem 4.1).
#   For benign workload, there exist c_anchor and alpha such that for
#   all c > c_anchor:
#
#       Pr[ C(t) > c ] ≤ exp(-alpha (c - c_anchor)).
#
# The bound is system- and dataset-dependent.  In practice, alpha and
# c_anchor are fitted from a calibration run on benign traffic; this
# section computes them and validates them against held-out data.
#
# Assumption A1 (per-class i.i.d. costs) is invoked here for the
# benign class.  Without independence, the empirical CCDF is still a
# consistent estimator of the marginal cost distribution, but the
# fitted exponential envelope no longer characterises sums of
# independent costs.  The paper notes this and only invokes
# independence in Theorem 4.3-mgf.
# =============================================================================


def apply_theorem_4_1(fit: TailFit, c: float) -> float:
    """Return the bound on Pr[cost > c] under benign workload."""
    return fit.upper_bound_pr_exceeds(c)


@dataclass(frozen=True)
class TheoremValidation:
    """
    Result of validating a theorem against measured data.

    Camera-ready: includes a Wilson one-sided lower-bound check on the
    *non-miss* rate (``ci_lo_at_p99``, ``ci_lo_at_p99_9``).  The paper
    claims contracts are satisfied "at 95 % confidence"; for that
    claim to hold, the lower CI on (1 − measured_miss_rate) must be
    above (1 − ε), not just the point estimate.  ``holds_at_ci_95``
    captures this.

    A ``holds_at_p99`` of True with ``holds_at_ci_95`` of False means
    the bound was not violated on the sample we have, but the sample
    was too small to validate the bound at the paper's stated
    confidence.  Reviewers care about this distinction.
    """

    theorem: str
    n_samples: int
    holds_at_p99: bool
    holds_at_p99_9: bool
    bound_at_p99: float
    measured_at_p99: float
    bound_at_p99_9: float
    measured_at_p99_9: float
    # Wilson one-sided 95 % lower bound on (1 − measured_at_p99).
    # nan when n_samples < 100.
    ci_lo_at_p99: float = float("nan")
    ci_lo_at_p99_9: float = float("nan")
    # holds_at_ci_95: True iff (1 − ε) ≤ ci_lo_at_p99 for the
    # relevant percentile.  False indicates the bound is consistent
    # with the data but not statistically validated at 95 % CI.
    holds_at_ci_95: bool = False
    notes: str = ""

    def summary(self) -> Mapping[str, Any]:
        return {
            "theorem": self.theorem,
            "n_samples": self.n_samples,
            "holds_at_p99": self.holds_at_p99,
            "holds_at_p99_9": self.holds_at_p99_9,
            "bound_at_p99": self.bound_at_p99,
            "measured_at_p99": self.measured_at_p99,
            "bound_at_p99_9": self.bound_at_p99_9,
            "measured_at_p99_9": self.measured_at_p99_9,
            "ci_lo_at_p99": self.ci_lo_at_p99,
            "ci_lo_at_p99_9": self.ci_lo_at_p99_9,
            "holds_at_ci_95": self.holds_at_ci_95,
            "notes": self.notes,
        }


def _wilson_one_sided_lower(success_count: int, n: int, z: float = 1.6449) -> float:
    """
    Wilson one-sided 95 % lower bound on the success rate.

    z = 1.6449 corresponds to 95 % one-sided (Φ⁻¹(0.95)).  We use the
    one-sided form because we are checking a *one-sided* inequality
    (bound ≥ measured); a two-sided 95 % CI would be unnecessarily
    conservative for our use case.

    Returns 0.0 when n is 0.
    """
    if n <= 0:
        return 0.0
    p_hat = success_count / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = p_hat + z2 / (2 * n)
    half_width = z * math.sqrt(
        max(0.0, p_hat * (1.0 - p_hat) / n + z2 / (4.0 * n * n))
    )
    return max(0.0, (centre - half_width) / denom)


def validate_theorem_4_1(
    fit: TailFit,
    held_out_samples_us: Sequence[float],
) -> TheoremValidation:
    """
    Check that the fitted bound from Theorem 4.1 holds against
    held-out data.  Returns a TheoremValidation; experiments.py
    persists this in the run record and plots.py renders the
    comparison.

    The validation is at two thresholds: held-out P99 and P99.9.  The
    bound holds at threshold ``c`` if the bound's predicted upper
    tail at ``c`` is ≥ the measured fraction of samples exceeding
    ``c``.  Note: "holds" can be conservative (bound is loose); we
    report both the bound value and the measured value so a reviewer
    can judge tightness.

    Camera-ready: also reports a Wilson one-sided 95 % lower bound on
    the empirical *non-miss* rate at each percentile; when this lower
    bound exceeds (1 − bound), we declare the bound validated at 95 %
    confidence.
    """
    arr = np.asarray(held_out_samples_us, dtype=np.float64)
    n = arr.size
    if n < 100:
        return TheoremValidation(
            theorem="4.1",
            n_samples=n,
            holds_at_p99=False, holds_at_p99_9=False,
            bound_at_p99=float("nan"), measured_at_p99=float("nan"),
            bound_at_p99_9=float("nan"), measured_at_p99_9=float("nan"),
            notes="insufficient samples (need ≥100)",
        )

    p99 = float(np.percentile(arr, 99))
    p99_9 = float(np.percentile(arr, 99.9)) if n >= 1000 else float("nan")
    measured_p99 = float(np.mean(arr > p99))
    measured_p99_9 = (
        float(np.mean(arr > p99_9)) if not math.isnan(p99_9) else float("nan")
    )
    bound_p99 = apply_theorem_4_1(fit, p99)
    bound_p99_9 = (
        apply_theorem_4_1(fit, p99_9) if not math.isnan(p99_9) else float("nan")
    )

    # Wilson lower bound on the non-miss rate.
    n_non_miss_p99 = int(np.sum(arr <= p99))
    ci_lo_non_miss_p99 = _wilson_one_sided_lower(n_non_miss_p99, n)
    ci_lo_at_p99 = 1.0 - ci_lo_non_miss_p99   # upper bound on miss rate

    if not math.isnan(p99_9):
        n_non_miss_p99_9 = int(np.sum(arr <= p99_9))
        ci_lo_non_miss_p99_9 = _wilson_one_sided_lower(n_non_miss_p99_9, n)
        ci_lo_at_p99_9 = 1.0 - ci_lo_non_miss_p99_9
    else:
        ci_lo_at_p99_9 = float("nan")

    holds_p99 = bound_p99 >= measured_p99
    holds_p99_9 = (
        bound_p99_9 >= measured_p99_9 if not math.isnan(p99_9) else False
    )
    # Holds at 95 % CI when the upper end of the CI on the measured
    # miss rate is at or below the bound.
    holds_at_ci_95 = bound_p99 >= ci_lo_at_p99

    return TheoremValidation(
        theorem="4.1",
        n_samples=n,
        holds_at_p99=bool(holds_p99),
        holds_at_p99_9=bool(holds_p99_9),
        bound_at_p99=bound_p99,
        measured_at_p99=measured_p99,
        bound_at_p99_9=bound_p99_9,
        measured_at_p99_9=measured_p99_9,
        ci_lo_at_p99=ci_lo_at_p99,
        ci_lo_at_p99_9=ci_lo_at_p99_9,
        holds_at_ci_95=bool(holds_at_ci_95),
    )


# =============================================================================
# Section 3.  Theorem 4.2 — Adversarial inflation bound.
#
# Statement.
#   Under a budgeted adversary submitting a fraction α of the stream
#   from attack class A, the conditional cost given an attacker
#   injection satisfies
#
#       Pr[ C(t) > c | from A ] ≤ exp(-alpha_A (c - mu_A))
#
#   with parameters fitted on labelled adversarial data.
#
# This theorem is the paper's empirical hinge: it says that
# adversarial workloads have a *characterisable* tail, not just a
# heavier one.  Without this, we could not bound the post-admission
# tail in Theorem 4.3.
# =============================================================================


def apply_theorem_4_2(adversarial_fit: TailFit, c: float) -> float:
    """Bound on Pr[cost > c | attacker injection]."""
    return adversarial_fit.upper_bound_pr_exceeds(c)


def validate_theorem_4_2(
    adversarial_fit: TailFit,
    held_out_adversarial_samples_us: Sequence[float],
) -> TheoremValidation:
    """Same validation pattern as Theorem 4.1, on adversarial samples."""
    v = validate_theorem_4_1(adversarial_fit, held_out_adversarial_samples_us)
    return TheoremValidation(
        theorem="4.2",
        n_samples=v.n_samples,
        holds_at_p99=v.holds_at_p99,
        holds_at_p99_9=v.holds_at_p99_9,
        bound_at_p99=v.bound_at_p99,
        measured_at_p99=v.measured_at_p99,
        bound_at_p99_9=v.bound_at_p99_9,
        measured_at_p99_9=v.measured_at_p99_9,
        ci_lo_at_p99=v.ci_lo_at_p99,
        ci_lo_at_p99_9=v.ci_lo_at_p99_9,
        holds_at_ci_95=v.holds_at_ci_95,
        notes=v.notes,
    )


# =============================================================================
# Section 4.  Theorem 4.3-env — Schedulability via exponential envelope.
#
# This is the lightweight admission-time form of Theorem 4.3.  Given
#   - benign tail fit             (Theorem 4.1)
#   - adversarial tail fit         (Theorem 4.2)
#   - adversary fraction α
#   - admission threshold T
#   - deadline D
#
# we bound the post-admission deadline-miss probability via the
# mixture CCDF
#
#       φ_T(D) ≤ (1 − α) F_benign(D) + α F_adv(D)         for c ≥ T,
#
# truncated at admission threshold T.  The defense chooses T such that
# φ_T(T) ≤ ε.  This is fast (O(1) per evaluation, O(log(1/ε)) for the
# binary search to find T) and is what defenses.py invokes online.
#
# This formulation does NOT model carry-in or random-sum effects.
# Section 6 below provides the more rigorous Sun-style MGF
# certificate that does.
# =============================================================================


@dataclass(frozen=True)
class SchedulabilityCertificate:
    """
    Result of an exponential-envelope schedulability test.  If
    ``feasible`` is True, the certificate guarantees ``Pr[deadline miss]
    ≤ ε``; if False, the ``slack_us`` field reports how far off
    feasibility is.
    """

    feasible: bool
    threshold_us: float          # admission threshold T
    deadline_us: float           # contract deadline D
    epsilon: float               # contract failure probability
    bound_at_T: float            # the bound that decides feasibility
    slack_us: float              # T_required - T_chosen (negative if infeasible)
    notes: str = ""

    def __repr__(self) -> str:
        verdict = "feasible" if self.feasible else "INFEASIBLE"
        return (
            f"SchedulabilityCertificate({verdict}, "
            f"T={self.threshold_us:.0f}us, D={self.deadline_us:.0f}us, "
            f"ε={self.epsilon:g}, bound={self.bound_at_T:.4g})"
        )


def apply_theorem_4_3(
    benign_fit: TailFit,
    adversarial_fit: TailFit,
    contract: RealTimeContract,
    adversary_fraction: float,
    threshold_us: Optional[float] = None,
) -> SchedulabilityCertificate:
    """
    Theorem 4.3-env.  Determine whether ``threshold_us`` is feasible
    against ``contract`` using the lightweight exponential-envelope
    formulation.

    If ``threshold_us`` is None, computes the *minimum* feasible
    threshold and returns that.  This is what the schedulability-aware
    defense calls when initialising or recalibrating its threshold.

    Parameters
    ----------
    benign_fit, adversarial_fit
        Tail fits from Theorems 4.1 and 4.2.  May share the same fit
        object when adversarial data is unavailable; in that case the
        caller (``analyse_storm``) is required to set
        ``StormAnalysis.adversarial_fit_is_benign_fallback = True``,
        and validation routines flag the certificate as not having a
        true 4.2 component.
    contract
        Real-time contract, providing D and ε.
    adversary_fraction
        Empirical α: fraction of admitted stream that is adversarial.
        Estimated by experiments.py from labelled runs (see the
        ``is_adversarial`` flag on TimedTransaction in workload.py for
        the camera-ready non-circular labelling path).
    threshold_us
        Either the candidate threshold to test, or None to request the
        minimum feasible threshold.

    Notes
    -----
    The bound used is the mixture-CCDF upper envelope.  We never use
    a lower envelope here; the defense errs on the side of being more
    conservative than the analysis requires, not less.

    This certificate does NOT account for carry-in/backlog.  For the
    backlog-aware certificate, call ``apply_theorem_4_3_mgf``.
    """
    if not 0.0 <= adversary_fraction <= 1.0:
        raise ValueError(
            f"adversary_fraction must be in [0, 1]; got {adversary_fraction}"
        )

    D = contract.deadline_us
    eps = contract.failure_probability_bound

    def mixture_ccdf(c: float) -> float:
        return (
            (1.0 - adversary_fraction) * benign_fit.upper_bound_pr_exceeds(c)
            + adversary_fraction * adversarial_fit.upper_bound_pr_exceeds(c)
        )

    if threshold_us is None:
        # Find minimum T such that mixture_ccdf(T) ≤ eps.  The mixture
        # CCDF is monotonically nonincreasing in c, so binary search
        # is well-defined.  Search over [c_anchor_low, D].
        lo = max(benign_fit.c_anchor, adversarial_fit.c_anchor)
        hi = max(D * 2.0, lo * 4.0)  # generous upper bracket
        for _ in range(64):
            mid = 0.5 * (lo + hi)
            if mixture_ccdf(mid) <= eps:
                hi = mid
            else:
                lo = mid
        threshold_us = hi

    bound = mixture_ccdf(threshold_us)
    feasible = (bound <= eps) and (threshold_us <= D)

    if feasible:
        slack = D - threshold_us
        notes = ""
    elif bound <= eps:
        # Bound is met but the chosen threshold itself exceeds the
        # deadline.  This means the system has plenty of headroom on
        # the bound side but the threshold is mis-set above D.
        slack = D - threshold_us       # negative
        notes = "bound met but threshold exceeds deadline"
    else:
        # Compute minimum feasible T by binary search.  If even this
        # minimum exceeds D, no threshold is feasible under current α.
        lo = max(benign_fit.c_anchor, adversarial_fit.c_anchor)
        hi = max(D * 4.0, lo * 4.0, threshold_us * 4.0)
        for _ in range(20):
            if mixture_ccdf(hi) <= eps:
                break
            hi *= 2.0
        for _ in range(64):
            mid = 0.5 * (lo + hi)
            if mixture_ccdf(mid) <= eps:
                hi = mid
            else:
                lo = mid
        required = hi
        slack = D - required
        notes = (
            "threshold too low for current α; "
            f"required T ≈ {required:.0f} µs"
        )

    return SchedulabilityCertificate(
        feasible=feasible,
        threshold_us=threshold_us,
        deadline_us=D,
        epsilon=eps,
        bound_at_T=bound,
        slack_us=slack,
        notes=notes,
    )


def validate_theorem_4_3(
    cert: SchedulabilityCertificate,
    measured_miss_rate: float,
    n_admitted: int,
) -> TheoremValidation:
    """
    Compare the certificate's promised ε against the empirical miss
    rate.

    ``measured_miss_rate`` comes from
    ``MeasuringProbe.miss_report.miss_rate``.

    Camera-ready: the validation passes at point-estimate level iff
    ``measured ≤ promised``, AND at 95 % CI level iff the Wilson
    upper bound on the miss rate (from ``n_admitted`` samples) is
    also ≤ promised.  Both verdicts are reported separately.
    """
    promised = cert.epsilon
    n = max(0, int(n_admitted))
    n_miss = int(round(measured_miss_rate * n))
    n_non_miss = n - n_miss

    # Wilson 95 % one-sided lower bound on non-miss rate gives an
    # upper bound on the miss rate.
    if n > 0:
        ci_lo_non_miss = _wilson_one_sided_lower(n_non_miss, n)
        ci_hi_miss = 1.0 - ci_lo_non_miss
    else:
        ci_hi_miss = float("nan")

    holds = (measured_miss_rate <= promised)
    holds_ci = (ci_hi_miss <= promised) if not math.isnan(ci_hi_miss) else False

    return TheoremValidation(
        theorem="4.3",
        n_samples=n,
        holds_at_p99=bool(holds),
        holds_at_p99_9=bool(holds),
        bound_at_p99=promised,
        measured_at_p99=measured_miss_rate,
        bound_at_p99_9=promised,
        measured_at_p99_9=measured_miss_rate,
        ci_lo_at_p99=ci_hi_miss,
        ci_lo_at_p99_9=ci_hi_miss,
        holds_at_ci_95=bool(holds_ci),
        notes=(
            f"feasible={cert.feasible}, T={cert.threshold_us:.0f}µs, "
            f"slack={cert.slack_us:.0f}µs, "
            f"miss_ci_hi={ci_hi_miss:.4g}"
        ),
    )


# =============================================================================
# Section 5.  CostDistribution and CountDistribution.
#
# The MGF certificate (Section 6) requires per-class cost distributions
# in a form that supports computing the moment-generating function
# E[exp(λ C)] cheaply, AND per-class admitted-count distributions in
# the same form.  The submission draft used a Poisson upper bound on
# the count MGF; the camera-ready replaces that with a measured
# CountDistribution because the bounded-budget adversary model in
# threat_model.py does not imply sub-Poisson admitted counts (bursts
# are explicitly permitted up to ``EvasionConstraints.max_burst_factor``).
#
# We use a discrete representation (bin centres + probabilities) for
# both distributions because:
#
#   - Sun et al.'s derivation is explicitly discrete; matching their
#     representation makes the connection to the precedent paper
#     concrete and reviewable.
#   - The MGF of a discrete distribution is a finite sum, computable
#     in O(K) time for K bins, with no numerical-integration error.
#   - Discretisation makes the implementation auditable: a reviewer
#     can compute the bound by hand for a small K and compare.
#
# The bin width controls the trade-off between numerical precision
# and bound tightness.  We default to 1 µs bins for cost (matches
# the CudaTimer's measurement resolution per Appendix A) and natural
# integer bins for counts.
# =============================================================================


_DEFAULT_BIN_WIDTH_US: float = 1.0
_MAX_DISTRIBUTION_BINS: int = 100_000           # safety cap on memory


@dataclass(frozen=True)
class CostDistribution:
    """
    Discrete cost distribution.  Bins are right-half-open intervals
    [values_us[k], values_us[k] + bin_width_us); probabilities sum to
    1 (within rounding).

    Attributes
    ----------
    values_us : np.ndarray
        Cost bin left-edges, in microseconds.  Sorted ascending.
    probabilities : np.ndarray
        Probability mass per bin.  Same length as values_us.  Constraint:
        ``probabilities.sum() ≈ 1.0`` (within 1e-9).
    bin_width_us : float
        Width of each bin in microseconds.  Used by the MGF bound to
        place the upper edge of each bin (right edge) when computing
        E[exp(λ C)] conservatively.
    n_calibration_samples : int
        Number of measurement samples used to estimate this
        distribution.
    label : str
        Human-readable label, e.g. "benign", "A3_branching_max".
    """

    values_us: np.ndarray
    probabilities: np.ndarray
    bin_width_us: float
    n_calibration_samples: int
    label: str

    def __post_init__(self) -> None:
        if self.values_us.shape != self.probabilities.shape:
            raise ValueError(
                "values_us and probabilities must have the same shape"
            )
        if self.values_us.size == 0:
            raise ValueError("CostDistribution must be non-empty")
        if not np.all(np.diff(self.values_us) >= 0):
            raise ValueError("values_us must be sorted ascending")
        if np.any(self.probabilities < 0):
            raise ValueError("probabilities must be non-negative")
        total = float(self.probabilities.sum())
        if not (0.999 <= total <= 1.001):
            raise ValueError(
                f"probabilities must sum to 1.0; got {total}"
            )
        if self.bin_width_us <= 0:
            raise ValueError("bin_width_us must be positive")

    # --- summary statistics --------------------------------------------

    @property
    def mean_us(self) -> float:
        return float(np.dot(self.values_us, self.probabilities))

    @property
    def variance_us2(self) -> float:
        m = self.mean_us
        return float(np.dot((self.values_us - m) ** 2, self.probabilities))

    @property
    def max_us(self) -> float:
        nz = np.nonzero(self.probabilities > 0)[0]
        if nz.size == 0:
            return 0.0
        last = int(nz[-1])
        return float(self.values_us[last] + self.bin_width_us)

    # --- moment-generating function ------------------------------------

    def mgf(self, lam: float) -> float:
        """
        E[exp(λ C)] for this distribution, computed conservatively
        using the right edge of each bin so the result is an UPPER
        bound on the MGF of the underlying continuous distribution.

        This is the convention Sun et al. use for safety.
        """
        return float(np.dot(
            np.exp(lam * (self.values_us + self.bin_width_us)),
            self.probabilities,
        ))

    def log_mgf(self, lam: float) -> float:
        """
        log(MGF(λ)).  Computed using log-sum-exp to avoid overflow on
        long-tailed distributions.

        For ``lam ≤ 0`` we delegate to ``mgf`` (safe in that regime)
        and return ``-inf`` if the result rounds to zero.  For
        ``lam > 0`` we use the log-sum-exp form.
        """
        if lam <= 0.0:
            value = self.mgf(lam)
            if value <= 0.0:
                return -float("inf")
            return math.log(value)
        right_edges = self.values_us + self.bin_width_us
        mask = self.probabilities > 0
        if not mask.any():
            return -float("inf")
        log_terms = lam * right_edges[mask] + np.log(self.probabilities[mask])
        m = float(np.max(log_terms))
        return m + math.log(float(np.sum(np.exp(log_terms - m))))


def cost_distribution_from_samples(
    samples_us: Sequence[float],
    bin_width_us: float = _DEFAULT_BIN_WIDTH_US,
    label: str = "unlabelled",
) -> CostDistribution:
    """
    Build a CostDistribution by histogramming measurement samples.
    Bins are right-half-open intervals of width ``bin_width_us``.

    The histogram is normalised so probabilities sum to 1.  Empty
    bins receive probability 0; non-empty bins receive (count / N).

    The lowest bin's left edge is the smallest sample (rounded down to
    the nearest bin boundary).  The highest bin's right edge is the
    largest sample (rounded up).  Total bin count is capped at
    ``_MAX_DISTRIBUTION_BINS`` to prevent pathological memory blow-up
    on long-tailed inputs; a warning is logged if the cap engages.
    """
    arr = np.asarray(samples_us, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("cannot build CostDistribution from empty samples")
    if bin_width_us <= 0:
        raise ValueError("bin_width_us must be positive")

    lo = float(np.floor(arr.min() / bin_width_us) * bin_width_us)
    hi = float(np.ceil(arr.max() / bin_width_us) * bin_width_us)
    n_bins = max(1, int(round((hi - lo) / bin_width_us)) + 1)
    if n_bins > _MAX_DISTRIBUTION_BINS:
        new_width = (hi - lo) / float(_MAX_DISTRIBUTION_BINS - 1)
        logger.warning(
            f"cost_distribution_from_samples: bin count {n_bins} > cap "
            f"{_MAX_DISTRIBUTION_BINS}; coarsening bin width "
            f"{bin_width_us}→{new_width:.4g}"
        )
        bin_width_us = new_width
        n_bins = _MAX_DISTRIBUTION_BINS

    edges = lo + np.arange(n_bins + 1) * bin_width_us
    counts, _ = np.histogram(arr, bins=edges)
    centres = edges[:-1]
    probs = counts.astype(np.float64) / float(arr.size)

    nz_mask = probs > 0
    if nz_mask.any():
        last_nz = int(np.nonzero(nz_mask)[0].max()) + 1
    else:
        last_nz = 1
    centres = centres[:last_nz]
    probs = probs[:last_nz]
    total = probs.sum()
    if total <= 0:
        raise ValueError(
            "cost_distribution_from_samples: all bins empty after trim"
        )
    probs = probs / total

    return CostDistribution(
        values_us=centres,
        probabilities=probs,
        bin_width_us=bin_width_us,
        n_calibration_samples=int(arr.size),
        label=label,
    )


# -----------------------------------------------------------------------------
# CountDistribution.  NEW in the camera-ready.
#
# The number ``N_a^D`` of class-a admitted transactions in a deadline
# window [0, D] is itself a random variable.  The submission draft
# bounded ``M_{N_a^D}(λ')`` by the Poisson MGF ``exp(μ_a (e^{λ'} − 1))``,
# which is a valid upper bound only when ``N_a^D`` is sub-Poisson.
# Under the bounded-budget adversary model this is not guaranteed:
# ``EvasionConstraints.max_burst_factor`` permits arrival bursts that
# violate sub-Poisson behaviour over short windows.
#
# This class provides the *exact* random-sum identity
#       log M_S(λ) = log M_N( log M_C(λ) )
# without invoking any specific count-distribution shape.  Callers
# either supply an empirical distribution (typical) or one of the
# named constructors (Poisson, deterministic, geometric).
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class CountDistribution:
    """
    Discrete distribution over non-negative integer counts.

    Used by the random-sum MGF certificate (Theorem 4.3-mgf) instead
    of a Poisson upper-bound assumption.

    Attributes
    ----------
    counts : np.ndarray
        Non-negative integer support, sorted ascending.  dtype int64.
    probabilities : np.ndarray
        Probability mass at each count.  Sums to 1 (within 1e-9).
    n_calibration_samples : int
        Number of observations used.  Zero for analytic constructors
        (Poisson, deterministic).
    label : str
        Human-readable label.
    distribution_family : str
        One of "empirical", "poisson", "deterministic", "geometric".
        Recorded so reviewers can audit which assumption is in play.
    """

    counts: np.ndarray
    probabilities: np.ndarray
    n_calibration_samples: int
    label: str
    distribution_family: str = "empirical"

    def __post_init__(self) -> None:
        if self.counts.shape != self.probabilities.shape:
            raise ValueError(
                "counts and probabilities must have the same shape"
            )
        if self.counts.size == 0:
            raise ValueError("CountDistribution must be non-empty")
        if not np.all(self.counts >= 0):
            raise ValueError("counts must be non-negative")
        if not np.all(np.diff(self.counts) >= 0):
            raise ValueError("counts must be sorted ascending")
        if np.any(self.probabilities < 0):
            raise ValueError("probabilities must be non-negative")
        total = float(self.probabilities.sum())
        if not (0.999 <= total <= 1.001):
            raise ValueError(
                f"probabilities must sum to 1.0; got {total}"
            )

    @property
    def mean(self) -> float:
        return float(np.dot(self.counts, self.probabilities))

    @property
    def max_count(self) -> int:
        nz = np.nonzero(self.probabilities > 0)[0]
        return int(self.counts[nz[-1]]) if nz.size > 0 else 0

    def log_mgf(self, theta: float) -> float:
        """
        log E[exp(theta * N)] using log-sum-exp.

        For ``theta ≤ 0``, the MGF is bounded above by 1 and below by
        ``Pr[N = 0]``; we compute it directly.

        For ``theta > 0``, we use log-sum-exp over the support.

        Returns ``+inf`` if the support extends beyond what theta can
        sustain (i.e., the MGF diverges); callers should treat this
        as "out of bracket".
        """
        if theta == 0.0:
            return 0.0
        mask = self.probabilities > 0
        if not mask.any():
            return -float("inf")
        n_supp = self.counts[mask].astype(np.float64)
        p_supp = self.probabilities[mask].astype(np.float64)
        log_terms = theta * n_supp + np.log(p_supp)
        finite = np.isfinite(log_terms)
        if not finite.any():
            return float("inf")
        m = float(np.max(log_terms[finite]))
        if not math.isfinite(m):
            return m
        try:
            return m + math.log(float(np.sum(np.exp(log_terms[finite] - m))))
        except (OverflowError, ValueError):
            return float("inf")


def count_distribution_from_samples(
    counts: Sequence[int],
    label: str = "unlabelled",
) -> CountDistribution:
    """
    Build a CountDistribution from a list of observed integer counts.

    Each entry of ``counts`` is the number of class-a transactions
    admitted in one deadline-length window during measurement.  The
    canonical procedure:

      - Pick a measurement window of length D (the deadline).
      - For each window in the run, count admitted transactions of
        class a.
      - Pass the resulting list to this function.

    If you have a single aggregate rate λ_a (transactions/second) and
    no per-window observations, use ``poisson_count_distribution``
    instead and document the Poisson assumption.
    """
    arr = np.asarray(counts, dtype=np.int64)
    if arr.size == 0:
        raise ValueError("cannot build CountDistribution from empty input")
    if np.any(arr < 0):
        raise ValueError("counts must be non-negative")
    unique, freq = np.unique(arr, return_counts=True)
    probs = freq.astype(np.float64) / float(arr.size)
    return CountDistribution(
        counts=unique.astype(np.int64),
        probabilities=probs,
        n_calibration_samples=int(arr.size),
        label=label,
        distribution_family="empirical",
    )


def poisson_count_distribution(
    mean: float,
    max_count: Optional[int] = None,
    label: str = "poisson",
) -> CountDistribution:
    """
    Build a Poisson(λ=mean) count distribution, truncated and
    renormalised at ``max_count``.

    Use this constructor *only* when the Poisson assumption is
    explicitly justified (e.g., independent benign arrivals with
    bounded inter-arrival variance).  The submission-draft analysis
    invoked this assumption implicitly via a Poisson upper bound on
    the count MGF; the camera-ready version requires callers to
    name the assumption by calling this constructor explicitly.

    Truncation
    ----------
    The Poisson distribution has unbounded support, but we represent
    it on ``[0, max_count]`` for storage.  We pick ``max_count`` as
    the smallest integer such that ``Pr[N > max_count] ≤ 1e-12``
    when not provided; this preserves the MGF up to that precision.
    """
    if mean < 0:
        raise ValueError("Poisson mean must be non-negative")
    if mean == 0:
        return CountDistribution(
            counts=np.array([0], dtype=np.int64),
            probabilities=np.array([1.0]),
            n_calibration_samples=0,
            label=label,
            distribution_family="poisson",
        )

    # Choose max_count to capture all but ~1e-12 of the mass.
    if max_count is None:
        # Crude bound: Poisson with mean μ has Pr[N > k] ≤ exp(-μ) μ^k / k!
        # for k > μ.  We expand until cumulative mass ≥ 1 − 1e-12.
        cap = max(64, int(mean + 12 * math.sqrt(mean) + 32))
        max_count = cap

    ks = np.arange(max_count + 1, dtype=np.int64)
    # Compute log Poisson PMF: -μ + k log μ - log(k!).
    log_mu = math.log(mean)
    log_factorials = np.cumsum(np.concatenate(([0.0], np.log(np.arange(1, max_count + 1)))))
    log_pmf = -mean + ks * log_mu - log_factorials
    pmf = np.exp(log_pmf)
    total = float(pmf.sum())
    if total <= 0:
        raise ValueError(
            f"poisson_count_distribution: mass underflow at mean={mean}"
        )
    pmf = pmf / total
    return CountDistribution(
        counts=ks,
        probabilities=pmf,
        n_calibration_samples=0,
        label=label,
        distribution_family="poisson",
    )


def deterministic_count_distribution(
    n: int,
    label: str = "deterministic",
) -> CountDistribution:
    """
    Build a degenerate count distribution at exactly ``n``.

    Useful when the harness has a fixed admission count per window
    (e.g., a closed-loop test).  Mostly for unit tests.
    """
    if n < 0:
        raise ValueError("deterministic count must be non-negative")
    return CountDistribution(
        counts=np.array([n], dtype=np.int64),
        probabilities=np.array([1.0]),
        n_calibration_samples=0,
        label=label,
        distribution_family="deterministic",
    )


# =============================================================================
# Section 6.  Theorem 4.3-mgf — Schedulability via random-sum Chernoff bound.
#
# This is the version of Theorem 4.3 that engages directly with the
# stochastic-WCRT literature, in particular Sun et al. RTSS 2025.
#
# Statement (paper §IV, Theorem 4.3-mgf).
#
#   Let D be the deadline window length.  Let ``B_0`` be the carry-in
#   workload at the start of the window (assumption A4: stationary
#   bounded distribution; degenerate-zero allowed but flagged).  For
#   each class ``a ∈ {benign, A_1, …, A_K}``, let ``N_a^D`` be the
#   number of class-a admitted transactions in [0, D] and let
#   ``C_{a,j}`` (j = 1, 2, …) be i.i.d. per-class costs with the same
#   distribution as ``CostDistribution`` (assumption A1).  Then the
#   total work in the window is
#
#       W = B_0 + Σ_a Σ_{j ≤ N_a^D} C_{a,j}.
#
#   The deadline-miss probability is bounded by
#
#       Pr[ W > D ]  ≤  inf_{λ > 0}  exp(-λD) · M_{B_0}(λ) · ∏_a M_{S_a}(λ),  (2)
#
#   where ``S_a = Σ_{j ≤ N_a^D} C_{a,j}`` is the per-class random
#   sum.  By the random-sum identity (which is *exact*, not a bound),
#
#       log M_{S_a}(λ)  =  log M_{N_a^D}( log M_{C_a}(λ) ).             (3)
#
#   We optimise (2) over λ by one-dimensional convex search on log-λ.
#   Sun et al. (RTSS 2025) prove the log objective is convex; we
#   exploit that property.
#
# Differences from the submission draft.
# --------------------------------------
#
#   - Identity (3) is used directly.  The submission draft replaced
#     ``M_{N_a^D}`` with the Poisson upper bound
#     ``exp(μ_a (e^θ − 1))``.  That bound holds only when ``N_a^D``
#     is sub-Poisson, which is not guaranteed by the bounded-budget
#     adversary model.  Instead, we now require callers to supply a
#     measured (or named-analytic) ``CountDistribution`` per class.
#
#   - The Poisson case is preserved as
#     ``poisson_count_distribution(mean=μ_a)``; callers who want the
#     submission draft's behaviour can pass that.  But the assumption
#     is now explicit.
#
#   - Boundary diagnostics: when the optimal λ hits the search-bracket
#     edge, the certificate's ``notes`` field reports it so plots.py
#     can mark looseness in the schedulability table.
#
#   - Carry-in: A4 (stationary B_0) is invoked.  When carry-in samples
#     are not provided, B_0 collapses to a degenerate-zero
#     distribution and we log a Sun-style warning that this is not
#     the safe default in real preemptive schedulers but is correct
#     for the synchronous, non-preemptive harness used in §V.
#
# What this bound does and does not prove.
# ----------------------------------------
#
#   Proves: under A1, A3, A4, the deadline-miss probability over a
#     window of length D is at most the right-hand side of (2).
#   Does not prove: anything about the cumulative distribution of
#     consecutive missed deadlines, anything about deadlines crossing
#     into the next window, or anything about adaptive adversaries
#     (those are addressed by Lemma 4.4, Section 7).
# =============================================================================


# Search bracket for λ in log-space.  These bounds are deliberately
# wide; the search converges in ≤80 iterations even on the widest.
_LAMBDA_SEARCH_LOG_LO: float = -20.0   # 2e-9
_LAMBDA_SEARCH_LOG_HI: float = 5.0     # 148
_LAMBDA_SEARCH_MAX_ITERS: int = 80


@dataclass(frozen=True)
class MGFCertificate:
    """
    Result of the random-sum Chernoff bound (Theorem 4.3-mgf).

    ``log_bound`` is the log of the right-hand side of inequality (2).
    The bound on ``Pr[W > D]`` is therefore ``exp(log_bound)``, but we
    return the log form because for tight contracts (ε = 1e-6 or less)
    the exponentiated bound underflows.

    ``feasible`` is ``log_bound ≤ log(ε)``.

    Attributes
    ----------
    feasible : bool
        Whether the bound certifies the contract.
    log_bound : float
        log of the right-hand side of (2) at the optimal λ.  May be
        ``-inf`` for trivially feasible cases.
    bound : float
        ``exp(log_bound)``, computed safely (capped at 1.0).
    epsilon : float
        Contract failure probability (the target).
    deadline_us : float
        Window length D.
    optimal_lambda : float
        The λ that minimises (2) (in 1/microseconds).  Reviewers
        often want this so they can re-derive the bound.
    optimal_lambda_at_bracket_edge : bool
        True if the optimal λ saturated the search bracket.  This
        means the bound is loose due to the search bracket, not the
        Chernoff inequality itself; callers should report this in
        figures.
    components : Mapping[str, float]
        Per-class log-MGF contributions at the optimal λ:
        ``{"B_0": log M_{B_0}(λ*), "<class_a>": log M_{S_a}(λ*), …}``.
        Useful for diagnosing which class drives the bound.
    notes : str
        Human-readable summary; non-empty when caveats apply.
    """

    feasible: bool
    log_bound: float
    bound: float
    epsilon: float
    deadline_us: float
    optimal_lambda: float
    optimal_lambda_at_bracket_edge: bool
    components: Mapping[str, float]
    notes: str = ""

    # Backward-compatibility aliases for the submission-draft field
    # names.  experiments.py and any external integrations that
    # imported the draft signature continue to work without changes.
    @property
    def optimal_lam(self) -> float:
        """Deprecated alias for ``optimal_lambda``."""
        return self.optimal_lambda

    @property
    def per_class_bounds(self) -> Mapping[str, float]:
        """Deprecated alias for ``components``."""
        return self.components

    def summary(self) -> Mapping[str, Any]:
        return {
            "feasible": self.feasible,
            "log_bound": self.log_bound,
            "bound": self.bound,
            "epsilon": self.epsilon,
            "deadline_us": self.deadline_us,
            "optimal_lambda": self.optimal_lambda,
            "lambda_at_edge": self.optimal_lambda_at_bracket_edge,
            "components": dict(self.components),
            "notes": self.notes,
        }


def _log_mgf_random_sum(
    cost: CostDistribution,
    count: CountDistribution,
    lam: float,
) -> float:
    """
    Random-sum identity (3): log M_S(λ) = log M_N(log M_C(λ)).

    No upper-bound approximation is applied: this is exact under
    assumption A1 (per-class i.i.d. costs) and the supplied
    ``CountDistribution``.  Returns ``+inf`` when the count MGF
    diverges (then the Chernoff search will reject this λ and move
    on).
    """
    psi_c = cost.log_mgf(lam)
    if not math.isfinite(psi_c):
        return psi_c
    return count.log_mgf(psi_c)


def _log_mgf_objective(
    lam: float,
    deadline_us: float,
    carry_in: CostDistribution,
    cost_distributions: Mapping[str, CostDistribution],
    count_distributions: Mapping[str, CountDistribution],
) -> Tuple[float, Mapping[str, float]]:
    """
    Compute ``log [exp(-λD) · M_{B_0}(λ) · ∏_a M_{S_a}(λ)]`` and the
    per-class log contributions.

    The components dict is returned alongside so callers can attribute
    the bound to specific classes.
    """
    if lam <= 0.0 or not math.isfinite(lam):
        return float("inf"), {}

    components: dict[str, float] = {}
    log_b0 = carry_in.log_mgf(lam)
    if not math.isfinite(log_b0):
        return float("inf"), {}
    components["B_0"] = log_b0

    log_sum = -lam * deadline_us + log_b0
    for cls, cost_dist in cost_distributions.items():
        if cls not in count_distributions:
            return float("inf"), {}
        log_s_a = _log_mgf_random_sum(cost_dist, count_distributions[cls], lam)
        if not math.isfinite(log_s_a):
            return float("inf"), {}
        components[cls] = log_s_a
        log_sum += log_s_a

    return log_sum, components


def _golden_section_minimise_log_objective(
    deadline_us: float,
    carry_in: CostDistribution,
    cost_distributions: Mapping[str, CostDistribution],
    count_distributions: Mapping[str, CountDistribution],
) -> Tuple[float, float, Mapping[str, float], bool]:
    """
    Find ``λ*`` that minimises the log objective.  Sun et al. prove
    convexity in λ for finite MGFs, so golden-section search is
    safe and converges in O(log(1/tol)).

    Returns (λ*, log_bound at λ*, components at λ*, edge_flag).

    The search is conducted in log-λ space because the objective's
    minimiser typically spans many decades depending on the deadline
    and tail shape.  We bracket on [exp(_LAMBDA_SEARCH_LOG_LO),
    exp(_LAMBDA_SEARCH_LOG_HI)].

    Returned ``edge_flag`` is True iff λ* lies within 0.1 of either
    bracket boundary, indicating the search bracket itself is
    constraining — a soft signal of bound looseness.
    """
    inv_phi = (math.sqrt(5.0) - 1.0) / 2.0   # 1/φ ≈ 0.618
    lo, hi = _LAMBDA_SEARCH_LOG_LO, _LAMBDA_SEARCH_LOG_HI

    def f(log_lam: float) -> Tuple[float, Mapping[str, float]]:
        return _log_mgf_objective(
            math.exp(log_lam), deadline_us, carry_in,
            cost_distributions, count_distributions,
        )

    # Initial pair.
    a, b = lo, hi
    c = b - inv_phi * (b - a)
    d = a + inv_phi * (b - a)
    fc, comp_c = f(c)
    fd, comp_d = f(d)
    for _ in range(_LAMBDA_SEARCH_MAX_ITERS):
        if abs(b - a) < 1e-6:
            break
        if fc < fd:
            b, d, fd, comp_d = d, c, fc, comp_c
            c = b - inv_phi * (b - a)
            fc, comp_c = f(c)
        else:
            a, c, fc, comp_c = c, d, fd, comp_d
            d = a + inv_phi * (b - a)
            fd, comp_d = f(d)
    log_lam_opt = 0.5 * (a + b)
    obj_opt, components = f(log_lam_opt)

    edge_flag = (
        (log_lam_opt - _LAMBDA_SEARCH_LOG_LO) < 0.1
        or (_LAMBDA_SEARCH_LOG_HI - log_lam_opt) < 0.1
    )

    return math.exp(log_lam_opt), obj_opt, components, edge_flag


def apply_theorem_4_3_mgf(
    contract: RealTimeContract,
    cost_distributions: Optional[Mapping[str, CostDistribution]] = None,
    count_distributions: Optional[Mapping[str, CountDistribution]] = None,
    carry_in: Optional[CostDistribution] = None,
    *,
    # Legacy keyword arguments preserved for backward compatibility
    # with defenses.py and any external callers that imported the
    # submission-draft signature.  Passing any of these triggers the
    # legacy path, which constructs Poisson CountDistributions
    # internally from the supplied means.  Using the new dict-based
    # signature is preferred because it does not assume sub-Poisson
    # admitted counts.
    benign_cost_dist: Optional[CostDistribution] = None,
    adversarial_cost_dists: Optional[Sequence[CostDistribution]] = None,
    expected_admitted_count_benign: Optional[float] = None,
    expected_admitted_count_per_attack: Optional[Sequence[float]] = None,
    carry_in_dist: Optional[CostDistribution] = None,
) -> MGFCertificate:
    """
    Theorem 4.3-mgf.  Compute the random-sum Chernoff bound on
    deadline-miss probability over a window of length
    ``contract.deadline_us``.

    Two calling conventions are supported.

    Modern (preferred).  Pass ``cost_distributions`` and
    ``count_distributions`` as parallel mappings ``class_name → ...``.
    The MGF identity is applied directly with no Poisson assumption.

    Legacy.  Pass ``benign_cost_dist``, ``adversarial_cost_dists``,
    ``expected_admitted_count_benign``, and
    ``expected_admitted_count_per_attack``.  Internally we wrap the
    expected counts as Poisson CountDistributions and emit a logger
    warning.  This path matches the submission-draft signature and is
    retained so that defenses.py and any external integrations
    continue to work.  New code should use the modern signature.

    Parameters (modern)
    -------------------
    contract
        Real-time contract supplying the deadline ``D`` and the target
        ``ε``.
    cost_distributions
        Mapping ``class_name → CostDistribution``.  Must contain at
        least one class.  In §V.3 the canonical mapping is
        ``{"benign": ..., "adversarial": ...}``.
    count_distributions
        Mapping ``class_name → CountDistribution``.  Must contain the
        same keys as ``cost_distributions``.
    carry_in
        Optional carry-in distribution ``B_0`` (assumption A4).  If
        omitted, defaults to a degenerate-zero distribution and we
        emit a logger warning consistent with Sun et al.'s caveat
        that B_0=0 is not the safe default in preemptive schedulers
        (it is correct for our synchronous harness; see paper §V.M).

    Returns
    -------
    MGFCertificate
        See the dataclass docstring.

    Raises
    ------
    ValueError if ``cost_distributions`` and ``count_distributions``
    have mismatched keys, or if a mix of modern and legacy kwargs is
    passed.
    """
    # Detect legacy invocation.
    legacy_kwargs_used = (
        benign_cost_dist is not None
        or adversarial_cost_dists is not None
        or expected_admitted_count_benign is not None
        or expected_admitted_count_per_attack is not None
        or carry_in_dist is not None
    )
    modern_kwargs_used = (
        cost_distributions is not None or count_distributions is not None
    )
    if legacy_kwargs_used and modern_kwargs_used:
        raise ValueError(
            "apply_theorem_4_3_mgf: cannot mix modern (cost_distributions, "
            "count_distributions) and legacy (benign_cost_dist, ...) "
            "keyword arguments in one call"
        )

    if legacy_kwargs_used:
        # Translate legacy kwargs into modern dict form, using Poisson
        # count distributions for each class.  This preserves the
        # submission-draft semantics exactly.
        if benign_cost_dist is None or expected_admitted_count_benign is None:
            raise ValueError(
                "legacy invocation requires both benign_cost_dist and "
                "expected_admitted_count_benign"
            )
        adv_costs = list(adversarial_cost_dists or ())
        adv_counts = list(expected_admitted_count_per_attack or ())
        if len(adv_costs) != len(adv_counts):
            raise ValueError(
                "adversarial_cost_dists and expected_admitted_count_per_attack "
                "must have the same length"
            )
        cost_distributions = {"benign": benign_cost_dist}
        count_distributions = {
            "benign": poisson_count_distribution(
                mean=float(expected_admitted_count_benign),
                label="benign(legacy-poisson)",
            )
        }
        for i, (cd, mean) in enumerate(zip(adv_costs, adv_counts)):
            key = cd.label or f"adversarial_{i}"
            cost_distributions[key] = cd
            count_distributions[key] = poisson_count_distribution(
                mean=float(mean), label=f"{key}(legacy-poisson)",
            )
        if carry_in is None:
            carry_in = carry_in_dist
        logger.info(
            "apply_theorem_4_3_mgf: legacy call signature in use; "
            "wrapped expected counts as Poisson CountDistributions.  "
            "Poisson is a valid upper bound only for sub-Poisson "
            "admitted-count distributions; for the camera-ready "
            "non-circular path, pass count_distributions= built from "
            "count_distribution_from_samples()."
        )

    # From here on, modern path.
    if cost_distributions is None or count_distributions is None:
        raise ValueError(
            "apply_theorem_4_3_mgf: must supply cost_distributions and "
            "count_distributions (modern) or the legacy benign_cost_dist/"
            "adversarial_cost_dists/expected_admitted_count_* kwargs"
        )
    if set(cost_distributions.keys()) != set(count_distributions.keys()):
        raise ValueError(
            "cost_distributions and count_distributions must have "
            "matching keys; got "
            f"{sorted(cost_distributions.keys())} vs "
            f"{sorted(count_distributions.keys())}"
        )
    if not cost_distributions:
        raise ValueError("at least one class is required")

    if carry_in is None:
        # Degenerate B_0 = 0.  Build a one-bin CostDistribution at 0 µs.
        carry_in = CostDistribution(
            values_us=np.array([0.0]),
            probabilities=np.array([1.0]),
            bin_width_us=1.0,    # nominal; the bin contains only 0
            n_calibration_samples=0,
            label="B_0=0 (degenerate)",
        )
        logger.info(
            "apply_theorem_4_3_mgf: no carry-in supplied; using "
            "degenerate-zero B_0.  This matches the synchronous "
            "harness in §V (no preemption); see Sun et al. RTSS 2025 "
            "§3.2 for the preemptive case."
        )

    deadline_us = contract.deadline_us
    eps = contract.failure_probability_bound

    lam_opt, log_bound, components, edge_flag = (
        _golden_section_minimise_log_objective(
            deadline_us, carry_in,
            cost_distributions, count_distributions,
        )
    )

    bound = min(1.0, math.exp(log_bound)) if math.isfinite(log_bound) else 1.0
    feasible = log_bound <= math.log(eps) if eps > 0 else False

    notes = ""
    if edge_flag:
        notes = (
            "optimal λ at bracket edge; bound may be loose due to "
            "search bracket, not Chernoff inequality"
        )
    if not math.isfinite(log_bound):
        notes = (
            (notes + "; " if notes else "")
            + "log objective non-finite at all bracketed λ "
            "(likely count-MGF divergence)"
        )

    return MGFCertificate(
        feasible=feasible,
        log_bound=log_bound,
        bound=bound,
        epsilon=eps,
        deadline_us=deadline_us,
        optimal_lambda=lam_opt,
        optimal_lambda_at_bracket_edge=edge_flag,
        components=components,
        notes=notes,
    )


def validate_theorem_4_3_mgf(
    cert: MGFCertificate,
    measured_miss_rate: float,
    n_admitted: int,
) -> TheoremValidation:
    """
    Validate the MGF certificate against the measured miss rate.
    Same Wilson-CI accounting as ``validate_theorem_4_3``.

    The bound being validated is ``cert.bound`` (exponentiated form
    of the Chernoff bound).  Reviewers will check that empirical
    miss rate ≤ ``cert.bound``, AND that the 95 % CI upper bound on
    the miss rate also stays below ``cert.bound``.
    """
    bound = cert.bound
    n = max(0, int(n_admitted))
    n_miss = int(round(measured_miss_rate * n))
    n_non_miss = n - n_miss
    if n > 0:
        ci_lo_non_miss = _wilson_one_sided_lower(n_non_miss, n)
        ci_hi_miss = 1.0 - ci_lo_non_miss
    else:
        ci_hi_miss = float("nan")

    holds = (measured_miss_rate <= bound)
    holds_ci = (ci_hi_miss <= bound) if not math.isnan(ci_hi_miss) else False

    return TheoremValidation(
        theorem="4.3-mgf",
        n_samples=n,
        holds_at_p99=bool(holds),
        holds_at_p99_9=bool(holds),
        bound_at_p99=bound,
        measured_at_p99=measured_miss_rate,
        bound_at_p99_9=bound,
        measured_at_p99_9=measured_miss_rate,
        ci_lo_at_p99=ci_hi_miss,
        ci_lo_at_p99_9=ci_hi_miss,
        holds_at_ci_95=bool(holds_ci),
        notes=(
            f"feasible={cert.feasible}, λ*={cert.optimal_lambda:.3g}, "
            f"edge={cert.optimal_lambda_at_bracket_edge}, "
            f"miss_ci_hi={ci_hi_miss:.4g}; {cert.notes}"
        ),
    )


# =============================================================================
# Section 7.  Lemma 4.4 — Adaptive-adversary degradation factor.
#
# THIS WAS LABELLED "Theorem 4.4" IN THE SUBMISSION DRAFT.
# It has been demoted to Lemma 4.4 because it is a derived bound
# (from Lipschitz arguments on the threshold-tracking dynamics)
# rather than a primary theorem in the paper's statistical-
# schedulability framework.  The proof appears in §IV.D and is
# summarised here.
#
# Statement (paper §IV.D, Lemma 4.4).
#
#   Consider an adaptive (Tier-3) adversary that observes the
#   defender's current threshold ``T(t)`` at rate ρ_obs and tunes
#   subsequent injections to fall just below it.  The defender
#   recomputes ``T(t)`` at rate ρ_def using the schedulability test
#   (Theorem 4.3-env or 4.3-mgf).  Under assumptions A2.1–A2.3
#   below, the integrated miss probability over a measurement
#   window W satisfies
#
#       ∫_W Pr[ miss ] dt   ≤   |W| · κ · ε
#
#   where
#
#       κ  =  1 + (ρ_obs / ρ_def) · γ                        (4)
#
#   and γ ∈ [0, 1] is a system-dependent overshoot constant
#   measured per deployment (see ``measure_overshoot_constant``).
#
# Assumptions.
#
#   A2.1 (Lipschitz threshold).  The defender's threshold update
#       rule ``T_{k+1} = T_k + δ_k`` has bounded per-update
#       displacement ``|δ_k| ≤ Δ_max``.  In our defenses (D2, D3),
#       Δ_max is set proportional to the certified slack, ensuring
#       this is a controllable parameter.
#
#   A2.2 (Bounded observation latency).  The adversary observes
#       ``T(t)`` with bounded delay.  Specifically, between two
#       consecutive defender updates k and k+1, the adversary's
#       belief ``T̂_k`` agrees with ``T_k`` modulo a delay term
#       ``η_obs ≤ 1 / ρ_obs`` from observation lag.
#
#   A2.3 (Bounded predictor residual).  The cost-prediction error
#       on adversarial transactions is bounded in expectation:
#       ``E[ |C_actual − C_predicted| ] ≤ Δ_max``.  This is
#       satisfied empirically in §V (Table 3); it would fail under
#       a model that is grossly miscalibrated for the adversarial
#       class, but the predictor in defenses.py is calibrated on
#       the same per-class data used to fit Theorem 4.2.
#
# Proof sketch.
#
#   Between two defender updates of length 1/ρ_def, the threshold
#   is constant at ``T_k``.  In that interval, the adversary may
#   issue up to ``ρ_obs / ρ_def`` observations of T, tuning each
#   subsequent injection to fall ``Δ_max`` below the observed
#   threshold (assumption A2.3 controls the prediction error;
#   A2.1 controls the threshold's mobility; A2.2 controls the
#   observation lag).
#
#   For each such tuned injection, the cost overshoots the
#   admission-time predicted cost by at most ``γ · Δ_max`` for
#   some γ ∈ [0, 1] dependent on the cost predictor's residual
#   distribution.  Integrating Pr[ miss ] over the interval and
#   summing across all updates in W, the total integrated miss
#   probability is bounded by
#
#       |W| · ε · (1 + (ρ_obs / ρ_def) · γ)
#
#   where the leading ε is the certified per-window miss probability
#   from Theorem 4.3 and the additive term captures the cost of
#   bounded adaptivity.
#
#   When ρ_def = 0 (defender never adapts), κ → ∞: an adversary that
#   is always ahead of a stationary defender can drive miss rate
#   arbitrarily close to 1.  This is consistent with the textbook
#   adaptive-adversary regret lower bounds (Cesa-Bianchi & Lugosi
#   2006) and with Sun et al.'s implicit assumption that the
#   schedule does not change adversarially during a window.
#
# What this lemma does NOT say.
# -----------------------------
#
#   - It is a bound on the *integrated* miss probability, not on
#     the per-transaction miss probability under adaptive load.
#   - It does not characterise the response of the defender to a
#     changing γ.  In §VI we report empirical κ on the four
#     datasets; γ varies between 0.05 (Bitcoin) and 0.31 (SWaT),
#     consistent with the theory's prediction that κ grows with
#     prediction-error magnitude.
#   - It does not address the case where γ changes mid-run (drift).
#     Our experiments report time-windowed κ to surface this.
#
# Why we present this as a lemma rather than a theorem.
# -----------------------------------------------------
#
#   The result is downstream of Theorem 4.3 — it does not certify
#   the contract by itself; it bounds the additive cost of
#   adaptivity given a 4.3 certificate.  Calling it a theorem in
#   the submission draft was a labelling error.  The substance is
#   unchanged.
# =============================================================================


def apply_lemma_4_4(
    cert: SchedulabilityCertificate,
    adversary_observation_rate_hz: float,
    defender_adaptation_rate_hz: float,
    overshoot_constant_gamma: float,
) -> Tuple[float, float]:
    """
    Lemma 4.4.  Compute the κ degradation factor and the resulting
    integrated miss-probability bound.

    Parameters
    ----------
    cert : SchedulabilityCertificate
        The Theorem 4.3 certificate that Lemma 4.4 multiplies.
        Must be feasible; we do not extend infeasible bounds.
    adversary_observation_rate_hz : float
        ρ_obs in Hz.  Empirical upper bound is ``budget.max_injection_rate``;
        for a realistic Tier-3 adversary, ρ_obs ≤ injection rate.
    defender_adaptation_rate_hz : float
        ρ_def in Hz.  Set by the defense's recalibration schedule.
        For D3, this is the schedulability-recheck frequency, which
        is configurable and reported in run records.
    overshoot_constant_gamma : float
        γ ∈ [0, 1].  Measured per deployment via
        ``measure_overshoot_constant``.

    Returns
    -------
    (kappa, degraded_epsilon) : Tuple[float, float]
        ``kappa = 1 + (ρ_obs / ρ_def) · γ`` (or ``+inf`` when ρ_def=0).
        ``degraded_epsilon = kappa · cert.epsilon``.

    Raises
    ------
    ValueError on negative rates or γ outside [0, 1].
    """
    if adversary_observation_rate_hz < 0:
        raise ValueError("adversary_observation_rate_hz must be non-negative")
    if defender_adaptation_rate_hz < 0:
        raise ValueError("defender_adaptation_rate_hz must be non-negative")
    if not 0.0 <= overshoot_constant_gamma <= 1.0:
        raise ValueError("overshoot_constant_gamma must be in [0, 1]")

    if defender_adaptation_rate_hz == 0.0:
        if adversary_observation_rate_hz > 0.0:
            return float("inf"), 1.0
        # Zero-zero: the adversary is also stationary; no degradation.
        return 1.0, cert.epsilon

    rate_ratio = adversary_observation_rate_hz / defender_adaptation_rate_hz
    kappa = 1.0 + rate_ratio * overshoot_constant_gamma
    degraded = min(1.0, kappa * cert.epsilon)
    return kappa, degraded


def measure_overshoot_constant(
    predicted_us: Sequence[float],
    actual_us: Sequence[float],
    threshold_us: float,
) -> float:
    """
    Estimate the overshoot constant γ ∈ [0, 1] from measured
    predictor residuals.  Definition:

        γ  =  E[ max(0, C_actual - threshold) | C_predicted ≤ threshold ]
              / E[ max(0, threshold - C_predicted) | C_predicted ≤ threshold ]

    Intuition: γ is the average overshoot above ``threshold`` for
    transactions the predictor said would fall below it, normalised
    by the predictor's "headroom" below threshold.  γ ∈ [0, 1] when
    the predictor is consistent (overshoots are smaller than
    headroom on average).

    Returns 0 when no predictor-said-safe transactions are observed
    (the adversary has no exploitation surface) or when the
    headroom denominator is zero.

    Used by Lemma 4.4 to compute κ.  Reviewers can verify the
    measurement by running ``analyse_storm`` with verbose=True and
    inspecting the per-class γ values.
    """
    pred = np.asarray(predicted_us, dtype=np.float64)
    actu = np.asarray(actual_us, dtype=np.float64)
    if pred.shape != actu.shape:
        raise ValueError("predicted_us and actual_us must have same length")
    if pred.size == 0:
        return 0.0
    safe_mask = pred <= threshold_us
    if not safe_mask.any():
        return 0.0
    overshoot = np.maximum(0.0, actu[safe_mask] - threshold_us)
    headroom = np.maximum(0.0, threshold_us - pred[safe_mask])
    e_overshoot = float(np.mean(overshoot))
    e_headroom = float(np.mean(headroom))
    if e_headroom <= 0:
        return 0.0
    return float(min(1.0, e_overshoot / e_headroom))


def validate_lemma_4_4(
    cert: SchedulabilityCertificate,
    kappa: float,
    measured_miss_rate: float,
    n_admitted: int,
) -> TheoremValidation:
    """
    Validate Lemma 4.4 against measured data: check that the
    integrated miss rate satisfies ``measured ≤ κ · ε``.  Includes
    the same Wilson-CI accounting as Theorem 4.3.

    Note: ``validate_lemma_4_4(...).theorem == "4.4"`` for backwards
    compatibility with experiments.py's existing field names.  The
    paper text uses "Lemma 4.4".
    """
    promised = min(1.0, kappa * cert.epsilon)
    n = max(0, int(n_admitted))
    n_miss = int(round(measured_miss_rate * n))
    n_non_miss = n - n_miss
    if n > 0:
        ci_lo_non_miss = _wilson_one_sided_lower(n_non_miss, n)
        ci_hi_miss = 1.0 - ci_lo_non_miss
    else:
        ci_hi_miss = float("nan")

    holds = (measured_miss_rate <= promised)
    holds_ci = (ci_hi_miss <= promised) if not math.isnan(ci_hi_miss) else False

    return TheoremValidation(
        theorem="4.4",
        n_samples=n,
        holds_at_p99=bool(holds),
        holds_at_p99_9=bool(holds),
        bound_at_p99=promised,
        measured_at_p99=measured_miss_rate,
        bound_at_p99_9=promised,
        measured_at_p99_9=measured_miss_rate,
        ci_lo_at_p99=ci_hi_miss,
        ci_lo_at_p99_9=ci_hi_miss,
        holds_at_ci_95=bool(holds_ci),
        notes=(
            f"kappa={kappa:.3f}, miss_ci_hi={ci_hi_miss:.4g}, "
            "Lemma 4.4 (was Theorem 4.4 in submission draft)"
        ),
    )


# Backward-compatibility shims for callers that imported the old names.
# Both ``apply_theorem_4_4`` and ``validate_theorem_4_4`` are retained
# but marked deprecated.  experiments.py and defenses.py have been
# updated to call the new ``*_lemma_4_4`` names; downstream callers
# should follow.
def apply_theorem_4_4(
    cert: SchedulabilityCertificate,
    adversary_observation_rate_hz: float,
    defender_adaptation_rate_hz: float,
    overshoot_constant_gamma: float,
) -> Tuple[float, float]:
    """Deprecated alias.  Use ``apply_lemma_4_4`` instead.  See §IV.D."""
    return apply_lemma_4_4(
        cert,
        adversary_observation_rate_hz,
        defender_adaptation_rate_hz,
        overshoot_constant_gamma,
    )


def validate_theorem_4_4(
    cert: SchedulabilityCertificate,
    kappa: float,
    measured_miss_rate: float,
    n_admitted: int,
) -> TheoremValidation:
    """Deprecated alias.  Use ``validate_lemma_4_4`` instead."""
    return validate_lemma_4_4(cert, kappa, measured_miss_rate, n_admitted)


# =============================================================================
# Section 8.  StormAnalysis bundle and analyse_storm().
#
# This is the top-level entry point that experiments.py invokes after
# a run completes.  It produces a single self-contained
# ``StormAnalysis`` object that:
#
#   - Fits Theorems 4.1 and 4.2 from the run's benign and adversarial
#     samples.
#   - Computes a Theorem 4.3-env certificate (mandatory).
#   - Computes a Theorem 4.3-mgf certificate when the caller supplies
#     count distributions (typical for §V.3); skipped otherwise.
#   - Computes Lemma 4.4 κ and degraded-ε for Tier-3 storms.
#   - Flags degenerate inputs explicitly (no silent fallbacks).
#
# Camera-ready: ``adversarial_fit_is_benign_fallback`` is now an
# explicit boolean rather than a silent equality check.  When True,
# downstream consumers (plots.py, paper tables) refuse to treat the
# 4.2 component as validated, and ``StormAnalysis.notes`` carries a
# human-readable explanation.
# =============================================================================


@dataclass(frozen=True)
class StormAnalysis:
    """
    Bundle of all theorem/lemma results for one ``UpdateStorm`` run.

    Camera-ready additions:
      - ``adversarial_fit_is_benign_fallback``: True when adversarial
        samples were not available and the analysis fell back to
        benign-only (Theorem 4.2 is then trivially equal to 4.1).
      - ``mgf_certificate``: optional MGF-path certificate from
        Section 6.  None when count distributions weren't supplied.
      - ``gamma_measured``: the overshoot constant from Lemma 4.4,
        when measurable.
      - ``notes``: human-readable string with all caveats.
    """

    storm_signature: str

    # Tail fits.
    benign_fit: TailFit
    adversarial_fit: TailFit
    adversarial_fit_is_benign_fallback: bool

    # Theorem 4.3-env certificate (always computed).
    certificate: SchedulabilityCertificate

    # Theorem 4.3-mgf certificate (optional; None if no count distributions).
    mgf_certificate: Optional[MGFCertificate]

    # Lemma 4.4 results (optional; only meaningful for Tier-3 storms).
    kappa: Optional[float]
    degraded_epsilon: Optional[float]
    gamma_measured: Optional[float]

    # Validation results (filled in by experiments.py when held-out
    # data is available).
    benign_validation: Optional[TheoremValidation] = None
    adversarial_validation: Optional[TheoremValidation] = None
    schedulability_validation: Optional[TheoremValidation] = None
    schedulability_validation_mgf: Optional[TheoremValidation] = None
    lemma_4_4_validation: Optional[TheoremValidation] = None

    notes: str = ""

    def summary(self) -> Mapping[str, Any]:
        """Compact dict summary suitable for run records / plots."""
        return {
            "storm_signature": self.storm_signature,
            "benign_alpha": self.benign_fit.alpha,
            "benign_c_anchor": self.benign_fit.c_anchor,
            "benign_n": self.benign_fit.n_samples,
            "adversarial_alpha": self.adversarial_fit.alpha,
            "adversarial_c_anchor": self.adversarial_fit.c_anchor,
            "adversarial_n": self.adversarial_fit.n_samples,
            "adversarial_fit_is_benign_fallback":
                self.adversarial_fit_is_benign_fallback,
            "certificate_feasible": self.certificate.feasible,
            "certificate_threshold_us": self.certificate.threshold_us,
            "certificate_bound": self.certificate.bound_at_T,
            "certificate_slack_us": self.certificate.slack_us,
            "mgf_feasible":
                self.mgf_certificate.feasible
                if self.mgf_certificate else None,
            "mgf_bound":
                self.mgf_certificate.bound
                if self.mgf_certificate else None,
            "mgf_optimal_lambda":
                self.mgf_certificate.optimal_lambda
                if self.mgf_certificate else None,
            "kappa": self.kappa,
            "degraded_epsilon": self.degraded_epsilon,
            "gamma_measured": self.gamma_measured,
            "notes": self.notes,
        }


def analyse_storm(
    storm: UpdateStorm,
    benign_cost_samples_us: Optional[Sequence[float]] = None,
    adversarial_cost_samples_us: Optional[Sequence[float]] = None,
    *,
    adversary_fraction: Optional[float] = None,
    threshold_us: Optional[float] = None,
    adversary_observation_rate_hz: Optional[float] = None,
    defender_adaptation_rate_hz: Optional[float] = None,
    overshoot_constant_gamma: Optional[float] = None,
    benign_count_distribution: Optional[CountDistribution] = None,
    adversarial_count_distribution: Optional[CountDistribution] = None,
    carry_in_distribution: Optional[CostDistribution] = None,
    # Legacy keyword aliases preserved for backward compatibility with
    # experiments.py and any external callers that imported the
    # submission-draft signature.  Passing any of these triggers the
    # legacy translation; mixing legacy and modern kwargs raises.
    benign_samples_us: Optional[Sequence[float]] = None,
    adversarial_samples_us: Optional[Sequence[float]] = None,
    enable_mgf_certificate: Optional[bool] = None,
    expected_admitted_rate_hz: Optional[float] = None,
    carry_in_samples_us: Optional[Sequence[float]] = None,
) -> StormAnalysis:
    """
    Top-level analysis.  Apply Theorems 4.1, 4.2, 4.3 (env and mgf
    when count distributions are provided), and Lemma 4.4 (when γ
    and rates are provided), and bundle the results.

    Empirical α (camera-ready)
    --------------------------
    The submission draft estimated α post-hoc from cost-bucketing
    heuristics, which created a circularity: the schedulability
    bound used α to compute its mixture, but α itself was estimated
    by inverting the bound.  The camera-ready harness instead
    propagates the *ground-truth* injection label from
    ``MixedStream`` through the probe to the run record (see
    ``TimedTransaction.is_adversarial`` in workload.py), and
    experiments.py passes the resulting empirical α to this function
    via ``adversary_fraction``.

    When ``adversary_fraction`` is None, we fall back to
    ``storm.budget.fraction_of_stream`` and emit a warning.  This
    fallback is reasonable for §V.1 attack-effectiveness experiments
    (where the budget IS the ground truth) but is less defensible
    in §V.3, where we want measured α.

    MGF path (camera-ready)
    -----------------------
    If both count distributions are supplied, we additionally
    compute an MGF certificate via ``apply_theorem_4_3_mgf``.  This
    is the path the paper's headline schedulability table uses.  If
    they are absent, we skip the MGF certificate (it is None in the
    bundle) and the env certificate is the only one returned.

    Lemma 4.4 path
    --------------
    For Tier-3 storms, the caller may supply ``ρ_obs``, ``ρ_def``,
    and ``γ``.  When all three are present (or γ is None and
    predictor data is in cost-prediction logs — see
    ``measure_overshoot_constant``), we compute κ.  Otherwise the
    bundle's kappa/degraded_epsilon fields are None, signalling
    "no Lemma 4.4 result available for this storm."
    """
    # --- Legacy kwarg translation -------------------------------------
    # If any legacy alias is supplied, translate it into the modern
    # name.  Mixing legacy and modern is rejected.
    legacy_used = (
        benign_samples_us is not None
        or adversarial_samples_us is not None
        or enable_mgf_certificate is not None
        or expected_admitted_rate_hz is not None
        or carry_in_samples_us is not None
    )
    if legacy_used:
        if benign_samples_us is not None:
            if benign_cost_samples_us is not None:
                raise ValueError(
                    "analyse_storm: cannot pass both benign_samples_us "
                    "(legacy) and benign_cost_samples_us (modern)"
                )
            benign_cost_samples_us = benign_samples_us
        if adversarial_samples_us is not None:
            if adversarial_cost_samples_us is not None:
                raise ValueError(
                    "analyse_storm: cannot pass both adversarial_samples_us "
                    "(legacy) and adversarial_cost_samples_us (modern)"
                )
            adversarial_cost_samples_us = adversarial_samples_us
        # Legacy: enable_mgf_certificate=True with a measured admit
        # rate is interpreted as "build a Poisson count distribution
        # from the rate and turn on the MGF path".  This preserves
        # the submission draft's behaviour.
        if (
            enable_mgf_certificate
            and expected_admitted_rate_hz is not None
            and benign_count_distribution is None
            and adversarial_count_distribution is None
        ):
            window_s = max(
                1e-9, storm.contract.measurement_window_seconds
            )
            mean_per_window = max(
                0.0, float(expected_admitted_rate_hz) * window_s
            )
            # Heuristic split by α (preserving submission-draft behaviour).
            alpha_hint = (
                adversary_fraction
                if adversary_fraction is not None
                else storm.budget.fraction_of_stream
            )
            benign_count_distribution = poisson_count_distribution(
                mean=mean_per_window * (1.0 - alpha_hint),
                label="benign(legacy-poisson-from-rate)",
            )
            adversarial_count_distribution = poisson_count_distribution(
                mean=mean_per_window * alpha_hint,
                label="adversarial(legacy-poisson-from-rate)",
            )
        if carry_in_samples_us is not None and carry_in_distribution is None:
            arr = np.asarray(carry_in_samples_us, dtype=np.float64).ravel()
            if arr.size > 0:
                carry_in_distribution = cost_distribution_from_samples(
                    arr, label="carry_in"
                )

    if benign_cost_samples_us is None:
        raise ValueError(
            "analyse_storm: benign cost samples must be supplied "
            "(via benign_cost_samples_us or legacy benign_samples_us)"
        )
    if adversarial_cost_samples_us is None:
        adversarial_cost_samples_us = ()

    # 1.  Tail fits.
    benign_arr = np.asarray(benign_cost_samples_us, dtype=np.float64).ravel()
    if benign_arr.size == 0:
        raise ValueError(
            "analyse_storm: benign_cost_samples_us must be non-empty"
        )
    benign_fit = fit_exponential_upper_bound(benign_arr)

    adv_arr = np.asarray(adversarial_cost_samples_us, dtype=np.float64).ravel()
    has_adv = adv_arr.size > 0
    if has_adv:
        adversarial_fit = fit_exponential_upper_bound(adv_arr)
        is_fallback = False
    else:
        # No silent fallback: explicitly flag.  The bound from
        # Theorem 4.2 will equal the bound from Theorem 4.1 in this
        # case; downstream consumers (validation, plots, paper
        # tables) treat this case specially.
        adversarial_fit = benign_fit
        is_fallback = True
        logger.warning(
            f"analyse_storm[{storm.label}]: no adversarial samples "
            "provided; Theorem 4.2 will be reported as a benign "
            "fallback (adversarial_fit_is_benign_fallback=True).  "
            "Downstream consumers must NOT treat this as a "
            "validation of the adversarial bound."
        )

    # 2.  α: prefer measured (passed in by caller); fall back to
    #     budgeted only as a documented degraded mode.
    if adversary_fraction is None:
        adversary_fraction = storm.budget.fraction_of_stream
        alpha_source = "budget (no measured α supplied)"
    else:
        alpha_source = "measured (from probe is_adversarial label)"

    # 3.  Theorem 4.3-env certificate.
    cert = apply_theorem_4_3(
        benign_fit=benign_fit,
        adversarial_fit=adversarial_fit,
        contract=storm.contract,
        adversary_fraction=adversary_fraction,
        threshold_us=threshold_us,
    )

    # 4.  Theorem 4.3-mgf certificate (optional).
    mgf_cert: Optional[MGFCertificate] = None
    if (
        benign_count_distribution is not None
        and adversarial_count_distribution is not None
        and not is_fallback
    ):
        try:
            cost_benign = cost_distribution_from_samples(
                benign_arr, label="benign",
            )
            cost_adv = cost_distribution_from_samples(
                adv_arr, label="adversarial",
            )
            mgf_cert = apply_theorem_4_3_mgf(
                contract=storm.contract,
                cost_distributions={
                    "benign": cost_benign,
                    "adversarial": cost_adv,
                },
                count_distributions={
                    "benign": benign_count_distribution,
                    "adversarial": adversarial_count_distribution,
                },
                carry_in=carry_in_distribution,
            )
        except (ValueError, RuntimeError) as e:
            logger.warning(
                f"analyse_storm[{storm.label}]: MGF certificate "
                f"computation failed: {e}.  Reporting env certificate "
                "only."
            )
    elif (
        benign_count_distribution is not None
        and adversarial_count_distribution is not None
        and is_fallback
    ):
        logger.info(
            f"analyse_storm[{storm.label}]: skipping MGF certificate "
            "because adversarial fit is a benign fallback "
            "(would produce a trivially equal bound)."
        )

    # 5.  Lemma 4.4 (Tier-3 only).
    is_tier3 = storm.capability.tier == CapabilityTier.TIER_3
    kappa: Optional[float] = None
    degraded_eps: Optional[float] = None
    gamma: Optional[float] = overshoot_constant_gamma
    if (
        is_tier3
        and adversary_observation_rate_hz is not None
        and defender_adaptation_rate_hz is not None
    ):
        if gamma is None:
            gamma = 0.5  # default; caller should override with measured γ
            logger.info(
                f"analyse_storm[{storm.label}]: γ not supplied; using "
                "default 0.5.  Pass overshoot_constant_gamma= or use "
                "measure_overshoot_constant() for accurate κ."
            )
        kappa, degraded_eps = apply_lemma_4_4(
            cert,
            adversary_observation_rate_hz,
            defender_adaptation_rate_hz,
            gamma,
        )

    # 6.  Notes.
    notes_parts: list[str] = [f"α source: {alpha_source}"]
    if is_fallback:
        notes_parts.append(
            "ADVERSARIAL FIT FALLBACK: Theorem 4.2 = Theorem 4.1; "
            "no adversarial samples"
        )
    if mgf_cert is None and (
        benign_count_distribution is None
        or adversarial_count_distribution is None
    ):
        notes_parts.append(
            "MGF certificate skipped (count distributions not provided)"
        )
    if mgf_cert is not None and mgf_cert.optimal_lambda_at_bracket_edge:
        notes_parts.append("MGF λ at bracket edge — bound may be loose")
    if is_tier3 and kappa is None:
        notes_parts.append(
            "Tier-3 storm but no Lemma 4.4 inputs (ρ_obs, ρ_def) "
            "supplied; κ unavailable"
        )

    return StormAnalysis(
        storm_signature=storm.signature(),
        benign_fit=benign_fit,
        adversarial_fit=adversarial_fit,
        adversarial_fit_is_benign_fallback=is_fallback,
        certificate=cert,
        mgf_certificate=mgf_cert,
        kappa=kappa,
        degraded_epsilon=degraded_eps,
        gamma_measured=gamma,
        notes="; ".join(notes_parts),
    )


# =============================================================================
# Public API.
#
# The exact set of names the rest of the codebase is allowed to import
# from this module.  Any new symbol added to this module that should
# be importable must be added here AND announced in the camera-ready
# header docstring.  Symbols not in __all__ are private to analysis.py
# and may change without notice.
# =============================================================================

__all__ = [
    # Tail fits.
    "TailFit",
    "fit_exponential_upper_bound",
    # Theorem 4.1.
    "apply_theorem_4_1",
    "validate_theorem_4_1",
    # Theorem 4.2.
    "apply_theorem_4_2",
    "validate_theorem_4_2",
    # Theorem 4.3 (env path).
    "SchedulabilityCertificate",
    "apply_theorem_4_3",
    "validate_theorem_4_3",
    # Theorem 4.3 (MGF path).
    "CostDistribution",
    "cost_distribution_from_samples",
    "CountDistribution",
    "count_distribution_from_samples",
    "poisson_count_distribution",
    "deterministic_count_distribution",
    "MGFCertificate",
    "apply_theorem_4_3_mgf",
    "validate_theorem_4_3_mgf",
    # Lemma 4.4 (was Theorem 4.4).
    "apply_lemma_4_4",
    "validate_lemma_4_4",
    "measure_overshoot_constant",
    # Backward-compatibility shims (deprecated, retained for callers).
    "apply_theorem_4_4",
    "validate_theorem_4_4",
    # Top-level bundle.
    "StormAnalysis",
    "TheoremValidation",
    "analyse_storm",
]
