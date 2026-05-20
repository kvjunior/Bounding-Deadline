"""
plots.py — Figures and tables for the paper.

One function per figure or table.  Each function reads run records
written by experiments.py and produces the corresponding paper
artefact under results/figures/.  Naming is explicit: figure_3()
produces Figure 3, table_2() produces Table 2.  A reviewer who
wants to verify a number in the paper can grep this file for the
table or figure name and read exactly one function.

Discipline
----------
- No interactive plotting.  Every figure goes to a PDF/PNG file.
- Tables emitted in BOTH LaTeX (for the paper) and CSV (for the
  README and reviewer verification).  The same data appears in both,
  so a reviewer can reconcile them.
- matplotlib only.  No seaborn or plotly: their default styles drift
  between versions, which is the enemy of reproducibility.
- Colors picked explicitly from a small accessible palette; no
  reliance on default cycle.
- Every function takes a `records` argument (list of RunRecords or
  Path to a directory of run records).  This makes per-figure
  regeneration cheap.

Mapping to the paper (camera-ready)
-----------------------------------
table_1  — System and Defense comparison        (Table 1, §I)
table_2  — Bound validation summary             (Table 2, §V.3)
table_3  — Attack effectiveness per dataset     (Table 3, §V.1)
table_4  — Defense deadline-miss rates          (Table 4, §V.2)
table_5  — Defense ablation                     (Table 5, §V.4)
table_6  — Adaptive-adversary case study        (Table 6, §VI)
table_7  — Age-violation rates (AoI)            (Table 7, §V.2)
table_8  — Schedulability certificate validation (Table 8, §V.3)
table_9  — Recovery time per defense             (Table 9, §V.4)

figure_3 — Bound validation CCDF curves         (Figure 3, §V.3)
figure_4 — Per-attack latency CCDF              (Figure 4, §V.1)
figure_5 — Defense deadline-miss bar chart      (Figure 5, §V.2)
figure_6 — Scalability curves                   (Figure 6, §V.3)
figure_7 — Cross-dataset comparison             (Figure 7, §V.5)
figure_8 — Adaptive-adversary kappa trajectory  (Figure 8, §VI)
figure_9 — Storm dynamics: latency, queue, throughput  (Figure 9, §V.1)
figure_10 — Freshness CCDFs (model age + propagation) (Figure 10, §V.2)

Camera-ready improvements over the submission draft
---------------------------------------------------
1. **Filter semantics fixed.**  The submission-draft
   ``LoadedRecords.filter`` used ``if attack is not None and
   r.get("attack") != attack: continue`` — meaning ``None`` was
   silently interpreted as "do not filter on this field."  All
   call sites — ``table_3``, ``table_4``, ``table_5``, ``table_6``,
   ``table_7``, ``table_9``, ``figure_3``, ``figure_5`` — used
   the pattern ``d_filter = None if defense == "no_defense" else
   defense`` *intending* ``None`` to mean "filter to records with
   no defense set."  In the submission draft those calls
   silently aggregated over every defense, so the "no-defense"
   row of every table mixed in D1, D2, D3 records.  The
   camera-ready introduces an ``_ANY`` sentinel as the default;
   passing ``None`` now correctly filters to records where the
   field is None, which is what every caller intended.  Existing
   call sites work unchanged because they all already pass None.

2. **A6 colour added to the attack palette.**  Camera-ready
   ``attacks.py`` exposes A6_evolutionary; the submission draft's
   ``_ATTACK_COLORS`` only listed A1..A5, so A6 records would
   render in fallback grey indistinguishable from A1.  The
   camera-ready maps A6 to a distinct purple.

3. **New extractors and field surfacing in table_8.**  Camera-
   ready experiments.py emits four new structured note fields:
   ``analysis_alpha`` (whether α came from ground-truth labels or
   benign fallback), ``benign_fallback`` (whether Theorem 4.2 was
   skipped because no adversarial samples were observed),
   ``lambda_at_edge`` (whether the MGF λ search saturated the
   bracket — looseness diagnostic), and ``holds_ci_95`` (Wilson
   95% CI on the apply/validate identity).  ``table_8`` parses
   and surfaces all four so reviewers can read certificate
   validity and tightness from the table directly.

4. **Documented assumption when ``alpha_source = benign-fallback``.**
   When the camera-ready experiments harness reports
   ``analysis_alpha: source=no per_txn_log; benign-only fallback``
   or the analysis layer reports
   ``adversarial_fit_is_benign_fallback=True``, table_8 marks the
   row so reviewers cannot mistake the numbers for a validated
   adversarial bound.

5. **F1 sentinel handling (defensive).**  Camera-ready
   experiments.py persists detection notes that may say
   ``"detection: F1 not computed (placeholder evaluator removed
   ...)"`` when the run had no real F1 evaluator.  No current
   figure plots F1, but ``_extract_streaming_f1`` filters such
   notes and returns NaN so that any future figure that consumes
   the field will see NaN rather than the submission draft's
   fabricated 1.0.

What this file does not do
--------------------------
- Aggregate run records across experiments where they should not be
  pooled (e.g. mixing exp_attack_effectiveness with
  exp_defense_efficacy in a single bar).  Each figure/table function
  is explicit about which experiment it consumes.
- Recompute statistics from raw probe samples when the run record
  already contains them.  The summary statistics in the run record
  are the source of truth for the paper; this file's job is to
  arrange and render them.
- Open files written by anything other than experiments.py.

Author: ANONYMOUS  (double-blind review)
"""

from __future__ import annotations

import csv
import json
import logging
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Section 1.  Style and palette.
#
# Pinned palette and rcParams so figures look identical across
# matplotlib versions.  The palette is colour-blind-safe (Okabe-Ito).
# =============================================================================


_PALETTE = {
    "blue":   "#0072B2",
    "orange": "#E69F00",
    "green":  "#009E73",
    "red":    "#D55E00",
    "purple": "#CC79A7",
    "yellow": "#F0E442",
    "cyan":   "#56B4E9",
    "grey":   "#999999",
    "black":  "#000000",
}

_ATTACK_COLORS: Mapping[str, str] = {
    "A1_random":         _PALETTE["grey"],
    "A2_high_degree":    _PALETTE["cyan"],
    "A3_branching_max":  _PALETTE["blue"],
    "A4_gradient_norm":  _PALETTE["orange"],
    "A5_adaptive":       _PALETTE["red"],
    # Camera-ready: A6 is the evolutionary cost-oracle attack
    # introduced in attacks.py (Sponge Examples, Shumailov et al.
    # EuroS&P 2021 inspired).  Without this entry, A6 records
    # rendered in fallback grey, indistinguishable from A1.
    "A6_evolutionary":   _PALETTE["purple"],
}

_DEFENSE_COLORS: Mapping[str, str] = {
    "no_defense":        _PALETTE["grey"],
    "D1_static":         _PALETTE["yellow"],
    "D2_adaptive":       _PALETTE["green"],
    "D3_schedulability": _PALETTE["blue"],
}

# Camera-ready: shading colors for warmup / storm / recovery
# intervals.  Light enough to not obscure overlaid lines, distinct
# enough that a reader sees three regions at a glance.
_INTERVAL_SHADING = {
    "warmup":   "#EEEEEE",   # very light grey
    "storm":    "#FFFFFF",   # transparent (no shading)
    "recovery": "#E0F0F8",   # very light blue
}


def _setup_matplotlib() -> Any:
    """Pin rcParams, return pyplot module."""
    import matplotlib
    matplotlib.use("Agg")              # file-only backend
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.6,
        "grid.linewidth": 0.4,
        "lines.linewidth": 1.2,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,        # editable in Illustrator etc.
        "ps.fonttype": 42,
    })
    return plt


# =============================================================================
# Section 2.  Run record loading.
#
# Records are written to results/raw/<experiment>/*.json by
# experiments.py.  We load them and group by (experiment, dataset,
# attack, defense, seed) for downstream aggregation.
#
# Camera-ready: ``LoadedRecords.filter`` semantics fixed.  See the
# class docstring for the rationale.
# =============================================================================


# Sentinel used by LoadedRecords.filter to distinguish "do not filter
# on this field" from "filter to records where this field is None."
# The submission draft conflated the two — every call site that
# wanted "filter to no_defense records" by passing
# ``defense=None`` instead silently returned every record.
_ANY: Any = object()


@dataclass
class LoadedRecords:
    """
    Convenience wrapper around a list of loaded RunRecord dicts.

    Filter semantics (camera-ready)
    -------------------------------
    ``filter(...)`` with no argument for a field returns every
    record (no filter on that field).  ``filter(field=None)`` keeps
    only records where that field IS None.  ``filter(field=value)``
    keeps only records where that field equals value.

    The submission draft used ``if attack is not None and ...``
    semantics, in which ``filter(attack=None)`` was identical to
    ``filter()`` — silently producing the wrong subset.  All
    submission-draft call sites passed ``None`` *intending* the
    "field equals None" semantics (e.g. ``d_filter = None if defense
    == "no_defense" else defense``); the camera-ready makes that
    intent the actual behaviour.

    Backwards compatibility: callers that want the submission-draft
    "do not filter" semantics now pass nothing — i.e.
    ``filter(experiment="exp_X")`` filters only on experiment, just
    as it did before.
    """

    records: List[Mapping[str, Any]]

    def filter(
        self,
        experiment: Any = _ANY,
        dataset: Any = _ANY,
        attack: Any = _ANY,
        defense: Any = _ANY,
    ) -> "LoadedRecords":
        out = []
        for r in self.records:
            if experiment is not _ANY and r.get("experiment") != experiment:
                continue
            if dataset is not _ANY and r.get("dataset") != dataset:
                continue
            if attack is not _ANY and r.get("attack") != attack:
                continue
            if defense is not _ANY and r.get("defense") != defense:
                continue
            out.append(r)
        return LoadedRecords(out)

    def __len__(self) -> int:
        return len(self.records)


def load_records(raw_dir: Path) -> LoadedRecords:
    """
    Load every JSON record under raw_dir (recursively) into a
    LoadedRecords.  Bad records are logged and skipped, not fatal.
    """
    records: List[Mapping[str, Any]] = []
    for path in sorted(raw_dir.rglob("*.json")):
        try:
            with open(path) as f:
                records.append(json.load(f))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"skipping {path}: {e}")
    logger.info(f"loaded {len(records)} run records from {raw_dir}")
    return LoadedRecords(records)


# =============================================================================
# Section 3.  Aggregation primitives.
#
# Two operations recur across every figure:
#   - compute mean ± 95% CI over the seed dimension;
#   - extract a specific summary statistic from a run record's
#     phase distributions or miss report.
#
# Camera-ready additions: extractors for the new freshness, time
# series, and recovery-diagnostics fields, plus a logarithmic CCDF
# sub-sampler so figures render in seconds rather than minutes at
# 10⁶+ samples.
# =============================================================================


def _mean_ci(values: Sequence[float], confidence: float = 0.95) -> Tuple[float, float, float]:
    """Returns (mean, ci_low, ci_high) using a t-distribution-free Wilson-like bound."""
    if not values:
        return (float("nan"), float("nan"), float("nan"))
    arr = np.asarray(values, dtype=np.float64)
    n = arr.size
    if n == 1:
        return (float(arr[0]), float(arr[0]), float(arr[0]))
    mean = float(arr.mean())
    sem = float(arr.std(ddof=1)) / math.sqrt(n)
    # t-multiplier for 95% CI at small n.  We use the normal approximation
    # for n ≥ 5 and a hardcoded t-table value for smaller n.
    z = 1.96 if n >= 30 else _t_approx(n - 1, confidence)
    half = z * sem
    return (mean, mean - half, mean + half)


def _t_approx(df: int, conf: float) -> float:
    """Cheap two-sided t-multiplier.  Hardcoded values for common (df, conf=.95)."""
    table = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57,
             6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26, 10: 2.23,
             15: 2.13, 20: 2.09, 25: 2.06}
    if df in table:
        return table[df]
    # Fall back to nearest.
    keys = sorted(table.keys())
    closest = min(keys, key=lambda k: abs(k - df))
    return table[closest]


def _miss_rate(record: Mapping[str, Any]) -> float:
    mr = record.get("miss_report")
    if not mr:
        return float("nan")
    rate = mr.get("miss_rate")
    return float(rate) if rate is not None else float("nan")


def _e2e_p99_us(record: Mapping[str, Any]) -> float:
    e2e = record.get("end_to_end")
    if not e2e:
        return float("nan")
    samples = e2e.get("samples_ns") or []
    if not samples:
        return float("nan")
    return float(np.percentile(samples, 99) / 1000.0)


def _e2e_samples_us(record: Mapping[str, Any]) -> np.ndarray:
    e2e = record.get("end_to_end")
    if not e2e:
        return np.zeros(0)
    samples = e2e.get("samples_ns") or []
    return np.asarray(samples, dtype=np.float64) / 1000.0


# --- Camera-ready: extractors for new measurement fields ----------------


def _age_violation_rate(record: Mapping[str, Any]) -> float:
    """Fraction of decisions whose model age exceeded A_max."""
    ar = record.get("age_violation_report")
    if not ar:
        return float("nan")
    rate = ar.get("violation_rate")
    return float(rate) if rate is not None else float("nan")


def _age_violation_samples(record: Mapping[str, Any]) -> Tuple[Optional[int], Optional[int], Optional[float]]:
    """Returns (n_violations, n_decisions, age_max_us) or (None, None, None)."""
    ar = record.get("age_violation_report")
    if not ar:
        return (None, None, None)
    return (ar.get("n_violations"), ar.get("n_decisions"), ar.get("age_max_us"))


def _propagation_samples_us(record: Mapping[str, Any]) -> np.ndarray:
    """commit − arrival latency per committed sample, in µs."""
    fr = record.get("freshness")
    if not fr:
        return np.zeros(0)
    samples = fr.get("propagation_latency_samples_us") or []
    return np.asarray(samples, dtype=np.float64)


def _age_at_decision_samples_us(record: Mapping[str, Any]) -> np.ndarray:
    """Model age (µs) at each recorded decision."""
    fr = record.get("freshness")
    if not fr:
        return np.zeros(0)
    samples = fr.get("age_at_decision_samples_us") or []
    return np.asarray(samples, dtype=np.float64)


def _average_model_age_us(record: Mapping[str, Any]) -> float:
    fr = record.get("freshness")
    if not fr:
        return float("nan")
    v = fr.get("average_model_age_us")
    return float(v) if v is not None and not math.isnan(float(v)) else float("nan")


def _peak_model_age_us(record: Mapping[str, Any]) -> float:
    fr = record.get("freshness")
    if not fr:
        return float("nan")
    v = fr.get("peak_model_age_us")
    return float(v) if v is not None and not math.isnan(float(v)) else float("nan")


def _recovery_time_s(record: Mapping[str, Any]) -> float:
    """Storm recovery time in seconds, or NaN if WSR was off / not observed."""
    rd = record.get("recovery_diagnostics")
    if not rd:
        return float("nan")
    rt = rd.get("recovery_time_s")
    if rt is None:
        return float("nan")
    return float(rt)


def _recovery_observed(record: Mapping[str, Any]) -> Optional[bool]:
    rd = record.get("recovery_diagnostics")
    if not rd:
        return None
    obs = rd.get("recovery_observed")
    return bool(obs) if obs is not None else None


def _recovery_baseline_us(record: Mapping[str, Any]) -> float:
    rd = record.get("recovery_diagnostics")
    if not rd:
        return float("nan")
    b = rd.get("baseline_latency_us")
    return float(b) if b is not None else float("nan")


def _time_series_field(record: Mapping[str, Any], key: str) -> Any:
    """Return record['time_series'][key] or None."""
    ts = record.get("time_series")
    if not ts:
        return None
    return ts.get(key)


def _latency_over_time(record: Mapping[str, Any]) -> List[Tuple[int, float]]:
    """List of (t_ns, latency_us) per completion, sorted by time."""
    pts = _time_series_field(record, "latency_over_time") or []
    return [(int(p["t_ns"]), float(p["latency_us"])) for p in pts]


def _queue_depth_events(record: Mapping[str, Any]) -> List[Tuple[int, int]]:
    """List of (t_ns, depth) at each event, sorted by time."""
    pts = _time_series_field(record, "queue_depth_events") or []
    return [(int(p["t_ns"]), int(p["depth"])) for p in pts]


def _throughput_buckets(record: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    pts = _time_series_field(record, "throughput_buckets") or []
    return list(pts)


def _run_start_ns(record: Mapping[str, Any]) -> Optional[int]:
    s = _time_series_field(record, "run_start_ns")
    return int(s) if s is not None else None


# --- Camera-ready: extractors for new experiments.py note fields --------


def _alpha_source_and_value(
    record: Mapping[str, Any],
) -> Tuple[Optional[str], float]:
    """
    Parse the ``analysis_alpha`` note (camera-ready experiments.py).

    The note has the form:
      "analysis_alpha: source=ground-truth (per_txn_log), measured=0.0473"
    or
      "analysis_alpha: source=no per_txn_log; benign-only fallback, measured=0.0000"

    Returns ``(source, measured)`` where source is the verbatim source
    description string (or None if the note is absent) and measured
    is the empirical α (or NaN).
    """
    for note in record.get("notes", []):
        if not isinstance(note, str) or not note.startswith("analysis_alpha:"):
            continue
        body = note[len("analysis_alpha:"):].lstrip()
        # The source field may contain commas (e.g. "ground-truth
        # (per_txn_log)"), so we must split only on the LAST
        # ", measured=" rather than any comma.
        m = re.search(r",\s*measured=([0-9.+-eE]+)\s*$", body)
        if m is None:
            return (body, float("nan"))
        source_str = body[: m.start()].strip()
        if source_str.startswith("source="):
            source_str = source_str[len("source="):]
        try:
            measured = float(m.group(1))
        except ValueError:
            measured = float("nan")
        return (source_str, measured)
    return (None, float("nan"))


def _benign_fallback_flag(record: Mapping[str, Any]) -> Optional[bool]:
    """
    Parse the ``benign_fallback`` field of the ``analysis_envelope``
    note.  Camera-ready experiments.py emits, e.g.,
      "analysis_envelope: T_min=Xus, feasible=True, slack=Yus,
       benign_fallback=False"
    True means Theorem 4.2 was reported as a benign fallback because
    no adversarial samples were observed (per camera-ready
    analysis.py); the row should be flagged so reviewers don't
    treat the bound as a validated adversarial bound.

    Returns None when the note is absent or the field is absent.
    """
    kv = _parse_kv_note(record, "analysis_envelope")
    return _kv_bool(kv, "benign_fallback")


def _lambda_at_edge_flag(record: Mapping[str, Any]) -> Optional[bool]:
    """
    Parse the ``lambda_at_edge`` flag from the ``analysis_mgf`` or
    ``d3_mgf_cert`` notes (camera-ready experiments.py).

    True means the MGF certificate's λ search saturated the bracket
    edge — a looseness diagnostic.  Bracket-edge saturation means
    the bound may be loose because the optimiser was constrained
    by the search range, not by the underlying mathematics.
    Reviewers seeing many such rows should ask whether the search
    bracket needs widening.

    Looks at ``d3_mgf_cert`` first (the defence-side certificate),
    then ``analysis_mgf`` (the post-hoc analysis).  Returns None
    when neither is present.
    """
    kv = _parse_kv_note(record, "d3_mgf_cert")
    flag = _kv_bool(kv, "lambda_at_edge")
    if flag is not None:
        return flag
    kv = _parse_kv_note(record, "analysis_mgf")
    return _kv_bool(kv, "lambda_at_edge")


def _holds_ci_95_flag(record: Mapping[str, Any]) -> Optional[bool]:
    """
    Parse the ``holds_ci_95`` flag from the ``validate_4_3_mgf``
    note (camera-ready experiments.py).

    True means the empirical miss rate's Wilson 95% CI upper bound
    is below the certified bound — a stronger statement than
    ``holds`` alone (which compares only the point estimate).
    The §V.3 schedulability claim explicitly uses 95% confidence.
    """
    kv = _parse_kv_note(record, "validate_4_3_mgf")
    return _kv_bool(kv, "holds_ci_95")


def _extract_streaming_f1(record: Mapping[str, Any]) -> float:
    """
    Parse the streaming F1 from the ``detection`` note.  Returns
    NaN when the camera-ready experiments.py reports
    ``detection: F1 not computed`` (the placeholder evaluator was
    removed in the camera-ready harness; see docs/F1_EVAL.md).

    The submission draft's ``streaming_predictions.append(
    (timed.label, timed.label))`` produced F1 = 1.0 by
    construction; this extractor is defensive code for any future
    figure that reads the detection note — it returns NaN for
    "F1 not computed" rather than reading a fabricated value.
    """
    for note in record.get("notes", []):
        if not isinstance(note, str) or not note.startswith("detection:"):
            continue
        body = note[len("detection:"):]
        if "not computed" in body:
            return float("nan")
        m = re.search(r"streaming_F1\s*=\s*([0-9.eE+-]+)", body)
        if m is None:
            return float("nan")
        try:
            return float(m.group(1))
        except ValueError:
            return float("nan")
    return float("nan")


# --- Camera-ready: logarithmic CCDF sub-sampling ------------------------


def _subsample_for_ccdf(
    arr_sorted: np.ndarray,
    max_points: int = 10_000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sub-sample a sorted array logarithmically for CCDF plotting.
    Keeps the tail well-resolved (denser sampling at large indices)
    while making 10⁶+-sample arrays plot in milliseconds.

    Returns (x, ccdf) where ccdf[i] = P(X > x[i]) computed against
    the FULL distribution, not the sub-sample, so the rendered curve
    is unbiased.
    """
    n = arr_sorted.size
    if n == 0:
        return arr_sorted, np.zeros(0)
    if n <= max_points:
        ranks = np.arange(1, n + 1)
        ccdf = (n - ranks + 1) / n
        return arr_sorted, ccdf
    # Logarithmically-spaced indices, spanning [0, n-1].  geomspace
    # would put too few points near the head; we mix linear (head)
    # with geometric (tail).
    head = np.linspace(0, max_points // 4 - 1, max_points // 4).astype(int)
    tail = np.geomspace(
        max_points // 4, n, max_points - max_points // 4
    ).astype(int) - 1
    idx = np.unique(np.concatenate([head, tail]))
    idx = idx[idx < n]
    sub = arr_sorted[idx]
    # CCDF computed against full n.
    ccdf = (n - (idx + 1) + 1) / n
    return sub, ccdf


# =============================================================================
# Section 4.  Note parsing.
#
# experiments.py persists structured information into RunRecord.notes:
#   - JSON-format notes (interval boundaries, per-phase stats):
#       "intervals: {<json>}"
#       "phase[warmup]: {<json>}"
#   - Key-value notes (certificates, validations):
#       "analysis_envelope: T_min=Xus, feasible=True, slack=Yus"
#       "analysis_mgf: feasible=False, bound=1.234, ..."
#       "validate_4_3_mgf: holds=True, bound=0.1, measured=0.05, ..."
#       "d3_envelope_cert: feasible=True, T=Xus, ..."
#       "d3_mgf_cert: feasible=True, bound=0.1, ..."
# These helpers parse them robustly, including notes that contain
# nested commas inside single-quoted strings.
# =============================================================================


def _parse_json_note(record: Mapping[str, Any], prefix: str) -> Optional[Mapping[str, Any]]:
    """Find the first note starting with `prefix:` and parse its JSON body."""
    full = prefix if prefix.endswith(":") else f"{prefix}:"
    for note in record.get("notes", []):
        if isinstance(note, str) and note.startswith(full):
            body = note[len(full):].lstrip()
            try:
                return json.loads(body)
            except (json.JSONDecodeError, TypeError):
                return None
    return None


def _parse_kv_note(record: Mapping[str, Any], prefix: str) -> Mapping[str, str]:
    """
    Parse a key-value note of the form
        "<prefix>: key1=val1, key2=val2, ..."
    Returns {} if no matching note found.  Robust to commas inside
    single-quoted string values (notes='..., ...').
    """
    full = prefix if prefix.endswith(":") else f"{prefix}:"
    for note in record.get("notes", []):
        if not isinstance(note, str) or not note.startswith(full):
            continue
        body = note[len(full):].lstrip()
        return _split_kv_string(body)
    return {}


def _split_kv_string(body: str) -> Mapping[str, str]:
    """
    Split a 'key1=val1, key2=val2, ...' string respecting single-quoted
    values.  Used by both _parse_kv_note and the cert table renderer.
    """
    parts: List[str] = []
    current: List[str] = []
    in_quote = False
    for ch in body:
        if ch == "'" and (not current or current[-1] != "\\"):
            in_quote = not in_quote
            current.append(ch)
        elif ch == "," and not in_quote:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        parts.append("".join(current).strip())
    out: Dict[str, str] = {}
    for part in parts:
        if "=" in part:
            k, _, v = part.partition("=")
            v = v.strip()
            # Strip wrapping single quotes from string values.
            if len(v) >= 2 and v[0] == "'" and v[-1] == "'":
                v = v[1:-1]
            # Strip trailing 'us' or 'µs' unit hints from numerics
            # for downstream parsing convenience.  We keep the key
            # itself (T_min=Xus) but the consumer can call
            # _kv_float() to drop the unit and convert.
            out[k.strip()] = v
    return out


def _kv_float(kv: Mapping[str, str], key: str, default: float = float("nan")) -> float:
    """Read a numeric value from a kv-parse, stripping common unit suffixes."""
    raw = kv.get(key)
    if raw is None:
        return default
    s = raw.strip()
    # Strip common unit suffixes.
    for suffix in ("us", "µs", "ms", "s", "ns"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
            break
    try:
        return float(s)
    except ValueError:
        return default


def _kv_bool(kv: Mapping[str, str], key: str) -> Optional[bool]:
    raw = kv.get(key)
    if raw is None:
        return None
    s = raw.strip().lower()
    if s in ("true", "1", "yes"):
        return True
    if s in ("false", "0", "no"):
        return False
    return None


def _phase_stats(record: Mapping[str, Any], phase_label: str) -> Optional[Mapping[str, Any]]:
    """Return the parsed phase[<phase_label>] note, or None."""
    return _parse_json_note(record, f"phase[{phase_label}]")


def _interval_boundaries(record: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    """Return the parsed intervals note, or None."""
    return _parse_json_note(record, "intervals")


# =============================================================================
# Section 5.  Tables.
# =============================================================================


def _write_table(
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
    out_dir: Path,
    name: str,
    caption: str = "",
) -> None:
    """Write rows to CSV and to LaTeX, both in `out_dir`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{name}.csv"
    tex_path = out_dir / f"{name}.tex"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in columns})
    with open(tex_path, "w") as f:
        f.write("% Auto-generated by plots.py.  Do not edit by hand.\n")
        f.write("\\begin{table}[t]\n\\centering\n")
        f.write(f"\\caption{{{caption}}}\n")
        f.write(f"\\label{{tab:{name}}}\n")
        f.write("\\begin{tabular}{" + "l" * len(columns) + "}\n")
        f.write("\\toprule\n")
        f.write(" & ".join(c.replace("_", "\\_") for c in columns) + " \\\\\n")
        f.write("\\midrule\n")
        for r in rows:
            f.write(
                " & ".join(_latex_format(r.get(c, "")) for c in columns) + " \\\\\n"
            )
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    logger.info(f"wrote table {name} to {csv_path} and {tex_path}")


def _latex_format(v: Any) -> str:
    if isinstance(v, float):
        if math.isnan(v):
            return "---"
        if abs(v) < 1e-3 or abs(v) >= 1e6:
            return f"\\num{{{v:.2e}}}"
        return f"{v:.3g}"
    if isinstance(v, int):
        return f"{v:,d}"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if v is None:
        return "---"
    return str(v).replace("_", "\\_")


def table_3(loaded: LoadedRecords, out_dir: Path) -> None:
    """Table 3 — Attack effectiveness per dataset (paper §V.1)."""
    target = loaded.filter(experiment="exp_attack_effectiveness")
    rows: List[Mapping[str, Any]] = []
    datasets = sorted({r["dataset"] for r in target.records})
    # Camera-ready: A6_evolutionary added to the enumeration.
    # The submission draft listed only A1..A5; with the camera-ready
    # registry-driven attack routing in attacks.py and threat_model.py,
    # ``attack_names()`` returns six entries and the harness produces
    # A6 records too.
    attacks = ["benign", "A1_random", "A2_high_degree",
               "A3_branching_max", "A4_gradient_norm", "A5_adaptive",
               "A6_evolutionary"]
    for ds in datasets:
        for atk in attacks:
            atk_filter = None if atk == "benign" else atk
            subset = target.filter(dataset=ds, attack=atk_filter)
            miss = [_miss_rate(r) for r in subset.records]
            p99 = [_e2e_p99_us(r) for r in subset.records]
            miss_clean = [m for m in miss if not math.isnan(m)]
            p99_clean = [m for m in p99 if not math.isnan(m)]
            mean_miss, lo_miss, hi_miss = _mean_ci(miss_clean)
            mean_p99, lo_p99, hi_p99 = _mean_ci(p99_clean)
            rows.append({
                "dataset": ds,
                "attack": atk,
                "n_runs": len(subset.records),
                "miss_rate_mean": mean_miss,
                "miss_rate_ci_lo": lo_miss,
                "miss_rate_ci_hi": hi_miss,
                "p99_latency_us_mean": mean_p99,
                "p99_latency_us_ci_lo": lo_p99,
                "p99_latency_us_ci_hi": hi_p99,
            })
    _write_table(
        rows,
        columns=["dataset", "attack", "n_runs", "miss_rate_mean",
                 "miss_rate_ci_lo", "miss_rate_ci_hi",
                 "p99_latency_us_mean", "p99_latency_us_ci_lo",
                 "p99_latency_us_ci_hi"],
        out_dir=out_dir,
        name="table_3_attack_effectiveness",
        caption="Attack effectiveness: deadline-miss rate and P99 "
                "end-to-end latency per (dataset, attack), no defense.",
    )


def table_4(loaded: LoadedRecords, out_dir: Path) -> None:
    """Table 4 — Defense deadline-miss rates (paper §V.2)."""
    target = loaded.filter(experiment="exp_defense_efficacy")
    rows: List[Mapping[str, Any]] = []
    datasets = sorted({r["dataset"] for r in target.records})
    attacks = sorted({r["attack"] for r in target.records if r.get("attack")})
    defenses = ["no_defense", "D1_static", "D2_adaptive", "D3_schedulability"]
    for ds in datasets:
        for atk in attacks:
            for defense in defenses:
                d_filter = None if defense == "no_defense" else defense
                subset = target.filter(dataset=ds, attack=atk, defense=d_filter)
                miss = [_miss_rate(r) for r in subset.records]
                miss_clean = [m for m in miss if not math.isnan(m)]
                mean, lo, hi = _mean_ci(miss_clean)
                rows.append({
                    "dataset": ds,
                    "attack": atk,
                    "defense": defense,
                    "n_runs": len(subset.records),
                    "miss_rate_mean": mean,
                    "miss_rate_ci_lo": lo,
                    "miss_rate_ci_hi": hi,
                })
    _write_table(
        rows,
        columns=["dataset", "attack", "defense", "n_runs",
                 "miss_rate_mean", "miss_rate_ci_lo", "miss_rate_ci_hi"],
        out_dir=out_dir,
        name="table_4_defense_efficacy",
        caption="Deadline-miss rate per (dataset, attack, defense).",
    )


def table_5(loaded: LoadedRecords, out_dir: Path) -> None:
    """Table 5 — Defense ablation (paper §V.4)."""
    target = loaded.filter(experiment="exp_ablation")
    rows: List[Mapping[str, Any]] = []
    defenses = ["no_defense", "D1_static", "D2_adaptive", "D3_schedulability"]
    datasets = sorted({r["dataset"] for r in target.records})
    for ds in datasets:
        for defense in defenses:
            d_filter = None if defense == "no_defense" else defense
            subset = target.filter(dataset=ds, defense=d_filter)
            miss = [_miss_rate(r) for r in subset.records]
            miss_clean = [m for m in miss if not math.isnan(m)]
            mean, lo, hi = _mean_ci(miss_clean)
            rows.append({
                "dataset": ds,
                "defense": defense,
                "n_runs": len(subset.records),
                "miss_rate_mean": mean,
                "miss_rate_ci_lo": lo,
                "miss_rate_ci_hi": hi,
            })
    _write_table(
        rows,
        columns=["dataset", "defense", "n_runs", "miss_rate_mean",
                 "miss_rate_ci_lo", "miss_rate_ci_hi"],
        out_dir=out_dir,
        name="table_5_ablation",
        caption="Defense ablation on the primary deadline-credible "
                "dataset under attack A3 (branching maximisation).",
    )


def table_6(loaded: LoadedRecords, out_dir: Path) -> None:
    """Table 6 — Adaptive-adversary case study (paper §VI)."""
    target = loaded.filter(experiment="exp_adaptive_adversary")
    rows: List[Mapping[str, Any]] = []
    datasets = sorted({r["dataset"] for r in target.records})
    defenses = ["no_defense", "D1_static", "D2_adaptive", "D3_schedulability"]
    for ds in datasets:
        for defense in defenses:
            d_filter = None if defense == "no_defense" else defense
            subset = target.filter(dataset=ds, defense=d_filter)
            miss = [_miss_rate(r) for r in subset.records]
            miss_clean = [m for m in miss if not math.isnan(m)]
            mean, lo, hi = _mean_ci(miss_clean)
            rows.append({
                "dataset": ds,
                "defense": defense,
                "n_runs": len(subset.records),
                "miss_rate_mean": mean,
                "miss_rate_ci_lo": lo,
                "miss_rate_ci_hi": hi,
            })
    _write_table(
        rows,
        columns=["dataset", "defense", "n_runs", "miss_rate_mean",
                 "miss_rate_ci_lo", "miss_rate_ci_hi"],
        out_dir=out_dir,
        name="table_6_adaptive_adversary",
        caption="A5 (adaptive white-box adversary) vs each defense.",
    )


def table_2(loaded: LoadedRecords, out_dir: Path) -> None:
    """Table 2 — Bound validation summary (paper §V.3)."""
    target = loaded.filter(experiment="exp_schedulability_validation")
    # Aggregate notes for theorem validations.  Notes are written as
    # free-form strings; we extract structured fields where present.
    rows: List[Mapping[str, Any]] = []
    datasets = sorted({r["dataset"] for r in target.records})
    for ds in datasets:
        subset = target.filter(dataset=ds)
        miss_rates = [_miss_rate(r) for r in subset.records]
        miss_clean = [m for m in miss_rates if not math.isnan(m)]
        mean, lo, hi = _mean_ci(miss_clean)
        # Pull contract epsilon from the first record's miss_report.
        eps = float("nan")
        for r in subset.records:
            mr = r.get("miss_report") or {}
            if "epsilon" in mr:
                eps = float(mr["epsilon"])
                break
        rows.append({
            "dataset": ds,
            "n_runs": len(subset.records),
            "epsilon": eps,
            "measured_miss_rate_mean": mean,
            "measured_miss_rate_ci_lo": lo,
            "measured_miss_rate_ci_hi": hi,
            "bound_holds": (mean <= eps) if not math.isnan(eps) else False,
        })
    _write_table(
        rows,
        columns=["dataset", "n_runs", "epsilon", "measured_miss_rate_mean",
                 "measured_miss_rate_ci_lo", "measured_miss_rate_ci_hi",
                 "bound_holds"],
        out_dir=out_dir,
        name="table_2_bound_validation",
        caption="Empirical validation of the schedulability bound: "
                "measured miss rate vs. promised epsilon.",
    )


def table_1(loaded: LoadedRecords, out_dir: Path) -> None:
    """
    Table 1 — System and defense comparison summary (paper §I).

    Drawn from exp_defense_efficacy, this is the headline table that
    appears early in the paper.  Reports aggregate miss rate per
    defense across all datasets and attacks.
    """
    target = loaded.filter(experiment="exp_defense_efficacy")
    rows: List[Mapping[str, Any]] = []
    defenses = ["no_defense", "D1_static", "D2_adaptive", "D3_schedulability"]
    for defense in defenses:
        d_filter = None if defense == "no_defense" else defense
        subset = target.filter(defense=d_filter)
        miss = [_miss_rate(r) for r in subset.records]
        miss_clean = [m for m in miss if not math.isnan(m)]
        mean, lo, hi = _mean_ci(miss_clean)
        rows.append({
            "defense": defense,
            "n_runs": len(subset.records),
            "miss_rate_mean": mean,
            "miss_rate_ci_lo": lo,
            "miss_rate_ci_hi": hi,
        })
    _write_table(
        rows,
        columns=["defense", "n_runs", "miss_rate_mean",
                 "miss_rate_ci_lo", "miss_rate_ci_hi"],
        out_dir=out_dir,
        name="table_1_summary",
        caption="Headline comparison: average deadline-miss rate per "
                "defense across all datasets and attacks.",
    )


# --- Camera-ready: new tables -------------------------------------------


def table_7(loaded: LoadedRecords, out_dir: Path) -> None:
    """
    Table 7 — Age-violation rates (Yates AoI demand, §V.2).

    For each (dataset, attack, defense), report the empirical
    fraction of decisions whose model age exceeded A_max.  The
    A_max threshold defaults to the contract deadline_us; the
    measurement layer tracks both the rate and its Wilson-score CI.

    This is the "freshness-violation" companion to the deadline-miss
    table (table_4): the latter answers "did the update finish in
    time?", the former answers "when a decision was made, was the
    model state fresh enough?"  Both are required to characterise
    update-storm impact under the AoI semantics.
    """
    target = loaded.filter(experiment="exp_defense_efficacy")
    if not target.records:
        # Fall back to attack-effectiveness records, which also have
        # AoI fields in the camera-ready pipeline.
        target = loaded.filter(experiment="exp_attack_effectiveness")
    rows: List[Mapping[str, Any]] = []
    datasets = sorted({r["dataset"] for r in target.records})
    attacks = sorted({r["attack"] for r in target.records if r.get("attack")})
    defenses = ["no_defense", "D1_static", "D2_adaptive", "D3_schedulability"]
    for ds in datasets:
        for atk in attacks:
            for defense in defenses:
                d_filter = None if defense == "no_defense" else defense
                subset = target.filter(dataset=ds, attack=atk, defense=d_filter)
                if not subset.records:
                    continue
                rates = [_age_violation_rate(r) for r in subset.records]
                rates_clean = [v for v in rates if not math.isnan(v)]
                if not rates_clean:
                    continue
                mean, lo, hi = _mean_ci(rates_clean)
                # Pick A_max from the first record that reports it.
                age_max_us = float("nan")
                for r in subset.records:
                    _, _, am = _age_violation_samples(r)
                    if am is not None:
                        age_max_us = float(am)
                        break
                # Average model age (time-average AoI).
                aoi_means = [
                    _average_model_age_us(r) for r in subset.records
                ]
                aoi_clean = [v for v in aoi_means if not math.isnan(v)]
                aoi_mean, _, _ = _mean_ci(aoi_clean)
                rows.append({
                    "dataset": ds,
                    "attack": atk,
                    "defense": defense,
                    "n_runs": len(subset.records),
                    "age_max_us": age_max_us,
                    "violation_rate_mean": mean,
                    "violation_rate_ci_lo": lo,
                    "violation_rate_ci_hi": hi,
                    "avg_model_age_us": aoi_mean,
                })
    _write_table(
        rows,
        columns=["dataset", "attack", "defense", "n_runs", "age_max_us",
                 "violation_rate_mean", "violation_rate_ci_lo",
                 "violation_rate_ci_hi", "avg_model_age_us"],
        out_dir=out_dir,
        name="table_7_age_violations",
        caption="Age-of-Information violation rate and time-average "
                "model age per (dataset, attack, defense).  An "
                "age violation occurs when a decision uses model "
                "state older than A_max (defaulting to the contract "
                "deadline).  Per Yates et al. JSAC 2021, the average "
                "model age is computed via the AoI sawtooth-area "
                "decomposition Q_n = ½ Y_n² + Y_n · T_n.",
    )


def table_8(loaded: LoadedRecords, out_dir: Path) -> None:
    """
    Table 8 — Schedulability certificate validation (paper §V.3 headline).

    For each (dataset) under D3_schedulability defense, parse the
    structured certificate notes written by experiments.py and
    display BOTH the envelope certificate (Theorem 4.3-env) and the
    MGF certificate (Theorem 4.3-mgf) side-by-side, plus the
    apply-vs-validate measurement.

    This is the headline §V.3 table because it is what
    distinguishes a paper-quality schedulability claim from a
    plotting-the-results claim: the certificates are *predictions*
    made before the evaluation finishes, and the measured miss rate
    is the *test*.  When the validate column says holds=true and
    the measured rate is below the certified bound, the proof is
    empirically tight; when holds=false, the bound was incorrectly
    derived (e.g., assumption violated).

    Camera-ready additions
    ----------------------
    - ``alpha_source``: from the ``analysis_alpha`` note.  Either
      "ground-truth (per_txn_log)" (when camera-ready experiments.py
      had attack-injected transactions to count) or "no per_txn_log;
      benign-only fallback" (no per_txn_log entries observed).  This
      reveals to reviewers whether α came from ground truth or
      from a fallback, which determines how much the row supports
      the paper's α-conditional schedulability claims.
    - ``alpha_measured``: numeric α from the same note.
    - ``benign_fallback_pct``: from the ``analysis_envelope`` note's
      ``benign_fallback=`` field.  Fraction of seeds where Theorem
      4.2 collapsed to the benign fit because no adversarial samples
      were observed (per camera-ready analysis.py).  Rows with high
      benign_fallback_pct should not be read as adversarial-bound
      validations; the table marks them so.
    - ``lambda_at_edge_pct``: from the ``d3_mgf_cert`` note's
      ``lambda_at_edge=`` field.  Fraction of seeds where the MGF
      λ search saturated the bracket edge.  High values indicate
      the bound may be loose because the optimiser was constrained
      by the search range, not by the underlying random-sum
      Chernoff structure.  See camera-ready analysis.py for the
      bracket-edge diagnostic.
    - ``validate_holds_ci_95_pct``: from the ``validate_4_3_mgf``
      note's ``holds_ci_95=`` field.  Fraction of seeds for which
      the Wilson 95% CI upper bound on the measured miss rate fell
      below the MGF-certified bound.  Stricter than
      ``validate_holds_pct`` (which compares only the point
      estimate); this is the column the paper's "at 95% confidence"
      language relies on.
    """
    target = loaded.filter(
        experiment="exp_schedulability_validation",
        defense="D3_schedulability",
    )
    rows: List[Mapping[str, Any]] = []
    datasets = sorted({r["dataset"] for r in target.records})
    for ds in datasets:
        subset = target.filter(dataset=ds)
        # Aggregate certificate fields across seeds.
        env_feasible: List[bool] = []
        env_T_us: List[float] = []
        env_bound: List[float] = []
        env_slack_us: List[float] = []
        mgf_feasible: List[bool] = []
        mgf_log_bound: List[float] = []
        mgf_lambda_star: List[float] = []
        mgf_lambda_at_edge: List[bool] = []     # camera-ready
        validate_holds: List[bool] = []
        validate_holds_ci_95: List[bool] = []   # camera-ready
        validate_bound: List[float] = []
        validate_measured: List[float] = []
        miss_rates: List[float] = []
        # Camera-ready: track α source and benign-fallback flag.
        alpha_sources: List[str] = []
        alpha_measured: List[float] = []
        benign_fallback_flags: List[bool] = []

        for r in subset.records:
            kv = _parse_kv_note(r, "d3_envelope_cert")
            if kv:
                fb = _kv_bool(kv, "feasible")
                if fb is not None:
                    env_feasible.append(fb)
                env_T_us.append(_kv_float(kv, "T"))
                env_bound.append(_kv_float(kv, "bound_at_T"))
                env_slack_us.append(_kv_float(kv, "slack"))
            kv = _parse_kv_note(r, "d3_mgf_cert")
            if kv:
                fb = _kv_bool(kv, "feasible")
                if fb is not None:
                    mgf_feasible.append(fb)
                mgf_log_bound.append(_kv_float(kv, "log_bound"))
                mgf_lambda_star.append(_kv_float(kv, "lambda*"))
                # Camera-ready: bracket-edge diagnostic.
                edge_flag = _kv_bool(kv, "lambda_at_edge")
                if edge_flag is not None:
                    mgf_lambda_at_edge.append(edge_flag)
            kv = _parse_kv_note(r, "validate_4_3_mgf")
            if kv:
                fb = _kv_bool(kv, "holds")
                if fb is not None:
                    validate_holds.append(fb)
                # Camera-ready: Wilson 95% CI flag.
                ci_flag = _kv_bool(kv, "holds_ci_95")
                if ci_flag is not None:
                    validate_holds_ci_95.append(ci_flag)
                validate_bound.append(_kv_float(kv, "bound"))
                validate_measured.append(_kv_float(kv, "measured"))
            miss_rates.append(_miss_rate(r))

            # Camera-ready: α source and benign-fallback diagnostics.
            src, val = _alpha_source_and_value(r)
            if src is not None:
                alpha_sources.append(src)
            if not math.isnan(val):
                alpha_measured.append(val)
            bf = _benign_fallback_flag(r)
            if bf is not None:
                benign_fallback_flags.append(bf)

        miss_clean = [m for m in miss_rates if not math.isnan(m)]
        miss_mean, _, _ = _mean_ci(miss_clean)

        def _frac_true(vals: Sequence[bool]) -> float:
            if not vals:
                return float("nan")
            return sum(1 for v in vals if v) / len(vals)

        # Camera-ready: α-source aggregation.  When all seeds agree,
        # report the common value; when they disagree, report
        # "mixed".  ground-truth and benign-fallback don't co-occur
        # within a single (dataset, defense) cell at the same α —
        # but seed-level data may genuinely vary if some seeds had
        # zero adversarial samples while others did.
        if alpha_sources:
            uniq = set(alpha_sources)
            alpha_source_summary = (
                next(iter(uniq)) if len(uniq) == 1 else "mixed"
            )
        else:
            alpha_source_summary = "(unknown)"
        alpha_measured_mean = (
            float(np.mean(alpha_measured)) if alpha_measured else float("nan")
        )

        rows.append({
            "dataset": ds,
            "n_runs": len(subset.records),
            # Camera-ready: α-provenance columns.
            "alpha_source": alpha_source_summary,
            "alpha_measured_mean": alpha_measured_mean,
            "benign_fallback_pct": _frac_true(benign_fallback_flags),
            # Envelope certificate (Theorem 4.3-env).
            "env_feasible_pct": _frac_true(env_feasible),
            "env_T_us_mean": float(np.mean(env_T_us)) if env_T_us else float("nan"),
            "env_bound_mean": float(np.mean(env_bound)) if env_bound else float("nan"),
            "env_slack_us_mean": float(np.mean(env_slack_us)) if env_slack_us else float("nan"),
            # MGF certificate (Theorem 4.3-mgf).
            "mgf_feasible_pct": _frac_true(mgf_feasible),
            "mgf_log_bound_mean": (
                float(np.mean(mgf_log_bound)) if mgf_log_bound else float("nan")
            ),
            "mgf_lambda_star_mean": (
                float(np.mean(mgf_lambda_star)) if mgf_lambda_star else float("nan")
            ),
            # Camera-ready: bracket-edge looseness diagnostic.
            "mgf_lambda_at_edge_pct": _frac_true(mgf_lambda_at_edge),
            # Apply/validate identity.
            "validate_holds_pct": _frac_true(validate_holds),
            # Camera-ready: Wilson 95% CI form of validate_holds.
            "validate_holds_ci_95_pct": _frac_true(validate_holds_ci_95),
            "validate_bound_mean": (
                float(np.mean(validate_bound)) if validate_bound else float("nan")
            ),
            "validate_measured_mean": (
                float(np.mean(validate_measured)) if validate_measured else float("nan")
            ),
            "miss_rate_mean": miss_mean,
        })
    _write_table(
        rows,
        columns=[
            "dataset", "n_runs",
            # α provenance
            "alpha_source", "alpha_measured_mean", "benign_fallback_pct",
            # Envelope certificate
            "env_feasible_pct", "env_T_us_mean",
            "env_bound_mean", "env_slack_us_mean",
            # MGF certificate (with bracket-edge diagnostic)
            "mgf_feasible_pct", "mgf_log_bound_mean",
            "mgf_lambda_star_mean", "mgf_lambda_at_edge_pct",
            # Apply/validate identity (point estimate AND Wilson CI)
            "validate_holds_pct", "validate_holds_ci_95_pct",
            "validate_bound_mean", "validate_measured_mean",
            "miss_rate_mean",
        ],
        out_dir=out_dir,
        name="table_8_schedulability_certificates",
        caption="Schedulability certificate validation under D3 "
                "(\\S V.3).  ``alpha_source`` reports whether $\\alpha$ "
                "was estimated from ground-truth labels via "
                "``Attack.is_adversarial_txn_id`` (the camera-ready "
                "non-circular path) or from the benign-only fallback.  "
                "``benign_fallback_pct`` is the fraction of seeds for "
                "which Theorem 4.2 collapsed to the benign fit because "
                "no adversarial samples were observed; rows with high "
                "values cannot be read as adversarial-bound "
                "validations.  The envelope certificate (Theorem 4.3-env) "
                "uses an exponential tail upper bound; the MGF "
                "certificate (Theorem 4.3-mgf) uses the Sun-style "
                "random-sum Chernoff bound with carry-in correction.  "
                "``mgf\\_lambda\\_at\\_edge\\_pct`` is the fraction of "
                "seeds where the $\\lambda$ search saturated the "
                "bracket edge — a looseness diagnostic.  The 'validate' "
                "columns report the apply-vs-validate identity: "
                "``validate\\_holds\\_pct`` is the fraction of seeds for "
                "which the empirical miss rate point estimate fell below "
                "the MGF-certified bound; ``validate\\_holds\\_ci\\_95\\_pct`` "
                "is the same comparison using the Wilson 95\\% CI upper "
                "bound on the miss rate (the form the paper's "
                "``at 95\\% confidence'' language relies on).",
    )


def table_9(loaded: LoadedRecords, out_dir: Path) -> None:
    """
    Table 9 — Recovery time per defense (Qing & Zheng demand, §V.4).

    For each (dataset, defense) cell, report the mean recovery time
    in seconds — the elapsed time from storm onset until end-to-end
    latency stays within (1 + tolerance) × baseline for at least
    hold_duration_s consecutive seconds.  Computed only on records
    with WSR enabled.

    This is the time-domain companion to table_5 (which reports
    miss rates).  table_5 says "during the storm, how often did
    the system miss its deadline?"; table_9 says "after the storm
    ended, how long until the system was healthy again?"  Both
    matter for production deployment.
    """
    # Collect from all WSR-enabled experiments.
    candidates = []
    for exp_name in (
        "exp_attack_effectiveness",
        "exp_defense_efficacy",
        "exp_ablation",
        "exp_adaptive_adversary",
    ):
        candidates.extend(loaded.filter(experiment=exp_name).records)
    target = LoadedRecords(candidates)
    rows: List[Mapping[str, Any]] = []
    datasets = sorted({r["dataset"] for r in target.records})
    defenses = ["no_defense", "D1_static", "D2_adaptive", "D3_schedulability"]
    for ds in datasets:
        for defense in defenses:
            d_filter = None if defense == "no_defense" else defense
            subset = target.filter(dataset=ds, defense=d_filter)
            recoveries = [_recovery_time_s(r) for r in subset.records]
            recoveries_clean = [v for v in recoveries if not math.isnan(v)]
            observed = [_recovery_observed(r) for r in subset.records]
            n_observed = sum(1 for o in observed if o is True)
            n_with_wsr = sum(1 for o in observed if o is not None)
            if not recoveries_clean:
                continue
            mean, lo, hi = _mean_ci(recoveries_clean)
            baselines = [_recovery_baseline_us(r) for r in subset.records]
            baselines_clean = [b for b in baselines if not math.isnan(b)]
            base_mean, _, _ = _mean_ci(baselines_clean)
            rows.append({
                "dataset": ds,
                "defense": defense,
                "n_runs": len(subset.records),
                "n_with_wsr": n_with_wsr,
                "n_recovered": n_observed,
                "baseline_us_mean": base_mean,
                "recovery_s_mean": mean,
                "recovery_s_ci_lo": lo,
                "recovery_s_ci_hi": hi,
            })
    _write_table(
        rows,
        columns=["dataset", "defense", "n_runs", "n_with_wsr",
                 "n_recovered", "baseline_us_mean",
                 "recovery_s_mean", "recovery_s_ci_lo", "recovery_s_ci_hi"],
        out_dir=out_dir,
        name="table_9_recovery_time",
        caption="Storm recovery time per (dataset, defense).  "
                "Recovery is declared when end-to-end latency stays "
                "within (1+tolerance) × baseline for hold\\_duration "
                "consecutive seconds, per the Qing & Zheng / DRRS "
                "stability criterion.  Baseline is the warmup-phase "
                "mean latency.",
    )


# =============================================================================
# Section 6.  Figures.
# =============================================================================


def figure_3(loaded: LoadedRecords, out_dir: Path) -> None:
    """
    Figure 3 — Empirical CCDF of end-to-end latency vs. fitted upper
    bound from Theorem 4.1.  One panel per dataset.  Demonstrates
    bound conformance.

    Camera-ready: uses _subsample_for_ccdf to handle 10⁶+-sample
    distributions without a multi-minute render time.
    """
    plt = _setup_matplotlib()
    target = loaded.filter(experiment="exp_schedulability_validation",
                           attack=None, defense=None)
    datasets = sorted({r["dataset"] for r in target.records})
    if not datasets:
        logger.warning("figure_3: no records found")
        return
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 2.6), squeeze=False)
    for i, ds in enumerate(datasets):
        ax = axes[0, i]
        subset = target.filter(dataset=ds)
        all_samples_us: List[float] = []
        for r in subset.records:
            all_samples_us.extend(_e2e_samples_us(r))
        if not all_samples_us:
            ax.set_title(f"{ds}\n(no data)")
            continue
        arr = np.sort(np.asarray(all_samples_us))
        # Empirical CCDF (sub-sampled for plotting).
        x_sub, ccdf_sub = _subsample_for_ccdf(arr)
        ax.semilogy(x_sub, ccdf_sub, color=_PALETTE["blue"], lw=1.0,
                    label="Empirical CCDF")
        # Fitted upper bound.
        from analysis import fit_exponential_upper_bound
        fit = fit_exponential_upper_bound(arr)
        c_grid = np.linspace(fit.c_anchor, arr.max() * 1.05, 200)
        bound = np.array([fit.upper_bound_pr_exceeds(c) for c in c_grid])
        ax.semilogy(c_grid, bound, color=_PALETTE["red"], lw=1.0, ls="--",
                    label="Theorem 4.1 bound")
        ax.set_xlabel("End-to-end latency (µs)")
        if i == 0:
            ax.set_ylabel("P(latency > c)")
        ax.set_title(ds)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="lower left")
    fig.suptitle("Theorem 4.1 — empirical bound conformance per dataset", y=1.02)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "figure_3_bound_validation.pdf")
    fig.savefig(out_dir / "figure_3_bound_validation.png")
    plt.close(fig)
    logger.info(f"wrote figure_3 to {out_dir}")


def figure_4(loaded: LoadedRecords, out_dir: Path) -> None:
    """
    Figure 4 — Per-attack end-to-end latency CCDF, one dataset focus
    (the deadline-credible one, CICIDS-2018).

    Camera-ready: sub-sampled CCDF for plotting efficiency.
    """
    plt = _setup_matplotlib()
    target = loaded.filter(
        experiment="exp_attack_effectiveness", dataset="cicids2018",
    )
    if not target.records:
        # Fall back to whichever dataset has data.
        all_target = loaded.filter(experiment="exp_attack_effectiveness")
        if not all_target.records:
            logger.warning("figure_4: no records found")
            return
        target = all_target

    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    attacks_present = sorted(
        {r["attack"] for r in target.records if r.get("attack")}
    )
    # Include benign control as well.
    all_attacks = ["benign"] + attacks_present
    for atk in all_attacks:
        f = target.filter(attack=None if atk == "benign" else atk)
        all_samples: List[float] = []
        for r in f.records:
            all_samples.extend(_e2e_samples_us(r))
        if not all_samples:
            continue
        arr = np.sort(np.asarray(all_samples))
        x_sub, ccdf_sub = _subsample_for_ccdf(arr)
        color = _PALETTE["black"] if atk == "benign" else _ATTACK_COLORS.get(
            atk, _PALETTE["grey"])
        ax.semilogy(x_sub, ccdf_sub, color=color, lw=1.0, label=atk)
    # Mark the contract deadline.
    for r in target.records:
        mr = r.get("miss_report") or {}
        if "deadline_us" in mr:
            ax.axvline(
                float(mr["deadline_us"]),
                color=_PALETTE["grey"], lw=0.6, ls=":",
                label="Deadline D",
            )
            break
    ax.set_xlabel("End-to-end latency (µs)")
    ax.set_ylabel("P(latency > c)")
    ax.set_title("Per-attack latency CCDF — CICIDS-2018")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower left", ncol=2)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "figure_4_attack_ccdf.pdf")
    fig.savefig(out_dir / "figure_4_attack_ccdf.png")
    plt.close(fig)
    logger.info(f"wrote figure_4 to {out_dir}")


def figure_5(loaded: LoadedRecords, out_dir: Path) -> None:
    """
    Figure 5 — Defense deadline-miss rate bar chart.

    For each (dataset, attack) cell, four bars (one per defense
    including no-defense baseline).  Error bars are 95% CIs over
    seeds.
    """
    plt = _setup_matplotlib()
    target = loaded.filter(experiment="exp_defense_efficacy")
    if not target.records:
        logger.warning("figure_5: no records found")
        return

    datasets = sorted({r["dataset"] for r in target.records})
    attacks = sorted({r["attack"] for r in target.records if r.get("attack")})
    defenses_order = ["no_defense", "D1_static", "D2_adaptive", "D3_schedulability"]

    fig, axes = plt.subplots(
        nrows=len(attacks), ncols=len(datasets),
        figsize=(2.8 * len(datasets), 2.0 * len(attacks)),
        squeeze=False,
    )
    for ai, atk in enumerate(attacks):
        for di, ds in enumerate(datasets):
            ax = axes[ai, di]
            means: List[float] = []
            errs_lo: List[float] = []
            errs_hi: List[float] = []
            for defense in defenses_order:
                d_filter = None if defense == "no_defense" else defense
                subset = target.filter(dataset=ds, attack=atk, defense=d_filter)
                miss = [_miss_rate(r) for r in subset.records]
                miss_clean = [m for m in miss if not math.isnan(m)]
                m, lo, hi = _mean_ci(miss_clean)
                means.append(m)
                errs_lo.append(max(0, m - lo))
                errs_hi.append(max(0, hi - m))
            x = np.arange(len(defenses_order))
            colors = [_DEFENSE_COLORS[d] for d in defenses_order]
            ax.bar(x, means, color=colors, yerr=[errs_lo, errs_hi],
                   capsize=2, edgecolor=_PALETTE["black"], lw=0.5)
            if ai == len(attacks) - 1:
                ax.set_xticks(x)
                ax.set_xticklabels(defenses_order, rotation=30, ha="right")
            else:
                ax.set_xticks([])
            if di == 0:
                ax.set_ylabel(f"{atk}\nmiss rate")
            if ai == 0:
                ax.set_title(ds)
            ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Defense efficacy: deadline-miss rate per (dataset, attack, defense)",
                 y=1.01)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "figure_5_defense_efficacy.pdf")
    fig.savefig(out_dir / "figure_5_defense_efficacy.png")
    plt.close(fig)
    logger.info(f"wrote figure_5 to {out_dir}")


def figure_6(loaded: LoadedRecords, out_dir: Path) -> None:
    """
    Figure 6 — Scalability curves: deadline-miss rate vs. stream size,
    two lines (no-defense, D3).
    """
    plt = _setup_matplotlib()
    target = loaded.filter(experiment="exp_scalability")
    if not target.records:
        logger.warning("figure_6: no records found")
        return
    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    # Group by (defense, n_transactions).  We need n_transactions; it's
    # in the run notes as "size=...".
    def extract_size(rec: Mapping[str, Any]) -> Optional[int]:
        for n in rec.get("notes", []):
            m = re.search(r"size=(\d+)", n)
            if m:
                return int(m.group(1))
        return None

    by_defense: Dict[Optional[str], Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in target.records:
        n = extract_size(r)
        if n is None:
            continue
        by_defense[r.get("defense")][n].append(_miss_rate(r))
    colors = {None: _PALETTE["red"], "D3_schedulability": _PALETTE["blue"]}
    labels = {None: "no defense", "D3_schedulability": "D3 (schedulability)"}
    for defense, sizes in by_defense.items():
        xs = sorted(sizes.keys())
        ys: List[float] = []
        lo: List[float] = []
        hi: List[float] = []
        for x in xs:
            vals = [v for v in sizes[x] if not math.isnan(v)]
            m, l, h = _mean_ci(vals)
            ys.append(m); lo.append(l); hi.append(h)
        ax.plot(xs, ys, marker="o", color=colors.get(defense, _PALETTE["grey"]),
                label=labels.get(defense, str(defense)))
        ax.fill_between(xs, lo, hi, alpha=0.2,
                        color=colors.get(defense, _PALETTE["grey"]))
    ax.set_xscale("log")
    ax.set_xlabel("Stream size (transactions)")
    ax.set_ylabel("Deadline-miss rate")
    ax.set_title("Scalability — Bitcoin ransomware, A3 attack")
    ax.legend(loc="best")
    ax.grid(True, which="both", alpha=0.3)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "figure_6_scalability.pdf")
    fig.savefig(out_dir / "figure_6_scalability.png")
    plt.close(fig)
    logger.info(f"wrote figure_6 to {out_dir}")


def figure_7(loaded: LoadedRecords, out_dir: Path) -> None:
    """Figure 7 — Cross-dataset attack comparison (one bar group per dataset)."""
    plt = _setup_matplotlib()
    target = loaded.filter(experiment="exp_cross_dataset")
    if not target.records:
        logger.warning("figure_7: no records found")
        return

    datasets = sorted({r["dataset"] for r in target.records})
    # Camera-ready: A6_evolutionary added (registry-driven enumeration
    # in attacks.py now includes the evolutionary search attack).
    attacks = ["A1_random", "A2_high_degree", "A3_branching_max",
               "A4_gradient_norm", "A5_adaptive", "A6_evolutionary"]
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    # Camera-ready: bar width adjusted from 0.15 to 0.13 and the
    # centring offset adjusted from (ai - 2) to (ai - 2.5) to
    # accommodate six bars per group instead of five.
    width = 0.13
    x = np.arange(len(datasets))
    for ai, atk in enumerate(attacks):
        means = []
        errs_lo = []
        errs_hi = []
        for ds in datasets:
            subset = target.filter(dataset=ds, attack=atk)
            miss = [_miss_rate(r) for r in subset.records]
            miss_clean = [m for m in miss if not math.isnan(m)]
            m, lo, hi = _mean_ci(miss_clean)
            means.append(m); errs_lo.append(max(0, m - lo)); errs_hi.append(max(0, hi - m))
        ax.bar(x + (ai - 2.5) * width, means, width=width,
               color=_ATTACK_COLORS.get(atk), yerr=[errs_lo, errs_hi],
               capsize=2, label=atk, edgecolor=_PALETTE["black"], lw=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=20, ha="right")
    ax.set_ylabel("Deadline-miss rate")
    ax.set_title("Cross-dataset attack effectiveness (no defense)")
    ax.legend(loc="best", ncol=2)
    ax.grid(True, axis="y", alpha=0.3)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "figure_7_cross_dataset.pdf")
    fig.savefig(out_dir / "figure_7_cross_dataset.png")
    plt.close(fig)
    logger.info(f"wrote figure_7 to {out_dir}")


def figure_8(loaded: LoadedRecords, out_dir: Path) -> None:
    """
    Figure 8 — Adaptive-adversary case study.  Bar chart of A5's
    miss rate against each defense (the worst-case scenario for the
    schedulability bound).
    """
    plt = _setup_matplotlib()
    target = loaded.filter(experiment="exp_adaptive_adversary")
    if not target.records:
        logger.warning("figure_8: no records found")
        return
    datasets = sorted({r["dataset"] for r in target.records})
    defenses = ["no_defense", "D1_static", "D2_adaptive", "D3_schedulability"]
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    width = 0.2
    x = np.arange(len(datasets))
    for di, defense in enumerate(defenses):
        d_filter = None if defense == "no_defense" else defense
        means = []
        errs_lo = []
        errs_hi = []
        for ds in datasets:
            subset = target.filter(dataset=ds, defense=d_filter)
            miss = [_miss_rate(r) for r in subset.records]
            miss_clean = [m for m in miss if not math.isnan(m)]
            m, lo, hi = _mean_ci(miss_clean)
            means.append(m); errs_lo.append(max(0, m - lo)); errs_hi.append(max(0, hi - m))
        ax.bar(x + (di - 1.5) * width, means, width=width,
               color=_DEFENSE_COLORS[defense], yerr=[errs_lo, errs_hi],
               capsize=2, label=defense, edgecolor=_PALETTE["black"], lw=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=20, ha="right")
    ax.set_ylabel("Deadline-miss rate (A5 attack)")
    ax.set_title("A5 (adaptive white-box) vs each defense")
    ax.legend(loc="best", ncol=2)
    ax.grid(True, axis="y", alpha=0.3)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "figure_8_adaptive_adversary.pdf")
    fig.savefig(out_dir / "figure_8_adaptive_adversary.png")
    plt.close(fig)
    logger.info(f"wrote figure_8 to {out_dir}")


# --- Camera-ready: new figures ------------------------------------------


def _shade_intervals(
    ax: Any,
    boundaries: Mapping[str, Any],
    plt_module: Any,
) -> None:
    """
    Shade warmup and recovery intervals on a time-axis plot.  Storm
    is unshaded (white) so the eye notices the contrast.  Adds
    vertical lines at the transitions and a small text label below
    the top of the axes.

    `boundaries` is the parsed intervals note from the run record;
    its keys are run_start_ns, warmup_end_ns, storm_end_ns,
    run_end_ns, plus the *_duration_s keys.
    """
    if not boundaries:
        return
    rs = boundaries.get("run_start_ns")
    we = boundaries.get("warmup_end_ns")
    se = boundaries.get("storm_end_ns")
    re_end = boundaries.get("run_end_ns")
    if None in (rs, we, se, re_end):
        return
    # Convert to seconds since run_start.
    t0 = float(rs) / 1e9
    we_s = float(we) / 1e9 - t0
    se_s = float(se) / 1e9 - t0
    re_s = float(re_end) / 1e9 - t0

    # Only shade if the warmup/recovery intervals are non-trivial.
    if we_s > 0:
        ax.axvspan(0.0, we_s, color=_INTERVAL_SHADING["warmup"], zorder=0)
        ax.axvline(we_s, color=_PALETTE["grey"], lw=0.5, ls=":")
    if re_s > se_s:
        ax.axvspan(se_s, re_s, color=_INTERVAL_SHADING["recovery"], zorder=0)
        ax.axvline(se_s, color=_PALETTE["grey"], lw=0.5, ls=":")


def _select_representative_record(
    records: Sequence[Mapping[str, Any]],
) -> Optional[Mapping[str, Any]]:
    """
    Pick the record whose run duration is closest to the median.
    Used by figure_9 to choose one seed per (dataset, attack,
    defense) cell when plotting time-series, since plotting all
    seeds clutters and an "average time series" requires
    re-aligning to a shared time axis.
    """
    if not records:
        return None
    durations: List[Tuple[float, Mapping[str, Any]]] = []
    for r in records:
        b = _interval_boundaries(r)
        if b is None:
            continue
        d = float(b.get("total_duration_s", 0.0))
        durations.append((d, r))
    if not durations:
        return records[0]
    durations.sort(key=lambda kv: kv[0])
    mid = len(durations) // 2
    return durations[mid][1]


def figure_9(
    loaded: LoadedRecords,
    out_dir: Path,
    dataset: str = "cicids2018",
    attack: str = "A3_branching_max",
) -> None:
    """
    Figure 9 — Storm dynamics (Qing & Zheng / DRRS-style §V.1
    headline).  Three stacked panels sharing x-axis (time, seconds
    since run start):
      panel 1: end-to-end latency over time, one line per defense
      panel 2: queue depth over time
      panel 3: throughput per bucket (admitted + rejected)

    Warmup and recovery intervals are shaded.  Storm interval is
    unshaded so the contrast highlights it.

    Selects records from exp_defense_efficacy with WSR enabled.  If
    no records match (dataset, attack), falls back to the first
    available cell.
    """
    plt = _setup_matplotlib()

    # Find records to plot: one representative per defense.
    target = loaded.filter(experiment="exp_defense_efficacy",
                           dataset=dataset, attack=attack)
    if not target.records:
        # Fallback: any defense_efficacy with WSR.
        candidates = loaded.filter(experiment="exp_defense_efficacy")
        if not candidates.records:
            logger.warning("figure_9: no records found")
            return
        # Pick whichever (dataset, attack) cell has WSR runs.
        for r in candidates.records:
            if _interval_boundaries(r) is not None:
                dataset = r["dataset"]
                attack = r.get("attack", "")
                target = candidates.filter(
                    dataset=dataset,
                    attack=attack if attack else None,
                )
                break
        if not target.records:
            logger.warning("figure_9: no WSR records found")
            return

    defenses = ["no_defense", "D1_static", "D2_adaptive", "D3_schedulability"]
    selected: Dict[str, Mapping[str, Any]] = {}
    for defense in defenses:
        d_filter = None if defense == "no_defense" else defense
        subset = target.filter(defense=d_filter)
        # Only consider WSR records (those with intervals notes).
        wsr_records = [
            r for r in subset.records
            if _interval_boundaries(r) is not None
        ]
        rep = _select_representative_record(wsr_records)
        if rep is not None:
            selected[defense] = rep

    if not selected:
        logger.warning("figure_9: no representative WSR records found")
        return

    fig, axes = plt.subplots(3, 1, figsize=(7.0, 6.0), sharex=True,
                              gridspec_kw={"height_ratios": [3, 2, 2]})
    ax_lat, ax_queue, ax_thr = axes

    # Use the FIRST defense's intervals to set the shading.  All
    # defenses use the same warmup/recovery fractions, so
    # boundaries should be very similar across defenses (modulo
    # processing-rate differences).  We accept this small
    # imprecision rather than fan out shading per defense, which
    # would clutter.
    representative_boundaries = None
    for defense in defenses:
        if defense in selected:
            representative_boundaries = _interval_boundaries(selected[defense])
            if representative_boundaries:
                break
    if representative_boundaries:
        _shade_intervals(ax_lat, representative_boundaries, plt)
        _shade_intervals(ax_queue, representative_boundaries, plt)
        _shade_intervals(ax_thr, representative_boundaries, plt)

    # Panel 1: latency over time.
    for defense in defenses:
        if defense not in selected:
            continue
        rec = selected[defense]
        lat = _latency_over_time(rec)
        if not lat:
            continue
        rs = _run_start_ns(rec) or lat[0][0]
        ts = np.array([(t - rs) / 1e9 for t, _ in lat])
        ys = np.array([y for _, y in lat])
        # Smooth with a small moving average for legibility (window=10).
        if ys.size >= 50:
            window = max(1, ys.size // 200)
            kernel = np.ones(window) / window
            ys_smooth = np.convolve(ys, kernel, mode="same")
        else:
            ys_smooth = ys
        ax_lat.plot(ts, ys_smooth, color=_DEFENSE_COLORS[defense],
                    lw=1.0, label=defense, alpha=0.85)
    ax_lat.set_yscale("log")
    ax_lat.set_ylabel("Latency (µs)")
    ax_lat.set_title(f"Storm dynamics — {dataset}, {attack}")
    ax_lat.legend(loc="upper right", ncol=2)
    ax_lat.grid(True, which="both", alpha=0.3)

    # Panel 2: queue depth over time.  Use one defense's queue
    # series (D3 if available, else first available) to keep the
    # plot legible — queue depth is a property of the harness, not
    # primarily of the defense, in this codebase.
    chosen_for_queue = "D3_schedulability" if "D3_schedulability" in selected else next(iter(selected))
    rec = selected[chosen_for_queue]
    qd = _queue_depth_events(rec)
    if qd:
        rs = _run_start_ns(rec) or qd[0][0]
        ts = np.array([(t - rs) / 1e9 for t, _ in qd])
        ds = np.array([d for _, d in qd])
        ax_queue.step(ts, ds, where="post",
                      color=_DEFENSE_COLORS[chosen_for_queue],
                      lw=0.8, alpha=0.9)
    ax_queue.set_ylabel("Queue depth")
    ax_queue.grid(True, alpha=0.3)

    # Panel 3: throughput (admitted + rejected stacked) for the
    # chosen defense.
    tb = _throughput_buckets(rec)
    if tb:
        rs = _run_start_ns(rec) or int(tb[0]["t_ns"])
        bw_ns = int(tb[0].get("bucket_width_ns", 100_000_000))
        bw_s = bw_ns / 1e9
        ts = np.array([(int(b["t_ns"]) - rs) / 1e9 for b in tb])
        adm = np.array([float(b.get("admitted", 0)) / bw_s for b in tb])
        rej = np.array([float(b.get("rejected", 0)) / bw_s for b in tb])
        # Stacked: admitted on bottom, rejected on top.
        ax_thr.fill_between(ts, 0, adm, step="mid",
                             color=_DEFENSE_COLORS[chosen_for_queue],
                             alpha=0.6, label="admitted")
        ax_thr.fill_between(ts, adm, adm + rej, step="mid",
                             color=_PALETTE["red"], alpha=0.4,
                             label="rejected")
    ax_thr.set_xlabel("Time (s) since run start")
    ax_thr.set_ylabel("Throughput (txn/s)")
    ax_thr.legend(loc="upper right")
    ax_thr.grid(True, alpha=0.3)

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "figure_9_storm_dynamics.pdf")
    fig.savefig(out_dir / "figure_9_storm_dynamics.png")
    plt.close(fig)
    logger.info(f"wrote figure_9 to {out_dir}")


def figure_10(
    loaded: LoadedRecords,
    out_dir: Path,
    dataset: str = "cicids2018",
) -> None:
    """
    Figure 10 — Freshness CCDFs (Yates AoI demand, §V.2).

    Two side-by-side panels:
      panel 1: model-age CCDF P(age > a) for each (attack, defense).
               Vertical line at A_max.
      panel 2: propagation-latency CCDF P(commit−arrival > τ) for
               each (attack, defense).  Vertical line at the
               contract deadline.

    Drawn from exp_defense_efficacy.  Falls back to
    exp_attack_effectiveness if exp_defense_efficacy is empty.
    """
    plt = _setup_matplotlib()

    target = loaded.filter(experiment="exp_defense_efficacy",
                           dataset=dataset)
    if not target.records:
        target = loaded.filter(experiment="exp_attack_effectiveness",
                               dataset=dataset)
    if not target.records:
        # Fall back to whichever dataset has data.
        for exp_name in ("exp_defense_efficacy",
                         "exp_attack_effectiveness"):
            cand = loaded.filter(experiment=exp_name)
            if cand.records:
                target = cand
                dataset = target.records[0]["dataset"]
                break
    if not target.records:
        logger.warning("figure_10: no records found")
        return

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    ax_age, ax_prop = axes

    # Group by defense so we can pool per-seed samples.
    defenses = ["no_defense", "D1_static", "D2_adaptive", "D3_schedulability"]
    age_max_us: Optional[float] = None
    deadline_us: Optional[float] = None

    for defense in defenses:
        d_filter = None if defense == "no_defense" else defense
        subset = target.filter(defense=d_filter)
        if not subset.records:
            continue

        # Pool model-age samples across seeds.
        age_pool: List[float] = []
        prop_pool: List[float] = []
        for r in subset.records:
            age_pool.extend(_age_at_decision_samples_us(r).tolist())
            prop_pool.extend(_propagation_samples_us(r).tolist())
            if age_max_us is None:
                _, _, am = _age_violation_samples(r)
                if am is not None:
                    age_max_us = float(am)
            if deadline_us is None:
                mr = r.get("miss_report") or {}
                if "deadline_us" in mr:
                    deadline_us = float(mr["deadline_us"])

        if age_pool:
            arr = np.sort(np.asarray(age_pool))
            x_sub, ccdf_sub = _subsample_for_ccdf(arr)
            ax_age.semilogy(x_sub, ccdf_sub,
                             color=_DEFENSE_COLORS[defense],
                             lw=1.0, label=defense)
        if prop_pool:
            arr = np.sort(np.asarray(prop_pool))
            x_sub, ccdf_sub = _subsample_for_ccdf(arr)
            ax_prop.semilogy(x_sub, ccdf_sub,
                              color=_DEFENSE_COLORS[defense],
                              lw=1.0, label=defense)

    if age_max_us is not None:
        ax_age.axvline(age_max_us, color=_PALETTE["grey"],
                        lw=0.6, ls=":", label="A_max")
    if deadline_us is not None:
        ax_prop.axvline(deadline_us, color=_PALETTE["grey"],
                         lw=0.6, ls=":", label="Deadline D")

    ax_age.set_xlabel("Model age at decision (µs)")
    ax_age.set_ylabel("P(age > a)")
    ax_age.set_title(f"AoI: model-age CCDF — {dataset}")
    ax_age.grid(True, which="both", alpha=0.3)
    ax_age.legend(loc="lower left", ncol=1)

    ax_prop.set_xlabel("Propagation latency (µs)")
    ax_prop.set_ylabel("P(τ > t)")
    ax_prop.set_title(f"AoI: propagation latency CCDF — {dataset}")
    ax_prop.grid(True, which="both", alpha=0.3)
    ax_prop.legend(loc="lower left", ncol=1)

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "figure_10_freshness_ccdf.pdf")
    fig.savefig(out_dir / "figure_10_freshness_ccdf.png")
    plt.close(fig)
    logger.info(f"wrote figure_10 to {out_dir}")


# =============================================================================
# Section 7.  All-figures-and-tables convenience.
# =============================================================================


def render_all(raw_dir: Path, out_dir: Path) -> None:
    """
    Render every figure and table in the paper from the raw run
    records.  Used by reproduce.sh after experiments.py finishes.

    Camera-ready: also renders the three new tables and two new
    figures.  Each renderer is independent and skips gracefully
    when its required data is absent.
    """
    loaded = load_records(raw_dir)
    # Existing tables (§I + §V + §VI).
    table_1(loaded, out_dir)
    table_2(loaded, out_dir)
    table_3(loaded, out_dir)
    table_4(loaded, out_dir)
    table_5(loaded, out_dir)
    table_6(loaded, out_dir)
    # New tables (§V.2 + §V.3 + §V.4).
    table_7(loaded, out_dir)
    table_8(loaded, out_dir)
    table_9(loaded, out_dir)
    # Existing figures.
    figure_3(loaded, out_dir)
    figure_4(loaded, out_dir)
    figure_5(loaded, out_dir)
    figure_6(loaded, out_dir)
    figure_7(loaded, out_dir)
    figure_8(loaded, out_dir)
    # New figures (§V.1 + §V.2).
    figure_9(loaded, out_dir)
    figure_10(loaded, out_dir)


# =============================================================================
# Section 8.  Public surface.
# =============================================================================

__all__ = [
    "LoadedRecords",
    "load_records",
    "render_all",
    "table_1", "table_2", "table_3", "table_4", "table_5", "table_6",
    "table_7", "table_8", "table_9",
    "figure_3", "figure_4", "figure_5", "figure_6", "figure_7", "figure_8",
    "figure_9", "figure_10",
]
