"""
test_plots.py — Tests for plots.py.

The submission-draft plots.py had no automated coverage.  This file
adds focused tests for the camera-ready additions, particularly
the filter-semantics fix (the most impactful correctness bug in
the file) and the new note-field extractors that consume
camera-ready experiments.py output.

What this file covers
---------------------
1. LoadedRecords.filter semantics: the camera-ready _ANY sentinel,
   and that None means "field equals None" rather than "do not
   filter."
2. Note parsers: the kv-format and JSON-format helpers, including
   the new analysis_alpha parser that handles commas and
   parentheses in the source string.
3. Camera-ready note-field extractors: alpha_source_and_value,
   benign_fallback_flag, lambda_at_edge_flag, holds_ci_95_flag,
   extract_streaming_f1.
4. Palette completeness: A6_evolutionary has a distinct colour.
5. Table generators: table_8 produces the camera-ready columns.

What this file deliberately skips
---------------------------------
- Figure rendering.  Verifying matplotlib output requires snapshot
  testing infrastructure not warranted for an academic paper.  The
  figure generators do call _setup_matplotlib() and matplotlib's
  Agg backend, but we do not assert pixel-level output.
- _subsample_for_ccdf statistical tightness.  The function is
  unbiased by construction (returns CCDF computed against the full
  array, not the subsample); we test it produces output of the
  expected shape, not its statistical properties.
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Any, Mapping

import pytest

import plots
from plots import (
    LoadedRecords,
    _alpha_source_and_value,
    _benign_fallback_flag,
    _extract_streaming_f1,
    _holds_ci_95_flag,
    _kv_bool,
    _kv_float,
    _lambda_at_edge_flag,
    _parse_kv_note,
    _parse_json_note,
    _split_kv_string,
    _subsample_for_ccdf,
    load_records,
    table_8,
)


# =============================================================================
# Helpers: build synthetic run records that exercise the parsers.
# =============================================================================


def _record(
    experiment: str = "exp_test",
    dataset: str = "synthetic",
    attack: Any = None,
    defense: Any = None,
    seed: int = 0,
    notes: list = None,
    miss_rate: float = 0.005,
    deadline_us: float = 2000.0,
) -> Mapping[str, Any]:
    return {
        "experiment": experiment,
        "dataset": dataset,
        "attack": attack,
        "defense": defense,
        "seed": seed,
        "miss_report": {
            "n_transactions": 10000,
            "n_misses": int(miss_rate * 10000),
            "miss_rate": miss_rate,
            "deadline_us": deadline_us,
            "epsilon": 0.001,
        },
        "end_to_end": {"samples_ns": [int(deadline_us * 0.5 * 1000)] * 100},
        "notes": notes or [],
    }


import numpy as np


# =============================================================================
# Section 1 — LoadedRecords.filter semantics (the most impactful fix).
# =============================================================================


class TestFilterSemantics:
    """The submission-draft filter() interpreted None as 'do not
    filter on this field'.  Every call site in plots.py used
    ``d_filter = None if defense == 'no_defense' else defense``
    intending the *opposite* semantics: 'filter to records where
    defense IS None'.

    Camera-ready: the _ANY sentinel separates the two cases.
    Calling filter() with no arguments returns everything; passing
    None for a field filters to records where that field equals None.
    """

    def setup_method(self):
        self.recs = LoadedRecords([
            _record(attack=None, defense=None),
            _record(attack="A3_branching_max", defense=None),
            _record(attack=None, defense="D1_static"),
            _record(attack="A3_branching_max", defense="D3_schedulability"),
        ])

    def test_filter_no_args_returns_all(self):
        assert len(self.recs.filter()) == 4

    def test_filter_attack_equals_none_keeps_only_benign(self):
        """Submission-draft would have returned all 4 records."""
        out = self.recs.filter(attack=None)
        assert len(out) == 2  # benign-no-defense + benign-D1
        for r in out.records:
            assert r["attack"] is None

    def test_filter_defense_equals_none_keeps_only_undefended(self):
        """Submission-draft would have returned all 4 records,
        silently mixing defended into the no-defense row of every
        table that used this filter."""
        out = self.recs.filter(defense=None)
        assert len(out) == 2  # benign-no-defense + A3-no-defense
        for r in out.records:
            assert r["defense"] is None

    def test_filter_both_none_keeps_only_no_attack_no_defense(self):
        out = self.recs.filter(attack=None, defense=None)
        assert len(out) == 1
        assert out.records[0]["attack"] is None
        assert out.records[0]["defense"] is None

    def test_filter_specific_value_filters_by_equality(self):
        out = self.recs.filter(attack="A3_branching_max")
        assert len(out) == 2
        for r in out.records:
            assert r["attack"] == "A3_branching_max"

    def test_filter_combined_specific_and_none(self):
        out = self.recs.filter(
            attack="A3_branching_max",
            defense=None,
        )
        assert len(out) == 1
        assert out.records[0]["attack"] == "A3_branching_max"
        assert out.records[0]["defense"] is None

    def test_filter_combined_specific_value_pair(self):
        out = self.recs.filter(
            attack="A3_branching_max",
            defense="D3_schedulability",
        )
        assert len(out) == 1

    def test_filter_returns_loaded_records(self):
        """Filter chains: filter() returns LoadedRecords, so further
        filtering works."""
        out = self.recs.filter(attack="A3_branching_max")
        assert isinstance(out, LoadedRecords)
        # Chain.
        out2 = out.filter(defense=None)
        assert len(out2) == 1


# =============================================================================
# Section 2 — Note parsers (kv-format and JSON-format).
# =============================================================================


class TestNoteParsers:
    def test_split_kv_string_basic(self):
        out = _split_kv_string("a=1, b=2, c=3")
        assert out == {"a": "1", "b": "2", "c": "3"}

    def test_split_kv_string_with_quoted_values(self):
        """Notes with single-quoted strings can contain commas
        without confusing the parser."""
        out = _split_kv_string("a=1, msg='hello, world', b=2")
        assert out == {"a": "1", "msg": "hello, world", "b": "2"}

    def test_split_kv_string_with_lambda_star(self):
        """Camera-ready experiments.py emits ``lambda*=0.0723``
        (the asterisk is part of the key name)."""
        out = _split_kv_string("lambda*=0.0723, n_classes=2")
        assert out == {"lambda*": "0.0723", "n_classes": "2"}

    def test_kv_float_strips_us_suffix(self):
        kv = {"T": "1500us"}
        assert _kv_float(kv, "T") == 1500.0

    def test_kv_float_strips_ms_suffix(self):
        kv = {"D": "10ms"}
        assert _kv_float(kv, "D") == 10.0

    def test_kv_float_handles_missing_key(self):
        assert math.isnan(_kv_float({}, "missing"))

    def test_kv_float_returns_default_on_missing(self):
        assert _kv_float({}, "missing", default=42.0) == 42.0

    def test_kv_bool_true_aliases(self):
        for v in ("True", "true", "1", "yes"):
            assert _kv_bool({"f": v}, "f") is True

    def test_kv_bool_false_aliases(self):
        for v in ("False", "false", "0", "no"):
            assert _kv_bool({"f": v}, "f") is False

    def test_kv_bool_returns_none_on_missing(self):
        assert _kv_bool({}, "missing") is None

    def test_parse_kv_note_extracts_first_match(self):
        rec = _record(notes=[
            "analysis_envelope: T_min=1500us, feasible=True, slack=300us",
            "other: x=1",
        ])
        kv = _parse_kv_note(rec, "analysis_envelope")
        assert kv["T_min"] == "1500us"
        assert kv["feasible"] == "True"

    def test_parse_kv_note_returns_empty_when_absent(self):
        rec = _record(notes=["other: x=1"])
        kv = _parse_kv_note(rec, "analysis_envelope")
        assert kv == {}

    def test_parse_json_note_extracts_intervals(self):
        rec = _record(notes=[
            'intervals: {"run_start_ns": 1000000000, "warmup_end_ns": 1500000000}',
        ])
        out = _parse_json_note(rec, "intervals")
        assert out is not None
        assert out["run_start_ns"] == 1000000000


# =============================================================================
# Section 3 — Camera-ready note-field extractors.
# =============================================================================


class TestCameraReadyExtractors:
    def test_alpha_source_and_value_ground_truth(self):
        """Camera-ready experiments.py emits
        ``analysis_alpha: source=ground-truth (per_txn_log), measured=0.0473``."""
        rec = _record(notes=[
            "analysis_alpha: source=ground-truth (per_txn_log), measured=0.0473",
        ])
        src, val = _alpha_source_and_value(rec)
        assert src == "ground-truth (per_txn_log)"
        assert abs(val - 0.0473) < 1e-9

    def test_alpha_source_and_value_benign_fallback(self):
        rec = _record(notes=[
            "analysis_alpha: source=no per_txn_log; benign-only fallback, measured=0.0000",
        ])
        src, val = _alpha_source_and_value(rec)
        # The source string contains a comma in the prose:
        # "no per_txn_log; benign-only fallback".  The parser must
        # split on the LAST ``, measured=`` so the comma in the
        # source value is preserved.
        assert "benign-only fallback" in src
        assert val == 0.0

    def test_alpha_source_and_value_absent_returns_none(self):
        rec = _record(notes=["other: x=1"])
        src, val = _alpha_source_and_value(rec)
        assert src is None
        assert math.isnan(val)

    def test_benign_fallback_flag_true(self):
        rec = _record(notes=[
            "analysis_envelope: T_min=1500us, feasible=True, "
            "slack=300us, benign_fallback=True",
        ])
        assert _benign_fallback_flag(rec) is True

    def test_benign_fallback_flag_false(self):
        rec = _record(notes=[
            "analysis_envelope: T_min=1500us, feasible=True, "
            "slack=300us, benign_fallback=False",
        ])
        assert _benign_fallback_flag(rec) is False

    def test_benign_fallback_flag_absent(self):
        rec = _record(notes=[])
        assert _benign_fallback_flag(rec) is None

    def test_lambda_at_edge_flag_from_d3_mgf_cert(self):
        rec = _record(notes=[
            "d3_mgf_cert: feasible=True, bound=0.001, log_bound=-6.91, "
            "lambda*=0.0723, lambda_at_edge=True, epsilon=0.001",
        ])
        assert _lambda_at_edge_flag(rec) is True

    def test_lambda_at_edge_flag_falls_back_to_analysis_mgf(self):
        rec = _record(notes=[
            "analysis_mgf: feasible=True, bound=0.001, "
            "log_bound=-6.91, lambda*=0.0723, lambda_at_edge=False, "
            "n_classes=2, notes='ok'",
        ])
        assert _lambda_at_edge_flag(rec) is False

    def test_lambda_at_edge_flag_absent(self):
        assert _lambda_at_edge_flag(_record(notes=[])) is None

    def test_holds_ci_95_flag_true(self):
        rec = _record(notes=[
            "validate_4_3_mgf: holds=True, holds_ci_95=True, "
            "bound=0.001, measured=0.0005, notes='ok'",
        ])
        assert _holds_ci_95_flag(rec) is True

    def test_holds_ci_95_flag_false(self):
        rec = _record(notes=[
            "validate_4_3_mgf: holds=True, holds_ci_95=False, "
            "bound=0.001, measured=0.0005, notes='ok'",
        ])
        assert _holds_ci_95_flag(rec) is False

    def test_extract_streaming_f1_not_computed(self):
        """Camera-ready experiments.py emits 'F1 not computed' when
        the placeholder evaluator is bypassed.  This MUST return
        NaN, not 1.0 (the submission-draft fabricated value)."""
        rec = _record(notes=[
            "detection: F1 not computed (placeholder evaluator removed; "
            "see docs/F1_EVAL.md for the planned real evaluator)",
        ])
        f1 = _extract_streaming_f1(rec)
        assert math.isnan(f1), (
            f"expected NaN for 'not computed', got {f1}"
        )

    def test_extract_streaming_f1_real_value(self):
        """When a real evaluator IS wired in, the parser reads the
        actual F1 value."""
        rec = _record(notes=[
            "detection: streaming_F1=0.8765, snapshot_F1=0.9012",
        ])
        f1 = _extract_streaming_f1(rec)
        assert abs(f1 - 0.8765) < 1e-9

    def test_extract_streaming_f1_absent(self):
        f1 = _extract_streaming_f1(_record(notes=[]))
        assert math.isnan(f1)


# =============================================================================
# Section 4 — Palette completeness.
# =============================================================================


class TestPalette:
    def test_a6_has_distinct_colour(self):
        """The submission draft's _ATTACK_COLORS only listed A1..A5;
        A6 records would render in fallback grey, indistinguishable
        from A1.  Camera-ready: A6 has its own colour."""
        assert "A6_evolutionary" in plots._ATTACK_COLORS
        assert (
            plots._ATTACK_COLORS["A6_evolutionary"]
            != plots._ATTACK_COLORS["A1_random"]
        )

    def test_all_six_attacks_have_colours(self):
        for atk in (
            "A1_random", "A2_high_degree", "A3_branching_max",
            "A4_gradient_norm", "A5_adaptive", "A6_evolutionary",
        ):
            assert atk in plots._ATTACK_COLORS, (
                f"{atk} missing from _ATTACK_COLORS"
            )

    def test_six_attacks_have_six_distinct_colours(self):
        colours = {
            plots._ATTACK_COLORS[atk]
            for atk in (
                "A1_random", "A2_high_degree", "A3_branching_max",
                "A4_gradient_norm", "A5_adaptive", "A6_evolutionary",
            )
        }
        assert len(colours) == 6, (
            f"colour collision among A1..A6: only {len(colours)} "
            f"distinct"
        )


# =============================================================================
# Section 5 — _subsample_for_ccdf.
# =============================================================================


class TestSubsampleForCCDF:
    def test_returns_full_array_when_short(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        x, ccdf = _subsample_for_ccdf(arr, max_points=10)
        assert len(x) == 5
        assert len(ccdf) == 5
        # CCDF is monotone non-increasing.
        for i in range(1, len(ccdf)):
            assert ccdf[i] <= ccdf[i - 1]

    def test_subsamples_when_long(self):
        arr = np.sort(np.random.RandomState(0).uniform(0, 100, 10_000))
        x, ccdf = _subsample_for_ccdf(arr, max_points=200)
        assert len(x) <= 200
        assert len(ccdf) == len(x)

    def test_empty_input(self):
        x, ccdf = _subsample_for_ccdf(np.zeros(0), max_points=10)
        assert len(x) == 0
        assert len(ccdf) == 0


# =============================================================================
# Section 6 — table_8 generation with camera-ready columns.
# =============================================================================


class TestTable8:
    def _make_loaded(self, tmpdir: Path) -> LoadedRecords:
        """Build a synthetic record set that exercises every
        camera-ready table_8 column."""
        recs = []
        for seed in range(5):
            recs.append(_record(
                experiment="exp_schedulability_validation",
                dataset="cicids2018",
                attack="A3_branching_max",
                defense="D3_schedulability",
                seed=seed,
                miss_rate=0.0005,
                notes=[
                    f"analysis_envelope: T_min=1500us, feasible=True, "
                    f"slack=300us, benign_fallback=False",
                    f"d3_envelope_cert: feasible=True, T=1500us, "
                    f"bound_at_T=0.0008, slack=300us, epsilon=0.001",
                    f"d3_mgf_cert: feasible=True, bound=0.001, "
                    f"log_bound=-6.91, lambda*=0.0723, "
                    f"lambda_at_edge={'True' if seed == 0 else 'False'}, "
                    f"epsilon=0.001",
                    f"validate_4_3_mgf: holds=True, "
                    f"holds_ci_95={'True' if seed != 4 else 'False'}, "
                    f"bound=0.001, measured=0.0005, notes='ok'",
                    f"analysis_alpha: source=ground-truth (per_txn_log), "
                    f"measured=0.0473",
                ],
            ))
        # Write to disk.
        for i, r in enumerate(recs):
            (tmpdir / f"rec_{i:03d}.json").write_text(json.dumps(r))
        return load_records(tmpdir)

    def test_generates_csv_with_camera_ready_columns(self, tmp_path):
        loaded = self._make_loaded(tmp_path)
        out_dir = tmp_path / "out"
        table_8(loaded, out_dir)
        csv_path = out_dir / "table_8_schedulability_certificates.csv"
        assert csv_path.exists()
        import csv
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            cols = set(reader.fieldnames or [])

        # Submission-draft columns still present.
        assert {
            "dataset", "n_runs", "env_feasible_pct", "env_T_us_mean",
            "mgf_feasible_pct", "mgf_log_bound_mean",
            "validate_holds_pct", "validate_bound_mean",
        }.issubset(cols)
        # Camera-ready columns added.
        assert {
            "alpha_source", "alpha_measured_mean", "benign_fallback_pct",
            "mgf_lambda_at_edge_pct", "validate_holds_ci_95_pct",
        }.issubset(cols), f"camera-ready columns missing.  present: {cols}"

    def test_lambda_at_edge_pct_aggregates_correctly(self, tmp_path):
        loaded = self._make_loaded(tmp_path)
        out_dir = tmp_path / "out"
        table_8(loaded, out_dir)
        csv_path = out_dir / "table_8_schedulability_certificates.csv"
        import csv
        with open(csv_path) as f:
            row = next(csv.DictReader(f))
        # 1 of 5 seeds had lambda_at_edge=True.
        assert abs(float(row["mgf_lambda_at_edge_pct"]) - 0.2) < 1e-9

    def test_validate_holds_ci_95_pct_aggregates_correctly(self, tmp_path):
        loaded = self._make_loaded(tmp_path)
        out_dir = tmp_path / "out"
        table_8(loaded, out_dir)
        csv_path = out_dir / "table_8_schedulability_certificates.csv"
        import csv
        with open(csv_path) as f:
            row = next(csv.DictReader(f))
        # 4 of 5 seeds had holds_ci_95=True.
        assert abs(float(row["validate_holds_ci_95_pct"]) - 0.8) < 1e-9

    def test_alpha_source_summary_when_seeds_agree(self, tmp_path):
        loaded = self._make_loaded(tmp_path)
        out_dir = tmp_path / "out"
        table_8(loaded, out_dir)
        csv_path = out_dir / "table_8_schedulability_certificates.csv"
        import csv
        with open(csv_path) as f:
            row = next(csv.DictReader(f))
        # All seeds agreed on the source string, so the row
        # reports it verbatim (no 'mixed').
        assert row["alpha_source"] == "ground-truth (per_txn_log)"

    def test_table_8_runs_with_no_records(self, tmp_path):
        """When no D3 schedulability validation records exist,
        table_8 should still produce a (possibly empty) CSV."""
        out_dir = tmp_path / "out"
        empty = LoadedRecords([])
        table_8(empty, out_dir)
        csv_path = out_dir / "table_8_schedulability_certificates.csv"
        assert csv_path.exists()


# =============================================================================
# Section 7 — load_records.
# =============================================================================


class TestLoadRecords:
    def test_loads_valid_json_files(self, tmp_path):
        rec = _record()
        (tmp_path / "good.json").write_text(json.dumps(rec))
        loaded = load_records(tmp_path)
        assert len(loaded) == 1

    def test_skips_invalid_json(self, tmp_path):
        """Bad records must be logged-and-skipped, not fatal —
        otherwise one corrupted file would block all rendering."""
        (tmp_path / "good.json").write_text(json.dumps(_record()))
        (tmp_path / "bad.json").write_text("{not valid json")
        loaded = load_records(tmp_path)
        # Only the good record loaded.
        assert len(loaded) == 1

    def test_loads_recursively(self, tmp_path):
        sub = tmp_path / "sub" / "nested"
        sub.mkdir(parents=True)
        (sub / "rec.json").write_text(json.dumps(_record()))
        loaded = load_records(tmp_path)
        assert len(loaded) == 1
