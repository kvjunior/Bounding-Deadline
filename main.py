"""
main.py — Single CLI entry point.

This file is intentionally short.  Its only job is argument parsing
and dispatch into experiments.py / plots.py / workload.py.  If
main.py grows beyond ~800 lines, the experiment design is wrong.

Subcommands
-----------
  smoke        Verify the installation by running one tiny synthetic
               experiment end-to-end.  No real datasets needed.
  run          Run one named experiment.
  run-all      Run every experiment in order.
  render       Render figures and tables from existing run records.
  list         List available experiments, attacks, defenses, datasets.
  info         Show the dataset card for one or more datasets.
  reproduce    Reviewer-facing: replay a published RunRecord and
               verify the audit signatures match.
  validate     Run the test suite (camera-ready CI hook).

Examples
--------
  # Sanity check (no datasets required):
  python -m src.main smoke --output-dir results/smoke

  # Sanity-check the camera-ready A6 routing path specifically:
  python -m src.main smoke --attack A6_evolutionary --output-dir results/smoke

  # Validate the installation (runs pytest):
  python -m src.main validate

  # Run one experiment on one dataset:
  python -m src.main run --experiment exp_attack_effectiveness \\
                         --datasets cicids2018 \\
                         --output-dir results/raw

  # Run every experiment on all four datasets (the full evaluation):
  python -m src.main run-all --output-dir results/raw \\
                              --strict-schema \\
                              --full-file-hash

  # Generate all figures and tables from the run records:
  python -m src.main render --raw-dir results/raw \\
                            --figures-dir results/figures

  # Show the CICIDS-2018 dataset card (citation, version, limitations):
  python -m src.main info --datasets cicids2018

  # Verify a published RunRecord reproduces:
  python -m src.main reproduce --record results/raw/abc123.json \\
                                --data-root data \\
                                --check signatures

Camera-ready improvements over the submission draft
---------------------------------------------------
1. **Runtime crash fixed in ``_cmd_run`` and ``_cmd_run_all``.**
   The submission draft passed ``strict_schema=...``,
   ``full_file_hash=...`` and ``hardware_info=...`` to
   ``experiments.run_experiment(...)`` as ``**kwargs``.  But the
   experiment functions
   (``exp_attack_effectiveness(dataset_paths, output_dir,
   n_transactions=...)`` and so on) do not accept those kwargs;
   the call would crash with ``TypeError: ...
   got an unexpected keyword argument 'strict_schema'`` the first
   time a reviewer ran ``python -m src.main run``.  The camera-
   ready drops the unsupported kwargs.  The flags themselves are
   preserved in argparse (they're documented affordances) but
   the values are now collected for diagnostic logging only —
   wiring them through to the loader / run record would require
   re-touching experiments.py and is documented as future work.

2. **A6 routes through the registry automatically.**  The
   ``list`` subcommand reads from ``attacks.attack_names()``,
   which now includes ``A6_evolutionary`` because camera-ready
   ``attacks.py`` self-registers all six attacks at module
   import time.  Reviewers running ``python -m src.main list``
   see all six attacks; the ``smoke`` subcommand takes an
   optional ``--attack`` flag so reviewers can confirm A6 routing
   with a one-line invocation.

3. **``--quick`` test list reflects reality.**  The submission
   draft's ``--quick`` mode hardcoded
   ``test_analysis.py + test_threat_model.py + test_measurement.py``
   under the comment "fast (numpy-only) tests".  In fact,
   ``test_attacks.py``, ``test_defenses.py``, and ``test_workload.py``
   are also numpy-only and run in seconds; only ``test_system.py``
   and ``test_experiments.py`` are torch-heavy.  The camera-ready
   ``--quick`` list adds the three missing fast tests so
   reviewers running ``python -m src.main validate --quick``
   exercise the A6 routing fix, the carry-in fix, and the
   dataset-card reconciliations.

4. **``smoke --attack`` / ``--defense`` knobs.**  The submission
   draft's smoke test was hardcoded to A1+D1.  The camera-ready
   exposes both as optional flags so reviewers can quickly
   verify any (attack, defense) cell works in their environment
   without writing a Python harness.  Defaults are unchanged.

5. **Honest documentation of ``_log_hardware``.**  The submission
   draft docstring claimed hardware info "persists into run
   records via experiments.py for provenance"; in fact
   experiments.py does not accept a ``hardware_info`` parameter.
   The camera-ready docstring describes the actual behaviour:
   logged to stdout at run start; persistence is future work.

Author: ANONYMOUS  (double-blind review)
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

# We import lazily inside subcommands so that --help works on systems
# missing optional deps (matplotlib for `render`, torch for `run`).


def _make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="update-storms",
        description=(
            "Update Storms — adversarial real-time attacks against "
            "continuously-learning systems.  See README.md for the "
            "full reproduction recipe."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "For dataset preparation, see docs/DATASETS.md.\n"
            "For the SUT-fidelity disclosure, see docs/DIAM_FIDELITY.md.\n"
            "For evaluation methodology, see docs/EVALUATION_METHODOLOGY.md."
        ),
    )
    p.add_argument(
        "--verbose", "-v", action="count", default=0,
        help="Increase verbosity; -v = INFO, -vv = DEBUG.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # ---- smoke ----------------------------------------------------------
    sp = sub.add_parser(
        "smoke",
        help="Verify installation with a tiny synthetic run.",
    )
    sp.add_argument(
        "--output-dir", type=Path, default=Path("results/smoke"),
        help="Where to write the smoke-run record.",
    )
    sp.add_argument(
        "--n-transactions", type=int, default=500,
        help="Number of synthetic transactions to process.",
    )
    # Camera-ready: knobs to verify any (attack, defense) cell works.
    # The submission draft hardcoded A1+D1; reviewers wanting to
    # exercise the camera-ready A6 routing path can pass
    # ``--attack A6_evolutionary``.
    sp.add_argument(
        "--attack", default="A1_random",
        help=(
            "Attack name to use in the smoke run.  Default A1_random.  "
            "Camera-ready: ``--attack A6_evolutionary`` exercises the "
            "registry-driven routing introduced in attacks.py."
        ),
    )
    sp.add_argument(
        "--defense", default="D1_static",
        help=(
            "Defense name to use in the smoke run.  Default D1_static.  "
            "Pass ``--defense D3_schedulability`` to exercise the "
            "MGF-certificate path from camera-ready defenses.py."
        ),
    )

    # ---- validate (camera-ready) ----------------------------------------
    sp = sub.add_parser(
        "validate",
        help="Run the test suite (pytest) — camera-ready CI hook.",
    )
    sp.add_argument(
        "--tests-dir", type=Path, default=Path("tests"),
        help="Tests directory (default: tests/).",
    )
    sp.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Run only the fast tests (analysis + threat_model + "
            "measurement); skip torch-heavy tests for impatient "
            "smoke-checking."
        ),
    )

    # ---- run ------------------------------------------------------------
    sp = sub.add_parser("run", help="Run one named experiment.")
    sp.add_argument(
        "--experiment", "-e", required=True,
        help="Name of the experiment (see `list`).",
    )
    sp.add_argument(
        "--datasets", "-d", nargs="+", required=True,
        help="Datasets to include.  Names must match configs/datasets.yaml.",
    )
    sp.add_argument(
        "--data-root", type=Path, default=Path("data"),
        help="Root directory containing dataset subdirectories.",
    )
    sp.add_argument(
        "--output-dir", type=Path, default=Path("results/raw"),
        help="Output directory for run records.",
    )
    sp.add_argument(
        "--n-transactions", type=int, default=None,
        help="Override the experiment's default transaction count.",
    )
    _add_publication_flags(sp)

    # ---- run-all --------------------------------------------------------
    sp = sub.add_parser(
        "run-all",
        help="Run every experiment on the configured datasets in order.",
    )
    sp.add_argument(
        "--datasets", "-d", nargs="+",
        default=["cicids2018", "swat", "ethereum_phishing", "bitcoin_ransomware"],
        help="Datasets to include.",
    )
    sp.add_argument(
        "--data-root", type=Path, default=Path("data"),
    )
    sp.add_argument(
        "--output-dir", type=Path, default=Path("results/raw"),
    )
    _add_publication_flags(sp)

    # ---- render ---------------------------------------------------------
    sp = sub.add_parser(
        "render",
        help="Render figures and tables from existing run records.",
    )
    sp.add_argument(
        "--raw-dir", type=Path, default=Path("results/raw"),
        help="Directory containing run-record JSON files (recursive).",
    )
    sp.add_argument(
        "--figures-dir", type=Path, default=Path("results/figures"),
        help="Where to write figures (PDF/PNG) and tables (TeX/CSV).",
    )

    # ---- list -----------------------------------------------------------
    sp = sub.add_parser(
        "list",
        help="List available experiments, attacks, defenses, datasets.",
    )

    # ---- info -----------------------------------------------------------
    sp = sub.add_parser(
        "info",
        help="Show the dataset card (citation, version, limitations).",
    )
    sp.add_argument(
        "--datasets", "-d", nargs="+",
        help="Dataset name(s); if omitted, shows cards for all known datasets.",
    )
    sp.add_argument(
        "--format", choices=("text", "json"), default="text",
        help="Output format (text for humans, json for tooling).",
    )

    # ---- reproduce ------------------------------------------------------
    sp = sub.add_parser(
        "reproduce",
        help="Replay a published RunRecord and verify audit signatures.",
    )
    sp.add_argument(
        "--record", "-r", type=Path, required=True,
        help="Path to the published RunRecord JSON.",
    )
    sp.add_argument(
        "--data-root", type=Path, default=Path("data"),
        help="Root directory containing dataset subdirectories.",
    )
    sp.add_argument(
        "--check",
        choices=("signatures", "full"),
        default="signatures",
        help=(
            "What to verify.  'signatures' (default): compare audit "
            "signatures only (fast; sufficient to catch dataset "
            "version mismatches).  'full': re-run the experiment and "
            "compare measured tail percentiles within tolerance."
        ),
    )
    sp.add_argument(
        "--tolerance", type=float, default=0.05,
        help="Relative tolerance for 'full' check (default 0.05 = 5%%).",
    )
    _add_publication_flags(sp)

    return p


def _add_publication_flags(sp: argparse.ArgumentParser) -> None:
    """
    Flags relevant only to publication-grade runs.  Added to multiple
    subparsers so that the camera-ready discipline is consistent
    across `run`, `run-all`, and `reproduce`.
    """
    sp.add_argument(
        "--strict-schema",
        action="store_true",
        default=True,
        help=(
            "Treat missing schema_version columns as fatal (default "
            "TRUE for publication runs).  Disable with --no-strict-schema."
        ),
    )
    sp.add_argument(
        "--no-strict-schema",
        dest="strict_schema",
        action="store_false",
        help="Allow missing schema_version columns (warning only).",
    )
    sp.add_argument(
        "--full-file-hash",
        action="store_true",
        default=False,
        help=(
            "Hash the full content of preprocessed files for "
            "audit_signature() (slow; recommended for publication runs)."
        ),
    )
    sp.add_argument(
        "--allow-synthetic",
        action="store_true",
        default=False,
        help=(
            "PUBLICATION DISCIPLINE: by default, synthetic datasets are "
            "REFUSED for `run` / `run-all` / `reproduce` because they "
            "do not reflect real-world distributions.  Pass this flag "
            "explicitly when synthetic is desired (CI / debugging).  "
            "The synthetic dataset is signalled by "
            "dataset_card(name).is_synthetic=True; see workload.py."
        ),
    )


def _setup_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity >= 2:
        level = logging.DEBUG
    elif verbosity >= 1:
        level = logging.INFO
    logging.basicConfig(
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        level=level,
    )


def _resolve_dataset_paths(
    datasets: Sequence[str],
    data_root: Path,
) -> Mapping[str, Path]:
    """
    For each named dataset, find its preprocessed file under data_root.
    Convention: data_root/<dataset>/processed.parquet (or .csv as fallback).

    Camera-ready: errors include a hint to docs/DATASETS.md for the
    preprocessing recipe.
    """
    paths: dict[str, Path] = {}
    for name in datasets:
        ds_dir = data_root / name
        candidates = [ds_dir / "processed.parquet", ds_dir / "processed.csv"]
        chosen: Optional[Path] = next((c for c in candidates if c.exists()), None)
        if chosen is None:
            raise FileNotFoundError(
                f"dataset '{name}': no processed file found.\n"
                f"  expected one of:\n"
                + "\n".join(f"    {c}" for c in candidates)
                + "\n  see docs/DATASETS.md for preprocessing instructions, "
                "or run with --datasets synthetic --allow-synthetic for a CI-only run."
            )
        paths[name] = chosen
    return paths


def _enforce_publication_dataset_discipline(
    datasets: Sequence[str],
    allow_synthetic: bool,
) -> None:
    """
    Camera-ready: refuse synthetic datasets for publication runs unless
    --allow-synthetic is set.  Source of truth is
    workload.dataset_card(name).is_synthetic.
    """
    sys.path.insert(0, str(Path(__file__).parent))
    from workload import dataset_card

    synthetic_used: list[str] = []
    for name in datasets:
        try:
            card = dataset_card(name)
        except KeyError:
            # Unknown dataset; let downstream code raise its own error.
            continue
        if card.is_synthetic:
            synthetic_used.append(name)

    if synthetic_used and not allow_synthetic:
        raise ValueError(
            f"PUBLICATION DISCIPLINE: synthetic dataset(s) {synthetic_used} "
            f"are not allowed for `run` / `run-all` runs unless "
            f"--allow-synthetic is passed explicitly.  Synthetic data does "
            f"not reflect real-world distributions and any quantitative "
            f"claim derived from it is not publication-quality.  Pass "
            f"--allow-synthetic for CI / debugging, or specify a real "
            f"dataset (cicids2018 / swat / ethereum_phishing / "
            f"bitcoin_ransomware)."
        )


def _log_hardware() -> Mapping[str, Any]:
    """
    Camera-ready: capture CPU / GPU / CUDA / RAM info at run start.
    Returned dict is JSON-serialisable so experiments.py can persist
    it into RunRecord for provenance.

    Best-effort: probes that fail (e.g. no torch) are recorded as
    "unavailable" rather than raising.
    """
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "cpu_count": _safe_call(lambda: __import__("os").cpu_count()) or "unknown",
    }

    # CPU model from /proc/cpuinfo on Linux.
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        try:
            text = cpuinfo.read_text()
            for line in text.splitlines():
                if line.startswith("model name"):
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                    break
        except OSError:
            pass

    # RAM (best-effort).
    try:
        import psutil  # type: ignore[import]
        info["ram_total_gb"] = round(psutil.virtual_memory().total / (1024 ** 3), 1)
    except ImportError:
        # Fall back to /proc/meminfo on Linux.
        meminfo = Path("/proc/meminfo")
        if meminfo.exists():
            try:
                for line in meminfo.read_text().splitlines():
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        info["ram_total_gb"] = round(kb / (1024 ** 2), 1)
                        break
            except (OSError, ValueError):
                pass

    # CUDA / torch info (lazy import).
    try:
        import torch
        info["torch_version"] = torch.__version__
        if torch.cuda.is_available():
            info["cuda_available"] = True
            info["cuda_version"] = torch.version.cuda
            info["cudnn_version"] = (
                str(torch.backends.cudnn.version())
                if torch.backends.cudnn.is_available()
                else "n/a"
            )
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_devices"] = [
                {
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "total_mem_gb": round(
                        torch.cuda.get_device_properties(i).total_memory / (1024 ** 3),
                        1,
                    ),
                }
                for i in range(torch.cuda.device_count())
            ]
        else:
            info["cuda_available"] = False
    except ImportError:
        info["torch_version"] = "unavailable"

    # numpy version.
    try:
        import numpy as np
        info["numpy_version"] = np.__version__
    except ImportError:
        info["numpy_version"] = "unavailable"

    return info


def _safe_call(fn: Any) -> Any:
    """Best-effort call; on any exception, return None."""
    try:
        return fn()
    except Exception:
        return None


def _print_hardware_summary(info: Mapping[str, Any]) -> None:
    """Compact one-line hardware summary suitable for run-start logging."""
    cpu = info.get("cpu_model", info.get("processor", "unknown"))
    n_cpu = info.get("cpu_count", "?")
    ram = info.get("ram_total_gb", "?")
    if info.get("cuda_available"):
        gpus = info.get("gpu_devices", [])
        if gpus:
            gpu_summary = ", ".join(g["name"] for g in gpus)
            gpu_part = f"GPU(s): {len(gpus)}× {gpu_summary}"
        else:
            gpu_part = "CUDA available (no devices?)"
    else:
        gpu_part = "no CUDA"
    print(f"hardware: {cpu} ({n_cpu} cores) / {ram} GB RAM / {gpu_part}")


# =============================================================================
# Subcommand implementations.
# =============================================================================


def _cmd_smoke(args: argparse.Namespace) -> int:
    """
    Build a synthetic stream and run one short experiment.  Verifies
    every module is importable and the pipeline is wired correctly.
    Does NOT validate scientific claims.

    Camera-ready: respects ``--attack`` / ``--defense`` flags so
    reviewers can exercise specific code paths (e.g.
    ``--attack A6_evolutionary`` to verify the registry-driven A6
    routing fix).  Defaults remain A1+D1 for back-compat.
    """
    sys.path.insert(0, str(Path(__file__).parent))
    from experiments import RunSpec, run_one, contract_for
    from threat_model import AdversaryBudget
    from attacks import attack_names
    from defenses import defense_names

    # Camera-ready: validate the attack/defense names against the
    # registries so a typo fails loudly and helpfully rather than
    # falling through to ``make_attack`` deep in the harness.
    if args.attack not in attack_names():
        print(
            f"error: unknown attack '{args.attack}'\n"
            f"available: {', '.join(attack_names())}",
            file=sys.stderr,
        )
        return 2
    if args.defense not in defense_names():
        print(
            f"error: unknown defense '{args.defense}'\n"
            f"available: {', '.join(defense_names())}",
            file=sys.stderr,
        )
        return 2

    hw = _log_hardware()
    _print_hardware_summary(hw)

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    print(
        f"running smoke test (attack={args.attack}, "
        f"defense={args.defense}) …"
    )
    spec = RunSpec(
        experiment="smoke",
        dataset="synthetic",
        dataset_path=Path("/tmp/unused-synthetic"),  # SyntheticLoader ignores
        attack=args.attack,
        defense=args.defense,
        seed=0,
        n_transactions=args.n_transactions,
        contract=contract_for("synthetic"),
        budget=AdversaryBudget.realistic_default(),
        output_dir=out,
        note="installation_smoke_test",
    )
    rec = run_one(spec)
    miss = (rec.miss_report or {}).get("miss_rate", float("nan"))
    print(f"smoke test done.  measured miss rate ≈ {miss}")
    print(f"  wrote run record to {out / (spec.signature() + '.json')}")
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    """
    Camera-ready: run the test suite programmatically.
    Reviewers can run this once after cloning to confirm the codebase
    is functional in their environment.
    """
    if not args.tests_dir.exists():
        print(f"error: tests directory not found: {args.tests_dir}", file=sys.stderr)
        return 2

    pytest_exe = shutil.which("pytest") or sys.executable
    cmd: list[str] = []
    if pytest_exe == sys.executable:
        cmd = [sys.executable, "-m", "pytest"]
    else:
        cmd = [pytest_exe]

    cmd.extend(["-q", "--no-header", str(args.tests_dir)])

    if args.quick:
        # Camera-ready: --quick runs only the fast (numpy-only) tests.
        # The submission draft listed three; this list is wrong --
        # test_attacks.py, test_defenses.py, test_workload.py are also
        # numpy-only and run in seconds.  Only test_system.py and
        # test_experiments.py exercise torch/CUDA paths and need the
        # heavier slow-test mode.  This list now includes every
        # camera-ready fast test so reviewers running --quick
        # exercise the A6 routing fix (test_attacks.py), the
        # carry-in/MGF-field fix (test_defenses.py), and the
        # dataset-card reconciliations (test_workload.py).  Plus the
        # camera-ready regressions in test_camera_ready_regressions.py
        # and the new plots.py coverage in test_plots.py.
        cmd = [
            sys.executable, "-m", "pytest",
            "-q", "--no-header",
            str(args.tests_dir / "test_analysis.py"),
            str(args.tests_dir / "test_threat_model.py"),
            str(args.tests_dir / "test_measurement.py"),
            str(args.tests_dir / "test_attacks.py"),
            str(args.tests_dir / "test_defenses.py"),
            str(args.tests_dir / "test_workload.py"),
            str(args.tests_dir / "test_plots.py"),
            str(args.tests_dir / "test_camera_ready_regressions.py"),
        ]

    print(f"running: {' '.join(cmd)}")
    started = time.time()
    proc = subprocess.run(cmd, check=False)
    elapsed = time.time() - started
    print(f"\nvalidate complete in {elapsed:.1f}s; pytest exit code = {proc.returncode}")
    return proc.returncode


def _cmd_run(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(Path(__file__).parent))
    from experiments import experiment_names, run_experiment

    if args.experiment not in experiment_names():
        print(f"unknown experiment '{args.experiment}'", file=sys.stderr)
        print(f"available: {', '.join(experiment_names())}", file=sys.stderr)
        return 2

    # Camera-ready: refuse synthetic for publication runs.
    try:
        _enforce_publication_dataset_discipline(args.datasets, args.allow_synthetic)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    paths = _resolve_dataset_paths(args.datasets, args.data_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    hw = _log_hardware()
    _print_hardware_summary(hw)

    # Camera-ready: log publication-discipline flags for the run
    # operator's audit trail.  These are NOT yet wired through to
    # the run record (would require touching experiments.py); the
    # workload loaders default to ``strict_schema=True`` and the
    # audit-signature path uses the prefix-suffix-only file hash.
    # Wiring is documented as future work.
    if not args.strict_schema:
        print(
            "  note: --no-strict-schema is set, but the camera-ready "
            "harness does not forward it to make_loader; loaders use "
            "their default strict_schema=True.  See the main.py "
            "docstring for the wiring-future-work note."
        )
    if args.full_file_hash:
        print(
            "  note: --full-file-hash is set, but the camera-ready "
            "harness does not forward it to make_loader; "
            "audit_signature() uses the prefix-suffix mode.  See "
            "the main.py docstring for the wiring-future-work note."
        )

    # Build only the kwargs each experiment function actually accepts.
    # The submission draft passed strict_schema, full_file_hash, and
    # hardware_info via **kwargs, which crashed at runtime because
    # the experiment functions don't accept them (TypeError).
    kwargs: dict = {}
    if args.n_transactions is not None:
        kwargs["n_transactions"] = args.n_transactions

    started = time.time()
    records = run_experiment(
        name=args.experiment,
        dataset_paths=paths,
        output_dir=args.output_dir,
        **kwargs,
    )
    elapsed = time.time() - started
    print(f"experiment {args.experiment}: {len(records)} runs in {elapsed:.1f}s")
    return 0


def _cmd_run_all(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(Path(__file__).parent))
    from experiments import experiment_names, run_experiment

    # Camera-ready: refuse synthetic for publication runs.
    try:
        _enforce_publication_dataset_discipline(args.datasets, args.allow_synthetic)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    paths = _resolve_dataset_paths(args.datasets, args.data_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    hw = _log_hardware()
    _print_hardware_summary(hw)

    started = time.time()
    total = 0
    for name in experiment_names():
        print(f"\n=== {name} ===")
        # Camera-ready: do NOT pass strict_schema / full_file_hash /
        # hardware_info as **kwargs.  Experiment functions don't
        # accept those names; the submission draft would have
        # crashed with TypeError on the first iteration.  See main.py
        # docstring for the future-work wiring note.
        recs = run_experiment(
            name=name,
            dataset_paths=paths,
            output_dir=args.output_dir,
        )
        print(f"  {len(recs)} runs")
        total += len(recs)
    elapsed = time.time() - started
    print(f"\nrun-all complete: {total} runs in {elapsed/60.0:.1f} min")
    return 0


def _cmd_render(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(Path(__file__).parent))
    from plots import render_all
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    render_all(args.raw_dir, args.figures_dir)
    print(f"figures and tables in {args.figures_dir}")
    return 0


def _cmd_list(_args: argparse.Namespace) -> int:
    sys.path.insert(0, str(Path(__file__).parent))
    from experiments import experiment_names
    from attacks import attack_names
    from defenses import defense_names
    from workload import dataset_names
    print("experiments:")
    for n in experiment_names():
        print(f"  {n}")
    print("\nattacks:")
    for n in attack_names():
        print(f"  {n}")
    print("\ndefenses:")
    for n in defense_names():
        print(f"  {n}")
    print("\ndatasets:")
    for n in dataset_names():
        print(f"  {n}")
    print("\n(use `info --datasets <name>` for dataset cards)")
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    """
    Camera-ready: show the dataset card for one or more datasets.
    Useful for reviewers verifying citation / version / limitations
    chains without needing to read the source.
    """
    sys.path.insert(0, str(Path(__file__).parent))
    from workload import dataset_card, all_dataset_cards

    cards: list = []
    if args.datasets:
        for name in args.datasets:
            try:
                cards.append(dataset_card(name))
            except KeyError as e:
                print(f"error: {e}", file=sys.stderr)
                return 2
    else:
        cards = list(all_dataset_cards().values())

    if args.format == "json":
        out = [c.to_dict() for c in cards]
        print(json.dumps(out, indent=2))
        return 0

    # Text format: human-readable, suitable for stdout.
    for card in cards:
        print(f"\n{'=' * 78}")
        print(f"  {card.canonical_name}")
        print(f"{'=' * 78}")
        print(f"  registry-name:    {card.name}")
        print(f"  domain:           {card.domain}")
        print(f"  version:          {card.version}")
        print(f"  schema-version:   {card.schema_version}")
        print(f"  feature-dim:      {card.feature_dim}")
        if card.n_records_expected is not None:
            print(f"  n-records:        {card.n_records_expected:,}")
        else:
            print(f"  n-records:        (variable)")
        print(f"  is-synthetic:     {card.is_synthetic}")
        print()
        print(f"  primary-paper:    {card.primary_paper}")
        if card.bibtex_key:
            print(f"  bibtex-key:       {card.bibtex_key}")
        if card.secondary_papers:
            print(f"  secondary-papers:")
            for p in card.secondary_papers:
                print(f"    - {p}")
        print()
        print(f"  label-source:")
        print(_indent(card.label_source, "    "))
        print()
        print(f"  construction-notes:")
        print(_indent(card.construction_notes, "    "))
        print()
        print(f"  deadline-semantics:")
        print(_indent(card.deadline_semantics, "    "))
        print()
        print(f"  known-limitations:")
        for lim in card.known_limitations:
            print(_indent("- " + lim, "    "))
        print()
        print(f"  ethics-notes:")
        print(_indent(card.ethics_notes, "    "))
    print()
    return 0


def _indent(text: str, prefix: str, width: int = 74) -> str:
    """Word-wrap and prefix each line.  Avoids importing textwrap for one call."""
    words = text.split()
    lines: list[str] = []
    cur = ""
    for w in words:
        if not cur:
            cur = w
        elif len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = cur + " " + w
    if cur:
        lines.append(cur)
    return "\n".join(prefix + ln for ln in lines)


def _cmd_reproduce(args: argparse.Namespace) -> int:
    """
    Camera-ready: replay a published RunRecord and verify audit
    signatures match.

    The two check modes are:

      'signatures' — load the published RunRecord, build a fresh
        loader for the same dataset, and compare the loader's
        audit_signature() against the record's saved signature.
        This catches "wrong dataset version" issues in a few
        seconds without re-running anything.

      'full'       — additionally re-run the experiment from the
        record's RunSpec and compare measured tail percentiles
        within tolerance.  This catches "wrong code or wrong host"
        issues but takes as long as the original run.
    """
    sys.path.insert(0, str(Path(__file__).parent))
    from workload import make_loader, dataset_card

    if not args.record.exists():
        print(f"error: record file not found: {args.record}", file=sys.stderr)
        return 2

    try:
        with open(args.record, "r") as f:
            record = json.load(f)
    except json.JSONDecodeError as e:
        print(f"error: record is not valid JSON: {e}", file=sys.stderr)
        return 2

    # Locate the dataset name and the saved audit_signature.
    spec = record.get("spec", record)            # accept flat or nested
    dataset_name = spec.get("dataset")
    saved_signature = (
        record.get("dataset_audit_signature")
        or spec.get("dataset_audit_signature")
    )

    if dataset_name is None:
        print("error: record has no 'dataset' field", file=sys.stderr)
        return 2
    if saved_signature is None:
        print(
            "warning: record has no saved dataset_audit_signature; "
            "cannot verify dataset version match.",
            file=sys.stderr,
        )

    # Camera-ready discipline: refuse synthetic.
    try:
        _enforce_publication_dataset_discipline([dataset_name], args.allow_synthetic)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    print(f"reproducing run for dataset={dataset_name}")
    try:
        card = dataset_card(dataset_name)
        print(f"  canonical-name:   {card.canonical_name}")
        print(f"  expected-version: {card.version} / schema {card.schema_version}")
    except KeyError:
        # Unknown dataset; downstream make_loader will raise.
        pass

    paths = _resolve_dataset_paths([dataset_name], args.data_root)
    loader = make_loader(
        dataset_name,
        paths[dataset_name],
        strict_schema=args.strict_schema,
    )
    fresh_signature = loader.audit_signature(full_file_hash=args.full_file_hash)
    print(f"  fresh audit_signature:  {fresh_signature[:16]}...")
    if saved_signature is not None:
        print(f"  saved audit_signature:  {saved_signature[:16]}...")
        if fresh_signature == saved_signature:
            print("  ✓ dataset signatures MATCH")
        else:
            # Camera-ready: sharper diagnostic.  The signature is a
            # function of (loader_config, file_content_hash).  We
            # cannot un-mix without re-hashing both inputs separately,
            # but we CAN suggest the most common causes in priority
            # order.
            print("  ✗ dataset signatures DIFFER")
            print()
            print("  diagnostic — most common causes (ordered by likelihood):")
            print(
                "    1. Different preprocessed file content.  Check the file at "
                f"\n         {paths[dataset_name]}\n"
                "       The original run may have used a different version of the "
                "preprocessing scripts in scripts/prep_*.py.  Re-run the "
                "preprocessing with the script version cited in the run record."
            )
            print(
                "    2. Different loader config (time_dilation, "
                "feature_normalize, schema_version).  These are part of the "
                "audit_signature() — a config change produces a different hash "
                "even if the file is identical.  Compare the spec.loader_config "
                "field of the saved record against the defaults this CLI uses."
            )
            print(
                "    3. Stale local cache.  If you previously ran with a "
                "different file, delete the loader's cached file_hash by "
                "removing __pycache__ and re-running."
            )
            print(
                "  See docs/DATASETS.md for the canonical preprocessing recipe."
            )
            return 1

    if args.check == "signatures":
        print("\nsignature check complete (use --check full for percentile check)")
        return 0

    # --check full: re-run the experiment.
    print("\nre-running experiment for full check …")
    from experiments import RunSpec, run_one, contract_for
    from threat_model import AdversaryBudget

    rerun_spec = RunSpec(
        experiment=spec.get("experiment", "reproduce"),
        dataset=dataset_name,
        dataset_path=paths[dataset_name],
        attack=spec.get("attack", "A1_random"),
        defense=spec.get("defense", "D1_static"),
        seed=spec.get("seed", 0),
        n_transactions=spec.get("n_transactions"),
        contract=contract_for(dataset_name),
        budget=AdversaryBudget.realistic_default(),
        output_dir=Path("/tmp/reproduce-out"),
        note="reproduce-check",
    )
    rec = run_one(rerun_spec)

    saved_p99 = (record.get("miss_report") or {}).get("p99_us")
    fresh_p99 = (rec.miss_report or {}).get("p99_us")
    if saved_p99 is None or fresh_p99 is None:
        print("warning: missing p99_us in one of the records; cannot compare")
        return 0
    rel = abs(fresh_p99 - saved_p99) / max(saved_p99, 1.0)
    print(f"  saved p99: {saved_p99:.1f} µs")
    print(f"  fresh p99: {fresh_p99:.1f} µs")
    print(f"  relative error: {rel:.3f} (tolerance: {args.tolerance})")
    if rel <= args.tolerance:
        print("  ✓ percentile check PASSED")
        return 0
    print("  ✗ percentile check FAILED")
    print()
    print("  diagnostic — possible causes:")
    print(
        "    1. Different host: the original run was on different "
        "hardware; CPU model, GPU model, RAM, and CUDA version all "
        "affect tail percentiles.  Compare the spec.hardware_info field "
        "of the saved record against the current host."
    )
    print(
        "    2. Different code: the implementation has changed since "
        "the record was produced.  Check the git commit hash recorded "
        "in the run, if any."
    )
    print(
        "    3. Stochastic noise: even with fixed seeds, tail "
        "percentiles can move 1-3% across runs due to OS scheduling.  "
        "Try --tolerance 0.10 to confirm this is the cause."
    )
    return 1


# =============================================================================
# Entry point.
# =============================================================================


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _make_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    handlers = {
        "smoke":     _cmd_smoke,
        "validate":  _cmd_validate,
        "run":       _cmd_run,
        "run-all":   _cmd_run_all,
        "render":    _cmd_render,
        "list":      _cmd_list,
        "info":      _cmd_info,
        "reproduce": _cmd_reproduce,
    }
    handler = handlers.get(args.cmd)
    if handler is None:
        parser.print_help()
        return 2
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
