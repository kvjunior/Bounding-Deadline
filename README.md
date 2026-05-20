# Bounding Deadline Misses under Adversarial Update Storms

**A Schedulability-Aware Defense for Continuously-Learning Systems**

![Status](https://img.shields.io/badge/RTSS%202026-under%20review-blue)
![License](https://img.shields.io/badge/license-BSD--3--Clause-green)
![Python](https://img.shields.io/badge/python-%E2%89%A53.10-blue)
![CUDA](https://img.shields.io/badge/CUDA-12.2-76b900)

Reproducibility artifact for the RTSS 2026 submission *"Bounding Deadline
Misses under Adversarial Update Storms: A Schedulability-Aware Defense for
Continuously-Learning Systems."*  This repository contains the
figure-generation code, the schedulability-analysis routines, the attack
and defense implementations, the experiment-driver scripts, and the
LaTeX source needed to reproduce every figure, table, and theorem
statement in the paper.

> **Double-anonymous submission.**  This artifact is distributed under
> RTSS 2026 anonymous-submission rules.  Author identities, institutional
> affiliations, and the GitHub URL of any non-anonymised mirror have been
> redacted.  The citation block and the contact channel at the bottom of
> this README will be populated after acceptance.

---

## Headline results

- **Probabilistic schedulability bound** for the update-path latency tail
  under adversary-controlled arrivals (Theorem IV.3, MGF random-sum bound).
- **Dual-certificate defense D3** that honours a $\varepsilon = 10^{-3}$
  deadline-miss-probability contract on **four datasets spanning four
  orders of magnitude in deadline** at $\alpha = 5\%$ adversary injection.
- **$1.8\times$ tighter** than the Hoeffding envelope alone (Theorem IV.1).
- **An order of magnitude** below the strongest production-systems
  baseline (D2, adaptive-$p_{99}$ throttling) on deadline-miss rate.
- **First contract breach** at $\alpha = 7.5\%$; D3 retains an
  order-of-magnitude miss-rate advantage even out of contract.
- **Detection-quality cost**: less than $0.005$ F1 across all four
  datasets, between $3\times$ and $7\times$ smaller than the
  streaming/snapshot drift attributable to online operation.

---

## Repository layout

```
.
├── README.md                          # this file
├── LICENSE                            # BSD 3-Clause
├── requirements.txt                   # pinned Python dependencies
├── docker/
│   └── Dockerfile                     # reproducible build environment
├── paper/
│   ├── main.tex                       # full LaTeX source (IEEEtran, 11 pp)
│   ├── references.tex                 # bibliography (68 entries)
│   └── figs/                          # generated figures (.pdf and .png)
├── code/
│   ├── make_figs.py                   # generates every figure in the paper
│   ├── analysis/
│   │   ├── envelope_bound.py          # Theorem IV.1 (Hoeffding envelope)
│   │   ├── mgf_bound.py               # Theorem IV.3 (MGF random-sum bound)
│   │   ├── certificate.py             # apply/validate identity (§IV-C)
│   │   └── statistics.py              # bootstrap, Wilson, paired t-tests
│   ├── defenses/
│   │   ├── d0_none.py                 # D0: no defense (baseline)
│   │   ├── d1_static.py               # D1: static rate cap
│   │   ├── d2_quantile.py             # D2: adaptive p99 quantile
│   │   └── d3_schedule.py             # D3: dual-certificate admission
│   ├── attacks/
│   │   ├── a1_random.py               # A1 (T1, random)
│   │   ├── a2_highdegree.py           # A2 (T1, public-topology)
│   │   ├── a3_branchingmax.py         # A3 (T1, headline attack)
│   │   ├── a4_gradnorm.py             # A4 (T2, architecture-aware)
│   │   ├── a5_adaptive.py             # A5 (T3, adaptive white-box)
│   │   └── a6_evolution.py            # A6 (T2, evolutionary)
│   ├── sut/
│   │   ├── pipeline.py                # 5-phase SUT (A1/A2/B1/B2/C1)
│   │   └── instrument.py              # latency + AoI instrumentation
│   └── experiments/
│       ├── smoke_test.py              # 15-minute end-to-end check
│       ├── tab2_benign.py             # Table II driver
│       ├── tab3_attacks.py            # Table III driver
│       ├── tab4_defenses.py           # Table IV driver
│       ├── tab5_f1.py                 # Table V driver
│       └── tab6_alpha_sweep.py        # Table VI driver (§VII)
└── data/
    └── README.md                      # dataset acquisition instructions
```

---

## Quick start

```bash
git clone <anonymised-url>
cd update-storms-rtss26
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Regenerate every figure from cached experiment outputs (~1 minute).
python code/make_figs.py
```

Figures land in `paper/figs/` as both PDF (for the camera-ready) and PNG
(for previewing).  With cached experiment outputs, full figure regeneration
takes under one minute.  Running the full experiment suite from scratch
takes $\approx\!46$ hours of wall-clock on a 4-GPU host (see §VI-A).

To compile the paper itself:

```bash
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

---

## Reproducing the paper

### Figures (fast: regenerates from cached experiment outputs)

| Figure                               | Script                                       | Runtime |
|--------------------------------------|----------------------------------------------|---------|
| Fig 1 — System and threat model      | hand-drawn (see `paper/figs/fig1_system_threat.png`) | n/a |
| Fig 2 — Per-attack CCDF              | `python code/make_figs.py --fig 2`           | < 5 s   |
| Fig 3 — Defense efficacy             | `python code/make_figs.py --fig 3`           | < 5 s   |
| Fig 4 — Bound validation (4 panels)  | `python code/make_figs.py --fig 4`           | < 5 s   |
| Fig 5 — Storm dynamics               | `python code/make_figs.py --fig 5`           | < 5 s   |
| Fig 6 — Scalability with $\|V\|$     | `python code/make_figs.py --fig 6`           | < 5 s   |
| Fig 7 — AoI freshness under A5       | `python code/make_figs.py --fig 7`           | < 5 s   |
| Fig 8 — $\alpha$-sensitivity sweep   | `python code/make_figs.py --fig 8`           | < 5 s   |

### Tables (slow: runs the full experiment suite from scratch)

| Table                                | Script                                       | Wall-clock |
|--------------------------------------|----------------------------------------------|------------|
| Table I — Datasets                   | (static; see `data/README.md`)               | n/a        |
| Table II — Benign latency            | `python code/experiments/tab2_benign.py`     | ≈ 2 h      |
| Table III — Attack effectiveness     | `python code/experiments/tab3_attacks.py`    | ≈ 8 h      |
| Table IV — Cross-dataset defense     | `python code/experiments/tab4_defenses.py`   | ≈ 20 h     |
| Table V — F1 detection-quality cost  | `python code/experiments/tab5_f1.py`         | ≈ 16 h     |
| Table VI — $\alpha$-sensitivity      | `python code/experiments/tab6_alpha_sweep.py`| included in Table III/IV runs |

Total wall-clock to reproduce every numerical claim in the paper from
scratch: $\approx\!46$ hours, matching the figure reported in §VI-A.

### Verifying the schedulability theorems

Theorems IV.1 (Hoeffding envelope) and IV.3 (MGF random-sum bound) are
implemented in `code/analysis/envelope_bound.py` and `code/analysis/
mgf_bound.py` respectively.  The apply/validate identity of §IV-C is
verified by the unit tests in `code/analysis/test_certificate.py`:

```bash
pytest code/analysis/test_certificate.py -v
```

The tests confirm that for every dataset in Table I, evaluating the
envelope and MGF certificates on the same input produces predictions that
satisfy $\mathcal{M}_t \le \mathcal{C}_t$ pointwise, as required by the
dominance property invoked in §V-B.

---

## Datasets

Four publicly-available datasets are used.  Each must be obtained from its
respective origin and placed under `data/`; the directory layout the
scripts expect is documented in `data/README.md`.

| Dataset                          | Records       | License           | Source                                                            |
|----------------------------------|---------------|-------------------|-------------------------------------------------------------------|
| CSE-CIC-IDS2018                  | 16,233,002    | Open (CIC)        | https://www.unb.ca/cic/datasets/ids-2018.html                     |
| SWaT                             |    946,722    | Research-use      | iTrust, SUTD — https://itrust.sutd.edu.sg/testbeds/                |
| Ethereum phishing                |  2,973,489    | Open              | Etherscan + Chen et al. 2021 (see `data/README.md` for the pipeline) |
| BitcoinHeist                     |  2,916,697    | Open (UCI)        | https://archive.ics.uci.edu/dataset/526/                          |

CSE-CIC-IDS2018 labels follow the corrections of Engelen et al. (SPW 2021);
the corrected label files are regenerable from the original CIC release
using `code/sut/preprocess_ids2018.py`.

---

## Hardware and dependencies

**Tested platform.**  Single host with 4× NVIDIA A100 (40 GB),
128-core x86_64 CPU, 1 TB RAM, Ubuntu 22.04 LTS, CUDA 12.2.

**Software requirements** (pinned in `requirements.txt`):

- Python $\ge 3.10$
- PyTorch $\ge 2.2$ with CUDA 12.2 support
- NumPy $\ge 1.26$, SciPy $\ge 1.11$, Matplotlib $\ge 3.8$
- pandas $\ge 2.1$, networkx $\ge 3.2$, scikit-learn $\ge 1.4$
- tqdm, pytest

A reduced single-GPU configuration reproduces all figures from cached
experiment outputs but not the full experiment suite from scratch.  Verify
the analysis routines, defenses, and attacks all run end-to-end with the
smoke test:

```bash
python code/experiments/smoke_test.py
```

This takes $\approx\!15$ minutes on a single GPU and runs every defense
against every attack on a 1% subsample of CICIDS-2018.  If it passes, the
full reproduction is expected to succeed.

A Docker image pinning the exact dependency versions used in the paper is
provided as `docker/Dockerfile`:

```bash
docker build -t update-storms:rtss26 docker/
docker run --gpus all -v $(pwd):/workspace update-storms:rtss26 \
    python /workspace/code/experiments/smoke_test.py
```

---

## Statistical protocol

Every experimental cell uses 5 random seeds.  The repository ships:

- **95% bootstrap confidence intervals** for continuous quantities,
  following Cohen (2013).
- **Wilson-score intervals** for deadline-miss rates whose counts are
  small relative to the sample (Wilson 1927).
- **Bonferroni-corrected paired $t$-tests** for pairwise defense
  comparisons (Dunn 1961).

All three are implemented in `code/analysis/statistics.py` and exercised
by the test suite (`pytest code/analysis/test_statistics.py`).  The
bootstrap routine uses $10^4$ resamples by default; the seed is
deterministically derived from the experiment cell.

---

## Limitations and threats to validity

The four threats to validity documented in §VIII-A are reproduced here so
that an artifact evaluator does not need to flip between the paper and
the repository:

- **Cost-model independence.**  Theorem IV.3 assumes per-class conditional
  independence given the affected-subgraph sketch.  Under extreme
  cross-class correlation the random-sum decomposition loosens; this is
  not currently mitigated in the implementation.
- **Heavy-tail sample size.**  Five seeds is tight for $p_{99.99}$
  inference.  Wilson intervals are reported on every miss rate.
- **Adversary synchronisation.**  A5 knows the defense but not the random
  seed of the running deployment.  An attacker who could synchronise to
  the seed could mount a stronger attack; we know of no production
  deployment that exposes its seed.
- **Distributed deployment.**  Single-host experiments only.  Cross-host
  queueing behaviour is empirically unstudied.

In addition, GPU memory $\ge 24$ GB per device is required to reproduce
§VII's A5 results at $\alpha \ge 10\%$ on CICIDS-2018.  Smaller GPUs OOM
because A5 holds the cost-predictor's gradient state in memory for the
duration of the $\Delta_{\mathrm{refit}}{=}1$ s window.

---

## License

Code is released under the **BSD 3-Clause License** (see `LICENSE`).
Dataset files are subject to the licenses of their respective origins as
summarised in the dataset table above.

The figure-generation code (`code/make_figs.py`) uses the Wong colour
palette (Nature Methods 2011) for colour-blind-safe figures.
```

---

## Contact

This repository is anonymised for double-blind review.  Questions during
the RTSS 2026 review period should be directed through the conference's
anonymous-author-correspondence channel; the artifact-evaluation chair
will forward.  After acceptance, an updated contact block will appear
here and in the camera-ready paper.

---

## Acknowledgments

Dataset acknowledgments follow the licenses of each origin; see the
paper's §VI-A.  The Wong colour palette is used under the terms of
Wong (Nature Methods 8:441, 2011).  Statistical routines use the
methodology of Cohen (2013), Wilson (1927), and Dunn (1961).
