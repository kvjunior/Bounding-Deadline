"""
workload.py — Dataset loading and adversarial stream generation.

This module turns disk-resident datasets into the canonical stream
form the rest of the codebase consumes:

    Iterable[ TimedTransaction ] = Iterable[ Tuple[float, Transaction, int] ]

where the float is the arrival timestamp (seconds, monotonic-equivalent),
the Transaction is from system.py, and the int is the ground-truth
label (0 benign, 1 illicit) when known and -1 otherwise.

Three responsibilities live here and nowhere else:

  1. Dataset adapters.  Four loaders, one per dataset.  They ingest a
     preprocessed file (documented in docs/DATASETS.md) and emit
     TimedTransactions in nondecreasing timestamp order.  We do not
     ship the datasets in this repo; the loaders open files from
     paths configured in configs/datasets.yaml.

  2. Stream mixing.  The MixedStream class interleaves benign and
     adversarial streams according to the AdversaryBudget from
     threat_model.py.  The mixing process is deterministic given a
     seed, replayable, and audited (the resulting stream's signature
     is hashed and recorded in run records so that two runs with the
     same hash provably saw the same workload).

  3. Dataset cards.  Each dataset has a `DatasetCard` describing its
     citation, version, feature dimensionality, label source,
     construction methodology, known limitations, and ethics notes.
     These cards are exposed via `dataset_card(name)` and persisted
     into run records so a reviewer can verify which version of which
     dataset produced which numbers.  Cards are the loader-side
     analogue of the deadline-framing notes in experiments.py: where
     experiments.py comments WHAT the deadline means, the card here
     describes WHAT the data are.

What this file deliberately does NOT do
---------------------------------------
- It does not push to the SUT.  It produces an iterator.  The caller
  in experiments.py pulls from it and decides timing.
- It does not run attacks.  Attacks (attacks.py) are independent of
  the workload; this file ASKS an Attack instance for candidates and
  interleaves them.
- It does not enforce defense admission.  The mixed stream feeds the
  defender; the defender decides what reaches the SUT.
- It does not parse raw blockchain or PCAP data.  Loaders assume a
  preprocessed columnar format described in docs/DATASETS.md.

The four datasets
-----------------
EthereumPhishing   (cryptocurrency)         — labels: Etherscan
BitcoinRansomware  (cryptocurrency)         — labels: BitcoinHeist
CICIDS-2018        (network intrusion)      — CSE-CIC-IDS2018, CIC ground truth
SWaT               (industrial control)     — iTrust ground truth

Each preprocessed file is a Parquet table with the columns:
  timestamp        float64  (seconds since epoch, sorted ascending)
  source           int64    (account or src endpoint id)
  target           int64    (account or dst endpoint id)
  amount           float64  (transaction value, normalised per dataset)
  features         list<float32>   (D-dim per-record feature vector)
  label            int8     (0 benign, 1 illicit, -1 unknown)
  dataset          string   (constant per file; redundant for safety)

The preprocessed file MAY include a `schema_version` column (a
constant int per file).  When present, loaders validate it against
the loader's `EXPECTED_SCHEMA_VERSION`.  When absent, loaders log a
warning and assume `schema_version=1` (the original schema).

Loaders fail fast if any required column is missing or has an
incompatible type; this is a contract with docs/DATASETS.md and the
preprocessing scripts.

Camera-ready improvements over the submission draft
---------------------------------------------------
1. **Dataset-card numbers reconciled with the §V.A LaTeX text.**
   The submission-draft cards had four mismatches against the paper
   text (``datasets_subsection_tex.txt``):

     a. CICIDS card said "8 attack days" → camera-ready says "10
        days, 7 attack families" (DoS, DDoS, Brute-Force, Bot, Web,
        Infiltration, Heartbleed), matching the official CIC
        description and the LaTeX paragraph.
     b. SWaT card said "Six attack scenarios" → camera-ready clarifies
        "41 documented attack instances grouped into 6 INTENT
        categories" (Adepu & Mathur 2016 intent-space taxonomy).
        The 41 vs. 6 distinction matters because the per-class F1
        appendix table reports support sizes that correspond to
        per-intent-category samples, not per-attack-instance samples.
     c. BitcoinHeist card said "28 ransomware families" → camera-
        ready says "24 ransomware families plus a benign class",
        matching the published BitcoinHeist artefact.
     d. SWaT feature_dim is 25 (a documented 25-dim subset of the
        51 sensors+actuators).  The camera-ready card now spells
        out the subset-selection rationale explicitly so a reviewer
        cross-checking against Goh 2017's "51 devices" sees the
        intentional reduction rather than an apparent error.

2. **Card-side known_limitations references the published Engelen
   et al. SPW 2021 corrections for CICIDS-2018** (not just for the
   2017 release).  Both releases share the CICFlowMeter feature
   extractor; the limitations transfer.

3. **TimedTransaction's ``is_adversarial`` flag now propagates
   ground truth.**  The flag was added in the submission draft but
   the documentation was implicit; the camera-ready spells out
   that this is the public ground-truth channel that camera-ready
   ``attacks.is_adversarial_txn_id`` mirrors and that camera-ready
   ``experiments._attach_storm_analysis`` consumes for non-circular
   α estimation.

4. **MixedStream fingerprint includes ``is_adversarial``.**
   Already true in the submission draft; the camera-ready
   docstring now documents this so reviewers can verify
   ground-truth-label fidelity by comparing fingerprints.

Author: ANONYMOUS  (double-blind review)
"""

from __future__ import annotations

import abc
import hashlib
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np

from threat_model import AdversaryBudget
from system import Transaction
from attacks import Attack, AttackCandidate

logger = logging.getLogger(__name__)


# =============================================================================
# Section 1.  TimedTransaction: the canonical stream element.
# =============================================================================


@dataclass(frozen=True, slots=True)
class TimedTransaction:
    """
    One element of a workload stream.

    `arrival_time` is in seconds, monotonic-equivalent (i.e. the
    consumer of the stream may compare arrival times directly to
    time.monotonic() to schedule processing).  The dataset loaders
    rebase the original timestamps to start at zero, then optionally
    apply a `time_dilation` factor (configured per experiment) to
    speed up or slow down replay.

    `is_adversarial` is set by the mixing layer and is True for
    transactions that came from an attack.  This is *experimental
    metadata*, not visible to the SUT or the defender — both treat
    every transaction identically.  It is recorded in run records so
    that experiments.py can compute attack-attribution metrics
    (rejection rate of adversarial vs benign, etc.) post-hoc.

    Camera-ready: this flag is also the public ground-truth channel
    used by ``experiments._attach_storm_analysis`` to estimate α
    non-circularly (replacing the submission draft's "top-α-by-cost"
    heuristic).  It mirrors ``attacks.Attack.is_adversarial_txn_id``,
    which classifies transactions by the attack-emitted txn_id range
    (large negative offsets); both channels agree by construction
    because ``MixedStream`` sets the flag to True iff the transaction
    came from ``attack._propose_one``.

    `label` is the ground-truth class label (0/1/-1).  Used only by
    the SUT's training signal, not by the defender.

    Note on `is_adversarial` vs `label`: these are independent.  An
    `is_adversarial=True` transaction is by convention also
    `label=1` (the attacker's injection is treated as illicit by
    construction), but a benign transaction with `label=1` (a
    real-world phishing transaction in the dataset) has
    `is_adversarial=False`.  The flag tracks "did this come from
    an attacker injection in this run", not "is this transaction
    illicit in ground truth".
    """

    arrival_time: float
    transaction: Transaction
    label: int
    is_adversarial: bool = False
    rationale: str = ""           # for adversarial: the attack's rationale string


# =============================================================================
# Section 2.  Dataset cards.
#
# Per Goldblum et al. ("Dataset Security for Machine Learning,"
# IEEE TPAMI 2023, the systematic survey of dataset failure modes in
# ML), every dataset used in published evaluation should be
# accompanied by a "card" specifying provenance, version, label
# source, and known limitations.  Cards make four things possible:
#
#   1. Reviewers can verify the citation chain (which paper
#      introduced the dataset, which papers corrected its issues).
#   2. Reproductions can use the same version: a run referring to
#      "CICIDS2018" without a version is potentially using one of
#      five preprocessed variants in circulation.
#   3. Limitations are surfaced at the loader, not buried in a
#      paper appendix that nobody reads.
#   4. Ethics review is straightforward: each card states the data's
#      consent / IRB status.
#
# Cards are immutable values constructed once at import time and
# returned by dataset_card(name).  The harness records the card
# alongside each RunRecord so that runs are self-describing.
# =============================================================================


@dataclass(frozen=True)
class DatasetCard:
    """
    Per-dataset metadata, citation, and limitations record.

    Fields
    ------
    name: registry key used by make_loader() and contract_for().
    canonical_name: the dataset's published name with version
        (e.g., "CSE-CIC-IDS2018", not "CICIDS2018"; the
        disambiguation the extraction reports flagged).
    domain: high-level category for grouping in plots.
    primary_paper: the paper that introduced this dataset (full
        citation string; bibtex_key gives the corresponding bib
        entry).
    bibtex_key: short citation key matching docs/refs.bib.
    secondary_papers: tuple of follow-up papers that surveyed,
        corrected, or extended the dataset.  Cited by the paper for
        evidence that limitations are publicly documented.
    version: dataset version identifier.  Where the upstream
        publishers do not version explicitly, we use a date-based
        proxy.
    schema_version: preprocessed-file schema version expected by
        this loader.  Bumped when columns or feature dims change.
    feature_dim: number of features per transaction record.
    n_records_expected: approximate size of the full dataset; None
        when the dataset is variable-sized or the figure is not
        publicly stable.
    label_source: short prose describing how labels were derived
        (e.g., "Etherscan phishing reports as of YYYY-MM").
    construction_notes: longer prose describing the construction
        pipeline.  Reviewers read this first to assess methodology.
    known_limitations: tuple of bullet-point limitation strings.
        Each should reference a published correction where one
        exists.
    ethics_notes: human-subjects / privacy posture.
    is_synthetic: True only for SyntheticLoader; experiments.py
        uses this to refuse synthetic data for publication-marked
        runs.
    deadline_semantics: explanation of what the contract deadline
        means for this dataset.  Pairs with the deadline_us values
        in experiments._DATASET_CONTRACTS so that the loader-side
        and contract-side framings are consistent.
    """

    name: str
    canonical_name: str
    domain: str
    primary_paper: str
    bibtex_key: str
    secondary_papers: Tuple[str, ...]
    version: str
    schema_version: int
    feature_dim: int
    n_records_expected: Optional[int]
    label_source: str
    construction_notes: str
    known_limitations: Tuple[str, ...]
    ethics_notes: str
    is_synthetic: bool
    deadline_semantics: str

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "name": self.name,
            "canonical_name": self.canonical_name,
            "domain": self.domain,
            "primary_paper": self.primary_paper,
            "bibtex_key": self.bibtex_key,
            "secondary_papers": list(self.secondary_papers),
            "version": self.version,
            "schema_version": self.schema_version,
            "feature_dim": self.feature_dim,
            "n_records_expected": self.n_records_expected,
            "label_source": self.label_source,
            "construction_notes": self.construction_notes,
            "known_limitations": list(self.known_limitations),
            "ethics_notes": self.ethics_notes,
            "is_synthetic": self.is_synthetic,
            "deadline_semantics": self.deadline_semantics,
        }


# Cards are constructed at import time and are immutable.  Editing
# any field below requires bumping schema_version on the affected
# loader and re-running the preprocessing scripts in scripts/prep_*.

_CARD_ETHEREUM_PHISHING = DatasetCard(
    name="ethereum_phishing",
    canonical_name="Ethereum Phishing Transactions (Etherscan-labelled)",
    domain="cryptocurrency",
    primary_paper=(
        "Chen et al., 'Phishing Scams Detection in Ethereum "
        "Transaction Network,' ACM TOIT 21(1), 2021."
    ),
    bibtex_key="Chen2021TOIT",
    secondary_papers=(
        "Wu et al., 'Who Are the Phishers? Phishing Scam Detection "
        "on Ethereum via Network Embedding,' IEEE TSMC 52(2), 2022.",
        "Lin et al., 'T-EDGE: Temporal Weighted MultiDiGraph "
        "Embedding for Ethereum Transaction Network Analysis,' "
        "Frontiers in Physics 8, 2020.",
    ),
    version="2021-09",
    schema_version=2,
    feature_dim=8,
    n_records_expected=2_973_489,
    label_source=(
        "Etherscan-flagged phishing addresses as of the cutoff date "
        "in the primary paper.  Positives are addresses that "
        "appeared on Etherscan's published phishing list at any "
        "time prior to the cutoff; negatives are randomly sampled "
        "non-flagged addresses."
    ),
    construction_notes=(
        "Per-account graph of Ethereum transfers; features are "
        "derived from per-transaction value, gas, gas-price, "
        "to-contract flag, plus four account-level statistics "
        "(in/out degree, in/out value).  All features are computed "
        "per-record without leaking future state."
    ),
    known_limitations=(
        "Look-back window: an address may have been benign for "
        "years before being reported as phishing; the label "
        "reflects retrospective ground truth and is therefore "
        "biased toward addresses with public exposure.",
        "Etherscan reports rely on community submissions; some "
        "true phishing addresses are missing from the positive "
        "class.",
        "Class imbalance is severe (~0.05% positives); aggregate "
        "miss rates are dominated by negatives.",
    ),
    ethics_notes=(
        "Public blockchain data; addresses are pseudonymous by "
        "design.  No PII risk."
    ),
    is_synthetic=False,
    deadline_semantics=(
        "4 s deadline = post-flow update window for incremental "
        "model updates against block-inclusion latency on Ethereum "
        "mainnet (~12 s post-Merge), conservatively halved to "
        "leave headroom for propagation."
    ),
)


_CARD_BITCOIN_RANSOMWARE = DatasetCard(
    name="bitcoin_ransomware",
    canonical_name="BitcoinHeist (Akcora et al. IJCAI 2020)",
    domain="cryptocurrency",
    primary_paper=(
        "Akcora et al., 'BitcoinHeist: Topological Data Analysis "
        "for Ransomware Prediction on the Bitcoin Blockchain,' "
        "IJCAI 2020."
    ),
    bibtex_key="Akcora2020IJCAI",
    secondary_papers=(
        "Akcora et al., 'Forecasting Bitcoin Price with Graph "
        "Chainlets,' PAKDD 2018 (precursor topology-features "
        "methodology).",
        "Bartoletti et al., 'A Data Science Approach to Detect "
        "Ransomware Families,' Future Generation Computer Systems "
        "115, 2021 (cross-validates BitcoinHeist labels).",
    ),
    version="2020 release (IJCAI artefact)",
    schema_version=2,
    feature_dim=8,
    n_records_expected=2_916_697,
    label_source=(
        "Ransomware family labels from BitcoinHeist's authors, who "
        "joined chain-state with WalletExplorer ground truth and "
        "the open ransomware-tracking community.  The published "
        "BitcoinHeist artefact contains 24 ransomware families plus "
        "a benign class; the harness collapses these into a single "
        "'illicit' class for binary detection (per-family F1 is "
        "reported in the paper's appendix)."
    ),
    construction_notes=(
        "Per-address features encode UTXO-graph topology.  The "
        "original Akcora et al. artefact reports six topological "
        "features (income, neighbours, weight, count, looped, "
        "length); we extend with two account-level statistics to "
        "harmonise the feature schema with the Ethereum loader, "
        "giving an eight-dimensional record.  Computed "
        "deterministically from the Bitcoin chain state at the "
        "labelled block height.  See docs/BITCOINHEIST.md for the "
        "extension rationale and the per-feature provenance."
    ),
    known_limitations=(
        "Family bias: among the 24 ransomware families, a small "
        "number (e.g., Locky, CryptoLocker, Cerber) dominate the "
        "positive class; minority families have fewer than 100 "
        "labelled addresses.  Per-family miss rates may be very "
        "different from aggregate miss rates.",
        "The negative class is sampled from non-flagged addresses "
        "and does not include other types of illicit-but-non-"
        "ransomware activity (mixers, exchanges with KYC issues, "
        "etc.); a benign-looking address may be illicit in ways "
        "this dataset does not catch.",
        "Akcora et al. note that the 'ransomware addresses' label "
        "covers payment endpoints, not the on-chain laundering "
        "mechanics — downstream addresses in laundering chains "
        "are unlabelled.",
    ),
    ethics_notes=(
        "Public blockchain data; addresses are pseudonymous.  "
        "Akcora et al. collaborated with the open ransomware-"
        "tracking community for label provenance.  No PII risk."
    ),
    is_synthetic=False,
    deadline_semantics=(
        "10 s deadline = post-flow update window.  Generous given "
        "Bitcoin's ~10-minute block time, leaves headroom for "
        "feature extraction from the chain state."
    ),
)


_CARD_CICIDS2018 = DatasetCard(
    name="cicids2018",
    canonical_name="CSE-CIC-IDS2018 (NOT the CICIDS2017 of Sharafaldin et al. ICISSP 2018)",
    domain="network_intrusion",
    primary_paper=(
        "Sharafaldin et al., 'Toward Generating a New Intrusion "
        "Detection Dataset and Intrusion Traffic Characterization,' "
        "ICISSP 2018.  NOTE: the ICISSP paper evaluates the 2017 "
        "CICIDS dataset; the 2018 successor (CSE-CIC-IDS2018, used "
        "here) is published by the Communications Security "
        "Establishment / Canadian Institute for Cybersecurity in "
        "the same series and uses the same CICFlowMeter feature "
        "extractor."
    ),
    bibtex_key="Sharafaldin2018ICISSP",
    secondary_papers=(
        "Engelen et al., 'Troubleshooting an Intrusion Detection "
        "Dataset: The CICIDS2017 Case Study,' IEEE SPW 2021 "
        "(documents labelling errors and CICFlowMeter bugs in "
        "CICIDS2017; analogous concerns apply to CSE-CIC-IDS2018 "
        "since the same feature extractor is used).",
        "Lashkari et al., 'Characterization of Tor Traffic using "
        "Time-based Features,' ICISSP 2017 (CICFlowMeter origin).",
        "Leevy & Khoshgoftaar, 'A Survey and Analysis of "
        "Intrusion Detection Models Based on CSE-CIC-IDS2018 Big "
        "Data,' Journal of Big Data 7(1), 2020 (survey of "
        "downstream uses of CSE-CIC-IDS2018 specifically)."
    ),
    version="CSE-CIC-IDS2018 (10 days; 7 attack families)",
    schema_version=2,
    feature_dim=16,
    n_records_expected=16_233_002,
    label_source=(
        "Per-flow ground-truth labels supplied by the dataset "
        "publishers, mapping (src_ip, dst_ip, src_port, dst_port, "
        "protocol, time-window) tuples to attack class.  We "
        "collapse the multi-class attack labels into a binary "
        "benign/illicit label for the schedulability evaluation."
    ),
    construction_notes=(
        "Network capture from a 50-machine victim network and "
        "10-machine attacker network on AWS over ten days, "
        "labelled across seven attack families: DoS, DDoS, "
        "Brute-Force, Bot, Web, Infiltration, and Heartbleed.  "
        "Per-flow features extracted by CICFlowMeter (duration, "
        "packet counts, byte counts, IAT statistics, flag counts; "
        "we use 16 of the 80 published features, selected via a "
        "stability-and-informativeness filter on a held-out "
        "warm-up split).  IMPORTANT: CICFlowMeter features are "
        "computed AFTER a flow has been observed for some time or "
        "after termination; the 2 ms deadline therefore applies "
        "after a flow record is emitted to the SUT, not at first-"
        "packet ingress.  This is the post-flow update freshness "
        "interpretation noted in experiments.py."
    ),
    known_limitations=(
        "Engelen et al. document several flow-construction bugs "
        "in the original CICFlowMeter (TCP termination handling, "
        "duplicate flows, asymmetric flow direction).  Our "
        "preprocessor applies the patches recommended in their "
        "paper; runs without those patches will produce different "
        "tail behaviour (typically more outliers in the IAT "
        "features).",
        "The label-by-IP-tuple convention misclassifies "
        "intermittent attacks: an attacker host may also generate "
        "benign traffic during the attack window, but all of its "
        "flows during that window are labelled as attack.",
        "The deadline-credibility of network IDS at 2 ms requires "
        "post-flow-record interpretation; an inline blocking IDS "
        "is a different problem with different SLAs and is "
        "explicitly out of scope (see Background §II)."
    ),
    ethics_notes=(
        "Synthetic-yet-realistic capture from a controlled "
        "testbed; no real user PII captured.  Attack traffic was "
        "generated by the dataset authors against their own "
        "victim network."
    ),
    is_synthetic=False,
    deadline_semantics=(
        "2 ms deadline = POST-FLOW-RECORD update freshness "
        "deadline.  Once a flow record is emitted to the SUT, the "
        "incremental update must complete within 2 ms.  This is "
        "NOT a packet-forwarding or first-packet blocking "
        "deadline; the SUT is not an inline IDS.  See "
        "experiments.py _DATASET_CONTRACTS for the matching "
        "contract-side framing."
    ),
)


_CARD_SWAT = DatasetCard(
    name="swat",
    canonical_name="Secure Water Treatment (SWaT) — 41 attacks across 6 intent categories",
    domain="industrial_control",
    primary_paper=(
        "Goh et al., 'A Dataset to Support Research in the Design "
        "of Secure Water Treatment Systems,' International "
        "Conference on Critical Information Infrastructures "
        "Security (CRITIS) 2017."
    ),
    bibtex_key="Goh2017CRITIS",
    secondary_papers=(
        "Mathur & Tippenhauer, 'SWaT: A Water Treatment Testbed "
        "for Research and Training on ICS Security,' "
        "International Workshop on Cyber-Physical Systems for "
        "Smart Water Networks (CySWater) 2016 (testbed design).",
        "Adepu & Mathur, 'Distributed Detection of Single-Stage "
        "Multipoint Cyber Attacks in a Water Treatment Plant,' "
        "ASIA CCS 2016 (attack-scenario design).",
    ),
    version="A1/A2 dataset (2017 release; 41 attacks, 6 intent categories)",
    schema_version=2,
    feature_dim=25,
    n_records_expected=946_722,
    label_source=(
        "Ground-truth attack labels supplied by iTrust (the "
        "Singapore University of Technology and Design / SUTD "
        "research lab that operates the testbed).  The published "
        "dataset comprises 11 days of continuous operation: 7 days "
        "of normal behaviour followed by 4 days during which the "
        "iTrust team launched 41 documented cyber-physical attack "
        "instances on the testbed.  These 41 instances are grouped "
        "into 6 INTENT CATEGORIES per the Adepu & Mathur (ASIA CCS "
        "2016) intent-space taxonomy (the same model used in the "
        "appendix of Goh et al. 2017).  Transactions during attack "
        "windows are labelled illicit at 1 Hz sample granularity. "
        "Note for reviewers: the 41-vs-6 distinction matters because "
        "per-class F1 results in the appendix table report support "
        "sizes that correspond to per-intent-category samples, not "
        "to the number of distinct attack instances."
    ),
    construction_notes=(
        "The full SWaT testbed has 51 sensors and actuators "
        "distributed across six process stages (raw water intake, "
        "chemical dosing, ultrafiltration, dechlorination, reverse "
        "osmosis, and disposal/recycle), sampled at 1 Hz.  The "
        "feature_dim=25 used by this loader is a documented "
        "subset of the 51 signals: we retain only the analog "
        "continuous-valued sensors (level transmitters, flow "
        "indicators, pressure transmitters, analyzer readings) and "
        "exclude the binary actuator state and pump on/off "
        "indicators because (i) they vary on much slower timescales "
        "than the analog signals and add little tail-percentile "
        "information, and (ii) several actuator readings on the "
        "raw-water side are reported as constants in the publicly "
        "released A1/A2 file.  Reviewers wanting full-device "
        "fidelity can re-preprocess with the full 51-D feature "
        "vector; the schedulability bound's distribution-level form "
        "(Theorem 4.3) does not depend on this choice — it bounds "
        "Pr(latency > T) given an empirical cost distribution.  See "
        "docs/DATASETS.md for the per-channel inclusion list.  The "
        "physical-process semantics make labels unambiguous: an "
        "attack that manipulates the valve state during a labelled "
        "window is illicit even when the analog readings remain in "
        "normal range."
    ),
    known_limitations=(
        "Smaller than the other datasets (~946K records); "
        "statistical power for tail-percentile claims is the "
        "binding constraint here.",
        "Per-instance support is small for several of the 41 "
        "attack instances; we therefore report SWaT results with "
        "explicit confidence intervals (per the methodology in "
        "§V.A of the paper) and treat per-instance tail percentiles "
        "as illustrative rather than determinative.  The six "
        "intent-categories (Adepu & Mathur 2016) have larger "
        "support and are the unit at which per-class F1 is "
        "credibly comparable across runs.",
        "Generalisation beyond the six intent categories is not "
        "directly tested; the testbed runs at a different physical "
        "scale from production water treatment plants, so absolute "
        "thresholds in the feature distributions are testbed-"
        "specific.",
    ),
    ethics_notes=(
        "iTrust testbed data; no human subjects.  iTrust grants "
        "research access on application; we use the publicly "
        "released A1/A2 release."
    ),
    is_synthetic=False,
    deadline_semantics=(
        "10 ms deadline = control-loop bound.  Derived from the "
        "SWaT testbed's 10 Hz PLC scan rate; the SUT's update "
        "must complete within one control cycle to integrate the "
        "fresh sensor reading into the next decision."
    ),
)


_CARD_SYNTHETIC = DatasetCard(
    name="synthetic",
    canonical_name="Synthetic power-law transaction stream (CI test fixture)",
    domain="synthetic",
    primary_paper="(none — generated in this codebase)",
    bibtex_key="",
    secondary_papers=(),
    version="schema-2",
    schema_version=2,
    feature_dim=8,
    n_records_expected=10_000,
    label_source="Bernoulli draw with configurable illicit_fraction.",
    construction_notes=(
        "Account ids drawn with rank-1/r popularity, features are "
        "i.i.d. standard normal.  The graph topology is "
        "approximately scale-free.  Used by tests/ and CI; "
        "experiments.py refuses synthetic for publication-marked "
        "runs (the is_synthetic=True flag here is the source of "
        "truth for that check)."
    ),
    known_limitations=(
        "Synthetic — does NOT reflect real-world transaction "
        "patterns and MUST NOT be used for any quantitative "
        "claim in the paper.",
    ),
    ethics_notes="Synthetic data; no privacy concerns.",
    is_synthetic=True,
    deadline_semantics="5 ms — chosen for fast CI runs, no operational meaning.",
)


_DATASET_CARDS: Mapping[str, DatasetCard] = {
    _CARD_ETHEREUM_PHISHING.name: _CARD_ETHEREUM_PHISHING,
    _CARD_BITCOIN_RANSOMWARE.name: _CARD_BITCOIN_RANSOMWARE,
    _CARD_CICIDS2018.name: _CARD_CICIDS2018,
    _CARD_SWAT.name: _CARD_SWAT,
    _CARD_SYNTHETIC.name: _CARD_SYNTHETIC,
}


def dataset_card(name: str) -> DatasetCard:
    """
    Return the DatasetCard for the named dataset.

    Raises KeyError on unknown names.  The card is the source of
    truth for citation, version, limitations, and ethics; callers
    that need to display or persist these (plots.py, experiments.py)
    should always call this rather than maintaining their own
    parallel metadata.
    """
    if name not in _DATASET_CARDS:
        raise KeyError(
            f"unknown dataset '{name}'; known: {sorted(_DATASET_CARDS)}"
        )
    return _DATASET_CARDS[name]


def all_dataset_cards() -> Mapping[str, DatasetCard]:
    """Return a copy of the full card registry, for serialization."""
    return dict(_DATASET_CARDS)


# =============================================================================
# Section 3.  Dataset loader abstract base.
#
# Camera-ready additions:
# - audit_signature():  deterministic hash over loader configuration +
#   file content, persisted into RunRecord for reviewer-facing
#   provenance.
# - Schema version validation: warn-or-fail when the preprocessed
#   file's schema_version disagrees with what the loader expects.
# - Stricter type validation in _iter_parquet().
# =============================================================================


class DatasetLoader(abc.ABC):
    """
    One loader per dataset.  Implementations:
      - EthereumPhishingLoader
      - BitcoinRansomwareLoader
      - CICIDS2018Loader
      - SWaTLoader
      - SyntheticLoader (CI-only)
    """

    name: str = "abstract"
    feature_dim: int = 0
    expected_columns: Tuple[str, ...] = (
        "timestamp", "source", "target", "amount", "features", "label", "dataset",
    )
    optional_columns: Tuple[str, ...] = ("schema_version",)

    # Subclasses set this to the schema_version they expect.  When
    # the preprocessed file declares a schema_version, the loader
    # validates equality.  When it does not, the loader logs a
    # warning and proceeds.
    EXPECTED_SCHEMA_VERSION: int = 2

    def __init__(
        self,
        path: Path,
        time_dilation: float = 1.0,
        feature_normalize: bool = True,
        strict_schema: bool = True,
    ) -> None:
        self._path = path
        self._time_dilation = float(time_dilation)
        self._feature_normalize = feature_normalize
        # When True, missing schema_version causes a fatal error;
        # when False, it logs a warning.  Default True for
        # publication runs; CI runs may opt out.
        self._strict_schema = strict_schema
        # Cached file content hash for audit_signature().  Populated
        # on first call; not on construction (so loaders can be
        # constructed in environments where the file is unavailable).
        self._file_hash_cache: Optional[str] = None

    @abc.abstractmethod
    def stream(self) -> Iterator[TimedTransaction]:
        """
        Yield TimedTransactions in nondecreasing arrival-time order.
        Concrete subclasses implement this; common file-reading logic
        lives in `_iter_records` below.
        """
        raise NotImplementedError

    # --- card and audit -------------------------------------------------

    def card(self) -> DatasetCard:
        """Return this loader's dataset card."""
        return dataset_card(self.name)

    def audit_signature(self, full_file_hash: bool = False) -> str:
        """
        SHA-256 over loader configuration + file content.  Persisted
        into RunRecord so a reviewer can verify the run used the
        expected dataset version.

        When full_file_hash=False (default), hashes only the file's
        first and last 64 KB plus its size — fast for large files
        and sufficient to catch accidental swaps.  When True, hashes
        the full file content (slower; use for publication runs).
        """
        h = hashlib.sha256()
        h.update(self.name.encode())
        h.update(str(self.feature_dim).encode())
        h.update(str(self._time_dilation).encode())
        h.update(str(int(self._feature_normalize)).encode())
        h.update(str(self.EXPECTED_SCHEMA_VERSION).encode())
        try:
            h.update(self._file_hash(full=full_file_hash).encode())
        except FileNotFoundError:
            h.update(b"<file-missing>")
        return h.hexdigest()

    def _file_hash(self, full: bool = False) -> str:
        """
        Hash of the underlying preprocessed file.  Cached.  When
        full=False, prefix+suffix+size hash; when True, content
        hash.
        """
        if self._file_hash_cache is None:
            self._file_hash_cache = self._compute_file_hash(full)
        return self._file_hash_cache

    def _compute_file_hash(self, full: bool) -> str:
        if not self._path.exists():
            raise FileNotFoundError(f"dataset file not found: {self._path}")
        st = self._path.stat()
        size = st.st_size
        h = hashlib.sha256()
        h.update(f"size={size};".encode())
        if full:
            with open(self._path, "rb") as f:
                while True:
                    chunk = f.read(1024 * 1024)
                    if not chunk:
                        break
                    h.update(chunk)
            return f"full:{h.hexdigest()}"
        # Prefix + suffix mode.
        prefix_n = min(64 * 1024, size)
        suffix_n = min(64 * 1024, max(0, size - prefix_n))
        with open(self._path, "rb") as f:
            h.update(f.read(prefix_n))
            if suffix_n > 0:
                f.seek(size - suffix_n, 0)
                h.update(f.read(suffix_n))
        return f"prefix-suffix:{h.hexdigest()}"

    # --- helpers --------------------------------------------------------

    def _iter_records(self) -> Iterator[Mapping[str, Any]]:
        """
        Yield raw records from the preprocessed file in arrival order.
        The implementation prefers parquet (columnar, fast); falls back
        to CSV with a clear warning.

        We import inside the method so that loaders are usable in CI
        environments without parquet support.
        """
        suffix = self._path.suffix.lower()
        if suffix in (".parquet", ".pq"):
            yield from self._iter_parquet()
        elif suffix == ".csv":
            logger.warning(
                f"loading {self._path} as CSV; parquet is preferred for "
                f"reproducibility (deterministic ordering, lower I/O variance)"
            )
            yield from self._iter_csv()
        else:
            raise ValueError(f"unsupported file format: {self._path}")

    def _iter_parquet(self) -> Iterator[Mapping[str, Any]]:
        try:
            import pyarrow.parquet as pq          # type: ignore[import]
        except ImportError as e:                  # pragma: no cover
            raise RuntimeError(
                "pyarrow is required to load parquet datasets; "
                "install via `pip install pyarrow`"
            ) from e
        table = pq.read_table(self._path)
        names = table.schema.names

        # Required columns must be present.
        for col in self.expected_columns:
            if col not in names:
                raise ValueError(
                    f"{self._path}: missing required column '{col}'; "
                    f"present columns: {names}"
                )

        # Type validation.  We don't enforce exact types because
        # different preprocessing pipelines may use slightly
        # different integer widths; we enforce *families*.
        self._validate_column_types(table)

        # Schema version check.
        self._check_schema_version(table)

        # Iterate row-batched for memory safety on large files.
        present_cols = list(self.expected_columns)
        for opt in self.optional_columns:
            if opt in names:
                present_cols.append(opt)
        for batch in table.to_batches(max_chunksize=100_000):
            cols = {n: batch.column(n).to_pylist() for n in present_cols}
            n = len(cols["timestamp"])
            for i in range(n):
                yield {k: cols[k][i] for k in present_cols}

    def _validate_column_types(self, table: Any) -> None:
        """
        Check that the schema's column types are compatible with
        what we expect.  Raises ValueError with a clear diagnostic
        on mismatch.
        """
        import pyarrow as pa   # type: ignore[import]
        type_expectations: Mapping[str, Tuple[type, ...]] = {
            "timestamp": (pa.types.is_floating,),
            "source":    (pa.types.is_integer,),
            "target":    (pa.types.is_integer,),
            "amount":    (pa.types.is_floating,),
            "label":     (pa.types.is_integer,),
            "dataset":   (pa.types.is_string,),
        }
        for col, predicates in type_expectations.items():
            t = table.schema.field(col).type
            ok = any(p(t) for p in predicates)
            if not ok:
                raise ValueError(
                    f"{self._path}: column '{col}' has incompatible "
                    f"type {t}; expected one of: "
                    f"{[p.__name__ for p in predicates]}"
                )
        # `features` must be a list-of-floats.
        feats_t = table.schema.field("features").type
        if not pa.types.is_list(feats_t):
            raise ValueError(
                f"{self._path}: column 'features' must be list-typed, "
                f"got {feats_t}"
            )

    def _check_schema_version(self, table: Any) -> None:
        """
        If the file declares a schema_version, validate it; if not,
        warn (or fail when strict_schema=True).
        """
        names = table.schema.names
        if "schema_version" not in names:
            msg = (
                f"{self._path}: no schema_version column; assuming "
                f"v1 (legacy).  Loader expects "
                f"v{self.EXPECTED_SCHEMA_VERSION}."
            )
            if self._strict_schema:
                raise ValueError(msg + " Set strict_schema=False to bypass.")
            else:
                logger.warning(msg)
            return
        # Read the first row's schema_version (it is constant per
        # file by convention).
        first = table.column("schema_version")[0].as_py()
        if first != self.EXPECTED_SCHEMA_VERSION:
            raise ValueError(
                f"{self._path}: schema_version={first}, but "
                f"{self.__class__.__name__} expects "
                f"v{self.EXPECTED_SCHEMA_VERSION}.  Re-run the "
                f"preprocessing script in scripts/prep_{self.name}.py "
                f"to regenerate the file with the current schema."
            )

    def _iter_csv(self) -> Iterator[Mapping[str, Any]]:
        import csv
        with open(self._path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Features stored as a |-separated string in CSV.
                feats = [float(x) for x in row["features"].split("|") if x]
                yield {
                    "timestamp": float(row["timestamp"]),
                    "source": int(row["source"]),
                    "target": int(row["target"]),
                    "amount": float(row["amount"]),
                    "features": feats,
                    "label": int(row["label"]),
                    "dataset": row.get("dataset", self.name),
                }

    def _make_timed_transaction(
        self,
        rec: Mapping[str, Any],
        rebased_time: float,
        txn_id: int,
    ) -> TimedTransaction:
        feats = np.asarray(rec["features"], dtype=np.float32)
        if feats.size != self.feature_dim:
            raise ValueError(
                f"{self.name}: expected feature_dim={self.feature_dim}, "
                f"got {feats.size}"
            )
        if self._feature_normalize:
            # Per-record L2 normalisation: makes datasets with
            # heterogeneous feature scales comparable.  We do not
            # cross-record normalise here because that would require
            # two passes; the preprocessing scripts handle global
            # normalisation if requested.
            norm = float(np.linalg.norm(feats))
            if norm > 0.0:
                feats = feats / norm
        txn = Transaction(
            txn_id=txn_id,
            source=int(rec["source"]),
            target=int(rec["target"]),
            timestamp=rebased_time,
            amount=float(rec["amount"]),
            features=feats,
        )
        return TimedTransaction(
            arrival_time=rebased_time,
            transaction=txn,
            label=int(rec["label"]),
            is_adversarial=False,
            rationale="benign",
        )


# =============================================================================
# Section 4.  Concrete dataset loaders.
#
# Each loader specifies its expected feature_dim and adds dataset-
# specific validation if needed.  The common pattern in the base
# class does most of the work; subclasses are deliberately thin.
# Construction notes, citations, and limitations live on the
# DatasetCard registered in Section 2 — NOT in subclass docstrings,
# so that updating provenance does not require editing this file's
# code.
# =============================================================================


class EthereumPhishingLoader(DatasetLoader):
    """Ethereum phishing dataset.  See dataset_card('ethereum_phishing')."""

    name = "ethereum_phishing"
    feature_dim = 8

    def stream(self) -> Iterator[TimedTransaction]:
        first_ts: Optional[float] = None
        txn_id = 0
        for rec in self._iter_records():
            if first_ts is None:
                first_ts = float(rec["timestamp"])
            t = (float(rec["timestamp"]) - first_ts) / self._time_dilation
            yield self._make_timed_transaction(rec, t, txn_id)
            txn_id += 1


class BitcoinRansomwareLoader(DatasetLoader):
    """BitcoinHeist ransomware dataset.  See dataset_card('bitcoin_ransomware')."""

    name = "bitcoin_ransomware"
    feature_dim = 8

    def stream(self) -> Iterator[TimedTransaction]:
        first_ts: Optional[float] = None
        txn_id = 0
        for rec in self._iter_records():
            if first_ts is None:
                first_ts = float(rec["timestamp"])
            t = (float(rec["timestamp"]) - first_ts) / self._time_dilation
            yield self._make_timed_transaction(rec, t, txn_id)
            txn_id += 1


class CICIDS2018Loader(DatasetLoader):
    """
    CSE-CIC-IDS2018 network intrusion dataset.  Source/target are
    int64 encodings of (src_ip, src_port) and (dst_ip, dst_port)
    tuples respectively; features are 16 numeric per-flow statistics
    (duration, packet counts, byte counts, IAT statistics, flag
    counts), selected from the 80 CICFlowMeter features by a
    stability-and-informativeness filter on a held-out warm-up
    split.  This is the primary deadline-credible dataset.

    See dataset_card('cicids2018') for the version disambiguation
    (CSE-CIC-IDS2018, NOT the CICIDS2017 evaluated in the original
    Sharafaldin et al. ICISSP 2018 paper) and for the deadline
    semantics (post-flow-record update freshness, NOT first-packet
    blocking).  Preprocessing is expected to apply the
    Engelen et al. SPW 2021 corrections (TCP-termination handling,
    deduplicated flows, direction normalisation); see
    docs/DATASETS.md.
    """

    name = "cicids2018"
    feature_dim = 16

    def stream(self) -> Iterator[TimedTransaction]:
        first_ts: Optional[float] = None
        txn_id = 0
        for rec in self._iter_records():
            if first_ts is None:
                first_ts = float(rec["timestamp"])
            t = (float(rec["timestamp"]) - first_ts) / self._time_dilation
            yield self._make_timed_transaction(rec, t, txn_id)
            txn_id += 1


class SWaTLoader(DatasetLoader):
    """
    SWaT industrial control dataset.  See dataset_card('swat').
    """

    name = "swat"
    feature_dim = 25

    def stream(self) -> Iterator[TimedTransaction]:
        first_ts: Optional[float] = None
        txn_id = 0
        for rec in self._iter_records():
            if first_ts is None:
                first_ts = float(rec["timestamp"])
            t = (float(rec["timestamp"]) - first_ts) / self._time_dilation
            yield self._make_timed_transaction(rec, t, txn_id)
            txn_id += 1


# Registry — used by the experiment harness to load by name.
_LOADER_REGISTRY: Mapping[str, type[DatasetLoader]] = {
    EthereumPhishingLoader.name: EthereumPhishingLoader,
    BitcoinRansomwareLoader.name: BitcoinRansomwareLoader,
    CICIDS2018Loader.name: CICIDS2018Loader,
    SWaTLoader.name: SWaTLoader,
}


def dataset_names() -> Sequence[str]:
    return tuple(_LOADER_REGISTRY.keys())


def make_loader(
    name: str,
    path: Path,
    time_dilation: float = 1.0,
    feature_normalize: bool = True,
    strict_schema: bool = True,
) -> DatasetLoader:
    """
    Instantiate a DatasetLoader by name.

    `strict_schema=True` (default) makes a missing schema_version
    column a fatal error.  CI runs and quick experiments may pass
    `strict_schema=False`; publication runs should leave it True so
    that loading an out-of-date preprocessed file fails loudly.
    """
    if name not in _LOADER_REGISTRY:
        raise KeyError(
            f"unknown dataset '{name}'; known: {sorted(_LOADER_REGISTRY)}"
        )
    cls = _LOADER_REGISTRY[name]
    # Synthetic loader has a different constructor signature; route
    # through the registry-aware factory below.
    if cls is SyntheticLoader:
        return cls(
            path=path,
            time_dilation=time_dilation,
            feature_normalize=feature_normalize,
        )
    return cls(
        path=path,
        time_dilation=time_dilation,
        feature_normalize=feature_normalize,
        strict_schema=strict_schema,
    )


def feature_dim_for(name: str) -> int:
    if name not in _LOADER_REGISTRY:
        raise KeyError(f"unknown dataset '{name}'")
    return _LOADER_REGISTRY[name].feature_dim


# =============================================================================
# Section 5.  Stream mixing.
#
# The central correctness primitive of this file.  Given a benign
# stream and an Attack instance, produce a mixed stream whose
# adversarial fraction matches the budget's `fraction_of_stream`,
# whose attacker emissions respect the budget's rate constraints,
# and whose interleaving is deterministic given a seed.
#
# Method
# ------
# We use Poisson thinning.  At each benign arrival at time t_b, we
# draw a Bernoulli with success probability
#
#     p = α / (1 - α)        where α = budget.fraction_of_stream.
#
# On success, we ask the attack for one candidate at time t_b + δ
# (δ a small jitter) and inject it.  This makes the long-run
# adversarial fraction equal to α exactly (in expectation), and ties
# adversarial arrivals to benign arrivals so that the stream's
# burstiness pattern is preserved.
#
# When the attacker's stream is exhausted (either because the time
# horizon is reached or because all controlled sources are
# rate-limited), the mixer stops requesting candidates and yields
# only the benign tail.  The harness records this in run notes.
#
# A deterministic fingerprint is produced over the full sequence of
# (timestamp, source, target, label, is_adversarial) tuples and
# returned as MixedStream.fingerprint() so that two runs with the
# same fingerprint provably saw the same workload.
# =============================================================================


@dataclass(frozen=True)
class MixingConfig:
    seed: int = 0
    arrival_jitter_s: float = 1e-4         # δ in the discussion above
    enforce_budget_rate: bool = True       # drop attacker candidates that
                                           # would exceed budget.max_injection_rate


class MixedStream:
    """
    Iterable of TimedTransactions interleaving a benign dataset stream
    with an attacker's candidate stream.

    Construction is deferred — the attacker is asked for candidates
    only when the consumer pulls.  This means the entire pipeline is
    incrementally evaluable; you can stop consumption at any point
    without wasting attacker computation.

    Lifecycle
    ---------
    Once iterated, a MixedStream is consumed; iterate again would
    silently produce an empty sequence.  This is intentional — the
    underlying benign stream is also single-pass — and matches the
    semantics of file replay.  The harness explicitly constructs a
    fresh MixedStream per repetition.
    """

    def __init__(
        self,
        benign: Iterable[TimedTransaction],
        attack: Optional[Attack],
        budget: AdversaryBudget,
        config: MixingConfig,
        time_horizon_s: Optional[float] = None,
    ) -> None:
        self._benign = benign
        self._attack = attack
        self._budget = budget
        self._config = config
        self._time_horizon_s = time_horizon_s
        self._rng = np.random.default_rng(config.seed)

        # Statistics captured during iteration; readable after.
        self.n_benign_emitted = 0
        self.n_adversarial_emitted = 0
        self.n_adversarial_skipped_no_candidate = 0
        self.n_adversarial_skipped_rate_limit = 0
        self._fingerprint = hashlib.sha256()
        self._exhausted = False

        # Sliding-window of attacker emission times (last 1 s) for the
        # rate-limit check.  We track *aggregate* rate; per-source
        # rate is enforced inside the attack.
        self._recent_attacker_emissions: List[float] = []

    # --- public iteration -----------------------------------------------

    def __iter__(self) -> Iterator[TimedTransaction]:
        if self._exhausted:
            return iter(())
        return self._iterate()

    def fingerprint(self) -> str:
        """
        Stable SHA-256 over the emitted sequence.  Available after
        iteration completes; calling before is permitted but returns
        a partial hash.
        """
        return self._fingerprint.hexdigest()

    def statistics(self) -> Mapping[str, Any]:
        total = self.n_benign_emitted + self.n_adversarial_emitted
        return {
            "n_benign_emitted": self.n_benign_emitted,
            "n_adversarial_emitted": self.n_adversarial_emitted,
            "n_adversarial_skipped_no_candidate":
                self.n_adversarial_skipped_no_candidate,
            "n_adversarial_skipped_rate_limit":
                self.n_adversarial_skipped_rate_limit,
            "empirical_adversarial_fraction":
                (self.n_adversarial_emitted / total) if total > 0 else 0.0,
            "exhausted": self._exhausted,
            "fingerprint": self._fingerprint.hexdigest(),
        }

    # --- internal -------------------------------------------------------

    def _iterate(self) -> Iterator[TimedTransaction]:
        alpha = self._budget.fraction_of_stream
        if not 0.0 <= alpha < 1.0:
            raise ValueError(f"adversary fraction α must be in [0, 1); got {alpha}")
        # Probability of injecting an attacker txn between consecutive
        # benign txns: solve  α = p / (1 + p)  →  p = α / (1 - α).
        p_inject = alpha / (1.0 - alpha) if alpha < 1.0 else float("inf")

        for b_timed in self._benign:
            if self._time_horizon_s is not None and b_timed.arrival_time > self._time_horizon_s:
                break
            # Yield the benign transaction first.
            self._record_emit(b_timed)
            yield b_timed

            # Decide whether to inject an attacker txn after this benign.
            if self._attack is None or alpha == 0.0:
                continue
            should_inject = (self._rng.random() < p_inject)
            if not should_inject:
                continue

            # Time-of-injection: just after the benign, with small jitter.
            t_inject = b_timed.arrival_time + (
                self._rng.exponential(self._config.arrival_jitter_s)
            )

            # Aggregate-rate check.
            if self._config.enforce_budget_rate:
                self._prune_recent_emissions(t_inject - 1.0)
                if len(self._recent_attacker_emissions) >= self._budget.max_injection_rate:
                    self.n_adversarial_skipped_rate_limit += 1
                    continue

            # Ask the attack for a candidate.
            cand = self._attack._propose_one(t_inject)   # noqa: SLF001
            if cand is None:
                self.n_adversarial_skipped_no_candidate += 1
                continue

            # Convert the candidate's transaction (which has its own
            # timestamp from the attack) so the SUT sees `t_inject` as
            # the arrival time but the transaction object retains its
            # attacker-set timestamp.
            adv = TimedTransaction(
                arrival_time=t_inject,
                transaction=cand.transaction,
                label=1,           # by convention attacker injections are illicit
                is_adversarial=True,
                rationale=cand.rationale,
            )
            self._recent_attacker_emissions.append(t_inject)
            self._record_emit(adv)
            yield adv

        self._exhausted = True

    def _record_emit(self, t: TimedTransaction) -> None:
        if t.is_adversarial:
            self.n_adversarial_emitted += 1
        else:
            self.n_benign_emitted += 1
        # Update fingerprint with stable bytes.
        h = self._fingerprint
        h.update(int(round(t.arrival_time * 1e6)).to_bytes(8, "big", signed=True))
        h.update(int(t.transaction.source).to_bytes(8, "big", signed=True))
        h.update(int(t.transaction.target).to_bytes(8, "big", signed=True))
        h.update(int(t.label).to_bytes(2, "big", signed=True))
        h.update(b"\x01" if t.is_adversarial else b"\x00")

    def _prune_recent_emissions(self, cutoff: float) -> None:
        # In-place prune; recent_attacker_emissions is short and
        # time-ordered.
        i = 0
        while i < len(self._recent_attacker_emissions) and self._recent_attacker_emissions[i] < cutoff:
            i += 1
        if i > 0:
            del self._recent_attacker_emissions[:i]


# =============================================================================
# Section 6.  Synthetic stream (for unit tests and CI).
#
# CI does not have access to the real datasets.  This loader produces
# a synthetic stream with controllable shape so that tests/ can run
# end-to-end against the full pipeline.  Production experiments do
# NOT use this loader; experiments.py refuses to use it for any
# experiment marked "publication" by checking
# `dataset_card(name).is_synthetic` (the source of truth).
# =============================================================================


class SyntheticLoader(DatasetLoader):
    """
    Synthetic stream for tests.  Produces a power-law degree
    distribution similar to real transaction graphs.  See
    dataset_card('synthetic') for the documented limitations.
    """

    name = "synthetic"
    feature_dim = 8

    def __init__(
        self,
        path: Path,
        time_dilation: float = 1.0,
        feature_normalize: bool = True,
        n_records: int = 10_000,
        n_accounts: int = 1_000,
        seed: int = 0,
        rate_hz: float = 100.0,
        illicit_fraction: float = 0.01,
    ) -> None:
        # Synthetic loader doesn't enforce schema_version because
        # there is no preprocessed file.  We still call the parent
        # constructor for the audit infrastructure, with
        # strict_schema=False to suppress warnings on the absent
        # file path.
        super().__init__(
            path=path,
            time_dilation=time_dilation,
            feature_normalize=feature_normalize,
            strict_schema=False,
        )
        self._n_records = n_records
        self._n_accounts = n_accounts
        self._seed = seed
        self._rate_hz = rate_hz
        self._illicit_fraction = illicit_fraction

    def audit_signature(self, full_file_hash: bool = False) -> str:
        """
        Synthetic-specific audit signature: hash over the
        generation parameters, since there is no preprocessed file
        to hash.  Overrides the parent implementation, which would
        try to hash a non-existent file.
        """
        h = hashlib.sha256()
        h.update(self.name.encode())
        h.update(f"n_records={self._n_records};".encode())
        h.update(f"n_accounts={self._n_accounts};".encode())
        h.update(f"seed={self._seed};".encode())
        h.update(f"rate_hz={self._rate_hz};".encode())
        h.update(f"illicit_fraction={self._illicit_fraction};".encode())
        h.update(f"feature_dim={self.feature_dim};".encode())
        h.update(f"feature_normalize={int(self._feature_normalize)};".encode())
        h.update(f"time_dilation={self._time_dilation};".encode())
        return h.hexdigest()

    def stream(self) -> Iterator[TimedTransaction]:
        rng = np.random.default_rng(self._seed)
        # Power-law degree distribution: account popularity ~ rank^-1.
        ranks = np.arange(1, self._n_accounts + 1, dtype=np.float64)
        weights = 1.0 / ranks
        weights = weights / weights.sum()
        accounts = np.arange(self._n_accounts)

        for txn_id in range(self._n_records):
            t = txn_id / self._rate_hz / self._time_dilation
            src = int(rng.choice(accounts, p=weights))
            tgt = int(rng.choice(accounts, p=weights))
            while tgt == src:
                tgt = int(rng.choice(accounts, p=weights))
            features = rng.standard_normal(self.feature_dim).astype(np.float32)
            if self._feature_normalize:
                norm = float(np.linalg.norm(features))
                if norm > 0:
                    features = features / norm
            label = 1 if rng.random() < self._illicit_fraction else 0
            txn = Transaction(
                txn_id=txn_id,
                source=src,
                target=tgt,
                timestamp=t,
                amount=float(rng.uniform(0.1, 10.0)),
                features=features,
            )
            yield TimedTransaction(
                arrival_time=t,
                transaction=txn,
                label=label,
                is_adversarial=False,
                rationale="benign-synthetic",
            )


# Add synthetic to the registry under a guarded name.  Callers that
# care about publication-vs-CI distinctions should consult
# `dataset_card(name).is_synthetic`, which is True only here.
_LOADER_REGISTRY = dict(_LOADER_REGISTRY)
_LOADER_REGISTRY["synthetic"] = SyntheticLoader        # type: ignore[assignment]


# =============================================================================
# Section 7.  Bounded-window replay.
#
# Some experiments need to replay only a portion of a dataset (e.g.
# the first hour, or the longest contiguous burst region).  This
# helper wraps any TimedTransaction iterable with a time window.  It
# is used by experiments.py to produce reproducible time-bounded
# replays without modifying the loaders.
# =============================================================================


def replay_window(
    stream: Iterable[TimedTransaction],
    t_start: float,
    t_end: float,
) -> Iterator[TimedTransaction]:
    """
    Yield only transactions with t_start ≤ arrival_time ≤ t_end.
    Assumes the input is sorted by arrival_time (true for all
    DatasetLoader outputs).  Stops iterating as soon as t_end is
    exceeded — does NOT consume the rest of the stream.
    """
    for t in stream:
        if t.arrival_time < t_start:
            continue
        if t.arrival_time > t_end:
            return
        yield t


# =============================================================================
# Section 8.  Public surface.
# =============================================================================

__all__ = [
    # Core type
    "TimedTransaction",
    # Cards
    "DatasetCard",
    "dataset_card",
    "all_dataset_cards",
    # Loaders
    "DatasetLoader",
    "EthereumPhishingLoader",
    "BitcoinRansomwareLoader",
    "CICIDS2018Loader",
    "SWaTLoader",
    "SyntheticLoader",
    "dataset_names",
    "make_loader",
    "feature_dim_for",
    # Mixing
    "MixingConfig",
    "MixedStream",
    # Replay helpers
    "replay_window",
]
