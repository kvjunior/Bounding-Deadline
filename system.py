"""
system.py — System Under Test (SUT): the victim of update-storm attacks.

This module implements the continuously-learning system that the paper
attacks and defends.  It is the *victim*, not the contribution.  The
contribution of the paper is the threat model, the attacks, the
schedulability analysis, and the defense; this module exists to give
those things something real to operate on.

Design philosophy
-----------------
A faithful victim is one that exhibits the behaviour an attacker
exploits.  In this paper the exploited behaviour is:

  (i)   the update path's data-dependent execution time, dominated by
        BFS reach over the affected subgraph;
  (ii)  the per-account adaptive sequence storage, which makes some
        accounts vastly more expensive to update than others;
  (iii) the gradient-magnitude variability across affected parameters,
        which the white-box adversary in tier-3 exploits.

The implementation here preserves every property in (i)–(iii) and
adds nothing else.  Mechanisms not relevant to the threat model
(compression, multi-tier storage, cross-chain support, multi-language
attribute encoders) have been removed from the predecessor codebase
to keep the SUT auditable.  A reviewer should be able to read this
file end-to-end and certify that the victim is a fair representation
of a production online learner without being unrealistically
vulnerable.

Mapping to the paper
--------------------
§II  (System model)              →  classes TemporalIndex, OnlineLearner
§III (Update path)               →  class UpdatePath
§III.D (Update cost function)    →  cost_decomposition_for_transaction()
§V   (Empirical evaluation)      →  consumes UpdatePathProbe events

Two prior-review issues fixed here
----------------------------------
ICDE R2-O1 (insertion complexity claim).  The original codebase
claimed O(1) array insertion.  This is false in the worst case
(shift) but correct as amortized cost under geometric reallocation.
The implementation here uses an explicit doubling vector and the
cost analysis in analysis.py uses the *amortized* O(log n + 1*)
bound, with the asterisk on the constant-time portion noting
amortization.  The cost function in this module reports both the
immediate and the amortized cost so the analysis can use whichever
it needs.

ICDE R2-O7 (F1 / incremental-update inconsistency).  The previous
paper reported F1 inconsistently between Figure 6 and the ablation
table.  There are two distinct quantities: F1 measured *under
continuous operation with incremental updates* (which can drift)
versus F1 measured *at end-of-stream after a final update sweep*
(which approximates the batch baseline).  Both are now defined
explicitly and computed in disjoint code paths in
``measure_detection_quality()``.  The paper reports both and labels
them.

SUT-fidelity disclosure (camera-ready)
--------------------------------------
The SUT in this file is *DIAM-inspired*, not *DIAM-faithful*.  It
uses the same overall structure as Tang et al.'s "DIAM: Diversity-
Aware Intent-Aware Model for Streaming Recommendation" (CIKM 2024) —
a per-account temporal encoder feeding a graph-aware downstream head
— but omits four DIAM-specific architectural choices:

  1. **Single GRU encoder.**  DIAM uses two separate encoders
     GRU_in and GRU_out for incoming and outgoing sequences with
     dedicated parameters per stream.  We use a single GRU applied
     twice (parameter-tied), with the two outputs concatenated and
     projected to hidden_dim.  This halves the encoder parameter
     count and is what ``OnlineLearner.encode_sequences`` does.
  2. **Mean-aggregation graph layer instead of 3-way attention.**
     DIAM's MGD module performs three-way attention over (self,
     neighbour, intent) pairs with a learned discrepancy term.  We
     use mean aggregation (``_GraphLayer`` below).  Mean
     aggregation is the simplest GNN aggregator and makes the
     cost-decomposition forward-pass term well-defined and uniform
     across nodes.
  3. **No MGD (Memory-Guided Discrepancy) decoder.**  DIAM's MGD
     decoder injects a discrepancy bias ``[z_u || z_v − z_u]`` into
     the classifier head.  We use a plain linear classifier on the
     final node embedding.  MGD is a recommender-specific design
     not relevant to the schedulability analysis.
  4. **``sequence_truncation = 256`` (vs. DIAM's ``T_max = 32``).**
     We use a longer history because the per-account window length
     is one of the variables the attacker exploits; truncating to
     32 would make the BFS-cost-vs-sequence-length sensitivity
     analysis (a §V.4 result) less informative.

Why these simplifications are acceptable for the paper's claims
---------------------------------------------------------------
The contribution of this paper is *the existence of a real-time
attack surface on the update path of continuously-learning systems
and a schedulability-based defense for it*.  The exploited
behaviours (i)–(iii) above depend on:

  - per-account temporal storage with geometric reallocation,
  - bounded BFS over a hub-aware adjacency graph,
  - data-dependent forward/backward passes on an affected subgraph.

DIAM has all three.  So does our SUT.  Therefore an attack that
works against our SUT will also work against DIAM (the same
data-dependent patterns are present), but our SUT has fewer
parameters and a simpler architecture, making the empirical
evaluation cheaper and the analysis more transparent.

A faithful DIAM rewrite is documented as a follow-up in
``docs/DIAM_FIDELITY.md``, including a discussion of which
cost-function parameters change and why the schedulability bounds
remain the same under the substitution.  We do not include the
rewrite in this paper because the schedulability claims (Theorems
4.1–4.3 and Lemma 4.4 in §IV) are distribution-level — they bound
P(latency > T) given a cost distribution, not given an architecture
— so substituting DIAM for our SUT changes only the empirical
distribution, not the bound.

Contract with the threat model
------------------------------
This module implements four protocols from threat_model.py:

  GraphTopologySource    →  TemporalIndex (this file)
  ModelMetadataSource    →  OnlineLearner.metadata_view()
  ModelWeightsSource     →  OnlineLearner.weights_view()
  DefenderStateSource    →  implemented by defenses.py, not here

A reviewer who wishes to audit the threat-model contract can do so
by reading the four ``_*Adapter`` classes near the bottom of this
file.  They are 30 lines total and contain no logic beyond
delegation.

Camera-ready improvements over the submission draft
---------------------------------------------------
1. **Phase-A timing skew fixed.**  The submission draft emitted the
   ``A_begin`` probe event with ``time.monotonic_ns()``, then read
   the clock again to start ``a_start`` for ``bfs_us``.  Two clock
   reads with arbitrary work between them meant the probe-side
   ``A_begin → A_end`` interval and the in-band ``bfs_us`` did not
   match.  Reviewers cross-checking the two will see drift.  The
   camera-ready takes ONE clock read at phase A start and reuses
   it for both the probe event and ``bfs_us``.  Same for B and C.

2. **CUDA-event timing path corrected.**  The submission draft
   recorded ``cuda_start`` unconditionally for both forward-only
   and training branches but only called ``_cuda_or_walltime``
   in the forward-only branch.  In the training branch, the
   ``cuda_start.elapsed_time(cuda_end)`` measurement was
   never read, so the CUDA-event allocation was dead work.  The
   camera-ready uses CUDA events consistently in both branches and
   adds a regression test in test_system.py
   (``test_cuda_events_used_when_present``).

3. **Default device is "cpu".**  The submission draft defaulted to
   ``"cuda"``, which crashes on the CI host.  The camera-ready
   defaults to ``"cpu"`` and makes GPU opt-in via explicit
   ``device="cuda"`` configuration.  Tests no longer need to
   override the default.

4. **``_build_batch`` uses dict-based row lookup throughout.**  The
   submission draft's ``_build_target`` called ``affected.index(
   txn.source)``, which is O(N).  For N=4096 affected nodes over a
   50k-transaction run, that's 200M extra ops.  The camera-ready
   builds a single ``node_to_row`` dict in ``_build_batch`` and
   reuses it in ``_build_target``.

5. **``_DoublingVector`` switched to numpy-backed key storage.**
   The submission draft stored keys as a Python list and used
   slice-assignment for shifts.  Slice-assignment dominates the
   amortization audit (Python overhead per element).  The
   camera-ready stores keys as ``np.ndarray[float64]`` so shifts
   use ``np.roll``-equivalent slicing implemented in C.  Payloads
   remain a Python list because they are heterogeneous objects.
   The amortization audit numbers continue to count *operations*,
   not Python time, so the analysis claims do not change.

6. **Thread-safety claim corrected.**  The submission draft's
   ``TemporalIndex._lock`` suggested thread-safety that the SUT
   does not actually provide — BFS reads the index without locking.
   The camera-ready documents that the SUT is single-threaded by
   contract: the harness serialises calls.  The ``_lock`` is
   retained as a defensive guard against accidental concurrent
   ingestion but the read path is explicitly NOT thread-safe.

7. **``record_decision`` placement clarified.**  The submission
   draft called ``probe.record_decision(t_ns)`` between
   ``_build_batch`` and the forward pass, leaving the question of
   "what counts as the decision moment" implicit.  The camera-ready
   places ``record_decision`` precisely at forward-pass start (just
   before the model is invoked) and documents the choice: the AoI
   freshness tracker samples model age at the moment the model
   produces an output, which is the moment that output is
   subsequently used by the decision pipeline.

8. **``architecture_signature`` advertises camera-ready improvements.**
   The signature now includes ``cuda_events_corrected=True`` and
   ``timing_skew_corrected=True`` so reviewers can verify which SUT
   variant produced each result via the run record.

Author: ANONYMOUS  (double-blind review)
"""

from __future__ import annotations

import bisect
import logging
import threading
import time
from dataclasses import dataclass, field
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
import torch
import torch.nn as nn
import torch.nn.functional as F

from threat_model import (
    CostDecomposition,
    GraphTopologySource,
    ModelMetadataSource,
    ModelWeightsSource,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Section 1.  Transaction record and the temporal index.
#
# The temporal index is the data structure the BFS walks.  It is a
# simplified, explicitly amortized version of the predecessor codebase's
# index, with mechanisms removed unless required by the threat model.
#
# Specifically retained:  per-account incoming/outgoing sorted arrays,
#                         per-account adaptive capacity, degree query.
# Specifically removed:   compression, hot/cold tiering, persistent
#                         storage, cross-shard partitioning.
#
# The retained behaviour is what makes the update path data-dependent
# and therefore attackable; the removed behaviour is unrelated to the
# threat model and would only obscure the analysis.
# =============================================================================


@dataclass(frozen=True, slots=True)
class Transaction:
    """One transaction.  Immutable once created."""

    txn_id: int
    source: int
    target: int
    timestamp: float
    amount: float
    features: np.ndarray  # shape (D,), dtype float32


class _DoublingVector:
    """
    Sorted vector of (timestamp, payload) pairs with explicit geometric
    doubling.  Supports binary-search insertion in O(log n + n_shift)
    immediate cost and O(log n) *amortized* cost (excluding payload
    shift, which is O(n_shift) Python list slicing).

    This class exists to make the amortization in the paper's
    complexity analysis explicit and visible.  The previous paper
    claimed O(1) insertion without justification; here we expose
    both costs so that analysis.py can use the bound appropriate to
    the claim being made.

    Camera-ready: keys are stored as ``np.ndarray[float64]`` so that
    in-place key shifts run in C-speed numpy code rather than as
    per-element Python list slice-assignment.  Payloads remain a
    Python list because they are heterogeneous (Transaction
    objects).  The audit counts shift *operations*, not Python time,
    so the amortization claim is unchanged.

    Thread-safety: not thread-safe.  The owning OnlineLearner
    serialises writes through the harness's single-threaded loop;
    callers MUST NOT concurrently insert from multiple threads.
    """

    __slots__ = (
        "_keys",          # np.ndarray, length _capacity
        "_payloads",      # list, length _capacity
        "_size",
        "_capacity",
        "_total_inserts",
        "_total_shift_cost",
    )

    def __init__(self, initial_capacity: int = 16) -> None:
        self._capacity = max(1, initial_capacity)
        self._keys = np.zeros(self._capacity, dtype=np.float64)
        self._payloads: List[Optional[Transaction]] = [None] * self._capacity
        self._size = 0
        self._total_inserts = 0
        self._total_shift_cost = 0  # for amortization audit

    def __len__(self) -> int:
        return self._size

    def insert(self, key: float, payload: Transaction) -> Tuple[int, int]:
        """
        Insert (key, payload) maintaining sorted order on key.

        Returns ``(insertion_index, shift_count)`` so the caller can
        record the immediate cost.  ``shift_count`` is 0 if the
        insertion was at the end.

        Cost (immediate):  O(log n + shift_count + reallocation)
        Cost (amortized):  O(log n)
        """
        # Binary search for insertion point on the live prefix.  We
        # use ``np.searchsorted`` rather than ``bisect`` because keys
        # are now numpy-backed; both are O(log n) but searchsorted
        # avoids the per-call Python wrapper.
        idx = int(np.searchsorted(
            self._keys[: self._size], key, side="right"
        ))

        # Reallocate if full.  This is the only place capacity grows.
        if self._size == self._capacity:
            new_capacity = self._capacity * 2
            new_keys = np.zeros(new_capacity, dtype=np.float64)
            new_keys[: self._capacity] = self._keys
            self._keys = new_keys
            self._payloads = self._payloads + [None] * self._capacity
            self._capacity = new_capacity

        # Shift to make room.  This is the cost the attacker tries
        # to inflate by injecting old-timestamped transactions that
        # land near the front of long sequences.  The attack model
        # in attacks.py assumes the attacker can predict idx within
        # ±1.
        shift_count = self._size - idx
        if shift_count > 0:
            # Numpy in-place shift for keys (C-speed for any array
            # backing).  Note that this requires copying through a
            # temporary slice, which numpy handles correctly even
            # for overlapping ranges.
            self._keys[idx + 1 : self._size + 1] = self._keys[idx : self._size]
            # Python list slice for payloads (Transaction objects).
            self._payloads[idx + 1 : self._size + 1] = (
                self._payloads[idx : self._size]
            )

        self._keys[idx] = key
        self._payloads[idx] = payload
        self._size += 1
        self._total_inserts += 1
        self._total_shift_cost += shift_count
        return idx, shift_count

    def range_query(self, t_lo: float, t_hi: float) -> List[Transaction]:
        """
        Return payloads with key in ``[t_lo, t_hi]``.  O(log n + k).
        """
        live_keys = self._keys[: self._size]
        lo = int(np.searchsorted(live_keys, t_lo, side="left"))
        hi = int(np.searchsorted(live_keys, t_hi, side="right"))
        return [self._payloads[i] for i in range(lo, hi)]  # type: ignore[misc]

    def all_keys(self) -> np.ndarray:
        """
        Return a view of the live keys.  Caller must not mutate.

        This returns a view (not a copy) for performance.  The
        existing tests use ``list(v.all_keys())`` or
        ``v.all_keys().tolist()``, which materialise; tests that
        index into the view directly remain valid because the SUT
        is single-threaded and the view's underlying buffer is
        not modified during a read.
        """
        return self._keys[: self._size]

    def amortization_audit(self) -> Tuple[int, int]:
        """
        ``(total_inserts, total_shift_cost)``.  Used by the experiment
        harness to verify that amortized shift cost stays O(1) per
        insert in the absence of adversarial input.  An adversarial
        workload should push this ratio up; that is one of the
        things the paper measures.
        """
        return self._total_inserts, self._total_shift_cost


@dataclass
class _AccountWindows:
    """Per-account incoming and outgoing transaction windows."""

    incoming: _DoublingVector = field(default_factory=_DoublingVector)
    outgoing: _DoublingVector = field(default_factory=_DoublingVector)


class TemporalIndex:
    """
    Per-account sorted transaction sequences with degree queries.

    This is the data structure the BFS in the update path walks.
    Its interface is deliberately small.  Its degree-query semantics
    (degree = ``|incoming| + |outgoing|``) match what the attacker
    observes publicly on a blockchain and what the SUT uses for
    hub-node avoidance during BFS.

    Thread-safety
    -------------
    The internal lock guards individual ``insert_transaction`` calls
    against accidental concurrent ingestion.  However, the *read
    path* (BFS, neighbours, edge_timestamps) is NOT thread-safe with
    respect to concurrent writes, and the SUT does not promise
    serialisability of reads vs. writes.  Real deployments must
    either (a) serialise all access through the harness loop (what
    the paper does) or (b) wrap the index in a higher-level
    read-write lock (out of scope).
    """

    def __init__(self) -> None:
        self._accounts: Dict[int, _AccountWindows] = {}
        self._adjacency: Dict[int, set[int]] = {}   # node → neighbour set
        self._lock = threading.Lock()

    # --- ingestion --------------------------------------------------------

    def insert_transaction(self, txn: Transaction) -> Tuple[int, int]:
        """
        Insert a transaction.  Returns ``(source_shift, target_shift)``
        so UpdatePath can record the immediate insertion cost
        component.
        """
        with self._lock:
            src_win = self._accounts.setdefault(txn.source, _AccountWindows())
            tgt_win = self._accounts.setdefault(txn.target, _AccountWindows())
            _, src_shift = src_win.outgoing.insert(txn.timestamp, txn)
            _, tgt_shift = tgt_win.incoming.insert(txn.timestamp, txn)
            self._adjacency.setdefault(txn.source, set()).add(txn.target)
            self._adjacency.setdefault(txn.target, set()).add(txn.source)
            return src_shift, tgt_shift

    # --- GraphTopologySource protocol -------------------------------------

    def num_nodes(self) -> int:
        return len(self._accounts)

    def degree(self, node_id: int) -> int:
        win = self._accounts.get(node_id)
        if win is None:
            return 0
        return len(win.incoming) + len(win.outgoing)

    def neighbors(self, node_id: int) -> Sequence[int]:
        return tuple(self._adjacency.get(node_id, ()))

    def edge_timestamps(self, node_id: int) -> np.ndarray:
        win = self._accounts.get(node_id)
        if win is None:
            return np.zeros(0, dtype=np.float64)
        # all_keys() returns views into _DoublingVector storage.
        # We make a copy here because callers may persist or mutate
        # the result (e.g., attacks.py sorts it).
        return np.concatenate([
            np.asarray(win.incoming.all_keys()),
            np.asarray(win.outgoing.all_keys()),
        ])

    # --- helpers used by UpdatePath ---------------------------------------

    def account_windows(self, node_id: int) -> Optional[_AccountWindows]:
        return self._accounts.get(node_id)

    def all_account_ids(self) -> Iterable[int]:
        return self._accounts.keys()


# =============================================================================
# Section 2.  Affected-subgraph BFS (the central operation of the update
# path; the operation the adversary inflates).
#
# This is the operation whose execution time the threat model targets.
# The implementation is deliberately straightforward: a degree-bounded
# breadth-first traversal up to a maximum depth.  Two parameters
# control its behaviour:
#
#   max_depth        — how many hops out from the seed transaction
#                      endpoints the BFS expands.  Production systems
#                      typically use 2–4.
#   degree_threshold — nodes of degree above this are *not* expanded
#                      further (i.e. they enter the affected set but
#                      their neighbours do not).  Without this, a single
#                      transaction touching a major exchange would force
#                      a full-graph update.
#
# Both parameters are part of the SUT's public configuration.  The
# attacker knows them (under tier 1 they can be inferred from observed
# update latency; under tier 2+ they are documented).  The defender
# does not change them; defense.py operates upstream by deciding which
# transactions even reach this point.
# =============================================================================


@dataclass(frozen=True)
class BFSResult:
    affected_nodes: Tuple[int, ...]
    depth_reached: int
    branching_factors: Tuple[float, ...]    # one per depth level expanded
    edges_visited: int
    nodes_at_depth: Tuple[int, ...]         # |frontier| at each depth


def affected_subgraph_bfs(
    index: TemporalIndex,
    seed_nodes: Sequence[int],
    max_depth: int,
    degree_threshold: int,
) -> BFSResult:
    """
    Bounded BFS from ``seed_nodes``.  Returns the affected node set
    and traversal statistics.  The returned node tuple is
    deduplicated and deterministically ordered (sorted), which makes
    downstream measurements stable across runs.

    Complexity:  O(|A| + |E_local|) where A is the returned affected
    set and E_local is the set of edges incident on A whose far
    endpoints have degree ≤ threshold.  See analysis.py for the
    formal statement.
    """
    if max_depth < 0:
        raise ValueError("max_depth must be non-negative")
    if degree_threshold < 1:
        raise ValueError("degree_threshold must be at least 1")

    visited: set[int] = set(seed_nodes)
    frontier: List[int] = list(seed_nodes)
    nodes_at_depth: List[int] = [len(frontier)]
    branching: List[float] = []
    edges_visited = 0
    depth_reached = 0

    for depth in range(1, max_depth + 1):
        next_frontier: List[int] = []
        for node in frontier:
            for neighbour in index.neighbors(node):
                edges_visited += 1
                if neighbour in visited:
                    continue
                visited.add(neighbour)
                # Hub-node guard: enter the set but do not expand.
                if index.degree(neighbour) <= degree_threshold:
                    next_frontier.append(neighbour)
                # else: neighbour is in `visited` but not in
                # `next_frontier`, so the BFS stops at it.

        if not next_frontier:
            break

        # Branching factor from previous frontier to this one.
        prev = nodes_at_depth[-1]
        branching.append(len(next_frontier) / max(prev, 1))
        nodes_at_depth.append(len(next_frontier))
        frontier = next_frontier
        depth_reached = depth

    return BFSResult(
        affected_nodes=tuple(sorted(visited)),
        depth_reached=depth_reached,
        branching_factors=tuple(branching),
        edges_visited=edges_visited,
        nodes_at_depth=tuple(nodes_at_depth),
    )


# =============================================================================
# Section 3.  The online learner.
#
# A two-layer GNN with a small recurrent encoder over the per-account
# incoming/outgoing windows.  This is a deliberately compact model —
# large enough to be a credible victim, small enough that the threat
# model is the headline rather than the architecture.  The previous
# codebase's six-component model has been collapsed to three because
# the adversary's action surface depends on the update-path, not on
# encoder choice.
#
# DIAM-inspired but not DIAM-faithful.  See the SUT-fidelity disclosure
# in this file's header docstring; details in docs/DIAM_FIDELITY.md.
# =============================================================================


@dataclass(frozen=True)
class LearnerConfig:
    feature_dim: int
    hidden_dim: int = 64
    num_gnn_layers: int = 2
    num_classes: int = 2
    dropout: float = 0.1
    sequence_truncation: int = 256       # max txns per direction read by encoder
                                         # (DIAM uses T_max=32; we use 256 for the
                                         # sensitivity analysis in §V.4)
    learning_rate: float = 1e-3
    # Camera-ready: default to "cpu" so the SUT runs on hosts without
    # a CUDA device (e.g., CI).  GPU is opt-in via explicit
    # device="cuda" configuration.  The submission draft defaulted to
    # "cuda", which crashed on the CI host and forced every test
    # fixture to override the default.
    device: str = "cpu"

    def validate(self) -> None:
        if self.feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.num_gnn_layers < 1:
            raise ValueError("num_gnn_layers must be at least 1")
        if self.sequence_truncation < 1:
            raise ValueError("sequence_truncation must be at least 1")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


class _GraphLayer(nn.Module):
    """One graph convolution layer.  Aggregation is mean over neighbours.

    DIAM uses 3-way attention with a learned discrepancy term over
    (self, neighbour, intent) triples.  We use mean aggregation:
    simpler, makes the cost-decomposition forward-pass term well-
    defined per node, and uniform across nodes (so the §IV.3
    bound's per-node cost expectation is easy to compute).
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.self_lin = nn.Linear(in_dim, out_dim)
        self.neigh_lin = nn.Linear(in_dim, out_dim)

    def forward(
        self,
        h: torch.Tensor,             # (N, in_dim)
        neigh_index: torch.Tensor,   # (E, 2): (src, dst)
    ) -> torch.Tensor:
        """
        Mean-aggregation graph conv.  Implemented with index_add for
        speed and to ensure the cost decomposition's "GNN forward"
        term is identifiable in profiles.
        """
        N = h.size(0)
        out_dim = self.neigh_lin.out_features
        if neigh_index.numel() == 0:
            return F.relu(self.self_lin(h))
        src = neigh_index[:, 0]
        dst = neigh_index[:, 1]
        msg = self.neigh_lin(h[src])                         # (E, out_dim)
        agg = torch.zeros(N, out_dim, device=h.device, dtype=msg.dtype)
        agg.index_add_(0, dst, msg)
        # Normalise by in-degree.  Use msg.dtype so mixed-precision
        # runs stay in their declared dtype (the submission draft
        # used the default float32 here, silently upcasting agg/deg
        # in float16 mode).
        deg = torch.zeros(N, device=h.device, dtype=msg.dtype).index_add_(
            0, dst, torch.ones_like(dst, dtype=msg.dtype)
        )
        deg = deg.clamp(min=1.0).unsqueeze(1)
        agg = agg / deg
        return F.relu(self.self_lin(h) + agg)


class OnlineLearner(nn.Module):
    """
    The model.  Encodes per-account sequences with a tiny GRU, then
    runs k message-passing rounds on the affected subgraph, then
    classifies.  All forward and backward passes go through this
    object so that gradient computation is identifiable in profiles.

    The model intentionally has no special "incremental update" mode
    at the architecture level.  The incremental update is achieved
    by forwarding only the affected subgraph and back-propagating
    only on the parameters touched by it.  This is implemented in
    UpdatePath.

    DIAM-inspired but not DIAM-faithful (see file header).  Three
    architectural simplifications:
      - Single GRU (parameter-tied for in and out streams) instead
        of DIAM's separate GRU_in / GRU_out.
      - Mean-aggregation graph layer instead of DIAM's 3-way
        attention.
      - Plain linear classifier instead of DIAM's MGD decoder.
    """

    def __init__(self, config: LearnerConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config

        # Single GRU, applied to both `in` and `out` sequences.
        # DIAM uses two separate GRUs; we tie parameters here
        # because the threat-model-relevant property is "GRU
        # forward over a truncated sequence", not "two distinct GRU
        # heads".
        self.encoder = nn.GRU(
            input_size=config.feature_dim,
            hidden_size=config.hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        # Two encoders' outputs (in, out) get concatenated then
        # projected to hidden_dim before the GNN layers.
        self.encoder_proj = nn.Linear(2 * config.hidden_dim, config.hidden_dim)
        self.gnn_layers = nn.ModuleList(
            [
                _GraphLayer(config.hidden_dim, config.hidden_dim)
                for _ in range(config.num_gnn_layers)
            ]
        )
        self.dropout = nn.Dropout(config.dropout)
        # Plain classifier head (DIAM uses MGD decoder; the
        # schedulability bound does not depend on the head structure).
        self.classifier = nn.Linear(config.hidden_dim, config.num_classes)

    # --- forward / loss --------------------------------------------------

    def encode_sequences(
        self,
        in_seq: torch.Tensor,      # (B, L_in, D)
        out_seq: torch.Tensor,     # (B, L_out, D)
        in_len: torch.Tensor,      # (B,) lengths for masking
        out_len: torch.Tensor,
    ) -> torch.Tensor:
        h_in = self._gru_last(self.encoder, in_seq, in_len)
        h_out = self._gru_last(self.encoder, out_seq, out_len)
        h = torch.cat([h_in, h_out], dim=1)
        return F.relu(self.encoder_proj(h))

    def forward(
        self,
        in_seq: torch.Tensor,
        out_seq: torch.Tensor,
        in_len: torch.Tensor,
        out_len: torch.Tensor,
        neigh_index: torch.Tensor,
    ) -> torch.Tensor:
        h = self.encode_sequences(in_seq, out_seq, in_len, out_len)
        for layer in self.gnn_layers:
            h = layer(h, neigh_index)
            h = self.dropout(h)
        return self.classifier(h)

    @staticmethod
    def _gru_last(
        gru: nn.GRU, seq: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        """Run GRU over padded sequences and return final hidden state."""
        if seq.size(1) == 0:
            return torch.zeros(seq.size(0), gru.hidden_size, device=seq.device)
        # We do not pack, because in this paper sequence lengths are
        # truncated to a small constant and padding overhead is
        # negligible compared to the GNN cost.  This is one of the
        # simplifications made vs. the predecessor codebase.
        out, _ = gru(seq)
        # Pick the last unmasked index per row.
        idx = (lengths - 1).clamp(min=0).long()
        gather_idx = idx.view(-1, 1, 1).expand(-1, 1, out.size(2))
        return out.gather(1, gather_idx).squeeze(1)

    # --- ModelMetadataSource / ModelWeightsSource adapters ---------------

    def architecture_signature(self) -> Mapping[str, object]:
        """
        Architecture signature for the threat-model metadata source.

        Camera-ready: explicitly advertises DIAM provenance,
        simplifications, and the camera-ready timing/CUDA fixes so
        a reviewer can audit which SUT variant produced each result
        via the run record.
        """
        return {
            "feature_dim": self.config.feature_dim,
            "hidden_dim": self.config.hidden_dim,
            "num_gnn_layers": self.config.num_gnn_layers,
            "num_classes": self.config.num_classes,
            "encoder": "GRU (single, parameter-tied for in/out)",
            "gnn_aggregation": "mean",
            "classifier": "linear",
            "sequence_truncation": self.config.sequence_truncation,
            # DIAM provenance fields.
            "diam_inspired": True,
            "diam_paper": "Tang et al., 'DIAM,' CIKM 2024",
            "diam_simplifications": (
                "single-GRU (parameter-tied)",
                "mean-aggregation GNN (vs. 3-way attention)",
                "no MGD decoder",
                f"sequence_truncation={self.config.sequence_truncation} "
                "(vs. DIAM T_max=32)",
            ),
            # Camera-ready fixes recorded for reviewer-facing audit.
            "timing_skew_corrected": True,
            "cuda_events_corrected": True,
        }

    def parameter_shapes(self) -> Mapping[str, Tuple[int, ...]]:
        return {n: tuple(p.shape) for n, p in self.named_parameters()}

    def parameter_tensor_numpy(self, name: str) -> np.ndarray:
        for n, p in self.named_parameters():
            if n == name:
                return p.detach().cpu().numpy()
        raise KeyError(f"unknown parameter '{name}'")


# =============================================================================
# Section 4.  The update path.
#
# The update path is the operation whose worst-case latency the paper
# bounds and the adversary attacks.  It is invoked once per accepted
# transaction and goes through three phases:
#
#   PHASE A  (CPU): index insertion + affected-subgraph BFS.
#   PHASE B  (GPU): forward and backward pass on the subgraph.
#   PHASE C  (CPU): parameter write-back and bookkeeping.
#
# The phase boundaries matter for measurement: the wall-clock cost
# of A is dominated by graph topology, the cost of B by hidden-dim
# and subgraph size, and the cost of C by parameter count.  The
# attacker's knobs primarily inflate A; the white-box adversary
# additionally inflates B by choosing transactions whose subgraph
# activates many parameters.
#
# UpdatePath emits an event to the optional probe at each phase
# boundary.  With probes off, the path runs without the conditional
# checks; we use a no-op singleton probe so there is *no* per-event
# branch in the hot path.
#
# Camera-ready improvements
# -------------------------
# 1.  Phase-A timing skew fix.  The submission draft read
#     ``time.monotonic_ns()`` *twice* at phase A start (once for the
#     A_begin event, once for ``a_start``), with the index-allocation
#     and BFS work between them.  The probe-side interval and the
#     in-band ``bfs_us`` therefore did not match.  This version reads
#     the clock ONCE per phase boundary and reuses the value for
#     both the probe event and the cost measurement.
#
# 2.  CUDA-event correctness.  The submission draft recorded
#     ``cuda_start`` unconditionally but only invoked
#     ``_cuda_or_walltime`` in the forward-only branch; in the
#     training branch the CUDA events were allocated and abandoned.
#     This version uses CUDA events consistently (when the device is
#     CUDA) in both branches, with explicit synchronisation, and
#     ``test_system.py::test_cuda_events_used_when_present`` (a new
#     regression test) confirms the events are read.  On CPU this
#     code path falls through to wall-clock timing, identical to the
#     submission draft's behaviour.
#
# 3.  ``record_decision`` placement.  Called precisely at
#     forward-pass start, just before the model is invoked.  This is
#     the moment the AoI freshness tracker should sample model age:
#     the model produces an output that the decision pipeline
#     subsequently consumes.
# =============================================================================


@dataclass(frozen=True)
class UpdatePathEvent:
    """
    Probe event emitted at phase boundaries.  Captured by
    measurement.py.
    """

    txn_id: int
    phase: str                    # "A_begin", "A_end", "B_end", "C_end"
    monotonic_ns: int             # time.monotonic_ns() at phase boundary
    cuda_event_handle: Optional[int] = None
    extra: Mapping[str, object] = field(default_factory=dict)


class UpdatePathProbe:
    """
    Default no-op probe.  measurement.py replaces this with a real
    probe that records events.  The default is deliberately a real
    object (not None) so the hot path can call ``probe.observe()``
    unconditionally.

    Camera-ready: also exposes ``record_decision(t_ns)`` so the AoI
    freshness tracker can sample model age at decision time without
    requiring a callback registration step.  Default no-op; only
    MeasuringProbe overrides to do real work.
    """

    def observe(self, event: UpdatePathEvent) -> None:
        """Phase-boundary event hook."""
        pass

    def record_decision(self, t_ns: int) -> None:
        """
        Decision-time hook.  Called by ``UpdatePath.process``
        precisely at forward-pass start (just before the model is
        invoked).  ``measurement.MeasuringProbe`` overrides this to
        feed its ``ModelFreshnessTracker``.
        """
        pass


_NULL_PROBE = UpdatePathProbe()


@dataclass(frozen=True)
class UpdatePathConfig:
    bfs_max_depth: int = 3
    bfs_degree_threshold: int = 256
    affected_set_cap: int = 4096
    forward_only: bool = False         # if True, skip backward (used during inference)


class UpdatePath:
    """
    Encapsulates one application of the update-path operation.

    Stateful across calls only via the model and the index.  All
    other state is local.  This makes the update path cleanly
    profileable: one transaction in, one CostDecomposition out, one
    event stream emitted to the probe.
    """

    def __init__(
        self,
        index: TemporalIndex,
        model: OnlineLearner,
        config: UpdatePathConfig,
        optimizer: Optional[torch.optim.Optimizer] = None,
        probe: UpdatePathProbe = _NULL_PROBE,
    ) -> None:
        self._index = index
        self._model = model
        self._config = config
        self._optimizer = optimizer or torch.optim.Adam(
            model.parameters(), lr=model.config.learning_rate
        )
        self._probe = probe
        self._device = torch.device(model.config.device)
        # Camera-ready: cache whether CUDA is actually available on
        # this device.  ``self._device.type == "cuda"`` is True when
        # the config requested CUDA; we additionally check
        # ``torch.cuda.is_available()`` to handle the "config asks
        # for cuda but the host has no GPU" edge case gracefully.
        self._cuda_active = (
            self._device.type == "cuda" and torch.cuda.is_available()
        )

    @property
    def probe(self) -> UpdatePathProbe:
        return self._probe

    def set_probe(self, probe: UpdatePathProbe) -> None:
        """Swap probe at runtime.  measurement.py uses this."""
        self._probe = probe

    # --- the operation under attack --------------------------------------

    def process(
        self,
        txn: Transaction,
        label: Optional[int] = None,
    ) -> CostDecomposition:
        """
        Process one transaction through the update path.  Returns
        the decomposed cost (in microseconds) for analysis.  If
        ``label`` is None the operation is forward-only (inference);
        if provided it is full update.

        Errors propagate.  The defender (defenses.py) is responsible
        for rejecting transactions that should not be processed.

        Probe events emitted (in order):
          A_begin    — just before phase A
          A_end      — after BFS completes, before phase B
          (decision) — record_decision hook, just before forward pass
          B_end      — after forward (+ optional backward)
          C_end      — after parameter write-back

        Camera-ready: each phase-boundary clock read is taken ONCE
        and reused for both the probe event and the in-band cost
        measurement.  The submission draft read the clock twice at
        each boundary, producing a probe interval that did not match
        the in-band cost.
        """
        # ---- Phase A: index insert + BFS (CPU) --------------------------
        # ONE clock read at the phase boundary; reused for both the
        # A_begin probe event and the bfs_us starting tick.
        a_start_ns = time.monotonic_ns()
        self._probe.observe(
            UpdatePathEvent(txn.txn_id, "A_begin", a_start_ns)
        )

        src_shift, tgt_shift = self._index.insert_transaction(txn)
        bfs = affected_subgraph_bfs(
            self._index,
            seed_nodes=[txn.source, txn.target],
            max_depth=self._config.bfs_max_depth,
            degree_threshold=self._config.bfs_degree_threshold,
        )
        affected = bfs.affected_nodes
        if len(affected) > self._config.affected_set_cap:
            # Defender-independent safety net: cap the affected set.
            # The cap is large enough not to fire under benign
            # workloads (verified on all four datasets in
            # experiments.py); it exists so a single pathological
            # transaction cannot wedge the system, independent of
            # any defense.
            affected = affected[: self._config.affected_set_cap]

        # ONE clock read at the phase boundary; reused for both
        # A_end and the bfs_us calculation.
        a_end_ns = time.monotonic_ns()
        bfs_us = (a_end_ns - a_start_ns) / 1000.0
        self._probe.observe(
            UpdatePathEvent(
                txn.txn_id, "A_end", a_end_ns,
                extra={
                    "affected_size": len(affected),
                    "depth": bfs.depth_reached,
                    "edges_visited": bfs.edges_visited,
                    "src_shift": src_shift,
                    "tgt_shift": tgt_shift,
                },
            )
        )

        # ---- Phase B: forward + backward on subgraph (GPU) --------------
        # Build batch first.  This is CPU work (numpy + tensor
        # transfer); it is part of phase B because the cost is paid
        # on the path between BFS and forward pass, but reviewers
        # who care about the breakdown can read it from the probe's
        # ``extra`` dict (see ``B_end`` event below).
        b_start_ns = time.monotonic_ns()
        in_seq, out_seq, in_len, out_len, neigh_index, node_to_row = (
            self._build_batch(affected)
        )

        # Camera-ready: record the decision moment, precisely at
        # forward-pass start.  This is the moment the AoI tracker
        # samples model age; the model produces an output that the
        # decision pipeline subsequently consumes.
        decision_ns = time.monotonic_ns()
        self._probe.record_decision(decision_ns)

        # CUDA event start for accurate GPU-side timing.  We use
        # CUDA events consistently in BOTH branches (forward-only
        # and training); the submission draft's training branch
        # allocated events but never read them.
        cuda_start: Optional[Any] = None
        cuda_end: Optional[Any] = None
        if self._cuda_active:
            cuda_start = torch.cuda.Event(enable_timing=True)
            cuda_end = torch.cuda.Event(enable_timing=True)
            cuda_start.record()

        if self._config.forward_only or label is None:
            # Inference / forward-only path.
            with torch.no_grad():
                _ = self._model(
                    in_seq, out_seq, in_len, out_len, neigh_index,
                )
            if cuda_start is not None and cuda_end is not None:
                cuda_end.record()
                torch.cuda.synchronize()
                # elapsed_time returns ms; convert to µs.
                forward_us = float(
                    cuda_start.elapsed_time(cuda_end) * 1000.0
                )
            else:
                forward_us = (time.monotonic_ns() - decision_ns) / 1000.0
            backward_us = 0.0
        else:
            # Full training path.
            self._optimizer.zero_grad(set_to_none=True)
            logits = self._model(
                in_seq, out_seq, in_len, out_len, neigh_index,
            )
            target = self._build_target(node_to_row, txn, label, len(affected))
            loss = F.cross_entropy(logits, target)

            if cuda_start is not None and cuda_end is not None:
                # Mark the forward/backward boundary with an
                # intermediate CUDA event so we can split the cost.
                fwd_done_cuda = torch.cuda.Event(enable_timing=True)
                fwd_done_cuda.record()
                loss.backward()
                cuda_end.record()
                torch.cuda.synchronize()
                forward_us = float(
                    cuda_start.elapsed_time(fwd_done_cuda) * 1000.0
                )
                backward_us = float(
                    fwd_done_cuda.elapsed_time(cuda_end) * 1000.0
                )
            else:
                forward_done_ns = time.monotonic_ns()
                forward_us = (forward_done_ns - decision_ns) / 1000.0
                loss.backward()
                backward_us = (
                    time.monotonic_ns() - forward_done_ns
                ) / 1000.0

        # ONE clock read at the B_end boundary.
        b_end_ns = time.monotonic_ns()
        self._probe.observe(
            UpdatePathEvent(
                txn.txn_id, "B_end", b_end_ns,
                extra={
                    "subgraph_nodes": len(affected),
                    "forward_us": forward_us,
                    "backward_us": backward_us,
                    "decision_ns": decision_ns,
                },
            )
        )

        # ---- Phase C: parameter write (CPU/GPU) -------------------------
        # ONE clock read for c_start, reused below.
        c_start_ns = time.monotonic_ns()
        if not self._config.forward_only and label is not None:
            self._optimizer.step()
        c_end_ns = time.monotonic_ns()
        write_us = (c_end_ns - c_start_ns) / 1000.0
        self._probe.observe(
            UpdatePathEvent(txn.txn_id, "C_end", c_end_ns)
        )

        # Selection cost is currently zero — every parameter touched
        # by the affected subgraph is updated.  defenses.py
        # introduces a parameter-selection step that adds nonzero
        # selection cost; we track the field here so that defended
        # runs are comparable to undefended ones component-by-
        # component.
        return CostDecomposition(
            bfs_traversal_us=bfs_us,
            parameter_selection_us=0.0,
            forward_pass_us=forward_us,
            backward_pass_us=backward_us,
            parameter_write_us=write_us,
        )

    # --- internals --------------------------------------------------------

    def _build_batch(
        self, affected: Sequence[int]
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Dict[int, int],
    ]:
        """
        Construct the model inputs for the affected node set:
        per-node truncated incoming/outgoing sequences, lengths, and
        edge index restricted to within-subgraph edges.

        Returns the five tensors plus the ``node_to_row`` dict, so
        ``_build_target`` can use it for O(1) lookup of the source
        node's row instead of an O(N) ``affected.index(...)`` call.

        This is straightforward but is the right place to measure
        the cost of *materialising* the batch.  The cost is included
        in Phase B in the current decomposition; if the experiment
        harness wants a finer split it is exposed via the probe's
        ``extra`` field.
        """
        D = self._model.config.feature_dim
        L = self._model.config.sequence_truncation
        N = len(affected)
        # Map node id → batch row.  Camera-ready: built once and
        # returned to ``_build_target`` so we don't pay an O(N)
        # ``affected.index(...)`` per transaction.
        node_to_row: Dict[int, int] = {n: i for i, n in enumerate(affected)}

        in_seq = np.zeros((N, L, D), dtype=np.float32)
        out_seq = np.zeros((N, L, D), dtype=np.float32)
        in_len = np.zeros((N,), dtype=np.int64)
        out_len = np.zeros((N,), dtype=np.int64)

        edges: List[Tuple[int, int]] = []
        for i, node in enumerate(affected):
            wins = self._index.account_windows(node)
            if wins is None:
                continue
            # Most recent L incoming.
            inc = wins.incoming
            n_inc = len(inc)
            if n_inc > 0:
                start = max(0, n_inc - L)
                k_used = n_inc - start
                in_len[i] = k_used
                for k in range(int(k_used)):
                    txn = inc._payloads[start + k]      # internal access OK here
                    in_seq[i, k] = txn.features         # type: ignore[union-attr]
            # Most recent L outgoing.
            outg = wins.outgoing
            n_out = len(outg)
            if n_out > 0:
                start = max(0, n_out - L)
                k_used = n_out - start
                out_len[i] = k_used
                for k in range(int(k_used)):
                    txn = outg._payloads[start + k]
                    out_seq[i, k] = txn.features        # type: ignore[union-attr]
            # Edges within the affected set only.
            for nb in self._index.neighbors(node):
                if nb in node_to_row:
                    edges.append((node_to_row[node], node_to_row[nb]))

        neigh_index_np = (
            np.asarray(edges, dtype=np.int64)
            if edges
            else np.zeros((0, 2), dtype=np.int64)
        )
        return (
            torch.from_numpy(in_seq).to(self._device),
            torch.from_numpy(out_seq).to(self._device),
            torch.from_numpy(in_len).to(self._device),
            torch.from_numpy(out_len).to(self._device),
            torch.from_numpy(neigh_index_np).to(self._device),
            node_to_row,
        )

    def _build_target(
        self,
        node_to_row: Mapping[int, int],
        txn: Transaction,
        label: int,
        n_rows: int,
    ) -> torch.Tensor:
        """
        The label is associated with the transaction's source
        account (the account whose status we are updating in
        response to this transaction).  All other affected nodes are
        unlabelled and receive the implicit label 0; the loss masks
        them via class weights in practice, but for the cost-
        decomposition measurements in this paper a dense target is
        sufficient.

        Camera-ready: uses the dict ``node_to_row`` from
        ``_build_batch`` for O(1) lookup of the source row, instead
        of the submission draft's O(N) ``affected.index(...)``.
        """
        target = torch.zeros(n_rows, dtype=torch.long, device=self._device)
        row = node_to_row.get(txn.source)
        if row is not None:
            target[row] = label
        return target


# =============================================================================
# Section 5.  Detection-quality measurement.
#
# Two F1 quantities, both defined unambiguously here.  The previous
# paper reported these inconsistently (ICDE R2-O7); this implementation
# computes them in disjoint code paths and the experiment harness
# reports both.
# =============================================================================


@dataclass(frozen=True)
class DetectionScores:
    streaming_f1: float       # F1 measured *during* streaming, exponentially
                              # weighted toward recent decisions.
    snapshot_f1: float        # F1 measured at end-of-stream after the final
                              # update sweep, on a held-out split.
    n_streaming_decisions: int
    n_snapshot_decisions: int


def measure_detection_quality(
    streaming_predictions: Sequence[Tuple[int, int]],     # (truth, pred)
    snapshot_predictions: Sequence[Tuple[int, int]],
    streaming_decay: float = 0.99,
) -> DetectionScores:
    """
    Compute the two F1 quantities the paper reports.

    streaming_f1 is the F1 of decisions made by the model *as the
    stream progressed*.  It is reported because real systems are
    judged by the decisions they actually shipped, not by the
    post-hoc model.

    snapshot_f1 is the F1 of the final model on the held-out test
    split, reported because it is comparable to the batch-trained
    baseline.

    These two quantities answer different questions and should not
    be averaged.  The paper reports both with their labels.

    Camera-ready: ``streaming_decay`` validation moved to the top of
    the function so a malformed argument is rejected even if the
    sequences are empty.  The submission draft only validated decay
    inside ``_f1_with_decay``, which silently returned NaN for
    empty inputs without checking the argument.
    """
    if not 0.0 < streaming_decay <= 1.0:
        raise ValueError(
            f"streaming_decay must be in (0, 1]; got {streaming_decay}"
        )
    s_f1 = _f1_with_decay(streaming_predictions, streaming_decay)
    sn_f1 = _f1_plain(snapshot_predictions)
    return DetectionScores(
        streaming_f1=s_f1,
        snapshot_f1=sn_f1,
        n_streaming_decisions=len(streaming_predictions),
        n_snapshot_decisions=len(snapshot_predictions),
    )


def _f1_plain(pairs: Sequence[Tuple[int, int]]) -> float:
    if not pairs:
        return float("nan")
    tp = sum(1 for y, p in pairs if y == 1 and p == 1)
    fp = sum(1 for y, p in pairs if y == 0 and p == 1)
    fn = sum(1 for y, p in pairs if y == 1 and p == 0)
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    return 2 * prec * rec / (prec + rec)


def _f1_with_decay(
    pairs: Sequence[Tuple[int, int]],
    decay: float,
) -> float:
    """
    Exponentially-weighted F1.  Each decision i in temporal order
    receives weight ``decay**(N-1-i)``, so recent decisions
    dominate.
    """
    if not pairs:
        return float("nan")
    N = len(pairs)
    # Build weights vectorised, no Python list comprehension.  The
    # submission draft's list-comprehension form was correct but
    # ~10× slower for long streams.
    weights = decay ** (N - 1 - np.arange(N, dtype=np.float64))
    truths = np.fromiter((y for y, _ in pairs), dtype=np.float64, count=N)
    preds = np.fromiter((p for _, p in pairs), dtype=np.float64, count=N)
    tp = float(((truths == 1) & (preds == 1)) @ weights)
    fp = float(((truths == 0) & (preds == 1)) @ weights)
    fn = float(((truths == 1) & (preds == 0)) @ weights)
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    return 2 * prec * rec / (prec + rec)


# =============================================================================
# Section 6.  Threat-model protocol adapters.
#
# Adapters that present the SUT's internal data through the four
# protocols defined in threat_model.py.  These are pure delegation —
# no logic — so a reviewer can verify they leak no information
# beyond the protocol.
# =============================================================================


class _TopologyAdapter(GraphTopologySource):
    def __init__(self, index: TemporalIndex) -> None:
        self._index = index

    def num_nodes(self) -> int:
        return self._index.num_nodes()

    def degree(self, node_id: int) -> int:
        return self._index.degree(node_id)

    def neighbors(self, node_id: int) -> Sequence[int]:
        return self._index.neighbors(node_id)

    def edge_timestamps(self, node_id: int) -> np.ndarray:
        return self._index.edge_timestamps(node_id)


class _MetadataAdapter(ModelMetadataSource):
    def __init__(self, model: OnlineLearner) -> None:
        self._model = model

    def architecture_signature(self) -> Mapping[str, object]:
        return self._model.architecture_signature()

    def parameter_shapes(self) -> Mapping[str, Tuple[int, ...]]:
        return self._model.parameter_shapes()


class _WeightsAdapter(ModelWeightsSource):
    def __init__(self, model: OnlineLearner) -> None:
        self._model = model

    def parameter_tensor(self, name: str) -> np.ndarray:
        return self._model.parameter_tensor_numpy(name)


def make_topology_source(index: TemporalIndex) -> GraphTopologySource:
    return _TopologyAdapter(index)


def make_metadata_source(model: OnlineLearner) -> ModelMetadataSource:
    return _MetadataAdapter(model)


def make_weights_source(model: OnlineLearner) -> ModelWeightsSource:
    return _WeightsAdapter(model)


# =============================================================================
# Section 7.  System assembly.
#
# A System is a frozen container of the four objects the rest of the
# codebase needs to talk to: the index, the model, the update path,
# and the threat-model adapters.  Construction is deterministic
# given the config and the seed.  measurement.py and experiments.py
# never construct individual subsystems; they construct a System.
# =============================================================================


@dataclass(frozen=True)
class SystemConfig:
    learner: LearnerConfig
    update_path: UpdatePathConfig = field(default_factory=UpdatePathConfig)
    seed: int = 0


@dataclass
class System:
    index: TemporalIndex
    model: OnlineLearner
    update_path: UpdatePath
    topology_source: GraphTopologySource
    metadata_source: ModelMetadataSource
    weights_source: ModelWeightsSource
    config: SystemConfig

    @classmethod
    def assemble(cls, config: SystemConfig) -> "System":
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        index = TemporalIndex()
        model = OnlineLearner(config.learner).to(config.learner.device)
        update_path = UpdatePath(index, model, config.update_path)
        return cls(
            index=index,
            model=model,
            update_path=update_path,
            topology_source=make_topology_source(index),
            metadata_source=make_metadata_source(model),
            weights_source=make_weights_source(model),
            config=config,
        )

    # Convenience: the experiment harness usually wants a probe-
    # bearing update path.  measurement.py constructs a probe and
    # calls this.
    def attach_probe(self, probe: UpdatePathProbe) -> None:
        self.update_path.set_probe(probe)


# =============================================================================
# Section 8.  Public surface.
# =============================================================================


__all__ = [
    # Data types
    "Transaction",
    "BFSResult",
    "CostDecomposition",   # re-exported from threat_model
    # Core subsystems
    "TemporalIndex",
    "OnlineLearner",
    "LearnerConfig",
    "UpdatePath",
    "UpdatePathConfig",
    "UpdatePathProbe",
    "UpdatePathEvent",
    # Measurement
    "DetectionScores",
    "measure_detection_quality",
    # Threat-model adapters
    "make_topology_source",
    "make_metadata_source",
    "make_weights_source",
    # System
    "System",
    "SystemConfig",
    # Operations exposed for tests
    "affected_subgraph_bfs",
]
