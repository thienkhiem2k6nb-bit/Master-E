"""N-best edit-level voting (consensus filtering).

Setting. Beam search returns N hypotheses h_1..h_N for a source s with scores z_1..z_N (length-
normalised log-probabilities). Each hypothesis is aligned to the source and decomposed into a set of
edits E(h_k) (see `alignment.extract_edits`). An edit is identified by its source span and its
correction string.

Edit-level voting (this module, `consensus_correction`). Hypothesis weights
    w_k = exp(z_k / T) / sum_j exp(z_j / T)          (T = temperature; T -> inf gives uniform weights)
Support of an edit e:
    supp(e) = sum_k w_k * 1[e in E(h_k)]
An edit is kept when supp(e) >= tau. Overlapping kept edits are resolved greedily by support
(`edits.resolve_overlaps`). The output is the source with the kept edits applied. This is the
single-system analogue of edit-based system combination (Kantor et al. 2019 "Learning to combine
GEC"; Qorib, Na & Ng 2022 "ESC") where the "systems" are the beams of one model, and of the
confidence thresholds used by GECToR (Omelianchuk et al. 2020): tau trades recall for precision,
which F0.5 rewards.

Complexity. For N hypotheses of length <= m and source length n: N alignments O(N*n*m), voting
O(N*k) for k distinct edits, overlap resolution O(k^2).
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Sequence

from .alignment import as_tokens, extract_edits
from .edits import Edit, apply_edits, resolve_overlaps

EditExtractor = Callable[[Sequence[str], Sequence[str]], list[Edit]]


# -------------------------------------------------------------------------------------------- weights


def hypothesis_weights(scores: Sequence[float], temperature: float = 1.0, mode: str = "softmax") -> list[float]:
    """Normalised weights for the n-best list.

    mode = "softmax": w_k ∝ exp(z_k / T)   (z are log-probs, so this is a tempered posterior)
    mode = "uniform": w_k = 1/N
    mode = "rank":    w_k ∝ 1/(rank_k)     (Borda-like, insensitive to score scale)
    """
    n = len(scores)
    if n == 0:
        return []
    if mode == "uniform":
        return [1.0 / n] * n
    if mode == "rank":
        order = sorted(range(n), key=lambda k: -scores[k])
        raw = [0.0] * n
        for r, k in enumerate(order):
            raw[k] = 1.0 / (r + 1)
        z = sum(raw)
        return [x / z for x in raw]
    if mode == "softmax":
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        mx = max(scores)
        ex = [math.exp((s - mx) / temperature) for s in scores]
        z = sum(ex)
        return [x / z for x in ex]
    raise ValueError(f"unknown weight mode {mode}")


# --------------------------------------------------------------------------------------------- voting


@dataclass
class VoteResult:
    tokens: list[str]                      # corrected source tokens
    kept: list[Edit]                       # edits applied
    support: dict[tuple, float]            # edit key -> support
    candidates: dict[tuple, Edit]          # edit key -> edit object (first occurrence)
    n_candidates: int


def edit_support(source: Sequence[str], hypotheses: Sequence[Sequence[str]], weights: Sequence[float],
                 extractor: EditExtractor = extract_edits) -> tuple[dict[tuple, float], dict[tuple, Edit]]:
    """Weighted support of every distinct edit across the n-best list."""
    support: dict[tuple, float] = defaultdict(float)
    candidates: dict[tuple, Edit] = {}
    for hyp, w in zip(hypotheses, weights):
        seen: set[tuple] = set()
        for e in extractor(source, hyp):
            if e.key in seen:            # an edit counts once per hypothesis
                continue
            seen.add(e.key)
            support[e.key] += w
            candidates.setdefault(e.key, e)
    return dict(support), candidates


def consensus_correction(source: Sequence[str], hypotheses: Sequence[Sequence[str]], scores: Sequence[float],
                         tau: float = 0.5, temperature: float = 1.0, weight_mode: str = "softmax",
                         restrict_to_top1: bool = False,
                         extractor: EditExtractor = extract_edits) -> VoteResult:
    """Keep edits whose weighted n-best support reaches `tau`, resolve overlaps, apply to the source.

    Two variants:
      * voting   (restrict_to_top1=False): every edit proposed by any hypothesis is a candidate, so
                 the output can combine edits from different beams;
      * filtering (restrict_to_top1=True): only the edits of the best hypothesis are candidates and
                 the other beams act as a jury that can veto them (output is a subset of the top-1 edits).

    tau = 0 with a single hypothesis reproduces that hypothesis (up to alignment/merging), so the
    top-1 beam output is the special case N=1.
    """
    if not 0.0 <= tau <= 1.0:
        raise ValueError("tau must be in [0, 1]")
    if not hypotheses:
        raise ValueError("at least one hypothesis is required")
    if len(scores) != len(hypotheses):     # zip() below would silently drop beams
        raise ValueError(f"got {len(hypotheses)} hypotheses but {len(scores)} scores")
    weights = hypothesis_weights(scores, temperature, weight_mode)
    support, candidates = edit_support(source, hypotheses, weights, extractor)
    if restrict_to_top1:
        top_keys = {e.key for e in extractor(source, hypotheses[0])}
        support = {k: v for k, v in support.items() if k in top_keys}
        candidates = {k: v for k, v in candidates.items() if k in top_keys}
    scored = [(candidates[k], s) for k, s in support.items() if s >= tau - 1e-12]
    kept = resolve_overlaps(scored)
    plain = [t.text for t in as_tokens(source)]          # source may carry Token features
    return VoteResult(apply_edits(plain, kept), kept, support, candidates, len(candidates))


