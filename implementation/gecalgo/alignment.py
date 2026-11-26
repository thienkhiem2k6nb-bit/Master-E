"""Token-level sequence alignment by dynamic programming.

Algorithms implemented here:

* `levenshtein_matrix` - Wagner & Fischer (1974) O(n*m) dynamic programme for the Levenshtein (1966)
  edit distance with unit costs.
* `align` - Damerau-Levenshtein alignment in the restricted (optimal string alignment) form of
  Lowrance & Wagner (1975), extended with a *linguistically-enhanced substitution cost* in the style of
  Felice, Bryant & Briscoe (2016, COLING) and ERRANT (Bryant et al. 2017):

      sub_cost(a, b) = 0                              if a == b (exact token)
                     = lemma_cost + pos_cost + char_cost   otherwise, where
      lemma_cost = 0 if lemma(a) == lemma(b) else 0.499
      pos_cost   = 0 if pos(a) == pos(b) else 0.25 (0.5 if neither token is content/function-compatible)
      char_cost  = 1 - Ratcliff/Obershelp similarity(a, b)      in [0, 1]

  so a substitution never costs less than 0 and at most ~1.999, while insert/delete cost 1 and a
  transposition of adjacent tokens costs 1. The costs are the ones published for ERRANT's aligner
  (lemma 0.499, POS 0.25/0.5, char similarity), reproduced here as our own implementation.
  Linguistic features (lemma, POS) are optional: when no `Token` features are supplied the aligner
  uses the feature-free cost 0.5 + char_cost (range 0.5..1.5), which keeps the module independent of
  spaCy while still preferring insert+delete (2.0) over two unrelated substitutions (3.0).
* `backtrace` - recovers the operation sequence (M = match, S = substitute, I = insert, D = delete,
  T = transpose) from the cost matrix.
* `merge_ops` - groups consecutive non-match operations into edit spans (a simplified version of the
  ERRANT merging rules: adjacent operations are merged, with one split rule that separates an
  insertion/deletion of punctuation from a neighbouring substitution).

Complexity: time O(n*m*k) where k is the cost of one substitution-cost evaluation (char similarity is
O(|a|*|b|) on token lengths), space O(n*m) for the matrix plus O(n*m) for the backpointers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Sequence

from .edits import Edit

# ---------------------------------------------------------------------------------------------- tokens


@dataclass(frozen=True)
class Token:
    """A token with optional linguistic features. `text` is what is compared and emitted."""

    text: str
    lemma: str | None = None
    pos: str | None = None

    @property
    def lower(self) -> str:
        return self.text.lower()


def as_tokens(seq: Sequence[str] | Sequence[Token]) -> list[Token]:
    return [t if isinstance(t, Token) else Token(str(t)) for t in seq]


# ---------------------------------------------------------------------------------------------- costs

LEMMA_COST = 0.499
POS_COST_SAME_CLASS = 0.25
POS_COST_DIFF_CLASS = 0.5
_CONTENT_POS = {"ADJ", "ADV", "NOUN", "PROPN", "VERB", "NUM"}


def char_similarity(a: str, b: str) -> float:
    """Ratcliff/Obershelp similarity in [0, 1] (difflib.SequenceMatcher.ratio)."""
    if a == b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def substitution_cost(a: Token, b: Token) -> float:
    """Linguistically-enhanced substitution cost (0 for identical tokens)."""
    if a.text == b.text:
        return 0.0
    if a.lower == b.lower:          # case-only difference: cheap, still an edit (ORTH)
        return 0.1
    char_cost = 1.0 - char_similarity(a.lower, b.lower)
    has_features = a.lemma is not None and b.lemma is not None and a.pos is not None and b.pos is not None
    if not has_features:
        # Feature-free fallback. A constant 0.5 keeps the substitution of two unrelated tokens
        # (cost 1.5 each, 3.0 for a pair) more expensive than an insertion plus a deletion (2.0), as the
        # lemma/POS terms do in the linguistic variant; related tokens ("go"/"goes") stay cheap.
        return 0.5 + char_cost
    cost = char_cost
    cost += 0.0 if a.lemma.lower() == b.lemma.lower() else LEMMA_COST
    if a.pos == b.pos:
        cost += 0.0
    elif (a.pos in _CONTENT_POS) == (b.pos in _CONTENT_POS):
        cost += POS_COST_SAME_CLASS
    else:
        cost += POS_COST_DIFF_CLASS
    return cost


# --------------------------------------------------------------------------------------- Levenshtein


def levenshtein_matrix(a: Sequence[str], b: Sequence[str]) -> list[list[int]]:
    """Wagner-Fischer dynamic programme with unit costs. Returns the (n+1) x (m+1) matrix."""
    n, m = len(a), len(b)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        d[i][0] = i
    for j in range(1, m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + sub)
    return d


def levenshtein_distance(a: Sequence[str], b: Sequence[str]) -> int:
    return levenshtein_matrix(a, b)[len(a)][len(b)]


# ------------------------------------------------------------------------- Damerau-Levenshtein alignment


@dataclass
class Alignment:
    ops: list[tuple[str, int, int]]      # (op, i, j): op in {M,S,I,D,T}; i/j are 0-based source/target indexes
    cost: float
    matrix: list[list[float]] = field(default_factory=list, repr=False)


def align(source: Sequence[str] | Sequence[Token], target: Sequence[str] | Sequence[Token]) -> Alignment:
    """Restricted Damerau-Levenshtein (optimal string alignment) with linguistic substitution costs.

    Operation costs: insert 1, delete 1, substitute `substitution_cost`, adjacent transposition 1
    (only when the two tokens are an exact swap, as in ERRANT).
    """
    src, tgt = as_tokens(source), as_tokens(target)
    n, m = len(src), len(tgt)
    INF = float("inf")
    cost = [[INF] * (m + 1) for _ in range(n + 1)]
    back: list[list[str]] = [[""] * (m + 1) for _ in range(n + 1)]
    cost[0][0] = 0.0
    for i in range(1, n + 1):
        cost[i][0], back[i][0] = float(i), "D"
    for j in range(1, m + 1):
        cost[0][j], back[0][j] = float(j), "I"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            s = substitution_cost(src[i - 1], tgt[j - 1])
            best, op = cost[i - 1][j - 1] + s, ("M" if s == 0.0 else "S")
            if cost[i - 1][j] + 1.0 < best:
                best, op = cost[i - 1][j] + 1.0, "D"
            if cost[i][j - 1] + 1.0 < best:
                best, op = cost[i][j - 1] + 1.0, "I"
            if (i > 1 and j > 1 and src[i - 1].lower == tgt[j - 2].lower and src[i - 2].lower == tgt[j - 1].lower
                    and src[i - 1].lower != src[i - 2].lower):
                if cost[i - 2][j - 2] + 1.0 < best:
                    best, op = cost[i - 2][j - 2] + 1.0, "T"
            cost[i][j], back[i][j] = best, op
    ops = backtrace(back, n, m)
    return Alignment(ops, cost[n][m], cost)


def backtrace(back: list[list[str]], n: int, m: int) -> list[tuple[str, int, int]]:
    ops: list[tuple[str, int, int]] = []
    i, j = n, m
    while i > 0 or j > 0:
        op = back[i][j]
        if op in ("M", "S"):
            ops.append((op, i - 1, j - 1)); i, j = i - 1, j - 1
        elif op == "D":
            ops.append((op, i - 1, j)); i -= 1
        elif op == "I":
            ops.append((op, i, j - 1)); j -= 1
        elif op == "T":
            ops.append((op, i - 2, j - 2)); i, j = i - 2, j - 2
        else:  # pragma: no cover - defensive
            raise RuntimeError("corrupt backpointer table")
    ops.reverse()
    return ops


# ------------------------------------------------------------------------------------ merging to edits

_PUNCT = set(".,;:!?\"'()[]{}-")


def merge_ops(ops: list[tuple[str, int, int]], source: Sequence[str] | Sequence[Token],
              target: Sequence[str] | Sequence[Token]) -> list[Edit]:
    """Group consecutive non-match operations into `Edit` spans.

    Rules (simplified from ERRANT's `merge` heuristics):
      1. Consecutive non-M operations form one candidate span.
      2. A transposition T is always its own edit covering two source tokens.
      3. Inside a span, a boundary is inserted between a pure punctuation insertion/deletion and a
         following/preceding word-level operation, so that "I like it ." -> "I liked it !" yields two edits.
    """
    src, tgt = [t.text for t in as_tokens(source)], [t.text for t in as_tokens(target)]
    edits: list[Edit] = []
    group: list[tuple[str, int, int]] = []

    def flush() -> None:
        if not group:
            return
        for part in _split_group(group, src, tgt):
            s_lo = min(i for op, i, _ in part if op != "I") if any(op != "I" for op, _, _ in part) else part[0][1]
            s_hi = max(i + (2 if op == "T" else 1) for op, i, _ in part if op != "I") if any(op != "I" for op, _, _ in part) else part[0][1]
            t_lo = min(j for op, _, j in part if op != "D") if any(op != "D" for op, _, _ in part) else part[0][2]
            t_hi = max(j + (2 if op == "T" else 1) for op, _, j in part if op != "D") if any(op != "D" for op, _, _ in part) else part[0][2]
            edits.append(Edit(s_lo, s_hi, tuple(src[s_lo:s_hi]), tuple(tgt[t_lo:t_hi])))
        group.clear()

    for op in ops:
        if op[0] == "M":
            flush()
        elif op[0] == "T":
            flush()
            group.append(op)
            flush()
        else:
            group.append(op)
    flush()
    return edits


def _split_group(group: list[tuple[str, int, int]], src: list[str], tgt: list[str]):
    """Split a run of operations at punctuation-only insertions/deletions adjacent to word operations."""
    if len(group) == 1:
        return [group]
    parts, cur = [], [group[0]]
    for prev, op in zip(group, group[1:]):
        if _is_punct_op(prev, src, tgt) != _is_punct_op(op, src, tgt):
            parts.append(cur); cur = [op]
        else:
            cur.append(op)
    parts.append(cur)
    return parts


def _is_punct_op(op: tuple[str, int, int], src: list[str], tgt: list[str]) -> bool:
    kind, i, j = op
    if kind == "I":
        return tgt[j] in _PUNCT
    if kind == "D":
        return src[i] in _PUNCT
    if kind == "S":
        return src[i] in _PUNCT and tgt[j] in _PUNCT
    return False


def extract_edits(source: Sequence[str] | Sequence[Token], target: Sequence[str] | Sequence[Token]) -> list[Edit]:
    """Convenience: align then merge. Returns edits sorted by source position."""
    al = align(source, target)
    return merge_ops(al.ops, source, target)
