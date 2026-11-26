"""Edit spans and operations on them: apply to a source, resolve overlaps, read/write M2."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True, order=True)
class Edit:
    """Replace source tokens [start, end) by `cor` tokens. start == end denotes an insertion."""

    start: int
    end: int
    orig: tuple[str, ...]
    cor: tuple[str, ...]
    etype: str = "UNK"   # error type label, optional (e.g. "R:VERB:SVA")

    @property
    def key(self) -> tuple[int, int, tuple[str, ...]]:
        """Identity used for voting: same span and same correction string."""
        return (self.start, self.end, self.cor)

    @property
    def op(self) -> str:
        if not self.orig:
            return "M"   # missing -> insertion
        if not self.cor:
            return "U"   # unnecessary -> deletion
        return "R"       # replacement

    def overlaps(self, other: "Edit") -> bool:
        if self.start == self.end and other.start == other.end:
            return self.start == other.start
        if self.start == self.end:
            return other.start <= self.start < other.end
        if other.start == other.end:
            return self.start <= other.start < self.end
        return self.start < other.end and other.start < self.end

    def m2_line(self, annotator: int = 0) -> str:
        cor = " ".join(self.cor) if self.cor else ""
        return f"A {self.start} {self.end}|||{self.etype}|||{cor}|||REQUIRED|||-NONE-|||{annotator}"


def apply_edits(source: Sequence[str], edits: Iterable[Edit]) -> list[str]:
    """Apply non-overlapping edits (any order) to the token list and return the new token list."""
    out = list(source)
    for e in sorted(edits, key=lambda x: (x.start, x.end), reverse=True):
        out[e.start:e.end] = list(e.cor)
    return out


def resolve_overlaps(scored: Sequence[tuple[Edit, float]]) -> list[Edit]:
    """Greedy selection of a non-overlapping edit set, preferring higher scores (ties: earlier, shorter).

    This is the classic interval-scheduling greedy heuristic; it is O(k log k + k^2) for k candidate edits.
    """
    chosen: list[Edit] = []
    for e, _ in sorted(scored, key=lambda es: (-es[1], es[0].start, es[0].end - es[0].start)):
        if not any(e.overlaps(c) for c in chosen):
            chosen.append(e)
    return sorted(chosen, key=lambda x: (x.start, x.end))


# ------------------------------------------------------------------------------------------- M2 I/O


@dataclass
class M2Sentence:
    source: list[str]
    edits_by_annotator: dict[int, list[Edit]]


def read_m2(path: str | Path) -> list[M2Sentence]:
    sentences: list[M2Sentence] = []
    cur: M2Sentence | None = None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("S "):
            cur = M2Sentence(line[2:].split(), {})
            sentences.append(cur)
        elif line.startswith("A ") and cur is not None:
            parts = line[2:].split("|||")
            s, e = (int(x) for x in parts[0].split())
            etype, cor, annot = parts[1], parts[2], int(parts[-1])
            cur.edits_by_annotator.setdefault(annot, [])
            if s == -1 or etype.lower() == "noop":
                continue
            cor_tokens = tuple(cor.split()) if cor and cor != "-NONE-" else tuple()
            cur.edits_by_annotator[annot].append(Edit(s, e, tuple(cur.source[s:e]), cor_tokens, etype))
    return sentences


def write_m2(path: str | Path, sentences: Iterable[tuple[Sequence[str], Sequence[Edit]]]) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        for source, edits in sentences:
            f.write("S " + " ".join(source) + "\n")
            if not edits:
                f.write("A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0\n")
            for e in edits:
                f.write(e.m2_line() + "\n")
            f.write("\n")
