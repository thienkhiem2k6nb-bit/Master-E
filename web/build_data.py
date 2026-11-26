#!/usr/bin/env python3
"""Precompute everything the browser needs, so the web page only does the cheap part.

The expensive steps (spaCy tokenisation, lemma/POS features, Damerau-Levenshtein alignment, edit
merging) do not depend on tau, temperature or mode, so they run once here. The page then performs
voting / filtering / overlap resolution / edit application live, which is O(edits) per sentence.

    .venv/bin/python web/build_data.py            # all three datasets
    .venv/bin/python web/build_data.py --set jfleg

Writes web/data/<set>.js (loaded with a plain <script> tag, so the page also works from file://).
Each file also carries the historical scores and a hash of every historical output file, so the page
can show the official score of a configuration and prove its own output matches the original one.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "implementation"))

from common import DATA, RESULTS, featurize, read_lines, tokenize  # noqa: E402
from gecalgo.alignment import extract_edits  # noqa: E402

SETS = ["bea19dev", "conll14", "jfleg"]
OUT = Path(__file__).resolve().parent / "data"


def fnv1a(text: str) -> str:
    """32-bit FNV-1a over UTF-16 code units; web/voting.js computes the identical hash."""
    h = 0x811C9DC5
    data = text.encode("utf-16-le")
    for k in range(0, len(data), 2):
        h ^= data[k] | (data[k + 1] << 8)
        h = (h * 0x01000193) & 0xFFFFFFFF
    return f"{h:08x}"


def build(name: str) -> Path:
    nbest = RESULTS / "nbest" / f"{name}__coedit__b4n4.jsonl"
    records = sorted((json.loads(s) for s in nbest.read_text(encoding="utf-8").splitlines() if s.strip()),
                     key=lambda r: r["id"])
    sources = read_lines(DATA / name / "source.txt")
    if len(records) != len(sources) or any(r["id"] != i for i, r in enumerate(records)):
        raise ValueError(f"{name}: n-best/source mismatch or missing IDs")

    # Same preprocessing as experiments/run_voting_filtering.py (spaCy tokens + linguistic features).
    htokens = [tokenize(r["hyps"]) for r in records]
    sf = featurize([s.split() for s in sources])
    flat = featurize([h for hs in htokens for h in hs])

    sentences, i = [], 0
    for idx, (rec, hs) in enumerate(zip(records, htokens)):
        hf = flat[i:i + len(hs)]
        i += len(hs)
        beams = []
        for text, feats in zip(rec["hyps"], hf):
            edits = extract_edits(sf[idx], feats)
            beams.append([text, [[e.start, e.end, list(e.cor)] for e in edits]])
        sentences.append({"src": sources[idx].split(), "top1": hs[0], "scores": rec["scores"], "beams": beams})

    hist_dir = RESULTS / "outputs" / name
    hashes = {}
    for p in sorted(hist_dir.glob("coedit__*.txt")):   # [hash, line count]; a wrong count flags a damaged file
        text = p.read_text(encoding="utf-8")
        hashes[p.stem] = [fnv1a(text), len(text.splitlines())]
    scores_path = RESULTS / "scores" / f"{name}.json"
    scores = json.loads(scores_path.read_text(encoding="utf-8")) if scores_path.exists() else {}

    payload = {"name": name, "sentences": sentences, "historical_hashes": hashes, "scores": scores}
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.js"
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    path.write_text(f"window.NBEST_DATA = window.NBEST_DATA || {{}};\nwindow.NBEST_DATA[{json.dumps(name)}] = {body};\n",
                    encoding="utf-8")
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=SETS, nargs="+", default=SETS)
    for name in ap.parse_args().set:
        p = build(name)
        print(f"{name}: wrote {p.relative_to(ROOT)} ({p.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
