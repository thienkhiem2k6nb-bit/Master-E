"""Shared helpers for the experiment scripts (paths, logging, tokenisation, timing, memory)."""
from __future__ import annotations

import json
import logging
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
sys.path.insert(0, str(ROOT / "implementation"))

SEED = 42


def setup_logging(name: str) -> logging.Logger:
    (RESULTS / "logs").mkdir(parents=True, exist_ok=True)
    log = logging.getLogger(name)
    log.setLevel(logging.INFO)
    if not log.handlers:
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        fh = logging.FileHandler(RESULTS / "logs" / f"{name}.log", encoding="utf-8")
        sh = logging.StreamHandler(sys.stderr)
        fh.setFormatter(fmt); sh.setFormatter(fmt)
        log.addHandler(fh); log.addHandler(sh)
    return log


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def write_lines(path: Path, lines) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def rss_mb() -> float:
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1e6
    except Exception:  # pragma: no cover
        return float("nan")


def env_info() -> dict:
    info = {"python": platform.python_version(), "platform": platform.platform(), "cpu_count": os.cpu_count(),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    import importlib
    for name in ("torch", "transformers", "numpy", "spacy", "errant"):   # record whatever is installed
        try:
            info[name] = importlib.import_module(name).__version__
        except Exception:  # pragma: no cover
            pass
    if "torch" in info:
        import torch
        info["torch_threads"] = torch.get_num_threads()
    return info


_NLP = None


def spacy_nlp():
    """Shared spaCy pipeline (same model as ERRANT: en_core_web_sm) for tokenisation and features."""
    global _NLP
    if _NLP is None:
        import spacy
        _NLP = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    return _NLP


def tokenize(texts: list[str], batch_size: int = 256) -> list[list[str]]:
    """spaCy word tokenisation (what ERRANT uses with -tok). Returns token lists."""
    nlp = spacy_nlp()
    out = []
    for doc in nlp.tokenizer.pipe(texts, batch_size=batch_size):
        out.append([t.text for t in doc if not t.is_space])
    return out


def featurize(token_lists: list[list[str]]):
    """Lemma/POS features for pre-tokenised sentences, for the linguistic alignment cost."""
    from spacy.tokens import Doc
    from gecalgo.alignment import Token

    nlp = spacy_nlp()
    docs = [Doc(nlp.vocab, words=toks if toks else ["."]) for toks in token_lists]
    out = []
    for toks, doc in zip(token_lists, nlp.pipe(docs, batch_size=256)):
        if not toks:
            out.append([]); continue
        out.append([Token(t.text, t.lemma_, t.pos_) for t in doc])
    return out


def dump_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
