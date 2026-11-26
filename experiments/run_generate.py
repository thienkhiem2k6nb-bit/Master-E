#!/usr/bin/env python3
"""Generate n-best lists with a fixed backbone and store them as JSONL (resumable).

    python run_generate.py --set bea19dev --model coedit --beams 4 --nret 4 --batch-size 8

Output: results/nbest/<set>__<model>__b<beams>n<nret>.jsonl with one record per sentence:
  {"id": i, "source": ..., "hyps": [...], "scores": [...], "truncated": [...], "gen_ms": amortised ms}
and a sidecar .meta.json with environment, settings, wall-clock time and peak RSS.

Baseline A (top-1 of beam search) is `hyps[0]` of the CoEdIT run; the same file feeds every
post-processing algorithm, so all variants share exactly the same model outputs.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from common import DATA, RESULTS, SEED, dump_json, env_info, read_lines, rss_mb, setup_logging

MODELS = {"coedit": ("grammarly/coedit-large", "coedit"), "flan": ("google/flan-t5-large", "flan"),
          "flan-base": ("google/flan-t5-base", "flan")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=["conll14", "bea19dev", "jfleg"])
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--beams", type=int, default=4)
    ap.add_argument("--nret", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    log = setup_logging(f"generate_{args.set}_{args.model}")
    from model_inference.seq2seq_backend import Seq2SeqBackend

    model_id, style = MODELS[args.model]
    sources = read_lines(DATA / args.set / "source.txt")
    if args.limit:
        sources = sources[: args.limit]
    out_path = RESULTS / "nbest" / f"{args.set}__{args.model}__b{args.beams}n{args.nret}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done: dict[int, dict] = {}
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line); done[rec["id"]] = rec
    todo = [i for i in range(len(sources)) if i not in done]
    log.info("set=%s model=%s beams=%d nret=%d batch=%d: %d sentences, %d already done", args.set, model_id,
             args.beams, args.nret, args.batch_size, len(sources), len(done))
    if not todo:
        log.info("nothing to do"); return 0

    backend = Seq2SeqBackend(model_id, prompt_style=style, threads=args.threads, seed=SEED)
    log.info("model loaded in %.1fs, RSS %.0f MB", backend.load_s, rss_mb())

    # length-bucketed batches reduce padding; ids are preserved in the records
    todo.sort(key=lambda i: len(sources[i].split()))
    t_start = time.perf_counter()
    peak = rss_mb()
    n_done = 0
    with out_path.open("a", encoding="utf-8") as f:
        for b in range(0, len(todo), args.batch_size):
            ids = todo[b: b + args.batch_size]
            batch = [sources[i] for i in ids]
            try:
                results = backend.nbest(batch, num_beams=args.beams, num_return=args.nret)
            except Exception as exc:  # pragma: no cover - keep going, record failure
                log.exception("batch failed (%s); retrying one by one", exc)
                results = []
                for s in batch:
                    try:
                        results.extend(backend.nbest([s], num_beams=args.beams, num_return=args.nret))
                    except Exception as exc2:
                        log.error("sentence failed: %s", exc2)
                        from model_inference.seq2seq_backend import NBest
                        results.append(NBest(s, [s] * args.nret, [0.0] * args.nret, [False] * args.nret, 0.0))
            for i, nb in zip(ids, results):
                rec = {"id": i, "source": nb.source, "hyps": nb.hypotheses, "scores": nb.scores,
                       "truncated": nb.output_truncated, "gen_ms": nb.gen_ms}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            n_done += len(ids)
            peak = max(peak, rss_mb())
            if (b // args.batch_size) % 5 == 0:
                el = time.perf_counter() - t_start
                log.info("%d/%d  %.1f s/sent  elapsed %.0fs  RSS %.0f MB", n_done, len(todo), el / n_done, el, rss_mb())
    wall = time.perf_counter() - t_start
    meta = {"set": args.set, "model_id": model_id, "prompt_style": style, "num_beams": args.beams, "num_return": args.nret,
            "batch_size": args.batch_size, "seed": SEED, "do_sample": False, "max_new_tokens": 128, "max_input_tokens": 256,
            "n_sentences_this_run": len(todo), "wall_clock_s": round(wall, 1), "s_per_sentence": round(wall / len(todo), 3),
            "peak_rss_mb": round(peak), "model_load_s": round(backend.load_s, 1), "env": env_info()}
    dump_json(out_path.with_suffix(".meta.json"), meta)
    log.info("done: %d sentences in %.0fs (%.2f s/sent), peak RSS %.0f MB", len(todo), wall, wall / len(todo), peak)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
