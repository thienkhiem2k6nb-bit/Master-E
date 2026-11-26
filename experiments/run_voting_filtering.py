#!/usr/bin/env python3
"""Re-run CoEdIT N-best edit voting/filtering on stored beams; no model needed."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from common import DATA, RESULTS, featurize, read_lines, tokenize, write_lines
from gecalgo.voting import consensus_correction


def fmt_num(x: float) -> str:
    """Filename-safe number that never collides: 0.5 -> '0.5', 1.0 -> '1', 0.05 -> '0.05', 0.55 -> '0.55'."""
    return f"{x:g}"


def fmt_tau(tau: float) -> str:
    """Keep the historical one-decimal names (t0.5, t1.0) but never round a finer tau onto a coarser one."""
    return f"{tau:.1f}" if round(tau, 1) == tau else fmt_num(tau)


def unique(values: list[float]) -> list[float]:
    return list(dict.fromkeys(values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--set', default='bea19dev', choices=['bea19dev', 'conll14', 'jfleg'])
    parser.add_argument('--mode', default='both', choices=['vote', 'filter', 'both'])
    parser.add_argument('--taus', nargs='+', type=float, default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    parser.add_argument('--temperatures', nargs='+', type=float, default=[0.1, 0.3, 1.0])
    parser.add_argument('--uniform', action='store_true')
    parser.add_argument('--no-features', action='store_true', help='Disable spaCy linguistic alignment features (ablation)')
    parser.add_argument('--whitespace', action='store_true', help='Tokenise with split() for quick smoke testing; NOT benchmark-comparable')
    parser.add_argument('--out-dir', type=Path, default=None,
                        help='Write outputs here instead of results/outputs/<set> (keeps historical outputs untouched)')
    args = parser.parse_args()

    nbest_path = RESULTS / 'nbest' / f'{args.set}__coedit__b4n4.jsonl'
    records = [json.loads(s) for s in nbest_path.read_text(encoding='utf-8').splitlines() if s.strip()]
    records = sorted(records, key=lambda x: x['id'])
    sources = read_lines(DATA / args.set / 'source.txt')
    if len(records) != len(sources) or any(r['id'] != i for i, r in enumerate(records)):
        raise ValueError('N-best/source mismatch or missing IDs')

    tok = (lambda texts: [x.split() for x in texts]) if args.whitespace else tokenize
    htokens = [tok(r['hyps']) for r in records]
    source_tokens = [s.split() for s in sources]
    if args.no_features or args.whitespace:
        sf, hf = source_tokens, htokens
    else:
        sf = featurize(source_tokens)
        flattened = featurize([hyp for hs in htokens for hyp in hs])
        hf, i = [], 0
        for hs in htokens:
            hf.append(flattened[i:i + len(hs)])
            i += len(hs)

    outputs = args.out_dir if args.out_dir is not None else RESULTS / 'outputs' / args.set
    outputs.mkdir(parents=True, exist_ok=True)
    write_lines(outputs / 'coedit__top1.txt', [' '.join(hs[0]) for hs in htokens])

    configurations = []
    weights = [('soft' + fmt_num(t), 'softmax', t) for t in unique(args.temperatures)]
    if args.uniform:
        weights.append(('unif', 'uniform', 1.0))
    for mode in (['vote', 'filter'] if args.mode == 'both' else [args.mode]):
        for label, wmode, temp in weights:
            for tau in unique(args.taus):
                name = f'coedit__{mode}_t{fmt_tau(tau)}_{label}'
                lines, edit_records, started = [], [], time.perf_counter()
                for idx, rec in enumerate(records):
                    ans = consensus_correction(sf[idx], hf[idx], rec['scores'], tau=tau, temperature=temp,
                                               weight_mode=wmode, restrict_to_top1=(mode == 'filter'))
                    out = ' '.join(ans.tokens)
                    lines.append(out)
                    edit_records.append({'id': idx, 'source': sources[idx], 'output': out,
                                         'kept': [{'start': e.start, 'end': e.end, 'original': list(e.orig),
                                                   'replacement': list(e.cor), 'support': round(ans.support[e.key], 6)}
                                                  for e in ans.kept]})
                write_lines(outputs / (name + '.txt'), lines)
                with (outputs / (name + '.edits.jsonl')).open('w', encoding='utf-8') as f:
                    for r in edit_records:
                        f.write(json.dumps(r, ensure_ascii=False) + '\n')
                elapsed = time.perf_counter() - started
                configurations.append({'config': name, 'ms_per_sentence': round(1000 * elapsed / len(records), 3),
                                       'sentences_changed': sum(a.strip() != b.strip() for a, b in zip(lines, sources))})
                print(name, 'done')

    run_info = {'dataset': args.set, 'size': len(records), 'feature_alignment': not (args.no_features or args.whitespace),
                'whitespace': args.whitespace, 'configurations': configurations}
    (outputs / 'voting_filtering_run.json').write_text(json.dumps(run_info, indent=2), encoding='utf-8')
    print('Saved to', outputs)


if __name__ == '__main__':
    main()
