# N-best Edit Voting + Confidence-based Filtering — isolated experiments

This archive was **extracted from the user-provided algorithm-research.zip**. Only the CoEdIT n-best beam post-processing algorithm, its alignment/edit utilities, necessary datasets, original beam lists, related source outputs and recorded score subsets are retained. Unrelated learning, dialogue, MBR, FLAN-T5 and iterative workflows have been removed.

## Scope
- **Baseline:** CoEdIT top-1 beam
- **Voting:** all beam-proposed edits are candidates; keep if weighted support >= tau
- **Filtering:** only edits from the first beam are candidates; other beams may veto them
- **Weights:** softmax of length-normalised beam scores, temperature T; optional uniform ablation
- **Backbone:** `grammarly/coedit-large`, frozen (not trained here)

## Math
For scores z_k and temperature T>0, weights are `w_k = exp(z_k/T) / Σ_j exp(z_j/T)`.
For edit e, support is `support(e) = Σ_k w_k * I[e ∈ E(h_k)]`.
Select edits with `support(e) >= τ`; resolve overlaps by support and apply them to the source.
Filtering additionally constrains candidates to the top-1 beam's edits. A candidate edit is identified by **source span + replacement tokens**.

## Run immediately without loading CoEdIT
The ZIP includes stored original beam lists (4 per source) for BEA-dev 500 sentences, CoNLL-2014 1312 sentences and JFLEG 747 sentences. This lets you re-run post-processing **without GPU or model downloads**.

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-minimal.txt
python -m spacy download en_core_web_sm
export PYTHONPATH=implementation   # Windows PowerShell: $env:PYTHONPATH="implementation"
python experiments/run_voting_filtering.py --set bea19dev --mode both --taus 0.5 0.6 --temperatures 0.1
python experiments/run_voting_filtering.py --set conll14 --mode both --taus 0.5 --temperatures 0.1
python experiments/run_voting_filtering.py --set jfleg --mode both --taus 0.5 --temperatures 0.1
python -m pytest -q tests/test_voting_filtering.py tests/test_alignment.py
```

For a lightweight logic-only run without spaCy, add `--whitespace --no-features`, but these scores **must not** be compared directly with the original spaCy-tokenised benchmark.

Outputs: `results/outputs/<dataset>/coedit__vote_*.txt`, `coedit__filter_*.txt`, matching `.edits.jsonl` with per-edit support, and `voting_filtering_run.json` (timing). By default these **overwrite the historical outputs of the same name**; pass `--out-dir <dir>` (e.g. `--out-dir runs/conll14`) to write a new run elsewhere and keep `results/outputs/` untouched.

## Web version (static HTML, runs in the browser)
`web/` is an interactive page that reruns voting/filtering live: τ slider, temperature, uniform weights, Voting/Filtering/Top-1, per-sentence diffs with every candidate edit's support, historical-score and sentences-changed charts across τ, and a `.txt` download of the output.

```bash
.venv/bin/python web/build_data.py      # once, or after changing alignment / data (needs spaCy)
xdg-open web/index.html                 # or open it in any browser; works from file://
```
The slow, τ-independent part (spaCy tokenisation, features, alignment, edit extraction) runs once in `build_data.py` and is stored in `web/data/<set>.js`. `web/voting.js` ports only weighting, support, thresholding, overlap resolution and edit application, and its outputs are byte-identical to the Python runner. The page checks this against hashes of `results/outputs/` and shows a ✓ badge. To host it, upload `web/index.html`, `web/voting.js` and `web/data/` to any static host (GitHub Pages, Netlify, nginx); no server code is needed. Check the dataset licences before making the page public, because `data/*.js` contains the source sentences.

## Scores and reporting
`results/scores/*.json` and pre-existing `results/outputs` are **historical results copied from the uploaded ZIP**, not newly re-measured. New output files replace the same name, so do not mistake them for original untouched results. The original report numbers include:

| Test | top-1 | filter τ=.5 T=.1 | vote τ=.5 T=.1 | Metric |
|---|---:|---:|---:|---|
| CoNLL-2014 | 53.76 | 53.81 | 53.97 | M2 F0.5 |
| CoNLL-2014 | 47.25 | 48.22 | 48.29 | ERRANT F0.5 |
| JFLEG | 58.98 | 58.30 | 58.54 | GLEU |

CoNLL official M2 improvement for the selected filtering configuration is **not statistically significant**, so don't claim a universal correction-quality gain. The algorithm's main supported claim is controllable precision/recall.

Scoring with official M2/ERRANT/GLEU is **not bundled as an executable one-command workflow**: original scorer contained machine-specific paths under `/home/claude/repos/` and therefore was excluded. Reference output scores are preserved for comparison. To produce new scores after changes, set up the official external scorers independently and compare output `.txt` against dataset gold/reference files. **BEA subset source.txt was reconstructed from the original n-best records** bundled in the supplied archive; BEA reference annotations were not present, so direct benchmark scoring there needs the original gold dataset. CoNLL and JFLEG source text is bundled.

To generate new CoEdIT beams (instead of replaying saved ones), install matching torch/transformers (consult `results/environment/requirements.lock.txt`), obtain model access, and run:
```bash
PYTHONPATH=implementation python experiments/run_generate.py --set conll14 --model coedit --beams 4 --nret 4 --batch-size 8
```

## Included files
- `implementation/gecalgo/voting.py`: isolated voting / filtering logic
- `implementation/gecalgo/alignment.py`, `edits.py`: token-level edit extraction and overlap resolution
- `experiments/run_voting_filtering.py`: dedicated experiment runner
- `experiments/run_generate.py`, `implementation/model_inference/*`: optional CoEdIT beam generation
- `data/`, `results/nbest/`, `results/outputs/`, `results/scores/`: relevant data and historical comparisons
- `tests/`: focused tests

All numerical comparison claims above are from the supplied experimental report; no new benchmark scores are invented.
