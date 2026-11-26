"""gecalgo - n-best edit voting / confidence-based filtering for grammatical error correction.

Every module here is pure Python logic that operates on token sequences, n-best lists and edit sets.
No module loads or calls a neural model; model inference lives in `model_inference/`. This
separation makes each algorithm unit-testable without PyTorch.

Modules
  alignment  Levenshtein / Damerau-Levenshtein token alignment with linguistic costs (Felice et al. 2016)
  edits      Edit spans, M2 I/O, applying edits, overlap resolution
  voting     N-best edit-level voting and top-1 filtering (system-combination style)
"""

__version__ = "1.0.0"
