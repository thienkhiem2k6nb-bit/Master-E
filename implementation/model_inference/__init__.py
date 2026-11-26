"""Model inference layer (pretrained backbones only). Kept separate from `gecalgo`, which contains the
algorithms. Nothing in this package implements an algorithm under study; it wraps Hugging Face
`generate()` so that the algorithms receive n-best lists and scores as plain Python data."""
