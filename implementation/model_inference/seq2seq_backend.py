"""Seq2seq backbone wrapper for CoEdIT (grammarly/coedit-large) and FLAN-T5 (google/flan-t5-*).

The wrapper returns, for every input sentence, the n-best list produced by beam search together with
the length-normalised sequence log-probabilities (`sequences_scores`). All algorithms (edit voting,
MBR, iterative correction) consume this plain data structure; they never touch the model.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

PROMPTS = {
    # CoEdIT instruction used throughout the project
    "coedit": "Fix grammatical errors in this sentence: {text}",
    # Zero-shot instruction for the un-tuned FLAN-T5 backbone (baseline B)
    "flan": "Correct the grammar of the following sentence. Output only the corrected sentence.\nSentence: {text}\nCorrected sentence:",
}


@dataclass
class NBest:
    source: str
    hypotheses: list[str]
    scores: list[float]            # length-normalised log-probs from beam search (higher is better)
    output_truncated: list[bool] = field(default_factory=list)
    gen_ms: float = 0.0            # amortised wall-clock ms per sentence for the batch


class Seq2SeqBackend:
    def __init__(self, model_id: str, prompt_style: str = "coedit", device: str = "cpu", threads: int | None = None,
                 max_input_tokens: int = 256, max_new_tokens: int = 128, seed: int = 42):
        import torch
        from transformers import AutoTokenizer, T5ForConditionalGeneration

        self.torch = torch
        if threads:
            torch.set_num_threads(threads)
        torch.manual_seed(seed)
        self.model_id = model_id
        self.prompt = PROMPTS[prompt_style]
        self.device = device
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        t0 = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = T5ForConditionalGeneration.from_pretrained(model_id).to(device).eval()
        self.load_s = time.perf_counter() - t0

    def nbest(self, sources: list[str], num_beams: int = 4, num_return: int = 4) -> list[NBest]:
        """Beam search with `num_return` returned beams per sentence (num_return <= num_beams)."""
        assert 1 <= num_return <= num_beams
        torch = self.torch
        prompts = [self.prompt.format(text=s) for s in sources]
        t0 = time.perf_counter()
        enc = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_input_tokens)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.inference_mode():
            out = self.model.generate(
                **enc, num_beams=num_beams, num_return_sequences=num_return, do_sample=False,
                max_new_tokens=self.max_new_tokens, output_scores=True, return_dict_in_generate=True,
                early_stopping=True,
            )
        ms = (time.perf_counter() - t0) * 1000 / len(sources)
        texts = self.tokenizer.batch_decode(out.sequences, skip_special_tokens=True)
        scores = out.sequences_scores.tolist() if out.sequences_scores is not None else [0.0] * len(texts)
        eos = self.tokenizer.eos_token_id
        truncated = [eos not in row[1:].tolist() for row in out.sequences]
        results = []
        for i, src in enumerate(sources):
            sl = slice(i * num_return, (i + 1) * num_return)
            results.append(NBest(src, [t.strip() for t in texts[sl]], scores[sl], truncated[sl], round(ms, 2)))
        return results

    def correct(self, sources: list[str], num_beams: int = 4) -> list[str]:
        return [nb.hypotheses[0] for nb in self.nbest(sources, num_beams=num_beams, num_return=1)]

    def generate_raw(self, prompts: list[str], num_beams: int = 4, max_new_tokens: int | None = None) -> list[str]:
        """Generate from already-formatted prompts (no instruction template), e.g. for dialogue responses."""
        torch = self.torch
        enc = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.inference_mode():
            out = self.model.generate(**enc, num_beams=num_beams, do_sample=False,
                                      max_new_tokens=max_new_tokens or self.max_new_tokens, early_stopping=True)
        return [t.strip() for t in self.tokenizer.batch_decode(out, skip_special_tokens=True)]
