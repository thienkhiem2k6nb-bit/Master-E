"""Unit tests for the alignment algorithms (no model, no spaCy)."""
from gecalgo.alignment import (Token, align, char_similarity, extract_edits, levenshtein_distance,
                               levenshtein_matrix, substitution_cost)
from gecalgo.edits import Edit, apply_edits


def test_levenshtein_matrix_matches_known_values():
    assert levenshtein_distance("kitten".split(), "sitting".split()) == 1  # token level: 1 substitution
    assert levenshtein_distance(list("kitten"), list("sitting")) == 3        # character level classic example
    assert levenshtein_distance([], ["a", "b"]) == 2
    assert levenshtein_distance(["a", "b"], ["a", "b"]) == 0
    m = levenshtein_matrix(list("ab"), list("ba"))
    assert m[2][2] == 2


def test_substitution_cost_properties():
    assert substitution_cost(Token("the"), Token("the")) == 0.0
    assert 0.0 < substitution_cost(Token("The"), Token("the")) < 0.2          # case only
    close = substitution_cost(Token("go", "go", "VERB"), Token("goes", "go", "VERB"))
    far = substitution_cost(Token("go", "go", "VERB"), Token("banana", "banana", "NOUN"))
    assert close < far
    assert 0.5 < substitution_cost(Token("cat"), Token("dog")) <= 1.5 + 1e-9  # no features: 0.5 + char cost
    assert substitution_cost(Token("go"), Token("goes")) < 1.0                 # related tokens stay cheap
    assert char_similarity("abc", "abc") == 1.0


def test_align_identity_and_simple_ops():
    al = align("I like it".split(), "I like it".split())
    assert [op for op, _, _ in al.ops] == ["M", "M", "M"] and al.cost == 0
    al = align("I likes it".split(), "I like it".split())
    assert [op for op, _, _ in al.ops] == ["M", "S", "M"]
    al = align("I it".split(), "I like it".split())
    assert [op for op, _, _ in al.ops] == ["M", "I", "M"]
    al = align("I really like it".split(), "I like it".split())
    assert [op for op, _, _ in al.ops] == ["M", "D", "M", "M"]


def test_align_detects_transposition():
    al = align("I only have one".split(), "I have only one".split())
    assert ("T", 1, 1) in al.ops
    assert al.cost == 1.0


def test_extract_edits_and_apply_roundtrip():
    src = "She don't likes speaking English .".split()
    tgt = "She doesn't like speaking English .".split()
    edits = extract_edits(src, tgt)
    assert apply_edits(src, edits) == tgt
    assert all(isinstance(e, Edit) for e in edits)
    assert [e.op for e in edits] == ["R"]  # one merged replacement "don't likes" -> "doesn't like"
    assert edits[0].start == 1 and edits[0].end == 3


def test_extract_edits_insertion_deletion_positions():
    src = "I went school yesterday".split()
    tgt = "I went to school".split()
    edits = extract_edits(src, tgt)
    assert apply_edits(src, edits) == tgt
    kinds = sorted(e.op for e in edits)
    assert kinds == ["M", "U"]
    ins = [e for e in edits if e.op == "M"][0]
    assert (ins.start, ins.end, ins.cor) == (2, 2, ("to",))


def test_punctuation_split_rule():
    src = "I like it .".split()
    tgt = "I liked it !".split()
    edits = extract_edits(src, tgt)
    assert apply_edits(src, edits) == tgt
    assert len(edits) == 2


def test_roundtrip_random_sentences():
    import random
    rng = random.Random(1)
    vocab = "a b c d e f g".split()
    for _ in range(200):
        src = [rng.choice(vocab) for _ in range(rng.randint(0, 8))]
        tgt = [rng.choice(vocab) for _ in range(rng.randint(0, 8))]
        assert apply_edits(src, extract_edits(src, tgt)) == tgt
