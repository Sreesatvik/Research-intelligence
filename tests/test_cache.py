import numpy as np

from cache.answer_cache import AnswerCache, key_tokens, normalize


def fake_embed(text: str) -> np.ndarray:
    """Tiny deterministic embedding: letter-frequency vector, unit length."""
    v = np.zeros(26)
    for ch in text.lower():
        if "a" <= ch <= "z":
            v[ord(ch) - 97] += 1
    return v / np.linalg.norm(v)


def test_normalize_makes_trivial_rewrites_equal():
    assert normalize("  What is RAG?? ") == normalize("what is rag")


def test_exact_hit():
    cache = AnswerCache()
    cache.put("What is RAG?", "answer")
    assert cache.get("what is rag") == ("answer", "exact")


def test_miss_returns_none_and_is_counted():
    cache = AnswerCache()
    assert cache.get("anything") is None and cache.stats["miss"] == 1


def test_scope_separates_entries():
    cache = AnswerCache()
    cache.put("q", "k3 answer", scope="k=3")
    assert cache.get("q", scope="k=5") is None


def test_lru_evicts_oldest():
    cache = AnswerCache(max_size=2)
    cache.put("one", 1)
    cache.put("two", 2)
    cache.get("one")          # "one" is now more recent than "two"
    cache.put("three", 3)     # evicts "two"
    assert cache.get("two") is None and cache.get("one") is not None


def test_semantic_hit_when_vectors_match_and_key_tokens_match():
    cache = AnswerCache(embed=fake_embed, semantic_threshold=0.9)
    cache.put("what is retrieval augmented generation", "A")
    assert cache.get("what is retrieval augmented generation again", ) == ("A", "semantic")


def test_guard_blocks_different_numbers_even_if_vectors_are_identical():
    cache = AnswerCache(embed=lambda t: np.ones(3) / np.sqrt(3), semantic_threshold=0.5)
    cache.put("What does Table 2 show", "table two")
    assert cache.get("What does Table 3 show") is None   # similarity 1.0, but 2 != 3


def test_guard_blocks_different_acronyms():
    cache = AnswerCache(embed=lambda t: np.ones(3) / np.sqrt(3), semantic_threshold=0.5)
    cache.put("score of FLARE on StrategyQA", "x")
    assert cache.get("score of FLARE on ASQA") is None


def test_key_tokens_extracts_numbers_and_names():
    assert key_tokens("What does Table 3 say about StrategyQA and FLARE?") == {"3", "StrategyQA", "FLARE"}


def test_clear_drops_everything():
    cache = AnswerCache()
    cache.put("q", "a")
    cache.clear()
    assert cache.get("q") is None
