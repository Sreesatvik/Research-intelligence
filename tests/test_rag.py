from core.schemas import Chunk, SearchResult
from rag.pipeline import RAG


class OneChunkRetriever:
    def search(self, query, k=3):
        return [SearchResult(Chunk("docA#0", "docA", "Dogs are mammals.", 0), 0.9)]


class FixedLLM:
    def __init__(self, reply):
        self.reply = reply

    def generate(self, system, user, temperature=0.0):
        return self.reply


def cited(reply):
    return RAG(OneChunkRetriever(), FixedLLM(reply)).ask("q").cited_chunk_ids


def test_square_bracket_citation():
    assert cited("Dogs are mammals [1].") == ["docA#0"]


def test_fullwidth_bracket_citation():
    assert cited("Dogs are mammals \u30101\u3011.") == ["docA#0"]


def test_citation_to_unknown_source_is_ignored():
    assert cited("Cats are mammals [7].") == []


def test_line_range_citation_format_is_recognised_and_cleaned():
    from rag.pipeline import normalize_citations

    text, numbers = normalize_citations("Defenses rely on fine-tuning \u30101\u2020L1-L4\u3011 and hierarchy \u30102\u2020L1-L3\u3011.")
    assert numbers == [1, 2]
    assert text == "Defenses rely on fine-tuning [1] and hierarchy [2]."


def test_multi_number_citation_is_split():
    from rag.pipeline import normalize_citations

    text, numbers = normalize_citations("Both agree [1, 3].")
    assert numbers == [1, 3] and text == "Both agree [1][3]."


def test_brackets_that_are_not_citations_are_left_alone():
    from rag.pipeline import normalize_citations

    text, numbers = normalize_citations("See [the appendix] and [section 2] for details.")
    assert numbers == [] and text == "See [the appendix] and [section 2] for details."


def test_ask_reports_cited_chunks_for_line_range_format():
    class RangeLLM:
        def generate(self, system, user, temperature=0.0):
            return "Dogs are mammals \u30101\u2020L1-L2\u3011."

    result = RAG(OneChunkRetriever(), RangeLLM()).ask("q")
    assert result.cited_chunk_ids == ["docA#0"] and result.answer == "Dogs are mammals [1]."
