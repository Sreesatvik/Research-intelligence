import re
from typing import Iterator

from core.interfaces import LLM, Retriever
from core.schemas import RAGAnswer
from rag.context.builder import build_prompt
from rag.prompts.templates import SYSTEM_RULES


REFUSAL = "I don't know based on the provided sources."

# Models cite in several shapes: [1]  [1, 2]  【1】  【1†L4-L9】 (the last carries a line range we do not use).
CITATION_RE = re.compile(r"[\[\u3010]\s*(\d+(?:\s*,\s*\d+)*)\s*(?:\u2020[^\]\u3011]*)?[\]\u3011]")


def normalize_citations(answer: str) -> tuple[str, list[int]]:
    """Rewrite every citation as plain [n] markers and return the source numbers that were cited."""
    numbers: set[int] = set()

    def clean(match: re.Match) -> str:
        found = [int(n) for n in re.findall(r"\d+", match.group(1))]
        numbers.update(found)
        return "".join(f"[{n}]" for n in found)

    return CITATION_RE.sub(clean, answer), sorted(numbers)


AGENT_CITATION_RE = re.compile(r"【([^】]+?)(?:†[^】]*)?】")


def normalize_agent_citations(answer: str) -> str:
    """The agent cites document ids, sometimes in the fullwidth form 【id】 or 【id†L1-L4】; show them as [id]."""
    return AGENT_CITATION_RE.sub(lambda m: f"[{m.group(1).strip()}]", answer)


class RAG:
    def __init__(self, retriever: Retriever, llm: LLM, k: int = 3,
                 min_score: float = 0.25):
        self.retriever = retriever
        self.llm = llm
        self.k = k
        self.min_score = min_score

    def ask(self, question: str, temperature: float = 0.0) -> RAGAnswer:
        results = self.retriever.search(question, k=self.k)
        if not results or results[0].score < self.min_score:
            # evidence too weak: refuse without spending an LLM call
            return RAGAnswer(question, REFUSAL, [], results, refused_by_retrieval=True)
        prompt, source_map = build_prompt(question, results)
        # build_prompt returns rules + sources + question as one string;
        # split the rules off so they go in the "system" role.
        user_part = prompt[len(SYSTEM_RULES):].strip()
        tokens_before = getattr(self.llm, "tokens_used", 0)
        answer = self.llm.generate(SYSTEM_RULES, user_part, temperature)
        tokens = getattr(self.llm, "tokens_used", 0) - tokens_before
        answer, cited_numbers = normalize_citations(answer)
        cited = [source_map[n] for n in cited_numbers if n in source_map]
        return RAGAnswer(question, answer, cited, results, tokens_used=tokens)

    def ask_stream(self, question: str, temperature: float = 0.0) -> Iterator[tuple[str, object]]:
        """Like ask(), but yields ("sources", results) once, then ("delta", text) pieces, then ("done", RAGAnswer).
        An LLM without a stream() method still works: its whole answer arrives as one piece."""
        results = self.retriever.search(question, k=self.k)
        if not results or results[0].score < self.min_score:
            yield "done", RAGAnswer(question, REFUSAL, [], results, refused_by_retrieval=True)
            return
        prompt, source_map = build_prompt(question, results)
        user_part = prompt[len(SYSTEM_RULES):].strip()
        yield "sources", results
        tokens_before = getattr(self.llm, "tokens_used", 0)
        stream = getattr(self.llm, "stream", None)
        pieces = stream(SYSTEM_RULES, user_part, temperature) if stream else iter([self.llm.generate(SYSTEM_RULES, user_part, temperature)])
        parts = []
        for piece in pieces:
            parts.append(piece)
            yield "delta", piece
        answer, cited_numbers = normalize_citations("".join(parts))
        cited = [source_map[n] for n in cited_numbers if n in source_map]
        yield "done", RAGAnswer(question, answer, cited, results, tokens_used=getattr(self.llm, "tokens_used", 0) - tokens_before)
