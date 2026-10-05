from core.schemas import SearchResult
from rag.prompts.templates import SYSTEM_RULES


def build_prompt(question: str, results: list[SearchResult]) -> tuple[str, dict[int, str]]:
    """Return (prompt text, {source number: chunk_id}) for the given evidence."""
    source_map = {}
    blocks = []
    for number, result in enumerate(results, start=1):
        source_map[number] = result.chunk.chunk_id
        blocks.append(f"[{number}] {result.chunk.text}")
    sources = "\n\n".join(blocks)
    prompt = f"{SYSTEM_RULES}\n\nSOURCES:\n{sources}\n\nQUESTION: {question}\n\nANSWER:"
    return prompt, source_map
