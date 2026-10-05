from dataclasses import dataclass, field


@dataclass
class Chunk:
    """One retrieval unit, with enough provenance to cite it (spec 5.2)."""
    chunk_id: str
    document_id: str
    text: str
    position: int  # index of this chunk within its document
    content_type: str = "text"
    metadata: dict = field(default_factory=dict)
    page: int | None = None  # 1-based page in the source PDF


@dataclass
class SearchResult:
    """A retrieved chunk together with its relevance score."""
    chunk: Chunk
    score: float


@dataclass
class RAGAnswer:
    """The generated answer plus the evidence it was based on."""
    question: str
    answer: str
    cited_chunk_ids: list[str]    # chunks the answer actually cites
    retrieved: list[SearchResult]  # everything the model was shown
    refused_by_retrieval: bool = False  # True if the LLM was never called
    tokens_used: int = 0  # LLM tokens spent on this answer (0 if the LLM was not called)
