from pydantic import BaseModel, Field

from agent.tools.base import Tool, ToolRegistry
from core.interfaces import Retriever
from core.schemas import Chunk


class SearchArgs(BaseModel):
    query: str = Field(min_length=1, description="what to look for, in natural language")
    k: int = Field(default=5, ge=1, le=10, description="number of results")


class Hit(BaseModel):
    chunk_id: str
    document_id: str
    score: float
    text: str


class SearchOutput(BaseModel):
    results: list[Hit]


class DocArgs(BaseModel):
    document_id: str = Field(min_length=1, description="a document_id returned by semantic_search")


class DocOutput(BaseModel):
    document_id: str
    text: str


class ClassifyOutput(BaseModel):
    document_id: str
    topic: str
    confidence: float


def build_tools(retriever: Retriever, chunks: list[Chunk], classifier=None) -> ToolRegistry:
    """Without a classifier the classify tool is left out (a user's own documents have no fixed topic list)."""
    by_doc: dict[str, list[Chunk]] = {}
    for c in chunks:
        by_doc.setdefault(c.document_id, []).append(c)

    def doc_text(document_id: str) -> str:
        if document_id not in by_doc:
            raise ValueError(f"unknown document_id '{document_id}'")
        return " ".join(c.text for c in by_doc[document_id] if c.content_type == "text")

    def semantic_search(query: str, k: int) -> SearchOutput:
        hits = retriever.search(query, k=k)
        return SearchOutput(results=[Hit(chunk_id=h.chunk.chunk_id, document_id=h.chunk.document_id,
                                         score=round(h.score, 3), text=h.chunk.text[:300]) for h in hits])

    def retrieve_document(document_id: str) -> DocOutput:
        return DocOutput(document_id=document_id, text=doc_text(document_id)[:3000])

    def classify_document(document_id: str) -> ClassifyOutput:
        topic, conf = classifier.classify(doc_text(document_id))
        return ClassifyOutput(document_id=document_id, topic=topic, confidence=round(conf, 3))

    tools = [
        Tool("semantic_search", "Search the document collection; returns the most relevant text chunks "
             "with their document_id.", SearchArgs, semantic_search),
        Tool("retrieve_document", "Get the text of one document by document_id.", DocArgs, retrieve_document),
    ]
    if classifier is not None:
        tools.append(Tool("classify_document", "Predict the research topic of one document by document_id, "
                          "with a confidence.", DocArgs, classify_document))
    return ToolRegistry(tools)
