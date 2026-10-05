from typing import Annotated, Literal

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    k: int = Field(default=5, ge=1, le=20)


class SourceItem(BaseModel):
    chunk_id: str
    document_id: str
    score: float
    text: str


class SearchResponse(BaseModel):
    results: list[SourceItem]


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    k: int = Field(default=3, ge=1, le=10)


class AskResponse(BaseModel):
    answer: str
    cited_chunk_ids: list[str]
    refused_by_retrieval: bool
    sources: list[SourceItem]
    tokens_used: int = 0  # LLM tokens this request cost (0 for cache hits and early refusals)
    cache_hit: str = "none"  # "none", "exact" or "semantic"
    route: str = "rag"  # "retrieval", "rag" or "agent" (the last means: use POST /research)


class ResearchRequest(BaseModel):
    question: str = Field(min_length=1)


class ResearchStarted(BaseModel):
    job_id: str
    status: str


class ToolCallInfo(BaseModel):
    step: int
    tool: str
    arguments: str
    is_error: bool


class ResearchStatus(BaseModel):
    job_id: str
    status: str  # "running", "done" or "failed"
    answer: str | None = None
    stopped_because: str | None = None
    tool_calls: list[ToolCallInfo] = []
    tokens_used: int = 0
    error: str | None = None
    owner: str | None = None  # who started the job; never shown to clients


class SendMessage(BaseModel):
    content: str = Field(min_length=1, max_length=4000)
    mode: Literal["ask", "research"] = "ask"


class RenameChat(BaseModel):
    title: str = Field(min_length=1, max_length=100)


class CheckFiles(BaseModel):
    hashes: list[Annotated[str, Field(min_length=64, max_length=64, pattern="^[0-9a-f]+$")]] = Field(max_length=50)
