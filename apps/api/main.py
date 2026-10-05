import json
import logging
import uuid
from typing import Callable

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse

from apps.api.schemas import (AskRequest, AskResponse, RenameChat, ResearchRequest, ResearchStarted,
                              ResearchStatus, SearchRequest, CheckFiles, SearchResponse, SendMessage, SourceItem, ToolCallInfo)
from apps.api.auth import Auth
from apps.api.job_store import InMemoryJobStore
from chats.store import ChatStore
from core.interfaces import LLM, Retriever
from core.schemas import SearchResult
from library.store import DuplicateDocument, Library, LibraryError
from quota.token_quota import QuotaExceeded, TokenQuota
from rag.cost_router import is_overview, route_by_rules
from rag.pipeline import RAG, normalize_agent_citations

log = logging.getLogger("api")

REWRITE_RULES = ("Rewrite the user's last question so it can be understood without the conversation: replace "
                 "pronouns and references with what they refer to. If it is already self-contained, repeat it "
                 "unchanged. Output only the question.")
FOLLOW_UP_WORDS = 14  # longer questions are assumed to stand on their own

OVERVIEW_RULES = ("You are summarizing a user's document library. For each document below, write one or two sentences "
                  "saying what it is about, using only the text shown. Use one bullet per document, starting with the "
                  "document name in bold. Do not invent details.")
OVERVIEW_DOCUMENTS = 12  # how many documents one overview covers
OVERVIEW_PASSAGES = 2  # opening passages shown per document (one when there are many documents)

EMPTY_LIBRARY = "Your library is empty. Add some documents in Library, then ask again."


def to_source(result: SearchResult) -> SourceItem:
    c = result.chunk
    return SourceItem(chunk_id=c.chunk_id, document_id=c.document_id,
                      score=round(result.score, 4), text=c.text)


def create_app(retriever: Retriever | None, llm: LLM, cache=None,
               agent_factory: Callable | None = None, quota: TokenQuota | None = None,
               job_store=None, auth: Auth | None = None, library: Library | None = None,
               chats: ChatStore | None = None, user_agent_factory: Callable | None = None) -> FastAPI:
    """Build the API from whichever retriever, LLM, cache, agent and quota we are given.

    With a `library`, every user searches only their own documents (`retriever` is then unused). Without one,
    a single shared `retriever` serves everybody, which is what the older tests use."""
    app = FastAPI(title="Research Intelligence")
    jobs = job_store or InMemoryJobStore()  # Redis-backed when several servers must share jobs

    def identify(request: Request, x_user_id: str = Header(default="anonymous")) -> str:
        """Who is calling. With accounts on, only a valid signed session counts and the header is ignored,
        because anyone can type any header. Without accounts (development) the header is trusted."""
        if auth is not None:
            return auth.require(request)
        return x_user_id

    if auth is not None:
        app.include_router(auth.router())

    def retriever_for(user: str):
        return library.retriever(user) if library is not None else retriever

    def enforce_quota(user: str) -> None:
        if quota is not None:
            try:
                quota.check(user)
            except QuotaExceeded as e:
                raise HTTPException(status_code=429, detail=str(e))

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/search", response_model=SearchResponse)
    def search(req: SearchRequest, user: str = Depends(identify)):
        index = retriever_for(user)
        results = index.search(req.query, k=req.k) if index is not None else []
        return SearchResponse(results=[to_source(r) for r in results])

    def overview_events(user: str, question: str, index, k: int):
        """Summarize the whole library from the opening of each document, instead of from top-k search hits."""
        by_document: dict[str, list] = {}
        for chunk in getattr(index, "chunks", []):
            if chunk.content_type == "text":
                by_document.setdefault(chunk.document_id, []).append(chunk)
        documents = list(by_document.items())[:OVERVIEW_DOCUMENTS]
        per_document = OVERVIEW_PASSAGES if len(documents) <= 8 else 1
        shown = [c for _, chunks in documents for c in chunks[:per_document]]
        yield "sources", [SourceItem(chunk_id=c.chunk_id, document_id=c.document_id, score=0.0, text=c.text) for c in shown]

        scope = f"{user}|{library.fingerprint(user) if library else '-'}|k={k}|overview"
        if cache is not None:
            hit = cache.get(question, scope)
            if hit is not None:
                cached, kind = hit
                yield "done", cached.model_copy(update={"tokens_used": 0, "cache_hit": kind})
                return
        enforce_quota(user)
        text = "\n\n".join(f"[{name}]\n" + "\n".join(c.text for c in chunks[:per_document]) for name, chunks in documents)
        before = getattr(llm, "tokens_used", 0)
        stream = getattr(llm, "stream", None)
        pieces = stream(OVERVIEW_RULES, text, 0.0) if stream else iter([llm.generate(OVERVIEW_RULES, text, 0.0)])
        parts = []
        for piece in pieces:
            parts.append(piece)
            yield "delta", piece
        tokens = getattr(llm, "tokens_used", 0) - before
        if quota is not None:
            quota.record(user, tokens)
        response = AskResponse(answer="".join(parts), cited_chunk_ids=[], refused_by_retrieval=False,
                               sources=[SourceItem(chunk_id=c.chunk_id, document_id=c.document_id, score=0.0, text=c.text)
                                        for c in shown], tokens_used=tokens, route="overview")
        if cache is not None:
            cache.put(question, response, scope)
        yield "done", response

    def answer_events(user: str, question: str, k: int):
        """The one place an ask is decided. Yields ("sources", [...]) and ("delta", text) while the model writes,
        and always ends with ("done", AskResponse). Answers that need no model (cache, lookups) just yield done."""
        index = retriever_for(user)
        if index is None:
            yield "done", AskResponse(answer=EMPTY_LIBRARY, cited_chunk_ids=[], refused_by_retrieval=True, sources=[])
            return
        if is_overview(question):
            yield from overview_events(user, question, index, k)
            return
        route = route_by_rules(question)
        if route == "agent":  # expensive and multi-step: do not run it behind the user's back
            yield "done", AskResponse(answer="This question needs multi-step research. Submit it to POST /research.",
                                      cited_chunk_ids=[], refused_by_retrieval=False, sources=[], route="agent")
            return
        if route == "retrieval":  # a lookup: return the best matches, no LLM at all
            results = index.search(question, k=k)
            docs = list(dict.fromkeys(r.chunk.document_id for r in results))
            yield "done", AskResponse(answer="Top matching documents: " + ", ".join(docs),
                                      cited_chunk_ids=[], refused_by_retrieval=False,
                                      sources=[to_source(r) for r in results], route="retrieval")
            return

        # Answers depend on k, on whose documents were searched and on which documents they currently are.
        scope = f"{user}|{library.fingerprint(user) if library else '-'}|k={k}"
        if cache is not None:
            hit = cache.get(question, scope)
            if hit is not None:  # cached answers are free: they never touch the quota
                cached, kind = hit
                yield "done", cached.model_copy(update={"tokens_used": 0, "cache_hit": kind})
                return
        enforce_quota(user)
        out = None
        for kind, payload in RAG(index, llm, k=k).ask_stream(question):
            if kind == "sources":
                yield "sources", [to_source(r) for r in payload]
            elif kind == "delta":
                yield "delta", payload
            else:
                out = payload
        if quota is not None:
            quota.record(user, out.tokens_used)
        response = AskResponse(
            answer=out.answer,
            cited_chunk_ids=out.cited_chunk_ids,
            refused_by_retrieval=out.refused_by_retrieval,
            sources=[to_source(r) for r in out.retrieved],
            tokens_used=out.tokens_used,
        )
        if cache is not None:
            cache.put(question, response, scope)
        yield "done", response

    def answer_question(user: str, question: str, k: int) -> AskResponse:
        for kind, payload in answer_events(user, question, k):
            if kind == "done":
                return payload

    @app.post("/ask", response_model=AskResponse)
    def ask(req: AskRequest, user: str = Depends(identify)):
        return answer_question(user, req.question, req.k)

    def run_job(job_id: str, question: str, user: str, message_id: int | None = None) -> None:
        def on_call(record) -> None:  # progress: record each tool call the moment it happens
            info = ToolCallInfo(step=record.step, tool=record.tool, arguments=record.arguments, is_error=record.is_error)
            current = jobs.get(job_id)
            if current is not None:  # expired or evicted: nothing to update
                jobs.save(current.model_copy(update={"tool_calls": [*current.tool_calls, info]}))
            if message_id is not None and chats is not None:
                meta = chats.message_meta(message_id)
                chats.update_message(message_id, meta={**meta, "tool_calls": [*meta.get("tool_calls", []), info.model_dump()]})

        try:
            agent = user_agent_factory(user) if user_agent_factory else agent_factory()
            result = agent.run(question, on_call=on_call)
            tokens = getattr(agent.llm, "tokens_used", 0)
            if quota is not None:
                quota.record(user, tokens)
            calls = [ToolCallInfo(step=c.step, tool=c.tool, arguments=c.arguments, is_error=c.is_error)
                     for c in result.calls]
            jobs.save(ResearchStatus(job_id=job_id, status="done", owner=user, answer=result.answer,
                                     stopped_because=result.stopped_because, tool_calls=calls, tokens_used=tokens))
            if message_id is not None and chats is not None:
                meta = chats.message_meta(message_id)
                chats.update_message(message_id, content=normalize_agent_citations(result.answer), status="done", meta={
                    **meta, "tool_calls": [c.model_dump() for c in calls], "tokens_used": tokens,
                    "stopped_because": result.stopped_because})
        except Exception as e:  # a failed job must be reported, never lost
            log.exception("research job %s failed", job_id)
            jobs.save(ResearchStatus(job_id=job_id, status="failed", error=str(e)[:300], owner=user))
            if message_id is not None and chats is not None:
                chats.update_message(message_id, content="The research could not be completed. Please try again.",
                                     status="failed")

    @app.post("/research", response_model=ResearchStarted, status_code=202)
    def start_research(req: ResearchRequest, background: BackgroundTasks, user: str = Depends(identify)):
        if agent_factory is None and user_agent_factory is None:
            raise HTTPException(status_code=503, detail="research agent is not configured")
        if retriever_for(user) is None:
            raise HTTPException(status_code=409, detail=EMPTY_LIBRARY)
        enforce_quota(user)
        job_id = uuid.uuid4().hex[:12]
        jobs.save(ResearchStatus(job_id=job_id, status="running", owner=user))
        background.add_task(run_job, job_id, req.question, user)
        return ResearchStarted(job_id=job_id, status="running")

    @app.get("/research/{job_id}", response_model=ResearchStatus)
    def research_status(job_id: str, user: str = Depends(identify)):
        status = jobs.get(job_id)
        # Someone else's job answers exactly like a job that does not exist, so ids cannot be probed.
        if status is None or (status.owner is not None and status.owner != user):
            raise HTTPException(status_code=404, detail="unknown job_id")
        return status.model_copy(update={"owner": None})

    # ---- the user's document library ----------------------------------------------

    if library is not None:
        @app.get("/library/documents")
        def list_documents(user: str = Depends(identify)):
            return library.documents(user)

        @app.put("/library/documents/{filename}", status_code=201)
        async def upload_document(filename: str, request: Request, user: str = Depends(identify)):
            if int(request.headers.get("content-length") or 0) > library.max_bytes:
                raise HTTPException(status_code=413, detail="The file is too large.")
            data = await request.body()
            try:
                return await run_in_threadpool(library.add, user, filename, data)
            except DuplicateDocument as e:
                raise HTTPException(status_code=409, detail=str(e))
            except LibraryError as e:
                raise HTTPException(status_code=400, detail=str(e))

        @app.post("/library/check")
        def check_files(req: CheckFiles, user: str = Depends(identify)):
            return {"duplicates": library.known_files(user, req.hashes)}

        @app.delete("/library/documents/{doc_id}")
        def delete_document(doc_id: str, user: str = Depends(identify)):
            if not library.delete(user, doc_id):
                raise HTTPException(status_code=404, detail="unknown document")
            return {"ok": True}

        @app.get("/library/map")
        def corpus_map(user: str = Depends(identify)):
            return library.corpus_map(user)

    # ---- saved chats ----------------------------------------------------------------

    if chats is not None:
        def chat_or_404(user: str, chat_id: int) -> dict:
            chat = chats.get(user, chat_id)
            if chat is None:  # someone else's chat looks exactly like a missing one
                raise HTTPException(status_code=404, detail="unknown chat")
            return chat

        @app.get("/chats")
        def list_chats(user: str = Depends(identify)):
            return chats.list(user)

        @app.post("/chats", status_code=201)
        def create_chat(user: str = Depends(identify)):
            return chats.create(user)

        @app.get("/chats/{chat_id}")
        def get_chat(chat_id: int, user: str = Depends(identify)):
            return chat_or_404(user, chat_id)

        @app.patch("/chats/{chat_id}")
        def rename_chat(chat_id: int, req: RenameChat, user: str = Depends(identify)):
            if not chats.rename(user, chat_id, req.title):
                raise HTTPException(status_code=404, detail="unknown chat")
            return chat_or_404(user, chat_id)

        @app.delete("/chats/{chat_id}")
        def delete_chat(chat_id: int, user: str = Depends(identify)):
            if not chats.delete(user, chat_id):
                raise HTTPException(status_code=404, detail="unknown chat")
            return {"ok": True}

        def recent_turns(messages: list[dict]) -> str:
            return "\n".join(f"{m['role']}: {m['content'][:300]}" for m in messages[-6:] if m["status"] == "done")

        def standalone_question(user: str, turns: str, question: str) -> str:
            """Turn "and what about its cost?" into a question that makes sense alone, so retrieval can find it."""
            if not turns or len(question.split()) > FOLLOW_UP_WORDS:
                return question
            before = getattr(llm, "tokens_used", 0)
            try:
                rewritten = llm.generate(REWRITE_RULES, f"Conversation:\n{turns}\n\nLast question: {question}\n\n"
                                         "Standalone question:", 0.0).strip().strip('"')
            except Exception:  # memory is a convenience: fall back to the question as typed
                return question
            if quota is not None:
                quota.record(user, getattr(llm, "tokens_used", 0) - before)
            return rewritten if 0 < len(rewritten) <= 300 else question

        def begin_turn(user: str, chat_id: int, req: SendMessage) -> str:
            """Checks the chat is usable, saves the user's message, and returns the earlier conversation as text."""
            chat = chat_or_404(user, chat_id)
            if any(m["status"] == "running" for m in chat["messages"]):
                raise HTTPException(status_code=409, detail="Wait for the current answer to finish.")
            if req.mode == "research" and agent_factory is None and user_agent_factory is None:
                raise HTTPException(status_code=503, detail="research agent is not configured")
            turns = recent_turns(chat["messages"])
            if chats.is_untitled(user, chat_id):
                chats.rename(user, chat_id, req.content)
            chats.add_message(user, chat_id, "user", req.content, meta={"mode": req.mode})
            return turns

        def sse(event: str, data) -> str:
            return f"event: {event}\ndata: {json.dumps(data)}\n\n"

        def ask_meta(result: AskResponse, question: str, content: str) -> dict:
            return {"mode": "ask", "standalone": question if question != content else None, "route": result.route,
                    "cache_hit": result.cache_hit, "tokens_used": result.tokens_used,
                    "refused": result.refused_by_retrieval, "cited_chunk_ids": result.cited_chunk_ids,
                    "sources": [s.model_dump() for s in result.sources]}

        @app.post("/chats/{chat_id}/stream")
        def stream_message(chat_id: int, req: SendMessage, user: str = Depends(identify)):
            """Answer in Ask mode as a stream of server-sent events: sources, then text pieces, then the saved chat."""
            if req.mode != "ask":
                raise HTTPException(status_code=400, detail="Only Ask mode streams; Research runs in the background.")
            turns = begin_turn(user, chat_id, req)  # errors here are ordinary HTTP errors, before streaming starts
            message_id = chats.add_message(user, chat_id, "assistant", "", status="running", meta={"mode": "ask"})

            def events():
                pieces: list[str] = []
                finished = False
                try:
                    question = standalone_question(user, turns, req.content)
                    if question != req.content:
                        yield sse("searching", {"question": question})
                    final = None
                    for kind, payload in answer_events(user, question, 3):
                        if kind == "sources":
                            yield sse("sources", [s.model_dump() for s in payload])
                        elif kind == "delta":
                            pieces.append(payload)
                            yield sse("delta", {"text": payload})
                        else:
                            final = payload
                    if final.route == "agent":
                        final = final.model_copy(update={"answer": "This looks like a multi-step question. Switch to "
                                                         "Research mode (next to the send button) and ask again."})
                    chats.update_message(message_id, content=final.answer, status="done",
                                         meta=ask_meta(final, question, req.content))
                    finished = True
                except HTTPException as e:  # e.g. the daily token limit: shown in the chat, not lost
                    chats.update_message(message_id, content=str(e.detail), status="failed")
                    finished = True
                except Exception:
                    log.exception("streaming answer failed")
                    chats.update_message(message_id, content="Something went wrong while answering. Please try again.",
                                         status="failed")
                    finished = True
                finally:
                    if not finished:  # the reader pressed Stop or left: keep what was written so far
                        chats.update_message(message_id, content="".join(pieces) or "Stopped.", status="done",
                                             meta={"mode": "ask", "stopped": True})
                yield sse("done", chats.get(user, chat_id))

            return StreamingResponse(events(), media_type="text/event-stream",
                                     headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

        @app.post("/chats/{chat_id}/messages")
        def send_message(chat_id: int, req: SendMessage, background: BackgroundTasks, user: str = Depends(identify)):
            turns = begin_turn(user, chat_id, req)

            def reply(content: str, status: str = "done", meta: dict | None = None) -> None:
                chats.add_message(user, chat_id, "assistant", content, status=status, meta={"mode": req.mode, **(meta or {})})

            if retriever_for(user) is None:
                reply(EMPTY_LIBRARY, meta={"refused": True})
            elif req.mode == "research":
                try:
                    enforce_quota(user)
                except HTTPException as e:
                    reply(str(e.detail), status="failed")
                else:
                    job_id = uuid.uuid4().hex[:12]
                    jobs.save(ResearchStatus(job_id=job_id, status="running", owner=user))
                    message_id = chats.add_message(user, chat_id, "assistant", "", status="running",
                                                   meta={"mode": "research", "job_id": job_id, "tool_calls": []})
                    task = f"Earlier in this conversation:\n{turns}\n\nNew question: {req.content}" if turns else req.content
                    background.add_task(run_job, job_id, task, user, message_id)
            else:
                try:
                    question = standalone_question(user, turns, req.content)
                    result = answer_question(user, question, 3)
                    if result.route == "agent":
                        result = result.model_copy(update={"answer": "This looks like a multi-step question. Switch "
                                                           "to Research mode (next to the send button) and ask again."})
                    reply(result.answer, meta=ask_meta(result, question, req.content))
                except HTTPException as e:  # e.g. the daily token limit: shown in the chat, not lost
                    reply(str(e.detail), status="failed")
                except Exception:
                    log.exception("answering failed")
                    reply("Something went wrong while answering. Please try again.", status="failed")
            return chat_or_404(user, chat_id)

    return app
