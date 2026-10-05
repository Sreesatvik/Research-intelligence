# Research Intelligence

Chat with your own documents. Upload PDFs, text or Markdown files, then ask questions and get answers that cite
the exact passages they came from. Everything is private to your account: search, chats and the document map only
ever see your own library.

- **Ask mode**: fast answers from the best-matching passages, streamed as they are written.
- **Research mode**: an agent that searches, reads and compares documents over several steps, showing each step live.
- **Library**: upload with live checks (format, size, duplicates) before anything is stored.
- **Corpus map**: your documents as a 3D map, grouped by topic.
- **Saved chats** with follow-up memory, plus email/password and Google sign-in.

## Contents

1. [Quick start](#quick-start)
2. [How it works](#how-it-works)
3. [Architecture](#architecture)
4. [Evaluation and benchmarks](#evaluation-and-benchmarks)
5. [Design decisions and trade-offs](#design-decisions-and-trade-offs)
6. [Security and privacy](#security-and-privacy)
7. [API](#api)
8. [Project layout](#project-layout)
9. [Testing](#testing)
10. [Limitations](#limitations)

## Quick start

You need Docker, and a [Groq](https://console.groq.com) API key for the language model.

```bash
cp .env.example .env        # then edit .env: set GROQ_API_KEY and AUTH_SECRET
docker compose up -d --build
```

Open http://localhost:8080, create an account, add documents in **Library**, then ask in **Chat**.

`AUTH_SECRET` signs session cookies and must be at least 32 random characters:

```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

### Running without Docker (development)

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m uvicorn apps.api.server:app --port 8000
```

In a second terminal:

```bash
cd ui
npm install
npm run dev                 # http://localhost:5173, proxies /api to port 8000
```

### Configuration (`.env`)

| Variable | Purpose |
|---|---|
| `GROQ_API_KEY` | Language model access. Required. Never committed; the Docker image does not contain it. |
| `AUTH_SECRET` | Signs session cookies (32+ characters). If unset, a random one is used and everyone is signed out on each restart. |
| `REDIS_URL` | Optional. Shares the cache, daily quotas and research jobs between servers and across restarts. Docker Compose sets it. |
| `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, `GOOGLE_REDIRECT_URI` | Optional Google sign-in. Without them the button is hidden. |
| `DATABASE_PATH` | SQLite file for accounts, chats and the document list. Uploaded files are stored beside it in `users/`. |
| `COOKIE_SECURE` | Set to `1` once the site is served over HTTPS. |
| `AUTH_ENABLED` | `0` turns accounts off for local experiments only. The API is then open to anyone. |

To enable Google sign-in, create an OAuth client of type **Web application** in the Google Cloud Console and add
`http://localhost:8080/api/auth/google/callback` (and `http://localhost:5173/api/auth/google/callback` for the dev
server) under **Authorized redirect URIs**. Leave "Authorized JavaScript origins" empty: the flow runs on the server.

## How it works

### 1. Adding a document

1. The browser checks the file (real PDF header, size, not selected twice) and asks the server whether the same
   file is already in your library, using a SHA-256 fingerprint of its bytes.
2. The server parses it (PDFs with PyMuPDF, including tables and figure captions) and cuts the text into
   overlapping **passages** of about 100 words.
3. Each passage is turned into a 384-number **embedding** with `all-MiniLM-L6-v2`, so passages with similar meaning
   have similar vectors even when they share no words.
4. A second fingerprint of the normalised *text* catches the same document saved in another format.
5. Files, passages and vectors are stored under your account only.

### 2. Asking a question (Ask mode, "RAG")

1. A short follow-up ("and what about cats?") is rewritten into a standalone question using the recent chat.
2. The question is embedded and compared with every passage by cosine similarity. The top 3 are kept.
3. **Retrieval gate**: if even the best passage scores below 0.25, the answer is "I don't know" and the language
   model is never called. This costs nothing and prevents invented answers.
4. Otherwise the passages are numbered `[1] [2] [3]` and sent to the model with the rule "answer only from these and
   cite the numbers". The answer streams back token by token.
5. Citations are normalised and shown with a Sources list so every claim can be checked.

Questions about the whole library ("brief all pdfs") cannot be answered from three matching passages, so they take
an **overview path**: the model is shown the opening of every document and writes one or two sentences for each.

### 3. Research mode (the agent)

For multi-step questions ("compare these two"), an agent loops: the model chooses a tool (`semantic_search` or
`retrieve_document`), the server runs it, and the result goes back to the model, for at most 6 steps. Each tool call
is shown live in the chat. Tool arguments are validated with Pydantic, tool output is wrapped as untrusted data, and a
loop guard stops repeated identical calls. If the step budget runs out, the agent writes a final answer from what it
has found.

### 4. Cost control

Language-model tokens are the only per-use cost, so the system tries the cheapest option first:

| Layer | Effect |
|---|---|
| Retrieval gate | Weak evidence is refused without a model call |
| Rule router | Lookups ("list documents about X") return search results with no model call; multi-step questions are sent to Research mode instead of being run silently |
| Answer cache | An exact match is free; a semantic match needs a very high similarity and the same numbers/names |
| Daily token quota | Per user, so one account cannot run up the bill |

Cache keys include the user and the current set of documents, so answers never leak between users and never go
stale after a document is added or deleted.

### 5. The corpus map

Each document becomes the average of its passage vectors. K-means groups them (best of 10 random starts, 1 to 7
groups depending on library size), the most distinctive words name each group, and PCA flattens the vectors to
three dimensions for the 3D view.

## Architecture

```
 Browser (React, Three.js, GSAP)
    │  /api  (nginx in Docker, Vite proxy in dev)
    ▼
 FastAPI ── identify user (signed cookie) ── quota ── route
    │
    ├── library/   per-user files, passages, embeddings, duplicate checks, corpus map
    ├── rag/       retrieve → gate → prompt → stream → citations
    ├── agent/     tool loop with typed tools and untrusted-output handling
    ├── chats/     saved conversations (SQLite)
    ├── auth/      scrypt passwords, signed sessions, Google OAuth (PKCE)
    ├── cache/     exact + guarded semantic answer cache
    └── quota/     daily token budget
         │
   SQLite file · Redis (optional, shared state) · Groq (language model)
```

| Part | Technology |
|---|---|
| Frontend | React 19, TypeScript, Vite, TanStack Query, react-router, GSAP, Lenis, Three.js |
| API | FastAPI, Pydantic, server-sent events for streaming |
| Retrieval | sentence-transformers (local), NumPy cosine search |
| Language model | Groq (`openai/gpt-oss-120b`) |
| Storage | SQLite (accounts, chats, document list), files on disk, Redis (optional) |
| Packaging | Docker Compose: nginx + API + Redis |

## Evaluation and benchmarks

The benchmark corpus is 47 arXiv papers on seven topics (RAG, prompt injection, web agents, sentence embeddings,
topic modelling, vision transformers, efficient transformers) with 22 hand-written queries. It lives in
`data/corpus_arxiv` and `data/eval` and is used for evaluation only; the running app does not load it.
All numbers below were produced on a CPU by experiment scripts named `demo_*.py`. Those scripts were removed from
the working tree to keep the project tidy; they are in the first commit of the git history
(`git show --stat <first-commit>`) if you want to re-run any of them.

### Retrieval: dense embeddings vs TF-IDF

`demo_eval_arxiv.py` (115 passages of ~100 words, 22 queries)

| k | Engine | Recall@k | Precision@k | MRR | ms/query |
|---|---|---|---|---|---|
| 1 | TF-IDF | 0.568 | 0.591 | 0.636 | 1.1 |
| 1 | Dense | **0.795** | 0.818 | **0.902** | 10.0 |
| 3 | TF-IDF | 0.682 | 0.242 | 0.680 | 0.8 |
| 3 | Dense | **0.977** | 0.348 | **0.902** | 9.0 |
| 5 | TF-IDF | 0.773 | 0.164 | 0.684 | 0.8 |
| 5 | Dense | **1.000** | 0.218 | **0.902** | 9.0 |

Dense retrieval is clearly better at finding the right paper; TF-IDF is about ten times faster. The app uses dense.

### Passage size

`demo_chunk_size.py` (Recall@3 / MRR, dense)

| Words per passage | Passages | Recall@3 | MRR |
|---|---|---|---|
| 30 | 351 | 0.955 | 0.780 |
| 50 | 214 | 0.909 | 0.861 |
| **100** | 115 | **0.977** | **0.902** |
| 200 | 59 | 0.955 | 0.848 |
| 1000 | 47 | 0.955 | 0.848 |

100 words was best on both measures, so it is the default.

### Refusing when there is no answer

`demo_threshold.py`. Best-match scores: answerable questions 0.31 to 0.76 (median 0.60); unanswerable
questions 0.09 to 0.42 (median 0.18).

| Threshold | Answerable kept | Unanswerable refused |
|---|---|---|
| 0.15 | 100% | 50% |
| 0.20 | 100% | 58% |
| **0.25** | **100%** | **67%** |
| 0.30 | 100% | 75% |

0.25 keeps every answerable question while refusing two thirds of unanswerable ones for free. The remainder reach
the model, which is told to refuse when the sources do not support an answer. Going higher starts risking
legitimate questions (the lowest answerable score was 0.31).

### Cost routing

`demo_cost_router.py` (45 labelled questions; "tokens" are estimated averages)

| Strategy | Accuracy | Under-routed | Over-routed | Avg tokens |
|---|---|---|---|---|
| Always agent | 0.33 | 0/45 | 30 | 4500 |
| Always plain RAG | 0.33 | 15/45 | 15 | 850 |
| Perfect oracle | 1.00 | 0 | 0 | 1783 |
| **Keyword rules (used)** | **0.98** | **0/45** | 1 | 1883 |
| Learned classifier (5x5 CV) | 0.73 | 26/225 | 34 | 1903 |

Simple whole-word rules beat a learned classifier at this data size and never send a hard question to the cheap
path. The single miss was a lookup sent to the agent (safe but costlier). "Under-routed" means a multi-step
question sent to plain RAG, which gives a poor answer, so it is the number to keep at zero.

### Topic classification and clustering

`demo_classify.py` (25 cross-validation folds over the 47 papers)

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Always the majority class | 0.236 | 0.054 |
| Nearest centroid | 0.931 | 0.925 |
| Softmax classifier (written from scratch) | 0.939 | 0.924 |
| scikit-learn LogisticRegression | 0.939 | 0.924 |

The from-scratch classifier matches scikit-learn. Most errors are between neighbouring topics (prompt injection
vs web agents).

`demo_clustering.py` and `demo_choose_k.py`: k-means with 7 groups reaches an Adjusted Rand Index of
0.72 against hand-labelled topics (best of 10 starts). ARI peaks around k = 6 to 8, which matches the seven real
topics; tightness always improves with more groups, so it cannot choose k by itself.

### Semantic cache threshold

`demo_cache_threshold.py`. Same-meaning pairs scored 0.61 to 0.97, but near-miss pairs that must **not** hit
(for example "FLARE on StrategyQA" vs "FLARE on ASQA") scored up to 0.94. The ranges overlap, so similarity alone is
unsafe. The cache therefore requires a very high threshold (0.96) **and** that numbers, acronyms and names match
between the two questions.

### Agent and multimodal results (earlier runs)

These need live model calls and were measured during development rather than re-run for this README:

- Prompt-injection test: 0 of 6 injected documents changed the agent's behaviour.
- Trimming old tool results from the agent's history blindly raised token use by about 48%, so it is off by
  default; the loop guard stayed.
- PDF tables and figures are answered by a vision model on a cropped region; scripts `demo_table_qa.py`,
  `demo_vision_vs_text.py` and `demo_final_vision.py` produced those runs. The vision path is not wired into the
  web app.

## Design decisions and trade-offs

- **Dense over TF-IDF**: much better recall (above), at a small latency cost.
- **Retrieval gate before the model**: it is the cheapest way to avoid both cost and hallucination.
- **Rules over a learned router**: more accurate here, easy to read and debug, and failures are predictable.
- **Cache safety over hit rate**: a wrong cached answer is worse than a miss.
- **Per-user everything**: libraries, indexes, cache keys, chats and jobs are scoped by user id from the start,
  rather than added later.
- **Local embeddings**: documents are never sent to a third party for embedding; only the few retrieved passages go
  to the language model.
- **Background jobs for Research, streaming for Ask**: a research run can take many seconds and several tool calls,
  so it runs in the background and the page follows it; a plain answer streams directly.
- **SQLite**: zero setup and fine for one server. Moving to several API servers means moving chats, documents and
  accounts to a networked database.

## Security and privacy

- Passwords are stored only as scrypt hashes (per-user salt). Unknown email and wrong password give the same
  message and take the same time. Repeated failures lock the account temporarily.
- Sessions are signed tokens (algorithm pinned) in an `httpOnly`, `SameSite=Lax` cookie, invisible to page scripts.
- Google sign-in uses the authorization-code flow with PKCE, `state` and `nonce`, requires a verified email, validates
  the ID token, and blocks open redirects. Signing in with Google takes over a password account that had the same
  email and revokes its unverified password.
- Another user's chat, job or document behaves exactly like a missing one (404), so ids cannot be probed.
- Uploads are checked in memory before anything touches disk; file names are reduced to a safe slug; each user's
  files live in a directory named by a hash of their id, so no input can become a path.
- Agent tool results are wrapped as untrusted data and the model is told never to follow instructions inside them.
- The API key is read from `.env` and is not copied into the Docker image.
- Before real deployment: serve over HTTPS and set `COOKIE_SECURE=1`.

## API

All endpoints except `/health` and `/auth/*` require a signed-in session.

| Method and path | Purpose |
|---|---|
| `GET /health` | Liveness check |
| `POST /auth/register`, `/auth/login`, `/auth/logout`, `GET /auth/me`, `/auth/providers` | Accounts |
| `GET /auth/google/start`, `/auth/google/callback` | Google sign-in (only when configured) |
| `GET /library/documents` | List your documents |
| `PUT /library/documents/{filename}` | Upload a file (raw body). `409` if it is a duplicate |
| `DELETE /library/documents/{id}` | Remove a document |
| `POST /library/check` | Which file fingerprints are already in your library |
| `GET /library/map` | 3D points and clusters for your documents |
| `GET/POST /chats`, `GET/PATCH/DELETE /chats/{id}` | Saved chats |
| `POST /chats/{id}/messages` | Send a message (Ask or Research). Research returns immediately; poll the chat |
| `POST /chats/{id}/stream` | Ask mode as server-sent events: `sources`, `delta`, `done` |
| `POST /ask`, `POST /search`, `POST /research`, `GET /research/{job_id}` | Stateless building blocks |

## Project layout

```
apps/api/       FastAPI app, auth routes, job store, server wiring
library/        per-user documents, indexes, duplicate detection, corpus map
chats/          saved conversations
rag/            pipeline, router, prompts, context builder, page locator, multimodal
agent/          research agent and typed tools
generation/     Groq text and vision clients
ingestion/      loaders, chunking, PDF parser
retrieval/      dense and TF-IDF retrievers
representations/ TF-IDF implementation
classification/ topic classifier (from scratch + sklearn comparison)
clustering/     k-means, cluster naming, PCA projection
cache/ quota/   answer cache and token quota
auth/           passwords, tokens, users, login limiter, Google OAuth
evaluation/     metrics and evaluation runners
data/eval/      benchmark queries and topic labels
ui/             React website (chat, library, map, home)
tests/          automated tests
```

## Testing

```bash
.venv\Scripts\python -m pytest tests
```

205 tests pass; 13 are skipped unless a real Redis server is available (the same behaviour is tested against
`fakeredis` otherwise). They cover retrieval and RAG behaviour, the agent loop and its failure modes, tools, the
cache, quotas and routing, accounts and Google sign-in (including attack cases), per-user isolation, duplicate
detection, chat memory, streaming and stopping mid-answer, and migration of older databases.

Frontend checks: `cd ui && npx tsc -b && npx oxlint && npm run build`.

## Limitations

- **Scale**: SQLite and in-process search suit one server. Many users or very large libraries need a networked
  database and a vector index.
- **Overviews read only the opening** of each document (title and abstract for papers).
- **Scanned PDFs** have no selectable text and are rejected; there is no OCR.
- **Research mode is not streamed**; its answer appears when the agent finishes.
- **Quality depends on the model**: answers can still be wrong or incomplete; the Sources list is how to check.
- **Benchmarks are small** (47 papers, 22 queries, one domain), so differences of a few points are not conclusive.
- **Vision (tables and figures)** was evaluated separately and is not part of the web app.
- **Animations** were checked for logic but not visually in an automated way; the 3D scene and transitions have not
  been profiled on low-end devices.
- **Language**: embeddings and prompts are tuned for English text.
