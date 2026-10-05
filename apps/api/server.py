"""Production entry point: real retriever, LLM, cache, agent and quota. Run with uvicorn."""
import os

from agent.agent import ResearchAgent
from agent.tools.research_tools import build_tools

from dotenv import load_dotenv

from apps.api.auth import Auth
from apps.api.job_store import InMemoryJobStore, RedisJobStore
from apps.api.main import create_app
from apps.api.schemas import AskResponse
from auth.google import GoogleOAuth
from auth.limiter import LoginLimiter, RedisLoginLimiter
from auth.users import UserStore
from cache.answer_cache import AnswerCache
from generation.groq_llm import GroqLLM
from chats.store import ChatStore
from library.store import Library
from quota.token_quota import RedisTokenQuota, TokenQuota
from sentence_transformers import SentenceTransformer

load_dotenv()

MODEL = "openai/gpt-oss-120b"
DAILY_TOKENS = 50_000

DATABASE_PATH = os.environ.get("DATABASE_PATH", "data/app.db")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
embed = lambda text: embedder.encode(text, normalize_embeddings=True)  # noqa: E731

# Nothing is preloaded: each user uploads their own documents, and each user searches only those.
library = Library(os.path.join(os.path.dirname(DATABASE_PATH) or ".", "users"), DATABASE_PATH, embedder)
chats = ChatStore(DATABASE_PATH)

# Shared state: with REDIS_URL set, cache hits, quotas and research jobs are shared by every server and survive
# restarts. Without it they live in this process only (fine for local work). A bad URL stops the server at
# startup instead of silently falling back, which would hide the misconfiguration.
redis_url = os.environ.get("REDIS_URL")
if redis_url:
    import redis

    client = redis.Redis.from_url(redis_url, decode_responses=True)
    client.ping()
    cache = AnswerCache(embed=embed, backend=client, dumps=lambda r: r.model_dump_json(), loads=AskResponse.model_validate_json)
    quota = RedisTokenQuota(client, daily_limit=DAILY_TOKENS)
    job_store = RedisJobStore(client)
    print("state: Redis (shared across servers)")
else:
    cache = AnswerCache(embed=embed)
    quota = TokenQuota(daily_limit=DAILY_TOKENS)
    job_store = InMemoryJobStore()
    print("state: in-memory (set REDIS_URL to share state between servers)")
# Accounts. AUTH_ENABLED=0 switches them off for quick local experiments only: the API is then open to anyone.
auth = None
if os.environ.get("AUTH_ENABLED", "1") != "0":
    secret = os.environ.get("AUTH_SECRET", "")
    if not secret:
        import secrets

        secret = secrets.token_urlsafe(48)
        print("WARNING: AUTH_SECRET is not set. Using a random one: everyone is signed out whenever the server "
              "restarts, and several servers will not accept each other's sessions. Set AUTH_SECRET in .env.")
    google = None
    if os.environ.get("GOOGLE_CLIENT_ID") and os.environ.get("GOOGLE_CLIENT_SECRET"):
        google = GoogleOAuth(
            os.environ["GOOGLE_CLIENT_ID"],
            os.environ["GOOGLE_CLIENT_SECRET"],
            # Must match, character for character, a redirect URI registered in the Google Cloud console.
            os.environ.get("GOOGLE_REDIRECT_URI", "http://localhost:8080/api/auth/google/callback"),
        )
    auth = Auth(
        UserStore(DATABASE_PATH),
        secret,
        limiter=RedisLoginLimiter(client) if redis_url else LoginLimiter(),
        cookie_secure=os.environ.get("COOKIE_SECURE", "0") == "1",  # set to 1 once the site is served over HTTPS
        google=google,
    )
    print("auth: accounts enabled" + (" (Google sign-in on)" if google else " (Google sign-in off: no GOOGLE_CLIENT_ID)"))
else:
    print("auth: DISABLED (development only: anyone can call the API)")

AGENT_LOG = "data/logs/agent_calls.jsonl"


def user_agent_factory(user: str) -> ResearchAgent:
    """A fresh agent (and LLM object, so token counts are per job) whose tools see only this user's documents."""
    index = library.retriever(user)
    registry = build_tools(index, index.chunks)
    return ResearchAgent(GroqLLM(MODEL), registry, max_steps=6, log_path=AGENT_LOG)


app = create_app(None, GroqLLM(MODEL), cache=cache, quota=quota, job_store=job_store, auth=auth,
                 library=library, chats=chats, user_agent_factory=user_agent_factory)
