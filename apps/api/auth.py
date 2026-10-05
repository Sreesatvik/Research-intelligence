import re

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field, field_validator

from auth.google import GoogleAuthError, GoogleIdentity, GoogleOAuth
from auth.limiter import LoginLimiter
from auth.passwords import DUMMY_HASH, hash_password, verify_password
from auth.tokens import check_secret, issue_token, read_token
from auth.users import NO_PASSWORD, EmailTaken, User, UserStore

EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
COOKIE_NAME = "session"
OAUTH_COOKIE = "oauth_attempt"
BAD_LOGIN = "Incorrect email or password"  # the same message whether the email exists or not


class _EmailMixin(BaseModel):
    email: str = Field(max_length=254)

    @field_validator("email")
    @classmethod
    def clean_email(cls, value: str) -> str:
        value = value.strip().lower()
        if not EMAIL_PATTERN.match(value):
            raise ValueError("enter a valid email address")
        return value


class RegisterRequest(_EmailMixin):
    password: str = Field(min_length=10, max_length=128)  # the cap stops absurdly long inputs from burning CPU


class LoginRequest(_EmailMixin):
    password: str = Field(min_length=1, max_length=128)  # no minimum here: a short password is simply wrong


class UserOut(BaseModel):
    email: str


class Auth:
    """Accounts and sessions. A session is a signed token in an httpOnly cookie, which page scripts cannot read."""

    def __init__(self, users: UserStore, secret: str, limiter: LoginLimiter | None = None,
                 cookie_secure: bool = False, ttl_seconds: int = 7 * 24 * 3600,
                 google: GoogleOAuth | None = None) -> None:
        check_secret(secret)
        self.users, self.secret, self.google = users, secret, google
        self.limiter = limiter or LoginLimiter()
        self.cookie_secure, self.ttl = cookie_secure, ttl_seconds

    def _claims(self, request: Request) -> dict | None:
        token = request.cookies.get(COOKIE_NAME)
        return read_token(self.secret, token) if token else None

    def require(self, request: Request) -> str:
        """The signed-in user's id, or HTTP 401."""
        claims = self._claims(request)
        if claims is None:
            raise HTTPException(status_code=401, detail="Not signed in")
        return str(claims["sub"])

    def _start_session(self, response: Response, user_id: int, email: str) -> None:
        response.set_cookie(
            COOKIE_NAME, issue_token(self.secret, user_id, email, self.ttl), max_age=self.ttl, path="/",
            httponly=True,  # invisible to page JavaScript, so an XSS bug cannot steal the session
            samesite="lax",  # not sent on cross-site POSTs, which blunts CSRF
            secure=self.cookie_secure,  # True once the site is served over HTTPS
        )

    def _google_user(self, identity: GoogleIdentity) -> User:
        """Find, link or create the account for a verified Google identity."""
        user = self.users.by_google_sub(identity.sub)  # a returning Google user: match on the permanent id
        if user is not None:
            return user
        existing = self.users.by_email(identity.email)
        if existing is not None:
            # Google proved who owns this email, so Google takes the account over and the password is revoked
            # (see UserStore.link_google for why).
            self.users.link_google(existing.id, identity.sub)
            return User(existing.id, existing.email, NO_PASSWORD, identity.sub)
        return self.users.create(identity.email, NO_PASSWORD, google_sub=identity.sub)

    def router(self) -> APIRouter:
        router = APIRouter(prefix="/auth", tags=["auth"])

        @router.get("/providers")
        def providers():
            return {"google": self.google is not None}  # the UI shows a Google button only when it can work

        if self.google is not None:
            self._add_google_routes(router)

        @router.post("/register", response_model=UserOut, status_code=201)
        def register(req: RegisterRequest, response: Response):
            try:
                user = self.users.create(req.email, hash_password(req.password))
            except EmailTaken:
                raise HTTPException(status_code=409, detail="An account with this email already exists")
            self._start_session(response, user.id, user.email)
            return UserOut(email=user.email)

        @router.post("/login", response_model=UserOut)
        def login(req: LoginRequest, response: Response):
            if self.limiter.blocked(req.email):
                raise HTTPException(status_code=429, detail="Too many failed attempts. Try again in a few minutes.")
            user = self.users.by_email(req.email)
            # Always verify against *some* hash, so an unknown email costs the same time as a wrong password.
            password_ok = verify_password(req.password, user.password_hash if user else DUMMY_HASH)
            if user is None or not password_ok:
                self.limiter.fail(req.email)
                raise HTTPException(status_code=401, detail=BAD_LOGIN)
            self.limiter.reset(req.email)
            self._start_session(response, user.id, user.email)
            return UserOut(email=user.email)

        @router.post("/logout")
        def logout(response: Response):
            response.delete_cookie(COOKIE_NAME, path="/")
            return {"ok": True}

        @router.get("/me", response_model=UserOut)
        def me(request: Request):
            claims = self._claims(request)
            if claims is None:
                raise HTTPException(status_code=401, detail="Not signed in")
            return UserOut(email=claims["email"])

        return router

    def _add_google_routes(self, router: APIRouter) -> None:
        google = self.google

        @router.get("/google/start")
        def google_start(next: str = "/ask"):
            url, attempt = google.start(self.secret, next)
            response = RedirectResponse(url, status_code=302)
            # Lax still sends this cookie on Google's redirect back (a top-level GET), but not on cross-site posts.
            response.set_cookie(OAUTH_COOKIE, attempt, max_age=600, path="/", httponly=True,
                                samesite="lax", secure=self.cookie_secure)
            return response

        @router.get("/google/callback")
        def google_callback(request: Request, code: str | None = None, state: str | None = None,
                            error: str | None = None):
            def failed() -> RedirectResponse:
                response = RedirectResponse("/login?error=google", status_code=302)
                response.delete_cookie(OAUTH_COOKIE, path="/")
                return response

            if error or not code:  # e.g. the person pressed "Cancel" on Google's screen
                return failed()
            try:
                identity, next_url = google.finish(self.secret, request.cookies.get(OAUTH_COOKIE), code, state)
            except GoogleAuthError:
                return failed()
            user = self._google_user(identity)
            response = RedirectResponse(next_url, status_code=302)
            self._start_session(response, user.id, user.email)
            response.delete_cookie(OAUTH_COOKIE, path="/")  # the attempt is over: it cannot be replayed
            return response
