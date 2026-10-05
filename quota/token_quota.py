from datetime import date


class QuotaExceeded(Exception):
    pass


class TokenQuota:
    """Daily LLM-token budget per user, kept in memory (a shared store is needed with many servers)."""

    def __init__(self, daily_limit: int = 50_000):
        self.daily_limit = daily_limit
        self.used: dict[tuple[str, date], int] = {}

    def check(self, user: str) -> None:
        """Raise QuotaExceeded if the user has already used up today's budget."""
        if self.used.get((user, date.today()), 0) >= self.daily_limit:
            raise QuotaExceeded(f"daily token limit of {self.daily_limit} reached")

    def record(self, user: str, tokens: int) -> None:
        key = (user, date.today())
        self.used[key] = self.used.get(key, 0) + tokens

    def remaining(self, user: str) -> int:
        return max(0, self.daily_limit - self.used.get((user, date.today()), 0))


class RedisTokenQuota(TokenQuota):
    """The same daily budget, counted in Redis so every server shares one tally per user.
    INCRBY is atomic, so two servers recording at once cannot lose an update."""

    def __init__(self, client, daily_limit: int = 50_000, prefix: str = "quota") -> None:
        super().__init__(daily_limit)
        self.client, self.prefix = client, prefix

    def _key(self, user: str) -> str:
        return f"{self.prefix}:{user}:{date.today().isoformat()}"

    def _used(self, user: str) -> int:
        return int(self.client.get(self._key(user)) or 0)

    def check(self, user: str) -> None:
        if self._used(user) >= self.daily_limit:
            raise QuotaExceeded(f"daily token limit of {self.daily_limit} reached")

    def record(self, user: str, tokens: int) -> None:
        key = self._key(user)
        pipe = self.client.pipeline()
        pipe.incrby(key, tokens)
        pipe.expire(key, 2 * 24 * 3600)  # yesterday's counters clean themselves up
        pipe.execute()

    def remaining(self, user: str) -> int:
        return max(0, self.daily_limit - self._used(user))
