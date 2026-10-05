import time


class LoginLimiter:
    """Blocks an account after too many failed logins in a window. Counts live in this process only."""

    def __init__(self, max_attempts: int = 5, window_seconds: int = 900) -> None:
        self.max_attempts, self.window = max_attempts, window_seconds
        self._failures: dict[str, list[float]] = {}

    def _recent(self, key: str) -> list[float]:
        cutoff = time.time() - self.window
        self._failures[key] = [t for t in self._failures.get(key, []) if t > cutoff]
        return self._failures[key]

    def blocked(self, key: str) -> bool:
        return len(self._recent(key)) >= self.max_attempts

    def fail(self, key: str) -> None:
        self._recent(key).append(time.time())

    def reset(self, key: str) -> None:
        self._failures.pop(key, None)


class RedisLoginLimiter(LoginLimiter):
    """The same limit, counted in Redis so every server enforces it together."""

    def __init__(self, client, max_attempts: int = 5, window_seconds: int = 900, prefix: str = "loginfail") -> None:
        super().__init__(max_attempts, window_seconds)
        self.client, self.prefix = client, prefix

    def _key(self, key: str) -> str:
        return f"{self.prefix}:{key}"

    def blocked(self, key: str) -> bool:
        return int(self.client.get(self._key(key)) or 0) >= self.max_attempts

    def fail(self, key: str) -> None:
        pipe = self.client.pipeline()
        pipe.incr(self._key(key))
        pipe.expire(self._key(key), self.window)  # the lock lifts by itself
        pipe.execute()

    def reset(self, key: str) -> None:
        self.client.delete(self._key(key))
