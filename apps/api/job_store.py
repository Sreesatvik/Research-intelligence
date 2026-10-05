from apps.api.schemas import ResearchStatus


class InMemoryJobStore:
    """Jobs live in this process only: fine for local work, lost on restart, not shared between servers."""

    def __init__(self) -> None:
        self._jobs: dict[str, ResearchStatus] = {}

    def save(self, status: ResearchStatus) -> None:
        self._jobs[status.job_id] = status

    def get(self, job_id: str) -> ResearchStatus | None:
        return self._jobs.get(job_id)


class RedisJobStore:
    """Jobs live in Redis: any server can report on a job another server started, and they survive restarts.
    Every job expires after `ttl_seconds`, so finished work cannot pile up forever."""

    def __init__(self, client, ttl_seconds: int = 3600, prefix: str = "job") -> None:
        self.client, self.ttl, self.prefix = client, ttl_seconds, prefix

    def save(self, status: ResearchStatus) -> None:
        self.client.set(f"{self.prefix}:{status.job_id}", status.model_dump_json(), ex=self.ttl)

    def get(self, job_id: str) -> ResearchStatus | None:
        raw = self.client.get(f"{self.prefix}:{job_id}")
        return ResearchStatus.model_validate_json(raw) if raw else None
