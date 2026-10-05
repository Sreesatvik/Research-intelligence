import re
from collections import defaultdict

from core.schemas import Chunk, SearchResult

LABEL_RE = re.compile(r"\b(table|figure|fig\.?)\s*(\d+)", re.I)


class PageLocator:
    """Decide which (document, page) to show the vision model."""

    def __init__(self, chunks: list[Chunk]):
        # (document_id, "figure 1") -> page, from caption chunks
        self.captions = {(c.document_id, c.metadata["label"]): c.page
                         for c in chunks if c.content_type == "caption"}

    def top1(self, results: list[SearchResult]) -> tuple[str, int] | None:
        """Baseline: the page of the single best chunk."""
        best = results[0].chunk
        return (best.document_id, best.page) if best.page else None

    def vote(self, results: list[SearchResult]) -> tuple[str, int] | None:
        """Sum scores per (document, page) across all retrieved chunks."""
        totals = defaultdict(float)
        for r in results:
            if r.chunk.page:
                totals[(r.chunk.document_id, r.chunk.page)] += r.score
        return max(totals, key=totals.get) if totals else None

    def locate(self, question: str, results: list[SearchResult]) -> tuple[str, int] | None:
        """If the question names 'Table 2' / 'Figure 1', look that caption up; else vote."""
        m = LABEL_RE.search(question)
        if m and results:
            kind = "figure" if m.group(1).lower().startswith("fig") else "table"
            label = f"{kind} {m.group(2)}"
            doc = results[0].chunk.document_id  # the document retrieval believes in
            if (doc, label) in self.captions:
                return doc, self.captions[(doc, label)]
        return self.vote(results)
