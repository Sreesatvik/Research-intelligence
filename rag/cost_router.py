import re

# whole-word patterns: "collect" must not match "collection", "contrast" must not match "contrastive"
AGENT_PATTERN = re.compile(
    r"\b(compar\w*|contrast|differ\w*|classif\w*|then|for each|group|shared idea|collect|"
    r"most recent|read the)\b", re.I)
LOOKUP_STARTS = ("which papers", "list", "find papers", "find documents", "find articles",
                 "show", "are there any", "any papers", "papers about", "which documents")


# Questions about the whole library ("brief all pdfs", "summarize my documents") cannot be answered from the few
# passages that happen to match their words, so they get their own path. Common typos of "brief" are included.
OVERVIEW_ACTION = re.compile(r"\b(brie?f|breif|summar\w*|overview|outline|tl;?dr|describe|what(?:'s| is| are) in)\b", re.I)
OVERVIEW_SCOPE = re.compile(
    r"\b(all|each|every|these|my|the|your)\s+(?:\w+\s+)?(pdfs?|documents?|docs?|files?|papers?|library|uploads?)\b|\blibrary\b",
    re.I)


def is_overview(question: str) -> bool:
    """True when the user wants a summary of the whole collection rather than an answer from a few passages."""
    return bool(OVERVIEW_ACTION.search(question) and OVERVIEW_SCOPE.search(question))


def route_by_rules(question: str) -> str:
    """Return 'agent', 'retrieval' or 'rag' from simple whole-word rules."""
    q = question.lower().strip()
    if AGENT_PATTERN.search(q):
        return "agent"
    if q.startswith(LOOKUP_STARTS):
        return "retrieval"
    return "rag"
