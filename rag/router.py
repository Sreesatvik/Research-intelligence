VISUAL_WORDS = ("table", "figure", "diagram", "chart", "plot", "illustration",
                "image", "picture", "screenshot")
NUMERIC_WORDS = ("score", "accuracy", "exact match", "f1", "how many",
                 "percent", "%", "number of")


def route(question: str) -> str:
    """Return 'visual' if the question likely needs a page image, else 'text'."""
    q = question.lower()
    if any(w in q for w in VISUAL_WORDS) or any(w in q for w in NUMERIC_WORDS):
        return "visual"
    return "text"
