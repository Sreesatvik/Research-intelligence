import re


def tokenize(text: str) -> list[str]:
    """Lowercase the text and split it into word tokens."""
    return re.findall(r"[a-z0-9]+", text.lower())


def term_counts(tokens: list[str]) -> dict[str, int]:
    """Map each word to the number of times it appears."""
    counts = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    return counts


if __name__ == "__main__":
    tokens = tokenize("The cat sat on the mat.")
    print(tokens)
    print(term_counts(tokens))
