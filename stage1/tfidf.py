import math

from tokenize_count import tokenize, term_counts


def compute_idf(docs_tokens: list[list[str]]) -> dict[str, float]:
    """idf(word) = log(N / df), where df = number of docs containing the word."""
    n_docs = len(docs_tokens)
    doc_freq = {}
    for tokens in docs_tokens:
        for word in set(tokens):  # set(): count each word once per document
            doc_freq[word] = doc_freq.get(word, 0) + 1
    return {word: math.log(n_docs / df) for word, df in doc_freq.items()}


def tfidf_vector(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    """Sparse TF-IDF vector as {word: weight}. Words unseen in the corpus are skipped."""
    counts = term_counts(tokens)
    return {w: c * idf[w] for w, c in counts.items() if w in idf}


if __name__ == "__main__":
    docs = [
        "The cat sat on the mat.",
        "The dog sat on the log.",
        "Cats and dogs are pets.",
    ]
    docs_tokens = [tokenize(d) for d in docs]
    idf = compute_idf(docs_tokens)
    for word in ["the", "sat", "cat", "pets"]:
        print(f"idf({word!r}) = {idf[word]:.3f}")
    print(tfidf_vector(docs_tokens[0], idf))
