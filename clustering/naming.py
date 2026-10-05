from representations.tfidf.tfidf import compute_idf, tfidf_vector
from representations.tfidf.tokenizer import tokenize

STOP = {"the", "of", "and", "to", "in", "a", "is", "for", "we", "on", "with", "that",
        "this", "are", "as", "by", "an", "our", "can", "from", "be", "or", "it", "which",
        "at", "also", "has", "have", "these", "their", "its", "such", "than", "not"}


def cluster_keywords(texts: list[str], labels, k: int, top_n: int = 6) -> list[list[str]]:
    """c-TF-IDF: treat each cluster as ONE document and find its most distinctive words."""
    cluster_tokens = []
    for j in range(k):
        words = []
        for text, label in zip(texts, labels):
            if label == j:
                words += [w for w in tokenize(text) if w not in STOP and len(w) > 2]
        cluster_tokens.append(words)
    idf = compute_idf(cluster_tokens)  # IDF across clusters, not across papers
    keywords = []
    for tokens in cluster_tokens:
        vec = tfidf_vector(tokens, idf)
        keywords.append([w for w, _ in sorted(vec.items(), key=lambda x: -x[1])[:top_n]])
    return keywords
