import numpy as np
from sentence_transformers import SentenceTransformer


class DenseSearch:
    def __init__(self, docs: list[str], model_name: str = "all-MiniLM-L6-v2"):
        self.docs = docs
        self.model = SentenceTransformer(model_name)
        # encode() -> array of shape (n_docs, 384). normalize_embeddings makes
        # every vector length 1, so cosine similarity reduces to a dot product.
        self.doc_vectors = self.model.encode(docs, normalize_embeddings=True)

    def search(self, query: str, k: int = 3) -> list[tuple[float, str]]:
        q_vec = self.model.encode(query, normalize_embeddings=True)  # shape (384,)
        scores = self.doc_vectors @ q_vec  # (n_docs, 384) @ (384,) -> (n_docs,)
        top = np.argsort(-scores)[:k]  # indices of the highest scores
        return [(float(scores[i]), self.docs[i]) for i in top]


if __name__ == "__main__":
    from search import TfidfSearch

    docs = [
        "Browser agents can leak personal data when navigating websites.",
        "We prevent leakage of private information in web-navigating LLM assistants.",
        "Convolutional networks achieve strong results on image classification.",
        "Transformers use attention to model relationships between tokens.",
    ]
    tfidf, dense = TfidfSearch(docs), DenseSearch(docs)
    print("embedding shape:", dense.doc_vectors.shape)
    for q in ["browser agents privacy", "attention in transformers"]:
        for name, engine in [("TF-IDF", tfidf), ("Dense ", dense)]:
            print(f"\n[{name}] {q}")
            for score, doc in engine.search(q):
                print(f"  {score:.3f}  {doc}")
