from evaluation.multimodal_cases import CASES, DOC
from ingestion.loader import load_pdfs
from rag.page_locator import PageLocator
from retrieval.indexes.dense_index import DenseRetriever

chunks = load_pdfs("data/pdfs")
print("caption chunks found:", [(c.document_id[:10], c.metadata["label"], "p%d" % c.page)
                                for c in chunks if c.content_type == "caption"][:12])
retriever = DenseRetriever(chunks)
locator = PageLocator(chunks)

hits = {"top1": 0, "vote": 0, "locate": 0}
for q, _, true_page, kind in CASES:
    results = retriever.search(q, k=5)
    row = []
    for name in hits:
        found = getattr(locator, name)(*([q, results] if name == "locate" else [results]))
        ok = found == (DOC, true_page)
        hits[name] += ok
        row.append(f"{name}={'OK ' if ok else 'BAD'}")
    print(f"{kind:6s} want p{true_page} | " + " ".join(row) + f" | {q[26:80]}")
print("\npage found correctly (out of %d):" % len(CASES), hits)
