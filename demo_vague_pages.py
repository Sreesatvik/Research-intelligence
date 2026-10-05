from evaluation.multimodal_cases import DOC
from ingestion.loader import load_pdfs
from rag.page_locator import PageLocator
from retrieval.indexes.dense_index import DenseRetriever

VAGUE = [
    ("Which retrieval approach performs best on StrategyQA in the FLARE paper?", 8),
    ("How does the percentage of retrieval affect FLARE performance on 2WikiMultihopQA?", 8),
    ("Show how FLARE regenerates a sentence about Joe Biden step by step", 2),
    ("In FLARE, how many past tokens work best as the query for the previous-window method?", 8),
    ("Compare using the previous sentence versus the next sentence for retrieval in FLARE", 8),
]
chunks = load_pdfs("data/pdfs")
retriever, locator = DenseRetriever(chunks), PageLocator(chunks)
hits = {"top1": 0, "vote": 0, "locate": 0}
for q, page in VAGUE:
    results = retriever.search(q, k=5)
    row = []
    for name in hits:
        arg = [q, results] if name == "locate" else [results]
        ok = getattr(locator, name)(*arg) == (DOC, page)
        hits[name] += ok
        row.append(f"{name}={'OK ' if ok else 'BAD'}")
    print(f"want p{page} | " + " ".join(row) + f" | {q[:70]}")
print("\nvague questions, page found (of %d):" % len(VAGUE), hits)
