from ingestion.loader import load_corpus
from rag.context.builder import build_prompt
from retrieval.indexes.dense_index import DenseRetriever

chunks = load_corpus("data/corpus_arxiv")
retriever = DenseRetriever(chunks)
question = "Can a model decide by itself when to retrieve documents while writing?"
results = retriever.search(question, k=3)
prompt, source_map = build_prompt(question, results)
print(prompt[:2200])
print("\n--- source map:", source_map)
print("--- prompt length:", len(prompt.split()), "words")
