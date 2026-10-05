from ingestion.loader import load_corpus

chunks = load_corpus("data/corpus", chunk_size=100, overlap=20)
print("total chunks:", len(chunks))
for c in chunks[:5]:
    print(c.chunk_id, "|", len(c.text.split()), "words |", c.text[:60], "...")
