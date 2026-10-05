from collections import Counter

from ingestion.loader import load_pdfs

chunks = load_pdfs("data/pdfs")
print("total chunks:", len(chunks))
print(Counter((c.document_id, c.content_type) for c in chunks))
text = next(c for c in chunks if c.document_id == "2305.06983v2" and c.page == 3)
print(f"\n--- text chunk {text.chunk_id} (page {text.page}) ---\n{text.text[:350]}")
table = next(c for c in chunks if c.content_type == "table")
print(f"\n--- table chunk {table.chunk_id} (page {table.page}) ---\n{table.text[:300]}")
