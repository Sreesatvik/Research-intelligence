"""Download arXiv abstracts into data/corpus_arxiv/ (one .txt file per paper)."""
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

TOPICS = {
    "rag": "retrieval augmented generation",
    "embeddings": "sentence embeddings contrastive learning",
    "transformer": "attention transformer architecture efficiency",
    "agents": "LLM web agents browser",
    "injection": "prompt injection attacks language models",
    "clustering": "topic modeling clustering embeddings",
    "vision": "vision transformer image classification",
    "evaluation": "evaluating retrieval augmented generation",
}
PER_TOPIC = 6
NS = {"a": "http://www.w3.org/2005/Atom"}
out_dir = Path("data/corpus_arxiv")
out_dir.mkdir(parents=True, exist_ok=True)

for label, query in TOPICS.items():
    params = urllib.parse.urlencode({
        "search_query": " AND ".join(f"abs:{w}" for w in query.split()), "start": 0, "max_results": PER_TOPIC,
    })
    url = f"http://export.arxiv.org/api/query?{params}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        root = ET.fromstring(resp.read())
    for entry in root.findall("a:entry", NS):
        arxiv_id = entry.find("a:id", NS).text.rsplit("/", 1)[-1]
        title = " ".join(entry.find("a:title", NS).text.split())
        abstract = " ".join(entry.find("a:summary", NS).text.split())
        (out_dir / f"{arxiv_id.replace('/', '_')}.txt").write_text(
            f"{title}. {abstract}", encoding="utf-8")
        print(f"[{label}] {arxiv_id}  {title[:70]}")
    time.sleep(3)  # arXiv asks clients to wait between requests
