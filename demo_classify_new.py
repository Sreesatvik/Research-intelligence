import json
from pathlib import Path

from classification.classifier import DocumentClassifier

truth = json.load(open("data/eval/topics.json", encoding="utf-8"))
paths = sorted(Path("data/corpus_arxiv").glob("*.txt"))
clf = DocumentClassifier().fit([p.read_text(encoding="utf-8") for p in paths],
                               [truth[p.stem] for p in paths])
NEW = [
    "We propose a method that retrieves relevant passages from Wikipedia and feeds them to a language model to answer open-domain questions.",
    "Attackers hide malicious instructions inside web pages so that an AI browsing assistant obeys them instead of the user.",
    "A new contrastive objective produces better sentence vectors for semantic textual similarity benchmarks.",
    "Preheat the oven to 200 degrees, mix flour, butter and sugar, and bake the cookies for twelve minutes until golden.",
    "We study how convolutional neural networks forecast protein structures from amino acid sequences.",
]
for text in NEW:
    label, conf = clf.classify(text)
    print(f"{label:22s} conf {conf:.2f} | {text[:70]}")
