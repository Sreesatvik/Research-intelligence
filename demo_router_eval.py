from rag.router import route

CASES = [
    ("What does Table 2 show?", "visual"),
    ("Describe the diagram of the FLARE pipeline", "visual"),
    ("What does the figure on page 2 illustrate?", "visual"),
    ("What is the exact match score of FLARE on ASQA?", "visual"),
    ("Which method has the highest accuracy on WikiAsp?", "visual"),
    ("How many topics does BERTopic find in the 20 newsgroups results?", "visual"),
    ("Which approach performs best on StrategyQA?", "visual"),
    ("Compare the results of the baselines across datasets", "visual"),
    ("What is the main idea of active retrieval?", "text"),
    ("How does contrastive learning work?", "text"),
    ("Why do language models hallucinate?", "text"),
    ("What is c-TF-IDF used for?", "text"),
    ("Explain the architecture of the retriever in plain words", "text"),
    ("What are the limitations the authors mention?", "text"),
    ("Who proposed supervised contrastive learning for sentences?", "text"),
    ("What score threshold should I use for refusing?", "text"),
]
wrong = [(q, want, route(q)) for q, want in CASES if route(q) != want]
print(f"router accuracy: {len(CASES) - len(wrong)}/{len(CASES)}")
for q, want, got in wrong:
    print(f"  WRONG  wanted {want:6s} got {got:6s} | {q}")
