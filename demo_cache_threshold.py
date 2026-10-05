from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
SAME = [
    ("What is the main idea of active retrieval augmented generation?", "Can you explain the core idea behind active retrieval augmented generation?"),
    ("How do attacks on browser agents work?", "How do attackers target browser agents?"),
    ("Which paper localizes injected prompts?", "What paper finds the part of the input that was injected?"),
    ("Explain contrastive learning for sentence embeddings", "How does contrastive learning produce sentence embeddings?"),
    ("What is the capital of France?", "capital of france?"),
]
DIFFERENT = [
    ("What exact-match score does FLARE get on StrategyQA?", "What exact-match score does FLARE get on ASQA?"),
    ("What does Table 2 show?", "What does Table 3 show?"),
    ("How do prompt injection attacks work?", "How do prompt injection defenses work?"),
    ("Which papers are about vision transformers?", "Which papers are about topic modeling?"),
    ("Who wrote the FLARE paper?", "Who wrote the BERTopic paper?"),
]
def sim(a, b):
    va, vb = model.encode([a, b], normalize_embeddings=True)
    return float(va @ vb)

print("SHOULD HIT (same meaning):")
same = [sim(a, b) for a, b in SAME]
for (a, b), s in zip(SAME, same):
    print(f"  {s:.3f} | {a[:48]} <> {b[:48]}")
print("MUST MISS (look similar, different answer):")
diff = [sim(a, b) for a, b in DIFFERENT]
for (a, b), s in zip(DIFFERENT, diff):
    print(f"  {s:.3f} | {a[:48]} <> {b[:48]}")
print(f"\nlowest 'same' = {min(same):.3f} | highest 'different' = {max(diff):.3f}")
