import re
import time

from generation.groq_llm import GroqLLM
from generation.groq_vision import GroqVLM
from ingestion.loader import load_pdfs
from ingestion.parsers.pdf import render_page
from rag.pipeline import RAG
from retrieval.indexes.dense_index import DenseRetriever

# (question, regex the correct answer must match, correct page, kind)
CASES = [
    ("In the FLARE paper's Table 2, what StrategyQA EM does FLARE get?", r"77\.3", 8, "table"),
    ("In the FLARE paper's Table 2, what ASQA EM does the Previous-window method get?", r"39\.9", 8, "table"),
    ("In the FLARE paper's Table 2, what WikiAsp UniEval score does single-time retrieval get?", r"52\.4", 8, "table"),
    ("In the FLARE paper's Table 2, what ASQA-hint D-F1 does Previous-sentence get?", r"35\.9", 8, "table"),
    ("In the FLARE paper's Table 4, what F1 does the 32-token setting get?", r"52\.4", 8, "table"),
    ("In the FLARE paper's Table 3, what 2WikiMultihopQA EM does the Next row get?", r"48\.8", 8, "table"),
    ("In the FLARE paper's Figure 1, what date does the final generated sentence give for Joe Biden's candidacy announcement?", r"April 25", 2, "figure"),
    ("In the FLARE paper's Figure 1, how many boxes labelled Search results are drawn?", r"\b(3|three)\b", 2, "figure"),
    ("In the FLARE paper's Figure 5, what is the StrategyQA EM at 0% retrieval?", r"72\.9", 8, "figure"),
    ("In the FLARE paper's Figure 1, which two steps trigger a new retrieval?", r"(2.*3|two and three|steps? 2)", 2, "figure"),
]
DOC = "2305.06983v2"
retriever = DenseRetriever(load_pdfs("data/pdfs"))
text_rag = RAG(retriever, GroqLLM(), k=4)
vlm = GroqVLM()

rows = []
for q, pattern, true_page, kind in CASES:
    results = retriever.search(q, k=4)
    top = results[0].chunk
    page_hit = top.document_id == DOC and top.page == true_page
    text_ans = text_rag.ask(q).answer
    t_ok = bool(re.search(pattern, text_ans, re.I | re.S))
    page_used = top.page if top.document_id == DOC else true_page
    img = render_page(f"data/pdfs/{top.document_id}.pdf", top.page)
    v_ans = vlm.ask_image(img, f"Answer using only this page image. Be brief.\n\nQuestion: {q}")
    v_ok = bool(re.search(pattern, v_ans, re.I | re.S))
    rows.append((kind, page_hit, t_ok, v_ok))
    print(f"{kind:6s} page_hit={page_hit!s:5} text={'OK ' if t_ok else 'BAD'} vision={'OK ' if v_ok else 'BAD'} | {q[26:90]}")
    time.sleep(0.3)

for kind in ("table", "figure"):
    sel = [r for r in rows if r[0] == kind]
    n = len(sel)
    print(f"\n{kind}: n={n} | retrieval found right page {sum(r[1] for r in sel)}/{n} | "
          f"text route {sum(r[2] for r in sel)}/{n} | vision route {sum(r[3] for r in sel)}/{n}")
