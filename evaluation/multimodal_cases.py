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
