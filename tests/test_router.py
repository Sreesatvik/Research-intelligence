from rag.router import route


def test_table_question_is_visual():
    assert route("What does Table 2 show?") == "visual"


def test_conceptual_question_is_text():
    assert route("Why do language models hallucinate?") == "text"


def test_known_weakness_implicit_table_question():
    # Documents a KNOWN failure: no keyword, but a human would want the table.
    assert route("Which approach performs best on StrategyQA?") == "text"
