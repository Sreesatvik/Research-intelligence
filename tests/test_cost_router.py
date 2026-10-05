from rag.cost_router import route_by_rules


def test_lookup_goes_to_retrieval():
    assert route_by_rules("Which papers are about topic modeling?") == "retrieval"


def test_single_question_goes_to_rag():
    assert route_by_rules("What does PromptLocate do?") == "rag"


def test_comparison_goes_to_agent():
    assert route_by_rules("Compare BERTopic and TopiCLEAR") == "agent"


def test_keyword_inside_a_longer_word_does_not_trigger_agent():
    # regression: "collect" is inside "collection", "contrast" inside "contrastive"
    assert route_by_rules("Which papers in the collection are about topic modeling?") == "retrieval"
    assert route_by_rules("How does supervised contrastive learning help?") == "rag"
