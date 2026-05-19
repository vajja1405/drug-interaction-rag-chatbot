"""
tests/test_agent.py
───────────────────
Unit tests for the DrugInteractionAgent and polypharmacy generation.
"""
from chatbot.interaction_agent import DrugCombination, DrugInteractionAgent

def test_drug_combination_alphabetical_key():
    """Ensure pair keys are always alphabetically sorted for cache stability."""
    combo1 = DrugCombination("warfarin", "ibuprofen")
    combo2 = DrugCombination("ibuprofen", "warfarin")
    
    assert combo1.pair_key == "ibuprofen+warfarin"
    assert combo2.pair_key == "ibuprofen+warfarin"

def test_polypharmacy_generation():
    """Ensure a list of N drugs generates N choose 2 combinations."""
    drugs3 = ["A", "B", "C"]
    agent = DrugInteractionAgent(retriever=None)
    from chatbot.interaction_agent import ReasoningTrace
    trace = ReasoningTrace()
    combinations = agent._stage1_combinations(drugs3, trace)
    assert len(combinations) == 3
    
    keys = {c.pair_key for c in combinations}
    # Expected alphabetically sorted pairs
    assert keys == {"a+b", "a+c", "b+c"}
    
    drugs4 = ["A", "B", "C", "D"]
    combinations4 = agent._stage1_combinations(drugs4, trace)
    # 4 choose 2 is 6
    assert len(combinations4) == 6
