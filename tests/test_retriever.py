"""
tests/test_retriever.py
───────────────────────
Integration-level test for the FAISS vector store.
Verifies that known seeded pairs successfully return documents.
"""
from rag_pipeline.retriever import DrugInteractionRetriever
from rag_pipeline.vector_store import DrugVectorStore

def test_retriever_finds_seeded_pair():
    """Ensure the retriever finds the known warfarin+aspirin interaction."""
    store = DrugVectorStore.load()
    retriever = DrugInteractionRetriever(vector_store=store)
    
    # Query with a known severe pair from the curated seed data
    # Note: retriever.retrieve() doesn't exist, Mode A is retrieve_for_drug_list
    result = retriever.retrieve_for_drug_list(["warfarin", "aspirin"])
    
    # The evidence should contain at least 1 document for the pair
    assert "aspirin+warfarin" in result
    evidence = result["aspirin+warfarin"]
    assert len(evidence) >= 1
    
    # Verify the mechanism text is retrieved
    top_doc = evidence[0]["text"]
    assert "bleeding" in top_doc.lower() or "haemorrhage" in top_doc.lower()

def test_retriever_empty_for_unknown_pair():
    """Ensure the retriever gracefully handles pairs with no documents."""
    store = DrugVectorStore.load()
    retriever = DrugInteractionRetriever(vector_store=store)
    
    # Query with a nonsense pair
    result = retriever.retrieve_for_drug_list(["fake-drug-xyz", "another-fake-drug-abc"])
    
    assert "another-fake-drug-abc+fake-drug-xyz" in result
    assert len(result["another-fake-drug-abc+fake-drug-xyz"]) == 0
