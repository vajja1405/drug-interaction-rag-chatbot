"""
tests/test_classifier.py
────────────────────────
Unit tests covering the severity classifier's fallback tiers.
"""
from models.interaction_classifier import InteractionClassifier

def test_tier1_structured_metadata():
    """Tier 1: Document metadata contains reliable severity tag."""
    clf = InteractionClassifier()
    result = clf.classify({
        "metadata": {"severity": "Moderate", "severity_source": "rxnorm"},
        "text": "Doesn't matter what this says.",
        "similarity_score": 0.9,
    })
    
    assert result.severity == "Moderate"
    assert result.method == "structured"
    assert result.confidence == 0.95

def test_tier2_keyword_extraction_severe():
    """Tier 2: Text contains severe keywords."""
    clf = InteractionClassifier()
    result = clf.classify({
        "metadata": {},
        "text": "This combination is fatal and strictly contraindicated to use together.",
        "similarity_score": 0.85,
    })
    
    assert result.severity == "Severe"
    assert result.method == "keyword"
    assert "fatal" in result.matched_keywords
    assert "contraindicated" in result.matched_keywords

def test_tier2_keyword_extraction_moderate():
    """Tier 2: Text contains moderate keywords."""
    clf = InteractionClassifier()
    result = clf.classify({
        "metadata": {},
        "text": "Monitor patients closely and use caution with this combination.",
        "similarity_score": 0.85,
    })
    
    assert result.severity == "Moderate"
    assert result.method == "keyword"
    assert "caution" in result.matched_keywords

def test_tier4_fallback():
    """Tier 4: No metadata and no identifiable clinical keywords."""
    clf = InteractionClassifier()
    result = clf.classify({
        "metadata": {},
        "text": "Bob went to the store to get some bread.",
        "similarity_score": 0.5,
    })
    
    assert result.severity == "Unknown"
    assert result.method == "default"

def test_classify_best_returns_highest_confidence_tier_1():
    """classify_best prioritizes structured severity across multiple docs."""
    clf = InteractionClassifier()
    results = [
        {
            "metadata": {"severity": "Low", "severity_source": "curated"},
            "text": "Structured doc",
            "similarity_score": 0.9
        },
        {
            "metadata": {},
            "text": "Some text",
            "similarity_score": 0.5
        }
    ]
    
    best = clf.classify_best(results)
    assert best.severity == "Low"
    assert best.method == "structured"
