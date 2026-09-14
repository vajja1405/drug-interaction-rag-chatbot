"""Behavioral coverage for terminology identity, label provenance and conservative failure."""
import asyncio
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler

from api.review import router, limiter
from medication_review.catalog import Catalog
from medication_review.evidence import parse_label, excerpts, compare_medications
from medication_review.service import ReviewService

SETID = "11111111-1111-1111-1111-111111111111"
XML = f'''<document xmlns="urn:hl7-org:v3"><setId root="{SETID}"/>
<versionNumber value="3"/><effectiveTime value="20240101"/>
<ingredient classCode="ACTIB"><ingredientSubstance><name>WARFARIN SODIUM</name></ingredientSubstance></ingredient>
<component><structuredBody><component><section><code code="34073-7"/>
<text>No significant interaction was observed with aspirin in this study.</text>
<component><section><code code="x"/><text>Other conditions require review.</text></section></component>
</section></component></structuredBody></component></document>'''.encode()


def med(name, rxcui, ingredients=None, status="label_available", text=""):
    return {"name": name, "rxcui": rxcui, "ingredients": ingredients or [], "status": status,
            "labels": [{"title": "Test label", "url": "https://dailymed.nlm.nih.gov/", "setid": SETID,
                        "version": "3", "sections": [{"section": "Drug interactions", "text": text, "truncated": False}]}]}


def test_catalog_is_large_unique_and_exact_name_first():
    c = Catalog()
    assert len(c.items) > 1000
    assert len(c.by_id) == len(c.items)
    assert c.search("warfarin")[0]["rxcui"] == "11289"
    assert c.search("TYLENOL")[0]["tty"] == "BN"
    assert not c.search("nonexistentmedicineZZZZ")
    assert not c.search("a")
    assert c.metadata["source"].startswith("https://rxnav.nlm.nih.gov/")


def test_label_identity_version_nested_sections_and_negation_preserved():
    label = parse_label(XML, SETID)
    assert label["version"] == "3" and len(label["sha256"]) == 64
    assert "Other conditions" in label["sections"][0]["text"]
    assert excerpts(label["sections"][0]["text"], "aspirin")[0].startswith("No significant interaction")
    with pytest.raises(ValueError):
        parse_label(XML, "22222222-2222-2222-2222-222222222222")
    with pytest.raises(ValueError):
        parse_label(b"<!DOCTYPE x><bad/>", SETID)
    with pytest.raises(ValueError):
        parse_label(b"broken XML", SETID)


def test_no_substring_false_match_and_no_class_inference():
    assert not excerpts("metoprolol and aspirin", "met")
    assert not excerpts("Avoid nonsteroidal anti-inflammatory drugs", "ibuprofen")
    pairs = compare_medications([med("ibuprofen", "1"), med("warfarin", "2", text="Avoid NSAIDs")])
    assert pairs[0]["status"] == "no_direct_mention"
    assert pairs[0]["clinical_severity"] == "not_assessed"


def test_negated_mention_is_not_converted_into_risk():
    pairs = compare_medications([med("aspirin", "1"), med("other", "2", text="No interaction was observed with aspirin.")])
    assert pairs[0]["status"] == "label_mention"
    assert pairs[0]["evidence"][0]["excerpt"].startswith("No interaction")
    assert pairs[0]["clinical_severity"] == "not_assessed"


def test_missing_source_is_distinct_from_no_direct_mention():
    pairs = compare_medications([med("a", "1", status="source_unavailable"), med("b", "2")])
    assert pairs[0]["status"] == "incomplete_sources"
    assert not pairs[0]["source_complete"]
    mentioned = compare_medications([med("aspirin", "1", status="source_unavailable"), med("b", "2", text="aspirin was studied")])[0]
    assert mentioned["status"] == "label_mention"
    assert not mentioned["source_complete"]


def test_possible_ingredient_overlap_by_identifier_not_spelling():
    ingredient = {"rxcui": "161", "name": "acetaminophen"}
    pair = compare_medications([med("Tylenol", "1", [ingredient]), med("acetaminophen", "2", [ingredient])])[0]
    assert pair["status"] == "ingredient_overlap"
    assert pair["shared_ingredients"] == ["acetaminophen"]


def test_combination_label_cannot_stand_in_for_single_ingredient():
    from medication_review.evidence import ingredient_set_matches
    assert not ingredient_set_matches(["ASPIRIN", "DIPYRIDAMOLE"], ["aspirin"])
    assert ingredient_set_matches(["WARFARIN SODIUM"], ["warfarin"])
    assert ingredient_set_matches(["ASPIRIN", "DIPYRIDAMOLE"], ["aspirin", "dipyridamole"])
    assert not ingredient_set_matches([], ["aspirin"])
    assert not ingredient_set_matches(["METOPROLOL"], ["met"])


def test_twenty_medications_yield_all_190_unique_pairs():
    pairs = compare_medications([med(f"drug{i}", str(i)) for i in range(20)])
    assert len(pairs) == 190
    assert len({tuple(p["ids"]) for p in pairs}) == 190
    assert all(p["status"] == "no_direct_mention" for p in pairs)


def test_oversized_context_preserves_explicit_truncation():
    from medication_review.evidence import MAX_SECTION_CHARS
    raw = XML.replace(b"No significant", b"a" * (MAX_SECTION_CHARS + 10) + b" No significant")
    label = parse_label(raw, SETID)
    assert label["sections"][0]["truncated"]
    assert len(label["sections"][0]["text"]) == MAX_SECTION_CHARS


def test_http_failure_is_not_cached_as_no_label_and_success_is_cached():
    async def run():
        calls = []
        def handler(request):
            calls.append(str(request.url))
            if len(calls) == 1:
                return httpx.Response(503)
            if request.url.path.endswith(".xml"):
                return httpx.Response(200, content=XML)
            return httpx.Response(200, json={"data": [{"setid": SETID, "title": "Test", "published_date": "Jan 01, 2024"}], "metadata": {"total_elements": 1}})
        s = ReviewService(client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        try:
            item = s.catalog.by_id["11289"]
            first = await s.medication(item)
            assert first["status"] == "source_unavailable"
            second = await s.medication(item)
            assert second["status"] == "label_available"
            before = len(calls)
            await s.medication(item)
            assert len(calls) == before
            # Expiring the cache forces retrieval again.
            for k, (_, v) in list(s.cache.items()): s.cache[k] = (0, v)
            await s.medication(item)
            assert len(calls) > before
        finally:
            await s.close()
    asyncio.run(run())


def test_label_filter_is_human_and_exact_rxcui():
    async def run():
        queries = []
        def handler(request):
            queries.append(dict(request.url.params))
            return httpx.Response(200, json={"data": []})
        s = ReviewService(client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        try:
            result = await s.medication(s.catalog.by_id["11289"])
            assert result["status"] == "no_matching_label"
            assert {q["doctype"] for q in queries} == {"34391-3", "34390-5"}
            assert all(q["rxcui"] == "11289" and "drug_name" not in q for q in queries)
        finally: await s.close()
    asyncio.run(run())


def test_service_skips_a_combination_label_before_selecting_matching_actives():
    async def run():
        wrong_id = "22222222-2222-2222-2222-222222222222"
        wrong_xml = XML.replace(SETID.encode(), wrong_id.encode()).replace(
            b"</document>", b'<ingredient classCode="ACTIB"><ingredientSubstance><name>ASPIRIN</name></ingredientSubstance></ingredient></document>')
        def handler(request):
            if wrong_id in request.url.path: return httpx.Response(200, content=wrong_xml)
            if SETID in request.url.path: return httpx.Response(200, content=XML)
            return httpx.Response(200, json={"data": [
                {"setid": wrong_id, "title": "Combination"}, {"setid": SETID, "title": "Single ingredient"}],
                "metadata": {"total_elements": 2}})
        s = ReviewService(client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        try:
            result = await s.medication(s.catalog.by_id["11289"])
            assert result["status"] == "label_available"
            assert result["labels"][0]["setid"] == SETID
            assert result["labels"][0]["active_ingredients"] == ["WARFARIN SODIUM"]
        finally: await s.close()
    asyncio.run(run())


def test_v2_request_validation_and_no_llm_dependency():
    app = FastAPI()
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.include_router(router)
    service = ReviewService()
    app.state.review_service = service
    with patch.object(limiter, "enabled", False), patch.object(service, "review", new_callable=AsyncMock) as review:
        review.return_value = {"pairs_reviewed": 190}
        client = TestClient(app)
        assert client.get("/api/v2/catalog").json()["terms"] > 1000
        assert client.get("/api/v2/medications?q=warfarin").json()["results"][0]["rxcui"] == "11289"
        assert client.post("/api/v2/review", json={"rxcuis": ["11289"]}).status_code == 422
        assert client.post("/api/v2/review", json={"rxcuis": ["11289", "11289"]}).status_code == 422
        assert client.post("/api/v2/review", json={"rxcuis": ["11289", "999999999999"]}).status_code == 422
        assert client.post("/api/v2/review", json={"rxcuis": ["11289", "5640"], "patient_name": "test"}).status_code == 422
        ids = [i["rxcui"] for i in service.catalog.items[:20]]
        assert client.post("/api/v2/review", json={"rxcuis": ids}).status_code == 200
        assert client.post("/api/v2/review", json={"rxcuis": ids + ["11289"]}).status_code == 422
        review.assert_awaited_once_with(ids)
    asyncio.run(service.close())
