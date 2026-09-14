"""Bounded, cached label retrieval from fixed NLM origins; no LLM or severity inference."""
import asyncio
import copy
import time
import uuid
from collections import OrderedDict, Counter
from datetime import datetime, timezone

import httpx

from medication_review.catalog import Catalog
from medication_review.evidence import parse_label, compare_medications, ingredient_set_matches

RX = "https://rxnav.nlm.nih.gov/REST"
DM = "https://dailymed.nlm.nih.gov/dailymed/services/v2"


class ReviewService:
    def __init__(self, catalog=None, client=None):
        self.catalog = catalog or Catalog()
        self.client = client or httpx.AsyncClient(timeout=httpx.Timeout(8), follow_redirects=False)
        self.cache = OrderedDict()
        self.cache_lock = asyncio.Lock()
        self.request_lock = asyncio.Lock()
        self.last_request = 0.0
        self.workers = asyncio.Semaphore(4)

    async def close(self):
        await self.client.aclose()

    async def get(self, url, params=None, xml=False):
        key = (url, tuple(sorted((params or {}).items())))
        async with self.cache_lock:
            cached = self.cache.get(key)
            if cached and cached[0] > time.monotonic():
                self.cache.move_to_end(key)
                return copy.deepcopy(cached[1])
            self.cache.pop(key, None)
        async with self.request_lock:
            await asyncio.sleep(max(0, .2 - (time.monotonic() - self.last_request)))
            self.last_request = time.monotonic()
        # Stream into a bounded buffer; reject redirects, errors, oversized XML/JSON.
        async with self.client.stream("GET", url, params=params) as response:
            response.raise_for_status()
            chunks, size = [], 0
            async for chunk in response.aiter_bytes():
                size += len(chunk)
                if size > 4_000_000:
                    raise ValueError("Upstream response too large")
                chunks.append(chunk)
        raw = b"".join(chunks)
        if xml:
            payload = raw
        else:
            import json
            payload = json.loads(raw)
        async with self.cache_lock:
            self.cache[key] = (time.monotonic() + 12 * 60 * 60, payload)
            self.cache.move_to_end(key)
            # Label documents can be large; cap both count and approximate byte size.
            while len(self.cache) > 180 or sum(len(v[1]) if isinstance(v[1], bytes) else len(str(v[1])) for v in self.cache.values()) > 40_000_000:
                self.cache.popitem(last=False)
        return copy.deepcopy(payload)

    async def ingredients(self, item):
        local = self.catalog.local_ingredients(item)
        if local:
            return local, "catalog"
        try:
            data = await self.get(f"{RX}/rxcui/{item['rxcui']}/related.json", {"tty": "IN"})
            groups = data.get("relatedGroup", {}).get("conceptGroup", [])
            found = {p["rxcui"]: {"rxcui": p["rxcui"], "name": p["name"], "tty": "IN"}
                     for group in groups if group.get("tty") == "IN"
                     for p in group.get("conceptProperties", [])}
            # Brand concepts can aggregate formulations. Do not claim an exact package match.
            return list(found.values()), "brand_or_concept_level" if found else "unavailable"
        except (httpx.HTTPError, ValueError, KeyError):
            return [], "unavailable"

    async def medication(self, item):
        result = {**item, "ingredients": [], "ingredient_resolution": "unavailable",
                  "labels": [], "status": "source_unavailable", "label_matches": 0,
                  "label_selection": "A sample matching human label; verify manufacturer, strength, route and formulation against your package."}
        async def load():
            result["ingredients"], result["ingredient_resolution"] = await self.ingredients(item)
            # Exact RxNorm linkage avoids matching a drug by a coincidental title substring.
            # Prioritize human prescription labeling, then human OTC. No veterinary labels.
            saw_records = False
            for doctype in ["34391-3", "34390-5"]:
                data = await self.get(f"{DM}/spls.json", {"rxcui": item["rxcui"], "doctype": doctype, "pagesize": 3})
                records = data.get("data", [])
                if records:
                    saw_records = True
                    result["label_matches"] = int(data.get("metadata", {}).get("total_elements", len(records)))
                for record in records:
                    setid = str(uuid.UUID(record["setid"]))
                    try:
                        raw = await self.get(f"{DM}/spls/{setid}.xml", xml=True)
                        parsed = parse_label(raw, setid)
                    except (httpx.HTTPError, ValueError):
                        continue
                    if not ingredient_set_matches(parsed["active_ingredients"], [i["name"] for i in result["ingredients"]]):
                        continue
                    result["labels"] = [{**parsed, "title": record["title"],
                        "published_date": record.get("published_date", ""),
                        "url": f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={setid}"}]
                    if any(s["section"] == "Drug interactions" for s in parsed["sections"]):
                        result["status"] = "label_available"
                        return
                    result["status"] = "interaction_section_unavailable"
                if result["labels"]:
                    return
            result["status"] = "label_match_unverified" if saw_records else "no_matching_label"
        async with self.workers:
            try:
                await asyncio.wait_for(load(), timeout=22)
            except (TimeoutError, httpx.HTTPError, ValueError, KeyError):
                # Keep any successfully obtained provenance, and distinguish incomplete sources.
                result["status"] = "source_unavailable"
        return result

    async def review(self, ids):
        started = time.perf_counter()
        medications = await asyncio.wait_for(asyncio.gather(
            *(self.medication(self.catalog.by_id[rxcui]) for rxcui in ids)), timeout=115)
        pairs = compare_medications(medications)
        counts = Counter(p["status"] for p in pairs)
        # Do not return full text; excerpts plus original links are the review artifact.
        for med in medications:
            for label in med["labels"]:
                label["sections"] = [{"section": s["section"], "characters": len(s["text"]),
                                      "truncated": s["truncated"]} for s in label["sections"]]
        return {"request_id": str(uuid.uuid4()), "medications": medications, "pairs": pairs,
                "pairs_reviewed": len(pairs), "counts": dict(counts),
                "pairs_with_incomplete_sources": sum(not p["source_complete"] for p in pairs),
                "catalog_terms": len(self.catalog.items), "catalog_version": self.catalog.metadata,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "processing_time_ms": round((time.perf_counter() - started) * 1000),
                "method": "Exact names/ingredient names in sampled label interaction sections; no severity prediction or LLM.",
                "limitations": [
                    "No direct mention does not mean no interaction or that a combination is safe.",
                    "Drug-class, dose, route, timing, disease, food and higher-order interactions are not assessed.",
                    "A label mention can describe no effect, a study or a conditional warning; read the passage in its original context.",
                    "Only one sample human label per selected concept is used; brands can cover multiple formulations.",
                    "Active-ingredient names are checked against the selected concept. Unrecognized naming variants and unmatched formulations are excluded; only a bounded candidate sample is examined.",
                    "Ingredient overlap is a question to review, not a recommendation to stop a medicine.",
                    "Sources may be cached for up to 12 hours; source dates and incomplete retrieval are visible.",
                ], "clinical_validation": "not_established"}
