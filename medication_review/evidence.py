"""Exact label-text matching. A mention is never classified as a clinical interaction."""
import hashlib
import itertools
import re
import xml.etree.ElementTree as ET

NS = {"s": "urn:hl7-org:v3"}
SECTIONS = {"34073-7": "Drug interactions", "34071-1": "Warnings",
            "43685-7": "Warnings and precautions", "34066-1": "Boxed warning",
            "34070-3": "Contraindications", "50569-7": "Ask a doctor or pharmacist"}
MAX_SECTION_CHARS = 120_000


def plain_text(element):
    return re.sub(r"\s+", " ", " ".join(element.itertext())).strip()


def parse_label(raw: bytes, expected_setid: str):
    if len(raw) > 4_000_000 or b"<!DOCTYPE" in raw.upper() or b"<!ENTITY" in raw.upper():
        raise ValueError("Unsupported label XML")
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        raise ValueError("Malformed label XML") from exc
    setid = root.find("s:setId", NS)
    if setid is None or setid.get("root", "").lower() != expected_setid.lower():
        raise ValueError("Label identity mismatch")
    version = root.find("s:versionNumber", NS)
    effective = root.find("s:effectiveTime", NS)
    active_ingredients = []
    for ingredient in root.findall(".//s:ingredient", NS):
        if ingredient.get("classCode", "").startswith("ACTI"):
            name = ingredient.find("s:ingredientSubstance/s:name", NS)
            if name is not None and name.text:
                active_ingredients.append(name.text.strip())
    sections = []
    seen = set()
    for section in root.findall(".//s:section", NS):
        code = section.find("s:code", NS)
        if code is None or code.get("code") not in SECTIONS:
            continue
        # Parent sections often contain nested subsections. Include their text once.
        blocks = section.findall(".//s:text", NS)
        texts = [plain_text(block) for block in blocks]
        text = "\n".join(dict.fromkeys(t for t in texts if t))
        if not text or text in seen:
            continue
        seen.add(text)
        sections.append({"section": SECTIONS[code.get("code")],
                         "text": text[:MAX_SECTION_CHARS],
                         "truncated": len(text) > MAX_SECTION_CHARS})
    return {"setid": expected_setid, "version": version.get("value", "") if version is not None else "",
            "effective_date": effective.get("value", "") if effective is not None else "",
            "sha256": hashlib.sha256(raw).hexdigest(), "sections": sections,
            "active_ingredients": sorted(set(active_ingredients))}


def ingredient_set_matches(label_ingredients, expected):
    """Conservative name match allowing salt suffixes, rejecting extra/missing actives.

    RxCUI linkage is required upstream as well. This is not a strength/route match.
    Unrecognized naming variants fail closed rather than broadening the label.
    """
    if not label_ingredients or not expected:
        return False
    def matches(active, name):
        return bool(re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", active, re.IGNORECASE))
    return (all(any(matches(active, name) for name in expected) for active in label_ingredients)
            and all(any(matches(active, name) for active in label_ingredients) for name in expected))


def excerpts(text: str, term: str, limit=2):
    if len(term.strip()) < 3:
        return []
    pattern = re.compile(r"(?<!\w)" + re.escape(term) + r"(?!\w)", re.IGNORECASE)
    result = []
    for match in pattern.finditer(text):
        start = max(0, match.start() - 220)
        end = min(len(text), match.end() + 300)
        # Preserve context (including negation); do not rewrite or infer severity.
        result.append(("…" if start else "") + text[start:end] + ("…" if end < len(text) else ""))
        if len(result) == limit:
            break
    return result


def compare_medications(medications):
    pairs = []
    for left, right in itertools.combinations(medications, 2):
        li = {x["rxcui"]: x["name"] for x in left.get("ingredients", [])}
        ri = {x["rxcui"]: x["name"] for x in right.get("ingredients", [])}
        overlap = [li[k] for k in sorted(li.keys() & ri.keys())]
        evidence = []
        for source, target in [(left, right), (right, left)]:
            terms = list(dict.fromkeys([target["name"]] + [i["name"] for i in target.get("ingredients", [])]))
            for label in source.get("labels", []):
                for section in label["sections"]:
                    # Warnings remain available on the source cards. Only the dedicated
                    # interaction section is searched for pair mentions.
                    if section["section"] != "Drug interactions":
                        continue
                    seen = set()
                    for term in terms:
                        for quote in excerpts(section["text"], term):
                            if quote in seen:
                                continue
                            seen.add(quote)
                            evidence.append({"source_drug": source["name"], "matched_term": term,
                                "section": section["section"], "excerpt": quote,
                                "source_url": label["url"], "source_title": label["title"],
                                "setid": label["setid"], "version": label["version"]})
        if overlap:
            status = "ingredient_overlap"
        elif evidence:
            status = "label_mention"
        elif any(m["status"] != "label_available" for m in [left, right]):
            status = "incomplete_sources"
        else:
            status = "no_direct_mention"
        pairs.append({"ids": [left["rxcui"], right["rxcui"]],
                      "names": [left["name"], right["name"]], "status": status,
                      "shared_ingredients": overlap, "evidence": evidence[:8],
                      "evidence_truncated": len(evidence) > 8,
                      "source_complete": all(m["status"] == "label_available" for m in [left, right]),
                      "clinical_severity": "not_assessed"})
    return pairs
