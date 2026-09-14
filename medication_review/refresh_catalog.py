"""Run explicitly to replace the terminology snapshot with the current RxNorm release."""
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from urllib.request import urlopen
from medication_review.catalog import CATALOG_PATH

URL = "https://rxnav.nlm.nih.gov/REST/allconcepts.json?tty=IN+MIN+BN"


def main():
    with urlopen(URL, timeout=30) as response:
        raw = response.read()
    items = json.loads(raw)["minConceptGroup"]["minConcept"]
    assert len(items) > 1000 and len({i["rxcui"] for i in items}) == len(items)
    with urlopen("https://rxnav.nlm.nih.gov/REST/version.json", timeout=20) as response:
        version = json.load(response)
    payload = {"metadata": {
        "source": URL, "source_version": version,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "counts": dict(Counter(i["tty"] for i in items)),
        "scope": "Active RxNorm ingredient, multiple-ingredient and brand concepts. Terminology availability does not establish human product availability or interaction coverage.",
    }, "items": sorted(items, key=lambda i: i["name"].casefold())}
    target = CATALOG_PATH.with_suffix(".tmp")
    target.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
    target.replace(CATALOG_PATH)
    print("Catalog terms:", len(items), payload["metadata"]["counts"])


if __name__ == "__main__":
    main()
