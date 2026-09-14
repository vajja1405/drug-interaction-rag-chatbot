"""Searchable, attributed RxNorm terminology snapshot; not an interaction database."""
import json
import re
from pathlib import Path

CATALOG_PATH = Path(__file__).with_name("catalog.json")


def normalized(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold()).strip()


class Catalog:
    def __init__(self, path=CATALOG_PATH):
        payload = json.loads(Path(path).read_text())
        self.metadata = payload["metadata"]
        self.items = payload["items"]
        self.by_id = {item["rxcui"]: item for item in self.items}
        self.ingredients = {normalized(i["name"]): i for i in self.items if i["tty"] == "IN"}

    def search(self, query: str, limit=15):
        query = normalized(query)
        if len(query) < 2:
            return []
        matches = [i for i in self.items if query in normalized(i["name"])]
        return sorted(matches, key=lambda i: (
            normalized(i["name"]) != query,
            not normalized(i["name"]).startswith(query),
            i["tty"] != "IN", len(i["name"]), i["name"],
        ))[:limit]

    def local_ingredients(self, item):
        if item["tty"] == "IN":
            return [item]
        if item["tty"] == "MIN":
            terms = [self.ingredients.get(normalized(n)) for n in item["name"].split(" / ")]
            return terms if terms and all(terms) else []
        return []
