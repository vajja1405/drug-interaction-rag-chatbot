"""
Tools the agent can call. Each call goes through ToolRegistry.call, which validates arguments,
retries transient failures with backoff, records a trajectory step, and can inject faults for tests.

  resolve_medication     free text / brand / misspelling → RxNorm ingredient (RxCUI) or 'unresolved'
  search_label_evidence  hybrid search over FDA label passages, filtered to cross-mentions of the pair
  get_patient_context    long-term patient memory (current medications, recent reviews)
  record_medications     write the reconciled medication list back to memory
  request_human_review   open a pharmacist review ticket (idempotent per case) and, when a webhook is
                         configured (AGENT_ESCALATION_WEBHOOK, e.g. the n8n workflow in integrations/n8n),
                         notify it; a failed notification never blocks the ticket
"""
from __future__ import annotations

import difflib
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx

from observability import span

# Brand → ingredient (extends rag_pipeline.retriever._DRUG_SYNONYMS, which lives behind heavy imports).
BRANDS = {
    "coumadin": "warfarin", "jantoven": "warfarin", "advil": "ibuprofen", "motrin": "ibuprofen",
    "bayer": "aspirin", "ecotrin": "aspirin", "glucophage": "metformin", "zestril": "lisinopril",
    "prinivil": "lisinopril", "plavix": "clopidogrel", "lanoxin": "digoxin", "cordarone": "amiodarone",
    "pacerone": "amiodarone", "prilosec": "omeprazole", "nexium": "esomeprazole", "tylenol": "acetaminophen",
    "paracetamol": "acetaminophen", "ultram": "tramadol", "zocor": "simvastatin", "lipitor": "atorvastatin",
    "crestor": "rosuvastatin", "prozac": "fluoxetine", "zoloft": "sertraline", "biaxin": "clarithromycin",
    "cipro": "ciprofloxacin", "zithromax": "azithromycin", "diflucan": "fluconazole", "dilantin": "phenytoin",
    "tegretol": "carbamazepine", "lasix": "furosemide", "aldactone": "spironolactone", "viagra": "sildenafil",
    "lopressor": "metoprolol", "toprol": "metoprolol", "calan": "verapamil", "trexall": "methotrexate",
    "zyrtec": "cetirizine", "pepcid": "famotidine", "singulair": "montelukast", "flomax": "tamsulosin",
    "proscar": "finasteride", "zetia": "ezetimibe", "januvia": "sitagliptin", "cozaar": "losartan",
    "neurontin": "gabapentin", "keppra": "levetiracetam", "protonix": "pantoprazole", "synthroid": "levothyroxine",
    "aricept": "donepezil", "namenda": "memantine", "rifadin": "rifampin", "rifampicin": "rifampin",
}

# Pharmacologic classes as they are written in FDA labels ("NSAIDs", "CYP3A4 inhibitors", ...).
# A label passage about drug A counts as evidence for pair (A, B) if it names B or one of B's classes.
CLASSES: dict[str, list[str]] = {}
for _cls, _members in {
    r"nsaids?|non-?steroidal anti-?inflammatory": "ibuprofen naproxen diclofenac celecoxib meloxicam ketorolac aspirin",
    r"antiplatelet|platelet aggregation inhibitors?|p2y12": "aspirin clopidogrel prasugrel ticagrelor",
    r"anticoagulants?|vitamin k antagonists?": "warfarin apixaban rivaroxaban dabigatran",
    r"ssris?|selective serotonin reuptake inhibitors?": "fluoxetine sertraline citalopram escitalopram paroxetine",
    r"snris?|serotonin[- ]norepinephrine reuptake inhibitors?": "venlafaxine duloxetine",
    r"serotonergic (drugs|agents)": "fluoxetine sertraline citalopram escitalopram paroxetine venlafaxine duloxetine "
                                    "tramadol fentanyl methadone sumatriptan trazodone linezolid",
    r"opioids?": "tramadol fentanyl methadone morphine oxycodone hydrocodone buprenorphine",
    r"(strong |moderate )?cyp3a4? inhibitors?": "clarithromycin erythromycin itraconazole ketoconazole voriconazole "
                                                 "ritonavir diltiazem verapamil fluconazole amiodarone",
    r"(strong )?cyp3a4? inducers?": "rifampin carbamazepine phenytoin",
    r"cyp2c9 inhibitors?": "fluconazole amiodarone voriconazole",
    r"beta[- ](adrenergic )?block(ers?|ing agents?)": "metoprolol atenolol carvedilol propranolol sotalol",
    r"ace inhibitors?|angiotensin[- ]converting enzyme inhibitors?": "lisinopril enalapril ramipril",
    r"angiotensin (ii )?receptor blockers?|arbs?": "losartan valsartan irbesartan",
    r"calcium channel blockers?": "verapamil diltiazem amlodipine nifedipine",
    r"(loop |thiazide )?diuretics?": "furosemide hydrochlorothiazide spironolactone",
    r"proton pump inhibitors?|ppis?": "omeprazole esomeprazole pantoprazole lansoprazole",
    r"statins?|hmg-coa reductase inhibitors?": "simvastatin atorvastatin lovastatin rosuvastatin pravastatin",
    r"(fluoro)?quinolones?": "ciprofloxacin levofloxacin moxifloxacin",
    r"macrolides?": "azithromycin clarithromycin erythromycin",
    r"antiarrhythmics?": "amiodarone dronedarone sotalol",
    r"digitalis|cardiac glycosides?": "digoxin",
    r"pde-?5 inhibitors?|phosphodiesterase": "sildenafil tadalafil",
    r"maois?|monoamine oxidase inhibitors?": "selegiline linezolid",
    r"benzodiazepines?|cns depressants?": "alprazolam lorazepam diazepam clonazepam zolpidem",
    r"sulfonylureas?|insulin secretagogues?": "glipizide glyburide glimepiride",
    r"antidiabetic|hypoglycemic agents?": "metformin glipizide glyburide glimepiride sitagliptin pioglitazone",
}.items():
    for _m in _members.split():
        CLASSES.setdefault(_m, []).append(_cls)


def mention_pattern(drug: str) -> re.Pattern:
    alts = [re.escape(drug)] + CLASSES.get(drug, [])
    return re.compile(r"\b(" + "|".join(alts) + r")\b", re.I)


class ToolError(Exception):
    def __init__(self, message: str, transient: bool = True):
        super().__init__(message)
        self.transient = transient


@dataclass
class FaultPlan:
    """Fault injection: fail the first `n` calls to a tool (n = -1 fails every call)."""
    failures: dict[str, int] = field(default_factory=dict)
    _seen: dict[str, int] = field(default_factory=dict)

    def check(self, tool: str) -> None:
        n = self.failures.get(tool, 0)
        k = self._seen.get(tool, 0)
        self._seen[tool] = k + 1
        if n == -1 or k < n:
            raise ToolError(f"injected fault in {tool} (call {k + 1})", transient=True)


@dataclass
class Step:
    tool: str
    args: dict
    ok: bool
    attempts: int
    latency_ms: float
    error: str | None = None
    node: str = ""


class ToolRegistry:
    def __init__(self, catalog, searcher, memory, review_queue=None, faults: FaultPlan | None = None,
                 max_retries: int = 2, backoff_s: float = 0.05, search_k: int = 8,
                 label_drugs: set[str] | None = None, escalation_webhook: str | None = None):
        self.catalog, self.searcher, self.memory = catalog, searcher, memory
        self.review_queue = review_queue if review_queue is not None else {}
        self.faults = faults or FaultPlan()
        self.max_retries, self.backoff_s, self.search_k = max_retries, backoff_s, search_k
        self.escalation_webhook = escalation_webhook or os.environ.get("AGENT_ESCALATION_WEBHOOK")
        self.label_drugs = label_drugs if label_drugs is not None else {p["drug"] for p in searcher.passages}
        self._ingredients = sorted(catalog.ingredients)
        self._common = sorted(self.label_drugs | set(BRANDS))
        self._tools: dict[str, Callable[..., Any]] = {
            "resolve_medication": self.resolve_medication,
            "search_label_evidence": self.search_label_evidence,
            "get_patient_context": self.get_patient_context,
            "record_medications": self.record_medications,
            "request_human_review": self.request_human_review,
        }

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def call(self, tool: str, /, node: str = "", **args) -> tuple[Any, dict]:
        """Run a tool with retries; returns (result or None, trajectory step)."""
        name = tool
        if name not in self._tools:
            step = Step(name, args, False, 0, 0.0, "unknown tool", node)
            return None, step.__dict__
        t0, attempts, err = time.perf_counter(), 0, None
        with span(f"tool.{name}", **{f"arg.{k}": str(v)[:80] for k, v in args.items()}):
            while attempts <= self.max_retries:
                attempts += 1
                try:
                    self.faults.check(name)
                    out = self._tools[name](**args)
                    return out, Step(name, args, True, attempts, (time.perf_counter() - t0) * 1000, None, node).__dict__
                except ToolError as exc:
                    err = str(exc)
                    if not exc.transient:
                        break
                    time.sleep(self.backoff_s * (2 ** (attempts - 1)))
                except (TypeError, ValueError, KeyError) as exc:  # bad arguments are not retried
                    err = f"{type(exc).__name__}: {exc}"
                    break
        return None, Step(name, args, False, attempts, (time.perf_counter() - t0) * 1000, err, node).__dict__

    # --- tools -----------------------------------------------------------------------------------
    def resolve_medication(self, name: str) -> dict:
        q = re.sub(r"\s+", " ", name.casefold()).strip()
        q = re.sub(r"\s+\d+(\.\d+)?\s*(mg|mcg|g|ml|units?)\b.*$", "", q)  # drop dose text
        if not q or len(q) < 3:
            raise ValueError("medication name too short")
        status = "resolved"
        if q in BRANDS:
            q, status = BRANDS[q], "brand_mapped"
        item = self.catalog.ingredients.get(q)
        if item is None:
            # Spelling correction: common drugs (labels in the corpus + known brands) at 0.8 similarity;
            # the 14k-ingredient long tail only at 0.9 and flagged, because a near miss there can be a
            # different substance ("insulin" vs "inulin").
            common = difflib.get_close_matches(q, self._common, n=1, cutoff=0.8)
            if common:
                q = BRANDS.get(common[0], common[0])
                item, status = self.catalog.ingredients.get(q), "spelling_corrected"
            if item is None:
                close = difflib.get_close_matches(q, self._ingredients, n=3, cutoff=0.9)
                if not close:
                    return {"query": name, "status": "unresolved", "candidates": []}
                item, status = self.catalog.ingredients[close[0]], "low_confidence_match"
        ing = item["name"].casefold()
        return {"query": name, "status": status, "rxcui": item["rxcui"], "ingredient": ing,
                "has_label": ing in self.label_drugs}

    def search_label_evidence(self, drug_a: str, drug_b: str, expand: bool = False) -> dict:
        if not drug_a or not drug_b:
            raise ValueError("two resolved drugs are required")
        queries = [f"{drug_a} {drug_b} drug interaction"]
        if expand:  # query rewriting: name each drug's pharmacologic class the way labels do
            for x, y in ((drug_a, drug_b), (drug_b, drug_a)):
                cls = [re.sub(r"[?()|\\]", " ", c).split()[0] for c in CLASSES.get(y, [])][:2]
                queries.append(f"{x} concomitant use {y} {' '.join(cls)}")
        k = self.search_k * (2 if expand else 1)
        seen, hits = set(), []
        for q in queries:  # label-scoped: rank only passages from the two drugs' own labels
            for h in self.searcher.search(q, k=k, drugs=(drug_a, drug_b)):
                if h.id not in seen:
                    seen.add(h.id)
                    hits.append(h)
        pa, pb = mention_pattern(drug_a), mention_pattern(drug_b)
        evidence = []
        for rank, h in enumerate(hits):
            if h.drug == drug_a and pb.search(h.text):
                evidence.append({"id": h.id, "label": h.drug, "about": drug_b, "rank": rank, "text": h.text})
            elif h.drug == drug_b and pa.search(h.text):
                evidence.append({"id": h.id, "label": h.drug, "about": drug_a, "rank": rank, "text": h.text})
        return {"queries": queries, "retrieved_ids": [h.id for h in hits], "evidence": evidence}

    def get_patient_context(self, patient_id: str) -> dict:
        if not re.fullmatch(r"P-\d{3,6}", patient_id or ""):
            raise ValueError("patient_id must look like P-123")
        return self.memory.get(patient_id)

    def record_medications(self, patient_id: str, medications: list[str], case_id: str, summary: str) -> dict:
        self.memory.record(patient_id, medications, case_id, summary)
        return {"patient_id": patient_id, "medications": medications}

    def request_human_review(self, case_id: str, reasons: list[str], priority: str = "routine") -> dict:
        if case_id in self.review_queue:  # idempotent: re-running a node must not open a second ticket
            return self.review_queue[case_id]
        ticket = {"ticket_id": f"RV-{uuid.uuid4().hex[:8]}", "case_id": case_id, "reasons": reasons,
                  "priority": priority, "status": "open"}
        if self.escalation_webhook:
            ticket["notified"] = self._notify(ticket)
        self.review_queue[case_id] = ticket
        return ticket

    def _notify(self, ticket: dict) -> dict:
        """POST the ticket to the routing webhook (n8n). Only ids, priority and reasons leave the agent."""
        try:
            r = httpx.post(self.escalation_webhook, json=ticket, timeout=5)
            r.raise_for_status()
            try:
                body = r.json()
            except ValueError:
                body = {}
            return {"ok": True, "status": r.status_code, "route": body.get("routed")}
        except httpx.HTTPError as exc:
            return {"ok": False, "error": type(exc).__name__}
