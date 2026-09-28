"""
Escalation policy and output guardrails. Deterministic on purpose: whether a case needs a human is a
governance decision, so it must not depend on how a model happens to phrase its answer.

The agent MAY on its own: resolve medication identities, retrieve and quote label evidence, summarize it.
The agent MUST escalate when: a documented high-severity interaction is found, evidence is missing or
conflicting, a medication cannot be resolved, a required tool stays unavailable, or the self-correction
budget runs out.
"""
from __future__ import annotations

import re

HIGH_RISK = re.compile(
    r"\b(contraindicat\w*|do not (use|coadminister|co-administer|administer)|avoid (concomitant|coadministration|co-administration|use)|"
    r"life-threatening|fatal|serotonin syndrome|qt (interval )?prolongation|torsades|rhabdomyolysis|"
    r"(serious|major|severe) (bleeding|hemorrhage|haemorrhage|toxicity|hypotension|hyperkalemia|hypoglycemia|adverse)|"
    r"bleeding risk|increased? (the )?risk of bleeding|lactic acidosis|respiratory depression)\b", re.I)

UNSAFE_CLAIM = re.compile(
    r"\b(safe to (take|use|combine)|no (known )?interaction(s)?( exists?| between)?|can be (safely )?(taken|used) together|"
    r"is safe|are safe|stop taking|discontinue your|double (the|your) dose)\b", re.I)

INJECTION = re.compile(r"ignore (all |any |the )?(previous|prior|above) (instructions|rules)|system prompt|"
                       r"you are now|pretend to be|disregard (the|your) (rules|policy)", re.I)

MAX_REVISIONS = 2

# Risk concepts a finding may mention only if its cited passages mention them too (hallucination guard).
RISK_CONCEPTS = {
    "bleeding": r"bleed|hemorrhag|haemorrhag",
    "serotonin syndrome": r"serotonin syndrome",
    "QT prolongation": r"\bqt\b|torsade",
    "myopathy or rhabdomyolysis": r"rhabdomyolysis|myopathy",
    "contraindication": r"contraindicat",
    "fatal or life-threatening harm": r"fatal|life-threatening|death",
    "hypotension": r"hypotension",
    "hyperkalemia": r"hyperkalemia",
    "hypoglycemia": r"hypoglyc",
    "lactic acidosis": r"lactic acidosis",
    "respiratory depression": r"respiratory depression",
    "renal failure": r"renal failure|kidney failure|renal impairment",
}


def ungrounded_concepts(summary: str, cited_text: str) -> list[str]:
    return [c for c, rx in RISK_CONCEPTS.items() if re.search(rx, summary, re.I) and not re.search(rx, cited_text, re.I)]


def high_risk_terms(texts: list[str]) -> list[str]:
    found = []
    for t in texts:
        found += [m.group(0).lower() for m in HIGH_RISK.finditer(t)]
    return sorted(set(found))


def unsafe_claims(text: str) -> list[str]:
    return sorted({m.group(0).lower() for m in UNSAFE_CLAIM.finditer(text)})


def injection_detected(text: str) -> bool:
    return bool(INJECTION.search(text))
