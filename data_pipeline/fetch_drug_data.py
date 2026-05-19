"""
data_pipeline/fetch_drug_data.py
─────────────────────────────────
Fetches drug-drug interaction records from two public, no-auth-required APIs:

  1. RxNorm / RxNav  (https://rxnav.nlm.nih.gov/REST)
     → structured DDI pairs with severity codes
  2. OpenFDA Drug Labels (https://api.fda.gov/drug/label.json)
     → free-text clinical narrative for richer embeddings

Canonical document schema produced by this module:
  {
    "drug_a": str,          # normalised lowercase name
    "drug_b": str,
    "rxcui_a": str | None,
    "rxcui_b": str | None,
    "severity": str,        # "Severe" | "Moderate" | "Low" | "Unknown"
    "severity_source": str, # "rxnorm" | "keyword" | "default"
    "mechanism": str,
    "clinical_effects": str,
    "management": str,
    "source": str,
    "raw_text": str,        # full narrative for embedding
  }

Usage:
  python -m data_pipeline.fetch_drug_data \
      --drugs warfarin aspirin metformin ibuprofen lisinopril \
      --out data/raw/interactions_raw.jsonl
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from itertools import combinations
from pathlib import Path

import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
RXNORM_BASE = "https://rxnav.nlm.nih.gov/REST"
OPENFDA_BASE = "https://api.fda.gov/drug/label.json"
CONCURRENCY = 5  # max simultaneous HTTP requests

# RxNorm severity code → human-readable label
RXNORM_SEVERITY_MAP = {
    "1": "Severe",        # contraindicated
    "2": "Severe",        # serious
    "3": "Moderate",      # significant
    "4": "Low",           # minor
    "N/A": "Unknown",
}

# ── Curated fallback knowledge base ──────────────────────────────────────────
# This seed data ensures the system works even when external APIs are
# unavailable. Records are based on well-established clinical literature.
SEED_INTERACTIONS: list[dict] = [
    {
        "drug_a": "warfarin", "drug_b": "aspirin",
        "rxcui_a": "11289", "rxcui_b": "1191",
        "severity": "Severe",
        "severity_source": "curated",
        "mechanism": "Pharmacodynamic synergy: additive anticoagulant and "
                     "antiplatelet effects. Aspirin also displaces warfarin "
                     "from plasma protein binding sites, raising free "
                     "warfarin concentration.",
        "clinical_effects": "Significantly increased risk of serious or "
                            "fatal bleeding, including gastrointestinal "
                            "haemorrhage and intracranial bleeding.",
        "management": "Avoid concurrent use if possible. If combination is "
                      "medically necessary, monitor INR at least weekly and "
                      "use lowest effective aspirin dose (≤100 mg/day). "
                      "Counsel patient to report any signs of bleeding.",
        "source": "curated",
        "source_url": "https://pubmed.ncbi.nlm.nih.gov/?term=warfarin+aspirin+drug+interaction+bleeding&sort=relevance",
        "source_page": "NIH PubMed – Warfarin + Aspirin Clinical Literature",
        "raw_text": (
            "Warfarin and aspirin interaction – Severe. "
            "Concomitant use of warfarin (anticoagulant) and aspirin "
            "(NSAID/antiplatelet) substantially increases the risk of "
            "haemorrhage. Aspirin inhibits thromboxane A2-dependent platelet "
            "aggregation and can cause gastric mucosal erosion, both "
            "potentiating warfarin's anticoagulant effect. Protein-binding "
            "displacement further increases free warfarin. "
            "Management: avoid or monitor INR weekly; lowest aspirin dose."
        ),
    },
    {
        "drug_a": "warfarin", "drug_b": "ibuprofen",
        "rxcui_a": "11289", "rxcui_b": "5640",
        "severity": "Severe",
        "severity_source": "curated",
        "mechanism": "Ibuprofen (NSAID) inhibits COX-1-mediated platelet "
                     "aggregation and causes gastric mucosal irritation. "
                     "It may also displace warfarin from albumin-binding "
                     "sites, transiently increasing free warfarin levels.",
        "clinical_effects": "Elevated bleeding risk, particularly "
                            "gastrointestinal bleeding. Possible INR "
                            "fluctuations.",
        "management": "Avoid combination; substitute with paracetamol "
                      "(acetaminophen) for analgesia if possible. "
                      "If NSAID is essential, monitor INR closely and "
                      "add a proton-pump inhibitor for GI protection.",
        "source": "curated",
        "source_url": "https://pubmed.ncbi.nlm.nih.gov/?term=warfarin+ibuprofen+NSAID+bleeding+INR&sort=relevance",
        "source_page": "NIH PubMed – Warfarin + NSAID Interaction Literature",
        "raw_text": (
            "Warfarin and ibuprofen interaction – Severe. "
            "Ibuprofen is a non-selective NSAID that inhibits platelet "
            "function and damages the gastric mucosa, both of which "
            "increase haemorrhagic risk in patients on warfarin. "
            "Protein-binding displacement can transiently raise free "
            "warfarin. Avoid; use acetaminophen if analgesic needed."
        ),
    },
    {
        "drug_a": "warfarin", "drug_b": "metformin",
        "rxcui_a": "11289", "rxcui_b": "6809",
        "severity": "Low",
        "severity_source": "curated",
        "mechanism": "No direct pharmacokinetic interaction. Metformin does "
                     "not significantly affect warfarin metabolism via "
                     "CYP2C9. No significant inhibition or induction of "
                     "CYP2C9 enzymes has been observed.",
        "clinical_effects": "Minimal interaction; routine INR monitoring "
                            "sufficient. Rare case reports of slightly "
                            "elevated INR.",
        "management": "Continue standard INR monitoring. No dose adjustment "
                      "typically required.",
        "source": "curated",
        "source_url": "https://pubmed.ncbi.nlm.nih.gov/?term=warfarin+metformin+drug+interaction+CYP2C9&sort=relevance",
        "source_page": "NIH PubMed – Warfarin + Metformin Interaction Literature",
        "raw_text": (
            "Warfarin and metformin interaction – Low. "
            "No clinically significant pharmacokinetic interaction has been "
            "established. Metformin does not inhibit or induce CYP2C9, the "
            "primary enzyme responsible for warfarin (S-enantiomer) "
            "metabolism. No significant change in INR has been documented "
            "in clinical studies. Routine INR monitoring is sufficient "
            "when co-administering these agents. No dose adjustment required."
        ),
    },
    {
        "drug_a": "aspirin", "drug_b": "ibuprofen",
        "rxcui_a": "1191", "rxcui_b": "5640",
        "severity": "Moderate",
        "severity_source": "curated",
        "mechanism": "Ibuprofen competitively inhibits the COX-1 binding "
                     "site of aspirin, blocking aspirin's irreversible "
                     "antiplatelet effect.",
        "clinical_effects": "Reduced cardioprotective effect of low-dose "
                            "aspirin; potential loss of antiplatelet benefit "
                            "in patients using aspirin for cardiovascular "
                            "prophylaxis.",
        "management": "Administer aspirin at least 30 minutes before "
                      "ibuprofen, or ≥8 hours after ibuprofen. Consider "
                      "alternative analgesic (paracetamol) in patients "
                      "requiring cardioprotection.",
        "source": "curated",
        "source_url": "https://pubmed.ncbi.nlm.nih.gov/?term=aspirin+ibuprofen+COX-1+antiplatelet+interaction&sort=relevance",
        "source_page": "NIH PubMed – Aspirin + Ibuprofen COX-1 Interaction Literature",
        "raw_text": (
            "Aspirin and ibuprofen interaction – Moderate. "
            "Ibuprofen competitively and reversibly occupies the COX-1 "
            "active site, preventing aspirin's irreversible acetylation and "
            "thus blunting its antiplatelet action. Patients on low-dose "
            "aspirin for cardiovascular protection may lose that benefit. "
            "Timing of administration matters: take aspirin ≥30 min before "
            "or ≥8 h after ibuprofen."
        ),
    },
    {
        "drug_a": "aspirin", "drug_b": "metformin",
        "rxcui_a": "1191", "rxcui_b": "6809",
        "severity": "Low",
        "severity_source": "curated",
        "mechanism": "No direct pharmacokinetic or pharmacodynamic "
                     "interaction. High-dose salicylates can theoretically "
                     "potentiate hypoglycaemia, but at typical aspirin "
                     "doses this is not clinically relevant.",
        "clinical_effects": "Minimal interaction. High-dose salicylates "
                            "(>3 g/day) may rarely contribute to "
                            "hypoglycaemia.",
        "management": "No action required at standard aspirin doses. "
                      "Monitor blood glucose if high-dose salicylate "
                      "therapy is initiated.",
        "source": "curated",
        "source_url": "https://pubmed.ncbi.nlm.nih.gov/?term=aspirin+metformin+salicylate+hypoglycemia&sort=relevance",
        "source_page": "NIH PubMed – Aspirin + Metformin Interaction Literature",
        "raw_text": (
            "Aspirin and metformin interaction – Low. "
            "No clinically significant interaction at standard aspirin doses. "
            "High-dose salicylates (anti-inflammatory dosing) may rarely "
            "contribute to hypoglycaemia via increased insulin sensitivity, "
            "but this is not a concern at typical antiplatelet doses."
        ),
    },
    {
        "drug_a": "ibuprofen", "drug_b": "metformin",
        "rxcui_a": "5640", "rxcui_b": "6809",
        "severity": "Moderate",
        "severity_source": "curated",
        "mechanism": "NSAIDs reduce renal prostaglandin synthesis, leading "
                     "to decreased renal perfusion and reduced metformin "
                     "clearance. This increases the risk of metformin "
                     "accumulation and lactic acidosis.",
        "clinical_effects": "Risk of acute kidney injury with consequent "
                            "metformin accumulation; rare but serious lactic "
                            "acidosis.",
        "management": "Use ibuprofen cautiously and for the shortest "
                      "duration necessary. Monitor renal function (serum "
                      "creatinine) and hold metformin if eGFR falls below "
                      "30 mL/min/1.73 m².",
        "source": "curated",
        "source_url": "https://pubmed.ncbi.nlm.nih.gov/?term=ibuprofen+metformin+renal+lactic+acidosis&sort=relevance",
        "source_page": "NIH PubMed – NSAID + Metformin Renal Interaction Literature",
        "raw_text": (
            "Ibuprofen and metformin interaction – Moderate. "
            "NSAIDs including ibuprofen can impair renal function by "
            "inhibiting prostaglandin-mediated afferent arteriolar dilation. "
            "Reduced GFR decreases metformin renal clearance, raising "
            "plasma metformin levels and increasing the risk of lactic "
            "acidosis. Monitor renal function; hold metformin if AKI occurs."
        ),
    },
    {
        "drug_a": "lisinopril", "drug_b": "ibuprofen",
        "rxcui_a": "29046", "rxcui_b": "5640",
        "severity": "Moderate",
        "severity_source": "curated",
        "mechanism": "NSAIDs blunt the antihypertensive and "
                     "cardioprotective effects of ACE inhibitors by "
                     "inhibiting prostaglandin-mediated vasodilation. "
                     "Combination also increases the risk of acute kidney "
                     "injury ('triple whammy' effect).",
        "clinical_effects": "Reduced antihypertensive efficacy; elevated "
                            "blood pressure. Increased risk of AKI "
                            "especially in volume-depleted or elderly "
                            "patients.",
        "management": "Avoid regular NSAID use in patients on ACE "
                      "inhibitors; use paracetamol for analgesia. "
                      "Monitor blood pressure and renal function if "
                      "short-term NSAID use is unavoidable.",
        "source": "curated",
        "source_url": "https://pubmed.ncbi.nlm.nih.gov/?term=lisinopril+ibuprofen+ACE+inhibitor+NSAID+renal&sort=relevance",
        "source_page": "NIH PubMed – ACE Inhibitor + NSAID Interaction Literature",
        "raw_text": (
            "Lisinopril (ACE inhibitor) and ibuprofen (NSAID) interaction – "
            "Moderate. NSAIDs inhibit prostaglandin synthesis and blunt "
            "the vasodilatory and natriuretic effects of ACE inhibitors, "
            "reducing antihypertensive efficacy. The combination impairs "
            "renal autoregulation, predisposing to AKI. Avoid concurrent "
            "use; prefer paracetamol for pain relief."
        ),
    },
    {
        "drug_a": "lisinopril", "drug_b": "metformin",
        "rxcui_a": "29046", "rxcui_b": "6809",
        "severity": "Low",
        "severity_source": "curated",
        "mechanism": "No direct pharmacokinetic interaction. "
                     "ACE inhibitors may improve insulin sensitivity, "
                     "potentially modestly enhancing metformin's "
                     "glucose-lowering effect.",
        "clinical_effects": "Generally favourable combination in diabetic "
                            "patients with hypertension. Rare mild "
                            "hypoglycaemia possible.",
        "management": "Monitor blood glucose periodically. "
                      "No dose adjustment typically required.",
        "source": "curated",
        "source_url": "https://pubmed.ncbi.nlm.nih.gov/?term=lisinopril+metformin+diabetes+hypertension&sort=relevance",
        "source_page": "NIH PubMed – Lisinopril + Metformin Interaction Literature",
        "raw_text": (
            "Lisinopril and metformin interaction – Low. "
            "This combination is frequently used and generally safe in "
            "patients with type 2 diabetes and hypertension. ACE inhibitors "
            "may modestly improve insulin sensitivity, complementing "
            "metformin's mechanism. Periodic blood glucose monitoring "
            "is standard care."
        ),
    },
    {
        "drug_a": "lisinopril", "drug_b": "aspirin",
        "rxcui_a": "29046", "rxcui_b": "1191",
        "severity": "Low",
        "severity_source": "curated",
        "mechanism": "High-dose aspirin (anti-inflammatory doses) may "
                     "reduce the vasodilatory benefit of ACE inhibitors via "
                     "prostaglandin inhibition. Low-dose aspirin "
                     "(≤100 mg/day) has minimal clinical impact.",
        "clinical_effects": "Slight reduction in antihypertensive "
                            "efficacy at high aspirin doses. Not clinically "
                            "significant at cardioprotective doses.",
        "management": "No specific action needed at low antiplatelet doses. "
                      "Prefer low-dose aspirin if antiplatelet therapy "
                      "is required.",
        "source": "curated",
        "source_url": "https://pubmed.ncbi.nlm.nih.gov/?term=lisinopril+aspirin+ACE+inhibitor+antihypertensive&sort=relevance",
        "source_page": "NIH PubMed – ACE Inhibitor + Aspirin Interaction Literature",
        "raw_text": (
            "Lisinopril and aspirin interaction – Low. "
            "High-dose salicylates may blunt the antihypertensive effect "
            "of ACE inhibitors. However, at standard cardioprotective "
            "aspirin doses (75–100 mg/day) the interaction is not "
            "clinically significant. No routine dose adjustment required."
        ),
    },
    {
        "drug_a": "warfarin", "drug_b": "lisinopril",
        "rxcui_a": "11289", "rxcui_b": "29046",
        "severity": "Low",
        "severity_source": "curated",
        "mechanism": "No significant pharmacokinetic interaction. "
                     "Neither drug substantially alters the other's "
                     "metabolism.",
        "clinical_effects": "No clinically relevant interaction. "
                            "Standard monitoring of INR is sufficient.",
        "management": "Continue routine INR monitoring. No special "
                      "precautions required beyond standard warfarin "
                      "management.",
        "source": "curated",
        "source_url": "https://pubmed.ncbi.nlm.nih.gov/?term=warfarin+lisinopril+drug+interaction+INR&sort=relevance",
        "source_page": "NIH PubMed – Warfarin + Lisinopril Interaction Literature",
        "raw_text": (
            "Warfarin and lisinopril interaction – Low. "
            "No clinically significant pharmacokinetic or pharmacodynamic "
            "interaction between warfarin and lisinopril has been "
            "established. Standard INR monitoring is appropriate."
        ),
    },
    # ── Additional common interactions ────────────────────────────────────────
    {
        "drug_a": "simvastatin", "drug_b": "amiodarone",
        "rxcui_a": "36567", "rxcui_b": "703",
        "severity": "Severe",
        "severity_source": "curated",
        "mechanism": "Amiodarone inhibits CYP3A4 and CYP2C9, significantly "
                     "reducing simvastatin metabolism and raising plasma "
                     "statin concentrations.",
        "clinical_effects": "Markedly increased risk of myopathy and "
                            "rhabdomyolysis with potential acute kidney injury.",
        "management": "Limit simvastatin dose to 20 mg/day when used with "
                      "amiodarone. Consider switching to a statin not "
                      "metabolised by CYP3A4 (e.g., rosuvastatin).",
        "source": "curated",
        "source_url": "https://pubmed.ncbi.nlm.nih.gov/?term=simvastatin+amiodarone+CYP3A4+rhabdomyolysis+myopathy&sort=relevance",
        "source_page": "NIH PubMed – Simvastatin + Amiodarone CYP3A4 Literature",
        "raw_text": (
            "Simvastatin and amiodarone interaction – Severe. "
            "Amiodarone is a potent inhibitor of CYP3A4 and CYP2C9, "
            "the primary enzymes metabolising simvastatin. "
            "Co-administration can increase simvastatin AUC up to 6-fold, "
            "greatly elevating the risk of myopathy and rhabdomyolysis. "
            "FDA recommends capping simvastatin at 20 mg/day with amiodarone "
            "or switching to a non-CYP3A4 substrate statin."
        ),
    },
    {
        "drug_a": "clopidogrel", "drug_b": "omeprazole",
        "rxcui_a": "174742", "rxcui_b": "7646",
        "severity": "Moderate",
        "severity_source": "curated",
        "mechanism": "Omeprazole inhibits CYP2C19, the enzyme responsible "
                     "for converting clopidogrel to its active metabolite. "
                     "This reduces clopidogrel's antiplatelet effect.",
        "clinical_effects": "Reduced platelet inhibition; potential "
                            "increased cardiovascular event risk in "
                            "high-risk patients (e.g., post-PCI).",
        "management": "Consider alternative PPI with lower CYP2C19 "
                      "inhibition (pantoprazole, rabeprazole) or H2 "
                      "blocker if gastroprotection is needed.",
        "source": "curated",
        "source_url": "https://pubmed.ncbi.nlm.nih.gov/?term=clopidogrel+omeprazole+CYP2C19+antiplatelet+PPI&sort=relevance",
        "source_page": "NIH PubMed – Clopidogrel + Omeprazole CYP2C19 Literature",
        "raw_text": (
            "Clopidogrel and omeprazole interaction – Moderate. "
            "Omeprazole is a moderate-to-strong CYP2C19 inhibitor. "
            "Clopidogrel requires CYP2C19-mediated bioactivation to "
            "exert antiplatelet effects. Co-administration significantly "
            "reduces active metabolite formation and platelet inhibition. "
            "Prefer pantoprazole or rabeprazole for GI protection in "
            "patients on dual antiplatelet therapy."
        ),
    },
    {
        "drug_a": "methotrexate", "drug_b": "ibuprofen",
        "rxcui_a": "7953", "rxcui_b": "5640",
        "severity": "Severe",
        "severity_source": "curated",
        "mechanism": "NSAIDs reduce renal tubular secretion of methotrexate, "
                     "significantly increasing methotrexate plasma levels "
                     "and half-life.",
        "clinical_effects": "Methotrexate toxicity: myelosuppression, "
                            "mucositis, hepatotoxicity, nephrotoxicity. "
                            "Can be life-threatening.",
        "management": "Avoid NSAIDs with high-dose methotrexate. "
                      "If low-dose methotrexate (e.g., for RA), monitor "
                      "CBC and renal/hepatic function closely. "
                      "Use paracetamol for analgesia instead.",
        "source": "curated",
        "source_url": "https://pubmed.ncbi.nlm.nih.gov/?term=methotrexate+ibuprofen+NSAID+toxicity+renal+tubular&sort=relevance",
        "source_page": "NIH PubMed – Methotrexate + NSAID Interaction Literature",
        "raw_text": (
            "Methotrexate and ibuprofen interaction – Severe. "
            "NSAIDs including ibuprofen inhibit renal tubular secretion "
            "of methotrexate via organic anion transport, leading to "
            "prolonged and elevated methotrexate exposure. This interaction "
            "can precipitate life-threatening myelosuppression, "
            "severe mucositis, and nephrotoxicity even at low methotrexate "
            "doses. Avoid combination; use paracetamol for pain relief."
        ),
    },
    {
        "drug_a": "digoxin", "drug_b": "amiodarone",
        "rxcui_a": "3407", "rxcui_b": "703",
        "severity": "Severe",
        "severity_source": "curated",
        "mechanism": "Amiodarone inhibits P-glycoprotein and reduces renal "
                     "clearance of digoxin, raising digoxin serum "
                     "concentrations by 70–100%.",
        "clinical_effects": "Digoxin toxicity: bradycardia, heart block, "
                            "nausea, visual disturbances, potentially "
                            "fatal arrhythmias.",
        "management": "Reduce digoxin dose by 30–50% when initiating "
                      "amiodarone. Monitor digoxin levels and ECG closely. "
                      "Adjust dose to maintain serum digoxin 0.5–0.9 ng/mL.",
        "source": "curated",
        "source_url": "https://pubmed.ncbi.nlm.nih.gov/?term=digoxin+amiodarone+P-glycoprotein+toxicity+bradycardia&sort=relevance",
        "source_page": "NIH PubMed – Digoxin + Amiodarone Interaction Literature",
        "raw_text": (
            "Digoxin and amiodarone interaction – Severe. "
            "Amiodarone inhibits P-glycoprotein-mediated renal tubular "
            "secretion of digoxin and reduces digoxin volume of "
            "distribution. Serum digoxin levels typically rise 70–100%. "
            "Symptoms of toxicity include bradycardia, AV block, "
            "and ventricular arrhythmias. Halve the digoxin dose when "
            "starting amiodarone; titrate to serum levels and clinical "
            "response."
        ),
    },
    # ── Fluoxetine + Tramadol ─────────────────────────────────────────────────
    {
        "drug_a": "fluoxetine", "drug_b": "tramadol",
        "rxcui_a": "41493", "rxcui_b": "41493",
        "severity": "Severe",
        "severity_source": "curated",
        "mechanism": "Fluoxetine inhibits serotonin reuptake (SSRI) and also "
                     "inhibits CYP2D6, the enzyme responsible for tramadol "
                     "activation. Tramadol itself inhibits serotonin and "
                     "noradrenaline reuptake. Combined serotonergic stimulation "
                     "greatly increases serotonin syndrome risk.",
        "clinical_effects": "Serotonin syndrome: agitation, hyperthermia, "
                            "tachycardia, clonus, diaphoresis — potentially "
                            "life-threatening. CYP2D6 inhibition by fluoxetine "
                            "also raises tramadol plasma levels. Lowered "
                            "seizure threshold.",
        "management": "Avoid concurrent use of fluoxetine and tramadol. "
                      "If pain management is required, consider an alternative "
                      "analgesic not metabolised by CYP2D6. Monitor closely "
                      "for signs of serotonin syndrome if combination is "
                      "unavoidable.",
        "source": "curated",
        "source_url": "https://pubmed.ncbi.nlm.nih.gov/?term=fluoxetine+tramadol+serotonin+syndrome+CYP2D6&sort=relevance",
        "source_page": "NIH PubMed – Fluoxetine + Tramadol Serotonin Syndrome Literature",
        "raw_text": (
            "Fluoxetine and tramadol interaction – Severe. "
            "Fluoxetine is a selective serotonin reuptake inhibitor (SSRI) "
            "and a potent CYP2D6 inhibitor. Tramadol inhibits serotonin and "
            "noradrenaline reuptake and requires CYP2D6-mediated activation. "
            "Co-administration produces additive serotonergic stimulation, "
            "greatly increasing the risk of serotonin syndrome (agitation, "
            "hyperthermia, clonus, autonomic instability). CYP2D6 inhibition "
            "by fluoxetine elevates tramadol exposure further. Avoid this "
            "combination. Use an alternative analgesic not affected by "
            "CYP2D6 inhibition. If combination cannot be avoided, use the "
            "lowest effective tramadol dose and monitor closely for serotonin "
            "syndrome symptoms."
        ),
    },
    # ── Lisinopril + Potassium Chloride ───────────────────────────────────────
    {
        "drug_a": "lisinopril", "drug_b": "potassium chloride",
        "rxcui_a": "29046", "rxcui_b": "8591",
        "severity": "Moderate",
        "severity_source": "curated",
        "mechanism": "ACE inhibitors such as lisinopril reduce aldosterone "
                     "secretion, decreasing renal potassium excretion. "
                     "Concurrent potassium chloride supplementation can "
                     "cause additive potassium retention, leading to "
                     "hyperkalemia.",
        "clinical_effects": "Hyperkalemia: muscle weakness, cardiac "
                            "arrhythmias, potentially life-threatening "
                            "in severe cases. Risk is amplified in patients "
                            "with renal impairment.",
        "management": "Monitor serum potassium levels regularly. "
                      "Adjust potassium chloride dose based on renal "
                      "function. Avoid high-dose potassium supplementation "
                      "in patients with renal impairment on ACE inhibitors.",
        "source": "curated",
        "source_url": "https://pubmed.ncbi.nlm.nih.gov/?term=lisinopril+potassium+ACE+inhibitor+hyperkalemia+aldosterone&sort=relevance",
        "source_page": "NIH PubMed – ACE Inhibitor + Potassium Hyperkalemia Literature",
        "raw_text": (
            "Lisinopril and potassium chloride interaction – Moderate. "
            "Lisinopril (ACE inhibitor) reduces aldosterone production, "
            "decreasing renal potassium clearance. Adding potassium "
            "chloride supplementation can cause clinically significant "
            "hyperkalemia, particularly in patients with renal impairment "
            "or diabetes. Serum potassium and renal function should be "
            "monitored closely. Adjust potassium supplementation doses "
            "accordingly."
        ),
    },
    # ── Warfarin + Amiodarone ─────────────────────────────────────────────────
    {
        "drug_a": "warfarin", "drug_b": "amiodarone",
        "rxcui_a": "11289", "rxcui_b": "703",
        "severity": "Severe",
        "severity_source": "curated",
        "mechanism": "Amiodarone and its active metabolite "
                     "desethylamiodarone inhibit CYP2C9 and CYP3A4, the "
                     "main enzymes responsible for warfarin metabolism. "
                     "This significantly impairs warfarin clearance and "
                     "raises free warfarin levels.",
        "clinical_effects": "Markedly elevated INR; substantially increased "
                            "risk of serious bleeding including intracranial "
                            "haemorrhage. Effect may persist weeks after "
                            "amiodarone is stopped due to its very long "
                            "half-life.",
        "management": "Reduce warfarin dose by approximately 33–50% when "
                      "initiating amiodarone. Monitor INR at least weekly "
                      "for the first month, then monthly. Be aware that the "
                      "interaction may persist for months after amiodarone "
                      "discontinuation.",
        "source": "curated",
        "source_url": "https://pubmed.ncbi.nlm.nih.gov/?term=warfarin+amiodarone+CYP2C9+CYP3A4+INR+bleeding&sort=relevance",
        "source_page": "NIH PubMed – Warfarin + Amiodarone CYP Inhibition Literature",
        "raw_text": (
            "Warfarin and amiodarone interaction – Severe. "
            "Amiodarone is a potent inhibitor of CYP2C9 and CYP3A4, the "
            "enzymes primarily responsible for warfarin (S- and R-enantiomer) "
            "metabolism. Co-administration dramatically reduces warfarin "
            "clearance, raising INR by 33–50% or more. Serious bleeding "
            "events including intracranial haemorrhage have been reported. "
            "Reduce warfarin dose by approximately 33% when initiating "
            "amiodarone. Monitor INR at least weekly for the first month. "
            "The interaction persists for weeks to months after amiodarone "
            "is stopped due to its very long half-life."
        ),
    },
    # ── Lithium + Ibuprofen ───────────────────────────────────────────────────
    {
        "drug_a": "lithium", "drug_b": "ibuprofen",
        "rxcui_a": "6448", "rxcui_b": "5640",
        "severity": "Severe",
        "severity_source": "curated",
        "mechanism": "NSAIDs including ibuprofen inhibit renal "
                     "prostaglandin synthesis, reducing renal blood flow "
                     "and sodium excretion. This decreases lithium "
                     "clearance by reducing lithium's renal tubular "
                     "secretion — the kidney reabsorbs lithium instead "
                     "of sodium when sodium is depleted.",
        "clinical_effects": "Lithium toxicity: tremor, nausea, diarrhoea, "
                            "confusion, seizures, cardiac arrhythmias. "
                            "Can be life-threatening at high lithium levels.",
        "management": "Avoid ibuprofen in patients on lithium. If an "
                      "NSAID is necessary, monitor lithium levels closely "
                      "and check renal function. Use paracetamol as the "
                      "preferred analgesic alternative.",
        "source": "curated",
        "source_url": "https://pubmed.ncbi.nlm.nih.gov/?term=lithium+ibuprofen+NSAID+toxicity+renal+sodium&sort=relevance",
        "source_page": "NIH PubMed – Lithium + NSAID Interaction Literature",
        "raw_text": (
            "Lithium and ibuprofen interaction – Severe. "
            "Ibuprofen (NSAID) inhibits prostaglandin-mediated renal "
            "vasodilation and reduces sodium excretion. The kidney "
            "compensates by increasing tubular reabsorption of both sodium "
            "and lithium, thereby reducing lithium clearance and raising "
            "plasma lithium levels by 25–60%. Lithium toxicity can present "
            "as tremor, confusion, and cardiac arrhythmia. Avoid ibuprofen; "
            "monitor lithium level and renal function if any NSAID is used."
        ),
    },
    # ── Aspirin + Clopidogrel ─────────────────────────────────────────────────
    {
        "drug_a": "aspirin", "drug_b": "clopidogrel",
        "rxcui_a": "1191", "rxcui_b": "174742",
        "severity": "Moderate",
        "severity_source": "curated",
        "mechanism": "Pharmacodynamic synergy: aspirin irreversibly inhibits "
                     "COX-1-dependent thromboxane A2 production (reducing "
                     "platelet activation), while clopidogrel irreversibly "
                     "blocks the P2Y12 ADP receptor. Dual antiplatelet "
                     "therapy produces additive platelet inhibition and "
                     "increased bleeding risk.",
        "clinical_effects": "Increased risk of bleeding, particularly "
                            "gastrointestinal bleeding and minor haemorrhages. "
                            "Beneficial in high-risk cardiovascular patients "
                            "(post-ACS, post-PCI) but requires careful "
                            "risk–benefit assessment.",
        "management": "Dual antiplatelet therapy is clinically indicated "
                      "in specific settings (e.g., ACS, coronary stenting). "
                      "Limit duration to guideline-recommended timeframes. "
                      "Monitor for signs of bleeding. Add gastroprotection "
                      "(PPI) to reduce GI bleeding risk.",
        "source": "curated",
        "source_url": "https://pubmed.ncbi.nlm.nih.gov/?term=aspirin+clopidogrel+dual+antiplatelet+bleeding+P2Y12&sort=relevance",
        "source_page": "NIH PubMed – Dual Antiplatelet Therapy Literature",
        "raw_text": (
            "Aspirin and clopidogrel interaction – Moderate. "
            "Dual antiplatelet therapy with aspirin and clopidogrel produces "
            "additive inhibition of platelet aggregation via complementary "
            "antiplatelet mechanisms — COX-1 inhibition (aspirin) and P2Y12 "
            "blockade (clopidogrel). While indicated in post-ACS and "
            "post-stent patients, the combination carries a meaningfully "
            "higher bleeding risk than monotherapy. Monitor closely for "
            "signs of bleeding including gastrointestinal haemorrhage. "
            "Co-prescribe a PPI for GI protection. Limit dual antiplatelet "
            "therapy duration to guideline-recommended timeframes."
        ),
    },
    {
        "drug_a": "ssri", "drug_b": "tramadol",
        "rxcui_a": "36437", "rxcui_b": "41493",
        "severity": "Severe",
        "severity_source": "curated",
        "mechanism": "SSRIs inhibit serotonin reuptake; tramadol also "
                     "inhibits serotonin and noradrenaline reuptake and "
                     "has weak opioid agonist activity. Combination "
                     "increases serotonergic tone markedly.",
        "clinical_effects": "Serotonin syndrome: agitation, hyperthermia, "
                            "tachycardia, clonus, diaphoresis. Can be "
                            "life-threatening. Also lowers seizure threshold.",
        "management": "Avoid combination. If concurrent use necessary, "
                      "use lowest effective tramadol dose and monitor "
                      "closely for serotonin syndrome symptoms. "
                      "Consider alternative analgesic.",
        "source": "curated",
        "source_url": "https://pubmed.ncbi.nlm.nih.gov/?term=SSRI+tramadol+serotonin+syndrome+seizure&sort=relevance",
        "source_page": "NIH PubMed – SSRI + Tramadol Serotonin Syndrome Literature",
        "raw_text": (
            "SSRI and tramadol interaction – Severe. "
            "Both agents increase synaptic serotonin levels. SSRIs block "
            "serotonin reuptake; tramadol inhibits both serotonin and "
            "noradrenaline reuptake in addition to its opioid activity. "
            "Their combination creates additive serotonergic stimulation, "
            "risking serotonin syndrome (agitation, hyperthermia, clonus, "
            "autonomic instability). Also increases seizure risk. Avoid."
        ),
    },
    # ── NEW PAIRS ─────────────────────────────────────────────────────────────
    # ── Sildenafil + Nitroglycerin ────────────────────────────────────────────
    {
        "drug_a": "nitroglycerin", "drug_b": "sildenafil",
        "rxcui_a": "4917", "rxcui_b": "269560",
        "severity": "Severe",
        "severity_source": "curated",
        "mechanism": "Both sildenafil (PDE5 inhibitor) and nitroglycerin "
                     "(nitrate) increase cyclic GMP, causing potent "
                     "vasodilation. Co-administration produces synergistic, "
                     "additive reduction in systemic vascular resistance.",
        "clinical_effects": "Profound, potentially fatal hypotension. "
                            "Severe drop in blood pressure may lead to "
                            "syncope, myocardial infarction, or stroke.",
        "management": "Contraindicated. Do not use nitrates within 24 hours "
                      "of sildenafil (or 48 hours for tadalafil). If urgent "
                      "nitrate therapy is needed in a patient who has taken "
                      "sildenafil, haemodynamic monitoring in ICU setting is "
                      "required.",
        "source": "curated",
        "source_url": "https://pubmed.ncbi.nlm.nih.gov/?term=sildenafil+nitroglycerin+nitrate+hypotension+PDE5&sort=relevance",
        "source_page": "NIH PubMed – PDE5 Inhibitor + Nitrate Hypotension Literature",
        "raw_text": (
            "Sildenafil and nitroglycerin interaction – Severe. "
            "Sildenafil inhibits phosphodiesterase type 5 (PDE5), preventing "
            "degradation of cyclic GMP and enhancing nitric oxide-mediated "
            "vasodilation. Nitroglycerin (organic nitrate) donates nitric "
            "oxide directly, also raising cGMP. The combination produces "
            "synergistic, severe hypotension that can be life-threatening. "
            "This combination is absolutely contraindicated. Do not administer "
            "nitrates within 24 hours of sildenafil use. Emergency nitrate "
            "use after sildenafil requires intensive haemodynamic monitoring."
        ),
    },
    # ── Warfarin + Fluconazole ────────────────────────────────────────────────
    {
        "drug_a": "fluconazole", "drug_b": "warfarin",
        "rxcui_a": "4450", "rxcui_b": "11289",
        "severity": "Severe",
        "severity_source": "curated",
        "mechanism": "Fluconazole is a potent CYP2C9 inhibitor and also "
                     "inhibits CYP3A4. As CYP2C9 is the primary enzyme "
                     "metabolising the more potent S-enantiomer of warfarin, "
                     "fluconazole dramatically reduces warfarin clearance.",
        "clinical_effects": "Significantly elevated INR with high risk of "
                            "serious or fatal bleeding. Even a single dose "
                            "of fluconazole can cause dangerous INR elevation.",
        "management": "Reduce warfarin dose proactively by 25–50% when "
                      "starting fluconazole. Monitor INR daily during "
                      "co-administration and for several days after fluconazole "
                      "is stopped. Consider alternative antifungal if feasible.",
        "source": "curated",
        "source_url": "https://pubmed.ncbi.nlm.nih.gov/?term=fluconazole+warfarin+CYP2C9+INR+bleeding+azole&sort=relevance",
        "source_page": "NIH PubMed – Fluconazole + Warfarin CYP2C9 Interaction Literature",
        "raw_text": (
            "Fluconazole and warfarin interaction – Severe. "
            "Fluconazole is a potent inhibitor of CYP2C9 and moderate "
            "inhibitor of CYP3A4. These are the primary enzymes metabolising "
            "warfarin enantiomers. Even a short course of fluconazole can "
            "double warfarin plasma levels and cause dangerous INR elevation. "
            "Reduce warfarin dose by 25–50% when fluconazole is prescribed. "
            "Monitor INR daily. Resume prior warfarin dosing after "
            "fluconazole course and recheck INR."
        ),
    },
    # ── Atorvastatin + Clarithromycin ─────────────────────────────────────────
    {
        "drug_a": "atorvastatin", "drug_b": "clarithromycin",
        "rxcui_a": "83367", "rxcui_b": "21212",
        "severity": "Severe",
        "severity_source": "curated",
        "mechanism": "Clarithromycin is a potent CYP3A4 inhibitor. "
                     "Atorvastatin is extensively metabolised by CYP3A4. "
                     "Co-administration greatly increases atorvastatin "
                     "plasma concentrations (AUC up to 4.5-fold).",
        "clinical_effects": "Markedly elevated statin levels; high risk of "
                            "myopathy and rhabdomyolysis with associated "
                            "acute kidney injury.",
        "management": "Temporarily suspend atorvastatin during the "
                      "clarithromycin course. Restart atorvastatin after "
                      "antibiotic is completed. If statin therapy cannot "
                      "be interrupted, use a lower dose or consider a "
                      "non-CYP3A4 substrate statin (rosuvastatin, pravastatin).",
        "source": "curated",
        "source_url": "https://pubmed.ncbi.nlm.nih.gov/?term=atorvastatin+clarithromycin+CYP3A4+rhabdomyolysis+statin&sort=relevance",
        "source_page": "NIH PubMed – Statin + Macrolide CYP3A4 Interaction Literature",
        "raw_text": (
            "Atorvastatin and clarithromycin interaction – Severe. "
            "Clarithromycin strongly inhibits CYP3A4, the enzyme primarily "
            "responsible for atorvastatin metabolism. Co-administration "
            "increases atorvastatin AUC by up to 4.5-fold, greatly elevating "
            "the risk of statin-induced myopathy and rhabdomyolysis. "
            "Temporarily suspend atorvastatin during the antibiotic course. "
            "Consider non-CYP3A4 statins if concurrent therapy cannot be avoided."
        ),
    },
    # ── Metoprolol + Verapamil ────────────────────────────────────────────────
    {
        "drug_a": "metoprolol", "drug_b": "verapamil",
        "rxcui_a": "6918", "rxcui_b": "11170",
        "severity": "Severe",
        "severity_source": "curated",
        "mechanism": "Both metoprolol (beta-1 selective blocker) and "
                     "verapamil (non-dihydropyridine calcium channel blocker) "
                     "depress sinoatrial and atrioventricular nodal conduction. "
                     "Verapamil also inhibits CYP2D6, raising metoprolol levels.",
        "clinical_effects": "Severe bradycardia, complete heart block, "
                            "and cardiogenic shock. Life-threatening cardiac "
                            "arrest has been reported.",
        "management": "Avoid combination if possible. If concurrent use is "
                      "necessary, start with very low doses and monitor "
                      "heart rate and ECG continuously. Have atropine and "
                      "cardiac pacing available.",
        "source": "curated",
        "source_url": "https://pubmed.ncbi.nlm.nih.gov/?term=metoprolol+verapamil+bradycardia+heart+block+beta+blocker&sort=relevance",
        "source_page": "NIH PubMed – Beta-Blocker + Verapamil Cardiac Interaction Literature",
        "raw_text": (
            "Metoprolol and verapamil interaction – Severe. "
            "The combination of a beta-blocker (metoprolol) and a "
            "non-dihydropyridine calcium channel blocker (verapamil) produces "
            "additive negative chronotropic and dromotropic effects on the "
            "heart. Both agents independently slow AV nodal conduction; "
            "together they can cause complete heart block, severe bradycardia, "
            "or cardiogenic shock. Verapamil also inhibits CYP2D6, increasing "
            "metoprolol plasma concentrations. Avoid this combination; "
            "if used, close cardiac monitoring is mandatory."
        ),
    },
    # ── Ciprofloxacin + Warfarin ──────────────────────────────────────────────
    {
        "drug_a": "ciprofloxacin", "drug_b": "warfarin",
        "rxcui_a": "2551", "rxcui_b": "11289",
        "severity": "Moderate",
        "severity_source": "curated",
        "mechanism": "Ciprofloxacin inhibits CYP1A2 and may disrupt gut "
                     "flora that produce vitamin K. Both mechanisms reduce "
                     "warfarin clearance and increase anticoagulant effect.",
        "clinical_effects": "Elevated INR with increased bleeding risk. "
                            "INR can rise significantly within 2–3 days "
                            "of starting ciprofloxacin.",
        "management": "Monitor INR closely when ciprofloxacin is started "
                      "or stopped. Reduce warfarin dose if INR rises "
                      "above therapeutic range. Resume monitoring for "
                      "several days after antibiotic course ends.",
        "source": "curated",
        "source_url": "https://pubmed.ncbi.nlm.nih.gov/?term=ciprofloxacin+warfarin+INR+CYP1A2+fluoroquinolone&sort=relevance",
        "source_page": "NIH PubMed – Fluoroquinolone + Warfarin Interaction Literature",
        "raw_text": (
            "Ciprofloxacin and warfarin interaction – Moderate. "
            "Ciprofloxacin inhibits CYP1A2 and disrupts intestinal flora "
            "that synthesise vitamin K, both of which potentiate warfarin's "
            "anticoagulant effect. INR can increase significantly within "
            "2–3 days of starting the antibiotic. Monitor INR at least "
            "every 2–3 days during concurrent therapy and for 5–7 days "
            "after ciprofloxacin is stopped."
        ),
    },
    # ── Rifampicin + Warfarin ─────────────────────────────────────────────────
    {
        "drug_a": "rifampicin", "drug_b": "warfarin",
        "rxcui_a": "9384", "rxcui_b": "11289",
        "severity": "Severe",
        "severity_source": "curated",
        "mechanism": "Rifampicin is a potent inducer of CYP2C9, CYP3A4, "
                     "and P-glycoprotein. This dramatically accelerates "
                     "warfarin metabolism and reduces its plasma levels "
                     "by up to 85%.",
        "clinical_effects": "Markedly reduced anticoagulant effect; "
                            "thrombotic events including stroke and "
                            "pulmonary embolism reported in patients whose "
                            "warfarin was subtherapeutic.",
        "management": "Increase warfarin dose substantially (often 2–5-fold) "
                      "during rifampicin therapy with daily INR monitoring. "
                      "After rifampicin is stopped, the inducing effect "
                      "wanes over 2–4 weeks; reduce warfarin dose gradually "
                      "and monitor INR closely to avoid rebound bleeding.",
        "source": "curated",
        "source_url": "https://pubmed.ncbi.nlm.nih.gov/?term=rifampicin+warfarin+CYP+induction+INR+anticoagulant&sort=relevance",
        "source_page": "NIH PubMed – Rifampicin + Warfarin CYP Induction Literature",
        "raw_text": (
            "Rifampicin and warfarin interaction – Severe. "
            "Rifampicin is one of the most potent known inducers of CYP2C9 "
            "and CYP3A4. It dramatically accelerates warfarin metabolism, "
            "reducing plasma warfarin levels by up to 85% and rendering "
            "standard doses completely subtherapeutic. Patients on warfarin "
            "who start rifampicin require large dose increases and daily INR "
            "monitoring. When rifampicin is stopped, enzyme induction persists "
            "for 2–4 weeks; warfarin dose must be tapered carefully."
        ),
    },
    # ── Spironolactone + Potassium ────────────────────────────────────────────
    {
        "drug_a": "potassium chloride", "drug_b": "spironolactone",
        "rxcui_a": "8591", "rxcui_b": "9997",
        "severity": "Severe",
        "severity_source": "curated",
        "mechanism": "Spironolactone is a potassium-sparing diuretic that "
                     "blocks aldosterone receptors in the collecting duct, "
                     "reducing urinary potassium excretion. Combined with "
                     "potassium supplementation, significant hyperkalaemia "
                     "can develop rapidly.",
        "clinical_effects": "Hyperkalaemia: weakness, paralysis, potentially "
                            "fatal ventricular arrhythmias and cardiac arrest.",
        "management": "Avoid routine potassium supplementation with "
                      "spironolactone unless hypokalaemia is documented. "
                      "Monitor serum potassium frequently. Reduce or "
                      "discontinue potassium supplements if levels rise "
                      "above 5.0 mmol/L.",
        "source": "curated",
        "source_url": "https://pubmed.ncbi.nlm.nih.gov/?term=spironolactone+potassium+hyperkalemia+aldosterone+diuretic&sort=relevance",
        "source_page": "NIH PubMed – Spironolactone + Potassium Hyperkalemia Literature",
        "raw_text": (
            "Potassium chloride and spironolactone interaction – Severe. "
            "Spironolactone (potassium-sparing diuretic) blocks aldosterone "
            "in the renal collecting duct, reducing potassium excretion. "
            "Concurrent potassium supplementation produces additive "
            "potassium retention, creating a high risk of life-threatening "
            "hyperkalaemia and cardiac arrhythmias. Avoid this combination "
            "unless documented hypokalaemia exists. Monitor serum potassium "
            "closely."
        ),
    },
    # ── Gentamicin + Furosemide ───────────────────────────────────────────────
    {
        "drug_a": "furosemide", "drug_b": "gentamicin",
        "rxcui_a": "4603", "rxcui_b": "4826",
        "severity": "Severe",
        "severity_source": "curated",
        "mechanism": "Both furosemide (loop diuretic) and gentamicin "
                     "(aminoglycoside) are independently nephrotoxic and "
                     "ototoxic. Furosemide reduces renal blood flow, "
                     "reducing gentamicin clearance and increasing its "
                     "accumulation in renal tubular cells and the cochlea.",
        "clinical_effects": "Synergistic nephrotoxicity (acute tubular "
                            "necrosis) and ototoxicity (irreversible "
                            "sensorineural hearing loss and vestibular "
                            "damage).",
        "management": "Avoid concurrent use when possible. If essential, "
                      "use lowest effective doses, monitor gentamicin trough "
                      "levels daily, and assess renal function and hearing. "
                      "Use once-daily gentamicin dosing to reduce cochlear "
                      "accumulation.",
        "source": "curated",
        "source_url": "https://pubmed.ncbi.nlm.nih.gov/?term=gentamicin+furosemide+ototoxicity+nephrotoxicity+aminoglycoside&sort=relevance",
        "source_page": "NIH PubMed – Aminoglycoside + Loop Diuretic Toxicity Literature",
        "raw_text": (
            "Furosemide and gentamicin interaction – Severe. "
            "Both agents carry independent risks of nephrotoxicity and "
            "ototoxicity; combination dramatically amplifies both risks. "
            "Furosemide-induced renal vasoconstriction reduces gentamicin "
            "clearance, increasing its accumulation in renal tubular cells "
            "and the stria vascularis of the cochlea. This can cause "
            "irreversible sensorineural hearing loss and AKI. Monitor "
            "gentamicin levels and renal function daily when co-administering."
        ),
    },
    # ── Sertraline + Warfarin ─────────────────────────────────────────────────
    {
        "drug_a": "sertraline", "drug_b": "warfarin",
        "rxcui_a": "36437", "rxcui_b": "11289",
        "severity": "Moderate",
        "severity_source": "curated",
        "mechanism": "Sertraline moderately inhibits CYP2C9 and may "
                     "also inhibit platelet serotonin uptake, reducing "
                     "platelet aggregation. Both effects increase bleeding "
                     "risk in anticoagulated patients.",
        "clinical_effects": "Modestly elevated INR; increased risk of "
                            "bleeding including gastrointestinal haemorrhage. "
                            "Risk of combined anticoagulant and antiplatelet "
                            "effect.",
        "management": "Monitor INR when sertraline is started, dose-adjusted, "
                      "or stopped. Counsel patient on bleeding signs. "
                      "Dose adjustment of warfarin may be required.",
        "source": "curated",
        "source_url": "https://pubmed.ncbi.nlm.nih.gov/?term=sertraline+warfarin+SSRI+INR+CYP2C9+bleeding&sort=relevance",
        "source_page": "NIH PubMed – SSRI + Warfarin Interaction Literature",
        "raw_text": (
            "Sertraline and warfarin interaction – Moderate. "
            "Sertraline (SSRI) moderately inhibits CYP2C9, the primary "
            "enzyme metabolising the pharmacologically active S-warfarin. "
            "Additionally, SSRIs reduce platelet serotonin uptake, impairing "
            "platelet aggregation. Together these effects elevate INR and "
            "increase bleeding risk. Monitor INR closely when sertraline is "
            "initiated or discontinued in patients on warfarin."
        ),
    },
    # ── Azithromycin + Warfarin ───────────────────────────────────────────────
    {
        "drug_a": "azithromycin", "drug_b": "warfarin",
        "rxcui_a": "18631", "rxcui_b": "11289",
        "severity": "Moderate",
        "severity_source": "curated",
        "mechanism": "Azithromycin may inhibit CYP3A4 to a modest degree "
                     "and disrupts gut flora that produce vitamin K. "
                     "The net effect is a modest increase in warfarin "
                     "anticoagulant activity.",
        "clinical_effects": "Elevated INR during azithromycin course, "
                            "with potential bleeding risk.",
        "management": "Monitor INR when azithromycin is prescribed in "
                      "patients on warfarin. A 5-day course may elevate "
                      "INR sufficiently to require dose adjustment. "
                      "Recheck INR after antibiotic course.",
        "source": "curated",
        "source_url": "https://pubmed.ncbi.nlm.nih.gov/?term=azithromycin+warfarin+INR+macrolide+antibiotic&sort=relevance",
        "source_page": "NIH PubMed – Azithromycin + Warfarin Interaction Literature",
        "raw_text": (
            "Azithromycin and warfarin interaction – Moderate. "
            "Azithromycin modestly inhibits CYP3A4 and reduces intestinal "
            "vitamin K production by altering gut flora. Both mechanisms "
            "increase warfarin effect. Multiple case reports document "
            "clinically significant INR elevation during a 5-day "
            "azithromycin course in stable anticoagulated patients. "
            "Check INR 2–3 days after starting azithromycin and after "
            "course completion."
        ),
    },
    # ── Carbamazepine + Oral Contraceptive ────────────────────────────────────
    {
        "drug_a": "carbamazepine", "drug_b": "ethinylestradiol",
        "rxcui_a": "2002", "rxcui_b": "3807",
        "severity": "Severe",
        "severity_source": "curated",
        "mechanism": "Carbamazepine is a potent inducer of CYP3A4, "
                     "the enzyme that metabolises ethinylestradiol (the "
                     "oestrogen component of oral contraceptives). This "
                     "dramatically increases oestrogen metabolism and "
                     "lowers plasma levels.",
        "clinical_effects": "Contraceptive failure with unintended pregnancy. "
                            "Carbamazepine reduces ethinylestradiol AUC by "
                            "approximately 50%.",
        "management": "Use a non-hormonal contraceptive method or a "
                      "progesterone-only injectable/implant (which is "
                      "less affected). If continuing combined oral "
                      "contraceptive, use highest oestrogen formulation "
                      "and add barrier method.",
        "source": "curated",
        "source_url": "https://pubmed.ncbi.nlm.nih.gov/?term=carbamazepine+oral+contraceptive+CYP3A4+enzyme+induction+failure&sort=relevance",
        "source_page": "NIH PubMed – Carbamazepine + Contraceptive Interaction Literature",
        "raw_text": (
            "Carbamazepine and ethinylestradiol (oral contraceptive) "
            "interaction – Severe. "
            "Carbamazepine strongly induces CYP3A4, dramatically accelerating "
            "metabolism of ethinylestradiol. Oestrogen AUC can fall by ~50%, "
            "leading to contraceptive failure and unintended pregnancy. "
            "Women of childbearing age on carbamazepine must use reliable "
            "non-hormonal or enzyme-resistant hormonal contraception. "
            "Condoms plus a non-interacting progesterone method is recommended."
        ),
    },
    # ── Insulin + Alcohol ─────────────────────────────────────────────────────
    {
        "drug_a": "insulin", "drug_b": "alcohol",
        "rxcui_a": "253182", "rxcui_b": "2670",
        "severity": "Moderate",
        "severity_source": "curated",
        "mechanism": "Alcohol inhibits hepatic gluconeogenesis, reducing "
                     "the liver's ability to compensate for insulin-induced "
                     "hypoglycaemia. It also masks sympathetic warning "
                     "symptoms of hypoglycaemia.",
        "clinical_effects": "Prolonged, severe hypoglycaemia that may "
                            "be difficult to recognise and treat. Risk "
                            "is highest several hours after alcohol "
                            "ingestion.",
        "management": "Advise patients on insulin to eat carbohydrates "
                      "when drinking alcohol and to avoid drinking on an "
                      "empty stomach. Monitor blood glucose closely. "
                      "Educate companions to recognise hypoglycaemia "
                      "if patient is impaired.",
        "source": "curated",
        "source_url": "https://pubmed.ncbi.nlm.nih.gov/?term=insulin+alcohol+hypoglycemia+gluconeogenesis+diabetes&sort=relevance",
        "source_page": "NIH PubMed – Insulin + Alcohol Hypoglycemia Literature",
        "raw_text": (
            "Insulin and alcohol interaction – Moderate. "
            "Ethanol suppresses hepatic gluconeogenesis, impairing the "
            "liver's ability to raise blood glucose in response to "
            "insulin-induced hypoglycaemia. Alcohol also blunts "
            "the adrenergic warning signs of hypoglycaemia. Patients "
            "treated with insulin who consume alcohol are at substantially "
            "higher risk of severe, prolonged hypoglycaemia. Educate "
            "patients to consume food when drinking and monitor glucose."
        ),
    },
    # ── Phenytoin + Warfarin ──────────────────────────────────────────────────
    {
        "drug_a": "phenytoin", "drug_b": "warfarin",
        "rxcui_a": "8143", "rxcui_b": "11289",
        "severity": "Severe",
        "severity_source": "curated",
        "mechanism": "Bidirectional interaction: phenytoin initially "
                     "inhibits warfarin metabolism (CYP2C9), raising INR; "
                     "chronic exposure induces CYP2C9, lowering INR. "
                     "Warfarin can also inhibit phenytoin metabolism, "
                     "raising phenytoin levels to toxic range.",
        "clinical_effects": "Unpredictable INR and phenytoin levels; "
                            "risk of both bleeding (if INR over-elevated) "
                            "and thrombosis (if INR under-elevated). "
                            "Phenytoin toxicity (nystagmus, ataxia, confusion) "
                            "is also possible.",
        "management": "Avoid combination if possible. If necessary, monitor "
                      "INR and phenytoin serum levels very closely after "
                      "any change in either drug. Frequent dose adjustments "
                      "may be required.",
        "source": "curated",
        "source_url": "https://pubmed.ncbi.nlm.nih.gov/?term=phenytoin+warfarin+CYP2C9+INR+bidirectional+interaction&sort=relevance",
        "source_page": "NIH PubMed – Phenytoin + Warfarin Bidirectional Interaction Literature",
        "raw_text": (
            "Phenytoin and warfarin interaction – Severe. "
            "This is a complex bidirectional drug interaction. Phenytoin "
            "initially inhibits CYP2C9 (warfarin's main metabolising enzyme), "
            "raising INR. With chronic use, phenytoin induces CYP2C9, "
            "eventually reducing warfarin efficacy. Warfarin can raise "
            "phenytoin levels to toxic concentrations by inhibiting "
            "phenytoin metabolism. Close monitoring of both INR and "
            "phenytoin serum levels is essential when this combination "
            "cannot be avoided."
        ),
    },
    # ── Metformin + Contrast Dye ──────────────────────────────────────────────
    {
        "drug_a": "iodinated contrast", "drug_b": "metformin",
        "rxcui_a": "203160", "rxcui_b": "6809",
        "severity": "Moderate",
        "severity_source": "curated",
        "mechanism": "Iodinated contrast agents can cause contrast-induced "
                     "nephropathy (acute tubular injury), reducing renal "
                     "function. Impaired kidneys cannot clear metformin, "
                     "causing drug accumulation and risk of lactic acidosis.",
        "clinical_effects": "Metformin-associated lactic acidosis in the "
                            "setting of contrast-induced nephropathy. "
                            "Though rare, lactic acidosis carries high "
                            "mortality (up to 50%).",
        "management": "Hold metformin 24–48 hours before elective contrast "
                      "procedures. Restart only after confirming stable "
                      "renal function (serum creatinine checked 48 h post-"
                      "procedure). Ensure adequate hydration.",
        "source": "curated",
        "source_url": "https://pubmed.ncbi.nlm.nih.gov/?term=metformin+contrast+dye+lactic+acidosis+nephropathy&sort=relevance",
        "source_page": "NIH PubMed – Metformin + Contrast Media Lactic Acidosis Literature",
        "raw_text": (
            "Iodinated contrast and metformin interaction – Moderate. "
            "Iodinated contrast agents can induce nephrotoxicity (contrast-"
            "induced nephropathy). Reduced renal function impairs metformin "
            "clearance, raising plasma levels and increasing the risk of "
            "metformin-associated lactic acidosis — a rare but potentially "
            "fatal complication. Current guidelines recommend holding "
            "metformin 48 hours before and after contrast procedures and "
            "confirming renal function before resuming."
        ),
    },
    # ── Tramadol + MAOIs ──────────────────────────────────────────────────────
    {
        "drug_a": "phenelzine", "drug_b": "tramadol",
        "rxcui_a": "8122", "rxcui_b": "41493",
        "severity": "Severe",
        "severity_source": "curated",
        "mechanism": "Phenelzine (MAOI) inhibits monoamine oxidase, the "
                     "enzyme that degrades serotonin. Tramadol inhibits "
                     "serotonin reuptake. Combined serotonin accumulation "
                     "produces life-threatening serotonin syndrome. "
                     "MAOIs also inhibit tramadol's first-pass metabolism.",
        "clinical_effects": "Severe serotonin syndrome: hyperthermia, "
                            "clonus, rhabdomyolysis, multi-organ failure, "
                            "death. Also risk of hypertensive crisis.",
        "management": "Absolutely contraindicated. Allow at least 14 days "
                      "after stopping an MAOI before initiating tramadol. "
                      "Allow at least 5 days after stopping tramadol before "
                      "initiating an MAOI.",
        "source": "curated",
        "source_url": "https://pubmed.ncbi.nlm.nih.gov/?term=phenelzine+tramadol+MAOI+serotonin+syndrome+contraindicated&sort=relevance",
        "source_page": "NIH PubMed – MAOI + Tramadol Serotonin Syndrome Literature",
        "raw_text": (
            "Phenelzine (MAOI) and tramadol interaction – Severe. "
            "This combination is absolutely contraindicated. MAOIs prevent "
            "serotonin degradation; tramadol inhibits serotonin reuptake. "
            "The synergistic serotonergic effect can produce life-threatening "
            "serotonin syndrome characterised by hyperthermia, clonus, "
            "rhabdomyolysis, and potentially death. A washout period of "
            "at least 14 days after stopping MAOIs is required before "
            "tramadol can be safely initiated."
        ),
    },
]


# ── HTTP helpers ──────────────────────────────────────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
async def _get_json(
    session: aiohttp.ClientSession,
    url: str,
    params: dict | None = None,
) -> dict | None:
    """GET request with automatic retry on transient failures."""
    try:
        async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status == 200:
                return await resp.json(content_type=None)
            logger.warning("HTTP %s for %s", resp.status, url)
            return None
    except Exception as exc:
        logger.error("Request failed for %s: %s", url, exc)
        raise


# ── RxNorm helpers ────────────────────────────────────────────────────────────

async def get_rxcui(session: aiohttp.ClientSession, drug_name: str) -> str | None:
    """Resolve a drug name to its RxNorm Concept Unique Identifier (RxCUI)."""
    data = await _get_json(
        session,
        f"{RXNORM_BASE}/rxcui.json",
        params={"name": drug_name, "search": "1"},
    )
    if not data:
        return None
    try:
        return data["idGroup"]["rxnormId"][0]
    except (KeyError, IndexError, TypeError):
        logger.warning("No RxCUI found for drug: %s", drug_name)
        return None


async def get_rxnorm_interactions(
    session: aiohttp.ClientSession, rxcui: str
) -> list[dict]:
    """
    Fetch all known drug interactions for a given RxCUI from RxNav.
    Returns a list of raw interaction dicts using the canonical schema.
    """
    data = await _get_json(
        session,
        f"{RXNORM_BASE}/interaction/interaction.json",
        params={"rxcui": rxcui},
    )
    if not data:
        return []

    interactions = []
    try:
        for group in data.get("interactionTypeGroup", []):
            for itype in group.get("interactionType", []):
                description = itype.get("comment", "")
                for pair in itype.get("interactionPair", []):
                    drugs = pair.get("interactionConcept", [])
                    if len(drugs) < 2:
                        continue

                    drug_a_name = drugs[0]["minConceptItem"]["name"].lower()
                    drug_b_name = drugs[1]["minConceptItem"]["name"].lower()
                    rxcui_a = drugs[0]["minConceptItem"]["rxcui"]
                    rxcui_b = drugs[1]["minConceptItem"]["rxcui"]

                    # Map RxNorm severity code
                    severity_code = str(pair.get("severity", "N/A"))
                    severity = RXNORM_SEVERITY_MAP.get(severity_code, "Unknown")

                    raw_text = pair.get("description", description)

                    interactions.append({
                        "drug_a": drug_a_name,
                        "drug_b": drug_b_name,
                        "rxcui_a": rxcui_a,
                        "rxcui_b": rxcui_b,
                        "severity": severity,
                        "severity_source": "rxnorm",
                        "mechanism": description,
                        "clinical_effects": raw_text,
                        "management": "",   # enriched in preprocessing
                        "source": "rxnorm",
                        "raw_text": raw_text,
                    })
    except Exception as exc:
        logger.error("Error parsing RxNorm interactions for RXCUI %s: %s", rxcui, exc)

    return interactions


# ── OpenFDA helpers ───────────────────────────────────────────────────────────

async def get_fda_narrative(
    session: aiohttp.ClientSession, drug_name: str
) -> str:
    """
    Retrieve the drug interactions section from an FDA label for
    richer narrative text used in embedding.
    """
    data = await _get_json(
        session,
        OPENFDA_BASE,
        params={
            "search": f'openfda.generic_name:"{drug_name}"',
            "limit": "1",
        },
    )
    if not data:
        return ""
    try:
        results = data.get("results", [])
        if results:
            sections = results[0].get("drug_interactions", [])
            return " ".join(sections)[:2000]  # cap to avoid token bloat
    except Exception:
        pass
    return ""


# ── Main fetch orchestration ──────────────────────────────────────────────────

async def fetch_all_interactions(
    drugs: list[str], include_seed: bool = True
) -> list[dict]:
    """
    Fetch drug-drug interaction data for the given drug list.

    Strategy:
      1. Start with curated SEED_INTERACTIONS (always reliable).
      2. Resolve each drug to its RxCUI.
      3. For each drug, fetch all known DDIs from RxNorm.
      4. Enrich with OpenFDA narrative text.
      5. Deduplicate by drug pair.
    """
    records: list[dict] = []
    seen_pairs: set[frozenset] = set()

    # Step 1 – seed data (filtered to requested drugs if list is provided)
    if include_seed:
        drug_set = {d.lower() for d in drugs} if drugs else None
        for record in SEED_INTERACTIONS:
            pair = frozenset([record["drug_a"], record["drug_b"]])
            # Include record if either drug is in requested set or no filter
            if drug_set is None or pair & drug_set:
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    records.append(record)

    # Step 2-4 – live API fetching
    semaphore = asyncio.Semaphore(CONCURRENCY)

    async with aiohttp.ClientSession(
        headers={"User-Agent": "DrugInteractionAI/1.0 (research; non-commercial)"}
    ) as session:

        async def fetch_drug(drug_name: str) -> list[dict]:
            async with semaphore:
                rxcui = await get_rxcui(session, drug_name)
                if not rxcui:
                    logger.warning("Skipping %s – no RxCUI resolved", drug_name)
                    return []
                logger.info("Fetching interactions for %s (RXCUI %s)", drug_name, rxcui)
                return await get_rxnorm_interactions(session, rxcui)

        tasks = [fetch_drug(d) for d in drugs]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                logger.error("Fetch task failed: %s", result)
                continue
            for record in result:
                pair = frozenset([record["drug_a"], record["drug_b"]])
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    records.append(record)

    logger.info("Fetched %d unique drug-pair interaction records", len(records))
    return records


def save_records(records: list[dict], output_path: str) -> None:
    """Persist records as newline-delimited JSON (JSONL)."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    logger.info("Saved %d records to %s", len(records), output_path)


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch drug interaction data")
    parser.add_argument(
        "--drugs",
        nargs="+",
        default=[],
        help="Drug names to fetch interactions for. "
             "Empty = include all seed data.",
    )
    parser.add_argument(
        "--out",
        default="data/raw/interactions_raw.jsonl",
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--no-seed",
        action="store_true",
        help="Exclude curated seed interactions",
    )
    args = parser.parse_args()

    records = asyncio.run(
        fetch_all_interactions(
            drugs=args.drugs,
            include_seed=not args.no_seed,
        )
    )
    save_records(records, args.out)
    print(f"Done – {len(records)} interaction records written to {args.out}")


if __name__ == "__main__":
    main()
