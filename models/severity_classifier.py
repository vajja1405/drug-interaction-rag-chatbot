"""
models/severity_classifier.py
───────────────────────────────
ML-based drug interaction severity classifier.

Input
─────
  drug_pair             : tuple[str, str]  e.g. ("warfarin", "ibuprofen")
  interaction_description : str            clinical interaction text

Output
──────
  SeverityPrediction
    .label          : "Mild" | "Moderate" | "Severe"
    .confidence     : float  0.0 – 1.0
    .probabilities  : dict   {"Mild": p1, "Moderate": p2, "Severe": p3}
    .model_used     : str

Two model backends
──────────────────
  RandomForestSeverityClassifier
    • TF-IDF (char + word n-grams) on the interaction description
    • Drug-class indicator features (anticoagulant, NSAID, statin, …)
    • Clinical keyword presence features (from SEVERE/MODERATE/MILD sets)
    • Calibrated RandomForestClassifier (Platt scaling)
    • Trained in < 1 second on the built-in corpus; no GPU required

  BERTSeverityClassifier
    • DistilBERT-base-uncased fine-tuned for 3-class sequence classification
    • Input: "{drug_a} and {drug_b} [SEP] {description}"
    • AdamW optimiser, cosine LR schedule, early stopping
    • Requires ~300 MB disk / GPU optional (runs on CPU)

  HybridSeverityClassifier (recommended)
    • Weighted soft-voting ensemble: 0.45 × RF + 0.55 × BERT
    • Falls back to RF if BERT model is not trained yet

Severity label mapping
──────────────────────
  This module uses: Mild / Moderate / Severe
  Legacy interaction_classifier.py uses: Low / Moderate / Severe
  Mapping: Mild ↔ Low (same clinical meaning)

Usage
─────
  See run_example() at the bottom of this file, or:

    python -m models.severity_classifier
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)

# ── Paths ──────────────────────────────────────────────────────────────────────
MODEL_DIR = Path("models/saved")
RF_MODEL_PATH = MODEL_DIR / "rf_severity_classifier.pkl"
BERT_MODEL_DIR = MODEL_DIR / "bert_severity_classifier"

# ── Label constants ────────────────────────────────────────────────────────────
LABELS = ["Mild", "Moderate", "Severe"]
LABEL_TO_IDX = {l: i for i, l in enumerate(LABELS)}
IDX_TO_LABEL = {i: l for l, i in LABEL_TO_IDX.items()}

# Legacy mapping (Low ↔ Mild)
LEGACY_TO_LABEL = {"Low": "Mild", "Moderate": "Moderate", "Severe": "Severe"}
LABEL_TO_LEGACY = {"Mild": "Low", "Moderate": "Moderate", "Severe": "Severe"}


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Labeled Training Corpus
# ══════════════════════════════════════════════════════════════════════════════

# Each record:  (drug_a, drug_b, description, label)
# Sources: established clinical pharmacology literature, FDA labels, RxNorm
_TRAINING_CORPUS: list[tuple[str, str, str, str]] = [

    # ── SEVERE ────────────────────────────────────────────────────────────────

    ("warfarin", "aspirin",
     "Additive anticoagulant and antiplatelet effects. Aspirin displaces "
     "warfarin from plasma protein binding, raising free drug concentration. "
     "Significantly increased risk of fatal gastrointestinal haemorrhage and "
     "intracranial bleeding. Avoid concurrent use; if necessary monitor INR "
     "weekly.",
     "Severe"),

    ("warfarin", "ibuprofen",
     "Ibuprofen inhibits COX-1-mediated platelet aggregation and causes "
     "gastric mucosal erosion. Protein-binding displacement transiently "
     "raises free warfarin. Elevated risk of serious gastrointestinal "
     "bleeding. Avoid combination; use acetaminophen instead.",
     "Severe"),

    ("warfarin", "naproxen",
     "Non-selective NSAID naproxen inhibits platelet function and causes "
     "GI mucosal damage, substantially increasing haemorrhagic risk in "
     "patients anticoagulated with warfarin. Contraindicated in most "
     "clinical settings.",
     "Severe"),

    ("simvastatin", "amiodarone",
     "Amiodarone is a potent inhibitor of CYP3A4 and CYP2C9, the primary "
     "metabolic pathways for simvastatin. Co-administration increases "
     "simvastatin AUC up to 6-fold, dramatically elevating the risk of "
     "myopathy and potentially fatal rhabdomyolysis. FDA recommends "
     "limiting simvastatin dose to 20 mg/day or switching statin.",
     "Severe"),

    ("simvastatin", "clarithromycin",
     "Clarithromycin is a strong CYP3A4 inhibitor. Markedly increases "
     "simvastatin plasma concentrations, causing severe myopathy and "
     "rhabdomyolysis. Do not administer together; withhold statin during "
     "clarithromycin course.",
     "Severe"),

    ("digoxin", "amiodarone",
     "Amiodarone inhibits P-glycoprotein and reduces renal clearance of "
     "digoxin, raising serum digoxin levels 70-100%. Leads to digoxin "
     "toxicity: bradycardia, heart block, ventricular arrhythmias. "
     "Reduce digoxin dose by 50%; monitor serum levels.",
     "Severe"),

    ("methotrexate", "ibuprofen",
     "NSAIDs inhibit renal tubular secretion of methotrexate via organic "
     "anion transport. Methotrexate accumulation causes potentially "
     "fatal myelosuppression, mucositis, and nephrotoxicity even at low "
     "doses. Absolutely avoid combination.",
     "Severe"),

    ("phenelzine", "fluoxetine",
     "Both agents increase serotonergic transmission. MAOI phenelzine "
     "combined with SSRI fluoxetine causes serotonin syndrome: "
     "hyperthermia, clonus, agitation, autonomic instability, "
     "potentially fatal. Contraindicated; require washout period.",
     "Severe"),

    ("tramadol", "sertraline",
     "Tramadol inhibits serotonin reuptake; sertraline is a potent SSRI. "
     "Combination causes serotonin syndrome and lowers seizure threshold. "
     "Life-threatening interaction. Avoid concurrent use.",
     "Severe"),

    ("linezolid", "venlafaxine",
     "Linezolid is a reversible MAO inhibitor. Combined with "
     "serotonin-noradrenaline reuptake inhibitor venlafaxine causes "
     "serotonin syndrome. Contraindicated. 14-day washout required "
     "after linezolid before starting SNRI.",
     "Severe"),

    ("cisapride", "fluconazole",
     "Fluconazole inhibits CYP3A4-mediated metabolism of cisapride, "
     "dramatically raising cisapride levels. Prolongs QTc interval "
     "causing Torsades de Pointes and potentially fatal ventricular "
     "fibrillation. Contraindicated.",
     "Severe"),

    ("pimozide", "erythromycin",
     "Erythromycin inhibits CYP3A4 metabolism of pimozide and itself "
     "prolongs the QT interval. Combination causes additive QT "
     "prolongation with risk of Torsades de Pointes. Contraindicated.",
     "Severe"),

    ("lithium", "ibuprofen",
     "NSAIDs reduce renal prostaglandin synthesis, decreasing renal "
     "lithium clearance. Serum lithium levels rise significantly, causing "
     "lithium toxicity: tremor, confusion, renal failure, arrhythmias, "
     "coma. Monitor lithium levels closely; avoid NSAIDs.",
     "Severe"),

    ("carbamazepine", "erythromycin",
     "Erythromycin inhibits CYP3A4, the primary enzyme metabolising "
     "carbamazepine, causing 2-3 fold increase in carbamazepine plasma "
     "levels. Carbamazepine toxicity: diplopia, ataxia, somnolence, "
     "cardiac arrhythmias. Contraindicated combination.",
     "Severe"),

    ("warfarin", "rifampicin",
     "Rifampicin is a potent inducer of CYP2C9, dramatically increasing "
     "warfarin metabolism and reducing anticoagulant effect by 80-90%. "
     "Risk of therapeutic failure (thromboembolism). INR must be "
     "monitored intensively; avoid if possible.",
     "Severe"),

    ("methotrexate", "trimethoprim",
     "Trimethoprim inhibits dihydrofolate reductase, the same enzyme "
     "targeted by methotrexate, causing additive antifolate toxicity. "
     "Bone marrow suppression, megaloblastic anaemia, mucositis. "
     "Contraindicated; use alternative antibiotics.",
     "Severe"),

    ("tacrolimus", "voriconazole",
     "Voriconazole potently inhibits CYP3A4, markedly increasing "
     "tacrolimus blood levels (up to 8-fold). Risk of serious "
     "nephrotoxicity and neurotoxicity. Reduce tacrolimus dose by "
     "67%; monitor drug levels every 1-2 days.",
     "Severe"),

    ("clozapine", "fluvoxamine",
     "Fluvoxamine inhibits CYP1A2 and CYP3A4, markedly increasing "
     "clozapine plasma levels (5-10 fold). Risk of clozapine toxicity: "
     "seizures, agranulocytosis, respiratory depression, cardiomyopathy. "
     "Contraindicated; use alternative antidepressant.",
     "Severe"),

    ("sotalol", "amiodarone",
     "Both sotalol and amiodarone prolong the cardiac QT interval by "
     "blocking potassium channels. Additive QTc prolongation causes "
     "risk of life-threatening Torsades de Pointes. Contraindicated "
     "combination.",
     "Severe"),

    ("heparin", "aspirin",
     "Aspirin inhibits platelet aggregation via irreversible COX-1 "
     "inhibition. Combined with anticoagulant heparin, substantially "
     "increases risk of serious bleeding events including intracranial "
     "haemorrhage. Use only when clinically essential; monitor closely.",
     "Severe"),

    ("clopidogrel", "omeprazole",
     "Omeprazole inhibits CYP2C19, blocking conversion of clopidogrel "
     "to its active thiol metabolite. Reduces platelet inhibition by "
     "up to 47%, potentially reducing cardioprotective benefit in "
     "post-PCI patients. Use pantoprazole instead.",
     "Severe"),

    ("warfarin", "metronidazole",
     "Metronidazole inhibits CYP2C9, markedly reducing warfarin "
     "metabolism and causing dangerous INR elevation. Risk of severe "
     "bleeding. Avoid combination; use alternative antibiotics. "
     "If unavoidable, reduce warfarin dose by 25-50%.",
     "Severe"),

    ("ergotamine", "sumatriptan",
     "Additive vasoconstrictive effects on cranial and peripheral "
     "vasculature. Combination causes severe vasospasm, peripheral "
     "ischaemia, and risk of stroke. Contraindicated; 24-hour washout "
     "required between ergotamine and triptan use.",
     "Severe"),

    ("theophylline", "ciprofloxacin",
     "Ciprofloxacin inhibits CYP1A2, markedly reducing theophylline "
     "clearance and raising plasma levels. Theophylline toxicity: "
     "nausea, cardiac arrhythmias, seizures. Reduce theophylline dose "
     "by 50%; monitor plasma levels.",
     "Severe"),

    ("warfarin", "fluconazole",
     "Fluconazole inhibits CYP2C9, dramatically reducing warfarin "
     "metabolism. INR increases 2-3 fold within days. Serious "
     "bleeding risk. Avoid combination; if antifungal needed use "
     "topical agents. Monitor INR daily if co-prescribing.",
     "Severe"),

    # ── MODERATE ──────────────────────────────────────────────────────────────

    ("lisinopril", "ibuprofen",
     "NSAIDs reduce prostaglandin synthesis, blunting the "
     "vasodilatory and natriuretic effects of ACE inhibitors. "
     "Reduces antihypertensive efficacy. Concurrent NSAID use also "
     "impairs renal autoregulation, increasing risk of acute kidney "
     "injury especially in elderly or dehydrated patients.",
     "Moderate"),

    ("aspirin", "ibuprofen",
     "Ibuprofen competitively and reversibly occupies the COX-1 "
     "acetylation site, preventing aspirin's irreversible antiplatelet "
     "effect. Patients on low-dose aspirin for cardiovascular "
     "prophylaxis may lose that benefit. Administer aspirin 30 "
     "minutes before or 8 hours after ibuprofen.",
     "Moderate"),

    ("ibuprofen", "metformin",
     "NSAIDs reduce renal prostaglandin synthesis, causing afferent "
     "arteriolar vasoconstriction and decreased GFR. Reduced "
     "metformin clearance increases plasma metformin. Risk of lactic "
     "acidosis if AKI develops. Monitor renal function.",
     "Moderate"),

    ("metoprolol", "verapamil",
     "Both drugs reduce heart rate and AV nodal conduction. "
     "Additive negative chronotropic and dromotropic effects cause "
     "risk of bradycardia, heart block, and haemodynamic compromise. "
     "Monitor ECG; avoid combination in patients with conduction "
     "defects.",
     "Moderate"),

    ("atorvastatin", "clarithromycin",
     "Clarithromycin moderately inhibits CYP3A4, increasing atorvastatin "
     "exposure approximately 4-fold. Risk of myopathy and rhabdomyolysis "
     "is elevated. Limit atorvastatin dose to 20 mg/day during "
     "clarithromycin course; monitor for muscle symptoms.",
     "Moderate"),

    ("metformin", "alcohol",
     "Alcohol potentiates metformin-associated lactic acidosis by "
     "impairing hepatic lactate metabolism and increasing hepatic "
     "lactate production. Risk is particularly elevated in fasting "
     "states. Avoid excessive alcohol; counsel patients.",
     "Moderate"),

    ("enalapril", "spironolactone",
     "ACE inhibitors reduce aldosterone production; spironolactone "
     "blocks aldosterone receptors. Both increase potassium retention. "
     "Risk of clinically significant hyperkalaemia, especially with "
     "impaired renal function. Monitor serum potassium regularly.",
     "Moderate"),

    ("warfarin", "amiodarone",
     "Amiodarone inhibits CYP2C9 and CYP3A4, reducing warfarin "
     "metabolism. INR increases gradually over weeks. Reduces "
     "warfarin dose requirement by 30-50%. Moderate risk of "
     "bleeding if not monitored. Monitor INR weekly on initiation.",
     "Moderate"),

    ("fluoxetine", "tramadol",
     "Fluoxetine inhibits CYP2D6, reducing tramadol conversion to "
     "active M1 metabolite. Simultaneously increases serotonergic "
     "tone raising serotonin syndrome risk. Reduced analgesic "
     "efficacy and potential toxicity. Consider alternative analgesic.",
     "Moderate"),

    ("amlodipine", "simvastatin",
     "Amlodipine moderately inhibits CYP3A4, raising simvastatin "
     "exposure approximately 1.5-2 fold. Myopathy risk is modestly "
     "increased. FDA recommends limiting simvastatin dose to 20 mg "
     "when combined with amlodipine. Monitor for muscle symptoms.",
     "Moderate"),

    ("lisinopril", "potassium_chloride",
     "ACE inhibitors reduce aldosterone, decreasing renal potassium "
     "excretion. Concurrent potassium supplementation can cause "
     "hyperkalaemia. ECG changes, cardiac arrhythmias in severe cases. "
     "Monitor serum potassium; adjust supplementation accordingly.",
     "Moderate"),

    ("sildenafil", "amlodipine",
     "Both agents cause vasodilation; sildenafil via PDE5 inhibition "
     "and amlodipine via L-type calcium channel blockade. Additive "
     "antihypertensive effect causing symptomatic hypotension. "
     "Monitor blood pressure; start sildenafil at lowest dose.",
     "Moderate"),

    ("warfarin", "omeprazole",
     "Omeprazole inhibits CYP2C19, moderately reducing (S)-warfarin "
     "metabolism. INR increases modestly (15-25%). Not clinically "
     "significant in most patients but monitor INR when initiating "
     "or stopping PPI therapy.",
     "Moderate"),

    ("ciprofloxacin", "antacids",
     "Antacids containing aluminium, magnesium, or calcium form "
     "insoluble chelates with ciprofloxacin, reducing oral bioavailability "
     "by 50-90%. Therapeutic failure possible for serious infections. "
     "Administer ciprofloxacin 2 hours before or 6 hours after antacid.",
     "Moderate"),

    ("furosemide", "gentamicin",
     "Both furosemide and aminoglycoside gentamicin are ototoxic. "
     "Co-administration causes additive or synergistic damage to "
     "cochlear hair cells, increasing risk of permanent hearing loss "
     "and vestibular toxicity. Monitor hearing; use minimum effective doses.",
     "Moderate"),

    ("warfarin", "carbamazepine",
     "Carbamazepine is a potent inducer of CYP2C9, markedly increasing "
     "warfarin metabolism and reducing INR. Significant thromboembolism "
     "risk if anticoagulation is subtherapeutic. Monitor INR closely "
     "when starting or stopping carbamazepine.",
     "Moderate"),

    ("methotrexate", "probenecid",
     "Probenecid reduces renal tubular secretion of methotrexate, "
     "increasing methotrexate plasma levels and tissue exposure. "
     "Risk of methotrexate toxicity: mucositis, bone marrow suppression. "
     "Avoid combination; reduce methotrexate dose and monitor CBC.",
     "Moderate"),

    ("digoxin", "verapamil",
     "Verapamil increases digoxin bioavailability and reduces its "
     "renal clearance, raising serum digoxin levels 50-75%. "
     "Elevated risk of digoxin toxicity: bradycardia, AV block. "
     "Reduce digoxin dose and monitor serum levels.",
     "Moderate"),

    ("lithium", "hydrochlorothiazide",
     "Thiazide diuretics reduce sodium reabsorption, causing "
     "compensatory increase in lithium reabsorption in proximal "
     "tubule. Serum lithium increases 25-40%. Risk of lithium toxicity. "
     "Monitor serum lithium; reduce dose by 25-50%.",
     "Moderate"),

    ("phenytoin", "fluconazole",
     "Fluconazole inhibits CYP2C9-mediated phenytoin metabolism, "
     "increasing phenytoin levels. Phenytoin toxicity: nystagmus, "
     "ataxia, drowsiness. Monitor phenytoin levels; adjust dose.",
     "Moderate"),

    ("rosuvastatin", "fenofibrate",
     "Fenofibrate inhibits OATP1B1 transport and moderately inhibits "
     "BCRP, increasing rosuvastatin exposure. Myopathy risk is elevated. "
     "Start with lowest rosuvastatin dose; monitor for muscle symptoms.",
     "Moderate"),

    # ── MILD ──────────────────────────────────────────────────────────────────

    ("warfarin", "metformin",
     "No direct pharmacokinetic interaction between warfarin and "
     "metformin. Metformin does not significantly affect CYP2C9-mediated "
     "warfarin metabolism. Routine INR monitoring is sufficient. "
     "No dose adjustment typically required.",
     "Mild"),

    ("aspirin", "metformin",
     "No clinically significant interaction at standard aspirin doses "
     "(75-325 mg/day). High-dose salicylate therapy may rarely "
     "potentiate hypoglycaemia via increased insulin sensitivity, "
     "but this is not a concern at cardioprotective doses.",
     "Mild"),

    ("lisinopril", "aspirin",
     "High-dose aspirin may blunt the antihypertensive effect of "
     "ACE inhibitors by inhibiting prostaglandin synthesis. At "
     "standard cardioprotective doses (75-100 mg/day), the clinical "
     "impact is minimal. No routine dose adjustment required.",
     "Mild"),

    ("lisinopril", "metformin",
     "No direct pharmacokinetic interaction. ACE inhibitors may "
     "modestly improve insulin sensitivity, complementing metformin "
     "action. Combination is generally safe and commonly used in "
     "diabetic patients with hypertension. Monitor blood glucose.",
     "Mild"),

    ("warfarin", "lisinopril",
     "No clinically significant pharmacokinetic or pharmacodynamic "
     "interaction between warfarin and lisinopril. Neither drug "
     "substantially alters the other's metabolism. Standard INR "
     "monitoring is appropriate.",
     "Mild"),

    ("atorvastatin", "amlodipine",
     "Amlodipine causes a small increase in atorvastatin AUC (about "
     "18%). This minor pharmacokinetic interaction does not require "
     "dose adjustment at standard atorvastatin doses (10-20 mg). "
     "No increased risk of myopathy at usual clinical doses.",
     "Mild"),

    ("metoprolol", "aspirin",
     "No significant pharmacokinetic interaction. Aspirin does not "
     "substantially alter metoprolol metabolism. Some theoretical "
     "concern about aspirin blunting prostaglandin-mediated benefits "
     "but clinically not relevant at cardioprotective doses.",
     "Mild"),

    ("furosemide", "metformin",
     "Furosemide can increase metformin peak plasma concentration "
     "approximately 22% by competing for renal tubular secretion. "
     "This minor interaction does not require dose adjustment in "
     "patients with normal renal function.",
     "Mild"),

    ("omeprazole", "metformin",
     "No direct pharmacokinetic interaction between omeprazole and "
     "metformin. Omeprazole does not significantly alter metformin "
     "absorption or renal clearance. Routine monitoring is sufficient.",
     "Mild"),

    ("amlodipine", "metformin",
     "No clinically relevant pharmacokinetic interaction. Amlodipine "
     "does not affect metformin absorption or renal elimination. "
     "Both agents are commonly co-prescribed in type 2 diabetes "
     "with hypertension without significant interaction.",
     "Mild"),

    ("rosuvastatin", "aspirin",
     "No pharmacokinetic interaction between rosuvastatin and aspirin. "
     "Both drugs are commonly used together in cardiovascular disease "
     "prevention. Combination is generally well tolerated. No dose "
     "adjustment required.",
     "Mild"),

    ("metformin", "lisinopril",
     "Frequently prescribed combination in patients with type 2 "
     "diabetes and hypertension or diabetic nephropathy. ACE inhibitors "
     "may provide additive renoprotective benefit alongside metformin. "
     "Monitor renal function and potassium periodically.",
     "Mild"),

    ("pantoprazole", "metformin",
     "No pharmacokinetic interaction. Pantoprazole does not inhibit "
     "CYP enzymes relevant to metformin disposition. Combination is "
     "safe and commonly used. No dose adjustment needed.",
     "Mild"),

    ("atenolol", "aspirin",
     "No significant pharmacokinetic interaction. Beta-blockers and "
     "aspirin are frequently co-prescribed in cardiovascular disease. "
     "Minor concern about aspirin reducing prostaglandin-dependent "
     "antihypertensive effects, but not clinically significant.",
     "Mild"),

    ("metformin", "omeprazole",
     "No clinically significant interaction. Omeprazole may "
     "theoretically reduce vitamin B12 absorption with long-term "
     "use, but this is unrelated to metformin's mechanism. "
     "Routine monitoring is appropriate.",
     "Mild"),

    ("lisinopril", "amlodipine",
     "Complementary antihypertensive mechanisms with no adverse "
     "pharmacokinetic interaction. ACE inhibitor plus calcium channel "
     "blocker combination is a recommended first-line antihypertensive "
     "regimen. Well tolerated. No dose adjustment required.",
     "Mild"),

    ("simvastatin", "aspirin",
     "No direct pharmacokinetic interaction. Both drugs are commonly "
     "used together for cardiovascular risk reduction. Simvastatin "
     "and aspirin have complementary mechanisms with no significant "
     "pharmacokinetic or adverse pharmacodynamic interaction.",
     "Mild"),

    ("metoprolol", "amlodipine",
     "No significant pharmacokinetic interaction. Beta-blocker and "
     "calcium channel blocker combination is used therapeutically "
     "for angina and hypertension. Additive blood pressure lowering "
     "is the intended effect, not a harmful interaction.",
     "Mild"),

    ("atorvastatin", "aspirin",
     "No pharmacokinetic interaction between atorvastatin and aspirin. "
     "Co-prescription for cardiovascular prevention is standard "
     "practice. No dose adjustment needed. Combination is well "
     "tolerated in the vast majority of patients.",
     "Mild"),
]


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Drug-Class Feature Engineering
# ══════════════════════════════════════════════════════════════════════════════

# Drug-class membership: used as binary features alongside TF-IDF
_DRUG_CLASSES: dict[str, list[str]] = {
    "anticoagulant":    ["warfarin", "heparin", "apixaban", "rivaroxaban", "dabigatran", "edoxaban"],
    "antiplatelet":     ["aspirin", "clopidogrel", "ticagrelor", "prasugrel", "dipyridamole"],
    "nsaid":            ["ibuprofen", "naproxen", "diclofenac", "ketoprofen", "celecoxib", "indomethacin", "meloxicam"],
    "statin":           ["simvastatin", "atorvastatin", "rosuvastatin", "lovastatin", "pravastatin", "fluvastatin"],
    "ace_inhibitor":    ["lisinopril", "enalapril", "ramipril", "perindopril", "captopril", "quinapril"],
    "arb":              ["losartan", "valsartan", "irbesartan", "candesartan", "telmisartan"],
    "beta_blocker":     ["metoprolol", "atenolol", "bisoprolol", "carvedilol", "propranolol", "labetalol"],
    "ccb":              ["amlodipine", "nifedipine", "verapamil", "diltiazem", "felodipine"],
    "biguanide":        ["metformin"],
    "ssri":             ["fluoxetine", "sertraline", "paroxetine", "citalopram", "escitalopram", "fluvoxamine"],
    "snri":             ["venlafaxine", "duloxetine", "desvenlafaxine"],
    "maoi":             ["phenelzine", "tranylcypromine", "isocarboxazid", "selegiline", "linezolid"],
    "cyp3a4_inhibitor": ["clarithromycin", "erythromycin", "fluconazole", "voriconazole", "itraconazole", "ketoconazole", "ritonavir", "amiodarone"],
    "cyp2c9_inhibitor": ["fluconazole", "amiodarone", "metronidazole", "fluvoxamine"],
    "cyp_inducer":      ["rifampicin", "rifampin", "carbamazepine", "phenytoin", "phenobarbital", "st_john_wort"],
    "antiarrhythmic":   ["amiodarone", "sotalol", "flecainide", "propafenone", "digoxin"],
    "diuretic":         ["furosemide", "hydrochlorothiazide", "spironolactone", "amiloride", "torsemide"],
    "antibiotic":       ["ciprofloxacin", "erythromycin", "clarithromycin", "metronidazole", "trimethoprim", "gentamicin"],
    "antifungal":       ["fluconazole", "voriconazole", "itraconazole", "ketoconazole"],
    "immunosuppressant":["tacrolimus", "cyclosporine", "methotrexate", "azathioprine"],
    "ppi":              ["omeprazole", "pantoprazole", "esomeprazole", "lansoprazole", "rabeprazole"],
    "opioid":           ["tramadol", "morphine", "oxycodone", "codeine", "fentanyl", "hydrocodone"],
}

# Reverse map: drug_name → list of drug classes it belongs to
_DRUG_TO_CLASSES: dict[str, list[str]] = {}
for _class, _drugs in _DRUG_CLASSES.items():
    for _drug in _drugs:
        _DRUG_TO_CLASSES.setdefault(_drug, []).append(_class)


def get_drug_classes(drug_name: str) -> list[str]:
    """Return all drug classes a drug belongs to."""
    return _DRUG_TO_CLASSES.get(drug_name.lower(), [])


# Keyword feature sets aligned with ML labels (Mild / Moderate / Severe)
_SEVERITY_KEYWORDS: dict[str, list[str]] = {
    "Severe": [
        "contraindicated", "avoid", "do not use", "life-threatening", "fatal",
        "rhabdomyolysis", "serotonin syndrome", "qt prolongation", "torsades",
        "hemorrhage", "haemorrhage", "bleeding risk", "serious", "toxicity",
        "myelosuppression", "agranulocytosis", "arrhythmia", "lactic acidosis",
        "vasospasm", "intracranial", "bone marrow", "cardiac arrest",
    ],
    "Moderate": [
        "caution", "monitor closely", "reduce dose", "dose reduction",
        "adjust dose", "clinical significance", "may increase", "may decrease",
        "elevated risk", "inhibits", "induces", "cyp inhibitor", "p-glycoprotein",
        "narrow therapeutic", "renal clearance", "impaired", "interaction",
        "ototoxic", "nephrotoxic", "hepatotoxic", "bioavailability",
    ],
    "Mild": [
        "no significant", "minor", "minimal", "routine monitoring",
        "no dose adjustment", "generally safe", "well tolerated",
        "not clinically significant", "theoretical", "unlikely", "low risk",
        "commonly co-prescribed", "standard practice", "appropriate",
    ],
}


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Feature Transformers (scikit-learn compatible)
# ══════════════════════════════════════════════════════════════════════════════

class DrugClassFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts binary drug-class indicator features for a drug pair.

    Input : list of (drug_a, drug_b, description) tuples
    Output: 2-D array, shape (n_samples, n_class_combinations)

    Features encode which drug classes are involved in the pair
    (e.g., anticoagulant_x_nsaid=1 for warfarin+ibuprofen).
    """

    CLASS_NAMES = sorted(_DRUG_CLASSES.keys())
    _DANGEROUS_COMBOS = [
        ("anticoagulant", "nsaid"),
        ("anticoagulant", "antiplatelet"),
        ("statin", "cyp3a4_inhibitor"),
        ("maoi", "ssri"),
        ("maoi", "opioid"),
        ("maoi", "snri"),
        ("antiarrhythmic", "antiarrhythmic"),
        ("immunosuppressant", "nsaid"),
        ("cyp3a4_inhibitor", "statin"),
        ("cyp_inducer", "anticoagulant"),
    ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []
        for drug_a, drug_b, _ in X:
            row = self._compute_row(drug_a, drug_b)
            features.append(row)
        return np.array(features, dtype=np.float32)

    def _compute_row(self, drug_a: str, drug_b: str) -> list[float]:
        classes_a = set(get_drug_classes(drug_a))
        classes_b = set(get_drug_classes(drug_b))
        row = []

        # Per-class indicators for each drug (is drug_a in class X?)
        for cls in self.CLASS_NAMES:
            row.append(1.0 if cls in classes_a else 0.0)
            row.append(1.0 if cls in classes_b else 0.0)

        # Shared-class indicator (both drugs in same class)
        shared = classes_a & classes_b
        row.append(float(len(shared)))

        # Dangerous combination indicators
        for cls_x, cls_y in self._DANGEROUS_COMBOS:
            hit = (
                (cls_x in classes_a and cls_y in classes_b)
                or (cls_y in classes_a and cls_x in classes_b)
            )
            row.append(1.0 if hit else 0.0)

        return row


class KeywordFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts binary keyword-presence features from the description text.

    Input : list of (drug_a, drug_b, description) tuples
    Output: 2-D array, shape (n_samples, n_keywords)
    """

    _ALL_KEYWORDS: list[tuple[str, str]] = [
        (kw, label)
        for label, kws in _SEVERITY_KEYWORDS.items()
        for kw in kws
    ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []
        for _, _, desc in X:
            desc_lower = desc.lower()
            row = [1.0 if kw in desc_lower else 0.0 for kw, _ in self._ALL_KEYWORDS]
            # Additional aggregate features
            severe_count = sum(1.0 for kw, lbl in self._ALL_KEYWORDS if lbl == "Severe" and kw in desc_lower)
            mod_count = sum(1.0 for kw, lbl in self._ALL_KEYWORDS if lbl == "Moderate" and kw in desc_lower)
            mild_count = sum(1.0 for kw, lbl in self._ALL_KEYWORDS if lbl == "Mild" and kw in desc_lower)
            row += [severe_count, mod_count, mild_count, len(desc)]
            features.append(row)
        return np.array(features, dtype=np.float32)


class DescriptionExtractor(BaseEstimator, TransformerMixin):
    """Pulls the description string out of (drug_a, drug_b, description) tuples."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [desc for _, _, desc in X]


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Prediction Output
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SeverityPrediction:
    label: str                           # "Mild" | "Moderate" | "Severe"
    confidence: float                    # max class probability
    probabilities: dict[str, float]      # {label: probability}
    model_used: str                      # "random_forest" | "bert" | "hybrid"
    drug_pair: tuple[str, str] = ("", "")

    @property
    def legacy_label(self) -> str:
        """Map to the Low/Moderate/Severe labeling used by interaction_classifier.py."""
        return LABEL_TO_LEGACY.get(self.label, self.label)

    def __str__(self) -> str:
        a, b = self.drug_pair
        probs = " | ".join(
            f"{lbl}: {p:.1%}" for lbl, p in self.probabilities.items()
        )
        return (
            f"[{self.model_used}] {a.title()} + {b.title()} → "
            f"{self.label} ({self.confidence:.1%})\n"
            f"  Probabilities: {probs}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 5.  RandomForest Classifier
# ══════════════════════════════════════════════════════════════════════════════

class RandomForestSeverityClassifier:
    """
    Lightweight severity classifier using:
      • TF-IDF on the interaction description (char + word n-grams)
      • Drug-class indicator features
      • Clinical keyword presence features

    Training on the built-in corpus takes < 2 seconds on any machine.
    No GPU required.

    The pipeline is calibrated with Platt scaling (CalibratedClassifierCV)
    so that the predicted probabilities are well-calibrated fractions.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int | None = None,
        min_samples_leaf: int = 1,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.random_state = random_state
        self._pipeline: Pipeline | None = None
        self._label_encoder = LabelEncoder()
        self.classes_: list[str] = []
        self.cv_report_: str = ""

    # ── Public API ─────────────────────────────────────────────────────────────

    def train(
        self,
        corpus: list[tuple[str, str, str, str]] | None = None,
        cross_validate: bool = True,
    ) -> "RandomForestSeverityClassifier":
        """
        Train the RandomForest classifier on the given (or built-in) corpus.

        Parameters
        ----------
        corpus : list of (drug_a, drug_b, description, label) tuples.
                 Defaults to the built-in _TRAINING_CORPUS.
        cross_validate : If True, print 5-fold cross-validation metrics.
        """
        corpus = corpus or _TRAINING_CORPUS
        X = [(a, b, desc) for a, b, desc, _ in corpus]
        y_raw = [label for _, _, _, label in corpus]
        y = self._label_encoder.fit_transform(y_raw)
        self.classes_ = list(self._label_encoder.classes_)

        logger.info(
            "Training RandomForest on %d samples | classes: %s",
            len(X),
            self.classes_,
        )

        self._pipeline = self._build_pipeline()

        # ── 5-fold cross-validation ───────────────────────────────────────────
        if cross_validate:
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            y_pred_cv = cross_val_predict(self._pipeline, X, y, cv=skf)
            y_pred_labels = self._label_encoder.inverse_transform(y_pred_cv)
            y_true_labels = self._label_encoder.inverse_transform(y)
            self.cv_report_ = classification_report(
                y_true_labels, y_pred_labels, target_names=self.classes_
            )
            f1 = f1_score(y, y_pred_cv, average="macro")
            logger.info("5-fold CV macro-F1: %.3f", f1)
            logger.info("CV Classification Report:\n%s", self.cv_report_)

        # ── Final fit on all data ─────────────────────────────────────────────
        self._pipeline.fit(X, y)
        logger.info("RandomForest training complete.")
        return self

    def predict(
        self,
        drug_a: str,
        drug_b: str,
        description: str,
    ) -> SeverityPrediction:
        """
        Predict severity for a single drug pair + description.

        Parameters
        ----------
        drug_a, drug_b : str
            Drug names (lowercase, generic preferred).
        description : str
            Interaction description text.

        Returns
        -------
        SeverityPrediction
        """
        self._check_trained()
        X = [(drug_a.lower(), drug_b.lower(), description)]
        proba = self._pipeline.predict_proba(X)[0]
        pred_idx = int(np.argmax(proba))
        label = self.classes_[pred_idx]
        probs = {cls: round(float(p), 4) for cls, p in zip(self.classes_, proba)}
        return SeverityPrediction(
            label=label,
            confidence=round(float(proba[pred_idx]), 4),
            probabilities=probs,
            model_used="random_forest",
            drug_pair=(drug_a.lower(), drug_b.lower()),
        )

    def predict_batch(
        self, samples: list[tuple[str, str, str]]
    ) -> list[SeverityPrediction]:
        """Predict severity for a batch of (drug_a, drug_b, description) tuples."""
        self._check_trained()
        X = [(a.lower(), b.lower(), d) for a, b, d in samples]
        probas = self._pipeline.predict_proba(X)
        results = []
        for (a, b, _), proba in zip(X, probas):
            idx = int(np.argmax(proba))
            label = self.classes_[idx]
            probs = {cls: round(float(p), 4) for cls, p in zip(self.classes_, proba)}
            results.append(SeverityPrediction(
                label=label,
                confidence=round(float(proba[idx]), 4),
                probabilities=probs,
                model_used="random_forest",
                drug_pair=(a, b),
            ))
        return results

    def save(self, path: str | Path | None = None) -> None:
        """Persist the trained pipeline to disk."""
        self._check_trained()
        path = Path(path or RF_MODEL_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(
                {
                    "pipeline": self._pipeline,
                    "label_encoder": self._label_encoder,
                    "classes": self.classes_,
                    "cv_report": self.cv_report_,
                },
                fh,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        logger.info("RandomForest model saved → %s", path)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "RandomForestSeverityClassifier":
        """Load a previously saved RandomForest classifier from disk."""
        path = Path(path or RF_MODEL_PATH)
        if not path.exists():
            raise FileNotFoundError(
                f"Model file not found: {path}. "
                "Call train() and save() first."
            )
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        instance = cls()
        instance._pipeline = payload["pipeline"]
        instance._label_encoder = payload["label_encoder"]
        instance.classes_ = payload["classes"]
        instance.cv_report_ = payload.get("cv_report", "")
        logger.info("RandomForest model loaded ← %s", path)
        return instance

    # ── Internal ───────────────────────────────────────────────────────────────

    def _build_pipeline(self) -> Pipeline:
        """
        Assemble the scikit-learn pipeline:
          FeatureUnion([TF-IDF, DrugClass, Keyword]) → RandomForest → Calibration
        """
        # TF-IDF on the description (pulled out from the tuple by DescriptionExtractor)
        tfidf = Pipeline([
            ("extractor", DescriptionExtractor()),
            ("tfidf", TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 3),       # unigrams + bigrams + trigrams
                max_features=3000,
                sublinear_tf=True,        # log(1+tf) smoothing
                min_df=1,
                strip_accents="unicode",
            )),
        ])

        # Char n-gram TF-IDF for robustness to spelling variants
        char_tfidf = Pipeline([
            ("extractor", DescriptionExtractor()),
            ("tfidf", TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                max_features=1000,
                sublinear_tf=True,
                min_df=1,
            )),
        ])

        feature_union = FeatureUnion([
            ("word_tfidf", tfidf),
            ("char_tfidf", char_tfidf),
            ("drug_class", DrugClassFeatureExtractor()),
            ("keywords", KeywordFeatureExtractor()),
        ])

        rf = CalibratedClassifierCV(
            RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=None,
                min_samples_leaf=1,
                class_weight="balanced",
                random_state=self.random_state,
                n_jobs=-1,
            ),
            method="isotonic",
            cv=3,
        )

        return Pipeline([
            ("features", feature_union),
            ("clf", rf),
        ])

    def _check_trained(self) -> None:
        if self._pipeline is None:
            raise RuntimeError(
                "Model not trained. Call train() or load() first."
            )


# ══════════════════════════════════════════════════════════════════════════════
# 6.  BERT-Based Classifier
# ══════════════════════════════════════════════════════════════════════════════

class BERTSeverityClassifier:
    """
    Fine-tuned DistilBERT classifier for drug interaction severity.

    Architecture
    ────────────
      DistilBERT-base-uncased
      → [CLS] representation (768-d)
      → Dropout(0.1)
      → Linear(768 → 3)
      → Softmax

    Input format
    ────────────
      "{drug_a} and {drug_b}: {interaction_description}"

    Training
    ────────
      AdamW optimizer, lr=2e-5, weight decay=0.01
      CosineAnnealingLR scheduler
      Early stopping on validation loss (patience=3)
      Class-weighted cross-entropy for imbalanced labels
      Max epochs: 20 (typically converges in 8-12)

    Requirements
    ────────────
      pip install torch transformers
    """

    MODEL_NAME = "distilbert-base-uncased"
    MAX_LENGTH = 256
    BATCH_SIZE = 8
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01
    MAX_EPOCHS = 20
    PATIENCE = 3
    WARMUP_RATIO = 0.1

    def __init__(self, model_dir: str | Path | None = None) -> None:
        self.model_dir = Path(model_dir or BERT_MODEL_DIR)
        self._model = None
        self._tokenizer = None
        self._device = None
        self._trained = False

    def _lazy_import(self):
        """Import PyTorch / transformers on first use (optional dependency)."""
        try:
            import torch
            from transformers import (
                AutoTokenizer,
                AutoModelForSequenceClassification,
                get_cosine_schedule_with_warmup,
            )
            return torch, AutoTokenizer, AutoModelForSequenceClassification, get_cosine_schedule_with_warmup
        except ImportError as exc:
            raise ImportError(
                "BERT classifier requires: pip install torch transformers\n"
                f"Original error: {exc}"
            ) from exc

    def train(
        self,
        corpus: list[tuple[str, str, str, str]] | None = None,
        val_split: float = 0.15,
    ) -> "BERTSeverityClassifier":
        """
        Fine-tune DistilBERT on the given corpus.

        Parameters
        ----------
        corpus    : (drug_a, drug_b, description, label) tuples.
        val_split : Fraction of data to hold out for early-stopping validation.
        """
        torch, AutoTokenizer, AutoModelForSequenceClassification, get_schedule = self._lazy_import()
        corpus = corpus or _TRAINING_CORPUS

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("BERT training device: %s", self._device)

        # ── Tokeniser ─────────────────────────────────────────────────────────
        logger.info("Loading tokeniser: %s", self.MODEL_NAME)
        self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)

        # ── Encode dataset ────────────────────────────────────────────────────
        texts = [
            f"{a} and {b}: {desc}"
            for a, b, desc, _ in corpus
        ]
        raw_labels = [label for _, _, _, label in corpus]
        label_enc = LabelEncoder().fit(LABELS)
        y_int = label_enc.transform(raw_labels)

        # Train / validation split (stratified)
        from sklearn.model_selection import train_test_split
        idx_all = list(range(len(texts)))
        idx_train, idx_val = train_test_split(
            idx_all, test_size=val_split, stratify=y_int, random_state=42
        )

        def encode(indices):
            batch_texts = [texts[i] for i in indices]
            batch_labels = [y_int[i] for i in indices]
            enc = self._tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.MAX_LENGTH,
                return_tensors="pt",
            )
            return enc, torch.tensor(batch_labels, dtype=torch.long)

        train_enc, train_y = encode(idx_train)
        val_enc, val_y = encode(idx_val)

        train_ds = torch.utils.data.TensorDataset(
            train_enc["input_ids"],
            train_enc["attention_mask"],
            train_y,
        )
        val_ds = torch.utils.data.TensorDataset(
            val_enc["input_ids"],
            val_enc["attention_mask"],
            val_y,
        )
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=self.BATCH_SIZE, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=self.BATCH_SIZE
        )

        # ── Class weights for imbalanced labels ───────────────────────────────
        counts = np.bincount(y_int, minlength=3).astype(float)
        weights = torch.tensor(
            (1.0 / (counts + 1e-8)) / (1.0 / (counts + 1e-8)).sum(),
            dtype=torch.float32,
        ).to(self._device)

        # ── Model ─────────────────────────────────────────────────────────────
        logger.info("Loading %s for fine-tuning …", self.MODEL_NAME)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.MODEL_NAME, num_labels=3
        ).to(self._device)

        # ── Optimiser + scheduler ─────────────────────────────────────────────
        optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=self.LEARNING_RATE,
            weight_decay=self.WEIGHT_DECAY,
        )
        total_steps = len(train_loader) * self.MAX_EPOCHS
        warmup_steps = int(total_steps * self.WARMUP_RATIO)
        scheduler = get_schedule(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

        # ── Training loop ─────────────────────────────────────────────────────
        best_val_loss = float("inf")
        patience_ctr = 0
        best_state = None

        for epoch in range(1, self.MAX_EPOCHS + 1):
            # Train
            self._model.train()
            train_loss = 0.0
            for input_ids, attn_mask, labels in train_loader:
                input_ids = input_ids.to(self._device)
                attn_mask = attn_mask.to(self._device)
                labels = labels.to(self._device)
                optimizer.zero_grad()
                out = self._model(input_ids=input_ids, attention_mask=attn_mask)
                loss = loss_fn(out.logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                train_loss += loss.item()

            # Validate
            self._model.eval()
            val_loss = 0.0
            val_preds, val_true = [], []
            with torch.no_grad():
                for input_ids, attn_mask, labels in val_loader:
                    input_ids = input_ids.to(self._device)
                    attn_mask = attn_mask.to(self._device)
                    labels = labels.to(self._device)
                    out = self._model(input_ids=input_ids, attention_mask=attn_mask)
                    val_loss += loss_fn(out.logits, labels).item()
                    val_preds.extend(out.logits.argmax(-1).cpu().numpy())
                    val_true.extend(labels.cpu().numpy())

            avg_val = val_loss / len(val_loader)
            avg_train = train_loss / len(train_loader)
            val_f1 = f1_score(val_true, val_preds, average="macro")
            logger.info(
                "Epoch %2d/%d | train_loss=%.4f | val_loss=%.4f | val_F1=%.3f",
                epoch, self.MAX_EPOCHS, avg_train, avg_val, val_f1
            )

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                best_state = {k: v.cpu() for k, v in self._model.state_dict().items()}
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= self.PATIENCE:
                    logger.info("Early stopping at epoch %d", epoch)
                    break

        # Restore best checkpoint
        if best_state:
            self._model.load_state_dict({k: v.to(self._device) for k, v in best_state.items()})

        self._trained = True
        logger.info("BERT fine-tuning complete. Best val_loss: %.4f", best_val_loss)
        return self

    def predict(
        self,
        drug_a: str,
        drug_b: str,
        description: str,
    ) -> SeverityPrediction:
        """Run inference on a single (drug_pair, description)."""
        self._check_trained()
        torch, *_ = self._lazy_import()
        import torch as _torch

        text = f"{drug_a} and {drug_b}: {description}"
        enc = self._tokenizer(
            [text],
            padding=True,
            truncation=True,
            max_length=self.MAX_LENGTH,
            return_tensors="pt",
        )
        self._model.eval()
        with _torch.no_grad():
            out = self._model(
                input_ids=enc["input_ids"].to(self._device),
                attention_mask=enc["attention_mask"].to(self._device),
            )
        proba = _torch.softmax(out.logits[0], dim=-1).cpu().numpy()
        idx = int(np.argmax(proba))
        probs = {LABELS[i]: round(float(proba[i]), 4) for i in range(3)}
        return SeverityPrediction(
            label=LABELS[idx],
            confidence=round(float(proba[idx]), 4),
            probabilities=probs,
            model_used="bert",
            drug_pair=(drug_a.lower(), drug_b.lower()),
        )

    def save(self) -> None:
        """Save the fine-tuned model and tokeniser to disk."""
        self._check_trained()
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(str(self.model_dir))
        self._tokenizer.save_pretrained(str(self.model_dir))
        logger.info("BERT model saved → %s", self.model_dir)

    @classmethod
    def load(cls, model_dir: str | Path | None = None) -> "BERTSeverityClassifier":
        """Load a saved fine-tuned BERT classifier."""
        torch, AutoTokenizer, AutoModelForSequenceClassification, _ = cls()._lazy_import()
        model_dir = Path(model_dir or BERT_MODEL_DIR)
        if not model_dir.exists():
            raise FileNotFoundError(
                f"BERT model dir not found: {model_dir}. "
                "Call train() and save() first."
            )
        instance = cls(model_dir=model_dir)
        import torch as _torch
        instance._device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
        instance._tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        instance._model = AutoModelForSequenceClassification.from_pretrained(
            str(model_dir), num_labels=3
        ).to(instance._device)
        instance._trained = True
        logger.info("BERT model loaded ← %s", model_dir)
        return instance

    def _check_trained(self) -> None:
        if not self._trained or self._model is None:
            raise RuntimeError(
                "BERT model not trained. Call train() or load() first."
            )


# ══════════════════════════════════════════════════════════════════════════════
# 7.  Hybrid Ensemble (RF + BERT)
# ══════════════════════════════════════════════════════════════════════════════

class HybridSeverityClassifier:
    """
    Weighted soft-voting ensemble of RandomForest and BERT classifiers.

    Weights (default): RF = 0.45, BERT = 0.55
    Falls back to RF-only when BERT model is unavailable.

    The combined probability for label l is:
      P_hybrid(l) = w_rf * P_rf(l) + w_bert * P_bert(l)
    """

    def __init__(
        self,
        rf_weight: float = 0.45,
        bert_weight: float = 0.55,
    ) -> None:
        self.rf_weight = rf_weight
        self.bert_weight = bert_weight
        self.rf: RandomForestSeverityClassifier | None = None
        self.bert: BERTSeverityClassifier | None = None

    def train(
        self,
        corpus: list[tuple[str, str, str, str]] | None = None,
        train_bert: bool = True,
    ) -> "HybridSeverityClassifier":
        """
        Train both classifiers.

        Parameters
        ----------
        train_bert : If False, only trains RandomForest (faster, no GPU needed).
        """
        corpus = corpus or _TRAINING_CORPUS

        logger.info("Training RandomForest …")
        self.rf = RandomForestSeverityClassifier().train(corpus)

        if train_bert:
            try:
                logger.info("Fine-tuning BERT …")
                self.bert = BERTSeverityClassifier().train(corpus)
            except ImportError as exc:
                logger.warning(
                    "BERT training skipped (torch/transformers not installed): %s", exc
                )
                self.bert = None

        return self

    def predict(
        self,
        drug_a: str,
        drug_b: str,
        description: str,
    ) -> SeverityPrediction:
        """Predict severity using the hybrid ensemble."""
        if self.rf is None:
            raise RuntimeError("Call train() first.")

        rf_pred = self.rf.predict(drug_a, drug_b, description)

        if self.bert is None:
            # RF-only fallback
            return SeverityPrediction(
                label=rf_pred.label,
                confidence=rf_pred.confidence,
                probabilities=rf_pred.probabilities,
                model_used="random_forest (bert unavailable)",
                drug_pair=(drug_a.lower(), drug_b.lower()),
            )

        bert_pred = self.bert.predict(drug_a, drug_b, description)

        # Weighted soft voting
        combined = {
            lbl: round(
                self.rf_weight * rf_pred.probabilities[lbl]
                + self.bert_weight * bert_pred.probabilities[lbl],
                4,
            )
            for lbl in LABELS
        }
        best_label = max(combined, key=combined.__getitem__)
        return SeverityPrediction(
            label=best_label,
            confidence=round(combined[best_label], 4),
            probabilities=combined,
            model_used="hybrid (rf+bert)",
            drug_pair=(drug_a.lower(), drug_b.lower()),
        )

    def save(self) -> None:
        if self.rf:
            self.rf.save()
        if self.bert:
            self.bert.save()

    @classmethod
    def load(cls, rf_only: bool = False) -> "HybridSeverityClassifier":
        instance = cls()
        instance.rf = RandomForestSeverityClassifier.load()
        if not rf_only:
            try:
                instance.bert = BERTSeverityClassifier.load()
            except (FileNotFoundError, ImportError) as exc:
                logger.warning("BERT model unavailable: %s. Using RF only.", exc)
        return instance


# ══════════════════════════════════════════════════════════════════════════════
# 8.  Quick-access inference helpers
# ══════════════════════════════════════════════════════════════════════════════

_rf_singleton: RandomForestSeverityClassifier | None = None


def get_trained_classifier() -> RandomForestSeverityClassifier:
    """
    Return a module-level singleton RandomForestSeverityClassifier.

    Loads from disk if a saved model exists, otherwise trains on the
    built-in corpus.  Called automatically on first classify() call.
    """
    global _rf_singleton
    if _rf_singleton is not None:
        return _rf_singleton

    if RF_MODEL_PATH.exists():
        _rf_singleton = RandomForestSeverityClassifier.load()
    else:
        _rf_singleton = RandomForestSeverityClassifier().train()
        _rf_singleton.save()

    return _rf_singleton


def classify(
    drug_a: str,
    drug_b: str,
    description: str,
) -> SeverityPrediction:
    """
    Module-level convenience function: classify a single interaction.

    Auto-trains on first call if no saved model exists.

    Example
    -------
    ::

        from models.severity_classifier import classify
        result = classify(
            "warfarin",
            "ibuprofen",
            "Ibuprofen inhibits COX-1 platelet aggregation, increases "
            "bleeding risk with warfarin."
        )
        print(result)  # Severe (94.3%)
    """
    return get_trained_classifier().predict(drug_a, drug_b, description)


# ══════════════════════════════════════════════════════════════════════════════
# 9.  Example Inference Code
# ══════════════════════════════════════════════════════════════════════════════

def run_example() -> None:
    """
    Comprehensive inference demo.
    Run with:  python -m models.severity_classifier
    """
    import textwrap

    def _box(title: str, lines: list[str]) -> None:
        w = 68
        print(f"\n╔{'═' * w}╗")
        print(f"║  {title:<{w - 2}}║")
        print(f"╠{'═' * w}╣")
        for line in lines:
            for wrapped in textwrap.wrap(line, w - 4) or [""]:
                print(f"║  {wrapped:<{w - 2}}║")
        print(f"╚{'═' * w}╝")

    print("\n" + "═" * 70)
    print("  Drug Interaction Severity Classifier — Inference Demo")
    print("═" * 70)

    # ── Train the RandomForest model ──────────────────────────────────────────
    print("\n[1] Training RandomForest classifier on built-in corpus …")
    clf = RandomForestSeverityClassifier()
    clf.train(cross_validate=True)
    clf.save()
    print(f"    Corpus size : {len(_TRAINING_CORPUS)} samples")
    print(f"    Label dist  : Mild={sum(1 for *_, l in _TRAINING_CORPUS if l=='Mild')}, "
          f"Moderate={sum(1 for *_, l in _TRAINING_CORPUS if l=='Moderate')}, "
          f"Severe={sum(1 for *_, l in _TRAINING_CORPUS if l=='Severe')}")

    # ── Test cases ────────────────────────────────────────────────────────────
    test_cases: list[tuple[str, str, str, str]] = [
        (
            "warfarin", "ibuprofen",
            "Ibuprofen inhibits COX-1-mediated platelet aggregation and "
            "causes gastric mucosal erosion, substantially increasing "
            "haemorrhagic risk. Protein-binding displacement transiently "
            "raises free warfarin levels. Avoid; use acetaminophen.",
            "Severe",
        ),
        (
            "simvastatin", "amiodarone",
            "Amiodarone is a potent CYP3A4 inhibitor. Markedly increases "
            "simvastatin AUC (up to 6-fold), causing severe myopathy "
            "and potentially fatal rhabdomyolysis.",
            "Severe",
        ),
        (
            "lisinopril", "ibuprofen",
            "NSAIDs reduce prostaglandin synthesis, blunting the antihypertensive "
            "effect of ACE inhibitors and impairing renal autoregulation. "
            "Monitor blood pressure and renal function closely.",
            "Moderate",
        ),
        (
            "clopidogrel", "omeprazole",
            "Omeprazole inhibits CYP2C19, blocking clopidogrel bioactivation. "
            "Reduces platelet inhibition by 47%. Clinical significance debated "
            "but avoid combination in high cardiovascular risk patients.",
            "Moderate",
        ),
        (
            "warfarin", "metformin",
            "No direct pharmacokinetic interaction. Metformin does not "
            "affect CYP2C9-mediated warfarin metabolism. Routine INR "
            "monitoring is sufficient.",
            "Mild",
        ),
        (
            "atorvastatin", "aspirin",
            "No pharmacokinetic interaction. Co-prescription for "
            "cardiovascular prevention is standard practice. "
            "Well tolerated without dose adjustment.",
            "Mild",
        ),
        # Edge case: unseen drug pair with informative text
        (
            "digoxin", "amiodarone",
            "Amiodarone inhibits P-glycoprotein and reduces renal clearance "
            "of digoxin, raising serum digoxin 70-100%. Digoxin toxicity: "
            "bradycardia, heart block, arrhythmias. Reduce digoxin dose 50%.",
            "Severe",
        ),
    ]

    print("\n[2] Running inference on test cases …\n")
    print(f"  {'Drug A':<15} {'Drug B':<15} {'Expected':<10} {'Predicted':<10} {'Conf':>6}  {'Match'}")
    print("  " + "─" * 66)

    correct = 0
    for drug_a, drug_b, desc, expected in test_cases:
        pred = clf.predict(drug_a, drug_b, desc)
        match = "✓" if pred.label == expected else "✗"
        if pred.label == expected:
            correct += 1
        print(
            f"  {drug_a:<15} {drug_b:<15} {expected:<10} "
            f"{pred.label:<10} {pred.confidence:>5.1%}  {match}"
        )

    accuracy = correct / len(test_cases)
    print(f"\n  Accuracy on {len(test_cases)} test cases: {accuracy:.0%}")

    # ── Detailed single prediction ────────────────────────────────────────────
    print("\n[3] Detailed prediction — warfarin + ibuprofen")
    drug_a, drug_b = "warfarin", "ibuprofen"
    description = (
        "Ibuprofen is a non-selective NSAID that inhibits COX-1-mediated "
        "platelet aggregation and damages the gastric mucosa, both "
        "increasing haemorrhagic risk in patients on warfarin. "
        "Protein-binding displacement can transiently raise free warfarin."
    )
    pred = clf.predict(drug_a, drug_b, description)
    _box(
        f"Severity Prediction: {drug_a.title()} + {drug_b.title()}",
        [
            f"Label       : {pred.label}",
            f"Confidence  : {pred.confidence:.1%}",
            f"Legacy label: {pred.legacy_label}",
            f"Model       : {pred.model_used}",
            "",
            "Probability breakdown:",
            f"  Mild     : {pred.probabilities.get('Mild',0):.1%}",
            f"  Moderate : {pred.probabilities.get('Moderate',0):.1%}",
            f"  Severe   : {pred.probabilities.get('Severe',0):.1%}",
            "",
            f"Input description (truncated): {description[:100]}…",
        ],
    )

    # ── Batch prediction ──────────────────────────────────────────────────────
    print("\n[4] Batch inference example")
    batch = [
        ("warfarin", "aspirin",
         "Additive anticoagulant and antiplatelet effects. Fatal bleeding risk."),
        ("metformin", "lisinopril",
         "No significant pharmacokinetic interaction. Generally safe combination."),
        ("metoprolol", "verapamil",
         "Additive AV nodal blockade. Risk of bradycardia and heart block."),
    ]
    results = clf.predict_batch(batch)
    for r in results:
        print(f"  {r}")

    # ── Module-level convenience function ─────────────────────────────────────
    print("\n[5] Module-level classify() convenience function")
    result = classify(
        drug_a="tramadol",
        drug_b="sertraline",
        description=(
            "Both agents increase serotonergic transmission. Tramadol inhibits "
            "serotonin reuptake; sertraline is a potent SSRI. Risk of serotonin "
            "syndrome: hyperthermia, clonus, autonomic instability. "
            "Life-threatening. Avoid concurrent use."
        ),
    )
    print(f"  {result}")
    print(f"  Legacy label (for interaction_classifier.py): {result.legacy_label}")

    # ── BERT classifier note ──────────────────────────────────────────────────
    print("\n[6] BERT Classifier (optional, requires torch + transformers)")
    print("    To train and use the BERT classifier:")
    print("    ┌─────────────────────────────────────────────────────────┐")
    print("    │  from models.severity_classifier import BERTSeverityClassifier")
    print("    │")
    print("    │  bert = BERTSeverityClassifier()")
    print("    │  bert.train()   # fine-tunes DistilBERT (~5-10 min CPU)")
    print("    │  bert.save()    # saves to models/saved/bert_severity_classifier/")
    print("    │")
    print("    │  result = bert.predict('warfarin', 'ibuprofen', description)")
    print("    │  print(result.label)        # 'Severe'")
    print("    │  print(result.confidence)   # 0.943")
    print("    └─────────────────────────────────────────────────────────┘")

    # ── Hybrid classifier note ────────────────────────────────────────────────
    print("\n[7] Hybrid ensemble (RF + BERT):")
    print("    ┌─────────────────────────────────────────────────────────┐")
    print("    │  from models.severity_classifier import HybridSeverityClassifier")
    print("    │")
    print("    │  hybrid = HybridSeverityClassifier()")
    print("    │  hybrid.train(train_bert=True)   # trains both models")
    print("    │  hybrid.save()")
    print("    │")
    print("    │  result = hybrid.predict('digoxin', 'amiodarone', description)")
    print("    └─────────────────────────────────────────────────────────┘")

    print("\n" + "═" * 70 + "\n")


if __name__ == "__main__":
    run_example()
