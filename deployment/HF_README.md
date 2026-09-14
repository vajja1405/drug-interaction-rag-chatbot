---
title: Drug Interaction AI
emoji: 💊
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
license: mit
---

Medication evidence workspace with two clearly separated modes.

[Open application](https://astra6-drug-interaction-ai.hf.space) · [Source, validation and limitations](https://github.com/vajja1405/drug-interaction-rag-chatbot)

**Medication label review:** search 23,651 RxNorm ingredient, combination and brand terms; select up to 20 concepts and inspect up to 190 pairs. DailyMed human labels are matched by RxCUI and checked for active-ingredient compatibility. Results show original passages, possible ingredient overlap, source versions and coverage gaps. Includes a pair matrix and a downloadable discussion report. No LLM is used for this mode. Terminology count is not verified interaction coverage. No direct mention does not mean safe.

**RAG research demo:** the original React/FastAPI/FAISS/classifier/LLM path remains available in its own tab with a five-medication limit. It uses a small curated research corpus and Qwen/Qwen3-4B-Instruct-2507 through nscale via Hugging Face Inference Providers. It is not clinically validated.

CPU Basic hosts the application. Runtime credentials stay in Space secrets. Label review uses bounded NLM requests and a 12-hour source cache; no patient records or user medication lists are persisted by the review application. Hosting/upstream services may retain logs. See the repository for operating limits and test scope.

This tool helps organize a discussion with a pharmacist. It does not assess personalized treatment, class-based or higher-order interactions, or certify medication combinations as safe.

This product uses publicly available data from the U.S. National Library of Medicine (NLM), National Institutes of Health, Department of Health and Human Services; NLM is not responsible for the product and does not endorse or recommend this or any other product.
