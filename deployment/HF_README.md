---
title: Drug Interaction AI
emoji: 💊
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
license: mit
---

Full React + FastAPI research application with embeddings, FAISS retrieval, severity classification and live LLM explanations.

[Open application](https://astra6-drug-interaction-ai.hf.space) · [Source and validation](https://github.com/vajja1405/drug-interaction-rag-chatbot) · [Browser-only backup](https://vajja1405.github.io/drug-interaction-rag-chatbot/)

The Space uses CPU Basic for retrieval/classification and Qwen/Qwen3-4B-Instruct-2507 through nscale via Hugging Face Inference Providers for explanations. Visitors need no account. Medication names are sent to the server and model provider. Use demonstration inputs only: the curated corpus is incomplete and not clinically adjudicated. This application is not for treatment decisions.

Runtime secrets/settings: OPENAI_API_KEY (secret), OPENAI_BASE_URL and LLM_MODEL. Credentials are never included in the source or browser bundle. Current hosted limits: five drugs, one concurrent uncached analysis, five requests/minute per observed client IP, 50 uncached analyses/process/UTC day, 1,800 output tokens and 40 seconds per model attempt. The process counter resets on restart and is not a billing cap. Provider failures are displayed explicitly and not cached as successful generations.

September 11 verification: a live supported-pair response, a cached repeat, an unsupported-pair abstention, concurrent readiness and the React browser flow passed. See the GitHub repository for the exact smoke-check scope. Engineering checks do not establish clinical effectiveness or generated-answer faithfulness.
