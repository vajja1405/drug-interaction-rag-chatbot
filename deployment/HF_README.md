---
title: Drug Interaction AI
emoji: 💊
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
license: mit
---

Full React + FastAPI research application. See the GitHub project for evidence limitations and validation results.

Set OPENAI_API_KEY, OPENAI_BASE_URL and LLM_MODEL as runtime settings. API keys must be secrets. Do not put keys in the image or README. CPU Basic has sufficient memory for retrieval/classification; a remote model provider supplies explanation inference. Creating a Docker Space currently requires Hugging Face Pro.
