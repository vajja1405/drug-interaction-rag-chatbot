# Drug Interaction Analysis Chatbot (RAG-Based)

Status: Ongoing (Feb 2026 – Present)

## Overview

This project develops a Retrieval-Augmented Generation (RAG) chatbot for multi-drug interaction analysis.

The system reduces hallucinations by grounding responses in retrieved clinical literature and classifies interaction severity levels.

## Key Features

- Multi-drug interaction analysis
- Retrieval from curated clinical literature
- Severity classification (Minor / Moderate / Severe)
- Risk escalation logic
- Hallucination reduction via grounded generation

## Architecture

User Query
    ↓
Embedding + Vector Search
    ↓
Relevant Clinical Literature Retrieval
    ↓
LLM Response Generation
    ↓
Severity Classification
    ↓
Risk Escalation Alert (if needed)

## Tech Stack

- Python
- LangChain / LlamaIndex
- OpenAI / LLM API
- FAISS / Chroma
- Streamlit

## Research Focus

- Reducing hallucination in medical AI
- Grounded generation
- Multi-drug reasoning
- Clinical safety applications
