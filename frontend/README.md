# Drug Interaction Analysis — Frontend

React + TypeScript UI for the Drug Interaction Analysis Chatbot.

## Quick Start

> **Prerequisite:** The FastAPI backend must be running on port 8000.

```bash
# Terminal 1 — Start the backend
cd DrugInteractionAI
uvicorn api.server:app --reload --port 8000

# Terminal 2 — Start the frontend
cd DrugInteractionAI/frontend
npm install
npm run dev
```

Open **http://localhost:5173** in your browser.

## Features

- **Drug Autocomplete** — Search by name with suggestions from RxNorm
- **Multi-Drug Selection** — Add/remove drugs as chips, then analyze all pairs
- **Interaction Results** — Severity-coded cards (Severe → Moderate → Low), collapsible with mechanism, clinical effects, management, and monitoring details
- **Loading & Errors** — Full-screen spinner during analysis, inline error banners)
- **Responsive** — Works on mobile and desktop

## Dev Proxy

Vite dev server proxies `/api/*` and `/analyze_interaction` to `http://localhost:8000`, so no CORS configuration is needed during development.

## Build for Production

```bash
npm run build     # outputs to dist/
npm run preview   # preview production build locally
```

## Stack

| Layer   | Tech                          |
|---------|-------------------------------|
| UI      | React 18 + TypeScript         |
| Build   | Vite 6                        |
| Styling | Vanilla CSS (dark theme)      |
| Backend | FastAPI (port 8000)           |
