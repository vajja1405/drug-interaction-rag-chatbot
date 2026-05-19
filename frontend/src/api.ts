/* ── Types ──────────────────────────────────────────────────────────────────── */

export interface DrugInteraction {
    pair: string[];
    severity: string;
    confidence: number;
    classification_method: string;
    mechanism: string;
    clinical_effects: string;
    management: string;
    monitoring: string[];
    interaction_type: string;
    source_name: string;
    source_url: string;
    source_page: string;
}

export interface AnalyzeResponse {
    request_id: string;
    drugs: string[];
    pairs_checked: string[];
    overall_risk: string;
    interactions: DrugInteraction[];
    explanation: string;
    monitoring_priorities: string[];
    retrieved_docs: number;
    processing_time_ms: number;
    disclaimer: string;
}

export interface SearchResponse {
    suggestions: string[];
    error?: string;
}

/* ── API Functions ─────────────────────────────────────────────────────────── */

export async function searchDrugs(query: string): Promise<string[]> {
    if (query.length < 2) return [];
    const res = await fetch(`/api/v1/drugs/search?q=${encodeURIComponent(query)}`);
    if (!res.ok) return [];
    const data: SearchResponse = await res.json();
    return data.suggestions ?? [];
}

export async function analyzeInteractions(drugs: string[]): Promise<AnalyzeResponse> {
    const res = await fetch('/analyze_interaction', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ drugs }),
    });
    if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: 'Analysis request failed' }));
        throw new Error(err.detail ?? `HTTP ${res.status}`);
    }
    return res.json();
}
