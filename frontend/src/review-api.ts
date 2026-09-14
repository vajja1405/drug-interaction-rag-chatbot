export interface Medication { rxcui: string; name: string; tty: string }
export interface Label { title: string; url: string; setid: string; version: string; published_date: string; effective_date: string; sha256: string; sections: {section: string; characters: number; truncated: boolean}[] }
export interface ReviewedMedication extends Medication { ingredients: Medication[]; ingredient_resolution: string; labels: Label[]; status: string; label_matches: number; label_selection: string }
export interface Pair { ids: string[]; names: string[]; status: string; shared_ingredients: string[]; evidence_truncated: boolean; source_complete: boolean; evidence: {source_drug: string; matched_term: string; section: string; excerpt: string; source_url: string; source_title: string; setid: string; version: string}[] }
export interface Review { request_id: string; medications: ReviewedMedication[]; pairs: Pair[]; pairs_reviewed: number; pairs_with_incomplete_sources: number; counts: Record<string, number>; generated_at: string; processing_time_ms: number; limitations: string[]; method: string; clinical_validation: string }
export interface CatalogInfo { terms: number; max_medications: number; max_pairs: number; metadata: { retrieved_at: string; source_version: Record<string, string>; counts: Record<string, number> } }

export async function requestJson<T>(url: string, options: RequestInit = {}): Promise<T> {
    const response = await fetch(url, options)
    if (!response.ok) {
        const data = await response.json().catch(() => ({}))
        throw new Error(typeof data.detail === 'string' ? data.detail : `Request unavailable (${response.status}). Please retry.`)
    }
    return response.json()
}

export const statusNames: Record<string, string> = {
    ingredient_overlap: 'Possible ingredient overlap', label_mention: 'Direct label mention',
    incomplete_sources: 'Incomplete sources', no_direct_mention: 'No direct mention',
    source_unavailable: 'Source unavailable', no_matching_label: 'No matching human label',
    interaction_section_unavailable: 'Interaction section unavailable', label_available: 'Interaction section retrieved',
    label_match_unverified: 'Matching ingredient set not verified',
}
