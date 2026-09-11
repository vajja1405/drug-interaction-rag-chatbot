import type { AnalyzeResponse } from './api';
export interface Fixture { drug_a: string; drug_b: string; severity: string; mechanism: string; clinical_effects: string }
export function searchFixtures(records: Fixture[], query: string): string[];
export function lookupFixtures(records: Fixture[], input: string[]): AnalyzeResponse;
