/** Deterministic research-fixture lookup. No ML, inference, or remote calls. */
export function searchFixtures(records, query) {
  if (query.trim().length < 2) return [];
  const names = [...new Set(records.flatMap(r => [r.drug_a, r.drug_b]))];
  return names.filter(n => n.includes(query.trim().toLowerCase())).sort().slice(0, 15);
}
export function lookupFixtures(records, input) {
  if (!Array.isArray(input) || input.some(d => typeof d !== 'string' || !d.trim())) throw new Error('Enter drug names.');
  const drugs = [...new Set(input.map(d => d.trim().toLowerCase()))];
  if (drugs.length < 2 || drugs.length > 10) throw new Error('Select between 2 and 10 different drugs.');
  const pairs = [];
  for (let i = 0; i < drugs.length; i++) for (let j = i + 1; j < drugs.length; j++) {
    const pair = [drugs[i], drugs[j]];
    const matches = records.filter(r => [r.drug_a, r.drug_b].sort().join('|') === [...pair].sort().join('|'));
    const labels = new Set(matches.map(r => r.severity));
    const record = matches[0];
    const supported = record && labels.size === 1 && ['Severe', 'Moderate', 'Low'].includes(record.severity);
    pairs.push({ pair, severity: supported ? record.severity : 'Unknown', confidence: 0,
      classification_method: supported ? 'curated_fixture_lookup' : 'abstain_no_consistent_fixture',
      mechanism: supported ? record.mechanism : 'No consistent record for this pair in the bundled research fixtures.',
      clinical_effects: supported ? record.clinical_effects : '',
      management: 'This demo cannot recommend treatment or medication changes. Consult a qualified clinician or pharmacist.',
      monitoring: [], interaction_type: '', source_name: 'Repository research fixture; not clinically adjudicated',
      source_url: 'https://github.com/vajja1405/drug-interaction-rag-chatbot/blob/main/data_pipeline/fetch_drug_data.py',
      source_page: 'Inspect the original curated fixture (not a clinical source)' });
  }
  // Missing evidence cannot establish low overall risk, even when another pair has a known fixture.
  const overall = pairs.some(p => p.severity === 'Unknown') ? 'Unknown' :
    pairs.some(p => p.severity === 'Severe') ? 'Severe' : pairs.some(p => p.severity === 'Moderate') ? 'Moderate' : 'Low';
  return { request_id: 'browser-demo', drugs, pairs_checked: pairs.map(p => p.pair.join(' + ')),
    overall_risk: overall, interactions: pairs,
    explanation: 'These are stored research-fixture labels, not a new medical assessment. Missing evidence means unknown, not safe. No language model, trained classifier, or Python API runs in this public demo.',
    monitoring_priorities: [], retrieved_docs: pairs.filter(p => p.severity !== 'Unknown').length,
    processing_time_ms: 0, disclaimer: 'Educational engineering prototype. Fixtures are incomplete and not clinically validated. Do not use this demo to make care decisions.' };
}
