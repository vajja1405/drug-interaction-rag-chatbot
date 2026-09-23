// Plain text avoids formula execution if names are copied into spreadsheet software.
export function reviewSummary(result) {
  const rows = [
    'Medication interaction research review',
    'Unvalidated prototype output. Not a diagnosis, treatment plan, or safety clearance.',
    `Medication names entered: ${result.drugs.join(', ')}`,
    `Request ID: ${result.request_id}`,
    `Evidence version: ${result.evidence_version || 'not recorded'}`,
    `Generation status: ${result.generation_status || 'not recorded'}`,
    `Pairs checked: ${result.pairs_checked.length}`,
    '',
  ];
  for (const item of result.interactions) {
    rows.push(`${item.pair.join(' + ')}: ${item.severity} (research label)`,
      `Method: ${item.classification_method}`,
      `Source: ${item.source_url || 'No linked source; evidence needs review'}`, '');
  }
  rows.push('Questions for a qualified reviewer:',
    '- Are the entered names and active ingredients correct, including combination products?',
    '- Does the linked source actually support the claim for this pair?',
    '- What additional information is needed before a clinical decision?',
    'Unknown or missing results mean insufficient evidence, not an absence of interactions.');
  return rows.join('\n');
}
