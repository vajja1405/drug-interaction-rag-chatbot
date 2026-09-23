import test from 'node:test';
import assert from 'node:assert/strict';
import { reviewSummary } from '../src/review-summary.mjs';
test('handoff preserves unknown evidence and source provenance without claiming clearance', () => {
 const text=reviewSummary({drugs:['a','b'], request_id:'r1', evidence_version:'v2',
 pairs_checked:['a|b'], interactions:[{pair:['a','b'],severity:'Unknown',classification_method:'abstain'}]});
 assert.match(text,/v2/); assert.match(text,/No linked source/); assert.match(text,/not an absence/);
 assert.match(text,/active ingredients/); assert.match(text,/r1/);
});
