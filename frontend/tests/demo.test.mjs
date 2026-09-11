import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { lookupFixtures, searchFixtures } from '../src/demo-core.mjs';
const data = JSON.parse(readFileSync(new URL('../src/demo-data.json', import.meta.url)));
test('known pair preserves fixture provenance without a probability claim', () => {
 const r = lookupFixtures(data, ['warfarin','ibuprofen']);
 assert.equal(r.interactions[0].classification_method, 'curated_fixture_lookup');
 assert.equal(r.interactions[0].confidence, 0);
 assert.match(r.interactions[0].source_name, /not clinically adjudicated/);
});
test('normalization, ordering and deduplication', () => {
 const a=lookupFixtures(data,['warfarin','ibuprofen']);
 const b=lookupFixtures(data,[' IBUPROFEN ', 'warfarin','warfarin']);
 assert.equal(a.overall_risk,b.overall_risk); assert.equal(b.interactions.length,1);
});
test('missing pair abstains and cannot imply low overall risk', () => {
 const r=lookupFixtures(data,['warfarin','ibuprofen','not-a-drug']);
 assert.equal(r.interactions.length,3); assert.equal(r.overall_risk,'Unknown');
 assert.equal(r.interactions.filter(p=>p.severity==='Unknown').length,2);
});
test('conflicting fixture labels abstain', () => {
 const d=[{drug_a:'a',drug_b:'b',severity:'Low'},{drug_a:'b',drug_b:'a',severity:'Severe'}];
 assert.equal(lookupFixtures(d,['a','b']).overall_risk,'Unknown');
});
test('validates input and protects source data', () => {
 const before=JSON.stringify(data); lookupFixtures(data,['warfarin','aspirin']); assert.equal(JSON.stringify(data),before);
 for(const d of [[],['warfarin'],['warfarin','warfarin'],[null,'a'],Array.from({length:11},(_,i)=>String(i))]) assert.throws(()=>lookupFixtures(data,d));
});
test('autocomplete uses only bundled vocabulary', () => {
 assert.ok(searchFixtures(data,' WAR ').includes('warfarin')); assert.deepEqual(searchFixtures(data,'zzyyxx'),[]);
});
