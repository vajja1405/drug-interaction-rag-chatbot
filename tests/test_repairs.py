import asyncio
import pytest
from cache_identity import evidence_version, pair_cache_key
from cache import LocalLRUCache

def test_cache_identity_tracks_evidence_filters_and_generation_settings():
 a=evidence_version(['text'],[{'source':'a'}]); b=evidence_version(['text'],[{'source':'b'}])
 key=lambda drugs=['a','b'], evidence=a, model='m', temp=0, tokens=20, low=True: pair_cache_key(drugs,evidence,model,temp,tokens,low)
 assert key()==key([' B ','a'])
 assert len({key(),key(evidence=b),key(model='other'),key(temp=1),key(tokens=30),key(low=False)})==6

def test_local_cache_expires_and_evicts(monkeypatch):
 clock=[0]; monkeypatch.setattr('cache.time.monotonic',lambda:clock[0])
 async def run():
  c=LocalLRUCache(capacity=2,ttl=5)
  await c.set('a','1'); await c.set('b','2'); assert await c.get('a')=='1'
  await c.set('c','3'); assert await c.get('b') is None
  clock[0]=5; assert await c.get('a') is None; assert await c.get('c') is None
  assert not c._expires
 asyncio.run(run())

def test_retired_interaction_endpoint_is_never_called():
 from data_pipeline.fetch_drug_data import get_rxnorm_interactions
 class NoNetwork:
  def get(self,*a,**k): raise AssertionError('Retired endpoint called')
 assert asyncio.run(get_rxnorm_interactions(NoNetwork(),['1','2']))==[]

def test_calibration_fits_features_inside_each_fold():
 from models.severity_classifier import RandomForestSeverityClassifier
 m=RandomForestSeverityClassifier(n_estimators=2,max_depth=3,min_samples_leaf=2)
 calibrated=m._build_pipeline()
 assert calibrated.method=='isotonic' and calibrated.cv==3
 assert calibrated.estimator.named_steps['rf'].max_depth==3
 assert calibrated.estimator.named_steps['rf'].min_samples_leaf==2
 # Unique tokens deliberately confined to different folds must not leak into every fitted vocabulary.
 samples=[('a','b',f'unique{chr(97+i)} common {i}') for i in range(12)]
 calibrated.fit(samples,[0,1]*6)
 vocabs=[set(c.estimator.named_steps['features'].transformer_list[0][1].named_steps['tfidf'].vocabulary_) for c in calibrated.calibrated_classifiers_]
 assert len({frozenset(v) for v in vocabs})>1

def test_api_body_and_cache_request_identity(client):
 p={'drugs':['warfarin','ibuprofen']}
 a=client.post('/analyze_interaction',json=p);b=client.post('/analyze_interaction',json=p)
 assert a.status_code==b.status_code==200
 assert a.json()['request_id']!=b.json()['request_id']
 assert client.post('/analyze_interaction',json={'drugs':['warfarin']}).status_code==422
