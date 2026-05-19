"""
benchmark_cache.py
──────────────────
Benchmarks the latency improvement of the cache.
"""
import time
from fastapi.testclient import TestClient
from api.server import app

def benchmark():
    payload = {"drugs": ["warfarin", "ibuprofen"]}
    
    print("Starting Drug Interaction AI Benchmark...")
    print("Initializing models and vector store (this takes a moment)...")
    
    with TestClient(app) as client:
        # First request (sets cache)
        print("\n--- First run (Cache Miss) ---")
        t0 = time.perf_counter()
        resp1 = client.post("/analyze_interaction", json=payload)
        t1 = time.perf_counter()
        
        if resp1.status_code != 200:
            print(f"API Error {resp1.status_code}: {resp1.text}")
            return
            
        ms1 = (t1 - t0) * 1000
        time_reported1 = resp1.json().get('processing_time_ms', 0)
        
        print(f"API Status      : {resp1.status_code}")
        print(f"Backend reported: {time_reported1:.1f} ms")
        print(f"Total Latency   : {ms1:.1f} ms")
        
        # Second request (gets cache)
        print("\n--- Second run (Cache Hit) ---")
        t0 = time.perf_counter()
        resp2 = client.post("/analyze_interaction", json=payload)
        t1 = time.perf_counter()
        
        ms2 = (t1 - t0) * 1000
        time_reported2 = resp2.json().get('processing_time_ms', 0)
        
        print(f"API Status      : {resp2.status_code}")
        print(f"Backend reported: {time_reported2:.1f} ms")
        print(f"Total Latency   : {ms2:.1f} ms")
        
        if ms2 < ms1:
            print(f"\n✅ Cache successfully accelerated response by {ms1/ms2:.1f}x!")
        else:
            print("\n❌ Cache did not improve latency.")

if __name__ == "__main__":
    benchmark()
