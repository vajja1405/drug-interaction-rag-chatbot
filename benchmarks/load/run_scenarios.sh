#!/bin/zsh
# Runs the load-test matrix against a local API and writes docs/benchmarks/load-<date>.json.
# usage: benchmarks/load/run_scenarios.sh <venv-bin-dir>
set -e
BIN=${1:-.venv/bin}; ROOT=${0:A:h:h:h}; OUT=$ROOT/docs/benchmarks; TMP=$(mktemp -d)
start_api() {  # $1 = rerank depth
  pkill -f "uvicorn api.server:app" 2>/dev/null || true; sleep 2
  (cd $ROOT && OPENAI_API_KEY=sk-placeholder TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 EVIDENCE_RERANK_DEPTH=$1 \
    $BIN/uvicorn api.server:app --host 127.0.0.1 --port 8000 --workers 1 > $TMP/api.log 2>&1 &)
  for i in {1..90}; do curl -s -m 2 127.0.0.1:8000/api/v1/health >/dev/null && break; sleep 2; done
  curl -s "127.0.0.1:8000/api/v3/evidence/search?q=warmup%20warfarin%20aspirin" >/dev/null
}
run() {  # $1 name  $2 users  $3 repeat share
  (cd $ROOT && REPEAT_SHARE=$3 $BIN/locust -f benchmarks/load/locustfile.py --headless -H http://127.0.0.1:8000 \
     -u $2 -r $2 -t 45s --csv $TMP/$1 --only-summary > /dev/null 2>&1)
  echo "$1,$2,$3,$(grep ',evidence_search,' $TMP/$1_stats.csv)" >> $TMP/all.csv
}
start_api 30; run depth30_u10 10 0; run depth30_u25 25 0
start_api 10; run depth10_u10 10 0; run depth10_u25 25 0; run depth10_u25_cache50 25 0.5
pkill -f "uvicorn api.server:app" || true
python3 - "$TMP/all.csv" "$OUT" <<'PY'
import csv, json, sys, datetime
rows = []
for r in csv.reader(open(sys.argv[1])):
    name, users, repeat = r[0], int(r[1]), float(r[2]); s = r[3:]
    # Locust stats columns: Type,Name,Request Count,Failure Count,Median,Average,Min,Max,Avg size,RPS,Failures/s,50%,66%,75%,80%,90%,95%,98%,99%,...
    rows.append({"scenario": name, "users": users, "repeat_share": repeat, "requests": int(s[2]), "failures": int(s[3]),
                 "rps": round(float(s[9]), 1), "p50_ms": float(s[11]), "p95_ms": float(s[16]), "p99_ms": float(s[18])})
out = {"date": str(datetime.date.today()), "endpoint": "/api/v3/evidence/search (hybrid_rerank, k=5)",
       "setup": "1 uvicorn worker, 4 CPU threads, Apple M3 laptop; Locust on the same machine; 45 s per scenario",
       "rows": rows}
path = f"{sys.argv[2]}/load-{out['date']}.json"; open(path, "w").write(json.dumps(out, indent=2)); print(json.dumps(rows, indent=1))
PY
