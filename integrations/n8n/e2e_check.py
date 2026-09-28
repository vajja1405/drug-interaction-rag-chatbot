"""
End-to-end check for the n8n escalation-routing workflow.

Starts a local receiver standing in for Slack/Teams, sends one urgent and one routine ticket to the
workflow's webhook through the agent's ToolRegistry, and verifies each landed in the right channel.

    N8N_NOTIFY_URL=http://127.0.0.1:8765/notify n8n start        # with the workflow imported and published
    python integrations/n8n/e2e_check.py --webhook http://localhost:5678/webhook/pharmacist-escalation
"""
from __future__ import annotations

import argparse
import http.server
import json
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--webhook", default="http://localhost:5678/webhook/pharmacist-escalation")
    ap.add_argument("--notify-port", type=int, default=8765)
    args = ap.parse_args()
    received: list[dict] = []

    class Receiver(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(200)
            self.end_headers()

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", args.notify_port), Receiver)
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    from agent.tools import ToolRegistry

    class Stub:  # the review tool needs no catalog or search
        passages, ingredients = [], {}

    tools = ToolRegistry(Stub(), Stub(), memory=None, escalation_webhook=args.webhook, label_drugs=set())
    urgent = tools.request_human_review("e2e-urgent", ["high_severity_interaction"], priority="urgent")
    routine = tools.request_human_review("e2e-routine", ["insufficient_evidence"], priority="routine")
    srv.shutdown()
    result = {
        "urgent_ticket": urgent["notified"], "routine_ticket": routine["notified"],
        "notifications": received,
        "passed": (urgent["notified"].get("route") == "urgent" and routine["notified"].get("route") == "routine"
                   and [r["channel"] for r in received] == ["pharmacist-urgent", "pharmacist-review-queue"]
                   and [r["case"] for r in received] == ["e2e-urgent", "e2e-routine"]),
    }
    print(json.dumps(result, indent=1))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
