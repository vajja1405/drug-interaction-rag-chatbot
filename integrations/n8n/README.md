# n8n: pharmacist escalation routing

When the clinical agent escalates a case, `request_human_review` opens a ticket and, if
`AGENT_ESCALATION_WEBHOOK` is set, POSTs it to this n8n workflow. The workflow routes on priority:

```
Webhook (POST /webhook/pharmacist-escalation)
  └─ Urgent? ── yes ─▶ Page on-call pharmacist  ─▶ respond {"routed": "urgent"}
              └ no ──▶ Add to review digest      ─▶ respond {"routed": "routine"}
```

Both notification steps are HTTP requests to `N8N_NOTIFY_URL` (a Slack or Teams incoming-webhook URL
in a real deployment; a local receiver in the check below). Only the ticket id, case id, priority and
escalation reasons leave the agent; no patient data or findings are sent. A failed notification is
recorded on the ticket (`notified.ok = false`) and never blocks the review itself.

Verified on 2026-09-28 against n8n 2.40.7 on Node 24 (n8n 2.40 needs Node 24+; its `isolated-vm`
dependency does not build on Node 25): both routes delivered, see `e2e-2026-09-28.json`.

Run it locally:

```bash
npm install n8n@2.40.7                       # Node 24
npx n8n import:workflow --input=integrations/n8n/escalation-routing.json
npx n8n publish:workflow --id=PharmEscRouting01
N8N_NOTIFY_URL=http://127.0.0.1:8765/notify N8N_BLOCK_ENV_ACCESS_IN_NODE=false npx n8n start
python integrations/n8n/e2e_check.py          # in another shell
export AGENT_ESCALATION_WEBHOOK=http://localhost:5678/webhook/pharmacist-escalation   # for the API
```

CI tests the agent side against a stand-in webhook (`tests/test_clinical_agent.py`); the workflow itself
is checked with `e2e_check.py` against a running n8n.
