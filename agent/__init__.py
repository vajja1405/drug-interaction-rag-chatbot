"""
Stateful clinical-evidence agent (LangGraph).

plan → resolve medications → retrieve label evidence → validate → propose → critique → judge
     ├─ REVISE   → loop back to propose (bounded self-correction)
     ├─ ESCALATE → human-review checkpoint (interrupt/resume) → finalize
     └─ PASS     → finalize (structured, cited, audited)
"""
