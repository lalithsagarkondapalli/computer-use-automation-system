"""Human-in-the-loop control-transfer mechanism.

State machine (AUTOMATION_CONTROL -> PAUSED_FOR_HUMAN -> HUMAN_CONTROL ->
RESUME_REQUESTED -> AUTOMATION_CONTROL) plus a minimal operator console
that lets a human act on the *same* live browser session automation was
using. Implemented in Phase 5.
"""
