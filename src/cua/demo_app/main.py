"""The local demo target: a small, deliberately-imperfect "legacy credit
union back office" search/results app (Phase 6). This is the SURFACE the
automation system operates on -- it never imports anything from `cua.*`
except its own `data` module, and nothing in `cua.replay`/`cua.agent`
imports this package back (a demo target is not part of the automation
system's dependency graph).

Deliberate legacy characteristics, per the Phase 6 brief:
  - table-based layout for the search form and result fields,
  - labels via `aria-label` rather than a wired-up `<label for>`,
  - no `data-testid` anywhere -- a reviewer/automation must target by
    role/accessible-name/text, the same way this repo's TargetDescriptor
    strategies are designed to work,
  - a simulated backend latency on every search,
  - a real validation state (a malformed member id bounces back to the
    search page with an inline error, not a 500),
  - a real not-found state,
  - a real unexpected interstitial (the Compliance Hold modal) that
    blocks the same page's content until a human dismisses it -- no
    server-side session/state involved, exactly like
    tests/fixtures/sample_page.html's spinner-toggle pattern already
    proven out in Phase 3/5.

Run directly with:
    uvicorn cua.demo_app.main:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from cua.demo_app.data import lookup

app = FastAPI(title="CU Ops Console (synthetic demo data only)")
_templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

# Simulated legacy backend latency on every search -- short enough not to
# slow the test suite, long enough to be a real, observable delay (and to
# exercise WAIT/timeout handling rather than everything resolving instantly).
_SIMULATED_LATENCY_S = 0.3


@app.get("/", response_class=HTMLResponse)
async def root() -> RedirectResponse:
    return RedirectResponse(url="/search")


@app.get("/search", response_class=HTMLResponse)
async def search_form(request: Request, error: str | None = None) -> HTMLResponse:
    return _templates.TemplateResponse(request, "search.html", {"error": error})


@app.get("/results", response_class=HTMLResponse)
async def results(request: Request, member_id: str = "") -> HTMLResponse:
    member_id = member_id.strip()
    if not (member_id.isdigit() and len(member_id) == 5):
        # Real validation state: bounce back to the search page with an
        # inline error rather than a 500 or a silent not-found.
        return RedirectResponse(url="/search?error=Enter+a+5-digit+member+ID")

    await asyncio.sleep(_SIMULATED_LATENCY_S)
    status, member = lookup(member_id)
    return _templates.TemplateResponse(
        request, "results.html", {"member_id": member_id, "status": status, "member": member}
    )
