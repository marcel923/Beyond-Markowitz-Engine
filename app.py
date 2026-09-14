"""
app.py
======
Top-level entrypoint. Run this file (not any module inside ui/) to start the
Quant Terminal.

Import order matters for exactly one reason: `ui.app_instance` must be
imported first so the shared `app` object exists before any tab module tries
to register a callback against it. After that, the 5 tab modules can import
in any order -- Python's module cache means `ui.tab4_rebalance`'s own
`from ui.tab3_tailrisk import compute_tail_penalty` will transparently import
tab3 first if it hasn't been already, so there is no need to hand-sequence
that dependency here. `ui.layout` is imported last purely for readability
(it assembles `app.layout` from pieces every tab module needs to already
have registered its component ids for).

Etap 0 architecture split (PROJECT_CONTEXT.md) -- replaces the old flat
quant_terminal.py. Run with: `python3 app.py`
"""
from ui.app_instance import app  # noqa: F401  (must be imported first)

import ui.tab1_market_data      # noqa: F401  (registers Stage 1 + Stage 2 callbacks)
import ui.tab2_fundamentals     # noqa: F401  (registers Stage 3 callbacks)
import ui.tab3_tailrisk         # noqa: F401  (registers Tail-Risk + Crash-Overlap callbacks)
import ui.tab4_rebalance        # noqa: F401  (registers Stage 4B solver callbacks)
import ui.tab5_sandbox          # noqa: F401  (registers Sandbox / Forward Tracker callbacks)
import ui.module_research       # noqa: F401  (registers Research / Company Dossier callbacks -- Etap 4)
import ui.module_relative_value # noqa: F401  (registers Relative Value pair-screener callbacks -- Etap 6)
import ui.layout                # noqa: F401  (sets app.layout as an import side-effect)

if __name__ == "__main__":
    # use_reloader=False, confirmed fix (2026-09-08, osiemnasty follow-up): debug=True
    # enables Werkzeug's auto-reloader by default, which spawns a SEPARATE monitor
    # process on top of the actual app process -- this is a documented source of
    # background-callback failures with DiskcacheManager (Etap 7q): the multiprocessing
    # worker can end up talking to the wrong process, so progress updates (and
    # sometimes the whole background task) never make it back to the browser at all,
    # with NO error shown -- confirmed root cause of "wlaczylem analize, zaden pasek
    # sie nie pojawil". debug=True itself (error pages, callback exception detail)
    # is kept; only the reloader is disabled. Restart the server manually after
    # editing code while this is off, since it will no longer auto-restart.
    app.run(debug=True, use_reloader=False, port=8050)