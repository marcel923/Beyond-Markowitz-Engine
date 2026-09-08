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
import ui.layout                # noqa: F401  (sets app.layout as an import side-effect)

if __name__ == "__main__":
    app.run(debug=True, port=8050)