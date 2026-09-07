"""
ui/theme.py
============
Shared visual constants: color palette, tab styling. Import THEME / TAB_STYLE
/ etc from here rather than redefining per-module.

Design direction (2026-09-07 conversation): "Institutional, softened" --
navy-tinted charcoal (not pure black) rejected the earlier "vibrant SaaS"
palette (8-color rainbow CLUSTER_PALETTE, mixed purple/orange usage for
unrelated concepts) in favor of an IBKR/Bloomberg-adjacent look: dense,
grid-ruled tables, ONE functional accent color, and colors that carry
meaning (green=gain, red=loss, amber=warning) rather than decoration. Also
explicitly rejected a harsher all-monospace/zero-radius "raw trading
workstation" first pass as too severe -- this is the settled middle ground:
normal sans-serif everywhere (tabular-nums for numeric alignment instead of
a monospace typeface swap), a small 6px radius on panels (not 0, not 16px),
and more breathing room in padding than a bare terminal grid.
"""

THEME = {
    "bg_base": "#0B0E14",        # page background -- navy-tinted charcoal, not pure black
    "bg_card": "#10141C",        # panel/card/tab-bar background (kept key name "bg_card" --
                                  # "card" isn't semantically wrong for a panel, unlike "purple"
                                  # below, which WOULD be wrong once it points at blue)
    "bg_head": "#0D1017",        # toolbar/header bar -- near bg_card but distinct, for the
                                  # topbar/section-label strips borrowed from the terminal mockup
    "bg_row_alt": "#12161F",     # alternating data-table row shading
    "bg_input": "#161B24",       # input fields, sliders, dropdowns -- slightly lighter than bg_card
    "border": "#1E2530",
    "border_strong": "#2C3644",  # heavier divider -- table header underline, tab bar bottom edge
    "text_white": "#E8EAED",     # kept key name for backward compat; value softened from pure
                                  # #FFFFFF to an off-white (pure white against a tinted dark bg
                                  # reads slightly harsh/glowing)
    "text_dim": "#7A8290",
    "text_label": "#5B6472",     # extra-muted -- tiny caps labels, table headers, KPI captions
    "accent": "#2E86FF",         # single functional accent -- vivid institutional blue (IBKR-strength,
                                  # not muted). Replaces the old "purple" key -- renamed, not just
                                  # re-colored, since a variable literally named "purple" holding a
                                  # blue hex value would be actively misleading in every file that
                                  # imports it. CORRECTION (2026-09-07 follow-up): an earlier pass
                                  # muted this to "#5B9FEF" while also softening structure (radius,
                                  # spacing) -- conflating the two was wrong. Structure should be
                                  # soft; semantic/functional colors must stay vivid so they read at
                                  # a glance, which is the entire point of color-coding a data terminal.
    "pos": "#00C853",            # semantic positive/gain green -- vivid, IBKR-strength saturation.
    "neg": "#FF3B30",            # semantic negative/loss red -- vivid, IBKR-strength saturation.
    "warn": "#FFB300",           # semantic warning/attention amber -- vivid gold, matches IBKR's
                                  # amber/yellow used for "SIMULATED TRADING" banners and caution flags.
    "orange": "#FFB300",         # DEPRECATED alias for "warn", same value -- kept only so any
                                  # call site not yet migrated to THEME["warn"] still renders
                                  # correctly instead of KeyError'ing. New code should use "warn".
    "font": "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif",
}

# Diverging blue-red colorscale for correlation/crash-overlap matrices (Tab 1 dendrogram heatmap,
# Tab 3 Jaccard/K heatmaps). Correction (2026-09-07 follow-up): an earlier pass used an
# amber-to-navy-to-blue scale here, which reads as arbitrary on a correlation matrix -- readers
# expect the conventional finance/statistics convention of red (negative) <-> blue (positive),
# same hues as THEME["neg"]/THEME["accent"] so it's visually consistent with the rest of the app.
# Gentle DIVERGING colorscale for true correlation-style matrices (data range -1..+1):
# red @ -1, neutral/background @ 0, blue @ +1. "Gentle" means the endpoints are blended
# partway toward the dark background rather than hitting full-saturation THEME["neg"]/
# THEME["accent"] -- on a dark theme, "soft like Excel's pastel 3-color-scale" means
# bringing the color CLOSER to the dark background, not lighter/whiter (which would
# fight the dark theme). Correction (2026-09-07, second follow-up): an earlier pass used
# full-strength THEME["neg"]/THEME["accent"] here, which read as harsh/oversaturated
# once numeric value labels were added on top of the cells.
MATRIX_COLORSCALE = [[0.0, "#7C2625"], [0.5, "#10141C"], [1.0, "#1E4782"]]

# Gentle SEQUENTIAL colorscale for non-negative similarity/coupling matrices (data range
# 0..max) -- the Discrete Crash-Overlap Matrix J/K (Section 3.3) and any distance metric
# that never goes negative. These have no meaningful "negative" pole, so there is
# deliberately no red here: 0 (no overlap) maps to the neutral background, same as the
# diverging scale's midpoint, and the maximum observed value maps to the same gentle blue
# used above -- one consistent "high value" color across every matrix in the app.
MATRIX_SEQUENTIAL_COLORSCALE = [[0.0, "#10141C"], [1.0, "#1E4782"]]

# Vivid, maximally-distinguishable qualitative palette for per-ticker line differentiation
# (Tab 1 multi-asset overlay chart) and cluster-group color-coding (dendrogram, donut,
# legend). Correction (2026-09-07, second follow-up): the previous 6-color muted set
# (all mid-tone blue/teal/amber/brown) was fine for a handful of dendrogram cells filling
# continuous space, but was NOT enough distinct hues for an 8+ ticker line-overlay chart --
# several lines rendered in near-identical shades. Expanded to 10 genuinely distinct vivid
# hues; CLUSTER_PALETTE reuses the first 6 for consistency with the overlay chart's colors
# rather than maintaining two independently-tuned palettes that could drift apart.
CHART_COLORS = ["#2E86FF", "#B983FF", "#FFB300", "#00D4D4", "#FF8A65", "#8BC34A", "#FF6EC7", "#64B5F6", "#00C853", "#FF3B30"]
CLUSTER_PALETTE = {i + 1: c for i, c in enumerate(CHART_COLORS[:6])}

# Muted, low-saturation multi-hue set for cluster differentiation (Tab 1 dendrogram/heatmap,
# Tab 4 donut chart). Replaces the old 8-color CHART_COLORS/CLUSTER_PALETTE rainbow (vivid
# purple/orange/cyan/red/green/yellow/magenta/orange) -- exactly the "vibrant, hard to execute
# consistently" look this redesign moved away from. Four muted variants is enough to visually
# tell clusters apart without competing with the semantic pos/neg/warn colors used elsewhere.

# ---------------------------------------------------------------------------
# Style dla głównych zakładek (dcc.Tabs / dcc.Tab) -- przekazywane bezpośrednio
# przez oficjalne propsy `style` / `selected_style`, a nie przez CSS klasy.
# Domyślny wygląd dcc.Tab to biały pasek z czarnym tekstem, co na ciemnym
# tle terminala dawało efekt "białe na białym". Ponieważ to propsy komponentu,
# a nie wewnętrzne nazwy klas DOM, to podejście przetrwa też ewentualne
# przyszłe zmiany w bibliotece dcc (w przeciwieństwie do CSS celującego
# w klasy takie jak .tab / .tab--selected, które mogą się zmieniać między
# wersjami Dash -- patrz notatka o dash<4.0 w requirements.txt).
# ---------------------------------------------------------------------------
TABS_CONTAINER_STYLE = {
    "marginBottom": "24px",
    "borderBottom": f"1px solid {THEME['border_strong']}",
}

TAB_STYLE = {
    "backgroundColor": THEME["bg_card"],
    "color": THEME["text_dim"],
    "border": "none",
    "borderRight": f"1px solid {THEME['border']}",
    "borderBottom": "2px solid transparent",
    "padding": "13px 20px",
    "fontWeight": "600",
    "fontSize": "12px",
    "letterSpacing": "0.01em",
    "fontFamily": THEME["font"],
}

TAB_SELECTED_STYLE = {
    "backgroundColor": THEME["bg_base"],
    "color": THEME["text_white"],
    "border": "none",
    "borderRight": f"1px solid {THEME['border']}",
    "borderBottom": f"2px solid {THEME['accent']}",
    "padding": "13px 20px",
    "fontWeight": "600",
    "fontSize": "12px",
    "letterSpacing": "0.01em",
    "fontFamily": THEME["font"],
    "boxShadow": f"inset 0 -2px 0 0 {THEME['accent']}",
}