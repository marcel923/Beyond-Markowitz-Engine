"""
ui/theme.py
============
Shared visual constants: color palette, tab styling. Import THEME / TAB_STYLE
/ etc from here rather than redefining per-module.

Moved out of quant_terminal.py (Etap 0 architecture split) with NO behavior
change.
"""

THEME = {
    "bg_base": "#0B0B0E",       
    "bg_card": "#13131A",       
    "bg_input": "#1C1C24",      
    "border": "#222230",        
    "text_white": "#FFFFFF",    
    "text_dim": "#8A8A98",      
    "purple": "#6F2CFF",        
    "orange": "#FF8A00",        
    "font": "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
}

CHART_COLORS = ["#6F2CFF", "#FF8A00", "#00E5FF", "#FF3366", "#00FF66", "#FFD600", "#B200FF", "#FF6600"]
CLUSTER_PALETTE = {1: "#6F2CFF", 2: "#00E5FF", 3: "#FF8A00", 4: "#FF3366", 5: "#00FF66", 6: "#FFD600"}

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
    "marginBottom": "30px",
    "borderBottom": f"1px solid {THEME['border']}",
}

TAB_STYLE = {
    "backgroundColor": THEME["bg_card"],
    "color": THEME["text_dim"],
    "border": "none",
    "borderBottom": "3px solid transparent",
    "padding": "16px 22px",
    "fontWeight": "600",
    "fontSize": "13px",
    "letterSpacing": "0.02em",
    "fontFamily": THEME["font"],
}

TAB_SELECTED_STYLE = {
    "backgroundColor": THEME["bg_card"],
    "color": THEME["text_white"],
    "border": "none",
    "borderBottom": f"3px solid {THEME['purple']}",
    "padding": "16px 22px",
    "fontWeight": "700",
    "fontSize": "13px",
    "letterSpacing": "0.02em",
    "fontFamily": THEME["font"],
    "boxShadow": f"inset 0 -1px 0 0 {THEME['purple']}",
}

