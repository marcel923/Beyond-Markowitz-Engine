"""
ui/app_instance.py
===================
The single shared Dash `app` object. Every ui/tabN_*.py module imports `app`
from HERE (never creates its own) so that `@app.callback` decorators across
different files all register against the same callback map. This is the
standard Dash multi-file pattern -- must be imported and constructed BEFORE
any tab module registers its callbacks, so app.py imports this module first.

Moved out of quant_terminal.py (Etap 0 architecture split, PROJECT_CONTEXT.md)
with NO behavior change.

NOTE on the CSS below: both `dark_css` and `app.index_string` are PLAIN
(non-f) triple-quoted strings with hardcoded hex values, deliberately NOT
f-strings interpolating ui.theme.THEME. An earlier version of this file
tried the f-string approach and broke at runtime (not caught by
py_compile!): CSS rule braces like `.Select-control { ... }` collide with
Python's f-string `{expr}` syntax, since every literal brace in an f-string
must be escaped as `{{`/`}}` -- easy to get wrong across dozens of CSS
rules, so this file trades the "single source of truth" elegance for
correctness. If you update ui/theme.py's palette, update the matching hex
values here too (search for old values first, e.g. via the previous
palette's hex codes) -- this file's CSS is the one place they're allowed to
drift out of sync with THEME without immediately breaking; it just needs a
person to remember to update it.
"""
import base64
import dash

dark_css = """
.Select-control, .Select, .Select-multi-value-wrapper {
    background-color: #161B24 !important;
    color: #E8EAED !important;
    border: 1px solid #1E2530 !important;
    border-radius: 4px !important;
    height: 36px !important;
}
.Select-placeholder {
    color: #7A8290 !important;
    line-height: 36px !important;
}
.Select-value-label, .Select-value {
    color: #E8EAED !important;
    background-color: #10141C !important;
    border: 1px solid #2C3644 !important;
    border-radius: 3px !important;
    line-height: 26px !important;
}
.Select--multi .Select-value {
    background-color: #0B0E14 !important;
    border: 1px solid #5B9FEF !important;
}
.Select-menu-outer, .Select-menu, .VirtualizedSelectOption {
    background-color: #161B24 !important;
    color: #E8EAED !important;
    border: 1px solid #1E2530 !important;
}
.VirtualizedSelectFocusedOption {
    background-color: #10141C !important;
    color: #E8EAED !important;
}
.data-inspector-panel {
    position: fixed;
    top: 0;
    right: -850px;
    width: 800px;
    height: 100vh;
    background-color: #10141C;
    border-left: 1px solid #1E2530;
    box-shadow: -20px 0px 50px rgba(0,0,0,0.8);
    transition: right 0.4s ease-in-out;
    z-index: 9999;
    padding: 40px;
    box-sizing: border-box;
    overflow-y: auto;
}
.data-inspector-panel.open {
    right: 0;
}
"""
encoded_css = base64.b64encode(dark_css.encode('utf-8')).decode('utf-8')
external_stylesheets = [f"data:text/css;base64,{encoded_css}"]

app = dash.Dash(__name__, title="Premium Quant Dashboard", external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)

# Naprawa "białego kontrastu" w komponencie dcc.Dropdown w Tab 5 (Forward Tracker):
# react-select (silnik pod spodem) domyślnie renderuje kontrolkę i menu na jasnym tle,
# niezależnie od stylu wrappera -- trzeba to nadpisać osobnym CSS wstrzykniętym w <head>.
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .dark-dropdown .Select-control, .dark-dropdown .Select-menu-outer, .dark-dropdown .Select-menu,
            .dark-dropdown .VirtualizedSelectOption, .dark-dropdown .Select--single > .Select-control .Select-value,
            .dark-dropdown div[class*="-control"], .dark-dropdown div[class*="-menu"], .dark-dropdown div[class*="-option"] {
                background-color: #161B24 !important;
                color: #E8EAED !important;
                border-color: #2C3644 !important;
            }
            .dark-dropdown .Select-placeholder, .dark-dropdown .Select-value-label, .dark-dropdown input,
            .dark-dropdown div[class*="-singleValue"], .dark-dropdown div[class*="-placeholder"] {
                color: #E8EAED !important;
            }

            /* Ciemny motyw dla dcc.Slider (rc-slider) -- domyślne style renderują tooltip
               i opisy na jasnym tle, niewidoczne na ciemnym tle terminala. */
            .rc-slider-rail { background-color: #1E2530 !important; }
            .rc-slider-track { background-color: #5B9FEF !important; }
            .rc-slider-handle {
                background-color: #5B9FEF !important;
                border-color: #5B9FEF !important;
                opacity: 1 !important;
            }
            .rc-slider-handle:hover, .rc-slider-handle:focus, .rc-slider-handle-active {
                border-color: #D4A62E !important;
                box-shadow: 0 0 0 4px rgba(212,166,46,0.2) !important;
            }
            .rc-slider-dot { background-color: #161B24 !important; border-color: #2C3644 !important; }
            .rc-slider-dot-active { border-color: #5B9FEF !important; }
            .rc-slider-mark-text { color: #7A8290 !important; }
            .rc-slider-mark-text-active { color: #E8EAED !important; }
            .rc-slider-tooltip { z-index: 9999 !important; }
            .rc-slider-tooltip-inner {
                background-color: #5B9FEF !important;
                color: #0B0E14 !important;
                border-radius: 4px !important;
                box-shadow: none !important;
                font-weight: 600 !important;
            }
            .rc-slider-tooltip-arrow { border-top-color: #5B9FEF !important; }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''