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
"""
import base64
import dash

dark_css = """
.Select-control, .Select, .Select-multi-value-wrapper {
    background-color: #1C1C24 !important;
    color: #FFFFFF !important;
    border: 1px solid #222230 !important;
    border-radius: 12px !important;
    height: 38px !important;
}
.Select-placeholder {
    color: #8A8A98 !important;
    line-height: 38px !important;
}
.Select-value-label, .Select-value {
    color: #FFFFFF !important;
    background-color: #2F2F3E !important;
    border: 1px solid #3F3F52 !important;
    border-radius: 6px !important;
    line-height: 28px !important;
}
.Select--multi .Select-value {
    background-color: #13131A !important;
    border: 1px solid #6F2CFF !important;
}
.Select-menu-outer, .Select-menu, .VirtualizedSelectOption {
    background-color: #1C1C24 !important;
    color: #FFFFFF !important;
    border: 1px solid #222230 !important;
}
.VirtualizedSelectFocusedOption {
    background-color: #2F2F3E !important;
    color: #FFFFFF !important;
}
.data-inspector-panel {
    position: fixed;
    top: 0;
    right: -850px;
    width: 800px;
    height: 100vh;
    background-color: #13131A;
    border-left: 1px solid #222230;
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
                background-color: #161B22 !important;
                color: #FFFFFF !important;
                border-color: #30363D !important;
            }
            .dark-dropdown .Select-placeholder, .dark-dropdown .Select-value-label, .dark-dropdown input,
            .dark-dropdown div[class*="-singleValue"], .dark-dropdown div[class*="-placeholder"] {
                color: #FFFFFF !important;
            }

            /* Ciemny motyw dla dcc.Slider (rc-slider) -- domyślne style renderują tooltip
               i opisy na jasnym tle, niewidoczne na ciemnym tle terminala. */
            .rc-slider-rail { background-color: #262630 !important; }
            .rc-slider-track { background-color: #6F2CFF !important; }
            .rc-slider-handle {
                background-color: #6F2CFF !important;
                border-color: #6F2CFF !important;
                opacity: 1 !important;
            }
            .rc-slider-handle:hover, .rc-slider-handle:focus, .rc-slider-handle-active {
                border-color: #FF8A00 !important;
                box-shadow: 0 0 0 4px rgba(255,138,0,0.2) !important;
            }
            .rc-slider-dot { background-color: #1C1C24 !important; border-color: #30363D !important; }
            .rc-slider-dot-active { border-color: #6F2CFF !important; }
            .rc-slider-mark-text { color: #8A8A98 !important; }
            .rc-slider-mark-text-active { color: #FFFFFF !important; }
            .rc-slider-tooltip { z-index: 9999 !important; }
            .rc-slider-tooltip-inner {
                background-color: #6F2CFF !important;
                color: #FFFFFF !important;
                border-radius: 6px !important;
                box-shadow: none !important;
                font-weight: 600 !important;
            }
            .rc-slider-tooltip-arrow { border-top-color: #6F2CFF !important; }
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

