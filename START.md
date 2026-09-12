# Beyond-Markowitz Quant Terminal — Setup Guide

A local Dash application for portfolio construction research, extending
mean-variance optimization with tail-risk-adjusted scoring, hierarchical
clustering, single-asset return projection, and statistical pair-relative
analysis. This guide covers installation and startup for reviewers running
the application locally.

## Prerequisites

- Python 3.10 or later
- A Python virtual environment (referred to below as `venv`)

## 1. Installation

1. Place the full project directory on disk. The application is organized
   as a package, not a single script — the following structure must be
   preserved as-is:

   ```
   quant-terminal/
   ├── app.py                  # Entry point
   ├── requirements.txt
   ├── engine/                 # Core quantitative logic (pure functions, no UI)
   ├── data/                   # Persistence and market data access
   ├── ui/                     # Dash layout and callbacks
   ├── storage/                # Runtime data (portfolios, universe, history)
   └── scripts/                # One-off maintenance/migration scripts
   ```

2. Activate the virtual environment:

   ```bash
   # Windows
   venv\Scripts\activate

   # macOS / Linux
   source venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## 2. Running the Application

From the project root, with the virtual environment active:

```bash
python3 app.py
```

The console will report:

```
Dash is running on http://127.0.0.1:8050/
```

Open this address in a browser. The application runs as a local process
with no external server dependency and no browser-IDE session timeouts.

## 3. Diagnosing Issues

Runtime errors are logged with full tracebacks to the terminal session
running `app.py`, not only summarized in the browser. When investigating
an issue, the terminal output — not the in-app error banner — is the
primary source of diagnostic information (exact file, line number, and
exception type).

## 4. Known Issues and Resolutions

| Symptom | Cause | Resolution |
|---|---|---|
| `YFRateLimitError`, or empty market data | Yahoo Finance is temporarily rate-limiting the client IP address | Wait several minutes before retrying; avoid repeated rapid refreshes |
| `JSONDecodeError` originating from `yfinance` | Upstream Yahoo Finance response format has changed; the installed `yfinance` version predates the change | `pip install --upgrade yfinance` |
| Sliders or dropdowns render unstyled despite custom CSS | `pip install` resolved Dash 4.x, which replaced the underlying `dcc.Slider` / `dcc.Dropdown` implementations (no longer built on `rc-slider` / `react-select`); the existing dark-theme CSS targets DOM classes that no longer exist | `requirements.txt` pins `dash<4.0`. Run `pip install -r requirements.txt --upgrade` inside the active virtual environment, then confirm with `python -c "import dash; print(dash.__version__)"` |
| `Address already in use` on startup | A prior Python process is still bound to the port | Terminate the existing process (Ctrl+C in its terminal), or change the port at the bottom of `app.py`: `app.run(debug=True, port=8060)` |

## 5. Feedback

When reporting an issue, please include the full terminal traceback rather
than a description of the on-screen symptom — this allows the underlying
cause to be addressed directly rather than inferred.