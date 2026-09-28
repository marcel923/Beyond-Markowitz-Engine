#!/usr/bin/env python3
"""
scripts/check_duplicate_outputs.py

Zaobserwowane 2026-09-28: Dash NIE odrzuca dwóch @app.callback z tym samym
Output(id, prop) (bez allow_duplicate=True) ani przy imporcie modułu, ani
przy starcie serwera (app.run()) -- ta walidacja jest wyłącznie po stronie
przeglądarki (dash-renderer), więc `python3 -m py_compile` i nawet
`python3 -c "import app"` nigdy tego nie złapią. Objawia się dopiero jako
czerwony błąd w przeglądarce ("Output N (x.value) is already in use...").
Patrz PROJECT_CONTEXT_2.md -- incydent z ANALIZA SOBOLA/Sandbox tego samego
dnia (slider-sb-rf.value w dwóch callbackach naraz).

To ten sam audyt server-side, jaki wykonuje dash-renderer po stronie klienta,
odtworzony tutaj z app._callback_list, żeby złapać ten sam problem PRZED
uruchomieniem przeglądarki. Uruchom po KAŻDEJ zmianie dodającej/zmieniającej
@app.callback:

    python3 scripts/check_duplicate_outputs.py

Exit code 0 = brak konfliktów, 1 = znaleziono (i wypisane na stdout).
"""
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def find_duplicate_outputs():
    import app  # noqa: F401 -- import samo w sobie rejestruje wszystkie @app.callback
    from ui.app_instance import app as dash_app

    counts = defaultdict(lambda: {"total": 0, "no_allow_duplicate": 0})
    for entry in dash_app._callback_list:
        out_str = entry.get("output", "")
        inner = out_str.strip(".")
        targets = inner.split("...") if "..." in inner else [inner]
        for t in targets:
            if not t:
                continue
            # Dash appends "@<hash>" to the output id.prop string when that
            # specific Output was registered with allow_duplicate=True.
            has_allow_duplicate = "@" in t
            base = t.split("@")[0]
            counts[base]["total"] += 1
            if not has_allow_duplicate:
                counts[base]["no_allow_duplicate"] += 1

    # A real conflict is >1 callback claiming the same (id, prop) output
    # WITHOUT allow_duplicate=True -- Dash requires all but one of them to
    # set it explicitly.
    return {k: v for k, v in counts.items() if v["no_allow_duplicate"] > 1}, len(counts)


if __name__ == "__main__":
    conflicts, n_total = find_duplicate_outputs()
    if conflicts:
        print(f"DUPLICATE OUTPUT CONFLICTS FOUND ({len(conflicts)} z {n_total} unikalnych outputów):")
        for output_id, info in conflicts.items():
            print(f"  {output_id}: {info['total']} rejestracji, {info['no_allow_duplicate']} bez allow_duplicate=True")
        print("\nNapraw: dodaj allow_duplicate=True do wszystkich-oprócz-jednej rejestracji "
              "tego Output (i prevent_initial_call=True na WSZYSTKICH callbackach, które go dzielą).")
        sys.exit(1)
    else:
        print(f"OK -- brak konfliktów duplikatów wśród {n_total} unikalnych targetów Output.")
        sys.exit(0)
