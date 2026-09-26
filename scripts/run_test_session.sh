#!/usr/bin/env bash
# scripts/run_test_session.sh
#
# Uruchamia aplikację ze wszystkimi zapisami (SAVE PORTFOLIO, company_history,
# universe.json) przekierowanymi do jednorazowego, wyrzucanego katalogu
# storage_test/ -- zamiast do prawdziwego storage/ (patrz QT_STORAGE_ROOT w
# data/snapshot_store.py, data/company_store.py, data/universe_store.py;
# potwierdzone 2026-09-26).
#
# Klikaj czym chcesz -- SAVE PORTFOLIO, ZAPISZ DO RESEARCH, cokolwiek -- nic
# nie trafi do prawdziwego storage/. Po zakończeniu sesji testowej po prostu
# usuń katalog:
#
#   rm -rf storage_test
#
# Żadnego commitowania/pullowania niczego dla samego testowania -- storage_test/
# jest już w .gitignore, więc nawet gdybyś go nie usunął, nic z niego nie
# trafi do repo przez przypadek.

set -euo pipefail
cd "$(dirname "$0")/.."

export QT_STORAGE_ROOT="storage_test"
echo "== Sesja testowa: zapisy idą do '${QT_STORAGE_ROOT}/', nie do 'storage/' =="
echo "== Po skończeniu: rm -rf ${QT_STORAGE_ROOT} =="
echo

python3 app.py
