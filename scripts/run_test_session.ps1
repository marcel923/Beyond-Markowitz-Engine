# scripts/run_test_session.ps1
#
# Windows/PowerShell wersja run_test_session.sh -- ten sam efekt: wszystkie
# zapisy (SAVE PORTFOLIO, company_history, universe.json) ida do jednorazowego
# katalogu storage_test/ zamiast do prawdziwego storage/ (QT_STORAGE_ROOT,
# patrz data/snapshot_store.py / company_store.py / universe_store.py).
#
# Uzycie (z aktywnym venv, z korzenia repo):
#   .\scripts\run_test_session.ps1
#
# Po sesji testowej: usun katalog recznie albo:
#   Remove-Item -Recurse -Force storage_test

$env:QT_STORAGE_ROOT = "storage_test"
Write-Host "== Sesja testowa: zapisy ida do '$($env:QT_STORAGE_ROOT)/', nie do 'storage/' =="
Write-Host "== Po skonczeniu: Remove-Item -Recurse -Force $($env:QT_STORAGE_ROOT) =="
Write-Host ""

python3 app.py
