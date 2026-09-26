# PROJECT_CONTEXT_2.md -- kontynuacja dziennika (od Etap 8d)

Ten plik jest KONTYNUACJĄ `PROJECT_CONTEXT.md` (Etap 1 -- Etap 8c), nie
nowym, niezależnym dokumentem. `PROJECT_CONTEXT.md` zrobił się bardzo długi,
więc od 2026-09-26 chronologiczny dziennik decyzji jedzie dalej tutaj.
Numeracja Etapów jest WSPÓLNA i ciągła między obydwoma plikami -- nie
zaczyna się od nowa. `PROJECT_CONTEXT.md` sam zostaje nietknięty i
zamrożony jako archiwum od tego momentu.

Te same zasady co w `CLAUDE.md`/`PROJECT_CONTEXT.md` obowiązują tu bez
zmian: append-only, NIE edytować istniejących wpisów, tylko dopisywać nowe
na końcu; przed dopisaniem większej, potwierdzonej zmiany -- pokazać
projekt wpisu i czekać na potwierdzenie.

---

### 8d. SAVE PORTFOLIO zawsze liczy i zapisuje wynik nakładki RV (complete)

Confirmed follow-up: adresuje priorytet #1 ("GŁÓWNY") z listy "Otwarte
priorytety" w `CLAUDE.md` -- nakładka RV nie trafiała do zapisu snapshotu.
Wymóg właściciela projektu: `SAVE PORTFOLIO` ma liczyć i zapisywać wynik
nakładki ZAWSZE, niezależnie od tego, czy w danej sesji kliknięto
"ZNAJDŹ PARY" -- ale tak, żeby Sandbox nadal mógł rozróżnić snapshoty z tą
nakładką od starszych, które jej nie mają. Jeśli "ZNAJDŹ PARY" nie zostało
kliknięte, zapis ma i tak policzyć nakładkę od zera z domyślną
theta=0.4, jawnie oznaczoną jako tymczasowa/niewalidowana do czasu przyszłej
kalibracji (priorytet #2 tamtej listy).

**Nowe funkcje w `engine/pairs.py`** (dopisane po istniejącym
`apply_post_solver_pair_overlay`, nic w nim nie zmienione):
- `DEFAULT_UNREVIEWED_THETA = 0.4` -- placeholder, patrz wyżej.
- `build_rv_overlay_record(weights, match_data, theta, theta_source)` --
  jedyne miejsce, które scala t-statystykę (dostępną tylko w
  `matched_pairs_info`) z wagami przed/po (dostępnymi tylko w
  `applied_pairs` z `apply_tilts_to_matched_pairs`) w jeden rekord --
  świadoma decyzja, żeby ta logika scalania nie duplikowała się w
  `tab4_rebalance.py` i `tab5_sandbox.py`. `theta_source` musi być
  `"user_reviewed"` albo `"default_unreviewed"` (walidowane, inaczej
  `ValueError`).
- `rv_overlay_unavailable(reason)` -- rekord dla przypadku, gdy liczenie
  nakładki się nie powiodło albo zostało pominięte (np. <2 tickery z
  dodatnią wagą, błąd pobierania cen) -- portfel i tak zapisuje się
  poprawnie, tylko nakładka jest oznaczona jako nieobliczona z podanym
  powodem.
- `apply_rv_overlay_weights(final_weights, rv_overlay)` -- czysty merge
  słownikowy, zero ponownego liczenia: rekonstruuje wagi po nakładce z
  `final_weights` + `rv_overlay["matched_pairs"]`.

**Format zapisu (`data/snapshot_store.py`)**: `save_snapshot()` przyjmuje
nowy opcjonalny parametr `rv_overlay: Optional[dict] = None`. Klucz
`"rv_overlay"` trafia do rekordu WYŁĄCZNIE gdy przekazany (nigdy jako
domyślny placeholder-dict) -- stare wywołania produkują bajtowo identyczne
rekordy jak przed tą zmianą. Trzy rozróżnialne stany, w tej kolejności
sprawdzane przez każdego czytelnika (Sandbox przede wszystkim):
1. Klucz w ogóle NIEOBECNY -> snapshot zapisany przed tą funkcją, traktuj
   jak `computed: False`.
2. `"computed": False` -> liczenie próbowane i nieudane/pominięte,
   `"reason"` opisuje czemu.
3. `"computed": True` -> `"matched_pairs"` może być pustą listą (to
   prawdziwy wynik, nie błąd); `"theta_source"` mówi, czy theta pochodzi z
   realnego przeglądu użytkownika czy z domyślnego placeholdera.

**`ui/tab4_rebalance.py`**: `save_snapshot_callback` przepisany na
`background=True` z `progress=[...]` (koszt liczenia nakładki od zera to
pełny, kosztowny two-stage theta-persistence gate + Maximum Weight
Matching -- per konwencja projektu, kosztowne operacje zawsze w tle z
raportowaniem postępu). Nowa logika cache-hit: jeśli zbiór tickerów w
`store-pair-overlay-match-cache` (populowany przez "ZNAJDŹ PARY") dokładnie
zgadza się z aktualnie wybranymi tickerami, użyj tego cache +
realnej thety ze slidera (`theta_source="user_reviewed"`); inaczej policz
`find_matched_pairs_for_overlay` od zera z `DEFAULT_UNREVIEWED_THETA`
(`theta_source="default_unreviewed"`), w `try/except` -> `rv_overlay_unavailable`
przy błędzie. Komunikat statusu po zapisie rozszerzony o wynik nakładki.

**`ui/tab5_sandbox.py`**: `update_forward_tracker` dostał odznakę w
`meta_info`, budowaną z `record.get("rv_overlay")`: "✓ ... (zweryfikowana)"
dla `user_reviewed`, "⚠ ... (domyślna, tymczasowa)" dla
`default_unreviewed`, "⚠ ... nie policzona (<reason>)" dla `computed: False`,
brak odznaki dla starych/nieobecnych zapisów -- wszystkie 6 punktów return
w tej funkcji dostają to automatycznie, bo używają tej samej zmiennej.

Zweryfikowane na syntetycznych danych przed wysłaniem: obie ścieżki
(cache-hit i cache-miss) produkują poprawny, zgodny ze schematem rekord;
stare snapshoty (bez klucza) nie wywołują błędu w Sandboxie. Solver
nietknięty -- zero zmian w `engine/optimizer.py`, `engine/returns.py`,
`engine/risk.py`. Zaimplementowane na branchu `rv-overlay-save-2026-09-26`,
PR: https://github.com/marcel923/Beyond-Markowitz-Engine/pull/1 (do
przeglądu/merge przez właściciela projektu -- nie zmergowane automatycznie).

---

### 8e. `QT_STORAGE_ROOT` -- bezpieczne sesje testowe bez ręcznego czyszczenia (complete)

Confirmed follow-up: właściciel projektu zgłosił obawę przed klikaniem po
aplikacji do testów, ponieważ realne kliknięcia (SAVE PORTFOLIO, "ZAPISZ DO
RESEARCH", dodanie do uniwersum) piszą prawdziwe pliki w
`storage/portfolio_snapshots/`, `storage/company_history/` i
`storage/universe.json`, które potem trzeba by ręcznie znajdować i usuwać.
Własna propozycja właściciela projektu (commit -> test-klikanie -> `git
pull` żeby "zrewertować") odrzucona jako rozwiązanie: nowe pliki snapshotów
są nieśledzone przez git (checkout/reset ich nie dotyka), a
`company_history`/`universe.json` mają z definicji narastać w czasie o
realne dane -- twardy reset ryzykowałby wycięciem prawdziwych wpisów razem
z testowymi, jeśli oba wylądowały w tym samym pliku między commitami.

**Rozwiązanie**: każdy z trzech modułów `data/` już przyjmował opcjonalny
override ścieżki na każdej publicznej funkcji (`storage_dir`/`base_dir`/
`file_path`) z domyślną wartością w jednej stałej modułowej
(`DEFAULT_STORAGE_DIR`, `DEFAULT_HISTORY_DIR`, `DEFAULT_FILE_PATH`). Ta
stała teraz sama rozwiązuje się ze zmiennej środowiskowej
`QT_STORAGE_ROOT` przy imporcie (`os.environ.get("QT_STORAGE_ROOT",
"storage")`), niezależnie w każdym module (zero nowego importu
międzymodułowego, zachowana istniejąca zasada braku zależności
między `data/snapshot_store.py`, `data/company_store.py`,
`data/universe_store.py`). Bez ustawienia zmiennej -- zero zmiany
zachowania, dokładnie `"storage"` jak wcześniej.

Sesja testowa: `QT_STORAGE_ROOT=storage_test python3 app.py` albo nowy
`scripts/run_test_session.sh` -- każdy zapis idzie do `storage_test/`,
realny `storage/` nigdy nie jest dotykany. Po sesji: `rm -rf storage_test`
(katalog już w `.gitignore`, więc nawet nieusunięty nie trafi do repo).
Żadnego commitowania/pullowania dla samego testowania.

Zweryfikowane: import wszystkich trzech modułów z ustawioną i nieustawioną
zmienną daje odpowiednio `storage_test/...` i `storage/...`; `py_compile`
czysty. Solver i silnik nietknięte -- zmiana wyłącznie w `data/` + `.gitignore`
+ nowy skrypt pomocniczy. Niezależne od Etap 8d -- zaimplementowane na
osobnym branchu `test-storage-isolation-2026-09-26` (od `main`, nie od
`rv-overlay-save-2026-09-26`), PR:
https://github.com/marcel923/Beyond-Markowitz-Engine/pull/2.

---
