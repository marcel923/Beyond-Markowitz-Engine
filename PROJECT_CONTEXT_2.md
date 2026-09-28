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

### 8f. Baseline regresji solvera z Etap 8a potwierdzony (complete)

Confirmed: właściciel projektu zatwierdził nowe wartości regresji solvera z
Etap 8a jako punkt odniesienia na przyszłość -- NVDA=0.35, AVGO=0.35,
HPE=0.2463, LYC.AX=0.0537 (tickery AVGO/NVDA/HPE/LYC.AX, seed=5). Poprzedni
baseline (HPE=0.246, LYC.AX=0.054, sprzed uniwersalnego dyskonta gamma na
mu_i) jest odtąd nieaktualny. `CLAUDE.md` zaktualizowany.

---

### 8g. Rozwój zakładki "ANALIZA SOBOLA" -- propozycja, w trakcie omawiania (open)

Właściciel projektu poprosił o propozycję rozwoju zakładki "ANALIZA SOBOLA"
(priorytet #3 z poprzedniej listy "Otwarte priorytety") -- trzy konkretne
kierunki, żadny jeszcze nie zaimplementowany, wszystkie pokazane właścicielowi
projektu do decyzji przed pisaniem kodu:

1. **Panel stabilności parametrów vs Sortino.** `run_sobol_batch()` w
   `engine/sobol_analysis.py` już liczy `raw_sample`/`raw_outputs` (pełna
   macierz N×18 próbek × 8 parametrów + CAGR/Sortino/MaxDrawdown per próbka)
   i trzyma je server-side (świadomie NIE wysyłane do przeglądarki przez
   `dcc.Store` -- zbyt duże na round-trip JSON), ale dziś te surowe dane nie
   są wizualizowane w ogóle poza zagregowanymi wskaźnikami Sobol/PRCC.
   Propozycja: siatka małych wykresów rozproszenia (8 paneli, po jednym na
   parametr) -- x = wartość parametru w tej próbce, y = Sortino tej próbki,
   z top ~10% próbek (najwyższy Sortino) podświetlonych innym kolorem na tle
   reszty wyszarzonej. Ciasny klaster podświetlonych punktów w wąskim
   zakresie danego parametru = stabilne, wiarygodne optimum; rozrzucone po
   całej osi = brak realnego sygnału (najlepszy wynik to szum próbki, nie
   coś do ufania). Zero dodatkowych wywołań solvera -- czysto serwerowa
   wizualizacja już policzonych danych, budowana w tym samym callbacku,
   który dziś je odrzuca po zbudowaniu wykresów S1/ST.

2. **R_f / hurdle rate: przemianowanie + opcjonalne auto-pobieranie.**
   Zweryfikowane w kodzie: matematycznie R_f i MAR Sortino są już
   poprawnie rozdzielone -- `engine/sobol_analysis.py` liczy
   `sortino = (mean_ann - 0.0) / downside_dev`, MAR na sztywno 0.0,
   NIGDY nie powiązane z przemiatanym parametrem `rf` (świadoma decyzja z
   Etap 8b, udokumentowana wprost w kodzie). Problem jest wyłącznie w
   etykiecie: `ui/components.py` (`STAGE4A_PARAMS_CONFIG`, pole "rf")
   nazywa to na karcie suwaka po prostu "Risk-Free Rate" -- mylące, skoro w
   `engine/optimizer.py` ten sam parametr jest jawnie "hurdle rate" w
   liczniku TPS (`(w.mu - Rf)/...`), nie stopa odniesienia dla Sortino.
   Propozycja: przemianować widoczną etykietę na "Stopa referencyjna /
   hurdle (R_f)" wszędzie gdzie się pojawia (karta parametru, suwak w
   Sandboxie), i dopisać krótkie zdanie na zakładce Sobol wprost mówiące,
   że Sortino liczy się względem MAR=0, niezależnie od ustawienia R_f.
   Co do auto-pobierania: w kodzie jest już komentarz "Ręczne wejście, brak
   automatycznego pobierania ^TNX (zgodnie z zasadą 'dane manualne')" --
   ale ta zasada nigdzie nie jest udokumentowana w PROJECT_CONTEXT/CLAUDE.md,
   więc nie zakładam jej uzasadnienia. Zweryfikowane: `^TNX` na Yahoo
   Finance (dokładnie to źródło, którego cała reszta projektu już używa
   przez `yfinance` w `data/market_data.py`) zwraca rentowność 10Y
   bezpośrednio w procentach (dziś ok. 5.22-5.23%, zgadza się z tym co
   podał właściciel projektu) -- żadnego przeliczania jednostek, żadnej
   nowej integracji. Otwarte pytanie do decyzji: (a) zostawić ręczne
   wejście jako domyślne, dodać obok przycisk "pobierz aktualną ^TNX" jako
   wygodę opcjonalną (nigdy nie nadpisuje cicho) -- najbliższe istniejącej
   zasadzie; czy (b) dla zakładki Sobol konkretnie, przy uruchamianiu
   analizy na starym zapisie, pobierać rentowność 10Y Z DNIA UTWORZENIA
   TEGO ZAPISU zamiast dzisiejszej -- poprawniejsze dla spójności
   walk-forward, ale wymaga historycznego szeregu ^TNX, nie tylko
   ostatniej ceny, więc trochę więcej pracy. Czeka na wybór właściciela
   projektu.

3. **Panel "Połączone Portfolio".** Potwierdzone wcześniej (przed Etap 8b),
   nigdy nie zbudowane -- właściciel projektu ręcznie wybiera i ustawia
   kolejność kilku zapisanych snapshotów, a śledzona krzywa equity po
   prostu PODMIENIA się na kolejny zapis w dniu jego utworzenia (żadnego
   wygładzania/mieszania między nimi). Celowo odseparowane od Sobola (Sobol
   nigdy nie skleja zapisów -- pomyliłoby to efekt parametru z efektem
   czasu rynkowego, patrz `engine/sobol_analysis.py` docstring). Propozycja:
   nowa zakładka-siostra obok "FORWARD TRACKER"/"ANALIZA SOBOLA" w
   Sandboxie -- lista zapisów z drag-to-reorder (albo prostszy numerowany
   wybór, jeśli drag-and-drop w Dash okaże się niepotrzebnie kosztowny),
   każdy z własną datą utworzenia jako punktem podmiany; renderowana krzywa
   equity to konkatenacja rzeczywistych, już zrealizowanych zwrotów forward
   każdego snapshotu między jego datą utworzenia a datą utworzenia
   następnego (albo dziś, dla ostatniego) -- bez interpolacji/wygładzania na
   styku, zgodnie z wcześniejszym ustaleniem.

Żadna z trzech rzeczy jeszcze nie zaimplementowana -- czeka na wybór
właściciela projektu (w szczególności co do (2b) auto-pobierania R_f) i
ogólne potwierdzenie zakresu przed pisaniem kodu.

---

### 8h. R_f przemianowany na hurdle rate + auto-pobieranie ^TNX w Sandboxie (complete)

Confirmed follow-up: realizuje punkt 2 propozycji z Etap 8g -- właściciel
projektu zatwierdził wariant (b) (auto-pobieranie historycznej rentowności
na dzień utworzenia zapisu, nie dzisiejszej) i poprosił o wdrożenie razem z
przemianowaniem etykiety.

**Przemianowanie**: `STAGE4A_PARAMS_CONFIG["rf"]["name"]` (`ui/components.py`)
z "Risk-Free Rate" na "Stopa referencyjna / Hurdle Rate" -- to jedno miejsce
zasila zarówno kartę parametru na Rebalance (Tab 4), jak i etykietę suwaka w
Sandboxie (`ui/layout.py`, `slider-sb-rf`). Dopisany explicit komentarz przy
tym wpisie configu, tłumaczący na stałe, że Sortino w `engine/sobol_analysis.py`
liczy się względem MAR=0.0, NIGDY niepowiązane z R_f (Etap 8b) -- żeby to
rozróżnienie nie zatarło się przy przyszłych zmianach.

**Auto-pobieranie**: nowa `fetch_treasury_yield_on_date(target_date)` w
`data/market_data.py` -- ten sam kanał yfinance co reszta projektu, pobiera
`^TNX` (CBOE 10Y Treasury Yield Index na Yahoo, notowany wprost w procentach
-- zweryfikowane) na/najbliżej przed podaną datą, zwraca jako ułamek albo
`None` przy dowolnym niepowodzeniu (nigdy nie rzuca wyjątku). Podłączone w
`ui/tab5_sandbox.py` w dwóch miejscach:
- Forward Tracker: wybór zapisu auto-wypełnia `slider-sb-rf` realną
  rentownością 10Y na dzień UTWORZENIA TEGO KONKRETNEGO ZAPISU (było
  statyczne 0.045) -- suwak zostaje suwakiem, można dalej ręcznie
  nadpisać; nieudane pobranie zostawia poprzednią wartość + pokazuje notkę
  zamiast cicho zostawić coś mylącego.
- Zakładka Sobol: ta sama historyczna rentowność pokazana jako czysto
  informacyjna podpowiedź przy wyborze zapisu -- CELOWO nie zmienia
  automatycznie zakresu przeszukiwania `sobol-range-rf-min/max` (zostaje
  szeroki, płaski `DEFAULT_PARAM_RANGES` z Etap 8g/8b -- osobna decyzja
  metodologiczna, właściciel projektu może ręcznie dostosować zakres
  patrząc na podpowiedź).

Główna zakładka Rebalance (`input-rf`) NIETKNIĘTA -- tam portfel powstaje
"dziś", ręczne wejście dzisiejszej wartości zostaje, zgodnie z ustaleniem.

Zweryfikowane: konwersja jednostek (%→ułamek) i fallback na najbliższy
wcześniejszy dzień sesyjny (weekend/święto) przetestowane na zamockowanej
odpowiedzi yfinance (live Yahoo niedostępne z tego środowiska); `py_compile`
czyste na wszystkich czterech dotkniętych plikach; pełny `import app`
przechodzi bez kolizji ID callbacków Dash. Solver i matematyka Sortino
nietknięte. PR: https://github.com/marcel923/Beyond-Markowitz-Engine/pull/5.

Pozostałe z Etap 8g wciąż otwarte: (1) panel stabilności parametrów vs
Sortino, (3) panel "Połączone Portfolio".

---
