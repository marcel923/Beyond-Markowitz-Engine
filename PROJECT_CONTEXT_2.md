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

### 8i. Panel stabilności parametrów vs Sortino w zakładce ANALIZA SOBOLA (complete)

Confirmed follow-up: realizuje punkt 1 propozycji z Etap 8g. Przed
implementacją pokazane właścicielowi projektu dokładne wyjaśnienie
mechaniki na dwóch konkretnych przykładach (parametr stabilny vs
niestabilny, z liczbami) -- zatwierdzone bez zmian co do koncepcji, z
trzema doprecyzowaniami: próg podświetlenia jako wybór z 4 opcji
(10/7.5/5/2.5%, nie suwak), tylko dla Sortino na razie (nie CAGR/MaxDrawdown),
i dopisanie liczbowego wskaźnika stabilności pod każdym wykresem.

**Mechanika**: dla każdego z 8 parametrów solvera, wykres rozproszenia
(wartość parametru w danym przebiegu Sobola, Sortino tego przebiegu) po
wszystkich przebiegach ostatniej analizy, z top N% (wybór: 10/7.5/5/2.5%,
`dcc.RadioItems`) podświetlonymi jednym kolorem na tle reszty wyszarzonej.
Pod każdym mini-wykresem liczbowy werdykt: zakres wartości parametru wśród
top N% jako % pełnej szerokości przemiatanego zakresu -- STABILNE (<25%,
zielone), UMIARKOWANE (25-60%, bursztynowe), NIESTABILNE (>60%, czerwone).
Jawnie udokumentowane w kodzie i w rozmowie z właścicielem projektu: to
pokazuje KORELACJĘ, nie przyczynowość (wszystkie 8 parametrów zmienia się
naraz w jednej próbce Saltelli) -- uzupełnienie istniejących S1/ST/PRCC,
nie ich zamiennik.

**Zero dodatkowych obliczeń solvera**: `run_sobol_batch` w
`engine/sobol_analysis.py` już liczy `raw_sample`/`raw_outputs` dla
każdego z N×18 przebiegów -- wcześniej używane tylko do zbudowania
zagregowanych wykresów S1/ST, potem odrzucane. Nowy moduł-poziomu cache
`_LAST_SOBOL_RAW` w `ui/tab5_sandbox.py` trzyma surowe dane ostatniego
przebiegu SERVER-SIDE (nigdy nie wysyłane do przeglądarki -- zgodnie z
własnym docstringiem `run_sobol_batch`, zbyt duże na round-trip JSON), więc
zmiana progu podświetlenia przerysowuje panel natychmiast przez osobny,
tani callback, bez ponownego uruchamiania analizy. Wiersze
`solver_success=False` są z tego panelu CAŁKOWICIE wykluczone (nie
median-imputowane jak w matematyce dekompozycji wariancji Sobola) -- sztuczny
punkt medianowy zniekształcałby wizualnie to, gdzie faktycznie klastrują
się zwycięskie przebiegi.

Zweryfikowane na syntetycznym zbiorze danych (Sortino skonstruowane tak,
żeby realnie zależało WYŁĄCZNIE od lambda, reszta 7 parametrów -- czysty
szum): lambda poprawnie wraca jako STABILNE z ciasnym zakresem top-10%
(14% pełnej szerokości), wszystkie pozostałe 7 parametrów poprawnie wraca
jako NIESTABILNE (~99-100% pełnej szerokości) -- potwierdza że mechanizm
faktycznie odróżnia realny sygnał od szumu, nie tylko wygląda sensownie.
Też zweryfikowane: przypadki brzegowe (mało poprawnych przebiegów, małe N),
`py_compile`, pełny `import app` bez kolizji ID callbacków. Solver i
matematyka Sortino nietknięte. PR:
https://github.com/marcel923/Beyond-Markowitz-Engine/pull/6.

Pozostały z Etap 8g: (3) panel "Połączone Portfolio".

---

### Etap 8j. CAGR-owy bliźniak panelu stabilności, "najlepsza kombinacja" i własne wartości w Sandboxie (complete)

Trzy potwierdzone, poproszone tego samego dnia (2026-09-28) rzeczy w
zakładce ANALIZA SOBOLA / Sandbox, plus wyjaśnienie osobnego incydentu z
gubieniem commitów na gałęzi po zmergowaniu PR (patrz na końcu).

1. **Panel stabilności parametrów, wersja CAGR.** `_build_stability_panel`
   (Etap 8i) przyjmuje teraz `metric` ("Sortino" domyślnie, lub "CAGR") --
   ten sam mechanizm (scatter + top-N% + werdykt STABILNE/UMIARKOWANE/
   NIESTABILNE), ale liczony względem kolumny CAGR z `raw_outputs`.
   Świadomie osobny panel z własnymi kontrolkami (`sobol-stability-pct-cagr`
   / `sobol-stability-custom-pct-cagr` / `sobol-stability-panel-cagr`), NIE
   przełącznik na jednym panelu -- ranking wg CAGR i wg Sortino może się
   realnie różnić (CAGR całkowicie ignoruje kształt downside), więc to dwa
   różne pytania, nie dwa widoki tego samego. Zero dodatkowych wywołań
   solvera -- ten sam cache co Etap 8i.

2. **Najlepsza pojedyncza kombinacja parametrów.** Nowa sekcja nad obiema
   panelami stabilności: `_build_best_combo_readout` czyta z tego samego
   cache'u wiersz `argmax(CAGR)` i osobno `argmax(Sortino)` spośród
   poprawnych przebiegów, pokazuje ich metryki i wszystkie 8 wartości
   parametrów. Jawnie oznaczone jako pojedynczy PRZEBIEG próbkowania
   Saltelli, nie ponowna optymalizacja ani centroid stabilnego regionu z
   paneli powyżej -- łatwo trafić na szczęśliwy outlier przy małym N, więc
   opisane jako punkt odniesienia, nie "zalecane ustawienia".

3. **Własne wartości suwaków w Sandboxie (poza zakresem suwaka).**
   Sprawdzone najpierw: zakładka Rebalance (`STAGE4A_PARAMS_CONFIG`/
   `build_param_card`) już dziś w pełni na to pozwala -- to zwykłe
   `dcc.Input(type="number")` bez HTML min/max i bez przycinania po stronie
   serwera; wpisanie wartości poza "Recommended" tylko podświetla pomarańczową
   plakietkę "OUTSIDE RANGE", wartość i tak trafia do solvera. Zmiana
   dotyczy więc wyłącznie Sandboxa, gdzie 8 parametrów to `dcc.Slider`
   (fizycznie nie da się przeciągnąć poza min/max). Przy każdym suwaku
   (`ui/layout.py`, `_slider_custom_input`) doszło małe pole "własna
   wartość"; jego callback (`apply_custom_sandbox_slider_value`,
   `ui/tab5_sandbox.py`) rozszerza min/max SAMEGO SUWAKA tak, żeby objąć
   wpisaną wartość, i ustawia jego `value` -- suwak zostaje jedynym
   źródłem prawdy, więc żaden z istniejących callbacków (m.in.
   `update_forward_tracker`), które czytają te suwaki, nie wymagał zmian.

**Osobny incydent (przyczyna "nie widzę commita na GitHubie"):** poprawka
błędu wielo-procesowego cache'u z Etap 8i (`ui/tab5_sandbox.py`, commit
`bdd9b48`) trafiła na gałąź `sobol-stability-panel-2026-09-28` PO tym, jak
PR #6 z tej gałęzi został już zmergowany (merge o 18:51:39 UTC, ten commit
wypchnięty 19:02:43 UTC -- 11 minut później) -- główny branch dostał tylko
wpis do dziennika (Etap 8i), nie samą poprawkę. Push do już zmergowanej
gałęzi nie tworzy nowego PR ani nie dopisuje się do main automatycznie.
Naprawione przez cherry-pick brakującego commita na nową gałąź
`sobol-cagr-and-custom-values-2026-09-28` cięta od świeżego `origin/main`,
razem z pracą z punktów 1-3 powyżej, PR:
https://github.com/marcel923/Beyond-Markowitz-Engine/pull/7. Wniosek na
przyszłość: po zmergowaniu PR-a wszelka dalsza praca nad tym samym tematem
idzie na NOWĄ gałąź od `origin/main`, nie kontynuacją starej (nawet tego
samego dnia).

Zweryfikowane: syntetyczny test dla obu metryk (CAGR-driving i
Sortino-driving parametr poprawnie wraca jako dominujący w
`_build_best_combo_readout`), `py_compile`, pełny `import app` bez kolizji
ID callbacków, na świeżej gałęzi po cherry-picku (nie tylko przed nim).

Pozostały z Etap 8g: (3) panel "Połączone Portfolio".

---

### Etap 8k. Naprawa konfliktu duplikatu Output (slider-sb-rf.value) + nowy audyt server-side (complete)

Błąd zgłoszony przez właściciela zaraz po zmergowaniu PR #7 (Etap 8j):
przeglądarka pokazywała czerwony błąd Dasha "Output 6 (slider-sb-rf.value)
is already in use", nic w Sandboksie nie reagowało (żaden suwak, żaden
przycisk, spółki się nie ładowały -- cała strona nie renderowała layoutu
poprawnie po tym błędzie).

**Przyczyna:** nowy callback `apply_custom_sandbox_slider_value` (Etap 8j)
ma `Output("slider-sb-rf", "value")` w swojej liście 24 outputów -- ale
`slider-sb-rf.value` jest JUŻ Outputem innego, wcześniejszego callbacku,
`autofetch_sandbox_rf` (Etap 8h, uzupełnia rentowność ^TNX przy zmianie
zapisu). Dash pozwala na to WYŁĄCZNIE gdy wszystkie-oprócz-jednej rejestracje
tego samego (id, property) mają `allow_duplicate=True` -- tu żadna nie
miała.

**Ważne odkrycie przy okazji naprawy:** ten konkretny typ błędu (duplikat
Output bez `allow_duplicate`) NIE wywala się ani na `python3 -m py_compile`,
ani na `python3 -c "import app"`, ani nawet na realnym starcie serwera
(`app.run()`) -- sprawdzone bezpośrednio na minimalnym przykładzie Dash: te
wszystkie trzy rzeczy przechodzą bezobjawowo. Walidacja jest WYŁĄCZNIE po
stronie przeglądarki (`dash-renderer` dostaje pełną listę callbacków przez
`/_dash-dependencies` i sam wykrywa konflikt) -- czyli KAŻDA moja dotychczasowa
"zweryfikowane: `import app` bez kolizji ID callbacków" w poprzednich wpisach
tego dziennika NIE łapała tej klasy błędu, tylko literalne kolizje ID
komponentów w layoutcie. Naprawione dwutorowo:
1. Poprawka: `allow_duplicate=True` na wszystkich 24 outputach
   `apply_custom_sandbox_slider_value` (ten callback rejestruje się PO
   `autofetch_sandbox_rf`, więc to on musi mieć `allow_duplicate=True`, nie
   odwrotnie).
2. Nowy `scripts/check_duplicate_outputs.py` -- odtwarza tę samą walidację
   co `dash-renderer`, czytając `app._callback_list` (to samo źródło co
   `/_dash-dependencies`) zamiast czekać na przeglądarkę. Uruchomiony po
   naprawie: 0 konfliktów wśród 216 unikalnych targetów Output w całej
   aplikacji. Zweryfikowany też negatywnie -- uruchomiony na kodzie SPRZED
   poprawki, poprawnie zgłasza dokładnie `slider-sb-rf.value` jako jedyny
   konflikt, więc audyt faktycznie łapie ten błąd, nie tylko wygląda
   sensownie. Dopisany do Konwencji: uruchamiać po KAŻDEJ zmianie/dodaniu
   `@app.callback`.

**Drugie odkrycie przy okazji:** środowisko weryfikacyjne miało domyślnie
zainstalowany Dash 4.4.1 -- dokładnie tę gałąź, przed którą ostrzega komentarz
w `requirements.txt` (dcc.Slider/dcc.Dropdown przepisane od zera w 4.x, custom
CSS przestaje trafiać). Doinstalowany Dash 3.4.0 (w zakresie `>=3.3,<4.0`) i
cała weryfikacja (poprawka + audyt duplikatów) powtórzona pod właściwą wersją.
Dopisane do Konwencji: sprawdzać wersję Dasha przed weryfikacją.

Ta poprawka, razem z nowym skryptem, poszła na nową gałąź cięta od świeżego
`origin/main` (PR #7 był już zmergowany, zanim zdążyłem wypchnąć tę
poprawkę -- ta sama zasada z Etap 8j zastosowana ponownie). PR:
https://github.com/marcel923/Beyond-Markowitz-Engine/pull/8.

Pozostały z Etap 8g: (3) panel "Połączone Portfolio".

---

### Etap 8l. Auto-fill fundamentów z company_store + kalkulator pozycji (complete)

Przed rebalansem (2026-09-30) właściciel poprosił o dwie niezależne zmiany:

1. **Stage 3 (FUNDAMENTAL INPUTS)** przy świeżym Stage 1 nie zeruje już
   fundamentów do 0.0 -- `build_stage3_initial_rows` (`ui/tab2_fundamentals.py`)
   teraz czyta `company_store.get_latest(ticker)` per ticker i wypełnia
   Target Consensus/High/Low, Analyst Coverage, EPS 2Y CAGR, 90d EPS Revision
   tym, co było ostatnio zapisane -- z nową, nieedytowalną kolumną
   "Fundamenty z dnia" (`ui/components.py`, `STAGE3_COLUMNS`), pokazującą datę
   tego zapisu ("brak zapisu" gdy tickera nigdy nie zapisano). Current Price
   (P0) wciąż ZAWSZE ze świeżego fetchu Stage 1, nigdy z company_store --
   tylko same fundamenty są auto-uzupełniane. Ręczne nadpisania w tabeli
   przed CONFIRM & EXPORT działają jak dotychczas (auto-fill działa
   WYŁĄCZNIE przy świeżym Stage 1 -- reset nadpisań / zmiana modelu
   bazowego w `sync_stage3_table` nie są tym dotknięte).

2. **KALKULATOR POZYCJI** -- nowy panel w Rebalansie (Tab 4), między
   ALLOCATION BREAKDOWN a nakładką RV (`ui/layout.py`). Czysty przelicznik:
   widzi WYŁĄCZNIE wagi solvera (`store-stage4b-results`), ceny na żywo
   (`fetch_current_prices`, już istniejące) i kursy walut na żywo (dwie NOWE
   funkcje w `data/market_data.py` -- `fetch_ticker_currencies`, per-ticker
   `.info` jak `fetch_company_profile`, i `fetch_fx_rate`, konwencja Yahoo
   `f"{from}{to}=X"` z fallbackiem na parę odwrotną, wzorowana na
   `fetch_treasury_yield_on_date`; obie nigdy nie podnoszą wyjątku, brak
   danych = `None`/pominięty ticker, ten sam kontrakt co reszta modułu).
   `shares_t = wartość_portfela * w_t * kurs(waluta_bazowa→waluta_t) / cena_t`,
   tylko spółki z DODATNIĄ wagą solvera. `background=True` + progres (bo
   `fetch_ticker_currencies` to wolne zapytanie per ticker), bezpieczna
   degradacja per-wiersz ("--") gdy cena/waluta/kurs się nie rozwiąże dla
   jednego tickera -- reszta portfela liczy się dalej. Na razie WYŁĄCZNIE
   ułamkowe akcje -- zaokrąglenie do pełnych EXPLICITE odłożone na wyraźną
   prośbę ("na razie zróbmy tylko ułamkowe").

Zweryfikowane: `py_compile` wszystkich zmienionych plików, `import app`,
`scripts/check_duplicate_outputs.py` (218 unikalnych targetów Output, 0
konfliktów), Dash 3.4.0 potwierdzony. PR:
https://github.com/marcel923/Beyond-Markowitz-Engine/pull/9.

Pozostały z Etap 8g: (3) panel "Połączone Portfolio" -- wciąż odłożone,
explicite potwierdzone jako niepilne przy tej samej prośbie.

---

### Etap 8m. Panele stabilności Sobola: werdykt odporny na outliery (gęstość/HDR) + wykrywanie dwukierunkowości (complete)

Ten sam dzień (2026-09-30), punkt 1 z priorytetowej listy właściciela --
explicite oznaczony jako priorytet, ale z prośbą "trzeba się dokładniej
zastanowić jak ma to działać" zanim zacznę implementować. Propozycja
(gęstość via KDE/HDR + porównanie z najgorszymi) przedstawiona w czacie z
dwoma wypracowanymi przykładami liczbowymi i 4 pytaniami doprecyzowującymi;
odpowiedzi właściciela: (1) domyślnie 80% pokrycia jądra gęstości, ale z
suwakiem, nie na sztywno; (2) dół ("najgorsze") jako OSOBNA, niezależna
kontrolka, nie związana z % górnego percentyla; (3) trzeci kolor na
wykresie punktowym dla dolnych punktów; (4) dobór dokładnej metody
wykrywania dwukierunkowości pozostawiony mnie, z prośbą o dokładne
wytłumaczenie.

**Problem, który to naprawia:** stary werdykt STABILNE/UMIARKOWANE/
NIESTABILNE (Etap 8i) liczył surowy min-max SPAN punktów top-N% na osi
parametru jako % pełnego zakresu suwaka -- jeden punkt, który trafił do
top-N% przypadkiem (np. przez interakcję z innym parametrem) na ekstremalnej
wartości, mógł samodzielnie przerzucić werdykt z STABILNE na NIESTABILNE,
nawet gdy 39/40 punktów siedziało w wąskim paśmie.

**Rozwiązanie -- jedno wspólne obliczenie dla obu asków:** `_hdr_regions()`
(`ui/tab5_sandbox.py`) liczy Highest-Density Region przy pokryciu `q`
(domyślnie 80%, suwak 50-95%) przez dopasowanie 1D Gaussian KDE
(`scipy.stats.gaussian_kde`, reguła Scotta, zero ręcznego tuningu) i
przycięcie go od góry gęstości, aż skumulowana masa osiągnie `q`% -- to
jest standardowa konstrukcja HDI/credible region ze statystyki bayesowskiej
(ta sama idea co `arviz.hdi`). Jedno to obliczenie odpowiada na oba
pytania naraz:
- **Odporność na outliery:** pojedynczy punkt ma tylko własny, niski,
  jednokrzywkowy wkład do gęstości, podczas gdy prawdziwy klaster wzmacnia
  się przez nakładanie się jąder -- więc komórki siatki o najwyższej
  gęstości prawie zawsze pochodzą z prawdziwego klastra, nie z pojedynczego
  outliera, który przy pokryciu <95% zwykle nigdy nie zostaje włączony do
  HDR. Surowy min-max jest wciąż pokazywany jako mała, nie-alarmująca
  notka (widoczność outliera bez wpływu na werdykt).
- **Dwukierunkowość:** przycinanie powierzchni gęstości może naturalnie
  zwrócić WIĘCEJ NIŻ JEDEN rozłączny region, gdy próbka jest faktycznie
  bimodalna (dobre przebiegi klastrują się w dwóch miejscach) -- bez
  osobnej, sztucznej heurystyki "szukaj przerwy". `_classify_direction()`
  porównuje HDR top-N% i dołu-N%: silne nakładanie -> "BEZ WYRAŹNEGO
  WPŁYWU" (parametr nie decyduje o wyniku); top-N% rozbity na 2+ regiony ->
  "DWUKIERUNKOWY" (wzmocnione, gdy dół-N% siedzi akurat w przerwie między
  nimi); inaczej -- "JEDNOKIERUNKOWY".

**Inne metody rozważone (wyjaśnione właścicielowi w czacie, NIE
zaimplementowane -- KDE/HDR uznane za wystarczające i najlepiej pasujące do
słowa "gęstość" z prośby):** odporne miary rozrzutu (IQR, MAD, trimmed/
winsorized range) jako prostsza alternatywa dla HDR; formalne testy
multimodalności (Hartiganowski dip test, kryterium pasma krytycznego
Silvermana, Gaussian Mixture Models + BIC/AIC); testy porównania dwóch
rozkładów (Kolmogorov-Smirnov, Mann-Whitney U, odległość Wassersteina) jako
alternatywa dla prostego overlap-ratio między HDR górą/dołem; klastrowanie
(k-means/k-medoids + statystyka gap/silhouette) jako alternatywa dla
przycinania gęstości przy wykrywaniu liczby "reżimów".

Oba panele (Sortino i CAGR) dostały to samo trzykrotne rozszerzenie, każdy
z własnym, niezależnym zestawem kontrolek (ten sam wzorzec co reszta pliku
-- panele nigdy nie dzielą stanu). Zero nowych wywołań solvera -- czysta
wizualizacja/reinterpretacja danych, które `run_sobol_batch` już policzył.

Zweryfikowane: `py_compile`, `import app`, `scripts/check_duplicate_outputs.py`
(218 unikalnych targetów Output, 0 konfliktów -- zero nowych Outputów, tylko
nowe Inputy do dwóch istniejących callbacków rebuild), oraz syntetyczny test
odtwarzający obydwa przykłady liczbowe z propozycji w czacie: surowy span
63% zwija się do 14% (robust/STABILNE) po odrzuceniu jednego outliera;
faktycznie bimodalna próbka poprawnie zwraca dwa rozłączne regiony HDR, z
dołem-N% wykrytym w przerwie między nimi -> DWUKIERUNKOWY.

Ta zmiana poszła na TĘ SAMĄ gałąź/PR co Etap 8l (`PR #9` wciąż niezmergowany
w momencie tej pracy -- zasada "nowa gałąź po merge" z Etap 8j/8k nie
dotyczy jeszcze niezmergowanego PR-a). PR:
https://github.com/marcel923/Beyond-Markowitz-Engine/pull/9.

Pozostały z Etap 8g: (3) panel "Połączone Portfolio" -- wciąż odłożone.

---
