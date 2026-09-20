# Beyond-Markowitz Quant Terminal

Python/Dash aplikacja rozszerzająca optymalizację Markowitza o silnik projekcji,
kalibrację walk-forward, klastrowanie, moduł Relative Value (kointegracja/theta,
nakładka post-solver na dopasowane pary) i infrastrukturę Sobol/Morris/PRCC do
analizy wrażliwości solvera.

## Uruchomienie
python3 app.py
# http://127.0.0.1:8050
# Wymaga: pip install -r requirements.txt (m.in. diskcache, multiprocess,
# networkx, SALib)
# UWAGA: app.py ma debug=True ale use_reloader=False -- autoreloader Werkzeug
# psuje DiskcacheManager (progress bary background-callbacków nigdy nie
# docierają do przeglądarki). Restart serwera ręczny po każdej edycji kodu.

## Mapa projektu -- gdzie czego szukać

engine/ (pure math, zero Dash, zero I/O na dysk/sieć):
- returns.py        -- mu_i: BaseReturn=(1-alpha)*U_raw+alpha*G_val, dyskont
                        D_i=exp(-gamma*S_i) UNIWERSALNY na całym mu_i (Etap 8a --
                        ZMIENIONE względem oryginalnego wzoru z Etap 2/Sekcji 3.1
                        PROJECT_CONTEXT.md, które go NIE odzwierciedlają)
- risk.py           -- Estrada Sigma_eps (Tikhonov), CDD, macierz K (crash-overlap)
- optimizer.py      -- solver TPS: Stage1 (intra-cluster SLSQP) -> Stage2
                        (inter-cluster SLSQP) -> Dynamic Singleton Split
                        (Binding Ceiling Throttle). Wzór:
                        TPS(w) = (w.mu-Rf) / (sigma_p*exp(nu*sigma_p)*exp(lam*k_p)+eps)
                        -- nu dodane w Etap 7z, domyślnie 0.0 (kompatybilność wsteczna)
- pairs.py          -- cały silnik Relative Value: kointegracja (Engle-Granger),
                        mechanizm theta, dwuetapowy filtr trwałości (tani ekran +
                        pełny test tylko na tym co przejdzie), Maximum Weight
                        Matching (networkx, dokładny algorytm, nie zachłanny),
                        nakładka post-solver "Droga B" (Etap 7v-7y)
- clustering.py     -- semi-kowariancja (wariant legacy), RMT denoising, DTW+K-Medoids
- evaluation.py     -- equity curve, drawdown, ewaluacja historyczna/forward
- single_asset.py   -- OSOBNY model (Research/Company Dossier). Nigdy nie woła
                        compute_composite_upside_row -- nietknięty przez cały
                        rozwój TPS/Relative Value, nie modyfikuj przy okazji tamtych
- sobol_analysis.py -- Sobol (Saltelli S1/ST/S2) + opcjonalny Morris + PRCC na
                        8 parametrach solvera (alpha, lambda, nu, gamma, kappa,
                        w_max, rf, n_ref)

data/ (persystencja + I/O rynkowe, zero matematyki, zero Dash):
- market_data.py    -- WSZYSTKIE wywołania yfinance, jedno źródło prawdy
- snapshot_store.py -- CRUD zapisów portfela (SAVE PORTFOLIO) -- dziś zapisuje
                        WYŁĄCZNIE store-stage4b-results, BEZ danych nakładki RV
                        (patrz "Otwarte priorytety" niżej)
- company_store.py  -- historia danych fundamentalnych, osobno per spółka
- universe_store.py -- uniwersum śledzonych spółek

ui/ (Dash layout + callbacks, woła tylko engine/ i data/):
- app_instance.py           -- obiekt `app`, DiskcacheManager
- components.py             -- współdzielone buildery: KPI, TPS formula
                                breakdown, build_pair_network_and_matrix,
                                build_pair_zscore_figure, build_pair_weight_history_figure
- layout.py                 -- cały layout (5 zakładek głównych)
- tab1_market_data.py       -- Stage 1 (uniwersum, klastrowanie, panel doboru par)
- tab2_fundamentals.py      -- Stage 3 (dwa przyciski: "ZAPISZ DO RESEARCH" vs
                                "EKSPERYMENTUJ BEZ ZAPISU")
- tab3_tailrisk.py          -- tail-risk, underwater chart
- tab4_rebalance.py         -- Stage 4B solver, panel nakładki RV, formula
                                breakdown, SAVE PORTFOLIO
- tab5_sandbox.py           -- Forward Tracker + zakładka "ANALIZA SOBOLA",
                                replika nakładki RV z toggle na żywą krzywą equity
- module_relative_value.py  -- moduł Relative Value, zakładki 1-6
- module_research.py        -- Research / Company Dossier (UI dla single_asset.py)
- theme.py                  -- stałe wizualne

## Kluczowe konwencje (cały projekt)

- Regresja solvera po KAŻDEJ zmianie w engine/: tickery AVGO/NVDA/HPE/LYC.AX,
  seed=5. Baseline POTWIERDZONY na dziś: NVDA=0.35, HPE=0.246, LYC.AX=0.054,
  AVGO=0.35. Po Etap 8a (gamma uniwersalne na mu_i) wychodzi HPE=0.2463,
  LYC.AX=0.0537 -- policzone i pokazane, ale NIEPOTWIERDZONE jako nowy
  punkt odniesienia. Stary baseline wiąże dopóki nie padnie wyraźne "tak".
- Fitting: zawsze `scipy.optimize.least_squares` (Trust Region Reflective),
  NIGDY `minimize()` na ręcznym MSE -- confirmed źle działa nawet na czystych danych.
- Zakładki 2 i 3 modułu Relative Value (mechanizm dyskretny 80/20) są CELOWO
  nietknięte przez cały rozwój mechanizmu theta -- nie modyfikuj bez wyraźnej prośby.
- Kolejność par (Ticker A/B) MUSI być kanonizowana przez
  `canonicalize_pair_order_by_theta` przed jakimkolwiek obliczeniem -- regresja
  OLS nie jest symetryczna.
- Kointegracja jest WYŁĄCZNIE informacyjna wszędzie poza Zakładkami 2/3 modułu
  RV -- jedyne kryterium kwalifikacji do matchingu to dwuetapowy filtr trwałości
  theta (`MATCH_MIN_WINDOWS=2`, `MATCH_MIN_T_STATISTIC=1.75` w `engine/pairs.py`,
  niepotwierdzone na realnym uniwersum).
- Nakładka RV post-solver ("Droga B"): działa WYŁĄCZNIE na już policzonych
  wagach solvera (nigdy nie dotyka mu_i/klastrowania), `w_max` jest przez nią
  CAŁKOWICIE ignorowane (świadoma decyzja), i jest dziś CZYSTO DIAGNOSTYCZNA --
  SAVE PORTFOLIO jej nie widzi.
- Sobol liczony na JEDNYM zamrożonym zapisie i JEDNYM horyzoncie na raz --
  NIGDY sklejanie snapshotów (myliłoby efekt parametru z efektem czasu rynkowego).
- Kosztowne operacje (>1s) zawsze jako osobny przycisk z `background=True` i
  raportowaniem postępu -- nigdy wpięte w szybki, reaktywny callback.

## Pełna historia decyzji

`PROJECT_CONTEXT.md` to JEDYNE źródło prawdy dla chronologii i uzasadnień --
append-only log ponumerowany "Etap X". NIE edytuj istniejących wpisów, tylko
dopisuj nowe na końcu. Sekcje 1-4 tego pliku (Problem Formulation, Mathematical
Framework) to opis teoretyczny, który w kilku miejscach NIE nadążył za kodem
(nie pokazuje Alpha Blend z Etap 2, `nu` z Etap 7z, ani gamma uniwersalnego z
Etap 8a) -- w razie sprzeczności ufaj najnowszemu wpisowi "Etap", nie Sekcji 3.
PRZECZYTAJ odpowiedni fragment PRZED każdą większą zmianą w danym obszarze --
nie trzeba czytać całości przy drobnych, niezwiązanych poprawkach.

## Po większej, potwierdzonej zmianie

Zaproponuj krótki wpis do `PROJECT_CONTEXT.md` w tym samym stylu co istniejące
(kolejny numerowany "Etap", dopisany na końcu, nic wcześniejszego nie ruszane).
Pokaż go i POCZEKAJ na potwierdzenie, zanim go dopiszesz.

## Otwarte priorytety (potwierdzone 2026-09-20)

1. [GŁÓWNY] Nakładka RV nie trafia do zapisu. `save_snapshot_callback` w
   `tab4_rebalance.py` czyta wyłącznie `store-stage4b-results`. Do zbudowania:
   przy zapisie snapshotu, dodatkowo zapisać jakie pary były wybrane, ich
   t-score, i jak zmieniły się wagi przed/po nakładką -- z jawnym znacznikiem
   w pliku, żeby Sandbox mógł bezpiecznie odróżnić snapshoty zapisane z
   nakładką od tych bez niej (bez tego ryzyko błędów przy wczytywaniu starszych
   zapisów).
2. Kalibracja `MATCH_MIN_WINDOWS`/`MATCH_MIN_T_STATISTIC` na realnym
   uniwersum -- niepilne, nakładka RV to dziś ostatni krok procesu, nie kluczowy.
3. Potwierdzenie nowego baseline'u regresji solvera z Etap 8a (patrz Konwencje
   wyżej) -- czeka na wyraźne "tak"/"nie" od właściciela projektu.
4. Rozwinięcie zakładki "ANALIZA SOBOLA" (czytelność/interpretacja wyników) --
   plus czeka na więcej realnych sesji handlowych od utworzenia pierwszego
   zapisu, żeby wyniki na prawdziwych danych były wiarygodne.