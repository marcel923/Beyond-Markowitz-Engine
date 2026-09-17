# Beyond-Markowitz Quant Terminal

Python/Dash aplikacja rozszerzajaca optymalizacje Markowitza o silnik projekcji,
kalibracje walk-forward, klastrowanie, i modul Relative Value (kointegracja/theta).

## Uruchomienie
python3 app.py
# http://127.0.0.1:8050
# Wymaga: pip install -r requirements.txt (m.in. diskcache, multiprocess, networkx)

## Mapa projektu -- gdzie czego szukac
- engine/single_asset.py -- silnik projekcji pojedynczego aktywa, kalibracja walk-forward,
  stabilnosc parametrow (Cov(theta) ~ sigma^2*(J^T*J)^-1)
- engine/clustering.py -- Ward na semi-kowariancji, DTW+K-Medoids, DBSCAN, RMT, Silhouette
- engine/pairs.py -- caly silnik Relative Value (skaner kointegracji, mechanizm theta,
  test trwalosci wielohoryzontowy, kanonizacja kolejnosci par, Maximum Weight Matching)
- ui/tab1_market_data.py -- Rebalance Stage 1 (uniwersum, klastrowanie, panel doboru par)
- ui/tab4_rebalance.py -- Stage 1/2 SLSQP solver, Singleton Split Loop, Active Binding Constraint
- ui/module_relative_value.py -- zakladki 1-6 modulu Relative Value
- ui/module_research.py -- Research / Company Dossier
- ui/app_instance.py -- konfiguracja Dash, DiskcacheManager dla progress bar

## Kluczowe konwencje (caly projekt)
- Kazda zmiana w engine/ konczy sie regresja solvera: tickery AVGO/NVDA/HPE/LYC.AX,
  seed=5, MUSI dac NVDA=0.35, HPE=0.246, LYC.AX=0.054, AVGO=0.35
- Fitting: zawsze scipy.optimize.least_squares (Trust Region Reflective), NIGDY
  minimize() na recznym MSE -- confirmed zle dziala nawet na czystych danych
- Zakladki 2 i 3 modulu Relative Value (mechanizm dyskretny 80/20) sa CELOWO
  nietkniete przez caly rozwoj mechanizmu theta -- nie modyfikuj bez wyraznej prosby
- Kolejnosc par (Ticker A/B) MUSI byc kanonizowana przez canonicalize_pair_order_by_theta
  przed jakimkolwiek obliczeniem -- regresja OLS nie jest symetryczna
- Kointegracja jest WYLACZNIE informacyjna wszedzie poza Zakladkami 2/3 --
  jedyne kryterium kwalifikacji to test trwalosci theta (p<0.10 jednostronne, t>0)

## Pelna historia decyzji
PROJECT_CONTEXT.md zawiera pelna, chronologiczna historie -- co zbudowane, co
odrzucone i dlaczego (m.in. usredniania divergence po calym oknie -- nie dziala,
autokorelacja ~0.99). PRZECZYTAJ odpowiedni fragment PRZED kazda wieksza zmiana
w danym obszarze, zeby nie powtorzyc juz odrzuconego eksperymentu. Nie trzeba
czytac calosci przy drobnych, niezwiazanych poprawkach.

## Po wiekszej, potwierdzonej zmianie
Zaproponuj krotki wpis do PROJECT_CONTEXT.md w tym samym stylu co istniejace
(numerowany "Etap", co zrobione, co potwierdzone testem, co jeszcze odlozone).
Pokaz mi go i POCZEKAJ na potwierdzenie, zanim go dopiszesz -- nie dopisuj
automatycznie bez pokazania. Przy drobnych poprawkach (literowka, mala korekta)
pomijaj ten krok.

## Aktualne priorytety
1. Wpiecie doboru par do realnych wag Stage 1 (dzis czysto informacyjne)
2. Reformulacja TPS z niepewnoscia prognozy w mianowniku
3. Kalibracja progow dwupoziomowego filtra (MATCH_MIN_WINDOWS, MATCH_MIN_T_STATISTIC)
   na realnym uniwersum
