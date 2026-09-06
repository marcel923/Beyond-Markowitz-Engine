# Jak odpalić Quant Terminal lokalnie

Masz już folder `quant-terminal` i `venv` – super, zostały 2 kroki.

## 1. Podmień plik i doinstaluj zależności

1. Zamień swój obecny `quant_terminal.py` na wersję z tego czatu (poniżej) — ma dodane logowanie
   pełnych błędów do konsoli zamiast ich ukrywania.
2. Wrzuć `requirements.txt` do tego samego folderu co `quant_terminal.py`.
3. Aktywuj venv i zainstaluj:

```bash
# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
```

## 2. Uruchom

**WAŻNE (po refaktorze Etap 0):** aplikacja to teraz kilka plików w folderach
`engine/`, `data/`, `ui/`, nie jeden `quant_terminal.py`. Rozpakuj wszystkie
foldery i pliki do jednego katalogu projektu (struktura musi zostać
zachowana — `engine/`, `data/`, `ui/` jako podfoldery obok `app.py`), a
`saved_portfolios.json` zostaje w katalogu głównym, obok `app.py`.

Uruchamiasz teraz `app.py`, nie `quant_terminal.py`:

```bash
python3 app.py
```

Terminal w konsoli pokaże coś w stylu:

```
Dash is running on http://127.0.0.1:8050/
```

Otwórz ten adres w przeglądarce. To Twój **własny proces**, bez cudzego serwera,
bez timeoutów przeglądarkowego IDE.

## 3. Jak teraz diagnozować błędy

Największa zmiana: jeśli coś się wywali, **nie patrz tylko na czerwony napis w appce** —
patrz w konsolę/terminal, w którym odpaliłeś `python3 quant_terminal.py`. Teraz zobaczysz
tam pełny traceback (dokładną linijkę i typ błędu), np.:

```
[STAGE 1] CRITICAL ERROR - PEŁNY TRACEBACK:
Traceback (most recent call last):
  File "quant_terminal.py", line ..., in run_stage_01_ingestion
    ...
yfinance.exceptions.YFRateLimitError: Too many requests
```

To mi (albo Tobie) mówi dokładnie co się dzieje, zamiast zgadywania.

## 4. Najczęstsze błędy i co znaczą

| Widzisz w konsoli | Co to znaczy | Co zrobić |
|---|---|---|
| `YFRateLimitError` / puste dane | Yahoo chwilowo blokuje Twoje IP po zbyt wielu requestach | Odczekaj kilka minut, nie odświeżaj appki w kółko |
| `JSONDecodeError` z yfinance | Yahoo zmienił format odpowiedzi, stara wersja `yfinance` tego nie obsługuje | `pip install --upgrade yfinance` |
| Callback error w przeglądarce, ale konsola milczy | To już nie powinno się zdarzać po poprawce — jeśli się zdarzy, wyślij mi zrzut konsoli |
| `Address already in use` / port zajęty | Poprzedni proces Pythona nadal działa | Zamknij terminal (Ctrl+C) albo zmień port na końcu pliku: `app.run(debug=True, port=8060)` |
| Slidery/dropdowny białe albo nieostylowane mimo custom CSS w `app.index_string` | `pip install` ściągnął Dash 4.x, który od zera przepisał `dcc.Slider`/`dcc.Dropdown` (bez `rc-slider`/`react-select`) — stary CSS celuje w klasy, których już nie ma w DOM | `requirements.txt` ma już pin `dash<4.0`. Zrób `pip install -r requirements.txt --upgrade` w aktywnym venv, żeby zejść na 3.x. Sprawdź wersję: `python -c "import dash; print(dash.__version__)"` |

## 5. Co dalej

Jak już to postawisz lokalnie i odpalisz Stage 1 (wpisując np. tickery z placeholdera),
daj znać co konkretnie widzisz w konsoli — jeśli coś nadal siada, poprawimy to punktowo
zamiast zgadywać na ślepo.

README zostaje na razie nietknięte (zgodnie z tym co mówiłeś — punkt 4 i wcześniejsze
kroki jeszcze do przemyślenia z Twojej strony).