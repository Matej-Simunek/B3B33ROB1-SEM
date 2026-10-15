# Postup u robotu CRS93 (první návštěva)

Prostředí: `git pull && python3 -m venv .venv --system-site-packages && source .venv/bin/activate && pip install -r requirements.txt`
(`uv` tam není). Před spuštěním zavři `pylonviewer` (kameru smí mít otevřenou jen jeden program).

## 0. Bez robotu (kamera + stolek)
1. `python scripts/camera_info.py` – model, rozlišení (čekáme 1920×1200), jméno kamery `camera-crs93`.
2. Zaostři kameru (kroužek objektivu) – snímky byly rozostřené, hrany markerů jsou důležité.
3. Zapni stolek (nejpomalejší rychlost: po zapnutí 2× SR), pak `python scripts/run.py` – vypíše směr a rychlost otáčení
   z markerů (v pixelech). Sekvence se uloží do `data/sequences/`; commitni ji, ať se dá vyhodnotit doma.
4. Znovu s opačným směrem otáčení (bonus +2 b).
5. Vyfoť čistou desku bez obruče: `python scripts/capture_image.py deska` a `python scripts/detect_all.py data/images/deska.png`.

## 1. Kalibrace kamera → robot (robot se hýbe!)
**Před začátkem odstraň ze stolu desku s puzzle a vše ostatní.** `jog_to_view.py` a `collect_calibration.py` kontrolují jen
výšku příruby nad stolem (≥ 10–12 cm), ne překážky; kolize s tyčkou hlídá až plánovač v `run.py`.
1. Zapni robot (červený vypínač + žlutý Arm Power), E-stop po ruce, klec zavřená.
2. Obruč na robotu. V pozici `home` míří příruba vodorovně dopředu, kroužek není shora vidět. Spusť
   `python scripts/jog_to_view.py` (příkaz `down` nastaví přírubu dolů, pak `x`/`y`/`z`/`yaw` po malých krocích;
   po každém pohybu vypíše, jestli kamera vidí kroužek a kde), dokud není celý kroužek v záběru ≥ 15 cm nad stolem. Skončit `q`.
3. `python scripts/collect_calibration.py --speed 0.2 --no-home` (robot už je nahomovaný a stojí v pozici z kroku 2). Skript sám jezdí
   kolem startovní pózy, v každé pozici najde obruč v obraze a uloží `data/calibration/samples.json`.
4. `python scripts/calibrate.py` – vypíše chybu reprojekce (cíl < 2–3 px), výšku kamery (měřeno 1,17 m), ohnisko (~4500 px)
   a TCP obruče vůči přírubě; uloží `data/calibration/camera_robot.json`.
5. Commitni `data/calibration/`.

## 2. Suchý běh (robot se nehýbe)
`python scripts/run.py --puzzle A` – vyfotí stolek, najde desku, naplánuje trajektorii a vypíše „plan OK“ nebo
důvod, proč není puzzle dosažitelné (robot se v tom případě nepohne, což zadání chce).

## 3. Ostrý běh
`python scripts/run.py --puzzle A --execute --speed 0.3`
- vždy první pokus pomalu (`--speed` 0.2–0.3), E-stop v ruce,
- skript chce potvrzení slovem `yes`, po potvrzení znovu naplánuje a jede v reálném čase,
- při odchylce robotu od plánu > 0,15 rad se sám zastaví.

## Co ověřit a případně doladit (hodnoty z modelu, ne z měření)
- `planning.Tool` / kalibrace: TCP obruče; poloha báze robotu (z = 0 na stole).
- `plate_pose.PUZZLE_FOOT_OFFSET`: posun paty puzzle vůči středu mezi markery.
- `puzzles.Y_SIGN_C`: směr druhého ohybu puzzle C (z výkresu nebylo jednoznačné).
- `puzzles.END_Z` a `planning.SUCCESS_HEIGHT`: hloubka konce (obruč < 4 cm nad podložkou s ArUco).
- Rychlost koordinovaných pohybů CRS: `run.py` sleduje odchylku skutečného q od plánu; pokud robot zaostává, sniž `--speed`.
- Předpoklad modelu: normála kroužku obruče = osa z příruby, rukojeť míří zpět k ose příruby. Zkontroluj na fotce
  `data/calibration/images/` (kroužek je vodorovně) a podle `tcp` z kalibrace (x, y = vzdálenost středu kroužku od osy příruby).
- Báze robotu leží na stole (z = 0): změř metrem výšku kroužku nad stolem v jedné póze a porovnej s FK
  (`robot.fk(q)[2, 3]` minus `tcp[2]` z kalibrace). Rozdíl > 1 cm = oprav posun báze před ostrým během.
- `run.py --execute` se při startu ptá, jestli je robot nahomovaný; odpověz `N` jen pokud smí robot jet homing.
