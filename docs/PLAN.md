# Plán implementace (větev `feature/full-pipeline`)

Cíl: jeden skript `scripts/run.py`, který za běhu (runtime) vyfotí scénu, odhadne rotaci stolku,
najde desku s puzzle, naplánuje trajektorii obruče skrz puzzle, ověří dosažitelnost a (jen s `--execute`)
provede pohyb robotu CRS93. Bez robotu a kamery jde vše spustit v simulaci / z uložených dat.

## Souřadné soustavy
- **robot** – báze CRS93, z nahoru, z=0 na stole (předpoklad, ověřit).
- **obraz** – pixely kamery (x doprava, y dolů), kamera ≈ 1,17 m nad stolem, ≈ kolmo dolů.
- **deska** – počátek ve středu mezi markery (ID1 vlevo nahoře, ID2 vpravo dole), osa x = hrana TL→TR markeru 1,
  z nahoru, rovina markerů je 33 mm nad stolem.
- **puzzle** – počátek v patě tyčky (střed desky + posun `PUZZLE_FOOT_OFFSET`), z = 34 mm nad rovinou markerů.
- **obruč (TCP)** – počátek ve středu kroužku, osa z = normála kroužku (směr průchodu), osa x = k rukojeti.

## Etapy
1. **Základ** (hotovo na `main`): kamera, ArUco, detekce obruče, odhad rotace stolku (`table_motion.py`).
2. **Modely puzzle A–D** (`puzzles.py`): osa tyčky jako polyline/oblouky z výkresů, hrana 10×10 mm,
   cesta obruče (od volného konce k patě), natočení obruče (normála = tečna, roll = paralelní transport).
3. **Kamerový model + kalibrace kamera→robot** (`camera_model.py`, `calibration.py`): pinhole (f, pozice, orientace) +
   TCP obruče, nelineární nejmenší čtverce z dvojic (q, pixel středu obruče). Skripty `collect_calibration.py`
   (jezdí robotem, pomalu, s potvrzením) a `calibrate.py`.
4. **Pose desky v souřadnicích robotu** (`plate_pose.py`): markery → rovina z=33 mm → rám desky; osa a směr rotace
   stolku v robotu; predikce pose desky v čase.
5. **Plánování** (`planning.py`): cesta v rámu puzzle → pose příruby → IK (CRS93 analytická) s kontinuitou větví,
   limity kloubů, výška nad stolem, kontrola celé cesty; hlášení „nedosažitelné“ bez pohybu.
6. **Provedení** (`executor.py`): sledování pohyblivého rámu (cíl se přepočítává podle času příjezdu), přiblížení → vsunutí
   → dokončení (obruč < 4 cm nad podložkou), časový limit 2 min, `SimRobot` pro dry-run.
7. **Integrace** (`pipeline.py`, `scripts/run.py`): jednorázově vše; `--puzzle`, `--execute`, `--replay`, `--sim`.
8. **Simulace a testy**: syntetická kamera (homografie) + rotující deska + SimRobot, test celého řetězce; testy modulů.
9. **Experimenty a zpráva**: mapa dosažitelnosti (`scripts/reachability_map.py`), osnova zprávy v `report/`.

## Nejistoty (nutné ověřit na robotu)
- TCP obruče vůči přírubě (kalibrace ho odhadne), z=0 báze robotu na stole.
- Posun paty puzzle vůči středu desky, směr y-posunu u puzzle C, rozměry puzzle E (nemodelováno).
- Rychlost `move_to_q` (koordinovaný pohyb je asynchronní): executor proto sleduje skutečné q.
- Ostrost kamery, expozice pro detekci hran při otáčení.

## Pravidla bezpečnosti
- Kód pro robota se nikdy nespouští bez `--execute` + interaktivního potvrzení; výchozí je dry-run.
- Nízké rychlosti, kontrola limitů, výšky nad stolem a dosažitelnosti před prvním pohybem.
