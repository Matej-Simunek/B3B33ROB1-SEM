# Technická zpráva – osnova (B3B33ROB1, bludiště na rotačním stolku, CRS93)

1. Úvod a zadání (zvolená varianta, robot, body)
2. Použitý hardware a souřadné soustavy (kamera 1,17 m, markery 40 mm, podložka 33 mm, puzzle +34 mm)
3. Řešení
   1. Detekce ArUco markerů a obruče (barva, `detect.py`)
   2. Odhad rotace stolku (lineární metoda nejmenších čtverců, osa mimo střed desky, směr ±)
   3. Kalibrace kamera → robot a TCP obruče (`calibration.py`)
   4. Pose desky a modely puzzle A–D (`plate_pose.py`, `puzzles.py`)
   5. Plánování: IK, časování podle limitů kloubů, čekání na dosažitelný úhel, kontrola kolizí a dosažitelnosti
   6. Provedení v reálném čase a sledování odchylky
4. **Experimenty** (povinná kapitola): mapa dosažitelnosti (`scripts/reachability_map.py`), chyby kalibrace a odhadu
   pozice, úspěšnost pokusů podle polohy a směru otáčení, simulace vs. realita
5. **Podíl členů týmu** (povinná kapitola): konkrétní technický přínos každého člena (ne „kód vs. zpráva“)
6. **Využití AI** (povinná kapitola, 2–5 odrážek): co bylo nejpřínosnější
7. Závěr, omezení, další práce
