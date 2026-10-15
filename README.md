# B3B33ROB1 – Semestrální práce: Bludiště na rotačním stolku

Tým 3 studentů, průmyslový robot (Bosch 4-DoF / CRS93, CRS97, RV6S 6-DoF), RGB kamera Basler, ArUco značky.
Podklady (rozměry bludiště, stolku, obruče): https://github.com/pilcsimo/ROB1-2025-hw
Pracoviště: https://robotics-robot-stations.readthedocs.io/en/latest/index.html

## Úloha
Robot drží kontrastní obruč, kterou musí provléct bludištěm / tyčkou na rotačním stolku
(nejpomalejší rychlost; zapnutí + 2× SR). Nedosažitelná poloha se musí detekovat a nahlásit.

- Úspěch: obruč na konci < 4 cm nad podložkou s ArUco; limit 2 min/pokus; max 3 neúspěšné pokusy.
- Diskvalifikace: kolize robota se stolem, stolkem nebo sám se sebou.

| Varianta | Popis | Body | Robot |
|---|---|---|---|
| A1 | Rovná tyčka bez stolku | 2 | 4-DoF, 6-DoF |
| A2 | Rovná tyčka na rotačním stolku | 8 | 4-DoF, 6-DoF |
| B | Složitější trajektorie v rovině | 12 | 4-DoF, 6-DoF |
| C | Pohyb mimo rovinu | 15 | 4-DoF, 6-DoF |
| D | Natočení obruče | 18 | pouze 6-DoF |
| Bonus | Oba směry otáčení stolku | +2 | A2–D |

## Odevzdání
- Technická zpráva do BRUTE před předvedením (kapitoly: Experimenty, Podíl členů týmu, Využití AI – 2–5 odrážek).
- Předvedení řešení cvičícímu (Krsek, Smutný, Petrík).
- Bonus za dřívější odevzdání zprávy: 17. 11. 2026 +5, 24. 11. +4, 1. 12. +3, 8. 12. +2, 15. 12. +1 (vždy 23:59).
- Řádný termín 5. 1. 2027 23:59 (−1 b / 3 dny zpoždění), nejzazší 24. 1. 2027 23:59.
- Rezervace robota: BRUTE. Konzultace: e-mail všem třem cvičícím.

## Spuštění (jeden skript)
```bash
uv run python scripts/run.py --sim --puzzle D              # celá scéna v simulaci (bez kamery a robotu)
python scripts/run.py                                      # jen rotace stolku z kamery (bez kalibrace)
python scripts/run.py --puzzle A                           # suchý běh: detekce, plán, dosažitelnost
python scripts/run.py --puzzle A --execute --speed 0.3     # ostrý běh s potvrzením, robot se pohne
```
Postup u robotu krok za krokem: [docs/ROBOT_CHECKLIST.md](docs/ROBOT_CHECKLIST.md), plán a nejistoty: [docs/PLAN.md](docs/PLAN.md).

## Prostředí
```bash
uv sync            # vytvoří .venv (Python 3.12) a nainstaluje závislosti
uv run pytest tests/test_maze_*.py     # testy projektu (cca 1,5 min, včetně simulace celého řetězce)
```
Bez `uv` (robotový PC): `python3 -m venv .venv --system-site-packages && pip install -r requirements.txt`.

## Struktura
- `src/robotics_toolbox/maze/` – projekt: `camera.py`, `aruco_base.py`, `detect.py`, `table_motion.py`, `camera_model.py`,
  `calibration.py`, `plate_pose.py`, `puzzles.py`, `planning.py`, `executor.py`, `pipeline.py`, `sim.py`, `sim_run.py`
- `src/robotics_toolbox/` – základ převzatý z `robotics_labs` (SE(2)/SE(3), roboti, IK, plánování, renderery)
- `scripts/` – `run.py` (vše najednou), `collect_calibration.py`, `calibrate.py`, `capture_image.py`, `detect_all.py`,
  `camera_info.py`, `reachability_map.py`
- `exercises/`, `tests/` – cvičení a testy z labů (část odpovídá nedokončeným domácím úkolům, mohou padat)
- `data/` – snímky, sekvence, kalibrace; `docs/` – podklady (`ROB1-2025-hw`), plán; `report/` – technická zpráva
