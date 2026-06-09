# CLAUDE.md

Guidance for working in this repository.

## Project overview

SCIE6063001 Computational Physics project (Even Semester 2025/2026): a 2D acoustic room
echo simulation. Sound is modelled as a scalar pressure field on a grid; a "clap" launches
wavefronts that reflect off walls, diffract through doorways, and are absorbed by materials.
The numerical method is FDTD (Finite Difference Time Domain) on the damped wave equation.
Deliverables: an interactive Pygame app, a PDF report (`report/`), and validation notebooks.

## Running the code

All source lives in `src/`; run from there so imports resolve:

```bash
cd src
python visualiser.py   # the interactive app (click-to-clap, build/recolour walls, energy chart)
python verify.py       # all checks: grid, room, scenes, solver, headless app  -> PASS/FAIL
```

Notebooks (`notebook/01_grid_check.ipynb`, `02_physics_check.ipynb`) add `../src` to
`sys.path` and render with the project's own pygame renderer — **no matplotlib**.

## Architecture

One engine, strict layered imports (each layer only imports from below):

```
config.py     constants + MATERIALS (no project imports)
   |
grid.py       RoomGrid: geometry + FDTD params (dx, dt, Courant number)
   |
utils.py      coord<->cell, empty_field, pressure_reflection
   |        \
room.py      physics_solver.py     scene builder (alpha-map + pieces)  |  FDTD WaveSolver
   |        /
scenes.py     room presets (two_rooms, tiny_home)
   |
render.py     field -> pixels (red-on-white wave, material-coloured walls)
   |
visualiser.py the interactive Pygame app   (verify.py drives all modules for tests)
```

`room.py` and `physics_solver.py` are siblings — neither imports the other; the solver
consumes the `(NY, NX)` α-map a `Room` produces. The two teammates split physics
(`room` + `physics_solver`) and visuals (`render` + `visualiser`) on the shared base
(`config` / `grid` / `utils`).

## Key design details

- `config.py` is the single source of truth for constants (`C_SOUND`, `FREQ`, `ROOM_W/H`,
  `PPW`, `CFL_FACTOR`, `BETA`) and `MATERIALS` (energy absorption coefficient α per preset).
- Fields and the α-map are `(NY, NX)` float64 arrays (row ↔ y, col ↔ x), built with
  `utils.empty_field`. Metre ↔ cell conversion goes through `utils.coord_to_cell` /
  `cell_to_coord` (which clamp).
- **Wave model:** lossless air by default (`BETA = 0`); all loss is at obstacles. A wall cell
  has α > 0 and reflects with `R_p = √(1−α)` (so absorbed energy = `1 − R_p² = α`) via a
  dissipative Mur-blend ghost `g = u − (1−R_p)/r·(u − u_prev)` in `WaveSolver`. The clap is a
  zero-mean Ricker wavelet. `WaveSolver.set_beta` adds reverberation decay at runtime.
- **Scenes / editing:** build rooms with `scenes.two_rooms` / `tiny_home`, or `Room`
  directly (`add_border`, `add_rectangle`/`add_block`, `add_doorway`). `Room` tracks
  selectable *pieces* so the app can recolour (`set_material`/`set_wall_material`), build, or
  delete (`remove_piece`) them; the outer shell is `protected`. After any edit call
  `solver.set_alpha(room.alpha)`.
- **Rendering:** `render.field_to_rgb` draws rest pressure white, compression red (the
  wavefront), rarefaction white, with a gamma lift so echoes stay visible; obstacles are
  tinted by material. The full FDTD derivation and error analysis live in
  `report/Final_Report.docx` (built from `report/build/generate.js`).
```
