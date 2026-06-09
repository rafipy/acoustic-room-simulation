# Acoustic Room Simulation — Repository Cheat Sheet

A presenter's map of the whole project: what the physics is, how the PDE becomes the
FDTD update, what every knob does, and where each idea lives in the code. Every claim is
anchored to a `file:line` you can open while talking.

> **One-line summary.** Sound pressure is a scalar field `u(x, y, t)` on a 2-D grid. A
> clap injects a pulse; the **damped wave equation** propagates it; walls **reflect and
> absorb** it; an energy curve and an interactive Pygame dashboard let you watch it decay.

---

## 0. Run it

```bash
cd src
python visualiser.py   # interactive app: click-to-clap, build/recolour walls, energy chart
python verify.py       # 18 checks across grid/room/scenes/solver/app -> ALL PASS
```

The grid auto-sizes to **124 × 93 cells** at `dx ≈ 6.5 cm`, `dt ≈ 120 µs`, Courant
`r = 0.636` (shown in the sidebar and printed by `RoomGrid.__repr__`,
[grid.py:31](src/grid.py:31)).

---

## 1. The physics

### 1.1 The governing equation (the PDE)

The model is the **2-D damped wave equation** for acoustic pressure `u`
([physics_solver.py:1](src/physics_solver.py:1) docstring):

```
∂²u/∂t²  +  β ∂u/∂t   =   c² ∇²u
 (inertia)   (damping)      (stiffness / propagation)
```

| Term | Meaning | In code |
|------|---------|---------|
| `u` | acoustic pressure (the field we solve for) | `solver.u_curr`, an `(NY, NX)` array |
| `c` | speed of sound, 343 m/s | `C_SOUND` [config.py:4](src/config.py:4) |
| `∇²u` | Laplacian — couples each cell to its neighbours; this is what makes a disturbance *propagate* at speed `c` | 5-point stencil [physics_solver.py:123](src/physics_solver.py:123) |
| `β ∂u/∂t` | **air damping** — a loss proportional to how fast pressure is changing; drains energy everywhere so the room reverberates and falls silent | `BETA` [config.py:16](src/config.py:16), `set_beta` [physics_solver.py:86](src/physics_solver.py:86) |

`β = 0` (the default) means **lossless air** — all energy loss happens *at the walls*,
not in the bulk. The interactive app overrides this to `β = 30` (`DAMP_DEFAULT`,
[visualiser.py:65](src/visualiser.py:65)) so claps audibly die out.

### 1.2 Two distinct loss mechanisms (don't conflate them)

This is the single most important conceptual point for the demo:

1. **Wall absorption `α`** — a *boundary* property of each material. `α ∈ [0, 1]` is the
   fraction of energy a surface swallows per bounce (`MATERIALS`,
   [config.py:21](src/config.py:21)). Concrete `α=0.02` (hard mirror) … Acoustic Foam
   `α=0.85` … Open `α=1.0` (anechoic). Converted to a **pressure reflection
   coefficient** `R_p = √(1−α)` in [utils.py:29](src/utils.py:29), so the absorbed-energy
   fraction is exactly `1 − R_p² = α`.

2. **Air damping `β`** — a *bulk* property of the whole medium. It decays the field
   uniformly in space (reverberation tail). Energy then falls off as `≈ exp(−βt)`.

> Rule of thumb: **`α` decides where echoes die (the walls); `β` decides how fast the
> whole room goes quiet.**

### 1.3 Boundaries: the reflecting / absorbing wall (Mur ghost)

Walls are simply cells with `α > 0` (`is_solid`, [room.py:160](src/room.py:160)). The
clever part is the boundary condition at an air↔solid face. When the Laplacian needs the
value *inside* a wall, the solver substitutes a **ghost value**
([physics_solver.py:123](src/physics_solver.py:123)):

```
g = u − (1 − R_p)/r · (u − u_prev)
```

- **Rigid wall** `α=0 → R_p=1`: `g = u`. The neighbour mirrors the cell → zero normal
  gradient (Neumann) → a perfect reflector, **lossless**.
- **Open / anechoic** `α=1 → R_p=0`: `g = u − (1/r)(u − u_prev)`. This is a **first-order
  Mur absorbing boundary** — the `(u − u_prev) ≈ Δt·∂u/∂t` term lets the wave *leave* the
  grid instead of bouncing.
- In between, `R_p` blends the two so the wall reflects *part* and absorbs the rest.

The `(u − u_prev)` time-difference is the key: a purely spatial ghost can only reflect;
adding the time term is **what makes absorption real**.

The grid edge is padded as solid + rigid ([physics_solver.py:61](src/physics_solver.py:61))
so a room with no explicit border still behaves like a closed box (no wrap-around).

### 1.4 The source: a Ricker "clap"

A clap is a **zero-mean Ricker wavelet** (`add_impulse`,
[physics_solver.py:98](src/physics_solver.py:98)):

```
s(t) = amp · (1 − 2a) · e^(−a),   a = (π f (t − t₀))²,   default f = FREQ/2
```

Zero-mean matters: a plain Gaussian would dump net pressure into a lossless cavity and
excite its DC mode (a non-physical drift). The Ricker integrates to zero, so it injects a
clean broadband pulse. Sources expire after their pulse passes
([physics_solver.py:111](src/physics_solver.py:111)).

### 1.5 The energy readout

`energy()` returns `Σ u²` over the field ([physics_solver.py:161](src/physics_solver.py:161))
— a **proxy** for acoustic energy (cheap, monotone with "loudness"), not the exact
kinetic+potential energy. It is what the chart plots and what the auto-stop watches.

---

## 2. The numerics (FDTD)

**FDTD = Finite-Difference Time-Domain**: replace every derivative with a difference
quotient on the grid and march forward in time.

### 2.1 Discretizing the derivatives

- **Time** (3 levels — past `n−1`, present `n`, future `n+1`):
  `∂²u/∂t² ≈ (uⁿ⁺¹ − 2uⁿ + uⁿ⁻¹)/Δt²`
- **Space** (the **5-point Laplacian** on square cells, [physics_solver.py:133](src/physics_solver.py:133)):
  `∇²u ≈ (u_N + u_S + u_E + u_W − 4u)/Δx²`

Square cells (`dx = dy`, [grid.py:15](src/grid.py:15)) keep the stencil isotropic (waves
travel at the same speed in x and y).

### 2.2 The update rule

Solving the discretized PDE for the future value gives the three-level leapfrog the
solver runs every step ([physics_solver.py:135](src/physics_solver.py:135)):

```
uⁿ⁺¹ = A·uⁿ − B·uⁿ⁻¹ + C·L
```

with coefficients ([physics_solver.py:42](src/physics_solver.py:42)), where
`r = c·Δt/Δx` is the Courant number and `d = 1 + βΔt/2`:

```
A = (2 − βΔt)/d      B = (1 − βΔt/2)/d      C = r²/d
```

**The lossless case is the anchor.** At `β = 0` (the default, and the regime *every*
`verify.py` check runs in) this reduces exactly to the classic lossless leapfrog:

```
A = 2,   B = 1,   C = r²        →    uⁿ⁺¹ = 2uⁿ − uⁿ⁻¹ + r²·L
```

The `d = 1 + βΔt/2` denominator is the damping correction: turning `β` up shrinks `A`/`B`
so each step bleeds a little amplitude, producing the `≈ exp(−βt)` reverb tail. `L` is the
*unnormalised* stencil (`Δx²` is folded into `C` via `r²`).

### 2.3 Stability — the CFL condition

A leapfrog scheme blows up unless the timestep is small enough. In 2-D the limit is
([grid.py:24](src/grid.py:24), asserted at construction):

```
r = c·Δt/Δx  <  1/√2 ≈ 0.707
```

The project picks `CFL_FACTOR = 0.636` ([config.py:13](src/config.py:13)) — exactly 0.9×
the limit, a safe margin. `dt` is then derived from `dx`
([grid.py:16](src/grid.py:16)). `verify.py` re-checks both CFL stability and that the
field stays finite over 1500 steps.

### 2.4 Accuracy — points-per-wavelength & the role of FREQ

This is **why `FREQ` exists and why it is required.** A discrete grid can only carry waves
that are sampled finely enough; too-coarse waves suffer **numerical dispersion** (they
travel at the wrong speed and smear). The cure is the **points-per-wavelength** rule
`PPW ≳ 10`. So the grid is sized straight from a target frequency
([grid.py:14](src/grid.py:14)):

```
dx = (c / FREQ) / PPW          # cell size = wavelength / points-per-wavelength
```

with `FREQ = 440 Hz` ([config.py:5](src/config.py:5)) and `PPW = 12`
([config.py:10](src/config.py:10)). So **`FREQ` sets the whole discretization**:
`FREQ → dx → dt → NX×NY`. It also caps the **source bandwidth**: the clap is centred at
`FREQ/2` ([physics_solver.py:104](src/physics_solver.py:104)) so its energy sits well
inside the band the 12-points-per-wavelength grid can resolve. Put energy near/above
`FREQ` and you'd under-sample it and corrupt echo timing.

> **`FREQ` is a *resolution / design* frequency, not a played pitch.** It is the highest
> frequency you promise to resolve accurately; everything else (cell size, timestep, grid
> dimensions, clap bandwidth) follows from it. Remove it and the grid has no way to choose
> a cell size.

---

## 3. Architecture map

Strict layered imports — **each layer only imports from the ones above it**:

```
config.py        constants + MATERIALS (no project imports)
   │
grid.py          RoomGrid: geometry + FDTD params (dx, dt, r)        [grid.py:8]
   │
utils.py         coord↔cell, empty_field, pressure_reflection        [utils.py]
   │        ┌───────────────┴───────────────┐
room.py    │                          physics_solver.py
(α-map +    │                          (FDTD WaveSolver)
 pieces)   [room.py:47]                [physics_solver.py:27]
   │        └───────────────┬───────────────┘
scenes.py        room presets: two_rooms, tiny_home                  [scenes.py]
   │
render.py        field → pixels; the energy chart                    [render.py]
   │
visualiser.py    the interactive Pygame app                          [visualiser.py:100]

verify.py        drives every layer headlessly → PASS/FAIL           [verify.py]
```

`room.py` and `physics_solver.py` are **siblings** — neither imports the other. The solver
just consumes the `(NY, NX)` `α`-map a `Room` produces (`solver.set_alpha(room.alpha)`,
[physics_solver.py:74](src/physics_solver.py:74)). This is the physics/visuals split: one
side owns `room + physics_solver`, the other `render + visualiser`, on the shared
`config / grid / utils` base.

**Field convention:** all fields are `(NY, NX)` `float64` (row ↔ y, col ↔ x), built by
`empty_field` [utils.py:24](src/utils.py:24). Metres ↔ cells go through
`coord_to_cell` / `cell_to_coord` [utils.py:12](src/utils.py:12) (which clamp).

---

## 4. The visualiser — full map

`Visualiser` ([visualiser.py:100](src/visualiser.py:100)) is a white dashboard around the
physics layer. Main loop ([visualiser.py:609](src/visualiser.py:609)):
**`handle()` events → `update()` physics → `draw()` → `clock.tick(60)`**.

### 4.1 Window layout

```
┌──────────────────────────────┬───────────────┐
│                              │  SIDEBAR      │
│   CANVAS  (cw × ch)          │  (controls)   │   cw = NX·SCALE, ch = NY·SCALE
│   the pressure field         │               │   SCALE = 6 px / cell
│                              │               │
├──────────────────────────────┤               │
│   ENERGY CHART (CHART_H)     │               │
└──────────────────────────────┴───────────────┘
```

### 4.2 How the field is drawn (red-on-white)

`field_to_rgb` ([render.py:19](src/render.py:19)) maps pressure → colour:

- **Rest pressure (`u ≈ 0`) → white**; **compression (`u > 0`) → red** (the wavefront);
  rarefaction (`u < 0`) stays white. So a clap looks like red ripples on white.
- A **gamma lift** (`gamma = 0.6`) brightens faint fronts so late echoes stay visible.
- **`p_scale` auto-ranges** the brightness: `max(peak, p_scale·0.97, 1e-4)`
  ([visualiser.py:472](src/visualiser.py:472)) — it tracks the current peak but decays
  slowly, so the image neither saturates nor goes black as the clap fades.
- **Walls are tinted by material** via a precomputed `wall_rgb` map
  (`_rebuild_wall_rgb`, [visualiser.py:281](src/visualiser.py:281); colours
  `MATERIAL_COLORS` [visualiser.py:44](src/visualiser.py:44)).
- `to_surface` ([render.py:48](src/render.py:48)) flips rows so **y points up** (grid row
  0 at the window bottom) and transposes to Pygame's `(NX, NY)` order.

> Note: the module-level docstring at the top of `render.py` describes an *older*
> grayscale theme; the live behaviour (and the function docstring) is the red-on-white
> theme above — that's what the code does and what you see on screen.

### 4.3 Modes, tools, materials (the right sidebar)

Built in `_build_widgets` ([visualiser.py:152](src/visualiser.py:152)); clicks routed by
`_click_panel` ([visualiser.py:407](src/visualiser.py:407)).

| Control | What it does | Code |
|---------|--------------|------|
| **MODE: Source** | click the canvas to drop a clap | `add_impulse` [physics_solver.py:98](src/physics_solver.py:98) |
| **MODE: Edit** | use a tool + a material to modify the room | |
| **Tool: Paint** | recolour a piece — clicking *any* wall recolours **all** walls as a group; furniture is individual | `_paint_at` [visualiser.py:417](src/visualiser.py:417), `set_wall_material` [room.py:128](src/room.py:128) |
| **Tool: Wall** | drag to lay a wall; ends **auto-snap** to nearby walls so rooms close up | `_extend_wall` [visualiser.py:320](src/visualiser.py:320), `SNAP_CELLS` |
| **Tool: Block** | drag a furniture rectangle | `add_block` [room.py:89](src/room.py:89) |
| **Tool: Erase** | delete a piece (the outer shell is **protected/locked**) | `_erase_at` [visualiser.py:430](src/visualiser.py:430), `remove_piece` [room.py:137](src/room.py:137) |
| **MATERIAL swatches** | 10 presets, each labelled with its `α` | `MATERIALS` [config.py:21](src/config.py:21) |
| **AMPLITUDE ± / `[ ]`** | clap loudness (0.25–4.0) | `_amp` [visualiser.py:233](src/visualiser.py:233) |
| **SPEED ± / `, .`** | solver steps per frame (0–24) — sim speed | `_speed` [visualiser.py:236](src/visualiser.py:236) |
| **DAMPING β ±** | air damping / reverb decay (0–150) | `_damp` → `set_beta` [visualiser.py:239](src/visualiser.py:239) |
| **Pause / `Space`** | freeze the sim | `_toggle_pause` [visualiser.py:243](src/visualiser.py:243) |
| **Clear / `R`** | zero the field + energy history | `_reset` [visualiser.py:249](src/visualiser.py:249) |
| **Undo / `Ctrl+Z`**, **Revert** | undo edits (40-deep), or rebuild the default scene | `_undo`/`_revert` [visualiser.py:263](src/visualiser.py:263) |

After **any** geometry edit the app calls `_apply_geometry`
([visualiser.py:289](src/visualiser.py:289)) → `solver.set_alpha(...)`
([physics_solver.py:74](src/physics_solver.py:74)) + rebuild wall colours. Edits are snapshotted for undo (`Room.snapshot`/`restore`,
[room.py:146](src/room.py:146)).

### 4.4 Coordinates

`_cell_at(px, py)` ([visualiser.py:294](src/visualiser.py:294)) converts a screen pixel to
a grid `(row, col)`, **flipping y** (screen is y-down, grid is y-up). `_cell_metres`
converts a cell back to metres for `add_impulse`/`add_rectangle`.

### 4.5 The energy chart — *now with numeric axes* (this update)

Drawn by `draw_line_chart` ([render.py:55](src/render.py:55)), wired in `_draw_chart`
([visualiser.py:547](src/visualiser.py:547)):

- **x-axis = elapsed simulation time in ms**, the **full history from t = 0** (the curve
  compresses as the run grows). Each frame stores `(energy, time)` pairs in the parallel
  lists `self.energy` / `self.energy_t` ([visualiser.py:474](src/visualiser.py:474)).
- **y-axis = energy `Σu²`**, ticked `0 / vmax/2 / vmax`.
- The polyline is **downsampled to the pixel width** so a long run stays cheap to draw.
- Memory is bounded by halving the history past `ENERGY_MAX_SAMPLES` (keeps t = 0 visible).
- Live readout shows `now / peak / t`, plus a **`· SETTLED`** tag when auto-stopped.

### 4.6 Auto-stop on silence — *this update*

`_check_auto_stop` ([visualiser.py:482](src/visualiser.py:482)) pauses the sim once the
clap has fully died out (canvas all white again):

- It **arms** only after a real clap (peak energy `> STOP_PEAK_FLOOR`) and once **no source
  is still firing** (`solver.has_active_sources` False, [physics_solver.py:171](src/physics_solver.py:171)) — so it never trips mid-clap.
- It pauses when energy stays below `STOP_ENERGY_FRAC` of its **peak** for
  `STOP_HOLD_FRAMES` consecutive frames (~0.5 s of quiet).
- Because the threshold is **relative to the peak**, a lossless rigid room (`β=0`, energy
  conserved → never decays) correctly **never** auto-stops; only a genuinely decaying room
  does.
- Pressing `Space` or dropping a fresh clap **re-arms** it (clears `auto_stopped`); it is a
  resumable pause, not a quit.

---

## 5. Constants cheat-table

| Name | Value | Meaning | Where |
|------|-------|---------|-------|
| `C_SOUND` | 343 m/s | speed of sound | [config.py:4](src/config.py:4) |
| `FREQ` | 440 Hz | resolution frequency → grid size + clap band | [config.py:5](src/config.py:5) |
| `ROOM_W × ROOM_H` | 8 × 6 m | room size | [config.py:8](src/config.py:8) |
| `PPW` | 12 | points per wavelength (≥~10 limits dispersion) | [config.py:10](src/config.py:10) |
| `CFL_FACTOR` | 0.636 | Courant `r`; 0.9× the `1/√2` 2-D limit | [config.py:13](src/config.py:13) |
| `BETA` | 0 | air damping (app uses 30) | [config.py:16](src/config.py:16) |
| `MATERIALS` | α 0.02–1.0 | absorption per material | [config.py:21](src/config.py:21) |
| **derived** `dx` | ≈ 0.065 m | `(c/FREQ)/PPW` | [grid.py:14](src/grid.py:14) |
| **derived** `dt` | ≈ 120 µs | `CFL_FACTOR·dx/c` | [grid.py:16](src/grid.py:16) |
| **derived** `NX × NY` | 124 × 93 | `ceil(W/dx) × ceil(H/dx)` | [grid.py:18](src/grid.py:18) |

---

## 6. Talking points (anticipated questions)

- **"Why FDTD and not a frequency-domain / ray method?"** FDTD solves the wave equation
  directly in time, so it captures *reflection, diffraction through doorways, and decay* in
  one model and produces a movie — ideal for an intuitive echo demo.
- **"Is it stable?"** Yes — enforced by the CFL check `r < 1/√2`
  ([grid.py:24](src/grid.py:24)); `verify.py` confirms a bounded field over 1500 steps.
- **"How do you know it's correct?"** `verify.py` measures the **wavefront speed** (≈342
  vs 343 m/s), checks **energy conservation** in a rigid room (<0.4 % drift), and an
  **absorption ladder** (Concrete > Wood > Carpet > Foam retain decreasing energy).
- **"What does α = 1 (Open) mean?"** A perfectly absorbing (anechoic) wall — the Mur
  boundary lets waves leave with no reflection.
- **"What's on the energy chart axes?"** Time in ms (x) vs `Σu²` energy proxy (y); the auto-
  stop fires when that curve flattens near zero.
