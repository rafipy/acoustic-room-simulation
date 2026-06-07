"""
physics_solver.py — 2D acoustic FDTD wave stepper (the "physics" layer).

================================================================================
WHAT THIS SOLVES
================================================================================
The damped scalar wave equation for acoustic pressure u(x, y, t):

        u_tt + β·u_t = c² · (u_xx + u_yy)

  u      acoustic pressure (arbitrary units)
  c      speed of sound (config.C_SOUND = 343 m/s)
  β      air (volumetric) damping; default 0 -> lossless air, so ALL energy loss
         happens at the walls/furniture.

================================================================================
DISCRETISATION  (Finite Difference Time Domain)
================================================================================
Time : 3-level central differences on a uniform step dt.
Space: the 5-point Laplacian stencil on a uniform grid (dx = dy).

  u_tt  ≈ (uⁿ⁺¹ − 2 uⁿ + uⁿ⁻¹) / dt²
  u_t   ≈ (uⁿ⁺¹ − uⁿ⁻¹) / (2 dt)                         (central)
  ∇²u   ≈ L / dx²,   L = u_E + u_W + u_N + u_S − 4 u      (5-point stencil)

Substituting and solving for the only future term uⁿ⁺¹ gives one explicit update.
With the Courant number r = c·dt/dx:

  denom = 1 + β·dt/2
  A = (2 − β·dt) / denom
  B = (1 − β·dt/2) / denom
  C = r² / denom
  uⁿ⁺¹ = A·uⁿ − B·uⁿ⁻¹ + C·L

  β = 0  ⇒  A=2, B=1, C=r²  ⇒  uⁿ⁺¹ = 2 uⁿ − uⁿ⁻¹ + r² L
                                (the classic lossless leapfrog scheme).

Stability (CFL): stable for r ≤ 1/√2 ≈ 0.707 in 2D.  RoomGrid sets r = CFL_FACTOR
and asserts this on construction.

================================================================================
WALLS & FURNITURE  (absorbing-boundary model)
================================================================================
Obstacles are cells with α > 0 in the material map (see room.py).  Each maps to a
target PRESSURE reflection coefficient

        R_p = √(1 − α)            (utils.pressure_reflection)

so the intended reflected-energy fraction is R_p² = 1 − α and the absorbed fraction
is 1 − R_p² = α (the tabulated absorption coefficient).

Implementation — a per-face "ghost" rule when building the Laplacian.  When an air
cell's neighbour is SOLID, that neighbour contributes a ghost value g in place of its
own (unused) stored value:

        g = u  −  (1 − R_p)/r · (u − u_prev)      (u, u_prev at THIS cell; r = Courant no.)

  - R_p = 1 (α=0):  g = u                     -> rigid mirror (Neumann), LOSSLESS.
  - R_p = 0 (α=1):  g = u − (1/r)(u − u_prev)  -> 1st-order Mur radiating boundary:
                                                 the wave leaves and does not return.
  - 0 < R_p < 1:    a blend of the two        -> a weaker echo; energy is removed.

WHY the (u − u_prev) term?  A purely spatial scaling of the ghost (g = R_p·u) is a
REACTIVE boundary: it changes the echo's amplitude/phase but CONSERVES energy, so it
cannot absorb (g = 0 is a pressure-release wall — 100% reflection, inverted).  Genuine
absorption needs the boundary to couple to the wave's motion, the time derivative
u_t ≈ (u − u_prev)/dt.  The term above is exactly that outgoing-wave (impedance)
condition, scaled by (1 − R_p) so a fully reflecting wall switches it off.

Exact at both limits and monotonic between (more absorptive -> weaker echo); like all
real-time impedance boundaries it is approximate at oblique incidence.  Solid cells are
never advanced (held at 0); only their R_p is read, through the ghost.  The outer array
edge is rigid by default, so neighbour shifts (edge replication) never wrap.

================================================================================
SOURCE  (the "clap")
================================================================================
A soft (additive) point source injected as   uⁿ⁺¹[src] += s(t).
"Soft" (added) rather than "hard" (overwritten) so the source point does not act as
a hard scatterer of waves passing back through it.

s(t) is a Ricker wavelet — the zero-mean form of a Gaussian pulse (its 2nd derivative):

        a    = (π·f·(t − t0))²
        s(t) = amp · (1 − 2a) · exp(−a)

Why zero-mean?  A plain (all-positive) Gaussian injects a NET volume of air.  In a
closed, lossless (β=0) room that pumps the spatially-uniform k=0 mode, which has no
restoring force (∂²U/∂t² = 0): its amplitude drifts linearly and the total energy
grows without bound — a real artifact, not a coding bug.  A Ricker pulse integrates
to zero, adds no net volume, and leaves that DC mode unexcited.  Its peak frequency f
defaults to FREQ/2 (low enough that PPW resolves the pulse, limiting dispersion);
t0 = 1/f delays the peak so injection ramps smoothly from ≈0.
================================================================================
"""

import numpy as np

from config import C_SOUND, FREQ
from grid import RoomGrid
from utils import empty_field, pressure_reflection, coord_to_cell


class WaveSolver:
    """Holds the pressure history and advances the FDTD scheme one step at a time."""

    def __init__(self, grid: RoomGrid, alpha, beta: float | None = None):
        self.g = grid
        self.alpha = np.asarray(alpha, dtype=np.float64)
        if self.alpha.shape != (grid.NY, grid.NX):
            raise ValueError(
                f"alpha map shape {self.alpha.shape} != grid (NY, NX) "
                f"({grid.NY}, {grid.NX})"
            )

        self.is_solid = self.alpha > 0.0
        self.is_air = ~self.is_solid
        self.Rp = pressure_reflection(self.alpha)        # (NY, NX) reflection field

        # Pre-shift the STATIC neighbour solidity / reflection fields once.  Padding
        # the domain edge as solid+rigid (True, R_p=1) makes a missing border behave
        # as a closed wall and guarantees shifts never wrap around.
        solid_pad = np.pad(self.is_solid, 1, mode="constant", constant_values=True)
        Rp_pad = np.pad(self.Rp, 1, mode="constant", constant_values=1.0)
        #            north = row+1,  south = row-1,  east = col+1,  west = col-1
        self._solidN, self._RpN = solid_pad[2:, 1:-1], Rp_pad[2:, 1:-1]
        self._solidS, self._RpS = solid_pad[:-2, 1:-1], Rp_pad[:-2, 1:-1]
        self._solidE, self._RpE = solid_pad[1:-1, 2:], Rp_pad[1:-1, 2:]
        self._solidW, self._RpW = solid_pad[1:-1, :-2], Rp_pad[1:-1, :-2]

        # Per-face outgoing-wave (Mur) weights  k = (1 - R_p_neighbour)/r — these turn
        # the reflective ghost into a dissipative one (see the WALLS docstring above).
        self._kN = (1.0 - self._RpN) / grid.r
        self._kS = (1.0 - self._RpS) / grid.r
        self._kE = (1.0 - self._RpE) / grid.r
        self._kW = (1.0 - self._RpW) / grid.r

        # Update coefficients (β default from the grid)
        beta = grid.beta if beta is None else beta
        self.beta = beta
        dt = grid.dt
        denom = 1.0 + beta * dt / 2.0
        self.A = (2.0 - beta * dt) / denom
        self.B = (1.0 - beta * dt / 2.0) / denom
        self.C = (grid.r ** 2) / denom

        # Pressure history (two persistent buffers)
        self.u_prev = empty_field(grid)                  # uⁿ⁻¹
        self.u_curr = empty_field(grid)                  # uⁿ

        self.t = 0.0
        self.n = 0
        self._sources = []                               # (row, col, signal_callable)

    # ── source ────────────────────────────────────────────────────────────────
    def add_impulse(self, x, y, amp=1.0, freq=None, t0=None):
        """Add a soft 'clap' at metre position (x, y) — a zero-mean Ricker wavelet.

        Chainable.  `freq` is the wavelet's peak frequency (default FREQ/2, kept low
        enough that PPW resolves the pulse); `t0` delays the peak so injection ramps
        up smoothly from ≈0 (default 1/freq).
        """
        row, col = coord_to_cell(x, y, self.g)
        if freq is None:
            freq = FREQ / 2.0
        if t0 is None:
            t0 = self.t + 1.0 / freq      # peak shortly after "now" (correct mid-run too)

        def signal(t, f=freq, t0=t0, amp=amp):
            a = (np.pi * f * (t - t0)) ** 2
            return amp * (1.0 - 2.0 * a) * np.exp(-a)

        self._sources.append((row, col, signal))
        return self

    def reset(self):
        """Zero the field and clock (keeps geometry and sources)."""
        self.u_prev[...] = 0.0
        self.u_curr[...] = 0.0
        self.t = 0.0
        self.n = 0
        return self

    # ── core step ─────────────────────────────────────────────────────────────
    def _laplacian(self, u, u_prev):
        """5-point Laplacian with the absorbing-boundary ghost at solid neighbours."""
        du = u - u_prev                                  # ~ dt * u_t (boundary loss term)
        up = np.pad(u, 1, mode="edge")                   # replicate edges (no wrap)
        north = up[2:, 1:-1]
        south = up[:-2, 1:-1]
        east = up[1:-1, 2:]
        west = up[1:-1, :-2]
        # a solid neighbour returns the impedance ghost  u - k*(u - u_prev)
        north = np.where(self._solidN, u - self._kN * du, north)
        south = np.where(self._solidS, u - self._kS * du, south)
        east = np.where(self._solidE, u - self._kE * du, east)
        west = np.where(self._solidW, u - self._kW * du, west)
        return north + south + east + west - 4.0 * u

    def step(self):
        """Advance one timestep; returns the new pressure field (u_curr)."""
        L = self._laplacian(self.u_curr, self.u_prev)
        u_next = self.A * self.u_curr - self.B * self.u_prev + self.C * L
        u_next *= self.is_air                            # obstacles stay 0

        t_next = self.t + self.g.dt
        for row, col, signal in self._sources:
            if self.is_air[row, col]:
                u_next[row, col] += signal(t_next)       # soft (additive) source

        self.u_prev, self.u_curr = self.u_curr, u_next
        self.t = t_next
        self.n += 1
        return self.u_curr

    def run(self, n_steps, record_energy=False):
        """Advance n_steps.  With record_energy, return per-step energies."""
        energies = []
        for _ in range(n_steps):
            self.step()
            if record_energy:
                energies.append(self.energy())
        return energies

    # ── diagnostics ─────────────────────────────────────────────────────────
    def energy(self):
        """A proxy for acoustic energy: Σ u² over the field."""
        return float(np.sum(self.u_curr ** 2))

    @property
    def field(self):
        """Current pressure field (NY, NX) — what a visualiser would render."""
        return self.u_curr


# ── Verification smoke test ───────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    from room import Room

    def banner(title):
        print("\n" + "=" * 66 + f"\n{title}\n" + "=" * 66)

    grid = RoomGrid()
    print(grid)
    cfl_limit = 1.0 / np.sqrt(2.0)
    print(f"CFL: r = {grid.r:.4f}  (limit 1/sqrt(2) = {cfl_limit:.4f})  -> "
          f"{'OK' if grid.r < cfl_limit else 'VIOLATED'}")

    SRC = (4.0, 3.0)

    # Test 1 — stability: rigid box (all-air, rigid padding), long run ----------
    banner("Test 1 - stability (rigid box, 1500 steps)")
    s = WaveSolver(grid, Room(grid).alpha)               # all air -> rigid padding
    s.add_impulse(*SRC)
    s.run(1500)
    mx = float(np.max(np.abs(s.field)))
    finite = bool(np.all(np.isfinite(s.field)))
    print(f"steps={s.n}  max|u|={mx:.4f}  finite={finite}")
    print("PASS" if finite and mx < 100 else "FAIL")

    # Test 2 — energy: rigid conserves (windowed mean), absorbing decays ---------
    banner("Test 2 - energy: rigid conserves vs absorbing decays")
    rs = WaveSolver(grid, Room(grid).alpha)              # rigid
    rs.add_impulse(*SRC)
    rs.run(400)                                          # finish injecting
    e_early = float(np.mean(rs.run(400, record_energy=True)))
    e_late = float(np.mean(rs.run(400, record_energy=True)))
    drift = abs(e_late - e_early) / e_early
    print(f"rigid:     mean E early={e_early:.3e}  late={e_late:.3e}  drift={drift*100:.2f}%")

    ab = WaveSolver(grid, Room(grid).add_border("Acoustic Foam").alpha)
    ab.add_impulse(*SRC)
    ab.run(400)
    a_early = float(np.mean(ab.run(400, record_energy=True)))
    a_late = float(np.mean(ab.run(400, record_energy=True)))
    print(f"absorbing: mean E early={a_early:.3e}  late={a_late:.3e}  ratio={a_late/a_early:.3f}")
    print("PASS" if drift < 0.10 and a_late < 0.5 * a_early else "FAIL")

    # Test 3 — leading wavefront travels at ~c (anechoic frame, no reflections) ---
    # Measure the front position along the +x ray at two times; the SPEED (Δr/Δt)
    # cancels the source's emission delay and the 2D wake behind the front.
    banner("Test 3 - wavefront speed ~ c (anechoic box)")
    free = WaveSolver(grid, Room(grid).add_border("Open").alpha)
    free.add_impulse(*SRC, freq=FREQ)                    # short pulse, finishes fast
    row_src, col_src = coord_to_cell(*SRC, grid)

    def front_radius(solver, thr_frac=0.01):
        line = np.abs(solver.field[row_src, col_src:])   # +x ray from the source
        thr = thr_frac * float(np.max(np.abs(solver.field)))
        idx = np.where(line > thr)[0]
        return idx.max() * solver.g.dx if len(idx) else 0.0

    free.run(45); r1, t1 = front_radius(free), free.t
    free.run(30); r2, t2 = front_radius(free), free.t
    speed = (r2 - r1) / (t2 - t1)
    print(f"front: t1={t1*1e3:.2f} ms -> {r1:.3f} m;  t2={t2*1e3:.2f} ms -> {r2:.3f} m")
    print(f"measured speed = {speed:.1f} m/s   (c = {C_SOUND:.1f} m/s)   "
          f"ratio = {speed/C_SOUND:.3f}")
    print("PASS" if 0.85 < speed / C_SOUND < 1.10 else "FAIL")

    # Test 4 — more absorbing wall -> less energy retained -----------------------
    banner("Test 4 - retained energy decreases with absorption")
    def retained(material):
        sv = WaveSolver(grid, Room(grid).add_border(material).alpha)
        sv.add_impulse(*SRC)
        sv.run(400)                                      # finish injecting
        e0 = float(np.mean(sv.run(200, record_energy=True)))
        sv.run(1000)
        e1 = float(np.mean(sv.run(200, record_energy=True)))
        return e1 / e0 if e0 > 0 else 0.0

    prev = None
    mono = True
    for m in ("Concrete", "Wood", "Carpet", "Acoustic Foam"):
        val = retained(m)
        if prev is not None and val > prev + 1e-6:
            mono = False
        prev = val
        print(f"  {m:>14}: energy retained after bounces = {val:.3f}")
    print("PASS" if mono else "FAIL")
