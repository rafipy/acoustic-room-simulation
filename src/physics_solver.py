"""
physics_solver.py — 2D acoustic FDTD wave stepper.

Solves the damped wave equation  u_tt + beta*u_t = c^2 * lap(u)  on a uniform grid with
a 3-level time scheme and the 5-point Laplacian.  With r = c*dt/dx (Courant number):

    u_next = A*u_curr - B*u_prev + C*L ,
    A = (2 - beta*dt)/d,  B = (1 - beta*dt/2)/d,  C = r^2/d,  d = 1 + beta*dt/2.

beta=0 is the classic lossless leapfrog scheme; CFL stability needs r <= 1/sqrt(2).

Walls/furniture are cells with alpha > 0.  At an air/solid face the Laplacian uses a
ghost value  g = u - (1 - R_p)/r * (u - u_prev),  with R_p = sqrt(1 - alpha):  R_p=1 is
a rigid mirror (lossless), R_p=0 is a 1st-order Mur absorbing boundary, and the
(u - u_prev) term is what makes the absorption real (a purely spatial ghost would only
reflect).  The clap is a soft, zero-mean Ricker wavelet (a plain Gaussian would pump the
lossless cavity's DC mode).  Full derivation + error analysis: report/Final_Report.docx.
"""

import numpy as np

from config import FREQ
from grid import RoomGrid
from utils import empty_field, pressure_reflection, coord_to_cell


class WaveSolver:
    """Holds the pressure history and advances the FDTD scheme one step at a time."""

    def __init__(self, grid: RoomGrid, alpha, beta: float | None = None):
        self.g = grid
        self.alpha = np.asarray(alpha, dtype=np.float64)
        if self.alpha.shape != (grid.NY, grid.NX):
            raise ValueError(f"alpha map shape {self.alpha.shape} != grid ({grid.NY}, {grid.NX})")

        self._build_walls()

        beta = grid.beta if beta is None else beta
        self.beta = beta
        dt = grid.dt
        denom = 1.0 + beta * dt / 2.0
        self.A = (2.0 - beta * dt) / denom
        self.B = (1.0 - beta * dt / 2.0) / denom
        self.C = (grid.r ** 2) / denom

        self.u_prev = empty_field(grid)                  # u^{n-1}
        self.u_curr = empty_field(grid)                  # u^n
        self.t = 0.0
        self.n = 0
        self._sources = []                               # (row, col, signal, t_expire)

    # ── wall fields ───────────────────────────────────────────────────────────
    def _build_walls(self):
        """(Re)derive obstacle masks, the reflection field, and the Mur weights from alpha."""
        self.is_solid = self.alpha > 0.0
        self.is_air = ~self.is_solid
        self.Rp = pressure_reflection(self.alpha)

        # Pre-shift the static neighbour solidity/reflection fields once.  Padding the edge
        # as solid+rigid makes a missing border act as a closed wall and avoids wrap-around.
        solid_pad = np.pad(self.is_solid, 1, mode="constant", constant_values=True)
        Rp_pad = np.pad(self.Rp, 1, mode="constant", constant_values=1.0)
        self._solidN, self._RpN = solid_pad[2:, 1:-1], Rp_pad[2:, 1:-1]
        self._solidS, self._RpS = solid_pad[:-2, 1:-1], Rp_pad[:-2, 1:-1]
        self._solidE, self._RpE = solid_pad[1:-1, 2:], Rp_pad[1:-1, 2:]
        self._solidW, self._RpW = solid_pad[1:-1, :-2], Rp_pad[1:-1, :-2]

        r = self.g.r
        self._kN = (1.0 - self._RpN) / r
        self._kS = (1.0 - self._RpS) / r
        self._kE = (1.0 - self._RpE) / r
        self._kW = (1.0 - self._RpW) / r

    def set_alpha(self, alpha):
        """Swap in a new alpha-map and rebuild the wall fields, keeping the pressure
        history (pressure in newly-solid cells is cleared)."""
        alpha = np.asarray(alpha, dtype=np.float64)
        if alpha.shape != (self.g.NY, self.g.NX):
            raise ValueError(f"alpha shape {alpha.shape} != grid ({self.g.NY}, {self.g.NX})")
        self.alpha = alpha
        self._build_walls()
        self.u_curr *= self.is_air
        self.u_prev *= self.is_air
        return self

    def set_beta(self, beta):
        """Change the air-damping coefficient at runtime (recomputes A, B, C).
        beta > 0 makes the room reverberate and fall silent; beta = 0 rings forever."""
        self.beta = beta
        dt = self.g.dt
        d = 1.0 + beta * dt / 2.0
        self.A = (2.0 - beta * dt) / d
        self.B = (1.0 - beta * dt / 2.0) / d
        self.C = (self.g.r ** 2) / d
        return self

    # ── source ────────────────────────────────────────────────────────────────
    def add_impulse(self, x, y, amp=1.0, freq=None, t0=None):
        """Add a soft 'clap' at metre (x, y) — a zero-mean Ricker wavelet.  Chainable.
        Fires relative to the current time, so clicks mid-run work."""
        row, col = coord_to_cell(x, y, self.g)
        if freq is None:
            freq = FREQ / 2.0
        if t0 is None:
            t0 = self.t + 1.0 / freq

        def signal(t, f=freq, t0=t0, amp=amp):
            a = (np.pi * f * (t - t0)) ** 2
            return amp * (1.0 - 2.0 * a) * np.exp(-a)

        self._sources.append((row, col, signal, t0 + 3.0 / freq))   # last term = expiry time
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
        du = u - u_prev                                  # ~ dt * u_t (the boundary loss term)
        up = np.pad(u, 1, mode="edge")
        north, south = up[2:, 1:-1], up[:-2, 1:-1]
        east, west = up[1:-1, 2:], up[1:-1, :-2]
        north = np.where(self._solidN, u - self._kN * du, north)
        south = np.where(self._solidS, u - self._kS * du, south)
        east = np.where(self._solidE, u - self._kE * du, east)
        west = np.where(self._solidW, u - self._kW * du, west)
        return north + south + east + west - 4.0 * u

    def step(self):
        """Advance one timestep; returns the new pressure field."""
        L = self._laplacian(self.u_curr, self.u_prev)
        u_next = self.A * self.u_curr - self.B * self.u_prev + self.C * L
        u_next *= self.is_air                            # obstacles stay 0

        t_next = self.t + self.g.dt
        for row, col, signal, _exp in self._sources:
            if self.is_air[row, col]:
                u_next[row, col] += signal(t_next)
        self.u_prev, self.u_curr = self.u_curr, u_next
        self.t = t_next
        self.n += 1
        if self._sources:                                # drop sources whose pulse has passed
            self._sources = [s for s in self._sources if s[3] > self.t]
        return self.u_curr

    def run(self, n_steps, record_energy=False):
        """Advance n_steps.  With record_energy, return the per-step energies."""
        energies = []
        for _ in range(n_steps):
            self.step()
            if record_energy:
                energies.append(self.energy())
        return energies

    def energy(self):
        """A proxy for acoustic energy: sum of u^2 over the field."""
        return float(np.sum(self.u_curr ** 2))

    @property
    def field(self):
        """Current pressure field (NY, NX) — what a visualiser renders."""
        return self.u_curr

    @property
    def has_active_sources(self) -> bool:
        """True while a clap's pulse is still being injected (its expiry is in the future).
        A frontend gates auto-stop on this so it never pauses mid-clap."""
        return bool(self._sources)
