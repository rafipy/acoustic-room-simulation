"""Numerical error and energy diagnostics for the acoustic FDTD solver.

Run:
    cd src
    python error.py

This is an analysis script, not part of the real-time app.  It uses the same
project grid constants and solver, then prints compact tables suitable for a
report or presentation.

What it demonstrates:
1. Interior truncation error:
   compare the finite-difference stencil against a known exact plane wave.
2. CFL stability:
   show that the project Courant number is below the 2D stability limit.
3. Energy proxy:
   compare the app's simple sum(p^2) chart against a fuller acoustic-energy
   estimate at one snapshot.
"""

import math

import numpy as np

from config import C_SOUND, FREQ
from grid import RoomGrid
from physics_solver import WaveSolver
from room import Room


RHO_AIR = 1.204          # kg/m^3, dry air near room temperature
SRC = (4.0, 3.0)


def _rms(a):
    """Root-mean-square: turns a whole field of errors into one representative size."""
    return float(np.sqrt(np.mean(np.asarray(a, dtype=float) ** 2)))


def _plane_wave(grid, t, *, freq=FREQ, angle_deg=31.0):
    """Smooth exact solution of p_tt = c^2 Laplacian(p).

    We use a known exact wave instead of the solver's own output because error
    analysis needs something to compare against.  For this wave, the continuous
    PDE residual is exactly zero:

        p_tt - c^2 (p_xx + p_yy) = 0

    Any nonzero residual after applying the discrete stencil is therefore
    numerical truncation error from dx and dt.
    """
    angle = math.radians(angle_deg)
    k = 2.0 * math.pi * freq / C_SOUND
    kx = k * math.cos(angle)
    ky = k * math.sin(angle)
    omega = C_SOUND * k

    x = np.arange(grid.NX, dtype=float) * grid.dx
    y = np.arange(grid.NY, dtype=float) * grid.dy
    X, Y = np.meshgrid(x, y)
    return np.sin(kx * X + ky * Y - omega * t)


def _laplacian_count(u):
    """The dimensionless five-point Laplacian used by WaveSolver.step().

    This returns neighbour_sum - 4*center.  Dividing by dx^2 converts it into
    the finite-difference approximation of p_xx + p_yy.
    """
    return (
        u[2:, 1:-1]
        + u[:-2, 1:-1]
        + u[1:-1, 2:]
        + u[1:-1, :-2]
        - 4.0 * u[1:-1, 1:-1]
    )


def truncation_error_rows(ppws=(6, 12, 24, 48)):
    """Measure local PDE residual of the discrete interior stencil on an exact wave.

    Process for each PPW:
    1. Build a grid.  Larger PPW means smaller dx and dt.
    2. Sample the exact wave at t-dt, t, and t+dt.
    3. Approximate p_tt with the central time difference:

           D_tt(p) = (p(t+dt) - 2p(t) + p(t-dt)) / dt^2

    4. Approximate p_xx + p_yy with the five-point spatial stencil:

           D_xx+yy(p) = (N + S + E + W - 4C) / dx^2

    5. Compute the discrete PDE residual:

           residual = D_tt(p) - c^2 D_xx+yy(p)

       The exact PDE residual is zero, so this residual is the local
       truncation error.

    6. Normalize by the size of exact p_tt so the result is dimensionless.
    7. Compare residuals across dx values to estimate observed order.
    """
    rows = []
    prev = None
    for ppw in ppws:
        grid = RoomGrid(ppw=ppw)
        t = 0.37 / FREQ
        u_prev = _plane_wave(grid, t - grid.dt)
        u_now = _plane_wave(grid, t)
        u_next_exact = _plane_wave(grid, t + grid.dt)

        time_second = (
            u_next_exact[1:-1, 1:-1]
            - 2.0 * u_now[1:-1, 1:-1]
            + u_prev[1:-1, 1:-1]
        ) / (grid.dt ** 2)
        space_second = _laplacian_count(u_now) / (grid.dx ** 2)

        # For an exact solution the continuous value below is zero.  The
        # nonzero discrete value is the finite-difference truncation error.
        residual = time_second - (C_SOUND ** 2) * space_second

        # Normalize by a characteristic exact acceleration size.  This makes
        # "rel RMS residual" independent of the arbitrary wave amplitude.
        omega = 2.0 * math.pi * FREQ
        reference = (omega ** 2) * _rms(u_now[1:-1, 1:-1])
        rel = _rms(residual) / max(reference, 1e-15)

        order = None
        if prev is not None:
            prev_dx, prev_rel = prev
            # If error ~= C*dx^p, then
            # p = log(error_old/error_new) / log(dx_old/dx_new).
            order = math.log(prev_rel / rel) / math.log(prev_dx / grid.dx)
        rows.append((ppw, grid.dx, grid.dt, rel, order))
        prev = (grid.dx, rel)
    return rows


def _max_leapfrog_amplification(r):
    """Worst-case von Neumann amplification for the 2D lossless leapfrog update.

    CFL stability asks whether grid Fourier modes stay bounded after each
    timestep.  Stable modes have amplification magnitude <= 1.  If the Courant
    number r is too large, the highest-frequency grid mode has amplification
    > 1, so numerical noise grows instead of oscillating.
    """
    q_max = 2.0  # sin^2(kx dx/2) + sin^2(ky dy/2), maximum on a square grid
    a = 2.0 - 4.0 * (r ** 2) * q_max
    if abs(a) <= 2.0:
        return 1.0
    return 0.5 * (abs(a) + math.sqrt(a * a - 4.0))


def cfl_rows():
    """Return project and comparison Courant numbers for the stability table."""
    grid = RoomGrid()
    limit = 1.0 / math.sqrt(2.0)
    cases = [
        ("project", grid.r),
        ("99% of limit", 0.99 * limit),
        ("101% of limit", 1.01 * limit),
        ("clearly unstable", 0.80),
    ]
    rows = []
    for name, r in cases:
        amp = _max_leapfrog_amplification(r)
        margin = limit - r
        status = "stable" if amp <= 1.0 + 1e-12 else "unstable growth"
        rows.append((name, r, margin, amp, status))
    return limit, rows


def energy_snapshot(steps=120, amp=1.0):
    """Compare the app's sum(p^2) proxy with estimated acoustic energy.

    The solver stores pressure only.  To estimate full acoustic energy at one
    instant, this integrates particle velocity from the pressure-gradient
    momentum equation v_t = -grad(p)/rho during the same pressure run.

    Because the project is 2D, the Joule values are per metre of out-of-plane
    depth.  The visualizer's proxy is intentionally unitless.
    """
    grid = RoomGrid()
    room = Room(grid)
    solver = WaveSolver(grid, room.alpha, beta=0.0)
    solver.add_impulse(*SRC, amp=amp, freq=FREQ)

    vx = np.zeros((grid.NY, grid.NX), dtype=float)
    vy = np.zeros((grid.NY, grid.NX), dtype=float)
    for _ in range(steps):
        solver.step()

        # Linear acoustics momentum equation:
        #
        #     dv/dt = -grad(p) / rho
        #
        # The solver does not store particle velocity, so this reconstructs an
        # approximate velocity field from the pressure gradient just for this
        # one diagnostic comparison.
        dpdy, dpdx = np.gradient(solver.field, grid.dy, grid.dx, edge_order=2)
        vx -= (grid.dt / RHO_AIR) * dpdx
        vy -= (grid.dt / RHO_AIR) * dpdy

    p = solver.field
    cell_area = grid.dx * grid.dy

    # Visualizer proxy: fast and monotonic enough for a decay chart, but unitless.
    proxy = float(np.sum(p ** 2))
    proxy_area = proxy * cell_area

    # Acoustic energy density has two parts:
    # pressure potential energy density = p^2 / (2 rho c^2)
    # kinetic energy density = 0.5 rho |v|^2
    # Multiplying by cell area gives J/m because this is a 2D slice.
    potential = float(np.sum((p ** 2) / (2.0 * RHO_AIR * C_SOUND ** 2)) * cell_area)
    kinetic = float(np.sum(0.5 * RHO_AIR * (vx ** 2 + vy ** 2)) * cell_area)
    total = potential + kinetic
    return {
        "steps": steps,
        "time": solver.t,
        "proxy": proxy,
        "proxy_area": proxy_area,
        "potential": potential,
        "kinetic": kinetic,
        "total": total,
        "potential_fraction": potential / total if total > 0.0 else float("nan"),
        "proxy_to_potential": potential / proxy if proxy > 0.0 else float("nan"),
    }


def print_truncation_error():
    print("\nInterior truncation error")
    print("Exact test wave: oblique smooth plane wave, no boundary cells included.")
    print("Residual form: D_tt(p) - c^2 D_xx+yy(p), normalized by exact |p_tt|.")
    print("Expected trend: second-order stencil gives about 4x less residual when dx halves.\n")
    print(f"{'PPW':>5}  {'dx (cm)':>9}  {'dt (us)':>9}  {'rel RMS residual':>18}  {'order':>7}")
    for ppw, dx, dt, rel, order in truncation_error_rows():
        order_text = "-" if order is None else f"{order:7.2f}"
        print(f"{ppw:5d}  {100.0 * dx:9.3f}  {1e6 * dt:9.3f}  {rel:18.3e}  {order_text}")


def print_cfl_stability():
    limit, rows = cfl_rows()
    print("\nCFL stability error")
    print(f"2D square-grid leapfrog limit: r <= 1/sqrt(2) = {limit:.6f}")
    print("If r exceeds this, high-frequency grid modes amplify instead of oscillating.\n")
    print(f"{'case':<18}  {'r':>8}  {'limit-r':>10}  {'max amp/step':>13}  status")
    for name, r, margin, amp, status in rows:
        print(f"{name:<18}  {r:8.4f}  {margin:10.4f}  {amp:13.6f}  {status}")


def print_energy_snapshot():
    e = energy_snapshot()
    print("\nEnergy proxy vs estimated acoustic energy")
    print("Snapshot uses one pressure run plus reconstructed velocity from v_t = -grad(p)/rho.")
    print("Joule values are per metre depth because the simulation is 2D.\n")
    print(f"time: {1e3 * e['time']:.2f} ms  steps: {e['steps']}")
    print(f"visual proxy sum(p^2):              {e['proxy']:.6e}  arbitrary units")
    print(f"area-weighted sum(p^2):             {e['proxy_area']:.6e}  Pa^2 m^2")
    print(f"pressure potential energy:          {e['potential']:.6e}  J/m")
    print(f"estimated kinetic energy:           {e['kinetic']:.6e}  J/m")
    print(f"estimated total acoustic energy:    {e['total']:.6e}  J/m")
    print(f"pressure-potential share of total:  {100.0 * e['potential_fraction']:.2f}%")
    print(f"proxy-to-potential conversion:      {e['proxy_to_potential']:.6e}  (J/m) per proxy unit")
    print("\nInterpretation: the app's energy chart is a relative pressure-energy proxy.")
    print("It tracks decay well, but it is not the full acoustic energy because kinetic")
    print("particle-velocity energy is not stored by the pressure-only solver.")


def main():
    print("Acoustic FDTD numerical error analysis")
    print_truncation_error()
    print_cfl_stability()
    print_energy_snapshot()


if __name__ == "__main__":
    main()
