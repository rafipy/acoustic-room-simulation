"""Interior truncation-error diagnostic for the acoustic FDTD solver.

Run:
    cd src
    python error.py

This is an analysis script, not part of the real-time app. It uses the same
project grid constants, compares the interior finite-difference stencil against
a known exact plane wave, and prints a compact convergence table suitable for a
report or presentation.

What it demonstrates:
Interior truncation error:
compare the finite-difference stencil against a known exact plane wave.
"""

import math

import numpy as np

from config import C_SOUND, FREQ
from grid import RoomGrid


def _rms(a):
    """Root-mean-square: turns a whole field of errors into one representative size."""
    return float(np.sqrt(np.mean(np.asarray(a, dtype=float) ** 2)))


def _plane_wave(grid, t, *, freq=FREQ, angle_deg=31.0):
    """Smooth exact solution of p_tt = c^2 Laplacian(p).

    We use a known exact wave instead of the solver's own output because error
    analysis needs something to compare against. For this wave, the continuous
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

    This returns neighbour_sum - 4*center. Dividing by dx^2 converts it into
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
    1. Build a grid. Larger PPW means smaller dx and dt.
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

        # For an exact solution the continuous value below is zero. The
        # nonzero discrete value is the finite-difference truncation error.
        residual = time_second - (C_SOUND ** 2) * space_second

        # Normalize by a characteristic exact acceleration size. This makes
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


def print_truncation_error():
    print("\nInterior truncation error")
    print("Exact test wave: oblique smooth plane wave, no boundary cells included.")
    print("Residual form: D_tt(p) - c^2 D_xx+yy(p), normalized by exact |p_tt|.")
    print("Expected trend: second-order stencil gives about 4x less residual when dx halves.\n")
    print(f"{'PPW':>5}  {'dx (cm)':>9}  {'dt (us)':>9}  {'rel RMS residual':>18}  {'order':>7}")
    for ppw, dx, dt, rel, order in truncation_error_rows():
        order_text = "-" if order is None else f"{order:7.2f}"
        print(f"{ppw:5d}  {100.0 * dx:9.3f}  {1e6 * dt:9.3f}  {rel:18.3e}  {order_text}")


def main():
    print("Acoustic FDTD interior truncation-error analysis")
    print_truncation_error()


if __name__ == "__main__":
    main()
