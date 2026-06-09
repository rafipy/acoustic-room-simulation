"""RoomGrid — room geometry and the derived FDTD parameters (dx, dt, Courant number)."""

import numpy as np

from config import C_SOUND, FREQ, ROOM_W, ROOM_H, PPW, CFL_FACTOR, BETA, MATERIALS


class RoomGrid:
    """Grid geometry + FDTD parameters.  Fields are (NY, NX) arrays (row<->y, col<->x)."""

    def __init__(self, width=ROOM_W, height=ROOM_H, ppw=PPW,
                 alpha=MATERIALS["Concrete"], beta=BETA):
        self.W, self.H = width, height
        self.dx = (C_SOUND / FREQ) / ppw       # cell size = wavelength / points-per-wavelength
        self.dy = self.dx                      # square cells keep the 5-point stencil isotropic
        self.dt = CFL_FACTOR * self.dx / C_SOUND
        self.r = C_SOUND * self.dt / self.dx   # Courant number (consumed by the solver)
        self.NX = int(np.ceil(self.W / self.dx))
        self.NY = int(np.ceil(self.H / self.dy))
        self.alpha = alpha                     # default outer-wall material (Room.add_border)
        self.beta = beta                       # air damping (0 = lossless)
        self._validate_cfl()

    def _validate_cfl(self):
        limit = 1.0 / np.sqrt(2.0)
        assert self.r < limit, f"CFL violated: r={self.r:.4f} must be < {limit:.4f}"

    def set_material(self, name: str):
        self.alpha = MATERIALS[name]

    def __repr__(self) -> str:
        return (f"RoomGrid {self.W}x{self.H} m | {self.NX}x{self.NY} cells | "
                f"dx={self.dx*100:.2f} cm  dt={self.dt*1e6:.2f} us  r={self.r:.4f}  beta={self.beta}")
