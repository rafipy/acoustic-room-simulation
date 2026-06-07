import numpy as np
from config import C_SOUND, FREQ, ROOM_W, ROOM_H, PPW, CFL_FACTOR, BETA, MATERIALS


class RoomGrid:
    """
    Geometry + FDTD parameters for the room.

    Fields built on this grid (pressure, the α material map) are (NY, NX) arrays:
    row index j ↔ y, column index i ↔ x — the image-row convention shared by the
    solver, the room builder, and (later) the visualiser.  See utils.empty_field.
    """

    def __init__(
        self,
        width:  float = ROOM_W,
        height: float = ROOM_H,
        ppw:    int   = PPW,
        alpha:  float = MATERIALS["Concrete"],
        beta:   float = BETA,
    ):
        self.W = width
        self.H = height

        # Spatial resolution derived from the wavelength of the resolution frequency
        lam     = C_SOUND / FREQ
        self.dx = lam / ppw
        self.dy = self.dx          # square cells keep the 5-point stencil isotropic

        # CFL-stable time step.  CFL_FACTOR is used directly as the Courant number r.
        self.dt = CFL_FACTOR * self.dx / C_SOUND
        self.r  = C_SOUND * self.dt / self.dx   # Courant number (consumed by the solver)

        # Grid dimensions
        self.NX = int(np.ceil(self.W / self.dx))
        self.NY = int(np.ceil(self.H / self.dy))

        self.alpha = alpha    # default outer-wall material α (used by Room.add_border)
        self.beta  = beta     # air damping coefficient β (default 0 → lossless air)

        self._validate_cfl()

    def _validate_cfl(self):
        cfl_limit = 1.0 / np.sqrt(2.0)
        assert self.r < cfl_limit, (
            f"CFL violated: r={self.r:.4f} must be < {cfl_limit:.4f}"
        )

    def set_material(self, name: str):
        """Switch the default wall material by preset name."""
        self.alpha = MATERIALS[name]

    def __repr__(self) -> str:
        return (
            f"RoomGrid  {self.W} m x {self.H} m  |  "
            f"{self.NX} x {self.NY} cells  |  "
            f"dx={self.dx*100:.2f} cm  dt={self.dt*1e6:.2f} us  "
            f"r={self.r:.4f}  alpha={self.alpha}  beta={self.beta}"
        )
