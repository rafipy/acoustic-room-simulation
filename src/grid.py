import numpy as np
from config import C_SOUND, FREQ, ROOM_W, ROOM_H, PPW, CFL_FACTOR, MATERIALS


class RoomGrid:

    def __init__(
        self,
        width:  float = ROOM_W,
        height: float = ROOM_H,
        ppw:    int   = PPW,
        alpha:  float = MATERIALS["Concrete"],
    ):
        self.W = width
        self.H = height

        # Spatial resolution derived from wavelength
        lam     = C_SOUND / FREQ
        self.dx = lam / ppw
        self.dy = self.dx          # square cells keep the stencil isotropic

        # CFL-stable time step
        self.dt = CFL_FACTOR * self.dx / C_SOUND
        self._r = C_SOUND * self.dt / self.dx   # Courant number (for reference)

        # Grid dimensions
        self.NX = int(np.ceil(self.W / self.dx))
        self.NY = int(np.ceil(self.H / self.dy))

        self.alpha = alpha

        self._validate_cfl()

    def _validate_cfl(self):
        cfl_limit = 1.0 / np.sqrt(2.0)
        assert self._r < cfl_limit, (
            f"CFL violated: r={self._r:.4f} must be < {cfl_limit:.4f}"
        )

    def set_material(self, name: str):
        """Switch wall material by preset name."""
        self.alpha = MATERIALS[name]

    def __repr__(self) -> str:
        return (
            f"RoomGrid  {self.W} m × {self.H} m  |  "
            f"{self.NX} × {self.NY} cells  |  "
            f"dx={self.dx*100:.2f} cm  dt={self.dt*1e6:.2f} µs  "
            f"r={self._r:.4f}  α={self.alpha}"
        )
