from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
from numpy import ndarray
from scipy.spatial.transform import Rotation


class Motif(ABC):
    @abstractmethod
    def place_in_space(
            self,
            positions: np.ndarray,
            orientations: Rotation
    ) -> Tuple[np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def place_on_grid(
            self,
            positions: np.ndarray,
            orientations: Rotation,
            grid_dimensions: Tuple[int, int, int]
    ) -> np.ndarray:
        pass
