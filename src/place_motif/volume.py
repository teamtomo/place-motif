from typing import Tuple, Sequence

import einops
import numpy as np
from scipy.spatial.transform import Rotation
from pydantic import BaseModel

from .motif import Motif
from .point_cloud import place_point_cloud, render_point_cloud_on_grid


class Volume(Motif, BaseModel):
    data: np.ndarray
    center_of_rotation: Tuple[int, int, int]

    def place_in_space(
            self,
            positions: np.ndarray,
            orientations: Rotation
    ) -> Tuple[np.ndarray, np.ndarray]:
        points, values = volume_as_point_cloud(self.data)
        points -= self.center_of_rotation


def volume_as_point_cloud(volume: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    indices = np.indices(dimensions=volume.shape)  # (3, n, n, n)
    points = einops.rearrange(indices, 'c i j k -> (i j k) c')  # (n^3, c)
    values = einops.rearrange(volume, 'i j k -> (i j k)')  # (n^3)
    return points, values


def place_volume(
        volume: np.ndarray,
        center_of_rotation: Tuple[float, float, float],
        positions: np.ndarray,
        orientations: Rotation
) -> Tuple[np.ndarray, np.ndarray]:
    points, values = volume_as_point_cloud(volume)
    points -= center_of_rotation
    placed_point_clouds = place_point_cloud(
        points=points,
        positions=positions,
        orientations=orientations,
    )
    return placed_point_clouds, values


def place_volume_on_grid(
        volume: np.ndarray,
        center_of_rotation: Tuple[float, float, float],
        positions: np.ndarray,
        orientations: Rotation,
        grid_dimensions: Tuple[int, int, int]
) -> np.ndarray:
    points, values = place_volume(
        volume=volume,
        center_of_rotation=center_of_rotation,
        positions=positions,
        orientations=orientations,
    )
    placed_volumes = render_point_cloud_on_grid(
        points=points,
        values=values,
        grid_dimensions=grid_dimensions
    )
    return placed_volumes
