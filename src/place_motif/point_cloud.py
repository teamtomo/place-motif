from typing import Optional, Tuple

import numpy as np
import einops
from pydantic import BaseModel
from scipy.spatial.transform import Rotation

from src.place_motif.motif import Motif


class PointCloud(Motif, BaseModel):
    data: np.ndarray
    values: Optional[np.ndarray] = None

    def place_in_space(
            self,
            positions: np.ndarray,
            orientations: Optional[Rotation]
    ) -> Tuple[np.ndarray, np.ndarray]:
        n_positions = len(positions)
        if orientations is None:
            orientations = Rotation.identity(num=n_positions)
        n_orientations = len(orientations)

        if n_positions != n_orientations:
            raise ValueError("must have the same number of positions and "
                             "orientations")
        if self.data.shape[-1] != 3 or self.data.ndim != 2:
            raise ValueError("points must have shape (n, 3)")
        # orient the points
        # (n, 3, 3) -> (1, n, 3, 3)
        orientations = einops.rearrange(orientations.as_matrix(),
                                        'n i j -> 1 n i j')

        # (m, 3) -> (m, 1, 3, 1)
        points = einops.rearrange(self.data, 'm i -> m 1 i 1')

        # (m, n, 3, 1) -> (m, n, 3)
        oriented_points = einops.rearrange(orientations @ points,
                                           'm n i 1 -> m n i')

        final_points = oriented_points + positions  # (m, n, 3) + (n, 3) -> (m, n, 3)

    def place_on_grid(
            self,
            positions: np.ndarray,
            orientations: Rotation,
            grid_dimensions: Tuple[int, int, int]
    ) -> np.ndarray:


def place_point_cloud(
        points: np.ndarray,
        positions: np.ndarray,
        orientations: Optional[Rotation] = None
) -> np.ndarray:
    """Place a point cloud at positions, with orientations.

    This function assumes that points (p) are oriented by orientations (R)
    by R @ p and makes no modifications to axis ordering in R or p.

    Parameters
    ----------
    points : (m, 3) np.ndarray
        Point cloud to be placed. Points will be rotated around the origin
        (0, 0, 0).
    positions : (n, 3) np.ndarray
        Positions at which to place point clouds.
    orientations: scipy.spatial.transform.Rotation | None
        Orientations (same length as positions) of point clouds.

    Returns
    -------
    placed_points : (m, n, 3) np.ndarray
    """
    n_positions = len(positions)
    if orientations is None:
        orientations = Rotation.identity(num=n_positions)
    n_orientations = len(orientations)

    if n_positions != n_orientations:
        raise ValueError("must have the same number of positions and "
                         "orientations")
    if points.shape[-1] != 3 or points.ndim != 2:
        raise ValueError("points must have shape (n, 3)")

    # orient the points
    # (n, 3, 3) -> (1, n, 3, 3)
    orientations = einops.rearrange(orientations.as_matrix(),  'n i j -> 1 n i j')

    # (m, 3) -> (m, 1, 3, 1)
    points = einops.rearrange(points, 'm i -> m 1 i 1')

    # (m, n, 3, 1) -> (m, n, 3)
    oriented_points = einops.rearrange(orientations @ points, 'm n i 1 -> m n i')

    return oriented_points + positions  # (m, n, 3) + (n, 3) -> (m, n, 3)


def flatten_point_clouds(points: np.ndarray, values: Optional[np.ndarray] = None):
    flattened_points = einops.rearrange(points, 'm n c -> (m n) c')
    if values is not None:
        values = einops.repeat(values, 'm -> (m n)', n=points.shape[1])
    return flattened_points, values


def render_point_cloud_on_grid(
        points: np.ndarray,
        grid_dimensions: Tuple[int, ...],
        values: Optional[np.ndarray] = None
) -> np.ndarray:
    points, values = flatten_point_clouds(points, values)
    volume, _ = np.histogramdd(sample=points, bins=grid_dimensions, weights=values)
    return volume

