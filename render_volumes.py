import numpy as np
from scipy.spatial.transform import Rotation as R

from place_motif.volume import place_volume_on_grid

n_points = 10
n_positions = 20
grid_dimensions = (500, 500, 500)

result = place_volume_on_grid(
    volume=np.arange(32** 3).reshape((32, 32, 32)),
    center_of_rotation=(16, 16, 16),
    positions=np.random.uniform(0, np.min(grid_dimensions),
                                size=(n_positions, 3)),
    orientations=R.random(num=n_positions),
    grid_dimensions=grid_dimensions
)

import napari

viewer = napari.Viewer(ndisplay=3)
viewer.add_image(result)
napari.run()
