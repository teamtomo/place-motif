import numpy as np
from scipy.spatial.transform import Rotation as R

from place_object.point_cloud import render_point_cloud_on_grid, \
    place_point_cloud

n_points = 10
n_positions = 20
grid_dimensions = (500, 500, 500)
point_clouds = place_point_cloud(
    points=np.random.normal(size=(n_points, 3), loc=0, scale=5),
    positions=np.random.uniform(0, np.min(grid_dimensions), size=(n_positions, 3)),
    orientations=R.random(num=n_positions)
)
result_3d = render_point_cloud_on_grid(
    points=point_clouds, grid_dimensions=grid_dimensions
)
result_2d = render_point_cloud_on_grid(
    points=point_clouds[:, :, -2:], grid_dimensions=(500, 500)
)
import napari



viewer = napari.Viewer(ndisplay=3)
viewer.add_image(result_3d)
viewer.add_image(result_2d)
napari.run()