import json
from pathlib import Path

import torch
from tqdm import tqdm

from nerfstudio.cameras.rays import Frustums, RayBundle, RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.pipelines.base_pipeline import VanillaPipeline
from nerfstudio.utils.eval_utils import eval_setup

print("creating config path")
config_path = Path("/home/stanlew/src/nerfstudio/outputs/ns-formatted/nerfacto/2023-05-03_114050/config.yml")

print("creating the model in inference mode")
trainer_config, pipeline, ckpt_path = eval_setup(config_path, test_mode="inference")

print("got model:")
print(pipeline.model)

print("reading the transforms file")
with(open("/home/stanlew/Downloads/drea_scan_ipad/2/ns-formatted/transforms.json", "r")) as f:
    tf_file_contents = json.load(f)

# loop through the 'frames' in the tf_file_contents and get the min/max x,y,z values
min_x = max_x = min_y = max_y = min_z = max_z = None
for frame in tf_file_contents["frames"]:
    x = frame["transform_matrix"][0][3]
    y = frame["transform_matrix"][1][3]
    z = frame["transform_matrix"][2][3]
    if min_x is None or x < min_x:
        min_x = x
    if max_x is None or x > max_x:
        max_x = x
    if min_y is None or y < min_y:
        min_y = y
    if max_y is None or y > max_y:
        max_y = y
    if min_z is None or z < min_z:
        min_z = z
    if max_z is None or z > max_z:
        max_z = z

# generate a random grid of size 'grid_size' in the x,y,z range
grid_size = 1000
x_range = max_x - min_x
y_range = max_y - min_y
z_range = max_z - min_z
x_step = x_range / grid_size
y_step = y_range / grid_size
z_step = z_range / grid_size
grid = []
for x in tqdm(range(grid_size)):
    for y in tqdm(range(grid_size), leave=False):
        for z in tqdm(range(grid_size), leave=False):
            grid.append((min_x + x * x_step, min_y + y * y_step, min_z + z * z_step))

# make grid into a torch tensor
grid = torch.tensor(grid)
# make a random vector like grid for the directions
directions = torch.rand(grid.shape)

ray_bundle = Frustums(
        origins=grid,
        directions=directions,
        pixel_area=None,
        starts=0,
        ends=0
    ).to(pipeline.device)
# create camera indices that match the number of grid points
camera_indices = list(range(grid_size ** 3))
camera_indices = torch.tensor(camera_indices).to(pipeline.device)
ray_samples = RaySamples(frustums=ray_bundle, camera_indices=camera_indices)

pipeline.model.eval()
results = pipeline.model.field(ray_samples)
density_results = results[FieldHeadNames.DENSITY]
print(density_results)
print(density_results.min())
print(density_results.max())