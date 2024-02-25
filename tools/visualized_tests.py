# Copyright (c) OpenMMLab. All rights reserved.
import os
import sys
import torch
import pytorch3d

import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import requests
from pytorch3d.structures import Pointclouds
# URL of the Python script
url = 'https://raw.githubusercontent.com/facebookresearch/pytorch3d/main/docs/tutorials/utils/plot_image_grid.py'
# Local path to save the script
filepath = '/app/tools/plot_image_grid.py'

response = requests.get(url)
if response.status_code == 200:
    with open(filepath, 'w') as f:
        f.write(response.text)
else:
    print("Failed to download the script")

from plot_image_grid import image_grid

# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, load_obj

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,

)
import torch
from pytorch3d.transforms import RotateAxisAngle

import sys
import os
sys.path.append(os.path.abspath(''))

dir_path = '/app/mmrotate/3Ddata/meshes'
os.makedirs(dir_path, exist_ok=True)


if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

# Set paths
DATA_DIR = "/app/mmrotate/3Ddata/"
obj_filename = os.path.join(DATA_DIR, "meshes/Container.obj")

# Load obj file
mesh = load_objs_as_meshes([obj_filename], device=device)
# mesh._verts_list[0] contains list of (x, y, z) verts
# tensor with shape [num of verts,]

def render_points_only(points, cameras):
    from pytorch3d.renderer import (
        PointsRasterizationSettings,
        PointsRenderer,
        PointsRasterizer,
        AlphaCompositor,
    )

    # Define rasterization settings for the point cloud
    raster_settings = PointsRasterizationSettings(
        image_size=1024, 
        radius = 0.003,  # The radius of each point in NDC units
        points_per_pixel = 11,  # Number of points to rasterize per pixel
    )

    # Create the point cloud renderer
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(cameras=cameras, raster_settings=raster_settings),
        compositor=AlphaCompositor()
    )
    # Render the point cloud
    images = renderer(points)

    # Assuming you're using IPython or a Jupyter notebook to visualize the output
    # from matplotlib import pyplot as plt
    # plt.figure(figsize=(10, 10))
    # plt.imshow(images[0, ..., :3].cpu().numpy())  # Assuming RGB channels
    # plt.axis("off")
    return images[0, ..., :3].cpu().numpy()

verts = mesh.verts_packed()  # Get the vertices of the mesh
features = torch.ones((len(verts), 3))
features[:, 0] = 0
points = Pointclouds(points=[verts], features=[features])

# Initialize a camera.
# With world coordinates +Y up, +X left and +Z in, the front of the cow is facing the -Z direction. 
# So we move the camera by 180 in the azimuth direction so it is facing the front of the cow. 
R, T = look_at_view_transform(70, elev=0, azim=180) 

for angle in [10, 15, 20, 30, 65, 40, 70, 127, 285, 339, 360]:

    folder_path = f"/app/data/test_injected/trainval/images/{angle}"
    os.makedirs(folder_path, exist_ok=True) 
    print(f"angle is - {angle}")
    print()

    rotation = RotateAxisAngle(angle=angle, axis="Z").get_matrix()
    R = rotation[:, :3, :3]
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)


    R_T = cameras.get_world_to_view_transform().get_matrix()

    # r = RotateAxisAngle(angle= (-angle), axis="Z").get_matrix()
    r = RotateAxisAngle(angle= (angle), axis="Z").get_matrix()
    R_T [:, :3, :3] = r[:, :3, :3]
    # R_T[:, 3, 0:2] = -0
    # Get the projection matrix (P)
    P = cameras.get_projection_transform().get_matrix()

    vertex = torch.concatenate([verts, torch.ones((len(verts), 1), device = verts.device)], dim=1)



    vertex_clip = (vertex @ R_T[0]) @ P[0]

    # Step 3: Perform perspective division to get NDC
    vertex_ndc = vertex_clip[:, :3] / vertex_clip[:, 3:] 

    R1 = R_T[0][:3, 0]
    R2 = R_T[0][:3, 1]
    R3 = R_T[0][:3, 2]
    T_VEC = R_T[0][3, :3]

    T_X = T_VEC[0]
    T_Y = T_VEC[1]
    T_Z = T_VEC[2]

    fov_weight = (1 / torch.tan(((np.pi / 180) * cameras.fov) / 2)) 

    xs = fov_weight * (verts @ R1 + T_X) / (verts @ R3 + T_Z)
    ys = fov_weight * (verts @ R2 + T_Y) / (verts @ R3 + T_Z)

    pix = torch.stack((xs, ys), dim=1)
    
    z = (torch.abs(pix - vertex_ndc[:, :2] ) > 1e-5).sum()

    # Step 4: Map NDC to screen coordinates, ndc range (-1, 1)
    width, height = 1024, 1024  # Example viewport dimensions
    # x_pixel = ((vertex_ndc[:, 0] + 1) / 2.0) * width
    x_pixel = ((1 - vertex_ndc[:, 0]) / 2.0) * width
    y_pixel = ((1 - vertex_ndc[:, 1]) / 2.0) * height 

    width, height = 1024, 1024  # Example dimensions
    x_pixel_int = np.clip(x_pixel.cpu().numpy().round().astype(int), 0, width - 1)
    y_pixel_int = np.clip(y_pixel.cpu().numpy().round().astype(int), 0, height - 1)


    x_pixel_int_min = x_pixel_int.argmin()
    x_pixel_int_max = x_pixel_int.argmax()

    y_pixel_int_min = y_pixel_int.argmin()
    y_pixel_int_max = y_pixel_int.argmax()


    z1 = x_pixel_int == x_pixel_int[x_pixel_int_min]
    y_for_x_min = y_pixel_int[z1].max()


    z2 = x_pixel_int == x_pixel_int[x_pixel_int_max]
    y_for_x_max = y_pixel_int[z2].min()


    z3 = y_pixel_int == y_pixel_int[y_pixel_int_min]
    x_for_y_min = x_pixel_int[z3].max()


    z4 = y_pixel_int == y_pixel_int[y_pixel_int_max]
    x_for_y_max = x_pixel_int[z4].min()


    pix_x = np.array([x_pixel_int[x_pixel_int_min], x_pixel_int[x_pixel_int_max], x_for_y_min, x_for_y_max])
    pix_y = np.array([y_for_x_min, y_for_x_max, y_pixel_int[y_pixel_int_min], y_pixel_int[y_pixel_int_max]])


    print(pix_y)
    print(pix_x)

    square_size = 5  # This will create a 5x5 square

    image = np.zeros((1024, 1024, 3)).astype(np.uint8)
    paint_color = np.array([255, 0, 0])  # Red


    pixels_to_highlight = np.stack((pix_y, pix_x), axis=1)
    for y, x in pixels_to_highlight:
        # Ensure the square stays within image bounds
        x_start = max(0, x - square_size // 2)
        y_start = max(0, y - square_size // 2)
        x_end = min(image.shape[1], x + square_size // 2 + 1)
        y_end = min(image.shape[0], y + square_size // 2 + 1)
        
        # Paint the square around each pixel
        image[y_start:y_end, x_start:x_end] = paint_color

    # Define the paint color

    # Paint the pixels using NumPy advanced indexing
    # image[pix_y, pix_x] = paint_color
    # image[pix_y, pix_x] = paint_color
    Image.fromarray((image).astype(np.uint8)).save(f"{folder_path}/hand_craft_extreme_point.png")

    image = np.zeros((1024, 1024, 3)).astype(np.uint8)
    # Paint the pixels using NumPy advanced indexing
    image[y_pixel_int, x_pixel_int] = paint_color
    # image[pix_y, pix_x] = paint_color
    Image.fromarray((image).astype(np.uint8)).save(f"{folder_path}/my_projection.png")




    ind = torch.concatenate([y_pixel, x_pixel])
    xxx = render_points_only(points, cameras)

    Image.fromarray((.5 * (xxx * 255) + .5 * image.astype(np.float32)).astype(np.uint8)).save(f"{folder_path}/my_proj_over_camera_proj.png")
    print("done")

    raster_settings = RasterizationSettings(
        image_size=1024, 
        blur_radius=0.0, 
        faces_per_pixel=1, 
        bin_size=0,  # Use naive rasterization
        max_faces_per_bin=None  # Optionally, adjust this value based on your needs
    )

    # Place a point light in front of the object. As mentioned above, the front of the cow is facing the 
    # -z direction. 
    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

    # Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will 
    # interpolate the texture uv coordinates for each vertex, sample from a texture image and 
    # apply the Phong lighting model
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device, 
            cameras=cameras,
            lights=lights
        )
    )

    im_for_rand = Image.open("/app/data/test_injected/trainval/images/P0005__1024__0___0.png")
    im_for_rand = np.array(im_for_rand).astype(np.float32) / 255.




    images = renderer(mesh)
    # images = renderer(points)
    mask = (images[0, ..., :3].cpu().numpy() == 1).astype(np.float32)
    plt.figure(figsize=(10, 10))
    # plt.imshow((1 - mask) * images[0, ..., :3].cpu().numpy() + mask * im_for_rand)
    plt.axis("off")

    plt.savefig("/app/mmrotate/3Ddata/rendering_images/first_try.png")


    end_im = (((1 - mask) * images[0, ..., :3].cpu().numpy() + mask * im_for_rand) * 255).astype(np.uint8)
    Image.fromarray(end_im).save(f"{folder_path}/pres_dota.png")
    Image.fromarray((xxx * 255).astype(np.uint8)).save(f"{folder_path}/point_cloud_there.png")



    Image.fromarray((.5 * (xxx * 255) + .5 * end_im.astype(np.float32)).astype(np.uint8)).save(f"{folder_path}/theres_over_obj.png")


    Image.fromarray((.5 * image + .5 * end_im.astype(np.float32)).astype(np.uint8)).save(f"{folder_path}/my_over-obj.png")


    print(f"done {angle}")
    print()
    print()


