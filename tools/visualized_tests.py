import os
import sys
import torch
from pathlib import Path

import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from pytorch3d.structures import Pointclouds


# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes  # , load_obj

# # Data structures and functions for rendering
# from pytorch3d.structures import Meshes
# from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
# from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib

from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
)
from pytorch3d.transforms import RotateAxisAngle

IMAGE_SIZE = 1024


def interpolate_line(img, start, end, color):
    """Draw a line from `start` to `end` on `img`."""
    y0, x0 = start
    y1, x1 = end
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1
    if dx > dy:
        err = dx / 2.0
        while x != x1:
            img[int(y), int(x)] = color
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            img[int(y), int(x)] = color
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    img[int(y), int(x)] = color  # Make sure the end point is drawn


def draw_bbox(img, corners, color=(255, 0, 0)):
    """Draw an angular bounding box defined by `corners` on `img`."""
    n = len(corners)
    for i in range(n):
        start = corners[i].astype(np.int16)
        end = corners[(i + 1) % n].astype(np.int16)
        interpolate_line(img, start, end, color)


def visualize_verts_over_object(
    R, T, y_pixel_int, x_pixel_int, extreme_y, extreme_x, vis_verts=True
):
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

    # Lighting
    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

    # Rasterization settings
    raster_settings = RasterizationSettings(
        image_size=1024,
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    # Renderer
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(device=device, cameras=cameras, lights=lights),
    )

    # Render the image
    image = renderer(meshes_world=mesh.to(device), R=cameras.R, T=cameras.T)
    image = np.array(image[0][:, :, :3])
    if vis_verts:
        pixels_to_highlight = np.stack((y_pixel_int, x_pixel_int), axis=1)
        square_size = 5  # This will create a 5x5 square
        paint_color = np.array([0, 255, 0])  # Red
        for y, x in pixels_to_highlight:
            # Ensure the square stays within image bounds
            x_start = int(max(0, x - square_size // 2))
            y_start = int(max(0, y - square_size // 2))
            x_end = int(min(image.shape[1], x + square_size // 2 + 1))
            y_end = int(min(image.shape[0], y + square_size // 2 + 1))

            # Paint the square around each pixel
            image[y_start:y_end, x_start:x_end] = paint_color

    square_size = 5  # This will create a 5x5 square
    paint_color = np.array([255, 0, 0])  # Red

    pixels_to_highlight = np.stack((extreme_y, extreme_x), axis=1)
    for y, x in pixels_to_highlight:
        # Ensure the square stays within image bounds
        x_start = int(max(0, x - square_size // 2))
        y_start = int(max(0, y - square_size // 2))
        x_end = int(min(image.shape[1], x + square_size // 2 + 1))
        y_end = int(min(image.shape[0], y + square_size // 2 + 1))

        # Paint the square around each pixel
        image[y_start:y_end, x_start:x_end] = paint_color

    # top-left, top-right, bottom-right, bottom-left
    draw_bbox(
        image,
        [
            pixels_to_highlight[2],
            pixels_to_highlight[1],
            pixels_to_highlight[3],
            pixels_to_highlight[0],
        ],
        color=(0, 0, 128),
    )
    return image


def render_points_only(points, cameras):

    """
    returns black image with the verts given in points render over it.
    to save it as PIL image one should multiple the result by 255 and move it to uint8
    """

    global IMAGE_SIZE
    # Define rasterization settings for the point cloud
    raster_settings = PointsRasterizationSettings(
        image_size=IMAGE_SIZE,
        radius=0.003,  # The radius of each point in NDC units
        points_per_pixel=11,  # Number of points to rasterize per pixel
    )

    # Create the point cloud renderer
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(cameras=cameras, raster_settings=raster_settings),
        compositor=AlphaCompositor(),
    )
    # Render the point cloud
    images = renderer(points)

    return (images[0, ..., :3]).cpu().numpy()


def render_mesh(mesh, angle):

    """
    returns white image with the mesh render over it using the given angle .
    to save it as PIL image one should multiple the result by 255 and move it to uint8
    """
    global IMAGE_SIZE
    R, T = look_at_view_transform(30, elev=90, azim=0, up=((0, 0, 1),), at=((0, 0, 0),))
    rotate_transform = RotateAxisAngle(angle=angle, axis="Y")
    rotation_matrix = rotate_transform.get_matrix()

    R = torch.bmm(rotation_matrix[:, :3, :3], R)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

    raster_settings = RasterizationSettings(
        image_size=IMAGE_SIZE,
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=0,  # Use naive rasterization
        max_faces_per_bin=None,  # Optionally, adjust this value based on your needs
    )

    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(device=device, cameras=cameras, lights=lights),
    )

    images = renderer(mesh)

    return (images[0, ..., :3]).cpu().numpy()


def find_closest(verts, x, y, z):
    """
    find the closest point to [x, y, z] from verts using the l2 simmilarity
    """
    recived_tensor = torch.tensor([x, y, z], device=verts.device)
    return verts[torch.norm(verts - recived_tensor, dim=1).argmin()]


def get_mesh_extreme_points_from_looking_above_view(mesh):

    """
    the current code only supports looking above view! ( R, T = look_at_view_transform(something, elev=90, azim=0, up=((0, 0, 1),), at=((0, 0, 0),)) )
    to extend this support one should decide how to use the full bounded rectangale and not only is top face as we did.
    the code assumes y axis is up.
    the points which are given to find closest are the bounded rectangale of the mesh and can be used also without find closest.

    """

    x_max = mesh._verts_packed[:, 0].max()
    x_min = mesh._verts_packed[:, 0].min()

    y_max = mesh._verts_packed[:, 1].max()
    y_min = mesh._verts_packed[:, 1].min()

    z_max = mesh._verts_packed[:, 2].max()
    z_min = mesh._verts_packed[:, 2].min()

    verts = mesh.verts_packed()

    x_max_y_max_z_max = find_closest(verts, x_max, y_max, z_max)
    x_max_y_min_z_max = find_closest(verts, x_max, y_min, z_max)
    x_min_y_max_z_max = find_closest(verts, x_min, y_max, z_max)
    x_min_y_min_z_max = find_closest(verts, x_min, y_min, z_max)

    x_max_y_max_z_min = find_closest(verts, x_max, y_max, z_min)
    x_max_y_min_z_min = find_closest(verts, x_max, y_min, z_min)
    x_min_y_max_z_min = find_closest(verts, x_min, y_max, z_min)
    x_min_y_min_z_min = find_closest(verts, x_min, y_min, z_min)

    return (
        x_max_y_max_z_min,
        x_min_y_max_z_min,
        x_max_y_max_z_max,
        x_min_y_max_z_max,
    )  # , x_max_y_min_z_max, x_min_y_min_z_max, x_max_y_min_z_min, x_min_y_min_z_min


def _view_definitions():
    AXIS = "Y"
    UP = ((0, 0, 1),)
    AT = ((0, 0, 0),)
    ELEV = 90
    AZIM = 0
    T_Z = 30
    return T_Z, AZIM, ELEV, AT, UP, AXIS


def _define_verts_ndc(verts, R, T, angle):

    """
    returns verts in ndc space
    """

    T_Z, AZIM, ELEV, AT, UP, AXIS = _view_definitions()
    R, T = look_at_view_transform(T_Z, elev=ELEV, azim=AZIM, up=UP, at=AT)

    rotation = RotateAxisAngle(angle=angle, axis=AXIS).get_matrix()
    R = rotation[:, :3, :3]
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    R_T = cameras.get_world_to_view_transform().get_matrix()

    r = RotateAxisAngle(angle=(angle), axis=AXIS).get_matrix()

    R_T[:, :3, :3] = r[:, :3, :3]
    P = cameras.get_projection_transform().get_matrix()

    vertex = torch.concatenate(
        [verts, torch.ones((len(verts), 1), device=verts.device)], dim=1
    )
    vertex_clip = (vertex @ R_T[0]) @ P[0]
    vertex_ndc = vertex_clip[:, :3] / vertex_clip[:, 3:]

    return vertex_ndc, R_T, cameras


def verts_to_ndc(verts, R_T, cameras):

    R1 = R_T[0][:3, 0]
    R2 = R_T[0][:3, 1]
    R3 = R_T[0][:3, 2]
    T_VEC = R_T[0][3, :3]

    T_X = T_VEC[0]
    T_Y = T_VEC[1]
    T_Z = T_VEC[2]

    fov_weight = 1 / torch.tan(((np.pi / 180) * cameras.fov) / 2)

    xs = fov_weight * (verts @ R1 + T_X) / (verts @ R3 + T_Z)
    ys = fov_weight * (verts @ R2 + T_Y) / (verts @ R3 + T_Z)

    return xs, ys


def pixels_to_image_coordinates(xs, ys):

    """
    xs and ys are pixels coordinates at (-1, 1) scale and we move them to (IMAGE_SIZE, IMAGE_SIZE) ints to match the image size
    """

    width, height = IMAGE_SIZE, IMAGE_SIZE  # Example viewport dimensions
    x_pixel = ((1 - xs) / 2.0) * width
    y_pixel = ((1 - ys) / 2.0) * height

    x_pixel_int = np.clip(x_pixel.cpu().numpy().round().astype(int), 0, width - 1)
    y_pixel_int = np.clip(y_pixel.cpu().numpy().round().astype(int), 0, height - 1)
    return x_pixel_int, y_pixel_int


def test_formula(verts):

    """
    test the basic formula
    """

    T_Z, AZIM, ELEV, AT, UP, AXIS = _view_definitions()
    R, T = look_at_view_transform(T_Z, elev=ELEV, azim=AZIM, up=UP, at=AT)

    for angle in [10 * i for i in range(1, 19)]:
        vertex_ndc, R_T, cameras = _define_verts_ndc(verts, R, T, angle)

        xs, ys = verts_to_ndc(verts, R_T, cameras)

        pix = torch.stack((xs, ys), dim=1)
        z = (torch.abs(pix - vertex_ndc[:, :2]) > 1e-5).sum()

        assert (
            z.item() == 0
        ), "formula for calculating verts position over the image is wrong"
    print("test formula is done")


def render_extreme_points(mesh, cameras, axis="Y", up_vec=((0, -1, 0),), angle=20):

    """
    this function maps the relavent points from the 3D bounded rectangle to image space
    and returns black image with the relevant points render over it
    """
    global IMAGE_SIZE
    square_size = 5

    paint = [
        np.array([1.0, 0, 0]),
        np.array([0, 1.0, 0]),
        np.array([0, 0, 1.0]),
        np.array([0.5, 0.5, 0.5]),
    ]

    image = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3)).astype(np.float32)

    (
        x_max_y_max_z_min,
        x_min_y_max_z_min,
        x_max_y_max_z_max,
        x_min_y_max_z_max,
    ) = get_mesh_extreme_points_from_looking_above_view(mesh)

    ver = torch.stack(
        [x_max_y_max_z_min, x_min_y_max_z_min, x_max_y_max_z_max, x_min_y_max_z_max],
        dim=0,
    )

    fov = (cameras.fov * np.pi) / 180  # angles to radins
    fov_weight = 1 / np.tan((fov / 2))

    R, T = look_at_view_transform(T_Z, elev=ELEV, azim=AZIM, up=up_vec, at=AT)
    rotate_transform = RotateAxisAngle(angle=angle, axis=axis)
    rotation_matrix = rotate_transform.get_matrix()

    R = torch.bmm(rotation_matrix[:, :3, :3], R)

    R1 = R[0][:3, 0]
    R2 = R[0][:3, 1]
    R3 = R[0][:3, 2]

    V_R_1 = ver @ R1
    V_R_2 = ver @ R2
    V_R_3 = ver @ R3

    xs = fov_weight * (V_R_1 + T[0, 0]) / (V_R_3 + T[0, 2])
    ys = fov_weight * (V_R_2 + T[0, 1]) / (V_R_3 + T[0, 2])

    x_pixel_int, y_pixel_int = pixels_to_image_coordinates(xs, ys)

    pixels_to_highlight = np.stack((y_pixel_int, x_pixel_int), axis=1)
    for i, (y, x) in enumerate(pixels_to_highlight):
        # Ensure the square stays within image bounds
        x_start = max(0, x - square_size // 2)
        y_start = max(0, y - square_size // 2)
        x_end = min(image.shape[1], x + square_size // 2 + 1)
        y_end = min(image.shape[0], y + square_size // 2 + 1)
        image[y_start:y_end, x_start:x_end] = paint[i]

    return image, pixels_to_highlight


if __name__ == "__main__":

    app_path = Path(__file__).parent.parent
    dir_path = f"{app_path}/mmrotate/3Ddata/meshes"
    os.makedirs(dir_path, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    # Set paths -
    DATA_DIR = f"{app_path}/mmrotate/3Ddata/"
    obj_filename = os.path.join(DATA_DIR, "meshes/Container.obj")

    # Load obj file
    mesh = load_objs_as_meshes([obj_filename], device=device)

    verts = mesh.verts_packed()  # Get the vertices of the mesh
    features = torch.ones((len(verts), 3))
    features[:, 0] = 0
    points = Pointclouds(points=[verts], features=[features])

    T_Z, AZIM, ELEV, AT, UP, AXIS = _view_definitions()

    for angle in [0, 20, 45]:

        folder_path = (
            f"{app_path}/data/test_injected/trainval/images/{angle}_up_{UP}_axis_{AXIS}"
        )
        os.makedirs(folder_path, exist_ok=True)
        print(f"angle is - {angle}", end="\n\n")

        R, T = look_at_view_transform(T_Z, elev=ELEV, azim=AZIM, up=UP, at=AT)
        rotate_transform = RotateAxisAngle(angle=angle, axis=AXIS)
        rotation_matrix = rotate_transform.get_matrix()

        R = torch.bmm(rotation_matrix[:, :3, :3], R)

        cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

        R_T = cameras.get_world_to_view_transform().get_matrix()

        P = cameras.get_projection_transform().get_matrix()

        vertex = torch.concatenate(
            [verts, torch.ones((len(verts), 1), device=verts.device)], dim=1
        )

        points_render_image, pixels_to_highlight = render_extreme_points(
            mesh=mesh, cameras=cameras, axis=AXIS, up_vec=UP, angle=angle
        )

        vertex_clip = (vertex @ R_T[0]) @ P[0]
        vertex_ndc = vertex_clip[:, :3] / vertex_clip[:, 3:]

        x_pixel_int, y_pixel_int = pixels_to_image_coordinates(
            xs=vertex_ndc[:, 0], ys=vertex_ndc[:, 1]
        )

        square_size = 5  # This will create a 5x5 square

        image_for_my_projection = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3)).astype(np.uint8)
        paint_color = np.array([255, 0, 0])  # Red

        for y, x in pixels_to_highlight:
            # Ensure the square stays within image bounds
            x_start = max(0, x - square_size // 2)
            y_start = max(0, y - square_size // 2)
            x_end = min(image_for_my_projection.shape[1], x + square_size // 2 + 1)
            y_end = min(image_for_my_projection.shape[0], y + square_size // 2 + 1)

            # Paint the square around each pixel
            image_for_my_projection[y_start:y_end, x_start:x_end] = paint_color

        Image.fromarray((image_for_my_projection).astype(np.uint8)).save(
            f"{folder_path}/hand_craft_extreme_point.png"
        )

        image_for_my_projection = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3)).astype(np.uint8)
        image_for_my_projection[y_pixel_int, x_pixel_int] = paint_color
        Image.fromarray((image_for_my_projection).astype(np.uint8)).save(
            f"{folder_path}/my_projection.png"
        )

        image_for_my_projection[
            pixels_to_highlight[:, 0], pixels_to_highlight[:, 1]
        ] = np.array(
            [0, 0, 255]
        )  # Blue

        torch_points_render_image = render_points_only(points, cameras)

        im_for_rand = Image.open(
            f"{app_path}/data/test_injected/trainval/images/P0005__1024__0___0.png"
        )
        im_for_rand = np.array(im_for_rand).astype(np.float32) / 255.0

        images = render_mesh(mesh=mesh, angle=angle)
        mask = (images == 1).astype(np.float32)
        end_im = (((1 - mask) * images + mask * im_for_rand) * 255).astype(np.uint8)

        print(f"start saving images, angle - {angle}")
        Image.fromarray((points_render_image * 255).astype(np.uint8)).save(
            f"{folder_path}/experiment_{AXIS}.png"
        )
        Image.fromarray(end_im).save(f"{folder_path}/pres_dota.png")
        Image.fromarray((torch_points_render_image * 255).astype(np.uint8)).save(
            f"{folder_path}/point_cloud_theirs.png"
        )
        Image.fromarray(
            (
                0.5 * (torch_points_render_image * 255)
                + 0.5 * end_im.astype(np.float32)
            ).astype(np.uint8)
        ).save(f"{folder_path}/theirs_over_obj.png")
        Image.fromarray(
            (0.5 * image_for_my_projection + 0.5 * end_im.astype(np.float32)).astype(
                np.uint8
            )
        ).save(f"{folder_path}/my_over-obj.png")
        Image.fromarray(
            (
                0.5 * (torch_points_render_image * 255)
                + 0.5 * image_for_my_projection.astype(np.float32)
            ).astype(np.uint8)
        ).save(f"{folder_path}/my_proj_over_camera_proj.png")

        print(
            f"done {angle}, visualizations can be found at - {folder_path}", end="\n\n"
        )
