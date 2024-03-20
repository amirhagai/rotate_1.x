# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from PIL import Image


# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    MeshRasterizer,
    MeshRenderer,
    PointLights,
    RasterizationSettings,
    SoftPhongShader,
    look_at_view_transform,
)
from pytorch3d.transforms import RotateAxisAngle
import torchvision

IMAGE_SIZE = 1024


def draw_pixels(image, y_pixel_int, x_pixel_int, square_size, paint_color):
    pixels_to_highlight = np.stack((y_pixel_int, x_pixel_int), axis=1)
    for i, (y, x) in enumerate(pixels_to_highlight):
        # Ensure the square stays within image bounds
        x_start = max(0, x - square_size // 2)
        y_start = max(0, y - square_size // 2)
        x_end = min(image.shape[1], x + square_size // 2 + 1)
        y_end = min(image.shape[0], y + square_size // 2 + 1)
        image[y_start:y_end, x_start:x_end] = paint_color


def find_angle_from_bbox(
    top_left, bottom_left, top_right, bottom_right, degrees=False
):

    y1, x1 = top_left
    y2, x2 = top_right
    dx, dy = x2 - x1, y2 - y1
    angle_radians = np.arctan2(dy, dx)

    if not degrees:
        return angle_radians

    angle_degrees = np.degrees(angle_radians)

    if dx == 0 and dy > 0:
        angle_degrees = np.array([-90])
    elif dx == 0 and dy < 0:
        angle_degrees = np.array([90])

    return -angle_degrees


class InjectedObject:
    def __init__(self, obj_file_path, TZ_start=70, T=None) -> None:

        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device('cpu')

        self.mesh = load_objs_as_meshes([obj_file_path], device=self.device)
        self.verts = self.mesh.verts_packed()  # Get the vertices of the mesh
        self.TZ_start = TZ_start

        self.elev = 90
        self.azim = 0
        self.up = ((0, 0, 1),)
        self.at = ((0, 0, 0),)
        self.rotation_axis = 'Y'

        self.base_R, self.base_T = look_at_view_transform(
            self.TZ_start, elev=self.elev, azim=self.azim, up=self.up, at=self.at
        )

        if T is not None:
            self.base_T = T

        self.base_R, self.base_R = self.base_R.to(self.device), self.base_R.to(
            self.device
        )

        self.camera = FoVPerspectiveCameras(
            device=self.device, R=self.base_R, T=self.base_T, zfar=500
        )

        self.f2 = (self.camera.zfar * self.camera.znear) / (
            self.camera.zfar - self.camera.znear
        )

        fov = (self.camera.fov * np.pi) / 180  # angles to radins
        self.fov_weight = 1 / torch.tan((fov / 2))
        self.natural_aspect_ratio = self.get_natural_aspect_ratio()

    def get_natural_aspect_ratio(self):
        (
            x_max_y_max_z_min,  # bl
            x_min_y_max_z_min,  # br
            x_max_y_max_z_max,  # tl
            x_min_y_max_z_max,  # tr
            x_max_y_min_z_max,
            x_min_y_min_z_max,
            x_max_y_min_z_min,
            x_min_y_min_z_min,
        ) = self.get_mesh_extreme_points_from_looking_above_view()

        dx = (x_max_y_max_z_min - x_min_y_max_z_min)[0]
        dy = (x_max_y_max_z_max - x_max_y_max_z_min)[2]

        return dy / dx

    def find_closest(self, x, y, z):
        verts = self.verts
        recived_tensor = torch.tensor([x, y, z], device=verts.device)
        return verts[torch.norm(verts - recived_tensor, dim=1).argmin()]

    def get_mesh_extreme_points_from_looking_above_view(self):
        """
        the current code only supports looking above view!
        ( R, T = look_at_view_transform(something,
        elev=90, azim=0, up=((0, 0, 1),), at=((0, 0, 0),)) )

        to extend this support one should decide how to use the full
        bounded rectangale and not only is top face as we did.
        the code assumes y axis is up.
        the points which are given to find closest are the bounded
        rectangale of the mesh and can be used also without find closest.

        """

        verts = self.mesh.verts_packed()
        x_max = verts[:, 0].max()
        x_min = verts[:, 0].min()

        y_max = verts[:, 1].max()
        y_min = verts[:, 1].min()

        z_max = verts[:, 2].max()
        z_min = verts[:, 2].min()

        # verts = self.mesh.verts_packed()

        x_max_y_max_z_min = torch.stack((x_max, y_max, z_min))  # bl
        # self.find_closest(x_max, y_max, z_min)
        x_min_y_max_z_min = torch.stack((x_min, y_max, z_min))  # br
        # self.find_closest(x_min, y_max, z_min)
        x_max_y_max_z_max = torch.stack((x_max, y_max, z_max))  # tl
        # self.find_closest(x_max, y_max, z_max)
        x_min_y_max_z_max = torch.stack((x_min, y_max, z_max))  # tr
        # self.find_closest(x_min, y_max, z_max)

        x_max_y_min_z_min = torch.stack((x_max, y_min, z_min))
        # self.find_closest(x_max, y_min, z_min)
        x_min_y_min_z_min = torch.stack((x_min, y_min, z_min))
        # self.find_closest(x_min, y_min, z_min)
        x_min_y_min_z_max = torch.stack((x_min, y_min, z_max))
        # self.find_closest(x_min, y_min, z_max)
        x_max_y_min_z_max = torch.stack((x_max, y_min, z_max))
        # self.find_closest(x_max, y_min, z_max)

        return (
            x_max_y_max_z_min,  # bl
            x_min_y_max_z_min,  # br
            x_max_y_max_z_max,  # tl
            x_min_y_max_z_max,  # tr
            x_max_y_min_z_max,
            x_min_y_min_z_max,
            x_max_y_min_z_min,
            x_min_y_min_z_min,
        )

    def render_mesh(self, T_z, angle, T=None, R=None, aspect_ratio=1):
        global IMAGE_SIZE
        R_, T_ = look_at_view_transform(
            T_z, elev=self.elev, azim=self.azim, up=self.up, at=self.at
        )
        rotate_transform = RotateAxisAngle(angle=angle, axis=self.rotation_axis)
        rotation_matrix = rotate_transform.get_matrix()
        R_ = torch.bmm(rotation_matrix[:, :3, :3], R_)

        if T is None:
            T = T_

        if R is None:
            R = R_
        cameras = FoVPerspectiveCameras(
            device=self.device, R=R, T=T, aspect_ratio=aspect_ratio, zfar=500
        )

        raster_settings = RasterizationSettings(
            image_size=IMAGE_SIZE,
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=0,
            max_faces_per_bin=None,
        )

        lights = PointLights(device=self.device, location=[[0.0, 0.0, -3.0]])

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras, raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                device=self.device, cameras=cameras, lights=lights
            ),
        )

        images = renderer(self.mesh)

        # plt.imshow(images[:, :, :, :3][0])
        return images[:, :, :, :3][0]

    def _get_rotated_verts(self, verts, R):
        R1 = R[0][:3, 0]
        R2 = R[0][:3, 1]
        R3 = R[0][:3, 2]

        V_R_1 = verts @ R1
        V_R_2 = verts @ R2
        V_R_3 = verts @ R3

        return V_R_1, V_R_2, V_R_3

    def get_extreme_pixels(
        self, angle, image_shape, width=IMAGE_SIZE, height=IMAGE_SIZE
    ):
        """vertex to pixel equation ->

        xs = fov_weight * (V @ R1 + T_X) / (verts @ R3 + T_Z)
        xy = fov_weight * (V @ R2 + T_Y) / (verts @ R3 + T_Z)

        we now doing it with base T as we only want to get the extreme pixels
        """

        (
            x_max_y_max_z_min,  # bl
            x_min_y_max_z_min,  # br
            x_max_y_max_z_max,  # tl
            x_min_y_max_z_max,  # tr
            x_max_y_min_z_max,
            x_min_y_min_z_max,
            x_max_y_min_z_min,
            x_min_y_min_z_min,
        ) = self.get_mesh_extreme_points_from_looking_above_view()

        verts = torch.stack(
            [
                x_max_y_max_z_min,  # bl
                x_min_y_max_z_min,  # br
                x_max_y_max_z_max,  # tl
                x_min_y_max_z_max,  # tr
                x_max_y_min_z_max,
                x_min_y_min_z_max,
                x_max_y_min_z_min,
                x_min_y_min_z_min,
            ],
            axis=0,
        )

        R = self.get_R(angle)
        R = R.to(self.device)

        V_R_1, V_R_2, V_R_3 = self._get_rotated_verts(verts, R)

        # here we do have T
        xs = (
            self.fov_weight
            * (V_R_1 + self.base_T[0, 0])
            / ((V_R_3 + self.base_T[0, 2]) * self.f2)
        )
        ys = (
            self.fov_weight
            * (V_R_2 + self.base_T[0, 1])
            / ((V_R_3 + self.base_T[0, 2]) * self.f2)
        )

        x_pixel = ((1 - xs) / 2.0) * width
        y_pixel = ((1 - ys) / 2.0) * height

        x_pixel_int = np.clip(x_pixel.cpu().numpy(), 0, width - 1)
        y_pixel_int = np.clip(y_pixel.cpu().numpy(), 0, height - 1)

        pix_x = x_pixel_int
        pix_y = y_pixel_int

        return (
            torch.stack([torch.tensor(pix_y[:4]), torch.tensor(pix_x[:4])], axis=1),
            verts[:4],
        )

    @staticmethod
    def calculate_distance(p1, p2):
        """Calculate the Euclidean distance between two points."""
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    @staticmethod
    def calculate_centered_square_bbox(corners, new_center=[512, 512]):
        # Assuming corners are [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
        # Calculate diagonal lengths and then find the average
        # to approximate the bbox size
        # Calculate the lengths of the sides of the rectangle
        height = InjectedObject.calculate_distance(corners[0], corners[2])
        width = InjectedObject.calculate_distance(corners[0], corners[1])

        # Calculate the corners of the new rectangle bbox
        # centered at the new_center
        half_width = width / 2
        half_height = height / 2
        rectangle_corners = [
            [new_center[0] + half_height, new_center[1] - half_width],  # bl
            [new_center[0] + half_height, new_center[1] + half_width],  # br
            [new_center[0] - half_height, new_center[1] - half_width],  # tl
            [new_center[0] - half_height, new_center[1] + half_width],  # tr
        ]

        return torch.tensor(rectangle_corners)

    def get_R(self, angle, rotation=None):
        if angle != 0:
            rotation = RotateAxisAngle(
                angle=angle, axis=self.rotation_axis
            ).get_matrix()
            R_final = torch.bmm(rotation[:, :3, :3].to(self.device), self.base_R)
        else:
            R_final = self.base_R
        R = R_final[:, :3, :3]
        return R

    def linear_alignment_of_bboxes(
        self,
        top_left_real,
        top_right_real,
        bottom_left_real,
        bottom_right_real,
        top_left_injected,
        top_right_injected,
        bottom_left_injected,
        bottom_right_injected,
    ):
        """
        return best a for ||a * injected_bbox_w_h - real_bbox_w_h||^2
        we don't want to find the best alignment as
        real_bbox not centarlized and injected_bbox is

        """

        # dx_real = top_left_real[1] - top_right_real[1]
        # dy_real = bottom_left_real[0] - top_left_real[0]

        # dx_injected = top_left_injected[1] - top_right_injected[1]
        # dy_injected = bottom_left_injected[0] - top_left_injected[0]

        # x = torch.tensor([
        #     dx_real,
        #     dy_real,
        #     top_left_real[0],
        #     top_right_real[0],
        #     bottom_left_real[0],
        #     bottom_right_real[0],
        #     top_left_real[1],
        #     top_right_real[1],
        #     bottom_left_real[1],
        #     bottom_right_real[1],
        # ]).reshape(-1)

        # y = torch.tensor([
        #     dx_injected,
        #     dy_injected,
        #     top_left_injected[0],
        #     top_right_injected[0],
        #     bottom_left_injected[0],
        #     bottom_right_injected[0],
        #     top_left_injected[1],
        #     top_right_injected[1],
        #     bottom_left_injected[1],
        #     bottom_right_injected[1],
        # ]).reshape(-1)

        x = torch.tensor(
            [
                bottom_left_injected[0],
                bottom_left_injected[1],
                bottom_right_injected[0],
                bottom_right_injected[1],
                top_left_injected[0],
                top_left_injected[1],
                top_right_injected[0],
                top_right_injected[1],
            ]
        )

        y = torch.tensor(
            [
                bottom_left_real[0],
                bottom_left_real[1],
                bottom_right_real[0],
                bottom_right_real[1],
                top_left_real[0],
                top_left_real[1],
                top_right_real[0],
                top_right_real[1],
            ]
        )
        # return (x @ y) / (x @ x)
        return y.to(torch.float32) / x.to(torch.float32)

    def _center_pixels(self, pixels_values, image_shape):
        """we get pixels values at image space and we want to take them to ndc
        space where our equation of verts to pixels is living.

        we know that
        x_pixel = ((vertex_ndc[:, 0] + 1) / 2.0) * width
        y_pixel = ((1 - vertex_ndc[:, 1]) / 2.0) * height

        which gives us

        vertex_ndc[:, 0] = ((x_pixel * 2.) / width ) - 1
        vertex_ndc[:, 1] = 1 - ((y_pixel * 2.) / height)
        """

        width, height = image_shape[1:]

        xs = 1 - ((pixels_values[:, 1].to(torch.float64) * 2.0) / width)
        # xs = 1 - ((pixels_values[:, 1].to(torch.float64)   * 2.) / width)
        ys = 1 - ((pixels_values[:, 0].to(torch.float64) * 2.0) / height)
        # ys = ((pixels_values[:, 0].to(torch.float64) * 2.) / height ) - 1
        return xs.to(self.device), ys.to(self.device)

    def find_T_Z(
        self, pixels_values, verts, R, image_shape, angle=0, aspect_ratio=1.0
    ):
        """
        this function assume T_X = T_Y = 0,
        vertex to pixel equation ->
        xs = fov_weight * (V @ R1 + T_X) / f2 *(verts @ R3 + T_Z)
        xy = fov_weight * (V @ R2 + T_Y) / f2 * (verts @ R3 + T_Z)
        for the time being T_X = 0, T_Y = 0 T_Z is our variable
        so we want to solve

        xs = fov_weight * (V @ R1 + T_X) / f2 *(verts @ R3 + T_Z)
        xy = fov_weight * (V @ R2 + T_Y) / f2 * (verts @ R3 + T_Z)

        where T_Z is our variable

        so the solution is

        T_Z = (fov_weight * (V @ R1 )) / f2 * xs - (verts @ R3)

        xs, ys assume to be centerlized and in the range of (-1, 1)

        """

        R = self.get_R(angle, R)

        V_R_1, V_R_2, V_R_3 = self._get_rotated_verts(verts, R)

        xs, ys = self._center_pixels(pixels_values, image_shape)

        T_Z_X = (
            (self.fov_weight * aspect_ratio * V_R_1 / (self.f2 * xs)) - V_R_3
        ).mean()
        T_Z_Y = ((self.fov_weight * V_R_2 / (self.f2 * ys)) - V_R_3).mean()

        return (T_Z_X + T_Z_Y) / 2.0

    def find_T_X_T_Y(
        self, pixels_values, verts, R, image_shape, T_Z, aspect_ratio=1.0
    ):
        """vertex to pixel equation ->

        xs = fov_weight * (V @ R1 + T_X) / f2 * (verts @ R3 + T_Z)
        xy = fov_weight * (V @ R2 + T_Y) / f2 * (verts @ R3 + T_Z)

        so

        T_X = ((f2 * xs * (verts @ R3 + T_Z)) /  fov_weight) - V @ R1
        T_Y = ((f2 * ys * (verts @ R3 + T_Z)) /  fov_weight) - V @ R2
        """

        # TODO - FIX CODE DUPLICATION

        V_R_1, V_R_2, V_R_3 = self._get_rotated_verts(verts, R.to(verts.device))

        xs, ys = self._center_pixels(pixels_values, image_shape)

        T_X = (
            ((self.f2 * xs * (V_R_3 + T_Z)) / (self.fov_weight * aspect_ratio))
            - V_R_1
        ).mean()
        T_Y = (((self.f2 * ys * (V_R_3 + T_Z)) / self.fov_weight) - V_R_2).mean()

        return T_X, T_Y

    @staticmethod
    def rotate_pixels(pixels, theta, center=torch.tensor([512, 512])):

        R = torch.tensor(
            [
                [torch.cos(theta), -torch.sin(theta)],
                [torch.sin(theta), torch.cos(theta)],
            ]
        )

        pix = (pixels - center) @ R.to(torch.float32)

        # pix = pixels @ R.to(torch.float32)
        return pix + center

    @staticmethod
    def get_center_translation(bbox_center, new_center=torch.tensor([512, 512])):
        """Adjust the center of a bounding box to a new center.

        :param bbox_center: Tuple of (y, x) representing the current bbox
          center.
        :param new_center: Tuple of (new_y, new_x), the desired new center.
        :return: New center of the bbox as a tuple (new_y, new_x).
        """
        # Calculate the difference (offset) needed to move the bbox center
        dy, dx = new_center[0] - bbox_center[0], new_center[1] - bbox_center[1]
        return dy, dx

    def align_aspect_ratio(self, aspect_ratio):

        current_ratio = self.natural_aspect_ratio
        return aspect_ratio / current_ratio

    def find_R_T_for_injection(
        self,
        top_left,
        top_right,
        bottom_left,
        bottom_right,
        image_shape,
        return_aspect_ratio=False,
    ):
        global IMAGE_SIZE

        rendering_angle = find_angle_from_bbox(
            top_left, bottom_left, top_right, bottom_right, degrees=True
        )
        extreme_pixels, extreme_verts = self.get_extreme_pixels(
            rendering_angle, image_shape
        )

        R = self.get_R(rendering_angle)
        # find T_z
        alpha = self.linear_alignment_of_bboxes(
            bottom_left_real=bottom_left,
            bottom_right_real=bottom_right,
            top_left_real=top_left,
            top_right_real=top_right,
            bottom_left_injected=extreme_pixels[0],
            bottom_right_injected=extreme_pixels[1],
            top_left_injected=extreme_pixels[2],
            top_right_injected=extreme_pixels[3],
        )  # todo - maybe it shouldn't be magic numbers?

        new_pixels_values = (alpha * extreme_pixels.reshape(-1)).reshape(
            extreme_pixels.shape
        )

        rotated_pix = InjectedObject.rotate_pixels(
            new_pixels_values,
            (rendering_angle * np.pi) / 180,
            center=torch.tensor([IMAGE_SIZE / 2.0, IMAGE_SIZE / 2.0]),
        )
        center_x = rotated_pix[0, 1] + (rotated_pix[1, 1] - rotated_pix[0, 1]) / 2
        center_y = rotated_pix[0, 0] + (rotated_pix[2, 0] - rotated_pix[0, 0]) / 2
        bbox_center = torch.stack([center_y, center_x])
        dy, dx = InjectedObject.get_center_translation(
            bbox_center=bbox_center, new_center=torch.tensor([512, 512])
        )
        centerlized_pix = rotated_pix + torch.stack([dy, dx])

        bl = centerlized_pix[0]
        br = centerlized_pix[1]
        tl = centerlized_pix[2]

        dx = (br - bl)[1]
        dy = (bl - tl)[0]

        aspect_ratio = dy / dx

        aspect_ratio = self.align_aspect_ratio(aspect_ratio=aspect_ratio)

        T_Z = self.find_T_Z(
            centerlized_pix,
            extreme_verts,
            torch.eye(3)[None, :, :],
            image_shape,
            aspect_ratio=1.0 / aspect_ratio,
        )

        T_X, T_Y = self.find_T_X_T_Y(
            pixels_values=new_pixels_values,
            verts=extreme_verts,
            R=R,
            image_shape=image_shape,
            T_Z=T_Z,
            aspect_ratio=1 / aspect_ratio,
        )

        if return_aspect_ratio is True:
            return R, torch.tensor([[T_X, T_Y, T_Z]]), extreme_pixels, aspect_ratio
        return R, torch.tensor([[T_X, T_Y, T_Z]]), extreme_pixels

    def __call__(
        self,
        top_left,
        top_right,
        bottom_left,
        bottom_right,
        image_shape,
        path,
        i,
        debug_draw=False,
    ):

        bbox = torch.stack([bottom_left, bottom_right, top_left, top_right])
        angle = find_angle_from_bbox(
            top_left, bottom_left, top_right, bottom_right, degrees=True
        )
        bbox = self.rotate_pixels(
            bbox.to(torch.float32), theta=torch.tensor([(np.pi * angle) / 180])
        )  # now the bbox is axis aligned

        R, T, extreme_pixels, aspect_ratio = self.find_R_T_for_injection(
            top_left=bbox[2],
            top_right=bbox[3],
            bottom_left=bbox[0],
            bottom_right=bbox[1],
            image_shape=image_shape,
            return_aspect_ratio=True,
        )

        image = self.render_mesh(
            angle=angle, T_z=3, T=T, R=self.base_R, aspect_ratio=aspect_ratio
        )

        image = image.permute(2, 0, 1)
        image = torchvision.transforms.functional.rotate(image, angle.item(), fill=1)
        image = image.permute(1, 2, 0)
        image = (image * 255).cpu().numpy().astype(np.uint8)

        segmantation_mask = (image != 255).astype(np.uint8)
        # image = np.transpose(image, (2, 0, 1))

        bbox = torch.stack([bottom_left, bottom_right, top_left, top_right])
        angle = angle.item()
        # Image.fromarray(image).save(f'{path}/bbox_{i}_angle{angle:.1f}.png')
        if debug_draw:
            draw_pixels(
                image,
                bbox[:, 0].numpy().astype(np.int32),
                bbox[:, 1].numpy().astype(np.int32),
                square_size=5,
                paint_color=np.array([255, 0, 0]).astype(np.uint8),
            )

            Image.fromarray(image).save(
                f'{path}/bbox_{i}_angle_{angle:.2f}.png'
            )
        # alpha_chanel = segmantation_mask[:, :, 0][:, :, None] * 255
        # rgba = np.dstack([image, alpha_chanel])

        return image, segmantation_mask
