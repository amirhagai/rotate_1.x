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
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesUV,
    TexturesVertex
)
import torch
from pytorch3d.transforms import RotateAxisAngle



# add path for demo utils functions 
import sys
import os
sys.path.append(os.path.abspath(''))



def find_angle_from_bbox(top_left, bottom_left, top_right, bottom_right, degrees=False):

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

    return angle_degrees

def get_extreme_pixels(x_pixel_int, y_pixel_int, indecis=False):

    " the code assume the sahpe is rectanguler and that dx > dy when the angle is 0"

    x_pixel_int_min = x_pixel_int.argmin()
    x_pixel_int_max = x_pixel_int.argmax()
    y_pixel_int_min = y_pixel_int.argmin()
    y_pixel_int_max = y_pixel_int.argmax()


    x_min = x_pixel_int[x_pixel_int_min]
    x_max = x_pixel_int[x_pixel_int_max]
    y_min = y_pixel_int[y_pixel_int_min]
    y_max = y_pixel_int[y_pixel_int_max]

    z1 = x_pixel_int == x_min
    y_for_x_min = y_pixel_int[z1].max()

    # bottom_left, top_right, top_left, bottom_right
    if y_for_x_min == y_max:

        assert not indecis, "not implemented yet for 0"
        # angle is 0
        if x_max - x_min > y_max - y_min:
            pix_x = np.array([x_min      ,    x_max      ,    x_min      ,    x_max])
            pix_y = np.array([y_min      ,    y_max      ,    y_max      ,    y_min])
        else:
        #angle is 90
            pix_x = np.array([x_max      ,    x_min      ,    x_min      ,    x_max])
            pix_y = np.array([y_max      ,    y_min      ,    y_max      ,    y_min])

        return pix_x, pix_y  

    z2 = x_pixel_int == x_max
    y_for_x_max = y_pixel_int[z2].min()

    z3 = y_pixel_int == y_min
    x_for_y_min = x_pixel_int[z3].max()

    z4 = y_pixel_int == y_max
    x_for_y_max = x_pixel_int[z4].min()

    # bottom_left, top_right, top_left, bottom_right, works for angle < 90

    pix_x = np.array([x_min      ,    x_max      ,    x_for_y_min,    x_for_y_max])
    pix_y = np.array([y_for_x_min,    y_for_x_max,    y_min      ,    y_max])

    if not indecis:
        return pix_x, pix_y
    else:
        return pix_x, pix_y, [[z1, y_pixel_int[z1].argmax()], [z2, y_pixel_int[z2].argmin()], [z3, x_pixel_int[z3].argmax()], [z4, x_pixel_int[z4].argmin()]]


def _define_verts_ndc(verts, R, T, angle):
    R, T = look_at_view_transform(70, elev=0, azim=180) 

    rotation = RotateAxisAngle(angle=angle, axis="Z").get_matrix()
    R = rotation[:, :3, :3]
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)    
    R_T = cameras.get_world_to_view_transform().get_matrix()

    r = RotateAxisAngle(angle= (angle), axis="Z").get_matrix()

    R_T [:, :3, :3] = r[:, :3, :3]
    P = cameras.get_projection_transform().get_matrix()

    vertex = torch.concatenate([verts, torch.ones((len(verts), 1), device = verts.device)], dim=1)
    vertex_clip = (vertex @ R_T[0]) @ P[0]
    vertex_ndc = vertex_clip[:, :3] / vertex_clip[:, 3:]   

    return vertex_ndc, R_T, cameras


def test_formula(verts):
    R, T = look_at_view_transform(70, elev=0, azim=180) 

    for angle in [10 * i for i in range(1, 19)]:
        vertex_ndc, R_T, cameras = _define_verts_ndc(verts, R, T, angle)

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

        assert z.item() == 0, "formula for calculating verts posiyion over the image is wrong"
    print("test formula is done")


def test_find_angle(verts):
    R, T = look_at_view_transform(70, elev=0, azim=180) 

    for angle in [10 * i for i in range(0, 10)]:
        vertex_ndc, _, _ = _define_verts_ndc(verts, R, T, angle)

        # Step 4: Map NDC to screen coordinates, ndc range from (-1, 1)
        width, height = 1024, 1024  # Example viewport dimensions
        x_pixel = ((1 - vertex_ndc[:, 0]) / 2.0) * width
        y_pixel = ((1 - vertex_ndc[:, 1]) / 2.0) * height 

        width, height = 1024, 1024  # Example dimensions
        x_pixel_int = np.clip(x_pixel.cpu().numpy().round().astype(int), 0, width - 1)
        y_pixel_int = np.clip(y_pixel.cpu().numpy().round().astype(int), 0, height - 1)

        if angle == 90:
            print()

        pix_x, pix_y = get_extreme_pixels(x_pixel_int, y_pixel_int)
        
        pixels_to_highlight = np.stack((pix_y, pix_x), axis=1)

        bottom_left, top_right, top_left, bottom_right = pixels_to_highlight

        x1 = find_angle_from_bbox(top_left, bottom_left, top_right, bottom_right, degrees=False)
        x2 = find_angle_from_bbox(top_left, bottom_left, top_right, bottom_right, degrees=True)

        assert (abs(x2.item() - angle) < 1), f"find angle from bbox is not working as excpected, angle is {angle}, found {x2.item()}"
    print("test find angle is done")


def visualization_tests():
    pass # todo

class InjectedObject:


    def __init__(self, obj_file_path, TZ_start = 70) -> None:

        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            torch.cuda.set_device(device)
        else:
            device = torch.device("cpu")

        self.mesh = load_objs_as_meshes([obj_file_path], device=device)
        # mesh._verts_list[0] contains list of (x, y, z) verts
        # tensor with shape [num of verts,]

        self.verts = mesh.verts_packed()  # Get the vertices of the mesh
        self.TZ_start = TZ_start

        self.base_R, self.base_T = look_at_view_transform(self.TZ_start, elev=0, azim=180) 
        self.camera = FoVPerspectiveCameras(device=device, R=self.base_R, T=self.base_T)


    def get_extreme_pixels(self, angle, image_shape, debug_seasion=False):

        """
        vertex to pixel equation -> 
        xs = fov_weight * (V @ R1 + T_X) / (verts @ R3 + T_Z)
        xy = fov_weight * (V @ R2 + T_Y) / (verts @ R3 + T_Z) 

        we now doing it with base T as we only want to get the extreme pixels
        """

        R = InjectedObject.get_R(angle)
        fov = (self.camera.fov * np.pi) / 180 # angles to radins
        fov_weight = 1 / np.tan((fov / 2))

        R1 = R[:3, 0][0]
        R2 = R[:3, 1][0]
        R3 = R[:3, 2][0]

        V_R_1 = verts @ R1
        V_R_2 = verts @ R2
        V_R_3 = verts @ R3

        xs = fov_weight * (V_R_1 + self.base_T[0, 0]) / (V_R_3 + self.base_T[0, 2])
        ys = fov_weight * (V_R_2 + self.base_T[0, 1]) / (V_R_3 + self.base_T[0, 2])

        width, height = image_shape[1], image_shape[2]  # Example viewport dimensions
        x_pixel = ((xs + 1) / 2.0) * width
        y_pixel = ((1 - ys) / 2.0) * height 

        x_pixel_int = np.clip(x_pixel.cpu().numpy().round().astype(int), 0, width - 1)
        y_pixel_int = np.clip(y_pixel.cpu().numpy().round().astype(int), 0, height - 1)
        
        # bottom_left, top_right, top_left, bottom_right
        # [[z1, y_pixel_int[z1].argmax()], [z2, y_pixel_int[z2].argmin()], [z3, x_pixel_int[z3].argmax()], [z4, x_pixel_int[z4].argmin()]]

        pix_x, pix_y, indecis = get_extreme_pixels(x_pixel_int, y_pixel_int, indecis=True)

        indecis_bottom_left, indecis_top_right, indecis_top_left, indecis_bottom_right = indecis
        v_bottom_left = verts[indecis_bottom_left[0]][indecis_bottom_left[1]]
        v_top_right = verts[indecis_top_right[0]][indecis_top_right[1]]
        v_top_left = verts[indecis_top_left[0]][indecis_top_left[1]]
        v_bottom_right = verts[indecis_bottom_right[0]][indecis_bottom_right[1]]


        if debug_seasion:
            assert (fov_weight * (v_bottom_left @ R1) / (v_bottom_left @R3 + self.base_T[0, 2]) - xs[indecis_bottom_left[0]][indecis_bottom_left[1]] ).abs() < 1e-5, "senaty test"
            assert (fov_weight * (v_top_right @ R1) / (v_top_right @R3 + self.base_T[0, 2]) - xs[indecis_top_right[0]][indecis_top_right[1]] ).abs() < 1e-5, "senaty test"
            assert (fov_weight * (v_top_left @ R1) / (v_top_left @R3 + self.base_T[0, 2]) - xs[indecis_top_left[0]][indecis_top_left[1]]).abs() < 1e-5, "senaty test"
            assert (fov_weight * (v_bottom_right @ R1) / (v_bottom_right @R3 + self.base_T[0, 2]) - xs[indecis_bottom_right[0]][indecis_bottom_right[1]] ).abs() < 1e-5, "senaty test"


            assert (fov_weight * (v_bottom_left @ R2) / (v_bottom_left @R3 + self.base_T[0, 2]) - ys[indecis_bottom_left[0]][indecis_bottom_left[1]] ).abs() < 1e-5, "senaty test"
            assert (fov_weight * (v_top_right @ R2) / (v_top_right @R3 + self.base_T[0, 2]) - ys[indecis_top_right[0]][indecis_top_right[1]] ).abs() < 1e-5, "senaty test"
            assert (fov_weight * (v_top_left @ R2) / (v_top_left @R3 + self.base_T[0, 2]) - ys[indecis_top_left[0]][indecis_top_left[1]]).abs() < 1e-5, "senaty test"
            assert (fov_weight * (v_bottom_right @ R2) / (v_bottom_right @R3 + self.base_T[0, 2]) - ys[indecis_bottom_right[0]][indecis_bottom_right[1]] ).abs() < 1e-5, "senaty test"


            print(f"bottom left - ({ys[indecis_bottom_left[0]][indecis_bottom_left[1]], xs[indecis_bottom_left[0]][indecis_bottom_left[1]]})")
            print(f"top right - ({ys[indecis_top_right[0]][indecis_top_right[1]], xs[indecis_top_right[0]][indecis_top_right[1]]})")
            print(f"top left - ({ys[indecis_top_left[0]][indecis_top_left[1]], xs[indecis_top_left[0]][indecis_top_left[1]]})")
            print(f"bottom right - ({ys[indecis_bottom_right[0]][indecis_bottom_right[1]], xs[indecis_bottom_right[0]][indecis_bottom_right[1]]})")
        return torch.stack([torch.tensor(pix_y), torch.tensor(pix_x)], axis=1), torch.stack([v_bottom_left, v_top_right, v_top_left, v_bottom_right], axis=0)

    @staticmethod
    def calculate_distance(p1, p2):
        """Calculate the Euclidean distance between two points."""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    @staticmethod
    def calculate_centered_square_bbox(corners, new_center = [451, 513]):
        # Assuming corners are [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
        # Calculate diagonal lengths and then find the average to approximate the bbox size
        # Calculate the lengths of the sides of the rectangle
        side1 = InjectedObject.calculate_distance(corners[0], corners[2])
        side2 = InjectedObject.calculate_distance(corners[1], corners[2])
        
        # Determine which is width and which is height by comparing the sides
        width, height = (side1, side2) if side1 < side2 else (side2, side1)
        
        # Calculate the corners of the new rectangle bbox centered at the new_center
        half_width = width / 2
        half_height = height / 2
        rectangle_corners = [
            [new_center[0] + half_width, new_center[1] - half_height],  # Top-right corner
            [new_center[0] - half_width, new_center[1] + half_height],  # Bottom-left corner
            [new_center[0] - half_width, new_center[1] - half_height],  # Top-eft corner 
            [new_center[0] + half_width, new_center[1] + half_height],  # Bottom-right corner 

        ]
        
        return torch.tensor(rectangle_corners)


    @staticmethod
    def get_R(angle):
        rotation = RotateAxisAngle(angle=angle, axis="Z").get_matrix()
        R = rotation[:, :3, :3]
        return R


    def linear_alignment_of_bboxes(self, top_left_real, top_right_real, bottom_left_real, bottom_right_real, 
                                   top_left_injected, top_right_injected, bottom_left_injected, bottom_right_injected):
        """
        return best a for ||a * injected_bbox_w_h - real_bbox_w_h||^2 
        we don't want to find the best alignment as real_bbox not centarlized and injected_bbox is
        
        """

        dx_real  = top_left_real[1] - top_right_real[1]
        dy_real = bottom_left_real[0] - top_left_real[0]

        dx_injected = top_left_injected[1] - top_right_injected[1]
        dy_injected =  bottom_left_injected[0] - top_left_injected[0]

        x = torch.tensor([dx_real    , dy_real    , top_left_real[0]    , top_right_real[0]    , bottom_left_real[0]    , bottom_right_real[0]
                          ,  top_left_real[1]    , top_right_real[1]    , bottom_left_real[1]    , bottom_right_real[1]]).reshape(-1)
        
        y = torch.tensor([dx_injected, dy_injected, top_left_injected[0], top_right_injected[0], bottom_left_injected[0], bottom_right_injected[0]
                          ,  top_left_injected[1], top_right_injected[1], bottom_left_injected[1], bottom_right_injected[1]]).reshape(-1)
        

        x = torch.tensor([bottom_left_injected[0], bottom_left_injected[1], top_right_injected[0], top_right_injected[1], top_left_injected[0], top_left_injected[1], bottom_right_injected[0], bottom_right_injected[1]])

        y = torch.tensor([bottom_left_real[0], bottom_left_real[1], top_right_real[0], top_right_real[1], top_left_real[0], top_left_real[1], bottom_right_real[0], bottom_right_real[1]])
        # return (x @ y) / (x @ x)
        return   y.to(torch.float32) / x.to(torch.float32) 
    


    def _center_pixels(self, pixels_values, image_shape):
        """
        we get pixels values at image space and we want to take them 
        to ndc space where our equation of verts to pixels is living

        we know that 
        x_pixel = ((vertex_ndc[:, 0] + 1) / 2.0) * width
        y_pixel = ((1 - vertex_ndc[:, 1]) / 2.0) * height 

        which gives us 

        vertex_ndc[:, 0] = ((x_pixel * 2.) / width ) - 1
        vertex_ndc[:, 1] = 1 - ((y_pixel * 2.) / height)

        """

        width, height = image_shape[1:]

        xs = ((pixels_values[:, 1].to(torch.float64) * 2.) / width ) - 1
        # xs = 1 - ((pixels_values[:, 1].to(torch.float64)   * 2.) / width)
        ys = 1 - ((pixels_values[:, 0].to(torch.float64)   * 2.) / height)
        # ys = ((pixels_values[:, 0].to(torch.float64) * 2.) / height ) - 1
        return xs, ys
    

     

    def find_T_Z(self, pixels_values, verts, R, image_shape):
        """
        this funciton assume T_X = T_Y = 0, 
        vertex to pixel equation -> 
        xs = fov_weight * (V @ R1 + T_X) / (verts @ R3 + T_Z)
        xy = fov_weight * (V @ R2 + T_Y) / (verts @ R3 + T_Z)
        for the time being T_X = 0, T_Y = 0 T_Z is our variable
        so we want to solve 

        xs = fov_weight * (V @ R1 ) / (verts @ R3 + T_Z)
        xy = fov_weight * (V @ R2 ) / (verts @ R3 + T_Z)

        where T_Z is our variable 

        so the solution is 

        T_Z = (fov_weight * (V @ R1 ) - (verts @ R3)) / xs

        xs, ys assume to be centerlized and in the range of (-1, 1)

        """

        fov = (self.camera.fov * np.pi) / 180 # angles to radins
        fov_weight = 1 / np.tan((fov / 2))

        R1 = R[:3, 0][0]
        R2 = R[:3, 1][0]
        R3 = R[:3, 2][0]


        xs, ys = self._center_pixels(pixels_values, image_shape)

        V_R_1 = verts @ R1
        V_R_2 = verts @ R2
        V_R_3 = verts @ R3

        # TODO - does returning the mean is the right thing? how can I test it?
        T_Z_X = ((fov_weight * V_R_1 / xs) - V_R_3).mean()
        T_Z_Y = ((fov_weight * V_R_2 / ys) - V_R_3).mean()

        return (T_Z_X + T_Z_Y) / 2.




    def find_T_X_T_Y(self, pixels_values, verts, R, image_shape, T_Z):
        """
        vertex to pixel equation -> 
        xs = fov_weight * (V @ R1 + T_X) / (verts @ R3 + T_Z)
        xy = fov_weight * (V @ R2 + T_Y) / (verts @ R3 + T_Z)       

        so

        T_X = ((xs * (verts @ R3 + T_Z)) /  fov_weight) - V @ R1
        T_Y = ((ys * (verts @ R3 + T_Z)) /  fov_weight) - V @ R2
        """


        # TODO - FIX CODE DUPLICATION
        fov = (self.camera.fov * np.pi) / 180 # angles to radins
        fov_weight = 1 / np.tan((fov / 2))

        R1 = R[:3, 0][0]
        R2 = R[:3, 1][0]
        R3 = R[:3, 2][0]


        xs, ys = self._center_pixels(pixels_values, image_shape)

        V_R_1 = verts @ R1
        V_R_2 = verts @ R2
        V_R_3 = verts @ R3

        #TODO - why does T_Y needs the "-" sign?
        T_X = -(((xs * (V_R_3 + T_Z)) / fov_weight) - V_R_1).mean()
        T_Y = (((ys * (V_R_3 + T_Z)) / fov_weight) - V_R_2).mean()

        return T_X, T_Y



    def find_R_T_for_injection(self, top_left, top_right, bottom_left, bottom_right, image_shape):

        rendering_angle = find_angle_from_bbox(top_left, bottom_left, top_right, bottom_right, degrees=True)
        extreme_pixels, extreme_verts = self.get_extreme_pixels(rendering_angle, image_shape)
        print()
        # print(F"EXTREME VERTS ARE - {extreme_verts}")
        print()
        R = self.get_R(rendering_angle)
        
        #bbox_aspect_ratio = float(top_right - top_left) / float(bottom_right - bottom_left) # Todo - should i use it?


        #top_left_injected, top_right_injected, bottom_left_injected, bottom_right_injected

        # find T_z 
        alpha = self.linear_alignment_of_bboxes(top_left_real=torch.tensor(top_left),
                                                top_right_real=torch.tensor(top_right), 
                                                bottom_left_real=torch.tensor(bottom_left),
                                                bottom_right_real=torch.tensor(bottom_right),

                                                top_left_injected=extreme_pixels[2],
                                                top_right_injected=extreme_pixels[1],
                                                bottom_left_injected=extreme_pixels[0],
                                                bottom_right_injected=extreme_pixels[3]) # todo - maybe it shouldn't be magic numbers?
        
        new_pixels_values = (alpha * extreme_pixels.reshape(-1)).reshape(extreme_pixels.shape)


        pix_for_T_Z = InjectedObject.calculate_centered_square_bbox(new_pixels_values)

        T_Z = self.find_T_Z(new_pixels_values, extreme_verts, R, image_shape)

        # print(f"\tnormal t-z is {T_Z}")



        T_Z = self.find_T_Z(pix_for_T_Z , extreme_verts, torch.eye(3)[None, :, :], [3, 1025, 902])

        # print(f"\tnew pixels, new R, new center - {T_Z}")

        

        T_X, T_Y = self.find_T_X_T_Y(new_pixels_values, extreme_verts, R, image_shape, T_Z)

        return R, torch.tensor([T_X, T_Y, T_Z]), extreme_pixels
    

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
        

def visualize_verts_over_object(R, T, y_pixel_int, x_pixel_int, extreme_y, extreme_x, vis_verts=True):
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T )

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
        shader=SoftPhongShader(device=device, cameras=cameras, lights=lights)
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

    #top-left, top-right, bottom-right, bottom-left
    draw_bbox(image, [pixels_to_highlight[2], pixels_to_highlight[1], pixels_to_highlight[3], pixels_to_highlight[0]], color=(0, 0, 128))
    return image




def test_injection():


    DATA_DIR = "/app/mmrotate/3Ddata/"
    obj_filename = os.path.join(DATA_DIR, "meshes/Container.obj")


    

    
    injection_object = InjectedObject(obj_filename)

    # define image

    R, T = look_at_view_transform(70, elev=0, azim=180) 

    
    for angle in [5, 20, 45, 78, 0]:
        print(f"angle - {angle}")
        for T in [
                  torch.tensor([[5., 5., 20.]]), torch.tensor([[5., 7., 40.]]), torch.tensor([[2., 3., 50.]]),
                  torch.tensor([[5., -3., 20.]]), torch.tensor([[-5., -6., 40.]]), torch.tensor([[-2., 7., 50.]]),
                  ]:

            rotation = RotateAxisAngle(angle=angle, axis="Z").get_matrix()
            R = rotation[:, :3, :3]           



            cameras = FoVPerspectiveCameras(device=device, R=R, T=T )



            # eps = kwargs.get("eps", None)
            # verts_view = cameras.get_world_to_view_transform(**kwargs).transform_points(
            #     verts_world, eps=eps
            # )
            # to_ndc_transform = cameras.get_ndc_camera_transform(**kwargs)
            # projection_transform = try_get_projection_transform(cameras, kwargs)
            # if projection_transform is not None:
            #     projection_transform = projection_transform.compose(to_ndc_transform)
            #     verts_ndc = projection_transform.transform_points(verts_view, eps=eps)
            # else:
            #     # Call transform_points instead of explicitly composing transforms to handle
            #     # the case, where camera class does not have a projection matrix form.
            #     verts_proj = cameras.transform_points(verts_world, eps=eps)
            #     verts_ndc = to_ndc_transform.transform_points(verts_proj, eps=eps)

            # verts_ndc[..., 2] = verts_view[..., 2]
            # meshes_ndc = meshes_world.update_padded(new_verts_padded=verts_ndc)


            # cameras = FoVPerspectiveCameras(device=device, R=R, T=T)



            R_T = cameras.get_world_to_view_transform().get_matrix()

            r = RotateAxisAngle(angle= (angle), axis="Z").get_matrix()
            

            R_T [:, :3, :3] = r[:, :3, :3]
            # R_T[:, 3, :3][0, :2] = -R_T[:, 3, :3][0, :2]

            P = cameras.get_projection_transform().get_matrix()

            vertex = torch.concatenate([verts, torch.ones((len(verts), 1), device = verts.device)], dim=1)

            vertex_clip = (vertex @ R_T[0]) @ P[0]

            # Step 3: Perform perspective division to get NDC
            vertex_ndc = vertex_clip[:, :2] / vertex_clip[:, 3:] 

            # Step 4: Map NDC to screen coordinates, ndc range from (-1, 1)
            width, height = 1024, 1024  # Example viewport dimensions
            x_pixel = ((1 - vertex_ndc[:, 0]) / 2.0) * width
            y_pixel = ((1 - vertex_ndc[:, 1]) / 2.0) * height 


            x_pixel_int = np.clip(x_pixel.cpu().numpy().round().astype(int), 0, width - 1)
            y_pixel_int = np.clip(y_pixel.cpu().numpy().round().astype(int), 0, height - 1)

            

            pix_x, pix_y = get_extreme_pixels(x_pixel_int, y_pixel_int)
            # im_vis = visualize_verts_over_object(R, T, y_pixel_int, x_pixel_int, extreme_y=pix_y, extreme_x=pix_x, vis_verts=True)
            # plt.imshow(im_vis)

            pixels_to_highlight = np.stack((pix_y, pix_x), axis=1)

            pix_for_T_Z = InjectedObject.calculate_centered_square_bbox(pixels_to_highlight)

            # print(pixels_to_highlight)

            print(f"\treal T - {T}")


            #bottom_left, top_right, top_left, bottom_right
            # R_new, T_new = injection_object.find_R_T_for_injection(bottom_left=pixels_to_highlight[0], top_right=pixels_to_highlight[1], top_left=pixels_to_highlight[2],
            #                                                 bottom_right=pixels_to_highlight[3], image_shape=[3, 1024, 1024])
            
            
            
            

            # im_vis = visualize_verts_over_object(R_new, T_new[None, :], y_pixel_int, x_pixel_int, extreme_y=pix_y, extreme_x=pix_x, vis_verts=True)

            # plt.imshow(im_vis)

            #bottom_left, top_right, top_left, bottom_right
            # RANDOM BBOX (284.0, 398.0), (265.0, 468.0)  (263.0, 399.0), (288.0, 468.0)
            pixels_to_highlight = torch.tensor([[284.0, 398.0], [265.0, 468.0], [263.0, 399.0], [288.0, 468.0]])




            corners = pixels_to_highlight

            # Calculate the centroid of the rectangle
            centroid = torch.mean(corners, axis=0)

            # Define the scale factor (e.g., 1.1 for 10% larger)
            scale_factor = 3

            # Calculate the new corners
            new_corners = np.empty_like(corners)
            for i, corner in enumerate(corners):
                vector_from_centroid = corner - centroid
                scaled_vector = vector_from_centroid * scale_factor
                new_corners[i] = centroid + scaled_vector

            pixels_to_highlight = new_corners
            R_new, T_new, extreme_pixels = injection_object.find_R_T_for_injection(bottom_left=pixels_to_highlight[0], top_right=pixels_to_highlight[1], top_left=pixels_to_highlight[2],
                                                            bottom_right=pixels_to_highlight[3], image_shape=[3, 1024, 1024])
            

            im_vis = visualize_verts_over_object(R_new, T_new[None, :], extreme_pixels[:, 0], extreme_pixels[:, 1], extreme_y=pixels_to_highlight[:, 0], extreme_x=pixels_to_highlight[:, 1], vis_verts=False)

            plt.imshow(im_vis)






            print(f"\tfound T - {T}")
            print()


            # render object with diffrent T and R over it (can be done implicitly)

            # validate that we get the same T and R
        print()
        print()
        


if __name__ == "__main__":

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

    verts = mesh.verts_packed()  # Get the vertices of the mesh

    test_injection()

    # test_formula(verts)
    # test_find_angle(verts)




# features = torch.ones((len(verts), 3))
# features[:, 0] = 0
# points = Pointclouds(points=[verts], features=[features])

# # Initialize a camera.
# # With world coordinates +Y up, +X left and +Z in, the front of the cow is facing the -Z direction. 
# # So we move the camera by 180 in the azimuth direction so it is facing the front of the cow. 






# R, T = look_at_view_transform(70, elev=0, azim=180) 
# angle = 45

# folder_path = f"/app/data/test_injected/trainval/images/{angle}"
# os.makedirs(folder_path, exist_ok=True) 
# print(f"angle is - {angle}")
# print()

# rotation = RotateAxisAngle(angle=angle, axis="Z").get_matrix()
# R = rotation[:, :3, :3]
# cameras = FoVPerspectiveCameras(device=device, R=R, T=T)


# R_T = cameras.get_world_to_view_transform().get_matrix()

# r = RotateAxisAngle(angle= (-angle), axis="Z").get_matrix()

# R_T [:, :3, :3] = r[:, :3, :3]
# # R_T[:, 3, 0:2] = -0
# # Get the projection matrix (P)
# P = cameras.get_projection_transform().get_matrix()

# vertex = torch.concatenate([verts, torch.ones((len(verts), 1), device = verts.device)], dim=1)



# vertex_clip = (vertex @ R_T[0]) @ P[0]

# # Step 3: Perform perspective division to get NDC
# vertex_ndc = vertex_clip[:, :3] / vertex_clip[:, 3:] 

# R1 = R_T[0][:3, 0]
# R2 = R_T[0][:3, 1]
# R3 = R_T[0][:3, 2]
# T_VEC = R_T[0][3, :3]

# T_X = T_VEC[0]
# T_Y = T_VEC[1]
# T_Z = T_VEC[2]

# fov_weight = (1 / torch.tan(((np.pi / 180) * cameras.fov) / 2)) 

# xs = fov_weight * (verts @ R1 + T_X) / (verts @ R3 + T_Z)
# ys = fov_weight * (verts @ R2 + T_Y) / (verts @ R3 + T_Z)

# pix = torch.stack((xs, ys), dim=1)

# z = (torch.abs(pix - vertex_ndc[:, :2] ) > 1e-5).sum()

# # Step 4: Map NDC to screen coordinates, ndc range from (-1, 1)
# width, height = 1024, 1024  # Example viewport dimensions
# x_pixel = ((vertex_ndc[:, 0] + 1) / 2.0) * width
# y_pixel = ((1 - vertex_ndc[:, 1]) / 2.0) * height 

# width, height = 1024, 1024  # Example dimensions
# x_pixel_int = np.clip(x_pixel.cpu().numpy().round().astype(int), 0, width - 1)
# y_pixel_int = np.clip(y_pixel.cpu().numpy().round().astype(int), 0, height - 1)


# x_pixel_int_min = x_pixel_int.argmin()
# x_pixel_int_max = x_pixel_int.argmax()

# y_pixel_int_min = y_pixel_int.argmin()
# y_pixel_int_max = y_pixel_int.argmax()


# z1 = x_pixel_int == x_pixel_int[x_pixel_int_min]
# y_for_x_min = y_pixel_int[z1].max()


# z2 = x_pixel_int == x_pixel_int[x_pixel_int_max]
# y_for_x_max = y_pixel_int[z2].min()


# z3 = y_pixel_int == y_pixel_int[y_pixel_int_min]
# x_for_y_min = x_pixel_int[z3].max()


# z4 = y_pixel_int == y_pixel_int[y_pixel_int_max]
# x_for_y_max = x_pixel_int[z4].min()


# pix_x = np.array([x_pixel_int[x_pixel_int_min], x_pixel_int[x_pixel_int_max], x_for_y_min, x_for_y_max])
# pix_y = np.array([y_for_x_min, y_for_x_max, y_pixel_int[y_pixel_int_min], y_pixel_int[y_pixel_int_max]])


# print(pix_y)
# print(pix_x)

# square_size = 5  # This will create a 5x5 square

# image = np.zeros((1024, 1024, 3)).astype(np.uint8)
# paint_color = np.array([255, 0, 0])  # Red


# pixels_to_highlight = np.stack((pix_y, pix_x), axis=1)

# bottom_left, top_right, top_left, bottom_right = pixels_to_highlight

# x1 = find_angle_from_bbox(top_left, bottom_left, top_right, bottom_right, degrees=False)
# x2 = find_angle_from_bbox(top_left, bottom_left, top_right, bottom_right, degrees=True)

# for y, x in pixels_to_highlight:
#     # Ensure the square stays within image bounds
#     x_start = max(0, x - square_size // 2)
#     y_start = max(0, y - square_size // 2)
#     x_end = min(image.shape[1], x + square_size // 2 + 1)
#     y_end = min(image.shape[0], y + square_size // 2 + 1)
    
#     # Paint the square around each pixel
#     image[y_start:y_end, x_start:x_end] = paint_color

# # Define the paint color

# # Paint the pixels using NumPy advanced indexing
# # image[pix_y, pix_x] = paint_color
# # image[pix_y, pix_x] = paint_color
# Image.fromarray((image).astype(np.uint8)).save(f"{folder_path}/hand_craft_extreme_point.png")

# image = np.zeros((1024, 1024, 3)).astype(np.uint8)
# # Paint the pixels using NumPy advanced indexing
# image[y_pixel_int, x_pixel_int] = paint_color
# # image[pix_y, pix_x] = paint_color
# Image.fromarray((image).astype(np.uint8)).save(f"{folder_path}/my_projection.png")




# ind = torch.concatenate([y_pixel, x_pixel])
# xxx = render_points_only(points, cameras)

# Image.fromarray((.5 * (xxx * 255) + .5 * image.astype(np.float32)).astype(np.uint8)).save(f"{folder_path}/my_proj_over_camera_proj.png")
# print("done")

# raster_settings = RasterizationSettings(
#     image_size=1024, 
#     blur_radius=0.0, 
#     faces_per_pixel=1, 
#     bin_size=0,  # Use naive rasterization
#     max_faces_per_bin=None  # Optionally, adjust this value based on your needs
# )

# # Place a point light in front of the object. As mentioned above, the front of the cow is facing the 
# # -z direction. 
# lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

# # Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will 
# # interpolate the texture uv coordinates for each vertex, sample from a texture image and 
# # apply the Phong lighting model
# renderer = MeshRenderer(
#     rasterizer=MeshRasterizer(
#         cameras=cameras, 
#         raster_settings=raster_settings
#     ),
#     shader=SoftPhongShader(
#         device=device, 
#         cameras=cameras,
#         lights=lights
#     )
# )

# im_for_rand = Image.open("/app/data/test_injected/trainval/images/P0005__1024__0___0.png")
# im_for_rand = np.array(im_for_rand).astype(np.float32) / 255.




# images = renderer(mesh)
# # images = renderer(points)
# mask = (images[0, ..., :3].cpu().numpy() == 1).astype(np.float32)
# plt.figure(figsize=(10, 10))
# # plt.imshow((1 - mask) * images[0, ..., :3].cpu().numpy() + mask * im_for_rand)
# plt.axis("off")

# plt.savefig("/app/mmrotate/3Ddata/rendering_images/first_try.png")


# end_im = (((1 - mask) * images[0, ..., :3].cpu().numpy() + mask * im_for_rand) * 255).astype(np.uint8)
# Image.fromarray(end_im).save(f"{folder_path}/pres_dota.png")
# Image.fromarray((xxx * 255).astype(np.uint8)).save(f"{folder_path}/point_cloud_there.png")



# Image.fromarray((.5 * (xxx * 255) + .5 * end_im.astype(np.float32)).astype(np.uint8)).save(f"{folder_path}/theres_over_obj.png")


# Image.fromarray((.5 * image + .5 * end_im.astype(np.float32)).astype(np.uint8)).save(f"{folder_path}/my_over-obj.png")


# print(f"done {angle}")
# print()
# print()


