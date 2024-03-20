from infer_camera_parameters import InjectedObject, draw_pixels
from pytorch3d.io import load_objs_as_meshes
import numpy as np
from PIL import Image
import os
from pathlib import Path
import torch


IMAGE_SIZE = 1024


def test_injection_sanity_check(obj_filename, app_path):

    global IMAGE_SIZE

    path = f"{app_path}/data/test_injected/trainval/images/sanity_check"
    os.makedirs(path, exist_ok=True)

    for angle in [0, 20, 50, 45, 78][::-1]:
        print(f"angle - {angle}")
        for i, T in enumerate(
            [
                torch.tensor([[5.0, 5.0, 50.0]]),
                torch.tensor([[5.0, 7.0, 40.0]]),
                torch.tensor([[2.0, 3.0, 50.0]]),
                torch.tensor([[5.0, -3.0, 20.0]]),
                torch.tensor([[-5.0, -6.0, 40.0]]),
                torch.tensor([[-2.0, 7.0, 50.0]]),
                torch.tensor([[2.0, -3.0, 60.0]]),
                torch.tensor([[-5.0, 1.0, 80.0]]),
                torch.tensor([[0.0, 0.0, 30.0]]),
            ]
        ):

            print(f"\treal T - {T}")
            # tr, bl, br, tl
            injection_object = InjectedObject(obj_filename, TZ_start=T[0, 2], T=T)

            pixels_to_highlight, verts_ = injection_object.get_extreme_pixels(
                angle=angle, image_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
            )

            injection_object = InjectedObject(obj_filename)

            (
                R_new,
                T_new,
                extreme_pixels,
            ) = injection_object.find_R_T_for_injection(
                bottom_left=pixels_to_highlight[0],
                top_right=pixels_to_highlight[3],
                top_left=pixels_to_highlight[2],
                bottom_right=pixels_to_highlight[1],
                image_shape=[3, IMAGE_SIZE, IMAGE_SIZE],
            )

            image = injection_object.render_mesh(
                angle=angle, T_z=3, T=T_new, R=R_new
            )
            image = (image * 255).cpu().numpy().astype(np.uint8)

            draw_pixels(
                image,
                pixels_to_highlight[:, 0].numpy().astype(np.int32),
                pixels_to_highlight[:, 1].numpy().astype(np.int32),
                square_size=5,
                paint_color=np.array([255, 0, 0]).astype(np.uint8),
            )

            Image.fromarray(image).save(f"{path}/T_{i}_angle_{angle}.png")

            image2, seg = injection_object(
                bottom_left=pixels_to_highlight[0],
                top_right=pixels_to_highlight[3],
                top_left=pixels_to_highlight[2],
                bottom_right=pixels_to_highlight[1],
                image_shape=[3, IMAGE_SIZE, IMAGE_SIZE],
                path=path,
                i=i,
            )

            im = np.abs(
                (image2.astype(np.float32) - image.astype(np.float32))
            ).astype(np.uint8)
            Image.fromarray(im).save(f"{path}/sub_T_{i}_angle_{angle}.png")
            print(f"\tfound T - {T_new}")
            print()

        print()


def understand_aspect_ratio(
    injection_object, T_new, angle, aspect_ratio, bbox, path, name
):
    """visualization for debug"""

    _, verts = injection_object.get_extreme_pixels(
        angle=angle, image_shape=(3, 1024, 1024)
    )

    R = injection_object.get_R(-angle)
    R = R.to(device)

    V_R_1, V_R_2, V_R_3 = injection_object._get_rotated_verts(verts, R)

    # here we do have T
    xs = (
        (1 / aspect_ratio)
        * injection_object.fov_weight
        * (V_R_1 + T_new[0, 0])
        / ((V_R_3 + T_new[0, 2]) * injection_object.f2)
    )
    ys = (
        injection_object.fov_weight
        * (V_R_2 + T_new[0, 1])
        / ((V_R_3 + T_new[0, 2]) * injection_object.f2)
    )

    x_pixel = ((1 - xs) / 2.0) * 1024
    y_pixel = ((1 - ys) / 2.0) * 1024

    x_pixel_int = np.clip(x_pixel.cpu().numpy(), 0, 1024 - 1)
    y_pixel_int = np.clip(y_pixel.cpu().numpy(), 0, 1024 - 1)

    pix_x = x_pixel_int
    pix_y = y_pixel_int

    image = injection_object.render_mesh(
        angle=angle, T_z=3, T=T_new, R=R, aspect_ratio=aspect_ratio
    )
    # image = (image * 255).cpu().numpy().astype(np.uint8)

    image = (image * 255).cpu().numpy().astype(np.uint8)
    # image = np.transpose(image, (2, 0, 1))

    draw_pixels(
        image,
        bbox[:, 0].numpy().astype(np.int32),
        bbox[:, 1].numpy().astype(np.int32),
        square_size=5,
        paint_color=np.array([255, 0, 0]).astype(np.uint8),
    )

    draw_pixels(
        image,
        pix_y.astype(np.int32),
        pix_x.astype(np.int32),
        square_size=5,
        paint_color=np.array([0, 0, 255]).astype(np.uint8),
    )

    Image.fromarray(image).save(f"{path}/{name}.png")
    return image


def test_injection_diffrent_aspect_ratio(obj_filename, app_path):

    global IMAGE_SIZE

    path = f"{app_path}/data/test_injected/trainval/images/final_bbox"
    os.makedirs(path, exist_ok=True)
    debug_path = f"{app_path}/data/test_injected/trainval/images/final_aspect"
    os.makedirs(debug_path, exist_ok=True)

    T_0 = [0, 0, 0, 0]

    for angle in (0, -20, -45, -60):

        for i, bbox in enumerate(
            [
                torch.tensor([[444, 359], [444, 578], [393, 359], [393, 578]]),
                torch.tensor([[494, 359], [494, 578], [393, 359], [393, 578]]),
                torch.tensor(
                    [
                        [494 + 40, 359 + 20],
                        [494 + 40, 578 + 20],
                        [393 + 40, 359 + 20],
                        [393 + 40, 578 + 20],
                    ]
                ),
                torch.tensor(
                    [
                        [494 + 200, 359 - 150],
                        [494 + 200, 578 - 150],
                        [393 + 200, 359 - 150],
                        [393 + 200, 578 - 150],
                    ]
                ),
            ]
        ):

            injection_object = InjectedObject(obj_filename)

            bbox = injection_object.rotate_pixels(
                bbox.to(torch.float32), theta=torch.tensor([(np.pi * angle) / 180])
            )
            print(f"real angle - {angle}")
            injection_object(
                top_left=bbox[2],
                top_right=bbox[3],
                bottom_left=bbox[0],
                bottom_right=bbox[1],
                image_shape=[3, 1024, 1024],
                path=path,
                i=i,
            )

            # (R_new, T_new, extreme_pixels, aspect_ratio) = (
            #     injection_object.find_R_T_for_injection(
            #         bottom_left=bbox[0],
            #         bottom_right=bbox[1],
            #         top_left=bbox[2],
            #         top_right=bbox[3],
            #         image_shape=[3, IMAGE_SIZE, IMAGE_SIZE],
            #         return_aspect_ratio=True,
            #     )
            # )
            # if angle == 0 and (i == 0 or i == 1 or i == 2 or i == 3):
            #     T_0[i] = T_new

            # print(f"T_new is {T_new}")
            # print(F"T_0 is {T_0[i]}")
            # image = injection_object.render_mesh(
            # angle=angle, T_z=3, T=T_0[i], R=injection_object.base_R,
            # aspect_ratio=aspect_ratio
            # )
            # # image = (image * 255).cpu().numpy().astype(np.uint8)
            # print()
            # import torchvision

            # image = image.permute(2, 0, 1)
            # image = torchvision.transforms.functional.rotate(image, -angle, fill=1)
            # image = image.permute(1, 2, 0)
            # image = (image * 255).cpu().numpy().astype(np.uint8)
            # image = np.transpose(image, (2, 0, 1))

            # draw_pixels(
            #     image,
            #     bbox[:, 0].numpy().astype(np.int32),
            #     bbox[:, 1].numpy().astype(np.int32),
            #     square_size=5,
            #     paint_color=np.array([255, 0, 0]).astype(np.uint8),
            # )
            # Image.fromarray(image).save(f'{path}/bbox_{i}_angle{angle}_t0.png')
            # understand_aspect_ratio(injection_object, T_new, angle, aspect_ratio,
            # bbox, debug_path, f"bbox_{i}_angle{angle}_tnew")

        print(f"\tdone angle - {angle}, bbox - {i}")
        print()

    print()


# Close the figure to prevent it from displaying


if __name__ == "__main__":

    app_path = Path(__file__).parent.parent
    dir_path = f"{app_path}/mmrotate/3Ddata/meshes"
    os.makedirs(dir_path, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    # Set paths
    DATA_DIR = f"{app_path}/mmrotate/3Ddata/"
    obj_filename = os.path.join(DATA_DIR, "meshes/Container.obj")

    # Load obj file
    mesh = load_objs_as_meshes([obj_filename], device=device)
    verts = mesh.verts_packed()  # Get the vertices of the mesh

    test_injection_sanity_check(obj_filename, app_path)
    # test_injection_diffrent_aspect_ratio(obj_filename, app_path)
