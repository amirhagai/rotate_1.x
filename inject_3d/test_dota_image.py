from parse_dota_file import parse_one_file, get_boxes, get_raw_bboxes
from infer_camera_parameters import InjectedObject
import torch
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2
import os
from pytorch3d.io import load_objs_as_meshes
import random

# import torch.multiprocessing as mp
from tqdm import tqdm
import argparse


def str_to_boll(boll_str):
    if boll_str == '0':
        return False
    elif boll_str == '1':
        return True
    else:
        raise Exception('random should be 0 for False or 1 for True')


parser = argparse.ArgumentParser()
parser.add_argument(
    '--random_colors',
    help='do you want to use random colors?',
    type=str_to_boll,
    default='1',
)

parser.add_argument(
    '--color_option',
    help='do you want to use random colors?',
    type=int,
    default=1,
)


parser.add_argument(
    '--random_materials',
    help='do you want to use random matirels?',
    type=str_to_boll,
    default='0',
)
parser.add_argument(
    '--random_shininess',
    help='do you want to use random shininess',
    type=str_to_boll,
    default='0',
)
parser.add_argument(
    '--save_median_restuls',
    help='save the midean results',
    type=str_to_boll,
    default='0',
)

parser.add_argument(
    '--save_ycbcr',
    help='save the midean results',
    type=str_to_boll,
    default='0',
)

args = parser.parse_args()

torch.set_printoptions(sci_mode=False)


def create_gif(image_folder, int_comp=False):

    # Directory containing images
    # image_folder = f'{app_path}/data/test_injected/trainval/images/final_bbox'
    images = [img for img in os.listdir(image_folder) if img.endswith('.png')]

    if int_comp:

        def key_func(x):
            return int(x[:-4])

        images.sort(key=key_func)  # Sort the images by name
    else:
        images.sort()

    # Load the first image to get the size
    first_image = Image.open(os.path.join(image_folder, images[0]))

    # Create a figure and axis to display the images
    fig, ax = plt.subplots(figsize=(10, 8))
    img_plot = ax.imshow(first_image, aspect='equal')

    # Function to update the figure with a new image
    def update(frame_id):
        img = Image.open(os.path.join(image_folder, images[frame_id]))
        img_plot.set_data(img)
        return [img_plot]

    # Create an animation
    ani = animation.FuncAnimation(fig, update, frames=len(images), blit=True)

    # Save the animation
    ani.save(f'{image_folder}/movie.gif', writer='pillow', fps=2)
    print(f'\n\ngif located at - {image_folder}/movie.gif', end='\n\n')

    plt.close(fig)


# create_gif(image_folder=images_path)


def draw_pixels(image, y_pixel_int, x_pixel_int, square_size, paint_colors):
    pixels_to_highlight = np.stack((y_pixel_int, x_pixel_int), axis=1)
    for i, (y, x) in enumerate(pixels_to_highlight):
        # Ensure the square stays within image bounds
        x_start = max(0, x - square_size // 2)
        y_start = max(0, y - square_size // 2)
        x_end = min(image.shape[1], x + square_size // 2 + 1)
        y_end = min(image.shape[0], y + square_size // 2 + 1)
        image[y_start:y_end, x_start:x_end] = paint_colors[i]


def get_pixels_in_oriented_bbox(corners, image_shape):
    """Get all pixels inside an oriented bounding box using OpenCV.

    :param corners: Four corners of the OBB as a list of (x, y) tuples.
    :param image_shape: Shape of the image or matrix (height, width).
    :return: A binary mask with the same dimensions as the input image,
    where pixels inside the OBB are set to 1 (True) and others are 0 (False).
    """
    # Create an empty mask
    mask = np.zeros(image_shape, dtype=np.uint8)
    converted_corners = [(x, y) for y, x in corners]

    # Rearrange to: bottom-left, top-left, top-right, bottom-right (clockwise order)
    reordered_corners = [
        converted_corners[0],
        converted_corners[2],
        converted_corners[3],
        converted_corners[1],
    ]

    # Fill the polygon defined by the OBB corners
    cv2.fillPoly(mask, pts=[np.array(reordered_corners, np.int32)], color=(1))

    return mask


def get_jaccard_ind(segmantation_mask, corners, image_shape):

    mask = get_pixels_in_oriented_bbox(corners, image_shape)
    pixels_inside_obb = np.argwhere(mask == 1)
    pixels_inside_segmantation = np.argwhere(segmantation_mask[:, :, 0] == 1)
    set_1 = set(map(tuple, pixels_inside_obb))
    set_2 = set(map(tuple, pixels_inside_segmantation))

    # Calculate intersection and union
    intersection = set_1.intersection(set_2)
    union = set_1.union(set_2)

    # Calculate Jaccard index
    jaccard_index = len(intersection) / len(union)
    return jaccard_index, mask


def parse_one_image(
    image_path,
    saving_path,
    obj_filename,
    annotation_folder_path,
    annotation_file_name,
    category='large-vehicle',
    return_T=False
):
    global args

    # x = torch.rand(5000, 3)
    image_name = Path(image_path).name

    if args.save_ycbcr:
        injection_ycbcr_path = Path(saving_path).parent / 'ycbcr'
        os.makedirs(injection_ycbcr_path, exist_ok=True)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()

    bboxes = parse_one_file(
        folder_path=annotation_folder_path,
        file_name=annotation_file_name,
        category=category,
    )

    # bboxes = get_boxes(
    #             folder_path=annotation_folder_path,
    #             file_name=annotation_file_name,
    #             category='small-vehicle',
    #             unwanted_category='large-vehicle'
    #         )
    dota_np = np.array(Image.open(f'{image_path}'))
    if len(bboxes) == 0:
        # Image.fromarray(dota_np).save(f"{injection_ycbcr_path}/{image_name}")
        # Image.fromarray(dota_np).save(f"{gif_images_path}/{image_name}")
        return dota_np

    os.makedirs(Path(saving_path).parent / 'mid_reults', exist_ok=True)
    
    if args.save_median_restuls is True:
        path_for_mid_results = (
            Path(saving_path).parent / 'mid_reults' / Path(image_name).stem
        )
        os.makedirs(path_for_mid_results, exist_ok=True)

    print(f'number of bboxes - {len(bboxes)}')
    injection = InjectedObject(obj_filename, device='cuda:0')
    torch.cuda.set_device(injection.device)

    jackards = []
    masks = []
    images = []
    segs = []

    for i in tqdm(range(len(bboxes))):

        bbox = torch.tensor(bboxes[i]).to(torch.float32)
        if bbox.min() < 0:
            continue
        corners = bbox.detach().cpu().numpy()

        image, segmantation_mask = injection(
            bbox=bboxes[i],
            image_shape=[3, 1024, 1024],
            random_colors=args.random_colors,
            random_materials=args.random_materials,
            random_shininess=args.random_shininess,
            color_option=args.color_option
        )
        if args.save_median_restuls is True:
            mid = image.copy()
            Image.fromarray(mid).save(f'{path_for_mid_results}/{i}.png')

            draw_pixels(
                mid,
                bbox[:, 0].cpu().numpy().astype(np.int32),
                bbox[:, 1].cpu().numpy().astype(np.int32),
                5,
                np.array([[255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0]]),
            )
            Image.fromarray(mid).save(
                f'{path_for_mid_results}/{i}_with bbox.png'
            )
            Image.fromarray(segmantation_mask[:, :, 0] * 255).save(
                f'{path_for_mid_results}/{i}_seg.png'
            )

        jaccard_index, mask = get_jaccard_ind(
            segmantation_mask, corners, [1024, 1024]
        )

        jackards.append(jaccard_index)
        images.append(image)
        masks.append(mask)
        segs.append(segmantation_mask)

        segmantation_mask = segmantation_mask[:, :, 0]

    jackards = np.array(jackards)

    sorted_jackards_indecis = np.argsort(jackards)

    if args.save_ycbcr:
        for j, i in enumerate(sorted_jackards_indecis[::-1]):
            # if jackards[i] < 0.6:
            # continue
            yuv_img = np.array(Image.fromarray(images[i]).convert('YCbCr'))
            yuv_origin = np.array(
                Image.fromarray((segs[i]) * dota_np).convert('YCbCr')
            )
            new_obj = np.concatenate(
                [
                    yuv_origin[:, :, 0][:, :, None],
                    yuv_img[:, :, 1][:, :, None],
                    yuv_img[:, :, 2][:, :, None],
                ],
                axis=2,
            ).astype(np.uint8)
            new_obj_im = Image.fromarray(new_obj, 'YCbCr').convert('RGB')
            # dota_np = (1 - segs[i]) * dota_np + segs[i] * images[i]
            dota_np = (1 - segs[i]) * dota_np + segs[i] * new_obj_im

        Image.fromarray(dota_np).save(f'{injection_ycbcr_path}/{image_name}')

    dota_np = np.array(Image.open(f'{image_path}'))
    for j, i in enumerate(sorted_jackards_indecis[::-1]):
        # if jackards[i] < 0.6:
        # continue
        dota_np = (1 - segs[i]) * dota_np + segs[i] * images[i]
        # Image.fromarray(dota_np).save(f"{gif_images_path}/{i}_{image_name}")
        
    Image.fromarray(dota_np).save(f'{saving_path}/{image_name}')
    
    
    # anns = get_raw_bboxes(
    #     folder_path=annotation_folder_path,
    #     file_name=annotation_file_name,
    #     category=category,)
    # d = dota_np.copy()
    # cv2.drawContours(d, np.array(anns).astype(np.int32).reshape(-1, 4, 1, 2), -1, (255, 0, 0), thickness=2)
    # im = np.hstack([dota_np, d])
    # plt.imshow(k)
    # plt.savefig("/app/data/90.png")

    # create_gif(gif_images_path, int_comp=True)
    # dota_np = np.array(Image.open(f"{image_path}"))

    # for i in range(len(masks)):
    #     dota_np = (1 - masks[i][:, :, None]) * dota_np + masks[i][:, :, None] * (
    #         masks[i][:, :, None] * 255
    #     )
    # Image.fromarray(dota_np).save(f'{gif_images_path}/basic.png')
    print('done one image')
    end.record()

    # Waits for everything to finish running
    torch.cuda.synchronize()

    print(f'elapsed_time - {start.elapsed_time(end) / 1000}\n\n\n')
    
    return dota_np


def process_image(
    annotations_folder,
    annotation_file,
    images_folder,
    saving_path,
    obj_filename,
    category='large-vehicle',
):
    # Extract the base file name without extension to match the image file
    base_name = os.path.splitext(annotation_file)[0]
    image_file = os.path.join(images_folder, f'{base_name}.png')
    os.makedirs(saving_path, exist_ok=True)

    # Check if the corresponding image file exists
    if os.path.exists(image_file):
        im = parse_one_image(
            image_path=image_file,
            saving_path=saving_path,
            obj_filename=obj_filename,
            annotation_folder_path=annotations_folder,
            annotation_file_name=f'{Path(annotation_file).stem}.txt',
            category=category,
        )
    torch.cuda.empty_cache()
    return im


def worker_init():
    # Explicitly create a new CUDA context
    if torch.cuda.is_available():
        torch.cuda.init()


def process_image_worker(data):
    (
        annotations_folder,
        annotation_file,
        images_folder,
        gif_images_path,
        obj_filename,
        category,
        gpu_id,
    ) = data
    # Set the current process to use the specific GPU
    torch.cuda.set_device(gpu_id)
    process_image(
        annotations_folder,
        annotation_file,
        images_folder,
        gif_images_path,
        obj_filename,
        category,
    )



def find_bbox_properties(image):
    # Read the image
    

    # Find contours
    
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Assume the largest contour is the one encompassing the bounding box
    contour = max(contours, key=cv2.contourArea)
    
    # Compute the minimum area rectangle
    rect = cv2.minAreaRect(contour)
    center, size, angle = rect
    
    # Get the corners of the rectangle
    box = cv2.boxPoints(rect)
    box = np.int0(box)  # Convert to integer
    
    # color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # cv2.drawContours(color_image, [box], -1, (1, 0, 0), thickness=-1)
    # x = np.hstack([color_image[:, :, 0], image[:, :, 0], np.abs(color_image[:, :, 0].astype(np.float32) - image[:, :, 0].astype(np.float32)).astype(np.uint8)])
    
    return center, size, angle, box


def rotate_image(image, angle, center=None):
    height, width = image.shape[:2]
    if center is None:
        center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine((image.cpu().numpy() * 255).astype(np.uint8), rotation_matrix, (width, height), borderValue=(0,0,0))
    return rotated_image

def save_annotation_file(annotations_folder, annotation_file_name, saved_data_path_annotatioins, base_name, lines=None, class_to_replace='', class_to_add=''):
    with open(f'{annotations_folder}/{annotation_file_name}', 'r') as file:
        content = file.readlines()
        file.close()
    
    if content is not None:    
        for i in range(len(content)):
            content[i] = content[i].replace(class_to_replace, class_to_add)
        
    if lines is not None:
        for i in range(len(lines)):
            lines[i] = lines[i].replace(class_to_replace, class_to_add)
        
    with open(f'{saved_data_path_annotatioins}/{base_name}.txt', 'w') as file:
        file.writelines(content)  # Write the original content
        if lines is not None:
            file.writelines(lines)  # Append a new line
        file.flush()
        file.close()
        # file.write('New line 2\n')    # Append another new line    
    

def inject_random_location():
    global args
    print('start')

    app_path = Path(__file__).parent.parent
    DATA_DIR = f'{app_path}/mmrotate/3Ddata/'

    obj_filename = os.path.join(DATA_DIR, 'meshes/Container/Container.obj')
    
    saved_data_path_images = '/app/data/split_ss_dota/added_container/images'
    saved_data_path_annotatioins = '/app/data/split_ss_dota/added_container/annfiles'

    os.makedirs(saved_data_path_images, exist_ok=True)
    os.makedirs(saved_data_path_annotatioins, exist_ok=True)
    # images_path = '/app/data/test_injected/finals'
    # os.makedirs(images_path, exist_ok=True)

    # gif_images_path = '/app/data/split_ss_dota/train_injected_container/images'
    # os.makedirs(gif_images_path, exist_ok=True)

    annotations_folder = '/app/data/split_ss_dota/train/annfiles/'
    images_folder = '/app/data/split_ss_dota/train/images'


    for filename in os.listdir(images_folder):


        image_name = filename
        category = 'small-vehicle'
        unwanted_category = 'large-vehicle'
        
        base_name = os.path.splitext(image_name)[0]
        image_file = os.path.join(images_folder, f'{base_name}.png')

        # Check if the corresponding image file exists
        if os.path.exists(f'{saved_data_path_images}/{base_name}.png'):
            continue
        
        if os.path.exists(image_file):
            image_path=image_file
            obj_filename=obj_filename
            annotation_folder_path=annotations_folder
            annotation_file_name=f'{Path(image_name).stem}.txt'
            category=category

            
            # x = torch.rand(5000, 3)
            image_name = Path(image_path).name

            bboxes = get_boxes(
                folder_path=annotation_folder_path,
                file_name=annotation_file_name,
                category=category,
            )

            dota_np = np.array(Image.open(f'{image_path}'))
            if len(bboxes) == 0:
                Image.fromarray(dota_np).save(f'{saved_data_path_images}/{base_name}.png')
                save_annotation_file(annotations_folder, annotation_file_name, saved_data_path_annotatioins,base_name, lines=None)
                continue


            print(f'number of bboxes - {len(bboxes)}')
            injection = InjectedObject(obj_filename, device='cuda:0')
            torch.cuda.set_device(injection.device)

            jackards = []
            masks = []
            images = []
            segs = []
            lines = []
            boxes = []

            Ts = []
            angles = []
            for i in tqdm(range(len(bboxes))):
                
                bbox = torch.tensor(bboxes[i]).to(torch.float32)
                corners = bbox.detach().cpu().numpy()
                image_shape = [3, 1024, 1024]


                bbox, cloned_origin, x, y, w, h, angle, bbox_center, dx, dy = \
                    injection.param_update(bbox, image_shape)
                if bbox is None:
                    continue

                R, T, extreme_pixels, aspect_ratio = injection.find_R_T_for_injection(
                    top_left=bbox[2],
                    top_right=bbox[3],
                    bottom_left=bbox[0],
                    bottom_right=bbox[1],
                    image_shape=image_shape,
                    return_aspect_ratio=True,
                )

                # image = injection.render_mesh(
                #     angle=angle,
                #     T_z=3,
                #     T=T,
                #     R=injection.base_R,
                #     aspect_ratio=aspect_ratio,
                #     random_colors=args.random_colors,
                #     random_materials=args.random_materials,
                #     random_shininess=args.random_shininess,
                # )

                # image_test = image.clone()
                # if image.sum() == 0:
                #     print(f"\n\n\nFATEL!!!!\n\n\n")
                #     continue
                # segmantation_mask = (
                #     (image[:, :, 0] != 0).cpu().numpy().astype(np.uint8)[:, :, None]
                # )
            
                # center, _, _, _ = find_bbox_properties(segmantation_mask)
                # bbox_center[0] = center[0]
                # bbox_center[1] = center[1]
                # nonzero_pixels, nonzero_indices, min_y, min_x, max_y, max_x, bbox_rotate, image, patch = injection.postprocess_image(image_test, torch.tensor([angle]), bbox_center, cloned_origin)
                
                # if image.sum() == 0:
                #     print(f"\n\n\nFATEL!!!! 2222222\n\n\n")
                #     continue
                # segmantation_mask = (
                #     (image[:, :, 0] != 0).cpu().numpy().astype(np.uint8)[:, :, None]
                # )
                
                # center, size, angle, box = find_bbox_properties(segmantation_mask)

                # bbox_new = np.empty(box.shape)
                # bbox_new[0] = box[3][::-1]
                # bbox_new[1] = box[2][::-1]
                # bbox_new[2] = box[0][::-1]
                # bbox_new[3] = box[1][::-1] # current order, writing order is like bbox
                
                if injection.obj_name == 'Container':
                    if dy > dx:
                        injection.natural_aspect_ratio = 1 / injection.natural_aspect_ratio
                        injection.mesh = load_objs_as_meshes(
                            [injection.obj_file_path], device=injection.device
                        )
                        injection.verts = (
                            injection.mesh.verts_packed()
                        ) 
            
                Ts.append(T)
                angles.append(angle)
                
                        
                # print(f"bbox is - {bbox}")
                # print(f"T is {T}")
                
                # print(f"image shape - {image_shape}")
                # print(f"angle is - {angle}")
                # print(f"baseR - {injection.base_R}")
                # print("\n\n\n")
                
            if len(Ts) == 0:
                Image.fromarray(dota_np).save(f'{saved_data_path_images}/{base_name}.png')
                save_annotation_file(annotations_folder, annotation_file_name, saved_data_path_annotatioins,base_name, lines=None)
                continue

                
            Ts = np.array(Ts)
            avg_T = Ts.mean(axis=(0, 1))
            Tx_max, Tx_min = Ts[:, :, 0].max(), Ts[:, :, 0].min()
            Ty_max, Ty_min = Ts[:, :, 1].max(), Ts[:, :, 1].min()
            
            
            angles = np.array(angles) 
            angle_min, angle_max = angles.min(), angles.max()
            
            min_value = -5
            max_value = 5
            times =  len(Ts) + random.randint(min_value, max_value)
            times = times if times > 0 else random.randint(-3, 3)

            for _ in range(times):
                
                # if injection.obj_name == 'Container':
                #     if dy is None or dy > dx:
                #         injection.natural_aspect_ratio = 1 / injection.natural_aspect_ratio
                #         injection.mesh = load_objs_as_meshes(
                #             [injection.obj_file_path], device=injection.device
                #         )
                #         injection.verts = (
                #             injection.mesh.verts_packed()
                #         ) 
                        
                        
                size = random.uniform(1.1, 2)
                angle = random.uniform(angle_min, angle_max)
                
                Tx = random.uniform(Tx_min, Tx_max)
                Ty = random.uniform(Ty_min, Ty_max)
                Tz = size * avg_T[2]
                # print(f'Tx- {Tx}, Ty - {Ty}, Tz - {Tz}, angle - {angle}')
                T = np.array([Tx, Ty, Tz])[None, :]
                image = injection.render_mesh(
                    angle=angle,
                    T_z=3,
                    T=T,
                    R=injection.base_R,
                    aspect_ratio=aspect_ratio,
                    random_colors=args.random_colors,
                    random_materials=args.random_materials,
                    random_shininess=args.random_shininess,
                )
                image_test = image.clone()
                if image.sum() == 0:
                    print(f"\n\n\nFATEL!!!!\n\n\n")
                    continue
                segmantation_mask = (
                    (image[:, :, 0] != 0).cpu().numpy().astype(np.uint8)[:, :, None]
                )
                
                center, _, _, b = find_bbox_properties(segmantation_mask)
                # bbox_center[0] = center[0]
                # bbox_center[1] = center[1]
                # f, ax = plt.subplots(1, 6)
                # ax[0].imshow(image.cpu().numpy())
                
                # nonzero_pixels, nonzero_indices, min_y, min_x, max_y, max_x, bbox_rotate, image, patch = injection.postprocess_image(image_test, torch.tensor([angle]), bbox_center, cloned_origin)
                _, _, _, _, _, _, _, _, dx, dy = injection.param_update(b.astype(np.float32), image_shape)
                if dy > dx :
                    image_rotataed = rotate_image(image, -angle, center=center)
                else:
                    image_rotataed = rotate_image(image, 90+angle, center=center)
                
                image = torch.tensor(image_rotataed.astype(np.float32) / 255)
                # ax[1].imshow(image.cpu().numpy())
                if image.sum() == 0:
                    print(f"\n\n\nFATEL!!!! 2222222\n\n\n")
                    continue
                segmantation_mask = (
                    (image[:, :, 0] != 0).cpu().numpy().astype(np.uint8)[:, :, None]
                )
                
                center, size, angle_, box = find_bbox_properties(segmantation_mask)
                boxes.append(box.reshape((-1, 1, 2)))
                
                x1, y1, x2, y2, x3, y3, x4, y4 = box.reshape(-1)
                category = 'container'
                difficulty = '0'
                
                line = f"{float(x1)} {float(y1)} {float(x2)} {float(y2)} {float(x3)} {float(y3)} {float(x4)} {float(y4)} {category} {difficulty}\n"
                lines.append(line)
                
                
                # bbox_xyxy_no_angle = RotatedBoxes.rbox2corner(bbox)
                image = (image * 255).cpu().numpy().astype(np.uint8)
                images.append(image)
                masks.append(segmantation_mask)
                
                # bbox_new = np.empty(box.shape)
                # bbox_new[0] = box[3][::-1]
                # bbox_new[1] = box[2][::-1]
                # bbox_new[2] = box[0][::-1]
                # bbox_new[3] = box[1][::-1] # current order, writing order is like bbox
                
                # _, _, _, _, _, _, _, _, dx, dy = \
                # injection.param_update(box.astype(np.float32), image_shape)
                # if injection.obj_name == 'Container':
                #     if dy is None or dy > dx:
                #         injection.natural_aspect_ratio = 1 / injection.natural_aspect_ratio
                #         injection.mesh = load_objs_as_meshes(
                #             [injection.obj_file_path], device=injection.device
                #         )
                #         injection.verts = (
                #             injection.mesh.verts_packed()
                #         ) 
            if times > 0:
                # original_annfile = open(f'/app/data/split_ss_dota/train/annfiles/{base_name}.txt')
                print("done creation")
                images = np.sum(np.array(images), axis=0).astype(np.float32)
                masks = np.sum(np.array(masks), axis=0).astype(np.float32)
                masks[masks > 0] = 1
                # added_im = (images * masks)
                # added = (images * masks).astype(np.uint8)
                # cv2.drawContours(added, boxes, -1, (255, 0, 0), 2)
                dota_np = (dota_np.astype(np.float32) * (1 - masks) + (images * masks)).astype(np.uint8)
            else:
                lines=None
                
            Image.fromarray(dota_np).save(f'{saved_data_path_images}/{base_name}.png')
            
            save_annotation_file(annotations_folder, 
                                 annotation_file_name, 
                                saved_data_path_annotatioins,
                                base_name,
                                lines=lines,)            
            
            # with open(f'{annotations_folder}/{annotation_file_name}', 'r') as file:
            #     content = file.readlines()
            #     file.close()
                
            # with open(f'{saved_data_path_annotatioins}/{base_name}.txt', 'w') as file:
            #     file.writelines(content)  # Write the original content
            #     file.write(lines)  # Append a new line
            #     file.flush()
            #     file.close()
                # file.write('New line 2\n')    # Append another new line

        torch.cuda.empty_cache()
        
        
        
def create_dataset():
    
    print('start')

    # file_name = 'P0005__1024__0___0.txt'

    app_path = Path(__file__).parent.parent
    DATA_DIR = f'{app_path}/mmrotate/3Ddata/'

    obj_filename = os.path.join(DATA_DIR, 'meshes/Container/Container.obj')


    saving_path_images = '/app/data/split_ss_dota/added_container/images'
    os.makedirs(saving_path_images, exist_ok=True)
    
    saving_path_ann = '/app/data/split_ss_dota/added_container/annfiles'
    os.makedirs(saving_path_ann, exist_ok=True)


    annotations_folder = '/app/data/split_ss_dota/train/annfiles/'
    images_folder = '/app/data/split_ss_dota/train/images'
    args.random_colors = True
    args.random_materials = True
    args.random_shininess = True

    for filename in os.listdir(images_folder):
        
        # if filename != 'P1412__1024__2472___0.png':
        #     continue
        # if os.path.exists(f"/app/data/split_ss_dota/added_container/images/{filename}"):
        #     continue

        print(f'start - {filename}')
        args.color_option = 1
        im = process_image(
            annotations_folder,
            filename,
            images_folder,
            saving_path_images,
            obj_filename,
        )
        Image.fromarray(im).save(f'{saving_path_images}/{Path(filename).name}')
        save_annotation_file(annotations_folder, 
                             f'{Path(filename).stem}.txt',
                             saving_path_ann,
                             f'{Path(filename).stem}', 
                             lines=None,
                             class_to_replace='large-vehicle', 
                            class_to_add='container'
                            )
        print(f'done - {filename}')

if __name__ == '__main__':

    create_dataset()
    # inject_random_location()
    # bs = 6

    # annotation_files = [
    #     f for f in os.listdir(annotations_folder) if f.endswith('.txt')
    # ]

    # start = torch.cuda.Event(enable_timing=True)
    # end = torch.cuda.Event(enable_timing=True)

    # start.record()

    # for ind in range(0, len(annotation_files), bs):

    #     annotation_files_batch = annotation_files[
    #         ind : min(ind + bs, len(annotation_files))
    #     ]

    #     # Get all annotation files from the folder
    #     category = 'large-vehicle'

    #     # Prepare the data with GPU assignments
    #     num_gpus = 2  # Number of GPUs available
    #     data = [
    #         (
    #             annotations_folder,
    #             annotation_file,
    #             images_folder,
    #             gif_images_path,
    #             obj_filename,
    #             category,
    #             i,
    #             i % num_gpus,
    #         )
    #         for i, annotation_file in enumerate(annotation_files_batch)
    #     ]

    #     # Set the start method to 'spawn'
    #     mp.set_start_method('spawn', force=True)
    #     print(f'cpu count - {mp.cpu_count()}')
    #     # Create a Pool of workers
    #     with mp.Pool(processes=mp.cpu_count()) as pool:
    #         pool.map(process_image_worker, data)

    #     pool.close()
    #     pool.join()
    #     torch.cuda.empty_cache()

    # end.record()

    # # Waits for everything to finish running
    # torch.cuda.synchronize()

    # print(f'elapsed_time - {start.elapsed_time(end) / 1000}')
