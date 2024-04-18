from parse_dota_file import parse_one_file
from infer_camera_parameters import InjectedObject
import torch
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2
import os

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
    'random_colors',
    help='do you want to use random colors?',
    type=str_to_boll,
    default='1',
)
parser.add_argument(
    'random_matirels',
    help='do you want to use random matirels?',
    type=str_to_boll,
    default='1',
)
parser.add_argument(
    'random_shininess',
    help='do you want to use random shininess',
    type=str_to_boll,
    default='1',
)
parser.add_argument(
    'save_mid_restuls',
    help='save the midean results',
    type=str_to_boll,
    default='1',
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
    gif_images_path,
    obj_filename,
    annotation_folder_path,
    annotation_file_name,
    category='large-vehicle',
):
    global args

    # x = torch.rand(5000, 3)
    image_name = Path(image_path).name

    injection_ycbcr_path = Path(gif_images_path).parent / 'ycbcr'
    os.makedirs(injection_ycbcr_path, exist_ok=True)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()

    bboxes = parse_one_file(
        folder_path=annotation_folder_path,
        file_name=annotation_file_name,
        category=category,
    )

    dota_np = np.array(Image.open(f'{image_path}'))
    if len(bboxes) == 0:
        # Image.fromarray(dota_np).save(f"{injection_ycbcr_path}/{image_name}")
        # Image.fromarray(dota_np).save(f"{gif_images_path}/{image_name}")
        return

    os.makedirs(Path(gif_images_path).parent / 'mid_reults', exist_ok=True)
    path_for_mid_results = (
        Path(gif_images_path).parent / 'mid_reults' / Path(image_name).stem
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
        corners = bbox.detach().cpu().numpy()

        image, segmantation_mask = injection(
            bbox=bboxes[i],
            image_shape=[3, 1024, 1024],
            random_colors=args.random_colors,
            random_materials=args.random_matirels,
            random_shininess=args.random_shininess,
        )
        if args.save_mid_restuls is True:
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

    Image.fromarray(dota_np).save(f'{gif_images_path}/{image_name}')

    # create_gif(gif_images_path, int_comp=True)
    # dota_np = np.array(Image.open(f"{image_path}"))

    # for i in range(len(masks)):
    #     dota_np = (1 - masks[i][:, :, None]) * dota_np + masks[i][:, :, None] * (
    #         masks[i][:, :, None] * 255
    #     )
    # Image.fromarray(dota_np).save(f'{gif_images_path}/basic.png')
    print('done')
    end.record()

    # Waits for everything to finish running
    torch.cuda.synchronize()

    print(f'elapsed_time - {start.elapsed_time(end) / 1000}')


def process_image(
    annotations_folder,
    annotation_file,
    images_folder,
    gif_images_path,
    obj_filename,
    category='large-vehicle',
):
    # Extract the base file name without extension to match the image file
    base_name = os.path.splitext(annotation_file)[0]
    image_file = os.path.join(images_folder, f'{base_name}.png')
    os.makedirs(gif_images_path, exist_ok=True)

    # Check if the corresponding image file exists
    if os.path.exists(image_file):
        parse_one_image(
            image_path=image_file,
            gif_images_path=gif_images_path,
            obj_filename=obj_filename,
            annotation_folder_path=annotations_folder,
            annotation_file_name=f'{Path(annotation_file).stem}.txt',
            category=category,
        )
    torch.cuda.empty_cache()


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


if __name__ == '__main__':

    print('start')

    # file_name = 'P0005__1024__0___0.txt'

    app_path = Path(__file__).parent.parent
    DATA_DIR = f'{app_path}/mmrotate/3Ddata/'
    obj_filename = os.path.join(
        DATA_DIR, 'meshes/TruckCGTrader/Truck_final.obj'
    )
    # obj_filename = os.path.join(DATA_DIR, 'meshes/Container/Container.obj')

    images_path = '/app/data/test_injected/finals'
    os.makedirs(images_path, exist_ok=True)

    gif_images_path = '/app/data/split_ss_dota/train_injected/images'
    os.makedirs(gif_images_path, exist_ok=True)

    debug_path = f'{images_path}/debug'
    os.makedirs(debug_path, exist_ok=True)

    annotations_folder = '/app/data/split_ss_dota/train/annfiles/'
    images_folder = '/app/data/split_ss_dota/train/images'

    allready_done_images = set(
        os.listdir('/app/data/split_ss_dota/train_injected/images')
    )

    for filename in os.listdir(images_folder):

        # if filename in allready_done_images:
        # continue

        # print(f"working on- {filename}")
        # if "P2710__1024__941___824" not in filename:
        # continue
        process_image(
            annotations_folder,
            filename,
            images_folder,
            gif_images_path,
            obj_filename,
        )

        print('done')

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
