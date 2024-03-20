from parse_dota_file import parse_one_file
from infer_camera_parameters import InjectedObject
import torch
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2
import os

folder_path = "/app/data/test_injected/trainval/annfiles/"
file_name = "P0005__1024__0___0.txt"

bboxes = parse_one_file(folder_path=folder_path, file_name=file_name)

app_path = Path(__file__).parent.parent
DATA_DIR = f"{app_path}/mmrotate/3Ddata/"
obj_filename = os.path.join(DATA_DIR, "meshes/Container.obj")

injection = InjectedObject(obj_filename)

images_path = "/app/data/test_injected/trainval/images/finals"
os.makedirs(images_path, exist_ok=True)

gif_images_path = f"{images_path}/gif_images"
os.makedirs(gif_images_path, exist_ok=True)


debug_path = f"{images_path}/debug"
os.makedirs(debug_path, exist_ok=True)

images = []
segs = []
print(len(bboxes))


def create_gif(image_folder, int_comp=False):

    # Directory containing images
    # image_folder = f'{app_path}/data/test_injected/trainval/images/final_bbox'
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]

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
    img_plot = ax.imshow(first_image, aspect="equal")

    # Function to update the figure with a new image
    def update(frame_id):
        img = Image.open(os.path.join(image_folder, images[frame_id]))
        img_plot.set_data(img)
        return [img_plot]

    # Create an animation
    ani = animation.FuncAnimation(fig, update, frames=len(images), blit=True)

    # Save the animation
    ani.save(f"{image_folder}/movie.gif", writer="pillow", fps=2)

    plt.close(fig)


# create_gif(image_folder=images_path)


start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()


def draw_pixels(image, y_pixel_int, x_pixel_int, square_size, paint_colors):
    pixels_to_highlight = np.stack((y_pixel_int, x_pixel_int), axis=1)
    for i, (y, x) in enumerate(pixels_to_highlight):
        # Ensure the square stays within image bounds
        x_start = max(0, x - square_size // 2)
        y_start = max(0, y - square_size // 2)
        x_end = min(image.shape[1], x + square_size // 2 + 1)
        y_end = min(image.shape[0], y + square_size // 2 + 1)
        image[y_start:y_end, x_start:x_end] = paint_colors[i]


def sort_bbox(bbox):
    bbox = np.array(bbox)
    sorted_by_y = bbox[bbox[:, 0].argsort()]
    bottom_points = sorted_by_y[:2]
    bottom_points = bottom_points[bottom_points[:, 1].argsort()]
    top_points = sorted_by_y[2:]
    top_points = top_points[top_points[:, 1].argsort()]
    sorted_bbox = np.vstack((top_points, bottom_points))
    return sorted_bbox


def get_pixels_in_oriented_bbox(corners, image_shape):
    """
    Get all pixels inside an oriented bounding box using OpenCV.

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


# create_gif(gif_images_path, int_comp=True)


jackards = []
masks = []
images = []
for i in tqdm(range(len(bboxes))):

    bbox = torch.tensor(sort_bbox(bboxes[i])).to(torch.float32)
    corners = bbox.detach().cpu().numpy()
    mask = get_pixels_in_oriented_bbox(corners, [1024, 1024])
    pixels_inside_obb = np.argwhere(mask == 1)

    image, segmantation_mask = injection(
        bottom_left=bbox[0],
        bottom_right=bbox[1],
        top_left=bbox[2],
        top_right=bbox[3],
        image_shape=[3, 1024, 1024],
        path=images_path,
        i=0,
    )

    pixels_inside_segmantation = np.argwhere(segmantation_mask[:, :, 0] == 1)

    set_1 = set(map(tuple, pixels_inside_obb))
    set_2 = set(map(tuple, pixels_inside_segmantation))

    # Calculate intersection and union
    intersection = set_1.intersection(set_2)
    union = set_1.union(set_2)

    # Calculate Jaccard index
    jaccard_index = len(intersection) / len(union)

    jackards.append(jaccard_index)
    images.append(image)
    masks.append(mask)
    segs.append(segmantation_mask)

    segmantation_mask = segmantation_mask[:, :, 0]

    # masks_im = np.hstack([mask * 255, segmantation_mask * 255,
    # np.abs(segmantation_mask - mask) * 255]).astype(np.uint8)

    # Image.fromarray(masks_im).save(f"{debug_path}/{i}.png")

    # draw_pixels(
    #             im,
    #             bbox[:, 1].numpy().astype(np.int32),
    #             bbox[:, 0].numpy().astype(np.int32),
    #             square_size=5,
    #             paint_colors=colors,
    #         )

# Image.fromarray(im).save(f"{images_path}/test_bbox_oriantation.png")

dota_np = np.array(
    Image.open("/app/data/test_injected/trainval/images/P0005__1024__0___0.png")
)
Image.fromarray(dota_np).save(f"{gif_images_path}/0.png")

jackards = np.array(jackards)
sorted_jackards_indecis = np.argsort(jackards)
# print(jackards[sorted_jackards_indecis])
# for j, i in enumerate(sorted_jackards_indecis[::-1]):
#     dota_np = (1 - segs[i]) * dota_np + segs[i] * images[i]
#     Image.fromarray(dota_np).save(f"{gif_images_path }/{j + 1}.png")


create_gif(gif_images_path, int_comp=True)
end.record()

dota_np = np.array(
    Image.open("/app/data/test_injected/trainval/images/P0005__1024__0___0.png")
)

for i in range(len(masks)):
    dota_np = (1 - masks[i][:, :, None]) * dota_np + masks[i][:, :, None] * (
        masks[i][:, :, None] * 255
    )

Image.fromarray(dota_np).save(f"{gif_images_path}/basic.png")

# Waits for everything to finish running
torch.cuda.synchronize()


print(f"elapsed time in seconds - {start.elapsed_time(end) / 1000}")
