import os 
import re
import numpy as np
from PIL import Image
import random


def get_files(folder_path, prob):
    files = os.listdir(folder_path)

    pattern_images = re.compile(r'^-?\d+(\.\d+)?\.png$')
    matching_images = [file for file in files if re.match(pattern_images, file)]

    pattern_segs = re.compile(r'^-?\d+(\.\d+)?\_seg.png$')
    matching_segs = [file for file in files if re.match(pattern_segs, file)]

    sampled_images = []
    sampled_segs = []

    print(f"number of images fetched - {len(matching_images) * 2}")
    # Iterate through each path in the list
    for i in range(len(matching_images)):
        # With probability p, add the path to the sampled list
        if random.random() < prob:
            sampled_images.append(matching_images[i])
            sampled_segs.append(matching_segs[i])
    

    sampled_images, sampled_segs = [np.array(Image.open(f"{folder_path}/{im}"))[:, :, ::-1] for im in sampled_images], [np.array(Image.open(f"{folder_path}/{im}"))[:, :, None] / 255 for im in sampled_segs]
    return sampled_images, sampled_segs



def transform(base_path, image_file_name, injection_type, probs):

    folder_path = f"{base_path}/mid_reults/{image_file_name[:-4]}"
    if not os.path.exists(folder_path):
        return
    
    for prob in probs:
        sampled_images, sampled_segs = get_files(folder_path, prob)


    dota_np = results['img'].astype(np.float32)
    if injection_type == 'ycbcr':

        for i in range(len(sampled_images)):

            yuv_sim_img = np.array(Image.fromarray(sampled_images[i]).convert('YCbCr'))
            yuv_origin = np.array(
                Image.fromarray((sampled_segs[i] * dota_np).astype(np.uint8)).convert('YCbCr')
            )

            new_obj = np.concatenate(
                [
                    yuv_origin[:, :, 0][:, :, None],
                    yuv_sim_img[:, :, 1][:, :, None],
                    yuv_sim_img[:, :, 2][:, :, None],
                ],
                axis=2,
            ).astype(np.uint8)

            new_obj_im = np.array(Image.fromarray(new_obj, 'YCbCr'))[:, :, ::-1]
            dota_np = (1 - sampled_segs[i]) * dota_np + sampled_segs[i] * new_obj_im

    elif injection_type == "simple":

        for i in range(len(sampled_images)):
            dota_np = (1 - sampled_segs[i]) * dota_np + sampled_segs[i] * sampled_images[i]

    
    return dota_np.astype(np.uint8)



if __name__ == '__main__':

    method = 'ycbcr'
    probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    paths = [f'/app/data/split_ss_dota/{p}' for p in probs]
    for pth in paths:
        os.makedirs(pth, exist_ok=True)
        os.makedirs(f"{pth}/annfiles", exist_ok=True)
        os.makedirs(f"{pth}/images", exist_ok=True)


    images_path = '/app/data/split_ss_dota/train/images'

    files = os.listdir(images_path)

    images_files = [file for file in files if file.endswith('.png')]

    for image in images_files:

        