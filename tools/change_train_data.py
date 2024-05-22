import os
import shutil


def swap_files(dir1, dir2):
    # List files in both directories
    files1 = set(os.listdir(dir1))
    files2 = set(os.listdir(dir2))

    # Find common files between two directories
    common_files = files1.intersection(files2)

    # Swap the files
    for file_name in common_files:

        file1_path = os.path.join(dir1, file_name)
        file2_path = os.path.join(dir2, file_name)
        shutil.copy(file1_path, file2_path)


# step0 ensure the original images are in
dir1 = "/app/data/split_ss_dota/train"
dir2 = "/app/data/split_ss_dota/train_modification/train/images"
# Swap the files
swap_files(dir1, dir2)


print(
    "finish step 0, /app/data/split_ss_dota/train_modification/train/images now contain the original images"
)

dir1 = "/app/data/split_ss_dota/train_injected/ycbcr"
dir2 = "/app/data/split_ss_dota/train_modification/train/images"

# Swap the files
swap_files(dir1, dir2)
