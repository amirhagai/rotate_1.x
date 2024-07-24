import os
from PIL import Image

# Specify the directory path where your PNG files are located
directory_path = "/app/data/split_ss_dota/trainval/images"

# List all files in the directory
file_list = os.listdir(directory_path)

# Filter the list to keep only the PNG files
png_files = [file for file in file_list if file.endswith(".png")]

# Iterate through the PNG files and get their dimensions
image_dimensions = set()
for png_file in png_files:
    file_path = os.path.join(directory_path, png_file)
    with Image.open(file_path) as img:
        width, height = img.size
    image_dimensions.add((width, height))

# Now, image_dimensions is a dictionary where keys are file names
# and values are tuples containing (width, height) in pixels
print(image_dimensions)
