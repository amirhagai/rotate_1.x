import os

# Path to the folder containing the files
folder_path = '/app/data/test_injected/trainval/annfiles/'

# Process each file in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith(".txt"):  # assuming the files are .txt format
        file_path = os.path.join(folder_path, file_name)
        
        with open(file_path, 'r') as file:
            print(file_path, end="\n\n\n")
            for line in file:
                parts = line.strip().split()
                x1, y1, x2, y2, x3, y3, x4, y4 = map(float, parts[:8])
                category, difficult = parts[8], parts[9]

                # Assuming the order is top-left, top-right, bottom-right, bottom-left
                top_left = (x1, y1)
                top_right = (x2, y2)
                bottom_right = (x3, y3)
                bottom_left = (x4, y4)

                # Now you have the bounding box in the desired format
                # Here, you can process these as needed, for example, print them
                print(f"\tCategory: {category}, Difficulty: {difficult}")
                print(f"\tTop-left: {top_left}, Top-right: {top_right}, Bottom-right: {bottom_right}, Bottom-left: {bottom_left}", end="\n\n")
