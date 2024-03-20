import os

# Path to the folder containing the files


# Process each file in the folder
# for file_name in os.listdir(folder_path):
#   if file_name.endswith(".txt"):  # assuming the files are .txt format"


def parse_one_file(folder_path, file_name):

    bboxes = []

    file_path = os.path.join(folder_path, file_name)
    with open(file_path, 'r') as file:
        print(file_path, end='\n\n\n')
        for line in file:
            parts = line.strip().split()
            x1, y1, x2, y2, x3, y3, x4, y4 = map(float, parts[:8])
            category, difficult = parts[8], parts[9]
            if category == 'large-vehicle':

                # Assuming the order is top-left,
                # top-right, bottom-right, bottom-left
                top_left = [y1, x1]
                top_right = [y2, x2]
                bottom_right = [y3, x3]
                bottom_left = [y4, x4]

                bbox = [bottom_left, bottom_right, top_left, top_right]
                bboxes.append(bbox)
    return bboxes
