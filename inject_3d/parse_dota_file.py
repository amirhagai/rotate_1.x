import os

# Path to the folder containing the files


# Process each file in the folder
# for file_name in os.listdir(folder_path):
#   if file_name.endswith(".txt"):  # assuming the files are .txt format"


def parse_one_file(folder_path, file_name, category='large-vehicle'):

    bboxes = []

    file_path = os.path.join(folder_path, file_name)
    with open(file_path, 'r') as file:
        print(file_path, end='\n\n\n')
        for line in file:
            parts = line.strip().split()
            x1, y1, x2, y2, x3, y3, x4, y4 = map(float, parts[:8])
            bbox_category, difficult = parts[8], parts[9]
            if bbox_category == category:

                # Assuming the order is top-left,
                # top-right, bottom-right, bottom-left
                top_left = [y1, x1]
                top_right = [y2, x2]
                bottom_right = [y3, x3]
                bottom_left = [y4, x4]

                bbox = [bottom_left, bottom_right, top_left, top_right]
                bboxes.append(bbox)
    return bboxes


def get_boxes(folder_path, file_name, category='small-vehicle', unwanted_category='large-vehicle'):

    bboxes = []

    file_path = os.path.join(folder_path, file_name)
    with open(file_path, 'r') as file:
        print(file_path, end='\n\n\n')
        for line in file:
            parts = line.strip().split()
            x1, y1, x2, y2, x3, y3, x4, y4 = map(float, parts[:8])
            bbox_category, difficult = parts[8], parts[9]
            if bbox_category == category:

                # Assuming the order is top-left,
                # top-right, bottom-right, bottom-left
                top_left = [y1, x1]
                top_right = [y2, x2]
                bottom_right = [y3, x3]
                bottom_left = [y4, x4]

                bbox = [bottom_left, bottom_right, top_left, top_right]
                bboxes.append(bbox)
            if bbox_category == unwanted_category:
                return []
    return bboxes

if __name__ == "__main__":

    import os
    from concurrent.futures import ThreadPoolExecutor, as_completed

    folder_path = '/app/data/split_ss_dota/test/annfiles'

    def process_file(file):
    # Wrapper function to call `parse_one_file` for each file
        return parse_one_file(folder_path, file, category='large-vehicle')

    

    # List all files in the directory
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    # Process files in parallel
    with ThreadPoolExecutor() as executor:
        # Map the process_file function to each file
        future_to_file = {executor.submit(process_file, file): file for file in files}
        all_bboxes = []
        for future in as_completed(future_to_file):
            bboxes = future.result()
            all_bboxes.extend(bboxes)  # Collect all bboxes

    # all_bboxes will now contain all the bounding boxes for 'large-vehicle' across all files
    print(len(all_bboxes))