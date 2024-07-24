import os

def get_directory_files(root_path):
    f = open("/app/data/files.txt","w+")
    directory_files = {}
    for dirpath, dirnames, filenames in os.walk(root_path):
        # Sort the directory and filenames
        dirnames.sort()
        filenames.sort()
        # Use sorted absolute paths for reliable comparison across different systems
        full_file_paths = [os.path.join(dirpath, file) for file in filenames]
        full_file_paths.sort()
        # Store the sorted list of filenames in each directory
        directory_files[dirpath] = full_file_paths
    f.write(directory_files)
    f.flush()
    return directory_files

get_directory_files("/app/data")