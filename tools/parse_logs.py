# import os
# import pandas as pd
# import re
# import matplotlib.pyplot as plt

# def parse_log_file(filepath):
#     data = {
#         'epoch': [],
#         'map0.1': [],
#         'map0.5': [],
#         'map0.8': []
#     }
#     exp_name = None
#     with open(filepath, 'r') as file:
#         for line in file:
#             if "Exp name" in line:
#                 # Extract experiment name
#                 match = re.search(r"Exp name: (\w+_\w+)", line)
#                 if match:
#                     exp_name = match.group(1)
#             if "dota/AP10" in line:
#                 # Extract epoch and MAP values
#                 epoch_match = re.search(r"Epoch\(val\) \[(\d+)\]", line)
#                 ap10_match = re.search(r"dota/AP10: (\d+\.\d+)", line)
#                 ap50_match = re.search(r"dota/AP50: (\d+\.\d+)", line)
#                 ap80_match = re.search(r"dota/AP80: (\d+\.\d+)", line)
#                 if epoch_match and ap10_match and ap50_match and ap80_match:
#                     data['epoch'].append(int(epoch_match.group(1)))
#                     data['map0.1'].append(float(ap10_match.group(1)))
#                     data['map0.5'].append(float(ap50_match.group(1)))
#                     data['map0.8'].append(float(ap80_match.group(1)))
#     return pd.DataFrame(data), exp_name

# # Example usage
# # log_data, experiment_name = parse_log_file('path_to_log_file.log')


# def find_log_files(root_dir):
#     log_files = []
#     visited_dirs = set()  # Set to keep track of visited directories

#     for subdir, dirs, files in os.walk(root_dir, followlinks=True):  # followlinks=True allows following symbolic links
#         real_path = os.path.realpath(subdir)
#         if real_path in visited_dirs:
#             continue  # Skip this directory as it has already been visited
#         visited_dirs.add(real_path)

#         for file in files:
#             if file.endswith('.log'):
#                 log_file_path = os.path.join(subdir, file)
#                 log_files.append(log_file_path)
#     return log_files


# def plot_single_experiment(data):
#     plt.figure(figsize=(10, 6))
#     plt.plot(data['epoch'], data['map0.1'], label='map0.1', marker='o')
#     plt.plot(data['epoch'], data['map0.5'], label='map0.5', marker='o')
#     plt.plot(data['epoch'], data['map0.8'], label='map0.8', marker='o')
#     plt.title('MAP Values Over Epochs')
#     plt.xlabel('Epoch')
#     plt.ylabel('MAP Value')
#     plt.legend()
#     plt.grid(True)
#     plt.show()

# # plot_single_experiment(log_data)


# def plot_maps_across_experiments(log_files):
#     all_data = {}
#     for filepath in log_files:
#         data, exp_name = parse_log_file(filepath)
#         all_data[exp_name] = data
    
#     maps = ['map0.1', 'map0.5', 'map0.8']
#     for map_type in maps:
#         plt.figure(figsize=(10, 6))
#         for exp_name, data in all_data.items():
#             plt.plot(data['epoch'], data[map_type], label=exp_name, marker='o')
#         plt.title(f'{map_type} Values Over Epochs Across Experiments')
#         plt.xlabel('Epoch')
#         plt.ylabel(map_type)
#         plt.legend()
#         plt.grid(True)
#         # plt.savefig(f"/app/work_dirs/parsing_visualization/")
#         plt.show()

# # Example usage assuming you have a list of file paths
# # plot_maps_across_experiments(['file1.log', 'file2.log', 'file3.log'])






# def main(root_directory):
#     # Find all log files
#     log_files = find_log_files(root_directory)
    
#     # Assuming you want to plot data from each file separately
#     for log_file in log_files:
#         try:
#             log_data, experiment_name = parse_log_file(log_file)
#             print(f"Processing {log_file} for experiment {experiment_name}")
#             plot_single_experiment(log_data)
#         except Exception as e:
#             print(f"Failed to process {log_file}: {e}")

#     # If you want to plot MAP values across all experiments:
#     if log_files:  # Ensure there are log files to process
#         plot_maps_across_experiments(log_files)

# # Run the main function with the root directory path
# if __name__ == "__main__":
#     root_directory = "/app/work_dirs/rotated_rtmdet_l-3x-dota"
#     main(root_directory)





import pandas as pd
import matplotlib.pyplot as plt
import os
import re

# def find_log_files(root_dir):
#     """ Recursively find all log files in the specified directory. """
#     log_files = []
#     for subdir, dirs, files in os.walk(root_dir):
#         for file in files:
#             if file.endswith('.log'):
#                 log_files.append(os.path.join(subdir, file))
#     return log_files

def parse_detailed_log_file(filepath):
    data = []
    current_epoch = None
    iou_thr = None
    mAP = {}  # To store mAP values for each epoch and IoU
    buffer = []  # Buffer to store data temporarily

    with open(filepath, 'r') as file:
        for line in file:
            # Check for and process epoch changes
            if "Epoch(val)" in line:
                # Flush the buffer if not empty and appropriate mAP is available
                if buffer:
                    for entry in buffer:
                        # Append buffered entries to the main dataset if mAP data is available
                        enter = False
                        if entry['epoch'] in mAP and entry['iou_thr'] in mAP[entry['epoch']]:
                            enter = True
                            entry['mAP'] = mAP[entry['epoch']][entry['iou_thr']]
                            data.append(entry)
                    if enter is True:
                        buffer = []  # Clear buffer after processing
                        enter = False

                # Capture new epoch number
                epoch_match = re.search(r"Epoch\(val\)\s*\[(\d+)\]", line)
                if epoch_match:
                    current_epoch = int(epoch_match.group(1))
                    mAP[current_epoch] = {}

            # Capture and update IoU threshold
            if "iou_thr:" in line:
                iou_match = re.search(r"iou_thr:\s*(\d+\.\d+)", line)
                if iou_match:
                    iou_thr = f'{float(iou_match.group(1)):.1f}'  # Standardize the IoU threshold format

            # Capture mAP values
            if "dota/mAP:" in line and current_epoch is not None:
                mAP_matches = re.search(r"dota/mAP: (\d+\.\d+).*dota/AP10: (\d+\.\d+).*dota/AP50: (\d+\.\d+).*dota/AP80: (\d+\.\d+)", line)
                if mAP_matches:
                    # Store mAP values for the current epoch
                    mAP[current_epoch] = {
                        '0.1': float(mAP_matches.group(2)),
                        '0.5': float(mAP_matches.group(3)),
                        '0.8': float(mAP_matches.group(4))
                    }

            # Capture and buffer data for 'large-vehicle'
            if "large-vehicle" in line and current_epoch is not None and iou_thr is not None:
                parts = line.split('|')
                if len(parts) < 6:
                    continue
                gts, dets, recall, ap = map(str.strip, parts[2:6])
                buffer.append({
                    'epoch': current_epoch,
                    'iou_thr': iou_thr,
                    'gts': int(gts),
                    'dets': int(dets),
                    'recall': float(recall),
                    'ap': float(ap)
                })

        # Process any remaining buffered entries
        for entry in buffer:
            if entry['epoch'] in mAP and entry['iou_thr'] in mAP[entry['epoch']]:
                entry['mAP'] = mAP[entry['epoch']][entry['iou_thr']]
                data.append(entry)

    return pd.DataFrame(data)

# def plot_class_data(data, class_name):
#     """ Generate and save plots of metrics over epochs for a specific class. """
#     fig, ax = plt.subplots(figsize=(12, 6))
#     for iou_thr, group_data in data.groupby('iou_thr'):
#         ax.plot(group_data['epoch'], group_data['recall'], label=f'Recall IoU {iou_thr}', marker='o')
#         ax.plot(group_data['epoch'], group_data['ap'], label=f'AP IoU {iou_thr}', marker='x')
    
#     ax.set_title(f'Performance Metrics for {class_name} Over Epochs')
#     ax.set_xlabel('Epoch')
#     ax.set_ylabel('Metric Value')
#     ax.legend()
#     ax.grid(True)
    
#     plt.savefig(f'{class_name}_metrics.png')
#     plt.show()

def plot_metrics_single_experiment(data, experiment_name):
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['b', 'g', 'r']  # Colors for different IoU thresholds
    markers = ['o', 'x', '^']  # Different markers for mAP, Precision (AP), Recall

    # Filter data for the specific experiment
    experiment_data = data[data['experiment'] == experiment_name]

    for idx, iou in enumerate(['0.1', '0.5', '0.8']):
        subset = experiment_data[experiment_data['iou_thr'] == iou]
        ax.plot(subset['epoch'], subset['mAP'], label=f'mAP{iou}', color=colors[idx], marker=markers[0])
        ax.plot(subset['epoch'], subset['ap'], label=f'precision{iou} (AP)', color=colors[idx], marker=markers[1])  # Using AP as proxy for precision
        ax.plot(subset['epoch'], subset['recall'], label=f'recall{iou}', color=colors[idx], marker=markers[2])

    ax.set_title(f'Metrics Over Epochs for {experiment_name}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Metric Value')
    ax.legend()
    ax.grid(True)
    plt.show()



def find_log_files_and_experiments(root_dir):
    log_files = {}
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.log'):
                experiment_name = subdir.split('/')[-1]  # Assuming folder name is experiment name
                experiment_name = experiment_name[:experiment_name.find("_120_epochs")]
                log_files[os.path.join(subdir, file)] = experiment_name
    return log_files

def plot_metrics_across_files(log_files):
    fig, axs = plt.subplots(3, 3, figsize=(18, 16))  # 9 plots
    metrics = ['mAP', 'ap', 'recall']  # Adjust as per available data

    all_data = []

    for filepath, exp_name in log_files.items():
        data = parse_detailed_log_file(filepath)
        data['experiment'] = exp_name
        all_data.append(data)

    all_data = pd.concat(all_data, ignore_index=True)



    for i, iou in enumerate(sorted(all_data['iou_thr'].unique())):
        for j, metric in enumerate(metrics):
            ax = axs[j, i]
            for exp_name, group_data in all_data[all_data['iou_thr'] == iou].groupby('experiment'):
                ax.plot(group_data['epoch'], group_data[metric], label=exp_name, marker='o')
            title = f'{metric} {iou} Across Experiments' if metric == 'mAP' else f'{metric} {iou} Across Experiments Large-Vehicle'
            ax.set_title(title)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(f'{metric.capitalize()} Value')
            ax.legend()
            ax.grid(True)
            ax.set_ylim(bottom=0, top=1)

    plt.tight_layout()
    plt.savefig('metrics_across_files.png')
    plt.show()

    for _, exp_name in log_files.items():
        plot_metrics_single_experiment(data=all_data, experiment_name=exp_name)

def main(root_directory):


    log_files = find_log_files_and_experiments(root_directory)
    plot_metrics_across_files(log_files)

    # log_files = find_log_files(root_directory)
    # all_data = pd.DataFrame()
    # for log_file in log_files:
    #     print(f'Processing file: {log_file}')
    #     log_data = parse_detailed_log_file(log_file)
    #     all_data = pd.concat([all_data, log_data], ignore_index=True)

    # if not all_data.empty:
    #     plot_class_data(all_data, "Large-Vehicle")

if __name__ == "__main__":
    root_directory = "/app/work_dirs/rotated_rtmdet_l-3x-dota"
    main(root_directory)
