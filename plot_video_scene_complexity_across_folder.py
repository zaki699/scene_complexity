import argparse
import json
import multiprocessing
from multiprocessing.dummy import freeze_support
import os
from itertools import cycle
import sys
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
# Import the calculate_average_scene_complexity function
from complexity_metrics import OutputType, calculate_scene_complexity


def plot_metrics_across_videos(folder_path, resize_width, resize_height, frame_interval, num_workers, batch_size, output_type):
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        sys.exit(1)

    # Get the list of video files in the folder
    video_files = [f for f in os.listdir(folder_path) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    
    # Check if the folder contains any video files
    if not video_files:
        print(f"Error: No video files found in '{folder_path}'. Please check the folder and ensure it contains valid video files.")
        sys.exit(1)

    metrics_per_video = {}
    max_length = 0
    
    # Iterate over videos and gather metric data
     # Iterate over videos and gather metric data
    for video_file in tqdm(video_files, desc="Processing Videos"):
        video_path = os.path.join(folder_path, video_file)
        metrics_dict = calculate_scene_complexity(
            video_path, resize_width=resize_width, resize_height=resize_height, frame_interval=frame_interval, num_workers=num_workers, batch_size=batch_size, output_type=output_type
        )

        # Debugging: Print the keys in metrics_dict and their first few values
        print(f"\nMetrics for {video_file}:")
        for metric_key, metric_values in metrics_dict.items():
            print(f"{metric_key}: {metric_values[:5]}...")  # Show first 5 values for each metric
        
        for metric_key in metrics_dict:
            metric_values = metrics_dict[metric_key]
            if len(metric_values) > max_length:
                max_length = len(metric_values)
            
            if metric_key not in metrics_per_video:
                metrics_per_video[metric_key] = []
            metrics_per_video[metric_key].append((video_file, metric_values))
    
    # Check if any metric is blank
    for metric_key, metric_data in metrics_per_video.items():
        if all(np.all(np.isclose(metric_values, 0)) for _, metric_values in metric_data):
            print(f"Warning: No valid data for {metric_key}!")
    
    # Continue plotting only for valid metrics
    x_values = range(0, max_length * frame_interval, frame_interval)
    color_cycle = cycle(plt.cm.Set3.colors)
    
    # Create subplots (4x2 layout for example)
    num_metrics = len(metrics_per_video)
    fig, axes = plt.subplots((num_metrics + 1) // 2, 2, figsize=(15, 5 * ((num_metrics + 1) // 2)), constrained_layout=True)
    axes = axes.flatten()[:num_metrics]  # Only use as many axes as needed

    # Plot for each metric
    for idx, (metric_key, metric_data) in enumerate(metrics_per_video.items()):
        if len(metric_data) == 0:  # Skip if no data
            continue
        for video_file, metric_values in metric_data:
            if len(metric_values) == 0:
                continue  # Skip if the metric values are empty
            if len(metric_values) < max_length:
                metric_values = np.pad(metric_values, (0, max_length - len(metric_values)), constant_values=0)
            axes[idx].plot(x_values, metric_values, label=video_file, color=next(color_cycle))
        
        axes[idx].set_title(f'{metric_key}')
        axes[idx].set_xlabel('Frame Number')
        axes[idx].set_ylabel('Complexity')
        axes[idx].legend(loc='upper left', fontsize='small', bbox_to_anchor=(1.05, 1), borderaxespad=0.)

    plt.show()

# Load configuration from a JSON file
def load_config(config_path='config.json'):
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Config file '{config_path}' not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Config file '{config_path}' is not a valid JSON.")
        sys.exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="""
        This script processes videos from a specified folder to analyze their scene complexity. 
        It extracts keyframes, analyzes frame-by-frame complexity metrics (such as motion, DCT, and histogram complexities), 
        and plots these metrics for each video. The configuration for resizing, frame intervals, batch sizes, and more is 
        read from a JSON configuration file. The results are displayed as plots for easy comparison of complexity across videos.
        """
    )
    parser.add_argument(
        'config_file', 
        type=str, 
        help="Path to the JSON configuration file that contains settings for video processing (e.g., resize dimensions, frame interval, batch size)."
    )

    parser.add_argument(
            'folder_path', 
            type=str, 
            help="Path to the folder containing the video files to be analyzed. Supported formats: .mp4, .avi, .mov, .mkv."
    )
    
    args = parser.parse_args()

    config = load_config(args.config_file)


    resize_width = config.get("resize_width", 64)
    resize_height = config.get("resize_height", 64)
    frame_interval = config.get("frame_interval", 10)
    smoothing_factor = config.get("smoothing_factor", 0.8)
    batch_size = config.get("batch_size", 100)
    num_workers = config.get("num_workers", multiprocessing.cpu_count())
    
    freeze_support()  # Ensure proper multiprocessing support
    plot_metrics_across_videos(args.folder_path, resize_width=resize_width,resize_height=resize_height,frame_interval=frame_interval,num_workers=num_workers,batch_size=batch_size,output_type=OutputType.PER_FRAME_METRICS)
