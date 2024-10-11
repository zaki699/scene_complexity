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


def classify_video(video_path, resize_width, resize_height, frame_interval, num_workers, batch_size, output_type):
    # Check if the folder exists
    if not os.path.exists(video_path):
        print(f"Error: Video '{video_path}' does not exist.")
        sys.exit(1)


    total_score = calculate_scene_complexity(
            video_path, resize_width=resize_width, resize_height=resize_height, frame_interval=frame_interval, num_workers=num_workers, batch_size=batch_size, output_type=output_type
        )
    
    # Classify based on total score
    if np.mean(total_score) > 0.9:
        complexity_class = "Extremely High Complexity"
    elif np.mean(total_score) > 0.75:
        complexity_class = "High Complexity"
    elif np.mean(total_score) > 0.6:
        complexity_class = "Moderately High Complexity"
    elif np.mean(total_score) > 0.5:
        complexity_class = "Moderate Complexity"
    elif np.mean(total_score) > 0.35:
        complexity_class = "Low to Moderate Complexity"
    elif np.mean(total_score) > 0.25:
        complexity_class = "Low Complexity"
    else:
        complexity_class = "Static/Low Motion"

    # Print results with improved readability
    print(f"--- Video Analysis: {video_path} ---")
    print(f"Total Complexity Score: {np.mean(total_score):.2f}")
    print(f"Classification: {complexity_class}")
    print("-----------------------------------")

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
            'video_path', 
            type=str, 
            help="Path to the video containing the video file to be analyzed. Supported formats: .mp4, .avi, .mov, .mkv."
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
    classify_video(args.video_path, resize_width=resize_width,resize_height=resize_height,frame_interval=frame_interval,num_workers=num_workers,batch_size=batch_size,output_type=OutputType.TOTAL_SCORE)