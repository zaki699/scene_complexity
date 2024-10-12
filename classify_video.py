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
    """
    Classify a video based on its total scene complexity score and assign VBV settings.
    
    Parameters:
        video_path (str): Path to the video file.
        resize_width (int): Width to resize frames.
        resize_height (int): Height to resize frames.
        frame_interval (int): Interval for frame extraction.
        num_workers (int): Number of workers for processing.
        batch_size (int): Batch size for frame processing.
        output_type (Enum): Type of output required from scene complexity calculation.

    Returns:
        None
    """
    # Check if video file exists
    if not os.path.exists(video_path):
        sys.exit(f"Error: Video '{video_path}' does not exist.")

    # Calculate total scene complexity score
    total_score = calculate_scene_complexity(
        video_path, 
        resize_width=resize_width, 
        resize_height=resize_height, 
        frame_interval=frame_interval, 
        num_workers=num_workers, 
        batch_size=batch_size, 
        output_type=output_type
    )
    
    # Classify the video complexity and assign VBV settings
    complexity_class = classify_complexity(np.mean(total_score))
    vbv_settings = assign_vbv_settings(np.mean(total_score))

    # Print results in a clean format
    print_video_classification(video_path, total_score, complexity_class, vbv_settings)


def classify_complexity(total_score):
    """
    Classify the video based on the total complexity score.
    
    Parameters:
        total_score (float): The total complexity score.
    
    Returns:
        str: The complexity classification.
    """
    if total_score > 0.9:
        return "Extremely High Complexity"
    elif total_score > 0.75:
        return "High Complexity"
    elif total_score > 0.6:
        return "Moderately High Complexity"
    elif total_score > 0.5:
        return "Moderate Complexity"
    elif total_score > 0.35:
        return "Low to Moderate Complexity"
    elif total_score > 0.25:
        return "Low Complexity"
    else:
        return "Static/Low Motion"


def assign_vbv_settings(total_score):
    """
    Assigns VBV settings (bitrate and buffer size) based on the scene complexity score.

    Parameters:
        total_score (float): The total complexity score of the video.

    Returns:
        dict: A dictionary with 'vbv_minrate', 'vbv_maxrate', and 'vbv_bufsize' values.
    """
    vbv_settings = {
        "Extremely High Complexity": {'vbv_minrate': 5000, 'vbv_maxrate': 10000, 'vbv_bufsize': 20000},
        "High Complexity": {'vbv_minrate': 4000, 'vbv_maxrate': 8000, 'vbv_bufsize': 16000},
        "Moderately High Complexity": {'vbv_minrate': 3000, 'vbv_maxrate': 6000, 'vbv_bufsize': 12000},
        "Moderate Complexity": {'vbv_minrate': 2500, 'vbv_maxrate': 5000, 'vbv_bufsize': 10000},
        "Low to Moderate Complexity": {'vbv_minrate': 2000, 'vbv_maxrate': 4000, 'vbv_bufsize': 8000},
        "Low Complexity": {'vbv_minrate': 1500, 'vbv_maxrate': 3000, 'vbv_bufsize': 6000},
        "Static/Low Motion": {'vbv_minrate': 1000, 'vbv_maxrate': 2000, 'vbv_bufsize': 4000}
    }
    
    complexity_class = classify_complexity(total_score)
    
    return vbv_settings[complexity_class]


def print_video_classification(video_path, total_score, complexity_class, vbv_settings):
    """
    Print video classification results and VBV settings.
    
    Parameters:
        video_path (str): Path to the video file.
        total_score (float): The total complexity score.
        complexity_class (str): The complexity classification.
        vbv_settings (dict): The assigned VBV settings.
    
    Returns:
        None
    """
    print(f"--- Video Analysis: {video_path} ---")
    print(f"Total Complexity Score: {np.mean(total_score):.2f}")
    print(f"Classification: {complexity_class}")
    print(f"Assigned VBV Minrate: {vbv_settings['vbv_minrate']} kbps")
    print(f"Assigned VBV Maxrate: {vbv_settings['vbv_maxrate']} kbps")
    print(f"Assigned VBV Buffer Size: {vbv_settings['vbv_bufsize']} kbps")
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
