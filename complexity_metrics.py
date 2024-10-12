from enum import Enum
import numpy as np  # Always import NumPy for CPU operations
import cv2
import pandas as pd
import logging
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor as Executor
import functools
import multiprocessing

class OutputType(Enum):
    AVERAGE_METRICS = "average"
    PER_FRAME_METRICS = "per_frame"
    TOTAL_SCORE = "score"

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Attempt to import CuPy for GPU processing (fallback to NumPy if unavailable)
try:
    import cupy as cp
    use_gpu = True
    logger.info("CuPy is available. Using GPU acceleration.")
except ImportError:
    cp = np  # Fallback to NumPy
    use_gpu = False
    logger.info("CuPy is not available. Using CPU processing.")

# Helper function to validate video path
def validate_video_path(input_path):
    """Check if the input is a valid video or frame file."""
    if not isinstance(input_path, str):
        raise ValueError("Invalid input path. Please provide a valid file path.")

    if input_path.endswith(('.mp4', '.avi', '.mov')):
        return 'video'
    elif input_path.endswith(('.jpg', '.png')):
        return 'frame'
    else:
        raise ValueError("Unsupported file type. Please provide a video or frame file.")
    
# Extract frame timestamps from the video
def extract_frame_timestamps(video_path, frame_interval=10):
    """
    Extract frame timestamps from a video file.

    Parameters:
        video_path (str): The path to the video file.
        frame_interval (int): The interval at which frames should be processed (default is 10).

    Returns:
        list: A list of timestamps for the frames in milliseconds.
    """
    validate_video_path(video_path)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    frame_timestamps = []

    try:
        if not cap.isOpened():
            logger.error(f"Error opening video file: {video_path}")
            return []

        while cap.isOpened():
            ret, _ = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
                frame_timestamps.append(timestamp)

            frame_count += 1
    finally:
        cap.release()

    return frame_timestamps

# Read frame pairs with a specified interval
def read_frame_pairs(video_path, frame_interval=10):
    """
    Read pairs of frames from a video file at a specified interval.

    Parameters:
        video_path (str): The path to the video file.
        frame_interval (int): The interval at which frames should be processed (default is 10).

    Returns:
        list: A list of tuples containing pairs of consecutive frames.
    """
    validate_video_path(video_path)

    cap = cv2.VideoCapture(video_path)
    frames = []
    prev_frame = None
    frame_count = 0

    try:
        if not cap.isOpened():
            logger.error(f"Error opening video file: {video_path}")
            return []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % frame_interval == 0:
                if prev_frame is not None:
                    frames.append((frame, prev_frame))  # Store current and previous frame as a pair
                prev_frame = frame
    finally:
        cap.release()

    return frames

# Exponential smoothing using GPU or CPU
def smooth_data(data, alpha=0.8):
    """
    Apply exponential smoothing to the data.

    Parameters:
        data (list or array): The data to be smoothed.
        alpha (float): The smoothing factor (default is 0.8).

    Returns:
        array: The smoothed data.
    """
    return pd.Series(data).ewm(alpha=alpha).mean().to_cupy() if use_gpu else pd.Series(data).ewm(alpha=alpha).mean().to_numpy()

# Process frames in batches with CuPy (GPU) or NumPy (CPU)
def process_in_batches(frames, process_func, num_workers, batch_size=100, **kwargs):
    """
    Process frames in batches using ProcessPoolExecutor.

    Args:
        frames: list of frames to process.
        process_func: the function to apply to each batch.
        num_workers: number of worker processes to use.
        batch_size: number of frames per batch.
        kwargs: additional arguments to pass to the process function.

    Returns:
        List of processed results.
    """
    results = []
    with Executor(max_workers=num_workers) as executor:
        for i in tqdm(range(0, len(frames), batch_size), desc="Batch Processing"):
            batch = frames[i:i+batch_size]
            process_with_args = functools.partial(process_func, **kwargs)
            results.extend(executor.map(process_with_args, batch))
    return results

def process_frame_interval_for_parallel(timestamps):
    """
    Calculate the frame rate interval between two consecutive timestamps.

    Parameters:
        timestamps (tuple): A tuple containing (prev_timestamp, curr_timestamp).

    Returns:
        float: The frame rate between two timestamps.
    """
    prev_timestamp, curr_timestamp = timestamps
    frame_interval = (curr_timestamp - prev_timestamp) / 1000.0  # Convert milliseconds to seconds

    if frame_interval > 0:
        return 1.0 / frame_interval  # Frame rate (in fps)
    return 0.0  # Return 0 if frame interval is invalid



def normalize(value, min_value, max_value):
    """Normalize a value to the range [0, 1]."""
    return (value - min_value) / (max_value - min_value) if max_value > min_value else 0

# Define the min and max values for normalization
min_max_values = {
    'advanced_motion_complexity': (0.0, 10.0),
    'dct_complexity': (1e6, 5e7),
    'temporal_dct_complexity': (0.0, 1e7),
    'histogram_complexity': (0.0, 8.0),
    'edge_detection_complexity': (0.0, 1000),
    'orb_feature_complexity': (0.0, 5000),
    'color_histogram_complexity': (0.0, 8.0),
    'framerate_variation_complexity': (0.0, 2.0)
}


# Calculate scene complexity using various metrics (DCT, ORB, etc.)
def calculate_scene_complexity(video_path, resize_width, resize_height, frame_interval=10, smoothing_factor=0.8, num_workers=None, batch_size=100, output_type=OutputType.AVERAGE_METRICS):
    """
    Calculate various scene complexity metrics from a video file.

    Parameters:
        video_path (str): The path to the video file.
        resize_width (int): The width to resize frames to.
        resize_height (int): The height to resize frames to.
        frame_interval (int): The interval at which frames should be processed (default is 10).
        smoothing_factor (float): The smoothing factor for exponential smoothing (default is 0.8).
        num_workers (int): Number of worker processes to use (default is half of CPU count).
        batch_size (int): Number of frames per batch (default is 100).

    Returns:
        tuple: A tuple containing average metrics calculated from the video.
    """

    frame_pairs = read_frame_pairs(video_path, frame_interval)

    if num_workers is None:
        num_workers = multiprocessing.cpu_count() // 2

    logger.info("Calculating advanced motion complexity...")
    advanced_motion_complexity = process_in_batches(frame_pairs, process_frame_complexity, num_workers, batch_size)
    smoothed_advanced_motion_complexity = smooth_data(advanced_motion_complexity, smoothing_factor)
    advanced_motion_complexity_norm = normalize(smoothed_advanced_motion_complexity, *min_max_values['advanced_motion_complexity'])

    frames = [pair[0] for pair in frame_pairs]  # Use only the first frame for other metrics

    logger.info("Calculating DCT scene complexity...")
    dct_complexity = process_in_batches(frames, functools.partial(process_dct_frame, resize_width=resize_width, resize_height=resize_height), num_workers, batch_size)
    smoothed_dct_complexity = smooth_data(dct_complexity, smoothing_factor)
    dct_complexity_norm = normalize(smoothed_dct_complexity, *min_max_values['dct_complexity'])

    logger.info("Calculating histogram complexity...")
    histogram_complexity = process_in_batches(frames, functools.partial(process_histogram_frame, resize_width=resize_width, resize_height=resize_height), num_workers, batch_size)
    smoothed_histogram_complexity = smooth_data(histogram_complexity, smoothing_factor)
    histogram_complexity_norm = normalize(smoothed_histogram_complexity, *min_max_values['histogram_complexity'])

    logger.info("Calculating edge detection complexity...")
    edge_detection_complexity = process_in_batches(frames, functools.partial(process_edge_frame, resize_width=resize_width, resize_height=resize_height), num_workers, batch_size)
    smoothed_edge_detection_complexity = smooth_data(edge_detection_complexity, smoothing_factor)
    edge_detection_complexity_norm = normalize(smoothed_edge_detection_complexity, *min_max_values['edge_detection_complexity'])

    logger.info("Calculating ORB feature complexity...")
    orb_feature_complexity = process_in_batches(frames, process_orb_frame_for_parallel, num_workers, batch_size)
    smoothed_orb_feature_complexity = smooth_data(orb_feature_complexity, smoothing_factor)
    orb_feature_complexity_norm = normalize(smoothed_orb_feature_complexity, *min_max_values['orb_feature_complexity'])

    logger.info("Calculating color histogram complexity...")
    color_histogram_complexity = process_in_batches(frames, functools.partial(process_color_histogram_frame, resize_width=resize_width, resize_height=resize_height), num_workers, batch_size)
    smoothed_color_histogram_complexity = smooth_data(color_histogram_complexity, smoothing_factor)
    color_histogram_complexity_norm = normalize(smoothed_color_histogram_complexity, *min_max_values['color_histogram_complexity'])

    logger.info("Calculating temporal DCT complexity...")
    temporal_dct_complexity = calculate_temporal_dct(video_path, resize_width, resize_height, frame_interval, smoothing_factor)
    smoothed_temporal_dct_complexity = smooth_data(temporal_dct_complexity, smoothing_factor)
    temporal_dct_complexity_norm = normalize(smoothed_temporal_dct_complexity, *min_max_values['temporal_dct_complexity'])
    


    frame_timestamps = extract_frame_timestamps(video_path, frame_interval)
    timestamp_pairs = list(zip(frame_timestamps[:-1], frame_timestamps[1:]))  # Consecutive timestamps
    framerate_variation_complexity = process_in_batches(timestamp_pairs, process_frame_interval_for_parallel, num_workers, batch_size)
    smoothed_framerate_variation_complexity = smooth_data(framerate_variation_complexity, smoothing_factor)
    framerate_variation_complexity_norm = normalize(smoothed_framerate_variation_complexity, *min_max_values['framerate_variation_complexity'])


    # Prepare the metrics dictionary
    metrics = {
        'advanced_motion_complexity': advanced_motion_complexity_norm,
        'dct_complexity': dct_complexity_norm,
        'temporal_dct_complexity': temporal_dct_complexity_norm,
        'histogram_complexity': histogram_complexity_norm,
        'edge_detection_complexity': edge_detection_complexity_norm,
        'orb_feature_complexity': orb_feature_complexity_norm,
        'color_histogram_complexity': color_histogram_complexity_norm,
        'framerate_variation_complexity': framerate_variation_complexity_norm
    }

    # Determine output based on the `output_type` argument
    if output_type == OutputType.AVERAGE_METRICS:
        return {key: np.mean(value) for key, value in metrics.items()}
    
    elif output_type == OutputType.PER_FRAME_METRICS:
        return metrics
    
    elif output_type == OutputType.TOTAL_SCORE:
        # Define weights for each metric
        weights = {
            'advanced_motion_complexity': 0.25,
            'dct_complexity': 0.15,
            'temporal_dct_complexity': 0.15,
            'histogram_complexity': 0.10,
            'edge_detection_complexity': 0.10,
            'orb_feature_complexity': 0.10,
            'color_histogram_complexity': 0.10,
            'framerate_variation_complexity': 0.05
        }

        # Apply weights and compute total complexity score
        total_score = sum(np.mean(metrics[metric]) * weight for metric, weight in weights.items())
        return total_score


# Optical Flow using GPU or CPU
def process_frame_complexity(frame_pair):
    """
    Calculate the motion complexity between two frames using optical flow.

    Parameters:
        frame_pair (tuple): A tuple containing the current and previous frames.

    Returns:
        float: The average motion magnitude between the two frames.
    """
    frame, prev_frame = frame_pair
    if frame is None or prev_frame is None:
        return 0.0

    curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    if use_gpu:
        # Use CUDA-accelerated Optical Flow on GPU
        gpu_curr_gray = cv2.cuda_GpuMat()
        gpu_prev_gray = cv2.cuda_GpuMat()
        gpu_curr_gray.upload(curr_gray)
        gpu_prev_gray.upload(prev_gray)
        optical_flow = cv2.cuda_FarnebackOpticalFlow_create(5, 0.5, False)
        flow = optical_flow.calc(gpu_prev_gray, gpu_curr_gray, None)
    else:
        # Fallback to CPU-based optical flow
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return np.mean(mag)

# DCT Scene Complexity Calculation using GPU or CPU
def process_dct_frame(frame, resize_width, resize_height):
    """
    Calculate the DCT (Discrete Cosine Transform) complexity of a frame.

    Parameters:
        frame: The input frame to process.
        resize_width (int): The width to resize the frame to.
        resize_height (int): The height to resize the frame to.

    Returns:
        float: The DCT complexity of the frame.
    """
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.resize(gray_frame, (resize_width, resize_height))
    if use_gpu:
        dct_frame = cv2.dct(cp.float32(gray_frame))  # Use CuPy for GPU
    else:
        dct_frame = cv2.dct(np.float32(gray_frame))  # Use NumPy for CPU
    return np.sum(dct_frame ** 2)

# ORB Feature Complexity using GPU or CPU
def process_orb_frame_for_parallel(frame, texture_threshold=10.0):
    """
    Calculate the number of ORB features in a frame.

    Parameters:
        frame: The input frame to process.
        texture_threshold (float): Threshold for detecting low texture.

    Returns:
        int: The number of keypoints detected or 0 if the texture is too low.
    """
    def is_low_texture(frame, threshold=texture_threshold):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        variance_of_laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()
        return variance_of_laplacian < threshold

    if is_low_texture(frame):
        logger.info("Low texture detected, skipping ORB computation.")
        return 0

    if use_gpu:
        orb = cv2.cuda_ORB_create()
        resized_frame = cv2.resize(frame, (64, 64))
        gpu_frame = cv2.cuda_GpuMat()
        gpu_frame.upload(resized_frame)
        gray_frame = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY)
        keypoints, _ = orb.detectAndCompute(gray_frame, None)
    else:
        orb = cv2.ORB_create()
        gray_frame = cv2.cvtColor(cv2.resize(frame, (64, 64)), cv2.COLOR_BGR2GRAY)
        keypoints, _ = orb.detectAndCompute(gray_frame, None)
    
    return len(keypoints) if keypoints is not None else 0

# Histogram complexity calculation using GPU or CPU
def process_histogram_frame(frame, resize_width, resize_height):
    """
    Calculate the histogram complexity of a frame.

    Parameters:
        frame: The input frame to process.
        resize_width (int): The width to resize the frame to.
        resize_height (int): The height to resize the frame to.

    Returns:
        float: The entropy of the histogram of the frame.
    """
    frame_resized = cv2.resize(frame, (resize_width, resize_height))
    gray_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
    
    if use_gpu:
        hist = cv2.calcHist([gray_frame], [0], None, [256], [0, 256])
        hist = cp.array(hist) / cp.sum(hist)
        entropy = -cp.sum(hist[hist > 0] * cp.log2(hist[hist > 0]))
    else:
        hist = cv2.calcHist([gray_frame], [0], None, [256], [0, 256])
        hist = hist / hist.sum()
        entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))

    return entropy

def process_color_histogram_frame(frame, resize_width, resize_height):
    """
    Calculate the color histogram complexity of a frame.

    Parameters:
        frame: The input frame to process.
        resize_width (int): The width to resize the frame to.
        resize_height (int): The height to resize the frame to.

    Returns:
        float: The entropy of the color histograms of the frame.
    """
    resized_frame = cv2.resize(frame, (resize_width, resize_height))
    
    if use_gpu:
        # GPU processing with CuPy
        hist_b = cv2.calcHist([resized_frame], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([resized_frame], [1], None, [256], [0, 256])
        hist_r = cv2.calcHist([resized_frame], [2], None, [256], [0, 256])
        
        # Avoid division by zero by checking sums
        hist_b_sum = cp.sum(hist_b)
        hist_g_sum = cp.sum(hist_g)
        hist_r_sum = cp.sum(hist_r)
        
        if hist_b_sum == 0 or hist_g_sum == 0 or hist_r_sum == 0:
            return float('nan')
        
        hist_b = cp.array(hist_b) / hist_b_sum
        hist_g = cp.array(hist_g) / hist_g_sum
        hist_r = cp.array(hist_r) / hist_r_sum
        
        entropy = - (cp.sum(hist_b * cp.log2(hist_b + 1e-8)) +
                     cp.sum(hist_g * cp.log2(hist_g + 1e-8)) +
                     cp.sum(hist_r * cp.log2(hist_r + 1e-8)))
    else:
        # CPU processing with NumPy
        hist_b = cv2.calcHist([resized_frame], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([resized_frame], [1], None, [256], [0, 256])
        hist_r = cv2.calcHist([resized_frame], [2], None, [256], [0, 256])
        
        # Avoid division by zero by checking sums
        hist_b_sum = hist_b.sum()
        hist_g_sum = hist_g.sum()
        hist_r_sum = hist_r.sum()
        
        if hist_b_sum == 0 or hist_g_sum == 0 or hist_r_sum == 0:
            return float('nan')
        
        hist_b = hist_b / hist_b_sum
        hist_g = hist_g / hist_g_sum
        hist_r = hist_r / hist_r_sum
        
        entropy = - (np.sum(hist_b * np.log2(hist_b + 1e-8)) +
                     np.sum(hist_g * np.log2(hist_g + 1e-8)) +
                     np.sum(hist_r * np.log2(hist_r + 1e-8)))

    return entropy

def process_edge_frame(frame, resize_width, resize_height):
    """
    Calculate edge detection complexity using Canny edge detection algorithm.

    Parameters:
        frame: The input frame to process.
        resize_width: The target width to resize the frame.
        resize_height: The target height to resize the frame.

    Returns:
        float: The number of edge pixels in the frame.
    """
    # Resize the frame
    frame_resized = cv2.resize(frame, (resize_width, resize_height))
    
    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
    
    # Apply Canny edge detection (using GPU or CPU)
    if use_gpu:
        gpu_frame = cv2.cuda_GpuMat()
        gpu_frame.upload(gray_frame)
        edge_detector = cv2.cuda.createCannyEdgeDetector(100, 200)
        edge_pixels = edge_detector.detect(gpu_frame)
        return cp.sum(edge_pixels.download() > 0)
    else:
        edge_pixels = cv2.Canny(gray_frame, 100, 200)
        return np.sum(edge_pixels > 0)

def calculate_temporal_dct(video_path, resize_width, resize_height, frame_interval=10, smoothing_factor=0.8):
    """
    Calculate temporal DCT complexity from a video file.

    Parameters:
        video_path (str): The path to the video file.
        resize_width (int): The width to resize frames to.
        resize_height (int): The height to resize frames to.
        frame_interval (int): The interval at which frames should be processed (default is 10).
        smoothing_factor (float): The smoothing factor for exponential smoothing (default is 0.8).

    Returns:
        float: The average temporal DCT complexity.
    """
    frames = read_frame_pairs(video_path, frame_interval)
    prev_gray_frame = None
    temporal_dct_energies = []

    for pair in frames:
        frame, previous_frame = pair
        if frame is None or previous_frame is None:
            continue

        # Convert to grayscale and resize
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.resize(gray_frame, (resize_width, resize_height))

        if prev_gray_frame is not None:
            temporal_energy = process_temporal_dct_frame(prev_gray_frame, gray_frame, resize_width, resize_height)
            temporal_dct_energies.append(temporal_energy)

        prev_gray_frame = gray_frame

    return temporal_dct_energies if len(temporal_dct_energies) > 0 else np.array([0.0])

def process_temporal_dct_frame(prev_gray_frame, curr_gray_frame, resize_width, resize_height):
    """
    Calculate the DCT difference between consecutive frames.

    Parameters:
        prev_gray_frame: The previous grayscale frame.
        curr_gray_frame: The current grayscale frame.
        resize_width (int): The width to resize the frames to.
        resize_height (int): The height to resize the frames to.

    Returns:
        float: The DCT difference between the two frames.
    """
    # Resize frames to the desired resolution
    prev_gray_frame = cv2.resize(prev_gray_frame, (resize_width, resize_height))
    curr_gray_frame = cv2.resize(curr_gray_frame, (resize_width, resize_height))

    if use_gpu:
        # Transfer frames to GPU using CuPy
        prev_frame_gpu = cp.array(prev_gray_frame, dtype=cp.float32)
        curr_frame_gpu = cp.array(curr_gray_frame, dtype=cp.float32)
        
        # Perform DCT on GPU (CuPy doesn't have a direct DCT, so we can use CuPy FFT as an alternative)
        prev_frame_dct = cp.fft.fft2(prev_frame_gpu)
        curr_frame_dct = cp.fft.fft2(curr_frame_gpu)
        
        # Compute the difference in DCT energies
        dct_difference = cp.sum(cp.abs(prev_frame_dct - curr_frame_dct))
        return float(dct_difference.get())  # Transfer result back to CPU
    else:
        # Perform DCT on CPU using OpenCV's cv2.dct
        # Ensure frames are grayscale and float32 before applying DCT
        # Ensure the frames are single-channel floating-point matrices
        prev_gray_frame = np.float32(prev_gray_frame)
        curr_gray_frame = np.float32(curr_gray_frame)

        # Apply DCT
        prev_frame_dct = cv2.dct(prev_gray_frame)
        curr_frame_dct = cv2.dct(curr_gray_frame)

        # Compute the difference in DCT energies
        dct_difference = np.sum(np.abs(prev_frame_dct - curr_frame_dct))
        return dct_difference
       
