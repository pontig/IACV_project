import numpy as np

alphas = [1.0, 2.0069, 1.9981, 2.0003, 2.0002, 1.1988, 2.3975]
betas = [0, -2320.6, -2255.38, -2440.13, -1997.91, -3618.11, -3745.56]

minimum_global_time = np.inf
maximum_global_time = -np.inf

def compute_global_time(frame_indices, alpha, beta = 0):
    """
    Convert frame indices to global timestamps and update min/max global time.
    
    frame_indices: List or array of frame indices
    alpha: Time scale factor (frame rate correction)
    beta: Time offset (initial shift)
    
    Returns: Global timestamps
    """
    global minimum_global_time, maximum_global_time
    global_times = frame_indices * alpha + beta
    minimum_global_time = min(minimum_global_time, np.min(global_times))
    maximum_global_time = max(maximum_global_time, np.max(global_times))
    return global_times

def get_rainbow_color(global_timestamp):
    """
    Map a global timestamp to a color sampled from a rainbow distribution.

    global_timestamp: The timestamp to map to a color.

    Returns: A tuple representing the RGB color.
    """
    global minimum_global_time, maximum_global_time

    # Normalize the timestamp to a range [0, 1]
    normalized_time = (global_timestamp - minimum_global_time) / (maximum_global_time - minimum_global_time)
    normalized_time = np.clip(normalized_time, 0, 1)  # Ensure it's within [0, 1]

    # Map the normalized time to a color in the rainbow spectrum
    # Adjust the phase shifts to follow the order: red, orange, yellow, etc.
    color = np.array([np.sin(2 * np.pi * normalized_time + 0),  # Red
                      np.sin(2 * np.pi * normalized_time + np.pi / 3),  # Green
                      np.sin(2 * np.pi * normalized_time + 2 * np.pi / 3)])  # Blue

    # Normalize the color to [0, 1]
    color = (color + 1) / 2
    return tuple(color)