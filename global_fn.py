import numpy as np
import cv2 as cv
import warnings
import logging

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

def evaluate_beta(b, frames, splines, camera_info, main_camera, secondary_camera, inliers_list=None, return_f=False):
    """
    Calculate inlier ratio for a given beta value between two cameras
    
    Parameters:
    -----------
    b : float
        Beta value to test
    frames : list
        List of frames from secondary camera with detection points
    splines : dict
        Dictionary of splines for all cameras
    camera_info : dict
        Dictionary of camera information
    main_camera : int
        ID of main camera
    secondary_camera : int
        ID of secondary camera
    inliers_list : list, optional
        List to append inlier ratio to
    return_f : boolean, optional
        Whether to return also the fundamental matrix for a specific beta passed
        
    Returns:
    --------
    float
        Inlier ratio for the given beta
    """
    correspondences = [] # List of (spline_x1, spline_y1), (x2, y2), global_ts
    for frame in frames:
        global_ts = compute_global_time(frame[0], camera_info[main_camera].fps/camera_info[secondary_camera].fps, b)
        for spline_x, spline_y, tss in splines[main_camera]:
            if np.min(tss) <= global_ts <= np.max(tss):
                x1 = float(spline_x(global_ts))
                y1 = float(spline_y(global_ts))
                correspondences.append((
                    (x1, y1),
                    (frame[1], frame[2]),
                    global_ts
                ))
                break
            
    if not correspondences:
        warnings.warn(f"No overlapping frames found between cameras {main_camera} and {secondary_camera} for beta: {b}")
        if inliers_list is not None:
            inliers_list.append(0)
        return 0

    # Estimate the fundamental matrix
    F, mask = cv.findFundamentalMat(
        np.array([x for x, _, _ in correspondences]),
        np.array([y for _, y, _ in correspondences]),
        cv.RANSAC
    )

    inlier_ratio = np.sum(mask) / len(mask) if mask is not None else 0
    # logging.info(f"Cameras {main_camera}-{secondary_camera}, beta: {b:.3f}, inliers: {inlier_ratio:.4f}, correspondences: {len(correspondences)}")
    
    if inliers_list is not None:
        inliers_list.append(inlier_ratio)
        
    if return_f:
        return F, mask, correspondences
    return inlier_ratio, len(correspondences)
