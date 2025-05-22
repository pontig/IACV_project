import warnings
import concurrent
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import pandas as pd
import os
import logging
from datetime import datetime
import time
import sys
from concurrent.futures import ProcessPoolExecutor
from scipy.interpolate import make_interp_spline

# import matplotlib
# matplotlib.use('Agg')

from data_loader import load_dataframe
from global_fn import *
from plotter import *
from bundle_adjustment import bundle_adjust_camera_pose

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Create logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# File handler - for logging to file
file_handler = logging.FileHandler(
    f"logs/beta_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# Console handler - for logging to stdout
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# Add both handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)
DATASET_NO = 1

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
    logging.info(f"Cameras {main_camera}-{secondary_camera}, beta: {b:.3f}, inliers: {inlier_ratio:.4f}, correspondences: {len(correspondences)}")
    
    if inliers_list is not None:
        inliers_list.append(inlier_ratio)
        
    if return_f:
        return F, mask, correspondences
    return inlier_ratio, len(correspondences)

def plot_combined_results(beta_values_coarse, inliers_coarse, best_beta_coarse, max_inliers_coarse,
                          beta_values_fine, inliers_fine, best_beta_fine, max_inliers_fine,
                          beta_values_finer, inliers_finer, best_beta_finer, max_inliers_finer,
                          beta_values_finest, inliers_finest, best_beta_finest, max_inliers_finest,
                          main_camera, secondary_camera, dataset_no):
    """
    Plot combined results of beta search
    
    Parameters:
    -----------
    (same as plot_refinement_process)
    """
    figsize = (28, 15)
    
    # Combined visualization in one plot with different colors and transparency
    plt.figure(figsize=figsize)
    plt.plot(beta_values_coarse, inliers_coarse, 'b-', alpha=0.5, linewidth=1, label='Coarse')
    plt.plot(beta_values_fine, inliers_fine, 'r-', alpha=0.6, linewidth=1.5, label='Fine')
    plt.plot(beta_values_finer, inliers_finer, 'g-', alpha=0.7, linewidth=2, label='Finer')
    plt.plot(beta_values_finest, inliers_finest, 'm-', alpha=1.0, linewidth=2.5, label='Finest')

    # Mark the best beta for each refinement level
    plt.scatter(best_beta_coarse, max_inliers_coarse, c='blue', marker='*', s=200, 
               label=f'Best β (Coarse)={best_beta_coarse}')
    plt.scatter(best_beta_fine, max_inliers_fine, c='red', marker='*', s=200, 
               label=f'Best β (Fine)={best_beta_fine}')
    plt.scatter(best_beta_finer, max_inliers_finer, c='green', marker='*', s=200, 
               label=f'Best β (Finer)={best_beta_finer}')
    plt.scatter(best_beta_finest, max_inliers_finest, c='magenta', marker='*', s=300, 
               label=f'Best β (Finest)={best_beta_finest}')

    plt.title(f'Multi-level Beta Search Refinement (Cameras {main_camera}-{secondary_camera})', fontsize=18)
    plt.xlabel('Beta', fontsize=14)
    plt.ylabel('Inlier Ratio', fontsize=14)
    plt.ylim(-0.1, 1.1)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    # Create plots directory if it doesn't exist
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/inliers_vs_beta_combined_cam{main_camera}-{secondary_camera}_ds{dataset_no}.png", dpi=300)
    plt.close()

def search_optimal_beta(frames, splines, camera_info, main_camera, secondary_camera, dataset_no, beta_shift):
    """
    Perform multi-level search to find the optimal beta between two cameras
    
    Parameters:
    -----------
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
    dataset_no : int
        Dataset number
    beta_shift : int, optional
        Initial beta search range, defaults to 4000
        
    Returns:
    --------
    float
        Optimal beta value
    float
        Maximum inlier ratio
    """
    logging.info(f"===== Starting beta search for cameras {main_camera}-{secondary_camera} =====")
    print(f"===== Starting beta search for cameras {main_camera}-{secondary_camera} =====")
    inliers_coarse = []
    inliers_fine = []
    inliers_finer = []
    inliers_finest = []
    
    coarse_step = 2
    fine_step = .5
    finer_step = 0.1
    finest_step = 0.01
    
    # Step 1: Coarse Search
    logging.info(f"Cameras {main_camera}-{secondary_camera}: Starting coarse search")
    beta_coarse = np.arange(-beta_shift, beta_shift, coarse_step)
    beta_values_coarse = []

    for b in beta_coarse:
        evaluate_beta(b, frames, splines, camera_info, main_camera, secondary_camera, inliers_coarse)
        beta_values_coarse.append(b)
        
    best_beta_coarse = beta_coarse[np.argmax(inliers_coarse)]
    max_inliers_coarse = np.max(inliers_coarse)
    logging.info(f"Cameras {main_camera}-{secondary_camera}: End of coarse search")
    logging.info(f"Cameras {main_camera}-{secondary_camera}: Best beta (coarse): {best_beta_coarse}, inliers: {max_inliers_coarse:.4f}")

    # Step 2: Fine Search
    logging.info(f"Cameras {main_camera}-{secondary_camera}: Starting fine search")
    beta_fine = np.arange(best_beta_coarse - 100, best_beta_coarse + 100, fine_step)
    beta_values_fine = []

    for b in beta_fine:
        evaluate_beta(b, frames, splines, camera_info, main_camera, secondary_camera, inliers_fine)
        beta_values_fine.append(b)

    best_beta_fine = beta_fine[np.argmax(inliers_fine)]
    max_inliers_fine = np.max(inliers_fine)
    logging.info(f"Cameras {main_camera}-{secondary_camera}: End of fine search")
    logging.info(f"Cameras {main_camera}-{secondary_camera}: Best beta (fine): {best_beta_fine}, inliers: {max_inliers_fine:.4f}")

    # Step 3: Finer Search
    logging.info(f"Cameras {main_camera}-{secondary_camera}: Starting finer search")
    beta_finer = np.arange(best_beta_fine - 10, best_beta_fine + 10, finer_step)
    beta_values_finer = []

    for b in beta_finer:
        evaluate_beta(b, frames, splines, camera_info, main_camera, secondary_camera, inliers_finer)
        beta_values_finer.append(b)

    best_beta_finer = beta_finer[np.argmax(inliers_finer)]
    max_inliers_finer = np.max(inliers_finer)
    logging.info(f"Cameras {main_camera}-{secondary_camera}: End of finer search")
    logging.info(f"Cameras {main_camera}-{secondary_camera}: Best beta (finer): {best_beta_finer}, inliers: {max_inliers_finer:.4f}")

    # Step 4: Finest Search
    logging.info(f"Cameras {main_camera}-{secondary_camera}: Starting finest search")
    beta_finest = np.arange(best_beta_finer - 1, best_beta_finer + 1, finest_step)
    beta_values_finest = []
    num_inliers = []

    for b in beta_finest:
        _, ni = evaluate_beta(b, frames, splines, camera_info, main_camera, secondary_camera, inliers_finest)
        beta_values_finest.append(b)
        num_inliers.append(ni)

    best_beta_finest = beta_finest[np.argmax(inliers_finest)]
    max_inliers_finest = np.max(inliers_finest)
    max_inliers_finest_abs_n = num_inliers[np.argmax(inliers_finest)]
    logging.info(f"Cameras {main_camera}-{secondary_camera}: End of finest search")
    logging.info(f"Cameras {main_camera}-{secondary_camera}: Best beta (finest): {best_beta_finest}, inliers: {max_inliers_finest:.4f}, that are {max_inliers_finest_abs_n} inliers")

    # Final result
    logging.info(f"Cameras {main_camera}-{secondary_camera}: Final Result")
    logging.info(f"Cameras {main_camera}-{secondary_camera}: Best beta: {best_beta_finest}, inliers: {max_inliers_finest:.4f}, that are {max_inliers_finest_abs_n} inliers")
    
    # Generate plots
    plot_refinement_process(
        beta_values_coarse, inliers_coarse, best_beta_coarse, max_inliers_coarse,
        beta_values_fine, inliers_fine, best_beta_fine, max_inliers_fine,
        beta_values_finer, inliers_finer, best_beta_finer, max_inliers_finer,
        beta_values_finest, inliers_finest, best_beta_finest, max_inliers_finest,
        main_camera, secondary_camera, dataset_no
    )
    
    plot_combined_results(
        beta_values_coarse, inliers_coarse, best_beta_coarse, max_inliers_coarse,
        beta_values_fine, inliers_fine, best_beta_fine, max_inliers_fine,
        beta_values_finer, inliers_finer, best_beta_finer, max_inliers_finer,
        beta_values_finest, inliers_finest, best_beta_finest, max_inliers_finest,
        main_camera, secondary_camera, dataset_no
    )
    
    return best_beta_finest, max_inliers_finest, max_inliers_finest_abs_n

def process_camera_pair(main_camera, secondary_camera, df, splines, camera_info, dataset_no, beta_shift):
    """Process a single camera pair to find optimal beta"""
    if main_camera == secondary_camera:
        return None

    logging.info(f"Processing camera pair: main={main_camera}, secondary={secondary_camera}")
    
    # Get frames from secondary camera with valid detections
    frames = df[df['cam_id'] == secondary_camera][['frame_id', 'detection_x', 'detection_y']].values
    frames = [frame for frame in frames if frame[1] != 0.0 and frame[2] != 0.0]
    
    if not frames:
        logging.warning(f"No valid frames found for camera {secondary_camera}")
        return None
        
    # Find optimal beta
    best_beta, max_inliers, max_inliers_abs = search_optimal_beta(
        frames, splines, camera_info, 
        main_camera, secondary_camera, 
        dataset_no, beta_shift
    )
    
    # Store result
    return {
        "key": f"{main_camera}-{secondary_camera}",
        "main_camera": main_camera,
        "secondary_camera": secondary_camera,
        "beta": best_beta,
        "inlier_ratio": max_inliers,
        "inlier_count": max_inliers_abs
    }

def first_beta_search(dataset_no=DATASET_NO, beta_shift=1):
    """
    Main function to find optimal beta values for all camera combinations
    """
    start_time = time.time()
    logging.info(f"Starting beta search process for dataset {dataset_no}")
        
    # Results dictionary to store all beta values
    results = {}
    
    # Create pairs of camera indices to process
    camera_pairs = [(m, s) for m in range(len(cameras)) for s in range(len(cameras)) if m != s]
    
    # Use ProcessPoolExecutor for parallel processing
    max_workers = min(10, os.cpu_count() or 4)  # Use min of 10 or available CPU cores
    logging.info(f"Using ProcessPoolExecutor with {max_workers} workers")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_pair = {
            executor.submit(
                process_camera_pair, 
                m, s, 
                df, splines, camera_info, dataset_no, beta_shift
            ): (m, s) for m, s in camera_pairs
        }
        
        for future in concurrent.futures.as_completed(future_to_pair):
            pair = future_to_pair[future]
            try:
                result = future.result()
                if result:
                    results[result["key"]] = {
                        "main_camera": result["main_camera"],
                        "secondary_camera": result["secondary_camera"],
                        "beta": result["beta"],
                        "inlier_ratio": result["inlier_ratio"],
                        "inlier_count": result["inlier_count"]
                    }
                    logging.info(f"Completed pair {pair[0]}-{pair[1]}")
            except Exception as exc:
                logging.error(f"Pair {pair[0]}-{pair[1]} generated an exception: {exc}")
                logging.error(f"Exception details: {str(exc)}")
                import traceback
                logging.error(traceback.format_exc())
    
    # Save all results to a separate log file
    result_log_path = f"logs/beta_results_dataset{dataset_no}.csv"
    with open(result_log_path, 'w') as f:
        f.write("main_camera,secondary_camera,beta,inlier_ratio,num_inliers\n")
        for key, value in results.items():
            f.write(f"{value['main_camera']},{value['secondary_camera']},{value['beta']:.4f},{value['inlier_ratio']:.4f},{value['inlier_count']}\n")
    
    logging.info(f"Results saved to {result_log_path}")
    logging.info(f"Total execution time: {(time.time() - start_time)/60:.2f} minutes")
    
    return results

def to_normalized_camera_coord(pts, K, distcoeff):
    """
    Convert points from image coordinates to normalized camera coordinates.
    
    Parameters:
    -----------
    pts : ndarray
        Points in image coordinates (N x 2)
    K : ndarray
        Camera matrix
    distcoeff : ndarray
        Distortion coefficients
    
    Returns:
    --------
    ndarray
        Points in normalized camera coordinates (N x 2)
    """
    pts_normalized = cv.undistortPoints(pts, K, distcoeff)
    
    # Convert from homogeneous coordinates to 2D
    pts_normalized = pts_normalized.reshape(-1, 2)
    
    return pts_normalized
  
def generate_3d_splines(points_3d_with_timestamps, already_existing_splines):
    """
    Generate 3D splines from 3D points with timestamps.
    
    Parameters:
    -----------
    points_3d_with_timestamps : list
        List of 3D points with timestamps
    already_existing_splines : list
        List of existing splines
    
    Returns:
    --------
    list
        List of generated splines
    """
    splines_3d_points = []
    this_spline = []
    
    if len(points_3d_with_timestamps) == 0:
        logging.warning("No points to generate splines from.")
        return already_existing_splines

    for i, current_point in enumerate(points_3d_with_timestamps[:-1]):
        next_point = points_3d_with_timestamps[i + 1]
        this_spline.append(points_3d_with_timestamps[i])

        # Check if the time difference exceeds the threshold
        if next_point[3] - current_point[3] >= 5:
            splines_3d_points.append(this_spline)
            this_spline = []

    # Append the last triangulated point to the current spline
    this_spline.append(points_3d_with_timestamps[-1])
    splines_3d_points.append(this_spline)

    # Filter out any empty splines or splines with less than 4 points
    splines_3d_points = [spline for spline in splines_3d_points if spline and len(spline) >= 4]
    
    # Generate 3D splines
    splines_3d = []
    for spline_points in splines_3d_points:
        spline_points = np.array(spline_points)
        ts = spline_points[:, 3]  # Use timestamps from triangulated points
        
        if len(ts) < 4:
            continue  # Skip if not enough points for spline fitting

        spline_x = make_interp_spline(ts, spline_points[:, 0], k=3)
        spline_y = make_interp_spline(ts, spline_points[:, 1], k=3)
        spline_z = make_interp_spline(ts, spline_points[:, 2], k=3)
        
        splines_3d.append((spline_x, spline_y, spline_z, ts))
        
    # Append the new splines to the existing ones only if the timestamps are not overlapping with any existing spline
    for new_spline in splines_3d:
        new_ts = new_spline[3]
        if not any(np.any(np.isin(new_ts, existing_spline[3])) for existing_spline in already_existing_splines):
            already_existing_splines.append(new_spline)
    return already_existing_splines
        
def merge_and_smooth_splines(spline_dicts, tpns_to_add_to_3d_with_timestamps, blend_window=5.0, smoothing_factor=0.1):
    """
    Merge splines with new points and smooth the junctions.
    
    Parameters:
    -----------
    spline_dicts : list
        List of dictionaries containing spline components
    tpns_to_add_to_3d_with_timestamps : ndarray
        New 3D points with timestamps to incorporate
    blend_window : float
        Time window for blending/merging splines
    smoothing_factor : float
        Factor controlling smoothness at junctions (0-1)
        
    Returns:
    --------
    list
        Updated list of spline dictionaries
    """
    # Sort new points by timestamp for efficient processing
    tpns_sorted = sorted(tpns_to_add_to_3d_with_timestamps, key=lambda pt: pt[3])
    extended_timestamps = set()
    
    # First pass: Extend existing splines with nearby points
    for pt_x, pt_y, pt_z, pt_ts in tpns_sorted:
        if pt_ts in extended_timestamps:
            continue
            
        # Find closest spline by timestamp
        closest_spline_idx = -1
        min_dist = float('inf')
        extend_direction = None  # 'forward' or 'backward'
        
        for i, spline in enumerate(spline_dicts):
            ts_min, ts_max = np.min(spline['ts']), np.max(spline['ts'])
            
            # Check if point can extend forward
            if 0 < pt_ts - ts_max < blend_window:
                dist = pt_ts - ts_max
                if dist < min_dist:
                    min_dist = dist
                    closest_spline_idx = i
                    extend_direction = 'forward'
            
            # Check if point can extend backward
            elif 0 < ts_min - pt_ts < blend_window:
                dist = ts_min - pt_ts
                if dist < min_dist:
                    min_dist = dist
                    closest_spline_idx = i
                    extend_direction = 'backward'
        
        # If a spline to extend was found
        if closest_spline_idx >= 0:
            spline = spline_dicts[closest_spline_idx]
            ts_min, ts_max = np.min(spline['ts']), np.max(spline['ts'])
            
            if extend_direction == 'forward':
                # Create a smooth transition between the spline end and the new point
                # Extract endpoint values and derivatives
                end_ts = ts_max
                end_x = float(spline['spline_x'](end_ts))
                end_y = float(spline['spline_y'](end_ts))
                end_z = float(spline['spline_z'](end_ts))
                
                # Calculate derivatives at the endpoint
                dx_dt = float(spline['spline_x'].derivative()(end_ts))
                dy_dt = float(spline['spline_y'].derivative()(end_ts))
                dz_dt = float(spline['spline_z'].derivative()(end_ts))
                
                # Predict where the trajectory should be at pt_ts
                delta_t = pt_ts - end_ts
                pred_x = end_x + dx_dt * delta_t
                pred_y = end_y + dy_dt * delta_t
                pred_z = end_z + dz_dt * delta_t
                
                # Blend actual point with prediction for smoothness
                smooth_x = smoothing_factor * pred_x + (1 - smoothing_factor) * pt_x
                smooth_y = smoothing_factor * pred_y + (1 - smoothing_factor) * pt_y
                smooth_z = smoothing_factor * pred_z + (1 - smoothing_factor) * pt_z
                
                # Append smoothed point
                ts_new = np.append(spline['ts'], pt_ts)
                x_new = np.append(spline['spline_x'](spline['ts']), smooth_x)
                y_new = np.append(spline['spline_y'](spline['ts']), smooth_y)
                z_new = np.append(spline['spline_z'](spline['ts']), smooth_z)
                
            else:  # backward extension
                # Create a smooth transition between the new point and the spline start
                start_ts = ts_min
                start_x = float(spline['spline_x'](start_ts))
                start_y = float(spline['spline_y'](start_ts))
                start_z = float(spline['spline_z'](start_ts))
                
                # Calculate derivatives at the start point
                dx_dt = float(spline['spline_x'].derivative()(start_ts))
                dy_dt = float(spline['spline_y'].derivative()(start_ts))
                dz_dt = float(spline['spline_z'].derivative()(start_ts))
                
                # Predict where the trajectory should have been at pt_ts
                delta_t = pt_ts - start_ts
                pred_x = start_x - dx_dt * (-delta_t)  # Note the negative delta_t
                pred_y = start_y - dy_dt * (-delta_t)
                pred_z = start_z - dz_dt * (-delta_t)
                
                # Blend actual point with prediction for smoothness
                smooth_x = smoothing_factor * pred_x + (1 - smoothing_factor) * pt_x
                smooth_y = smoothing_factor * pred_y + (1 - smoothing_factor) * pt_y
                smooth_z = smoothing_factor * pred_z + (1 - smoothing_factor) * pt_z
                
                # Prepend smoothed point
                ts_new = np.append(pt_ts, spline['ts'])
                x_new = np.append(smooth_x, spline['spline_x'](spline['ts']))
                y_new = np.append(smooth_y, spline['spline_y'](spline['ts']))
                z_new = np.append(smooth_z, spline['spline_z'](spline['ts']))
            
            # Ensure data is sorted by timestamp
            idx = np.argsort(ts_new)
            ts_sorted = ts_new[idx]
            x_sorted = x_new[idx]
            y_sorted = y_new[idx]
            z_sorted = z_new[idx]
            
            # Fit new spline with smoothing
            if len(ts_sorted) >= 4:
                s = 0.1  # Smoothing parameter for spline fitting
                spline_dicts[closest_spline_idx]['spline_x'] = make_interp_spline(ts_sorted, x_sorted, k=3)
                spline_dicts[closest_spline_idx]['spline_y'] = make_interp_spline(ts_sorted, y_sorted, k=3)
                spline_dicts[closest_spline_idx]['spline_z'] = make_interp_spline(ts_sorted, z_sorted, k=3)
                spline_dicts[closest_spline_idx]['ts'] = ts_sorted
                extended_timestamps.add(pt_ts)
    
    # Second pass: Try merging splines when a point bridges them
    merged_splines = True
    while merged_splines:
        merged_splines = False
        
        # Create a mapping of timestamps to spline indices for efficient lookup
        ts_to_spline = {}
        for i, spline in enumerate(spline_dicts):
            min_ts, max_ts = np.min(spline['ts']), np.max(spline['ts'])
            ts_to_spline[(min_ts, max_ts)] = i
        
        # For each point, check if it can merge two splines
        for pt_x, pt_y, pt_z, pt_ts in tpns_sorted:
            if pt_ts in extended_timestamps:
                continue
                
            # Find splines that could be merged using this point
            potential_merges = []
            for (min_ts1, max_ts1), i in ts_to_spline.items():
                for (min_ts2, max_ts2), j in ts_to_spline.items():
                    if i >= j:  # Avoid checking the same pair twice
                        continue
                        
                    # Check if point falls between splines within merge window
                    if (0 < pt_ts - max_ts1 < blend_window and 
                        0 < min_ts2 - pt_ts < blend_window):
                        potential_merges.append((i, j, abs(pt_ts - max_ts1) + abs(min_ts2 - pt_ts)))
            
            # Sort potential merges by total time gap
            potential_merges.sort(key=lambda x: x[2])
            
            if potential_merges:
                i, j, _ = potential_merges[0]
                s1, s2 = spline_dicts[i], spline_dicts[j]
                
                # Create overlap region for smooth transition
                end_ts1 = np.max(s1['ts'])
                start_ts2 = np.min(s2['ts'])
                
                # Extract endpoint derivatives
                dx1_dt = float(s1['spline_x'].derivative()(end_ts1))
                dy1_dt = float(s1['spline_y'].derivative()(end_ts1))
                dz1_dt = float(s1['spline_z'].derivative()(end_ts1))
                
                dx2_dt = float(s2['spline_x'].derivative()(start_ts2))
                dy2_dt = float(s2['spline_y'].derivative()(start_ts2))
                dz2_dt = float(s2['spline_z'].derivative()(start_ts2))
                
                # Create smooth interpolation at the junction
                # Hermite interpolation to blend the two splines
                def hermite_blend(t, p0, p1, m0, m1):
                    t2 = t * t
                    t3 = t2 * t
                    h00 = 2 * t3 - 3 * t2 + 1  # Hermite basis function
                    h10 = t3 - 2 * t2 + t
                    h01 = -2 * t3 + 3 * t2
                    h11 = t3 - t2
                    return h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1
                
                # Create transition points
                n_blend = 5  # Number of blend points
                blend_ts = np.linspace(end_ts1, start_ts2, n_blend + 2)[1:-1]  # Exclude endpoints
                
                end_x1 = float(s1['spline_x'](end_ts1))
                end_y1 = float(s1['spline_y'](end_ts1))
                end_z1 = float(s1['spline_z'](end_ts1))
                
                start_x2 = float(s2['spline_x'](start_ts2))
                start_y2 = float(s2['spline_y'](start_ts2))
                start_z2 = float(s2['spline_z'](start_ts2))
                
                # Generate blend points
                blend_x = []
                blend_y = []
                blend_z = []
                
                for t in blend_ts:
                    # Normalize t for hermite interpolation
                    t_norm = (t - end_ts1) / (start_ts2 - end_ts1)
                    
                    # Scale derivatives by time difference
                    dt = start_ts2 - end_ts1
                    m0_x = dx1_dt * dt
                    m0_y = dy1_dt * dt
                    m0_z = dz1_dt * dt
                    m1_x = dx2_dt * dt
                    m1_y = dy2_dt * dt
                    m1_z = dz2_dt * dt
                    
                    # Apply hermite blending
                    x_t = hermite_blend(t_norm, end_x1, start_x2, m0_x, m1_x)
                    y_t = hermite_blend(t_norm, end_y1, start_y2, m0_y, m1_y)
                    z_t = hermite_blend(t_norm, end_z1, start_z2, m0_z, m1_z)
                    
                    blend_x.append(x_t)
                    blend_y.append(y_t)
                    blend_z.append(z_t)
                
                # Create merged spline with smooth junction
                ts_merged = np.concatenate((s1['ts'], blend_ts, s2['ts']))
                x_merged = np.concatenate((s1['spline_x'](s1['ts']), blend_x, s2['spline_x'](s2['ts'])))
                y_merged = np.concatenate((s1['spline_y'](s1['ts']), blend_y, s2['spline_y'](s2['ts'])))
                z_merged = np.concatenate((s1['spline_z'](s1['ts']), blend_z, s2['spline_z'](s2['ts'])))
                
                # Sort by timestamp
                idx = np.argsort(ts_merged)
                ts_sorted = ts_merged[idx]
                x_sorted = x_merged[idx]
                y_sorted = y_merged[idx]
                z_sorted = z_merged[idx]
                
                # Create new spline with additional smoothing
                s = 0.1  # Smoothing parameter
                new_spline = {
                    'spline_x': make_interp_spline(ts_sorted, x_sorted, k=3),
                    'spline_y': make_interp_spline(ts_sorted, y_sorted, k=3),
                    'spline_z': make_interp_spline(ts_sorted, z_sorted, k=3),
                    'ts': ts_sorted
                }
                
                # Remove old splines and add the merged one
                new_spline_list = [s for k, s in enumerate(spline_dicts) if k != i and k != j]
                new_spline_list.append(new_spline)
                spline_dicts = new_spline_list
                extended_timestamps.add(pt_ts)
                merged_splines = True
                break  # Need to rebuild ts_to_spline mapping
    
    # Third pass: Create new splines for remaining points
    remaining_points = []
    for pt in tpns_sorted:
        if pt[3] not in extended_timestamps:
            remaining_points.append(pt)
    
    # Group remaining points by proximity in time
    if remaining_points:
        remaining_points.sort(key=lambda pt: pt[3])
        point_groups = []
        current_group = [remaining_points[0]]
        
        for i in range(1, len(remaining_points)):
            if remaining_points[i][3] - remaining_points[i-1][3] < blend_window:
                current_group.append(remaining_points[i])
            else:
                if len(current_group) >= 4:  # Need at least 4 points for cubic spline
                    point_groups.append(current_group)
                current_group = [remaining_points[i]]
        
        if len(current_group) >= 4:
            point_groups.append(current_group)
        
        # Create new splines for each group
        for group in point_groups:
            group_array = np.array(group)
            ts = group_array[:, 3]
            
            # Apply smoothing directly during spline creation
            s = 0.1  # Smoothing parameter
            new_spline = {
                'spline_x': make_interp_spline(ts, group_array[:, 0], k=3),
                'spline_y': make_interp_spline(ts, group_array[:, 1], k=3),
                'spline_z': make_interp_spline(ts, group_array[:, 2], k=3),
                'ts': ts
            }
            spline_dicts.append(new_spline)
    
    return spline_dicts
  
if __name__ == "__main__":
    
    
    # ============== Step 0: Load data and search for betas ==============
    
    camera_poses = []
    
    # Load camera info
    logging.info("Loading camera info")
    with open(f"drone-tracking-datasets/dataset{DATASET_NO}/cameras.txt", 'r') as f:
        cameras = f.read().strip().split()    
    cameras = cameras[2::3]
    logging.info(f"Loaded {len(cameras)} cameras")
    camera_poses = [{"cam_id": i, "R": None, "t": None, 'b': None} for i in range(len(cameras))]
    
    # Load dataframe and splines
    logging.info("Loading dataframe and splines")
    df, splines, contiguous, camera_info = load_dataframe(cameras, DATASET_NO)
    
    if not os.path.exists(f"beta_results_dataset{DATASET_NO}.csv"):
        logging.info("Beta results file not found, starting beta search")
        first_beta_search(DATASET_NO, beta_shift=2000)
    
    logging.info("Loading beta results")  
    betas_df = pd.read_csv(f"beta_results_dataset{DATASET_NO}.csv")
    betas_df = betas_df.sort_values(by='inlier_ratio', ascending=False)
    # Create a 2D array with all betas from betas_df
    beta_array = betas_df.pivot(index='secondary_camera', columns='main_camera', values='beta').fillna(0).to_numpy() # [sec][prim]
    
    # ============== Step 1: Generate first 3D point and splines with best beta ==============
    
    best_beta = betas_df.iloc[0]['beta']
    main_camera = betas_df.iloc[0]['main_camera']
    secondary_camera = betas_df.iloc[0]['secondary_camera']
    frames = df[df['cam_id'] == int(secondary_camera)][['frame_id', 'detection_x', 'detection_y']].values
    frames = [frame for frame in frames if frame[1] != 0.0 and frame[2] != 0.0]
    
    first_main_camera = main_camera
    first_beta = best_beta
    
    logging.info(f"Best beta: {best_beta}, main camera: {main_camera}, secondary camera: {secondary_camera}, inliers: {betas_df.iloc[0]['inlier_ratio']:.4f}")
    
    F, mask, correspondences = evaluate_beta(best_beta, frames, splines, camera_info, int(main_camera), int(secondary_camera), return_f=True)
    E = camera_info[int(main_camera)].K_matrix.T @ F @ camera_info[int(secondary_camera)].K_matrix
    
    main_pts_normalized = to_normalized_camera_coord(np.array([x for x, _, _ in correspondences]), camera_info[int(main_camera)].K_matrix, camera_info[int(main_camera)].distCoeff)
    secondary_pts_normalized = to_normalized_camera_coord(np.array([y for _, y, _ in correspondences]), camera_info[int(secondary_camera)].K_matrix, camera_info[int(secondary_camera)].distCoeff)
    
    _, R, t, mask, triangulated_points_first_step = cv.recoverPose(E, main_pts_normalized, secondary_pts_normalized, cameraMatrix=np.eye(3), distanceThresh=100)
    
    camera_poses[int(main_camera)] = {
        "cam_id": int(main_camera),
        "R": np.eye(3),
        "t": np.zeros((3, 1)),
    }
    camera_poses[int(secondary_camera)] = {
        "cam_id": int(secondary_camera),
        "R": R,
        "t": t,
    }
    
    triangulated_points_first_step /= triangulated_points_first_step[3]
    triangulated_points_first_step = triangulated_points_first_step[:3]
    triangulated_points_first_step = triangulated_points_first_step.T
    
    triangulated_points_first_step_with_timestamps = np.column_stack((triangulated_points_first_step, np.array([x[2] for x in correspondences])))
    
    # # Reproject 3D points onto both cameras
    # logging.info("Reprojecting 3D points onto both cameras")

    # # Main camera reprojection
    # main_reprojected_points, _ = cv.projectPoints(
    #     triangulated_points_first_step, np.zeros((3, 1)), np.zeros((3, 1)), 
    #     camera_info[int(main_camera)].K_matrix, camera_info[int(main_camera)].distCoeff
    # )
    # main_reprojected_points = main_reprojected_points.reshape(-1, 2)

    # # Secondary camera reprojection
    # secondary_reprojected_points, _ = cv.projectPoints(
    #     triangulated_points_first_step, R, t, 
    #     camera_info[int(secondary_camera)].K_matrix, camera_info[int(secondary_camera)].distCoeff
    # )
    # secondary_reprojected_points = secondary_reprojected_points.reshape(-1, 2)
    
    # Compute first 3D splines
    splines_3d = generate_3d_splines(triangulated_points_first_step_with_timestamps, [])
        
    # plot_triangulated_points(triangulated_points, main_camera, secondary_camera)

    # plot_reprojection_analysis(
    #     triangulated_points_first_step, np.array([y for _, y, _ in correspondences]),
    #     R, t,
    #     camera_info[int(secondary_camera)], int(secondary_camera)
    # )
    # plot_3d_splines(triangulated_points_first_step, correspondences, main_camera, secondary_camera)


    # ============== Step 2.1: Add cameras in decreasing order of inlier ratio ==============
    cameras_processed = [int(main_camera), int(secondary_camera)]
    cameras_skipped = []
    main_camera_index = 0
    old_main_camera = int(main_camera)
    stop = False
    
    while len(cameras_processed) < len(cameras):
        
        next_in_line = 0
        while True:
            # Filter betas_df for pairs where the main camera is in cameras_processed and the secondary camera is not
            betas_here_df = betas_df[
                (betas_df['main_camera'].isin(cameras_processed)) & 
                (~betas_df['secondary_camera'].isin(cameras_processed))
            ]
            betas_here_df = betas_here_df.sort_values(by='inlier_ratio', ascending=False)
            
            # print(betas_here_df)
            
            if betas_here_df.empty:
                logging.warning("No valid camera pairs found with the current conditions.")
                break
            
            # Select the next best pair
            new_camera = betas_here_df.iloc[next_in_line]['secondary_camera']
            nth_best_beta = betas_here_df.iloc[next_in_line]['beta']
            inlier_ratio = betas_here_df.iloc[next_in_line]['inlier_ratio']
            main_camera = betas_here_df.iloc[next_in_line]['main_camera']
            num_inliers = betas_here_df.iloc[next_in_line]['num_inliers'] if 'num_inliers' in betas_here_df.columns else np.inf
            next_in_line += 1
            
            if inlier_ratio < 0.5 or num_inliers < 200:
                logging.warning(f"Low inlier ratio or insufficient support for camera pair {main_camera}-{new_camera}: {inlier_ratio:.4f}, num_inliers: {num_inliers}")
                if next_in_line >= len(betas_here_df):
                    logging.warning(f"Unable to find a valid camera pair with sufficient inlier ratio. skipping camera")
                    cameras_skipped.append(new_camera)
                    stop = True
                    break
                continue
            
            if new_camera not in cameras_processed:
                if main_camera != old_main_camera:
                    logging.info(f"Switching main camera from {old_main_camera} to {main_camera}, adjusting 3d splines timestamps...")
                    beta_main_old_main_camera = betas_df[(betas_df['main_camera'] == main_camera) & (betas_df['secondary_camera'] == old_main_camera)]['beta'].values[0]
                    new_splines_3d = []
                    for spline_x, spline_y, spline_z, tss in splines_3d:
                        # Adjust timestamps for the new main camera
                        old_tss = tss.copy()
                        new_tss = compute_global_time(old_tss, camera_info[int(main_camera)].fps/camera_info[int(old_main_camera)].fps, beta_main_old_main_camera) # TODO: maybe here
                        tss[:] = new_tss
                        spline_x = make_interp_spline(tss, spline_x(old_tss), k=3)
                        spline_y = make_interp_spline(tss, spline_y(old_tss), k=3)
                        spline_z = make_interp_spline(tss, spline_z(old_tss), k=3)
                        new_splines_3d.append((spline_x, spline_y, spline_z, tss))
                        
                    splines_3d = new_splines_3d  
                    old_main_camera = main_camera
                                        
                break
            
        if stop:
            break
            
        logging.info(f"{len(cameras_processed)}-th best beta: {nth_best_beta}, main camera: {main_camera}, secondary camera: {new_camera}, inliers: {inlier_ratio:.4f}")
        
        # Load the frames for the nth best beta
        frames = df[df['cam_id'] == int(new_camera)][['frame_id', 'detection_x', 'detection_y']].values
        frames = [frame for frame in frames if frame[1] != 0.0 and frame[2] != 0.0]
        frames = np.array(frames)
        frames_ids = frames[:, 0]
        global_ts = compute_global_time(frames_ids, camera_info[int(main_camera)].fps/camera_info[int(new_camera)].fps, nth_best_beta)
        frames = np.column_stack((global_ts, frames[:, 1], frames[:, 2]))
        
        # Compute 3D - 2D correspondences
        correspondences_2 = [] # List of (spline_x1, spline_y1, spline_z1), (x2, y2), global_ts
        for frame in frames:
            global_ts = frame[0]
            for spline_x, spline_y, spline_z, tss in splines_3d:
                if np.min(tss) <= global_ts <= np.max(tss):
                    x1 = float(spline_x(global_ts))
                    y1 = float(spline_y(global_ts))
                    z1 = float(spline_z(global_ts))
                    correspondences_2.append((
                        (x1, y1, z1),
                        (frame[1], frame[2]),
                        global_ts
                    ))
                    break
        if not correspondences_2:
            warnings.warn(f"No overlapping frames found between cameras {main_camera} and {new_camera} for beta: {nth_best_beta}")
            
        # Perform pnp to locate camera pose
        retval, rvec, tvec, inliers = cv.solvePnPRansac(
            np.array([x for x, _, _ in correspondences_2], dtype=np.float32),
            np.array([y for _, y, _ in correspondences_2], dtype=np.float32),
            camera_info[int(new_camera)].K_matrix,
            camera_info[int(new_camera)].distCoeff,
            confidence=.999
        )
        
        camera_poses[int(new_camera)] = {
            "cam_id": int(new_camera),
            "R": cv.Rodrigues(rvec)[0],
            "t": tvec,
        }
        
        # ============== Step 2.2: Extend 3D splines with new camera detections ==============
        
        frames = np.column_stack((frames_ids, frames[:, 1], frames[:, 2])) # to restore original frame ids
        F, mask, correspondences = evaluate_beta(nth_best_beta, frames, splines, camera_info, int(main_camera), int(new_camera), return_f=True)
        
        P1 = camera_info[int(main_camera)].K_matrix @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P3 = camera_info[int(new_camera)].K_matrix @ np.hstack((cv.Rodrigues(rvec)[0], tvec.reshape(-1, 1)))
        
        triangulated_points_nth_step = cv.triangulatePoints(P1, P3,
            np.array([x for x, _, _ in correspondences], dtype=np.float32).T,
            np.array([y for _, y, _ in correspondences], dtype=np.float32).T
        )
        
        triangulated_points_nth_step /= triangulated_points_nth_step[3]
        triangulated_points_nth_step = triangulated_points_nth_step[:3]
        triangulated_points_nth_step = triangulated_points_nth_step.T
        
        tpns_with_timestamps = np.column_stack((triangulated_points_nth_step, np.array([x[2] for x in correspondences])))
        
        # Filter out points with timestamps outside the range of the splines
        mask = []
        for point in tpns_with_timestamps:
            point_ts = point[3]
            in_range = any(np.min(tss) <= point_ts <= np.max(tss) for _, _, _, tss in splines_3d)
            mask.append(in_range)

        # Keep only points within the range
        tpns_to_add_to_3d = triangulated_points_nth_step[np.logical_not(mask)]
        tpns_to_add_to_3d_with_timestamps = tpns_with_timestamps[np.logical_not(mask)]        
        # plot_triangulated_points(triangulated_points_nth_step, main_camera, third_camera)
        
        # Plot splines in red and triangulated_points in blue
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot triangulated_points in blue
        ax.scatter(
            tpns_to_add_to_3d[:, 0], 
            tpns_to_add_to_3d[:, 1], 
            tpns_to_add_to_3d[:, 2], 
            c='blue', s=1, label=f"Triangulated Points after adding camera {new_camera}"
        )

        # Plot splines in red
        for spline_x, spline_y, spline_z, ts in splines_3d:
            ax.plot(
                spline_x(ts), 
                spline_y(ts), 
                spline_z(ts), 
                c='red'
            )

        # Set plot labels and title
        ax.set_title("3D Points and Splines Visualization")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()
        
        splines_3d = generate_3d_splines(tpns_to_add_to_3d_with_timestamps, splines_3d)
        
        # Convert splines to mutable structure
        spline_dicts = [{
            'spline_x': spline_x,
            'spline_y': spline_y,
            'spline_z': spline_z,
            'ts': ts.copy()
        } for spline_x, spline_y, spline_z, ts in splines_3d]

        spline_dicts = merge_and_smooth_splines(spline_dicts, tpns_to_add_to_3d_with_timestamps)
                
        # Convert back to original format
        splines_3d = [(s['spline_x'], s['spline_y'], s['spline_z'], s['ts']) for s in spline_dicts]
        
        for camera_pose in camera_poses:
            if camera_pose["R"] is None or camera_pose["cam_id"] != int(new_camera):
                continue
            i = camera_pose["cam_id"]
            frames = df[df['cam_id'] == i][['frame_id', 'detection_x', 'detection_y']].values
            frames = [frame for frame in frames if frame[1] != 0.0 and frame[2] != 0.0]
            frames = np.array(frames)
            frames_ids = frames[:, 0]
            bbbb = beta_array[int(i)][int(main_camera)]
            global_ts = compute_global_time(frames_ids, camera_info[int(main_camera)].fps/camera_info[i].fps, bbbb)
            frames = np.column_stack((global_ts, frames[:, 1], frames[:, 2]))

            ret = bundle_adjust_camera_pose(
                splines_3d,
                frames,
                camera_info[i].K_matrix,
                camera_info[i].distCoeff,
                cv.Rodrigues(camera_poses[i]["R"])[0],
                camera_poses[i]["t"]
            )
            
            camera_poses[i]["R"] = ret['R']
            camera_poses[i]["t"] = ret['tvec']
            
            
        # plot splines
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        for spline in splines_3d:
            spline_x, spline_y, spline_z, ts = spline
            ax.plot(spline_x(ts), spline_y(ts), spline_z(ts))
        for camera_pose in camera_poses:
            if camera_pose["R"] is None:
                continue
            cam_id = camera_pose["cam_id"]
            R = camera_pose["R"]
            t = camera_pose["t"].flatten()
            # ax.scatter(t[0], t[1], t[2], label=f"Camera {cam_id}", s=30)
        ax.set_title(f"3D Splines")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        # ax.legend()
        
            
        # print(f"Rotation vector: {rvec}")
        # print(f"Translation vector: {tvec}")
        # print(f"Inlier ratio: {len(inliers) / len(correspondences_2)}")
        # print(f"Number of inliers: {len(inliers)}")
        
        # plot_reprojection_analysis(
        #     splines_3d, df[df['cam_id'] == int(third_camera)][['detection_x', 'detection_y']].values,
        #     cv.Rodrigues(camera_poses[int(third_camera)]["R"])[0], camera_poses[int(third_camera)]["t"],
        #     camera_info[int(third_camera)], int(third_camera),
        #     title=f"Refined Reprojection Analysis for Camera {third_camera} with Beta {second_best_beta}"
        # )
        
        # plot_reprojection_analysis(
        #     splines_3d, df[df['cam_id'] == int(main_camera)][['detection_x', 'detection_y']].values,
        #     cv.Rodrigues(camera_poses[int(main_camera)]["R"])[0], camera_poses[int(main_camera)]["t"],
        #     camera_info[int(main_camera)], int(main_camera)
        # )
        
        # # Plot camera poses with arrows along the z-direction and highlight the xy-plane
        # fig = plt.figure(figsize=(15, 10))
        # ax = fig.add_subplot(111, projection='3d')

        # # Plot camera poses
        # for pose in camera_poses:
        #     if pose["R"] is None:
        #         continue
        #     cam_id = pose["cam_id"]
        #     R = pose["R"]
        #     t = pose["t"].flatten()
            
        #     # Plot camera position
        #     ax.scatter(t[0], t[1], t[2], label=f"Camera {cam_id}", s=100)

        #     # Plot arrow along the z-direction
        #     z_dir = R @ np.array([0, 0, 1])  # Transform z-direction
        #     ax.quiver(
        #         t[0], t[1], t[2], 
        #         z_dir[0], z_dir[1], z_dir[2], 
        #         length=1.0, color='blue', linewidth=2
        #     )

        # # Set plot labels and title
        # ax.set_title("Camera Poses with Z-Direction Arrows")
        # ax.set_xlabel("X")
        # ax.set_ylabel("Y")
        # ax.set_zlabel("Z")
        # ax.legend()
        
        cameras_processed.append(int(new_camera))
        
    # main_camera = first_main_camera
    # beta = first_beta

    for camera_pose in camera_poses:
        if camera_pose["R"] is None:
            continue
        i = camera_pose["cam_id"]
        frames = df[df['cam_id'] == i][['frame_id', 'detection_x', 'detection_y']].values
        plot_reprojection_analysis(
                    splines_3d, frames[:, 1:],
                    cv.Rodrigues(camera_poses[i]["R"])[0], camera_poses[i]["t"],
                    camera_info[i], i,
                    title=f"Refined Reprojection Analysis for Camera {i}"
        )
    
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    plane_size = 10
    xx, yy = np.meshgrid(np.linspace(-plane_size, plane_size, 10), np.linspace(-plane_size, plane_size, 10))
    zz = np.zeros_like(xx)
    ax.plot_surface(xx, yy, zz, alpha=0.2, color='gray')
        
    for spline in splines_3d:
        spline_x, spline_y, spline_z, ts = spline
        ax.plot(spline_x(ts), spline_y(ts), spline_z(ts))
    for camera_pose in camera_poses:
        if camera_pose["R"] is None:
            continue
        cam_id = camera_pose["cam_id"]
        R = camera_pose["R"]
        t = camera_pose["t"].flatten()
        ax.scatter(t[0], t[1], t[2], label=f"Camera {cam_id}", s=30)
    ax.set_title(f"3D Splines with Camera Poses")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    
    plt.show()