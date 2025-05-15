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
DATASET_NO = 4

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
    return inlier_ratio

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

def search_optimal_beta(frames, splines, camera_info, main_camera, secondary_camera, dataset_no, beta_shift=4000):
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

    for b in beta_finest:
        evaluate_beta(b, frames, splines, camera_info, main_camera, secondary_camera, inliers_finest)
        beta_values_finest.append(b)

    best_beta_finest = beta_finest[np.argmax(inliers_finest)]
    max_inliers_finest = np.max(inliers_finest)
    logging.info(f"Cameras {main_camera}-{secondary_camera}: End of finest search")
    logging.info(f"Cameras {main_camera}-{secondary_camera}: Best beta (finest): {best_beta_finest}, inliers: {max_inliers_finest:.4f}")

    # Final result
    logging.info(f"Cameras {main_camera}-{secondary_camera}: Final Result")
    logging.info(f"Cameras {main_camera}-{secondary_camera}: Best beta: {best_beta_finest}, inliers: {max_inliers_finest:.4f}")
    
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
    
    return best_beta_finest, max_inliers_finest

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
    best_beta, max_inliers = search_optimal_beta(
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
        "inlier_ratio": max_inliers
    }

def first_beta_search(dataset_no=DATASET_NO, beta_shift=4000):
    """
    Main function to find optimal beta values for all camera combinations
    """
    start_time = time.time()
    logging.info(f"Starting beta search process for dataset {dataset_no}")
        
    # Results dictionary to store all beta values
    results = {}
    
    # Create pairs of camera indices to process
    camera_pairs = [(m, s) for m in range(len(cameras)) for s in range(len(cameras)) if m != s]
    
    # Use ProcessPoolExecutor instead of ThreadPoolExecutor
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
                        "inlier_ratio": result["inlier_ratio"]
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
        f.write("main_camera,secondary_camera,beta,inlier_ratio\n")
        for key, value in results.items():
            f.write(f"{value['main_camera']},{value['secondary_camera']},{value['beta']:.4f},{value['inlier_ratio']:.4f}\n")
    
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
        
  
if __name__ == "__main__":
    
    
    # ============== Step 0: Load data and search for betas ==============
    
    camera_poses = []
    
    # Load camera info
    logging.info("Loading camera info")
    with open(f"drone-tracking-datasets/dataset{DATASET_NO}/cameras.txt", 'r') as f:
        cameras = f.read().strip().split()    
    cameras = cameras[2::3]
    logging.info(f"Loaded {len(cameras)} cameras")
    camera_poses = [{"cam_id": i, "R": None, "t": None, "b": None} for i in range(len(cameras))]
    
    # Load dataframe and splines
    logging.info("Loading dataframe and splines")
    df, splines, contiguous, camera_info = load_dataframe(cameras, DATASET_NO)
    
    if not os.path.exists(f"beta_results_dataset{DATASET_NO}.csv"):
        logging.info("Beta results file not found, starting beta search")
        first_beta_search(DATASET_NO, beta_shift=2000)
      
    logging.info("Loading beta results")  
    betas_df = pd.read_csv(f"beta_results_dataset{DATASET_NO}.csv")
    betas_df = betas_df.sort_values(by='inlier_ratio', ascending=False)
    
    
    # ============== Step 1: Generate first 3D point and splines with best beta ==============
    
    best_beta = betas_df.iloc[0]['beta']
    main_camera = betas_df.iloc[0]['main_camera']
    secondary_camera = betas_df.iloc[0]['secondary_camera']
    frames = df[df['cam_id'] == int(secondary_camera)][['frame_id', 'detection_x', 'detection_y']].values
    frames = [frame for frame in frames if frame[1] != 0.0 and frame[2] != 0.0]
    
    F, mask, correspondences = evaluate_beta(best_beta, frames, splines, camera_info, int(main_camera), int(secondary_camera), return_f=True)
    E = camera_info[int(main_camera)].K_matrix.T @ F @ camera_info[int(secondary_camera)].K_matrix
    
    main_pts_normalized = to_normalized_camera_coord(np.array([x for x, _, _ in correspondences]), camera_info[int(main_camera)].K_matrix, camera_info[int(main_camera)].distCoeff)
    secondary_pts_normalized = to_normalized_camera_coord(np.array([y for _, y, _ in correspondences]), camera_info[int(secondary_camera)].K_matrix, camera_info[int(secondary_camera)].distCoeff)
    
    _, R, t, mask, triangulated_points_first_step = cv.recoverPose(E, main_pts_normalized, secondary_pts_normalized, cameraMatrix=np.eye(3), distanceThresh=100)
    
    camera_poses[int(main_camera)] = {
        "cam_id": int(main_camera),
        "R": np.eye(3),
        "t": np.zeros((3, 1)),
        "b": 0
    }
    camera_poses[int(secondary_camera)] = {
        "cam_id": int(secondary_camera),
        "R": R,
        "t": t,
        "b": best_beta
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
    num_cameras_processed = 2
    
    while num_cameras_processed < len(cameras):    
        # Take the second-best beta value that has current main camera as main camera
        betas_df = betas_df[betas_df['main_camera'] == int(main_camera)]
        betas_df = betas_df.sort_values(by='inlier_ratio', ascending=False)
        nth_best_beta = betas_df.iloc[num_cameras_processed - 1]['beta']
        second_best_main_camera = betas_df.iloc[num_cameras_processed - 1]['main_camera']
        new_camera = betas_df.iloc[num_cameras_processed - 1]['secondary_camera']
        inlier_ratio = betas_df.iloc[num_cameras_processed - 1]['inlier_ratio']
        
        logging.info(f"{num_cameras_processed}-th best beta: {nth_best_beta}, main camera: {second_best_main_camera}, secondary camera: {new_camera}, inliers: {inlier_ratio:.4f}")
        
        # Load the frames for the second best beta
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
            "b": nth_best_beta
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
            c='blue', s=1, label=f"Triangulated Points {num_cameras_processed}th step"
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

        tpns_sorted = sorted(tpns_to_add_to_3d_with_timestamps, key=lambda pt: pt[3])
        extended_timestamps = set()
        merge_window = 5.0  # seconds

        for pt_x, pt_y, pt_z, pt_ts in tpns_sorted:
            if pt_ts in extended_timestamps:
                continue

            extended = False

            # Try forward or backward extension
            for spline in spline_dicts:
                ts_min, ts_max = np.min(spline['ts']), np.max(spline['ts'])

                if 0 < pt_ts - ts_max <= 5:  # Forward
                    ts_new = np.append(spline['ts'], pt_ts)
                    x_new = np.append(spline['spline_x'](spline['ts']), pt_x)
                    y_new = np.append(spline['spline_y'](spline['ts']), pt_y)
                    z_new = np.append(spline['spline_z'](spline['ts']), pt_z)

                elif 0 < ts_min - pt_ts <= 5:  # Backward
                    ts_new = np.append(pt_ts, spline['ts'])
                    x_new = np.append(pt_x, spline['spline_x'](spline['ts']))
                    y_new = np.append(pt_y, spline['spline_y'](spline['ts']))
                    z_new = np.append(pt_z, spline['spline_z'](spline['ts']))
                else:
                    continue

                # Always sort by time
                idx = np.argsort(ts_new)
                ts_sorted = ts_new[idx]
                x_sorted = x_new[idx]
                y_sorted = y_new[idx]
                z_sorted = z_new[idx]

                if len(ts_sorted) >= 4:
                    spline['spline_x'] = make_interp_spline(ts_sorted, x_sorted, k=3)
                    spline['spline_y'] = make_interp_spline(ts_sorted, y_sorted, k=3)
                    spline['spline_z'] = make_interp_spline(ts_sorted, z_sorted, k=3)
                    spline['ts'] = ts_sorted
                    extended_timestamps.add(pt_ts)
                    extended = True
                    break

            if extended:
                continue

            # Try merging two splines if point bridges them
            for i, s1 in enumerate(spline_dicts):
                for j, s2 in enumerate(spline_dicts):
                    if i == j:
                        continue
                    end1, start2 = np.max(s1['ts']), np.min(s2['ts'])

                    if 0 < pt_ts - end1 <= merge_window and 0 < start2 - pt_ts <= merge_window:
                        # Bridge candidate
                        ts_merged = np.concatenate((s1['ts'], [pt_ts], s2['ts']))
                        x_merged = np.concatenate((s1['spline_x'](s1['ts']), [pt_x], s2['spline_x'](s2['ts'])))
                        y_merged = np.concatenate((s1['spline_y'](s1['ts']), [pt_y], s2['spline_y'](s2['ts'])))
                        z_merged = np.concatenate((s1['spline_z'](s1['ts']), [pt_z], s2['spline_z'](s2['ts'])))

                        idx = np.argsort(ts_merged)
                        ts_sorted = ts_merged[idx]
                        x_sorted = x_merged[idx]
                        y_sorted = y_merged[idx]
                        z_sorted = z_merged[idx]

                        if len(ts_sorted) >= 4:
                            new_spline = {
                                'spline_x': make_interp_spline(ts_sorted, x_sorted, k=3),
                                'spline_y': make_interp_spline(ts_sorted, y_sorted, k=3),
                                'spline_z': make_interp_spline(ts_sorted, z_sorted, k=3),
                                'ts': ts_sorted
                            }

                            # Replace old splines with merged one
                            new_spline_list = []
                            for k, s in enumerate(spline_dicts):
                                if k not in (i, j):
                                    new_spline_list.append(s)
                            new_spline_list.append(new_spline)
                            spline_dicts = new_spline_list
                            extended_timestamps.add(pt_ts)
                            extended = True
                            break
                if extended:
                    break
        # Convert back to original format
        splines_3d = [(s['spline_x'], s['spline_y'], s['spline_z'], s['ts']) for s in spline_dicts]
        
        for camera_pose in camera_poses:
            if camera_pose["R"] is None:
                continue
            i = camera_pose["cam_id"]
            frames = df[df['cam_id'] == i][['frame_id', 'detection_x', 'detection_y']].values
            frames = [frame for frame in frames if frame[1] != 0.0 and frame[2] != 0.0]
            frames = np.array(frames)
            frames_ids = frames[:, 0]
            bbbb = camera_poses[i]["b"]
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
            ax.scatter(t[0], t[1], t[2], label=f"Camera {cam_id}", s=30)
        ax.set_title(f"3D Splines")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()
            
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
        
        # Plot camera poses with arrows along the z-direction and highlight the xy-plane
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot the xy-plane
        plane_size = 10
        xx, yy = np.meshgrid(np.linspace(-plane_size, plane_size, 10), np.linspace(-plane_size, plane_size, 10))
        zz = np.zeros_like(xx)
        ax.plot_surface(xx, yy, zz, alpha=0.2, color='gray')

        # Plot camera poses
        for pose in camera_poses:
            if pose["R"] is None:
                continue
            cam_id = pose["cam_id"]
            R = pose["R"]
            t = pose["t"].flatten()
            
            # Plot camera position
            ax.scatter(t[0], t[1], t[2], label=f"Camera {cam_id}", s=100)

            # Plot arrow along the z-direction
            z_dir = R @ np.array([0, 0, 1])  # Transform z-direction
            ax.quiver(
                t[0], t[1], t[2], 
                z_dir[0], z_dir[1], z_dir[2], 
                length=1.0, color='blue', linewidth=2
            )

        # Set plot labels and title
        ax.set_title("Camera Poses with Z-Direction Arrows")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()
        
        num_cameras_processed += 1
        
    for camera_pose in camera_poses:
        if camera_pose["R"] is None:
            continue
        i = camera_pose["cam_id"]
        frames = df[df['cam_id'] == i][['frame_id', 'detection_x', 'detection_y']].values
        plot_reprojection_analysis(
                    splines_3d, frames[:, 1:],
                    cv.Rodrigues(camera_poses[i]["R"])[0], camera_poses[i]["t"],
                    camera_info[i], i,
                    title=f"Refined Reprojection Analysis for Camera {i} with Beta {bbbb}"
        )
    
    plt.show()