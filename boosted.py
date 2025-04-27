
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
from data_loader import load_dataframe
from global_fn import *
from concurrent.futures import ProcessPoolExecutor

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
    correspondences = [] # List of (spline_x1, spline_y1), (x2, y2)
    for frame in frames:
        global_ts = compute_global_time(frame[0], camera_info[main_camera].fps/camera_info[secondary_camera].fps, b)
        for spline_x, spline_y, tss in splines[main_camera]:
            if np.min(tss) <= global_ts <= np.max(tss):
                x1 = float(spline_x(global_ts))
                y1 = float(spline_y(global_ts))
                correspondences.append((
                    (x1, y1),
                    (frame[1], frame[2]),
                ))
                break
            
    if not correspondences:
        warnings.warn(f"No overlapping frames found between cameras {main_camera} and {secondary_camera} for beta: {b}")
        if inliers_list is not None:
            inliers_list.append(0)
        return 0

    # Estimate the fundamental matrix
    F, mask = cv.findFundamentalMat(
        np.array([x for x, _ in correspondences]),
        np.array([y for _, y in correspondences]),
        cv.RANSAC
    )

    inlier_ratio = np.sum(mask) / len(mask) if mask is not None else 0
    logging.info(f"Cameras {main_camera}-{secondary_camera}, beta: {b:.3f}, inliers: {inlier_ratio:.4f}, correspondences: {len(correspondences)}")
    
    if inliers_list is not None:
        inliers_list.append(inlier_ratio)
        
    if return_f:
        return F, mask, correspondences
    return inlier_ratio

def plot_refinement_process(beta_values_coarse, inliers_coarse, best_beta_coarse, max_inliers_coarse,
                            beta_values_fine, inliers_fine, best_beta_fine, max_inliers_fine,
                            beta_values_finer, inliers_finer, best_beta_finer, max_inliers_finer,
                            beta_values_finest, inliers_finest, best_beta_finest, max_inliers_finest,
                            main_camera, secondary_camera, dataset_no):
    """
    Plot the beta search refinement process
    
    Parameters:
    -----------
    beta_values_* : list
        Lists of beta values for each search level
    inliers_* : list
        Lists of inlier ratios for each search level
    best_beta_* : float
        Best beta value for each search level
    max_inliers_* : float
        Maximum inlier ratio for each search level
    main_camera : int
        ID of main camera
    secondary_camera : int
        ID of secondary camera
    dataset_no : int
        Dataset number
    """
    figsize = (28, 15)
    
    # Create a figure with 2 rows and 2 columns
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f'Beta Search Refinement Process (Cameras {main_camera}-{secondary_camera})', fontsize=20)

    # Coarse search plot
    axes[0, 0].plot(beta_values_coarse, inliers_coarse, 'b.-', markersize=3, label='Inliers')
    axes[0, 0].axvline(x=best_beta_coarse, color='r', linestyle='--', 
                      label=f'Best β={best_beta_coarse} (inliers={max_inliers_coarse:.4f})')
    axes[0, 0].set_title('Coarse Search')
    axes[0, 0].set_xlabel('Beta')
    axes[0, 0].set_ylabel('Inlier Ratio')
    axes[0, 0].set_ylim(-0.1, 1.1)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # Fine search plot
    axes[0, 1].plot(beta_values_fine, inliers_fine, 'r.-', markersize=3, label='Inliers')
    axes[0, 1].axvline(x=best_beta_fine, color='b', linestyle='--', 
                      label=f'Best β={best_beta_fine} (inliers={max_inliers_fine:.4f})')
    axes[0, 1].set_title(f'Fine Search (range={best_beta_coarse}±100)')
    axes[0, 1].set_xlabel('Beta')
    axes[0, 1].set_ylabel('Inlier Ratio')
    axes[0, 1].set_ylim(-0.1, 1.1)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # Finer search plot
    axes[1, 0].plot(beta_values_finer, inliers_finer, 'g.-', markersize=3, label='Inliers')
    axes[1, 0].axvline(x=best_beta_finer, color='b', linestyle='--', 
                      label=f'Best β={best_beta_finer} (inliers={max_inliers_finer:.4f})')
    axes[1, 0].set_title(f'Finer Search (range={best_beta_fine}±10)')
    axes[1, 0].set_xlabel('Beta')
    axes[1, 0].set_ylabel('Inlier Ratio')
    axes[1, 0].set_ylim(-0.1, 1.1)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    # Finest search plot
    axes[1, 1].plot(beta_values_finest, inliers_finest, 'm.-', markersize=5, label='Inliers')
    axes[1, 1].axvline(x=best_beta_finest, color='b', linestyle='--', 
                      label=f'Best β={best_beta_finest} (inliers={max_inliers_finest:.4f})')
    axes[1, 1].set_title(f'Finest Search (range={best_beta_finer}±1)')
    axes[1, 1].set_xlabel('Beta')
    axes[1, 1].set_ylabel('Inlier Ratio')
    axes[1, 1].set_ylim(-0.1, 1.1)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Create plots directory if it doesn't exist
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/inliers_vs_beta_refinement_cam{main_camera}-{secondary_camera}_ds{dataset_no}.png", dpi=300)
    plt.close()

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

def first_beta_search(dataset_no=4, beta_shift=4000):
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
  

if __name__ == "__main__":
    
        # Load camera info
    logging.info("Loading camera info")
    with open(f"drone-tracking-datasets/dataset{DATASET_NO}/cameras.txt", 'r') as f:
        cameras = f.read().strip().split()    
    cameras = cameras[2::3]
    logging.info(f"Loaded {len(cameras)} cameras")
    
    # Load dataframe and splines
    logging.info("Loading dataframe and splines")
    df, splines, contiguous, camera_info = load_dataframe(cameras, DATASET_NO)
    
    if not os.path.exists(f"beta_results_dataset{DATASET_NO}.csv"):
        first_beta_search(DATASET_NO, beta_shift=4000)
        
    betas_df = pd.read_csv(f"beta_results_dataset{DATASET_NO}.csv")
    betas_df = betas_df.sort_values(by='inlier_ratio', ascending=False)
    
    best_beta = betas_df.iloc[0]['beta']
    main_camera = betas_df.iloc[0]['main_camera']
    secondary_camera = betas_df.iloc[0]['secondary_camera']
    frames = df[df['cam_id'] == int(secondary_camera)][['frame_id', 'detection_x', 'detection_y']].values
    frames = [frame for frame in frames if frame[1] != 0.0 and frame[2] != 0.0]
    
    F, mask, correspondences = evaluate_beta(best_beta, frames, splines, camera_info, int(main_camera), int(secondary_camera), return_f=True)
    E = camera_info[int(main_camera)].K_matrix.T @ F @ camera_info[int(secondary_camera)].K_matrix
    
    main_pts_normalized = to_normalized_camera_coord(np.array([x for x, _ in correspondences]), camera_info[int(main_camera)].K_matrix, camera_info[int(main_camera)].distCoeff)
    secondary_pts_normalized = to_normalized_camera_coord(np.array([y for _, y in correspondences]), camera_info[int(secondary_camera)].K_matrix, camera_info[int(secondary_camera)].distCoeff)
    
    _, R, t, mask, triangulated_points = cv.recoverPose(E, main_pts_normalized, secondary_pts_normalized, cameraMatrix=np.eye(3), distanceThresh=100)
    
    triangulated_points /= triangulated_points[3]
    triangulated_points = triangulated_points[:3]
    triangulated_points = triangulated_points.T
    
    # Plot triangulated points
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(triangulated_points[:, 0], triangulated_points[:, 1], triangulated_points[:, 2], c='r', marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Triangulated Points')
    plt.show()
    plt.savefig(f"plots/triangulated_points_{main_camera}_{secondary_camera}.png", dpi=300)
    
 
    
    
    
    