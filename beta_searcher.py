import os
import sys
import time
import logging
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import warnings
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import argparse

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving plots

from global_fn import *
from plotter import plot_refinement_process, plot_combined_results
from data_loader import load_dataframe

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
        Initial beta search range
        
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
        main_camera, secondary_camera, dataset_no, f"plots/search_results/dataset{dataset_no}"
    )
    
    plot_combined_results(
        beta_values_coarse, inliers_coarse, best_beta_coarse, max_inliers_coarse,
        beta_values_fine, inliers_fine, best_beta_fine, max_inliers_fine,
        beta_values_finer, inliers_finer, best_beta_finer, max_inliers_finer,
        beta_values_finest, inliers_finest, best_beta_finest, max_inliers_finest,
        main_camera, secondary_camera, dataset_no, f"plots/search_results/dataset{dataset_no}"
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
    
    # Save all results to a separate log file (overwrite if exists)
    result_log_path = f"beta_results_dataset{dataset_no}.csv"
    with open(result_log_path, 'w') as f:
        f.write("main_camera,secondary_camera,beta,inlier_ratio,num_inliers\n")
        for key, value in results.items():
            f.write(f"{value['main_camera']},{value['secondary_camera']},{value['beta']:.4f},{value['inlier_ratio']:.4f},{value['inlier_count']}\n")
    
    logging.info(f"Results saved to {result_log_path}")
    logging.info(f"Total execution time: {(time.time() - start_time)/60:.2f} minutes")
    
    return results


# Parse positional argument for dataset_no
parser = argparse.ArgumentParser(description="Beta searcher for done tracking datasets")
parser.add_argument("dataset_no", type=int, help="Dataset number to process")
args = parser.parse_args()
DATASET_NO = args.dataset_no

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

first_beta_search(
    dataset_no=DATASET_NO,
    beta_shift=3000
)
logging.info("Beta search completed")