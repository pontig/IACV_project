import json
import warnings
import numpy as np
from scipy.interpolate import CubicSpline, make_interp_spline
from scipy.optimize import minimize
import matplotlib
import matplotlib.pyplot as plt
import random
import cv2 as cv
import pandas as pd
import open3d as o3d

from camera_info import Camera_info
from data_loader import load_dataframe
from global_fn import *
from bundles_adj import minimize_reprojection_error

DATASET_NO = 4
inliers_coarse = []
inliers_fine = []
inliers_finer = []
inliers_finest = []

figsize = (28, 15)

def to_normalized_camera_coord(pts, K, distcoeff, R, t):
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
    P = np.dot(K, np.hstack((R, t)))
    pts_normalized = cv.undistortPoints(pts, K, distcoeff)
    
    # Convert from homogeneous coordinates to 2D
    pts_normalized = pts_normalized.reshape(-1, 2)
    
    return pts_normalized
   
print("Loading camera info") 
with open(f"drone-tracking-datasets/dataset{DATASET_NO}/cameras.txt", 'r') as f:
    cameras = f.read().strip().split()    
cameras = cameras[2::3]
print(f"Loaded {len(cameras)} cameras")

print("Loading dataframe")
df, splines, contiguous, camera_info = load_dataframe(cameras, DATASET_NO)

main_camera = 3 #col
secondary_camera = 1 #row
xx = -1

print(f"MAX OVERLAP Main camera: {main_camera}, Secondary camera: {secondary_camera}, with {xx} frames")

print(f"fps: {camera_info[main_camera].fps}, {camera_info[secondary_camera].fps}")

main_camera_width = camera_info[main_camera].resolution[0]
main_camera_height = camera_info[main_camera].resolution[1]
secondary_camera_width = camera_info[secondary_camera].resolution[0]
secondary_camera_height = camera_info[secondary_camera].resolution[1]

frames = df[df['cam_id'] == secondary_camera][['frame_id', 'detection_x', 'detection_y']].values
frames = [frame for frame in frames if frame[1] != 0.0 and frame[2] != 0.0]
beta = 0  # Initialize with default value

def find_beta(b, inliers_list):
    correspondences = [] # List of (spline_x1, spline_y1), (x2, y2), (v2x, v2y), timestamps_2
    for frame in frames:
        global_ts = compute_global_time(frame[0], camera_info[main_camera].fps/camera_info[secondary_camera].fps, b)
        for spline_x, spline_y, tss in splines[main_camera]:
            if np.min(tss) <= global_ts <= np.max(tss):
                x1 = float(spline_x(global_ts))
                y1 = float(spline_y(global_ts))
                correspondences.append((
                    (x1, y1),
                    (frame[1], frame[2]),
                    (frame[1], frame[2]),
                    # (frame[3], frame[4]),
                    global_ts
                ))
                            
                break
            
    if not correspondences:
        warnings.warn("No overlapping frames found between the two cameras for beta: " + str(b))
        inliers_list.append(0)
        return 0

    # Estimate the fundamental matrix
    F, mask = cv.findFundamentalMat(
        np.array([x for x, _, _, _ in correspondences]),
        np.array([y for _, y, _, _ in correspondences]),
        cv.RANSAC
    )

    inlier_ratio = np.sum(mask) / len(mask) if mask is not None else 0
    print(f"beta: {b:.3f}, inliers: {inlier_ratio:.4f}, correspondences: {len(correspondences)}")
    inliers_list.append(inlier_ratio)
    return inlier_ratio


coarse_step = 5
fine_step = 1
finer_step = 0.1
finest_step = 0.01

# Step 1: Coarse Search
print("\n=== Coarse Search ===")
beta_shift = 100
beta_coarse = np.arange(-beta_shift, beta_shift, coarse_step)
beta_values_coarse = []

for b in beta_coarse:
    find_beta(b, inliers_coarse)
    beta_values_coarse.append(b)
    
best_beta_coarse = beta_coarse[np.argmax(inliers_coarse)]
max_inliers_coarse = np.max(inliers_coarse)
print(f"End of coarse search")
print(f"Best beta (coarse): {best_beta_coarse}, inliers: {max_inliers_coarse:.4f}")

# Step 2: Fine Search
print("\n=== Fine Search ===")
beta_fine = np.arange(best_beta_coarse - 100, best_beta_coarse + 100, fine_step)
beta_values_fine = []

for b in beta_fine:
    find_beta(b, inliers_fine)
    beta_values_fine.append(b)

best_beta_fine = beta_fine[np.argmax(inliers_fine)]
max_inliers_fine = np.max(inliers_fine)
print(f"End of fine search")
print(f"Best beta (fine): {best_beta_fine}, inliers: {max_inliers_fine:.4f}")

# Step 3: Finer Search
print("\n=== Finer Search ===")
beta_finer = np.arange(best_beta_fine - 10, best_beta_fine + 10, finer_step)
beta_values_finer = []

for b in beta_finer:
    find_beta(b, inliers_finer)
    beta_values_finer.append(b)

best_beta_finer = beta_finer[np.argmax(inliers_finer)]
max_inliers_finer = np.max(inliers_finer)
print(f"End of finer search")
print(f"Best beta (finer): {best_beta_finer}, inliers: {max_inliers_finer:.4f}")

# Step 4: Finest Search
print("\n=== Finest Search ===")
beta_finest = np.arange(best_beta_finer - 1, best_beta_finer + 1, finest_step)
beta_values_finest = []

for b in beta_finest:
    find_beta(b, inliers_finest)
    beta_values_finest.append(b)

best_beta_finest = beta_finest[np.argmax(inliers_finest)]
max_inliers_finest = np.max(inliers_finest)
print(f"End of finest search")
print(f"Best beta (finest): {best_beta_finest}, inliers: {max_inliers_finest:.4f}")

# Final result
print("\n=== Final Result ===")
print(f"Best beta: {best_beta_finest}, inliers: {max_inliers_finest:.4f}")
# Plot the results for all refinement levels
plt.figure(figsize=figsize)

# Create a figure with 2 rows and 2 columns
fig, axes = plt.subplots(2, 2, figsize=figsize)
fig.suptitle('Beta Search Refinement Process', fontsize=20)

# Coarse search plot
axes[0, 0].plot(beta_values_coarse, inliers_coarse, 'b.-', markersize=3, label='Inliers')
axes[0, 0].axvline(x=best_beta_coarse, color='r', linestyle='--', 
                  label=f'Best β={best_beta_coarse} (inliers={max_inliers_coarse:.4f})')
axes[0, 0].set_title('Coarse Search (step=50)')
axes[0, 0].set_xlabel('Beta')
axes[0, 0].set_ylabel('Inlier Ratio')
axes[0, 0].set_ylim(-0.1, 1.1)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend()

# Fine search plot
axes[0, 1].plot(beta_values_fine, inliers_fine, 'r.-', markersize=3, label='Inliers')
axes[0, 1].axvline(x=best_beta_fine, color='b', linestyle='--', 
                  label=f'Best β={best_beta_fine} (inliers={max_inliers_fine:.4f})')
axes[0, 1].set_title(f'Fine Search (step=5, range={best_beta_coarse}±100)')
axes[0, 1].set_xlabel('Beta')
axes[0, 1].set_ylabel('Inlier Ratio')
axes[0, 1].set_ylim(-0.1, 1.1)
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].legend()

# Finer search plot
axes[1, 0].plot(beta_values_finer, inliers_finer, 'g.-', markersize=3, label='Inliers')
axes[1, 0].axvline(x=best_beta_finer, color='b', linestyle='--', 
                  label=f'Best β={best_beta_finer} (inliers={max_inliers_finer:.4f})')
axes[1, 0].set_title(f'Finer Search (step=0.5, range={best_beta_fine}±10)')
axes[1, 0].set_xlabel('Beta')
axes[1, 0].set_ylabel('Inlier Ratio')
axes[1, 0].set_ylim(-0.1, 1.1)
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].legend()

# Finest search plot
axes[1, 1].plot(beta_values_finest, inliers_finest, 'm.-', markersize=5, label='Inliers')
axes[1, 1].axvline(x=best_beta_finest, color='b', linestyle='--', 
                  label=f'Best β={best_beta_finest} (inliers={max_inliers_finest:.4f})')
axes[1, 1].set_title(f'Finest Search (step=0.05, range={best_beta_finer}±1)')
axes[1, 1].set_xlabel('Beta')
axes[1, 1].set_ylabel('Inlier Ratio')
axes[1, 1].set_ylim(-0.1, 1.1)
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].legend()

plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.savefig(f"plots/inliers_vs_beta_refinement_{DATASET_NO}.png", dpi=300)

# Combined visualization in one plot with different colors and transparency
plt.figure(figsize=figsize)
plt.plot(beta_values_coarse, inliers_coarse, 'b-', alpha=0.5, linewidth=1, label='Coarse (step=50)')
plt.plot(beta_values_fine, inliers_fine, 'r-', alpha=0.6, linewidth=1.5, label='Fine (step=5)')
plt.plot(beta_values_finer, inliers_finer, 'g-', alpha=0.7, linewidth=2, label='Finer (step=0.5)')
plt.plot(beta_values_finest, inliers_finest, 'm-', alpha=1.0, linewidth=2.5, label='Finest (step=0.05)')

# Mark the best beta for each refinement level
plt.scatter(best_beta_coarse, max_inliers_coarse, c='blue', marker='*', s=200, 
           label=f'Best β (Coarse)={best_beta_coarse}')
plt.scatter(best_beta_fine, max_inliers_fine, c='red', marker='*', s=200, 
           label=f'Best β (Fine)={best_beta_fine}')
plt.scatter(best_beta_finer, max_inliers_finer, c='green', marker='*', s=200, 
           label=f'Best β (Finer)={best_beta_finer}')
plt.scatter(best_beta_finest, max_inliers_finest, c='magenta', marker='*', s=300, 
           label=f'Best β (Finest)={best_beta_finest}')

plt.title('Multi-level Beta Search Refinement', fontsize=18)
plt.xlabel('Beta', fontsize=14)
plt.ylabel('Inlier Ratio', fontsize=14)
plt.ylim(-0.1, 1.1)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig(f"plots/inliers_vs_beta_combined_{DATASET_NO}.png", dpi=300)

# Print final conclusion
print(f"\nFinal conclusion: The optimal time shift between cameras {main_camera} and {secondary_camera} is beta = {best_beta_finest:.2f}")