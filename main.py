import json
import numpy as np
from scipy.interpolate import CubicSpline, make_interp_spline
from scipy.optimize import minimize
from matplotlib import pyplot as plt
import random
import cv2 as cv
import pandas as pd
import open3d as o3d

from camera_info import Camera_info
from data_loader import load_dataframe, find_max_overlapping
from global_fn import compute_global_time
from bundles_adj import minimize_reprojection_error

DATASET_NO = 4

figsize = (28, 15)

def estimate_time_shift(F, xi, xk, vk, max_iterations=1000, threshold=0.1, min_inliers=10):
    """
    Estimate the time shift between two cameras using RANSAC.
    
    Parameters:
    -----------
    F : ndarray
        Fundamental matrix
    xi : ndarray
        Points from camera i (N x 2)
    xk : ndarray
        Points from camera k (N x 2)
    vk : ndarray
        Velocities from camera k (N x 2)
    max_iterations : int, optional
        Maximum number of RANSAC iterations
    threshold : float, optional
        Threshold for considering a point as an inlier
    min_inliers : int, optional
        Minimum number of inliers required for a model to be considered valid
    
    Returns:
    --------
    float
        Estimated time shift (beta) between cameras
    list
        Indices of inliers
    """

    # Make sure all inputs are numpy arrays
    xi = np.array(xi)
    xk = np.array(xk)
    vk = np.array(vk)
    
    if len(xi) < 3:
        raise ValueError("Need at least 3 point correspondences")
    
    best_beta = 0
    best_inliers = []
    best_inlier_count = 0
    
    # Calculate all possible beta values for later evaluation
    all_beta_values = []
    for i in range(len(xi)):
        x1_h = np.append(xi[i], 1)  # Homogeneous coordinates for point in camera i
        x2_h = np.append(xk[i], 1)  # Homogeneous coordinates for point in camera k
        v2_h = np.append(vk[i], 0)  # Homogeneous coordinates for velocity (append 0)
        
        numerator = np.dot(x2_h.T, np.dot(F, x1_h))
        denominator = np.dot(v2_h.T, np.dot(F, x1_h))
        
        if np.abs(denominator) > 1e-6:  # Avoid division by zero
            beta_ik = -numerator / denominator
            all_beta_values.append((i, beta_ik))
    
    # RANSAC implementation
    for _ in (range(max_iterations)):
        # 1. Randomly select a sample of 3 correspondences
        if len(all_beta_values) < 3:
            continue
            
        sample_indices = random.sample(range(len(all_beta_values)), 3)
        sample_betas = [all_beta_values[i][1] for i in sample_indices]
        
        # 2. Compute model from the sample (median beta value)
        model_beta = np.median(sample_betas)
        
        # Skip if the model is invalid
        if np.isnan(model_beta) or np.isinf(model_beta) or np.abs(model_beta) > 10:
            continue
        
        # 3. Calculate inliers based on reprojection error
        inliers = []
        for idx, beta_val in all_beta_values:
            # Calculate error as the difference between predicted beta and actual beta
            error = np.abs(beta_val - model_beta)
            if error < threshold:
                inliers.append(idx)
        
        # 4. Save the model if it has more inliers
        if len(inliers) > best_inlier_count and len(inliers) >= min_inliers:
            best_inlier_count = len(inliers)
            best_inliers = inliers
            
            # Recalculate beta using all inliers
            inlier_betas = [all_beta_values[i][1] for i in range(len(all_beta_values)) if all_beta_values[i][0] in inliers]
            best_beta = np.median(inlier_betas)
    
    # If RANSAC failed to find a good model, use median of all values
    if not best_inliers:
        print("RANSAC could not find a consistent model. Using median of all beta values.")
        best_beta = np.median([b for _, b in all_beta_values])
        best_inliers = list(range(len(xi)))
    
    return best_beta

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
   
    
with open(f"drone-tracking-datasets/dataset{DATASET_NO}/cameras.txt", 'r') as f:
    cameras = f.read().strip().split()    
cameras = cameras[2::3]

df, splines, contiguous, camera_info = load_dataframe(cameras, DATASET_NO)
main_camera, secondary_camera, xx = find_max_overlapping(df, contiguous)
print(f"MAX OVERLAP Main camera: {main_camera}, Secondary camera: {secondary_camera}, with {xx} frames")

main_camera_width = camera_info[main_camera].resolution[0]
main_camera_height = camera_info[main_camera].resolution[1]
secondary_camera_width = camera_info[secondary_camera].resolution[0]
secondary_camera_height = camera_info[secondary_camera].resolution[1]

correspondences = [] # List of (spline_x1, spline_y1), (x2, y2), (v2x, v2y), timestamps_2
frames = df[df['cam_id'] == secondary_camera][['frame_id', 'detection_x', 'detection_y', 'velocity_x', 'velocity_y', 'global_ts']].values
frames = [frame for frame in frames if frame[1] != 0.0 and frame[2] != 0.0]
beta = 0

for frame in frames:
    global_ts = frame[5]
    for spline_x, spline_y, tss in splines[main_camera]:
        if np.min(tss) <= global_ts <= np.max(tss):
            x1 = float(spline_x(global_ts))
            y1 = float(spline_y(global_ts))
            correspondences.append((
                (x1, y1),
                (frame[1], frame[2]),
                (frame[3], frame[4]),
                global_ts
            ))
            break
        
if not correspondences:
    raise ValueError("No overlapping frames found between the two cameras")
# Estimate the fundamental matrix
F, mask = cv.findFundamentalMat(
    np.array([x for x, _, _, _ in correspondences]),
    np.array([y for _, y, _, _ in correspondences]),
    cv.RANSAC, 3, 0.999
)


print(np.sum(mask))
print(correspondences.__len__())
print(f"Estimated fundamental matrix: {F}")
# correspondences = [correspondences[i] for i in range(len(correspondences)) if mask[i] == 1]

# Essential matrix
E = camera_info[secondary_camera].K_matrix.T @ F @ camera_info[main_camera].K_matrix
print(f"Estimated essential matrix:\n {E/E[2, 2]}")


_, E, R, t, mask = cv.recoverPose(
    np.array([x for x, _, _, _ in correspondences]),
    np.array([y for _, y, _, _ in correspondences]),
    camera_info[main_camera].K_matrix, camera_info[main_camera].distCoeff,
    camera_info[secondary_camera].K_matrix, camera_info[secondary_camera].distCoeff,
    cv.RANSAC, 0.999, 3
)
print(E/E[2, 2])
P1 = np.dot(camera_info[main_camera].K_matrix, np.hstack((np.eye(3), np.zeros((3, 1)))))
P2 = np.dot(camera_info[secondary_camera].K_matrix, np.hstack((R, t)))
# pts_camera_coord_1 = to_normalized_camera_coord(np.array([x for x, _, _, _ in correspondences]), camera_info[main_camera].K_matrix, camera_info[main_camera].distCoeff, np.eye(3), np.zeros((3,1)))
# pts_camera_coord_2 = to_normalized_camera_coord(np.array([y for _, y, _, _ in correspondences]), camera_info[secondary_camera].K_matrix, camera_info[secondary_camera].distCoeff, R, t)
pts_camera_coord_1 = np.array([x for x, _, _, _ in correspondences])
pts_camera_coord_2 = np.array([y for _, y, _, _ in correspondences])

fig, ax = plt.subplots(2, 2, figsize=(15, 7))

# Plot points in camera coordinates for main camera
ax[0, 0].scatter(pts_camera_coord_1[:, 0], -pts_camera_coord_1[:, 1], c='r', marker='o', s=1)
ax[0, 0].set_title('Main Camera Coordinates')
ax[0, 0].set_xlabel('X')
ax[0, 0].set_ylabel('Y')
ax[0, 0].axis('equal')

# Plot points in camera coordinates for secondary camera
ax[0, 1].scatter(pts_camera_coord_2[:, 0], -pts_camera_coord_2[:, 1], c='b', marker='o', s=1)
ax[0, 1].set_title('Secondary Camera Coordinates')
ax[0, 1].set_xlabel('X')
ax[0, 1].set_ylabel('Y')
ax[0, 1].axis('equal')

# Triangulate points
pts_3d = cv.triangulatePoints(P1, P2, pts_camera_coord_1.T, pts_camera_coord_2.T)
pts_3d /= pts_3d[3]

# Re-project points onto the image planes
# Convert rotation matrices to rotation vectors using Rodrigues
rvec_main, _ = cv.Rodrigues(np.eye(3))
rvec_secondary, _ = cv.Rodrigues(R)

# Project 3D points to 2D image plane
pts_2d_main = cv.projectPoints(pts_3d.T[:, :3], rvec_main, np.zeros(3), camera_info[main_camera].K_matrix, camera_info[main_camera].distCoeff)[0]
pts_2d_secondary = cv.projectPoints(pts_3d.T[:, :3], rvec_secondary, t, camera_info[secondary_camera].K_matrix, camera_info[secondary_camera].distCoeff)[0]
pts_2d_main = pts_2d_main.reshape(-1, 2)
pts_2d_secondary = pts_2d_secondary.reshape(-1, 2)

# Plot re-projected points for main camera
ax[1, 0].scatter(pts_2d_main[:,0], -pts_2d_main[:,1], c='r', marker='o', s=1)
ax[1, 0].set_title('Main Camera Reprojections')
ax[1, 0].set_xlabel('X')
ax[1, 0].set_ylabel('Y')
ax[1, 0].axis('equal')

# Plot re-projected points for secondary camera
ax[1, 1].scatter(pts_2d_secondary[:,0], -pts_2d_secondary[:,1], c='b', marker='o', s=1)
ax[1, 1].set_title('Secondary Camera Reprojections')
ax[1, 1].set_xlabel('X')
ax[1, 1].set_ylabel('Y')
ax[1, 1].axis('equal')

fig.savefig('plots/reprojection.png')

# Plot 3D points
fig_3d = plt.figure(figsize=(10, 10))
ax_3d = fig_3d.add_subplot(111, projection='3d')
ax_3d.scatter(pts_3d[0], pts_3d[1], pts_3d[2], c='g', marker='o', s=1)
ax_3d.set_title('3D Points')
ax_3d.set_xlabel('X')
ax_3d.set_ylabel('Y')
ax_3d.set_zlabel('Z')

# plt.show()