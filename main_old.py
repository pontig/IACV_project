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

def undistort(df, camera_info, camera_id):
    dist_coeffs = camera_info[camera_id].distCoeff
    K = camera_info[camera_id].K_matrix
    points = df[df['cam_id'] == camera_id][['detection_x', 'detection_y']].values
    undistorted_points = cv.undistortPoints(points.reshape(-1, 1, 2).astype(np.float32), K, dist_coeffs)
    
    plt.figure(figsize=figsize)
    plt.scatter(undistorted_points[:, 0, 0], undistorted_points[:, 0, 1], c='g', s=1)
    plt.title(f"Undistorted points in Camera {camera_id}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.savefig(f'plots/undistorted_points_camera{camera_id}.png')
   
    
with open(f"drone-tracking-datasets/dataset{DATASET_NO}/cameras.txt", 'r') as f:
    cameras = f.read().strip().split()    
cameras = cameras[2::3]

df, splines, contiguous, camera_info = load_dataframe(cameras, DATASET_NO)
main_camera, secondary_camera, xx = find_max_overlapping(df, contiguous)
print(f"MAX OVERLAP Main camera: {main_camera}, Secondary camera: {secondary_camera}, with {xx} frames")

main_camera_width = camera_info[main_camera].resolution[0]
main_camera_height = camera_info[main_camera].resolution[1]


plt.figure(figsize=figsize)

for spline_x, spline_y, tss in splines[main_camera]:
    plt.plot(spline_x(tss), main_camera_height-spline_y(tss))
    
plt.xlim(0, main_camera_width)
plt.ylim(0, main_camera_height)

plt.savefig('plots/splines_whole.png')

correspondences = [] # List of (spline_x1, spline_y1), (x2, y2), (v2x, v2y), timestamps_2
det = df[df['cam_id'] == secondary_camera][['frame_id', 'detection_x', 'detection_y', 'velocity_x', 'velocity_y', 'global_ts']].values
beta = 0

plt.figure(figsize=figsize)
valid_frames = [frame for frame in det if frame[1] != 0.0 or frame[2] != 0.0]
print(f"Plotting initial correspondences between Main and Secondary Camera (length = {len(valid_frames)})")

points_to_scatter = []

for frame in valid_frames:
    global_ts = compute_global_time(frame[0], camera_info[secondary_camera].fps, beta)
    for spline_x, spline_y, tss in splines[main_camera]:
        if np.min(tss) <= global_ts <= np.max(tss):
            points_to_scatter.append((float(spline_x(global_ts)), main_camera_height - float(spline_y(global_ts))))
            break

points_to_scatter = np.array(points_to_scatter)
plt.scatter(points_to_scatter[:, 0], points_to_scatter[:, 1], c='k', s=1)

plt.xlim(0, main_camera_width)
plt.ylim(0, main_camera_height)
plt.title("Correspondences between Main and Secondary Camera")
plt.xlabel("X")
plt.ylabel("Y")

plt.savefig('plots/correspondences.png')


for iteration in range(3):  # Iterate to refine F and beta
    correspondences = []
    for frame in det:
        # No point in secondary camera
        if frame[1] == 0.0 and frame[2] == 0.0:
            continue
        global_ts = compute_global_time(frame[0], camera_info[secondary_camera].fps, beta)
        
        # find the spline of primary camera that contains the global_ts
        found = False
        for spline_x, spline_y, tss in splines[main_camera]:
            if np.min(tss) <= global_ts <= np.max(tss):
                found = True
                correspondences.append(((float(spline_x(global_ts)), float(spline_y(global_ts))),
                                        (frame[1] + beta * frame[3], frame[2] + beta * frame[4]),
                                        (frame[3], frame[4]), global_ts))
                break
            
    if not correspondences:
        print("No correspondences found.")
        break
    
    F, mask = cv.findFundamentalMat(np.array([x for x, _, _, _ in correspondences]), 
                                  np.array([y for _, y, v, t in correspondences]),
                                    cv.FM_RANSAC, 0.1, 0.99)
                                  
    
    # Estimate the time shift
    beta = estimate_time_shift(F,
                            np.array([x for x, _, _, _ in correspondences]), 
                            np.array([y for _, y, _, _ in correspondences]), 
                            np.array([v for _, _, v, _ in correspondences]))
    print(f"Iteration {iteration + 1}: Estimated time shift beta = {beta}")

correspondences = [correspondences[i] for i in range(len(correspondences)) if  correspondences[i][2][0] is not None] # if mask[i] == 1

# Update the global timestamps for the secondary camera in the dataframe and in the correspondences
df.loc[df['cam_id'] == secondary_camera, 'global_ts'] = compute_global_time(
    df.loc[df['cam_id'] == secondary_camera, 'frame_id'], 
    camera_info[secondary_camera].fps, 
    beta
)

print(f"left correspondences: {len(correspondences)}")

E = camera_info[secondary_camera].K_matrix.T @ F @ camera_info[main_camera].K_matrix
pts1_camera_coord = np.array([(np.linalg.inv(camera_info[main_camera].K_matrix) @ np.array([x[0], x[1], 1]))[:2] for x, _, _, _ in correspondences], dtype=np.float32)
pts2_camera_coord = np.array([(np.linalg.inv(camera_info[secondary_camera].K_matrix) @ np.array([y[0], y[1], 1]))[:2] for _, y, _, _ in correspondences], dtype=np.float32)
# Recover pose
# Normalize the points by dividing by the focal length and subtracting the principal point
pts1_camera_coord = np.array([[(x[0] - camera_info[main_camera].K_matrix[0, 2]) / camera_info[main_camera].K_matrix[0, 0],
                              (x[1] - camera_info[main_camera].K_matrix[1, 2]) / camera_info[main_camera].K_matrix[1, 1]] 
                             for x, _, _, _ in correspondences], dtype=np.float32)

pts2_camera_coord = np.array([[(y[0] - camera_info[secondary_camera].K_matrix[0, 2]) / camera_info[secondary_camera].K_matrix[0, 0],
                              (y[1] - camera_info[secondary_camera].K_matrix[1, 2]) / camera_info[secondary_camera].K_matrix[1, 1]] 
                             for _, y, _, _ in correspondences], dtype=np.float32)
retval, R, t, mask = cv.recoverPose(E, pts1_camera_coord, pts2_camera_coord)

plt.figure(figsize=figsize)
plt.scatter(pts1_camera_coord[:, 0], pts1_camera_coord[:, 1], c='b', s=1)
plt.title("Correspondences in Camera 1 in normalized coordinates")
plt.xlabel("X")
plt.ylabel("Y")
plt.savefig('plots/norm_coords_camera1.png')

# Triangulate points
P1 = np.dot(camera_info[main_camera].K_matrix, np.hstack((np.eye(3), np.zeros((3, 1)))))
P2 = np.dot(camera_info[secondary_camera].K_matrix, np.hstack((R, t)))
pts_4d_hom = cv.triangulatePoints(P1, P2, pts1_camera_coord.T, pts2_camera_coord.T)

# Convert homogeneous coordinates to 3D
pts_3d = pts_4d_hom[:3] / pts_4d_hom[3]
pts_3d = pts_3d.T

# Keep only the valid correspondences
# correspondences = [correspondences[i] for i in range(len(correspondences)) if mask[i] == 255]
correspondences = [correspondences[i] + (pts_3d[i],) for i in range(len(correspondences)) if not np.isnan(pts_3d[i]).any()] # List of (spline_x1, spline_y1), (x2, y2), (v2x, v2y), timestamps_2, 3D point


# Split correspondences based on global timestamp
slices = []
this_slice = [correspondences[0]]

for i in range(1, len(correspondences)):
    if correspondences[i][3] - correspondences[i-1][3] > 0.5:
        slices.append(this_slice)
        this_slice = [correspondences[i]]
    else:
        this_slice.append(correspondences[i])
slices.append(this_slice)

# Plot slices of main camera
plt.figure(figsize=figsize)
for slice in slices:
    if len(slice) < 3:
        continue
    pts_main_camera = np.array([x[0] for x in slice])
    plt.plot(pts_main_camera[:, 0], main_camera_height - pts_main_camera[:, 1])
plt.xlim(0, main_camera_width)
plt.ylim(0, main_camera_height)
plt.title("Slices of Main Camera")
plt.xlabel("X")
plt.ylabel("Y")

plt.savefig('plots/slices_main_camera.png')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for slice in slices:
    if len(slice) < 3:
        continue
    pts_3d_slice = np.array([x[4] for x in slice])
    ax.scatter(pts_3d_slice[:, 0], pts_3d_slice[:, 1], pts_3d_slice[:, 2])
    
# Interpolate the 3D splines for each slice
splines_3d = [] # List of (spline_x, spline_y, spline_z, (start_ts, end_ts))

for slice in slices:
    if len(slice) < 3:
        continue
    pts_3d_slice = np.array([x[4] for x in slice])
    tss = np.array([x[3] for x in slice])
    spline_x = make_interp_spline(tss, pts_3d_slice[:, 0], k=3, bc_type='natural')
    spline_y = make_interp_spline(tss, pts_3d_slice[:, 1], k=3, bc_type='natural')
    spline_z = make_interp_spline(tss, pts_3d_slice[:, 2], k=3, bc_type='natural')

    
    splines_3d.append((spline_x, spline_y, spline_z, (tss[0], tss[-1])))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for spline_x, spline_y, spline_z, tss in splines_3d:
    ts = np.linspace(tss[0], tss[1], num=100)
    points = np.vstack((spline_x(ts), spline_y(ts), spline_z(ts))).T
    ax.plot(points[:, 0], points[:, 1], points[:, 2])

# Re project the 3D points to the primary camera

pts_3d = np.array([np.append(x[4], 1) for x in correspondences])
pts_3d = pts_3d.T
P1 = np.dot(camera_info[main_camera].K_matrix, np.hstack((np.eye(3), np.zeros((3, 1)))))
print(f"P1:\n{P1}")
P2 = np.dot(camera_info[secondary_camera].K_matrix, np.hstack((R, t)))
pts_2d = np.dot(P1, pts_3d)
pts_2d = pts_2d / pts_2d[2]

# Plot the reprojected points
plt.figure(figsize=figsize)
plt.scatter(pts_2d[0], main_camera_height - pts_2d[1], c='r', s=1)
# plt.xlim(0, main_camera_width)
# plt.ylim(0, main_camera_height)
plt.title("Reprojected 3D points to Main Camera")
plt.xlabel("X")
plt.ylabel("Y")
plt.savefig('plots/reprojected_points.png')

plt.show()
exit(0)




























# Bundle Adjustment
filtered_df = df[(df['cam_id'] == main_camera) | (df['cam_id'] == secondary_camera)]
filtered_df.to_csv('detections.csv', index=False)

alpha1 = camera_info[main_camera].fps
alpha2 = camera_info[secondary_camera].fps
res = minimize_reprojection_error(initial_poses=((np.eye(3), np.zeros(3)), (R, t)),
                                  calibration_matrices=(camera_info[main_camera].K_matrix, camera_info[secondary_camera].K_matrix),
                                    initial_splines=splines_3d, 
                                    dataframe=df[(df['cam_id'] == main_camera) | (df['cam_id'] == secondary_camera)],
                                    initial_alphas=(alpha1, alpha2), 
                                    initial_betas=(0, beta))

np.save('bundle_adjustment.npy', res)

res = np.load('bundle_adjustment.npy', allow_pickle=True).item()

new_3d_splines = res['splines']
new_Ps = res['Ps']

# Plot new 3D splines after bundle adjustment
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for spline_x, spline_y, spline_z, tss in new_3d_splines:
    ts = np.linspace(tss[0], tss[1], num=100)
    points = np.vstack((spline_x(ts), spline_y(ts), spline_z(ts))).T
    ax.plot(points[:, 0], points[:, 1], points[:, 2])

plt.title("New 3D Splines after Bundle Adjustment")
plt.xlabel("X")
plt.ylabel("Y")
ax.set_zlabel("Z")

print("New P1:\n", new_Ps[0])

plt.figure(figsize=figsize)
for i, (spline_x, spline_y, spline_z, tss) in enumerate(new_3d_splines):
    ts = np.linspace(tss[0], tss[1], num=100)
    points = np.vstack((spline_x(ts), spline_y(ts), spline_z(ts), np.ones_like(ts)))
    points = np.dot(new_Ps[0], points) # Project to main camera
    points = points / points[2]
    plt.plot(points[0], main_camera_height - points[1], label=f"Slice {i}")
    
# plt.xlim(0, main_camera_width)
# plt.ylim(0, main_camera_height)
plt.title("Reprojected 3D points to Main Camera after Bundle Adjustment")
plt.xlabel("X")
plt.ylabel("Y")
plt.savefig('plots/reprojected_points_after_bundle_adjustment.png')

plt.show()
