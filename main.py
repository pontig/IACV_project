import json
import numpy as np
from scipy.interpolate import CubicSpline
from matplotlib import pyplot as plt
import cv2 as cv
import pandas as pd
import open3d as o3d

from camera_info import Camera_info
from data_loader import load_dataframe, find_max_overlapping
from global_fn import compute_global_time

DATASET_NO = 3

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
    import numpy as np
    import random
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    
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

# def triangulate_points(P1, P2, pts1, pts2, K1, K2):
#     """
#     Triangulate points after normalizing them using camera intrinsics.
    
#     P1: Projection matrix for camera 1
#     P2: Projection matrix for camera 2
#     pts1: Points in image 1 (Nx2)
#     pts2: Points in image 2 (Nx2)
#     K1: Intrinsic matrix of camera 1
#     K2: Intrinsic matrix of camera 2
    
#     Returns: Nx3 array of 3D points
#     """
#     # Convert points to homogeneous coordinates
#     pts1_hom = np.hstack((pts1, np.ones((pts1.shape[0], 1)))).T  # Shape (3, N)
#     pts2_hom = np.hstack((pts2, np.ones((pts2.shape[0], 1)))).T  # Shape (3, N)

#     # Normalize image coordinates (convert to camera space)
#     pts1_norm = np.linalg.inv(K1) @ pts1_hom  # (3, N)
#     pts2_norm = np.linalg.inv(K2) @ pts2_hom  # (3, N)

#     # Triangulate points
#     pts_4d_hom = cv.triangulatePoints(P1, P2, pts1_norm[:2], pts2_norm[:2])  # OpenCV requires 2D (x,y)
    
#     # Convert from homogeneous coordinates
#     pts_3d_hom = pts_4d_hom / pts_4d_hom[3]
#     pts_3d = pts_3d_hom[:3].T  # Convert to (N, 3)

#     return pts_3d

# def in_front_of_both_cameras(R, t, K1, K2, pts1, pts2):
#     """
#     Check if the triangulated points are in front of both cameras.
    
#     R: Rotation matrix from camera 1 to camera 2 
#     t: Translation vector from camera 1 to camera 2
#     K1: Intrinsic matrix of camera 1
#     K2: Intrinsic matrix of camera 2
#     pts1: Points in camera 1
#     pts2: Points in camera 2
    
#     Returns: Number of points in front of both cameras and the 3D points
#     """
#     P1 = np.dot(K1, np.hstack((np.eye(3), np.zeros((3, 1)))))
#     P2 = np.dot(K2, np.hstack((R, t.reshape(3, 1))))
    
#     pts_3d = triangulate_points(P1, P2, pts1, pts2, K1, K2)
    
#     # Check if points are in front of camera 1
#     pts_3d_in_cam1 = pts_3d.copy()
#     in_front_cam1 = pts_3d_in_cam1[:, 2] > 0
#     n_in_front_cam1 = np.sum(in_front_cam1)
    
#     # Check if points are in front of camera 2
#     pts_3d_in_cam2 = np.dot(R, pts_3d.T).T + t.reshape(1, 3)
#     in_front_cam2 = pts_3d_in_cam2[:, 2] > 0
#     n_in_front_cam2 = np.sum(in_front_cam2)
    
#     # Points in front of both cameras
#     in_front_both = np.logical_and(in_front_cam1, in_front_cam2)
#     pts_3d_valid = pts_3d[in_front_both]
    
#     # Return the number of points in front of both cameras and the 3D points
#     return min(n_in_front_cam1, n_in_front_cam2), pts_3d, pts_3d_valid

# # Create Open3D visualization for camera
# def create_camera_visualization(R, t, size=0.1, color=[0, 0, 0], name="Camera"):
#     """Create a camera visualization with coordinate axes."""
#     # Create a coordinate frame
#     camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    
#     # Apply the rotation and translation
#     camera_frame.translate(t)
#     camera_frame.rotate(R)
#     return camera_frame

# # Function to visualize a configuration
# def visualize_configuration(R, t, K1, K2, pts1, pts2, config_name):
#     """
#     Visualize a specific R,t configuration with Open3D.
    
#     R: Rotation matrix from camera 1 to camera 2
#     t: Translation vector from camera 1 to camera 2
#     K1: Intrinsic matrix of camera 1
#     K2: Intrinsic matrix of camera 2
#     pts1: Points in camera 1
#     pts2: Points in camera 2
#     config_name: Name of this configuration (for window title)
#     """
#     # Get 3D points and count points in front of cameras
#     n_in_front, pts_3d, pts_3d_valid = in_front_of_both_cameras(R, t, K1, K2, pts1, pts2)
    
#     # Create point cloud for all 3D points (blue)
#     all_point_cloud = o3d.geometry.PointCloud()
#     all_point_cloud.points = o3d.utility.Vector3dVector(pts_3d)
#     all_point_cloud.paint_uniform_color([0.7, 0.7, 1.0])  # Light blue for all points
    
#     # Create point cloud for valid 3D points (green)
#     valid_point_cloud = o3d.geometry.PointCloud()
#     if len(pts_3d_valid) > 0:
#         valid_point_cloud.points = o3d.utility.Vector3dVector(pts_3d_valid)
#         valid_point_cloud.paint_uniform_color([0.0, 1.0, 0.0])  # Green for valid points
    
#     return n_in_front, pts_3d_valid
    
#     # Camera 1 (Main) - at origin with identity rotation
#     cam1_frame = create_camera_visualization(np.eye(3), np.zeros(3), size=0.1, name="main")
    
#     # Camera 2 (Secondary) - positioned according to R and t
#     # Camera 2 position is -R.T @ t
#     cam2_position = -R.T @ t
#     # The rotation for camera 2 in world coordinates
#     cam2_rotation = R.T
#     cam2_frame = create_camera_visualization(cam2_rotation, cam2_position, size=0.1, name="secondary")
    
#     # Create a visualizer
#     vis = o3d.visualization.Visualizer()
#     vis.create_window(window_name=f"Configuration: {config_name} - {n_in_front} points in front")
    
#     # Add geometries to the visualizer
#     vis.add_geometry(all_point_cloud)
#     vis.add_geometry(valid_point_cloud)
#     # vis.add_geometry(cam1_frame)
#     # vis.add_geometry(cam2_frame)
    
#     # Set initial viewpoint
#     ctr = vis.get_view_control()
#     if len(pts_3d_valid) > 0:
#         ctr.set_lookat(np.mean(pts_3d_valid, axis=0))
#     else:
#         ctr.set_lookat(np.mean(pts_3d, axis=0))
#     ctr.set_front([0, 0, -1])
#     ctr.set_up([0, -1, 0])
#     ctr.set_zoom(0.8)
    
#     # Run the visualizer
#     vis.run()
#     vis.destroy_window()
    
#     return n_in_front, pts_3d_valid

# def show_splines(splines):
#     """ Visualize the 3D splines using o3d """
#     # Splines is a list of (spline_x, spline_y, spline_z, (start_ts, end_ts))
#     vis = o3d.visualization.Visualizer()
#     vis.create_window(window_name="3D Trajectory Spline Interpolation")
    
#     for spline_x, spline_y, spline_z, (start_ts, end_ts) in splines:
#         # Generate points along the spline
#         ts = np.linspace(start_ts, end_ts, num=100)
#         points = np.vstack((spline_x(ts), spline_y(ts), spline_z(ts))).T
        
#         # Create a line set for visualization
#         lines = [[i, i + 1] for i in range(len(points) - 1)]
#         colors = [[1, 0, 0] for _ in range(len(lines))]  # Red color for the spline
        
#         line_set = o3d.geometry.LineSet()
#         line_set.points = o3d.utility.Vector3dVector(points)
#         line_set.lines = o3d.utility.Vector2iVector(lines)
#         line_set.colors = o3d.utility.Vector3dVector(colors)
        
#         vis.add_geometry(line_set)
    
#     vis.run()
#     vis.destroy_window()

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

for iteration in range(5):  # Iterate to refine F and beta
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
                                  np.array([y for _, y, _, _ in correspondences]))
                                  
    correspondences = [correspondences[i] for i in range(len(correspondences)) if mask[i] == 1 and correspondences[i][2][0] is not None]
    
    # Estimate the time shift
    beta = estimate_time_shift(F,
                            np.array([x for x, _, _, _ in correspondences]), 
                            np.array([y for _, y, _, _ in correspondences]), 
                            np.array([v for _, _, v, _ in correspondences]))
    print(f"Iteration {iteration + 1}: Estimated time shift beta = {beta}")


# Update the global timestamps for the secondary camera in the dataframe and in the correspondences
df.loc[df['cam_id'] == secondary_camera, 'global_ts'] = compute_global_time(
    df.loc[df['cam_id'] == secondary_camera, 'frame_id'], 
    camera_info[secondary_camera].fps, 
    beta
)

df_filtered = df[df['cam_id'].isin([main_camera, secondary_camera])]
df_filtered.to_csv('detections.csv', index=False)


E = camera_info[secondary_camera].K_matrix.T @ F @ camera_info[main_camera].K_matrix
pts1_camera_coord = np.array([(np.linalg.inv(camera_info[main_camera].K_matrix) @ np.array([x[0], x[1], 1]))[:2] for x, _, _, _ in correspondences], dtype=np.float32)
pts2_camera_coord = np.array([(np.linalg.inv(camera_info[secondary_camera].K_matrix) @ np.array([y[0], y[1], 1]))[:2] for _, y, _, _ in correspondences], dtype=np.float32)
retval, R, t, mask, pts_3d = cv.recoverPose(E, pts1_camera_coord, pts2_camera_coord, np.eye(3), distanceThresh=10.0)

pts_3d = pts_3d.T

# Keep only the valid correspondences
correspondences = [correspondences[i] for i in range(len(correspondences)) if mask[i] == 255]

correspondences = [correspondences[i] + (pts_3d[i],) for i in range(len(correspondences))] # List of (spline_x1, spline_y1), (x2, y2), (v2x, v2y), timestamps_2, 3D point

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
    spline_x = CubicSpline(tss, pts_3d_slice[:, 0])
    spline_y = CubicSpline(tss, pts_3d_slice[:, 1])
    spline_z = CubicSpline(tss, pts_3d_slice[:, 2])

    
    splines_3d.append((spline_x, spline_y, spline_z, (tss[0], tss[-1])))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for spline_x, spline_y, spline_z, tss in splines_3d:
    ts = np.linspace(tss[0], tss[1], num=100)
    points = np.vstack((spline_x(ts), spline_y(ts), spline_z(ts))).T
    ax.plot(points[:, 0], points[:, 1], points[:, 2])

# show_splines(splines_3d) 

# Re project the 3D points to the primary camera

pts_3d = np.array([x[4] for x in correspondences])
pts_3d = pts_3d.T
P1 = np.dot(camera_info[main_camera].K_matrix, np.hstack((np.eye(3), np.zeros((3, 1)))))
pts_2d = np.dot(P1, pts_3d)
pts_2d = pts_2d / pts_2d[2]

# Plot the reprojected points
plt.figure(figsize=figsize)
plt.scatter(pts_2d[0], main_camera_height - pts_2d[1], c='r', s=1)
plt.xlim(0, main_camera_width)
plt.ylim(0, main_camera_height)
plt.title("Reprojected 3D points to Main Camera")
plt.xlabel("X")
plt.ylabel("Y")

plt.savefig('plots/reprojected_points.png')


# plt.show()