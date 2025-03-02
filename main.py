import json
import numpy as np
from scipy.interpolate import CubicSpline
from matplotlib import pyplot as plt
import cv2 as cv
import pandas as pd
import open3d as o3d

from camera_info import Camera_info

MAIN_CAMERA = 0
SECONDARY_CAMERA = 2

def interpolate_trajectory(detections, time_stamps):
    """
    Interpolate the UAV's 2D trajectory using cubic splines.
    
    detections: List of (frame_id, x, y) for a camera
    time_stamps: Corresponding frame timestamps
    
    Returns: Spline function for interpolation
    """
    detections = np.array(detections)
    spline_x = CubicSpline(time_stamps, detections[:, 1])
    spline_y = CubicSpline(time_stamps, detections[:, 2])

    
    return spline_x, spline_y

def compute_global_time(frame_indices, alpha, beta = 0):
    """
    Convert frame indices to global timestamps.
    
    frame_indices: List or array of frame indices
    alpha: Time scale factor (frame rate correction)
    beta: Time offset (initial shift)
    
    Returns: Global timestamps
    """
    return frame_indices / alpha + beta

def load_detections(camera_id):
    """
    Get detections for a camera, reading the file detections/cam{camera_id}.txt
    
    camera_id: Camera id
    
    Returns: List of (frame_id, x, y)
    """
    detections = []
    with open(f'drone-tracking-datasets/dataset1/detections/cam{camera_id}.txt', 'r') as f:
        for line in f:
            frame_id, x, y = map(float, line.strip().split())
            detections.append((frame_id, x, y))
    return detections

def get_camera_info(camera_name):
    """
    Get camera information from the camera name.
    
    camera_name: Camera name
    
    Returns: object retrieved from the json file
    """
    with open(f'drone-tracking-datasets/calibration/{camera_name}/{camera_name}.json', 'r') as f:
        data = json.load(f)
    ret = Camera_info(data['comment'], np.array(data['K-matrix']), np.array(data['distCoeff']), data['fps'], data['resolution'])
    return ret

def find_contiguous_regions(detections):
    """
    Find the time intervals where the UAV is visible in the camera (x and y are not 0.0)
    
    detections: List of (frame_id, x, y)
    
    Returns: List of (start_frame, end_frame)
    """
    regions = []
    start_frame = None
    for i, (frame_id, x, y) in enumerate(detections):
        if x != 0.0 and y != 0.0:
            if start_frame is None:
                start_frame = frame_id
        elif start_frame is not None:
            regions.append((start_frame, frame_id))
            start_frame = None
    if start_frame is not None and detections[-1][0] - start_frame > 3:
        regions.append((start_frame, detections[-1][0]))
    return regions
       
def load_dataframe(cameras):
    splines = []
    camera_info = []
    contiguous = []
    data = {
        'cam_id': [],
        'cam_name': [],
        'frame_id': [],
        'global_ts': [],
        'detection_x': [],
        'detection_y': [],
        'velocity_x': [],
        'velocity_y': []
    }
    for i, camera in enumerate(cameras):
        camera_info_i = get_camera_info(camera)
        detections_i = load_detections(i)
        frame_indices_i = np.array([frame_id for frame_id, _, _ in detections_i])
        contiguous_i = find_contiguous_regions(detections_i)
        splines_i = []
        for start_frame, end_frame in contiguous_i:
            timestamps_j = compute_global_time(frame_indices_i[int(start_frame-1):int(end_frame-1)], camera_info_i.fps)
            spline_x, spline_y = interpolate_trajectory(detections_i[int(start_frame-1):int(end_frame-1)], timestamps_j)
            splines_i.append((spline_x, spline_y, timestamps_j))
        splines.append(splines_i)
        camera_info.append(camera_info_i)
        contiguous.append(contiguous_i)    
        for frame_id, x, y in detections_i:
            global_ts = compute_global_time(np.array([frame_id]), camera_info_i.fps)[0]
            data['cam_id'].append(i)
            data['cam_name'].append(camera)
            data['frame_id'].append(frame_id)
            data['detection_x'].append(x)
            data['detection_y'].append(y)
            data['global_ts'].append(global_ts)
            
            # Set default velocity to None
            data['velocity_x'].append(None)
            data['velocity_y'].append(None)

        # After processing all detections, compute forward velocity
        for j in range(len(data['frame_id']) - 1):
            # Check if we're looking at consecutive frames from the same camera
            if (data['cam_id'][j] == data['cam_id'][j+1] and 
                data['detection_x'][j] != 0.0 and data['detection_y'][j] != 0.0 and
                data['detection_x'][j+1] != 0.0 and data['detection_y'][j+1] != 0.0 and
                data['global_ts'][j+1] - data['global_ts'][j] > 0):
                
                # Calculate velocity (forward difference)
                velocity_x = (data['detection_x'][j+1] - data['detection_x'][j]) / (data['global_ts'][j+1] - data['global_ts'][j])
                velocity_y = (data['detection_y'][j+1] - data['detection_y'][j]) / (data['global_ts'][j+1] - data['global_ts'][j])
                
                data['velocity_x'][j] = velocity_x
                data['velocity_y'][j] = velocity_y

    print("To dataframe")
    df = pd.DataFrame(data)
    print("Writing")
    # Filter the dataframe to include only primary and secondary cameras
    df_filtered = df[df['cam_id'].isin([MAIN_CAMERA, SECONDARY_CAMERA])]
    df_filtered.to_csv('detections.csv', index=False)
    print("Done")
    return df, splines, contiguous, camera_info

def estimate_time_shift(F, xi, xk, vk):
    """
    Estimate the time shift between two cameras using RANSAC.
    
    F: Fundamental matrix
    xi: Points from camera i
    xk: Points from camera k
    vk: Velocities from camera k
    
    Returns: Time shift beta and inliers
    """
    
    beta_values = []
    
    for x1, x2, v2 in zip(xi, xk, vk):
        x1_h = np.append(x1, 1)
        x2_h = np.append(x2, 1)
        v2_h = np.append(v2, 0)
        
        numerator = np.dot(x2_h.T, np.dot(F, x1_h))
        denominator = np.dot(v2_h.T, np.dot(F, x1_h))
        
        if np.abs(denominator) > 1e-6:  # Avoid division by zero
            beta_ik = -numerator / denominator
            beta_values.append(beta_ik)
            
    # Plot all beta values
    plt.figure()
    plt.plot(beta_values)
    plt.hlines(np.median(beta_values), 0, len(beta_values), colors='r', linestyles='dashed')
    plt.title("Beta values")
    plt.xlabel("Correspondence index")
    plt.ylabel("Beta")
    
    return np.median(beta_values) 

def triangulate_points(P1, P2, pts1, pts2, K1, K2):
    """
    Triangulate points after normalizing them using camera intrinsics.
    
    P1: Projection matrix for camera 1
    P2: Projection matrix for camera 2
    pts1: Points in image 1 (Nx2)
    pts2: Points in image 2 (Nx2)
    K1: Intrinsic matrix of camera 1
    K2: Intrinsic matrix of camera 2
    
    Returns: Nx3 array of 3D points
    """
    # Convert points to homogeneous coordinates
    pts1_hom = np.hstack((pts1, np.ones((pts1.shape[0], 1)))).T  # Shape (3, N)
    pts2_hom = np.hstack((pts2, np.ones((pts2.shape[0], 1)))).T  # Shape (3, N)

    # Normalize image coordinates (convert to camera space)
    pts1_norm = np.linalg.inv(K1) @ pts1_hom  # (3, N)
    pts2_norm = np.linalg.inv(K2) @ pts2_hom  # (3, N)

    # Triangulate points
    pts_4d_hom = cv.triangulatePoints(P1, P2, pts1_norm[:2], pts2_norm[:2])  # OpenCV requires 2D (x,y)
    
    # Convert from homogeneous coordinates
    pts_3d_hom = pts_4d_hom / pts_4d_hom[3]
    pts_3d = pts_3d_hom[:3].T  # Convert to (N, 3)

    return pts_3d

def in_front_of_both_cameras(R, t, K1, K2, pts1, pts2):
    """
    Check if the triangulated points are in front of both cameras.
    
    R: Rotation matrix from camera 1 to camera 2
    t: Translation vector from camera 1 to camera 2
    K1: Intrinsic matrix of camera 1
    K2: Intrinsic matrix of camera 2
    pts1: Points in camera 1
    pts2: Points in camera 2
    
    Returns: Number of points in front of both cameras and the 3D points
    """
    P1 = np.dot(K1, np.hstack((np.eye(3), np.zeros((3, 1)))))
    P2 = np.dot(K2, np.hstack((R, t.reshape(3, 1))))
    
    pts_3d = triangulate_points(P1, P2, pts1, pts2, K1, K2)
    
    # Check if points are in front of camera 1
    pts_3d_in_cam1 = pts_3d.copy()
    in_front_cam1 = pts_3d_in_cam1[:, 2] > 0
    n_in_front_cam1 = np.sum(in_front_cam1)
    
    # Check if points are in front of camera 2
    pts_3d_in_cam2 = np.dot(R, pts_3d.T).T + t.reshape(1, 3)
    in_front_cam2 = pts_3d_in_cam2[:, 2] > 0
    n_in_front_cam2 = np.sum(in_front_cam2)
    
    # Points in front of both cameras
    in_front_both = np.logical_and(in_front_cam1, in_front_cam2)
    pts_3d_valid = pts_3d[in_front_both]
    
    # Return the number of points in front of both cameras and the 3D points
    return min(n_in_front_cam1, n_in_front_cam2), pts_3d, pts_3d_valid

# Create Open3D visualization for camera
def create_camera_visualization(R, t, size=0.1, color=[0, 0, 0], name="Camera"):
    """Create a camera visualization with coordinate axes."""
    # Create a coordinate frame
    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    
    # Apply the rotation and translation
    camera_frame.translate(t)
    camera_frame.rotate(R)
    return camera_frame

# Function to visualize a configuration
def visualize_configuration(R, t, K1, K2, pts1, pts2, config_name):
    """
    Visualize a specific R,t configuration with Open3D.
    
    R: Rotation matrix from camera 1 to camera 2
    t: Translation vector from camera 1 to camera 2
    K1: Intrinsic matrix of camera 1
    K2: Intrinsic matrix of camera 2
    pts1: Points in camera 1
    pts2: Points in camera 2
    config_name: Name of this configuration (for window title)
    """
    # Get 3D points and count points in front of cameras
    n_in_front, pts_3d, pts_3d_valid = in_front_of_both_cameras(R, t, K1, K2, pts1, pts2)
    
    # Create point cloud for all 3D points (blue)
    all_point_cloud = o3d.geometry.PointCloud()
    all_point_cloud.points = o3d.utility.Vector3dVector(pts_3d)
    all_point_cloud.paint_uniform_color([0.7, 0.7, 1.0])  # Light blue for all points
    
    # Create point cloud for valid 3D points (green)
    valid_point_cloud = o3d.geometry.PointCloud()
    if len(pts_3d_valid) > 0:
        valid_point_cloud.points = o3d.utility.Vector3dVector(pts_3d_valid)
        valid_point_cloud.paint_uniform_color([0.0, 1.0, 0.0])  # Green for valid points
    
    return n_in_front, pts_3d_valid
    
    # Camera 1 (Main) - at origin with identity rotation
    cam1_frame = create_camera_visualization(np.eye(3), np.zeros(3), size=0.1, name="main")
    
    # Camera 2 (Secondary) - positioned according to R and t
    # Camera 2 position is -R.T @ t
    cam2_position = -R.T @ t
    # The rotation for camera 2 in world coordinates
    cam2_rotation = R.T
    cam2_frame = create_camera_visualization(cam2_rotation, cam2_position, size=0.1, name="secondary")
    
    # Create a visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Configuration: {config_name} - {n_in_front} points in front")
    
    # Add geometries to the visualizer
    vis.add_geometry(all_point_cloud)
    vis.add_geometry(valid_point_cloud)
    # vis.add_geometry(cam1_frame)
    # vis.add_geometry(cam2_frame)
    
    # Set initial viewpoint
    ctr = vis.get_view_control()
    if len(pts_3d_valid) > 0:
        ctr.set_lookat(np.mean(pts_3d_valid, axis=0))
    else:
        ctr.set_lookat(np.mean(pts_3d, axis=0))
    ctr.set_front([0, 0, -1])
    ctr.set_up([0, -1, 0])
    ctr.set_zoom(0.8)
    
    # Run the visualizer
    vis.run()
    vis.destroy_window()
    
    return n_in_front, pts_3d_valid

with open('drone-tracking-datasets/dataset1/cameras.txt', 'r') as f:
    cameras = f.read().strip().split()    
cameras = cameras[2::3]

df, splines, contiguous, camera_info = load_dataframe(cameras)

plt.figure()
for spline_x, spline_y, tss in splines[MAIN_CAMERA]:
    plt.plot(spline_x(tss), 1080-spline_y(tss))
plt.xlim(0, 1920)
plt.ylim(0, 1080)
    
correspondences = [] # List of (x1, y1), (x2, y2), (v2x, v2y), timestamps_2
det = df[df['cam_id'] == SECONDARY_CAMERA][['frame_id', 'detection_x', 'detection_y', 'velocity_x', 'velocity_y', 'global_ts']].values

plt.figure()

for frame in det:
    if frame[1] == 0.0 and frame[2] == 0.0:
        continue
    global_ts = compute_global_time(frame[0], camera_info[SECONDARY_CAMERA].fps)
    
    # find the spline of cam0 that contains the global_ts
    found = False
    for spline_x, spline_y, tss in splines[MAIN_CAMERA]:
        if np.min(tss) <= global_ts <= np.max(tss):
            found = True
            correspondences.append(((frame[1], frame[2]), (float(spline_x(global_ts)), float(spline_y(global_ts))), (frame[3],frame[4]), global_ts))
            plt.scatter(float(spline_x(global_ts)), 1080-float(spline_y(global_ts)), s=1, c='black')
            break
    # if not found:
    #     print("No spline found for frame", frame[0])
    #     continue
    
plt.xlim(0, 1920)
plt.ylim(0, 1080)
    
# Initial estimation of the fundamental matrix
F, mask = cv.findFundamentalMat(np.array([x for x, _, _, _ in correspondences]), np.array([y for _, y, _, _ in correspondences]), cv.FM_RANSAC)

# Iteratively refine F and beta
for _ in range(1):  # Perform 5 iterations
    # Keep only the inliers
    correspondences = [correspondences[i] for i in range(len(correspondences)) if mask[i] == 1 and correspondences[i][2][0] is not None]

    # Estimate the time shift
    beta = estimate_time_shift(F, 
                            np.array([x for x, _, _, _ in correspondences]), 
                            np.array([y for _, y, _, _ in correspondences]), 
                            np.array([v for _, _, v, _ in correspondences]))

    print("Estimated time shift:", beta)

    # Update the global timestamps for the secondary camera in the dataframe and in the correspondences
    df.loc[df['cam_id'] == SECONDARY_CAMERA, 'global_ts'] = compute_global_time(
        df.loc[df['cam_id'] == SECONDARY_CAMERA, 'frame_id'], 
        camera_info[SECONDARY_CAMERA].fps, 
        beta
    )

    # Recompute F using the updated timestamps and remove outliers
    F, mask = cv.findFundamentalMat(np.array([x for x, _, _, _ in correspondences]), 
                                    np.array([y for _, y, _, _ in correspondences]), 
                                    cv.FM_RANSAC)

    correspondences = [correspondences[i] for i in range(len(correspondences)) if mask[i] == 1]

print("Final estimated time shift:", beta)

# Update the global timestamps for the secondary camera in the dataframe and in the correspondences
df.loc[df['cam_id'] == SECONDARY_CAMERA, 'global_ts'] = compute_global_time(
    df.loc[df['cam_id'] == SECONDARY_CAMERA, 'frame_id'], 
    camera_info[SECONDARY_CAMERA].fps, 
    beta
)

df_filtered = df[df['cam_id'].isin([MAIN_CAMERA, SECONDARY_CAMERA])]
df_filtered.to_csv('detections.csv', index=False)


E = camera_info[SECONDARY_CAMERA].K_matrix.T @ F @ camera_info[MAIN_CAMERA].K_matrix

R1, R2, t = cv.decomposeEssentialMat(E)

# Convert the correspondences to the right format
pts1 = np.array([np.array(x) for x, _, _, _ in correspondences])
pts2 = np.array([np.array(y) for _, y, _, _ in correspondences])

# Check all four configurations to see which one places most points in front of both cameras
configs = [
    (R1, t, "R1, t"), 
    (R1, -t, "R1, -t"), 
    (R2, t, "R2, t"), 
    (R2, -t, "R2, -t")
]

max_in_front = -1
best_config = None
best_pts_3d = None

# Visualize each configuration one by one
for i, (R, t, name) in enumerate(configs):
    print(f"\nEvaluating configuration {i+1}: {name}")
    
    # Visualize this configuration and get the number of points in front
    n_in_front, pts_3d_valid = visualize_configuration(
        R, t, 
        camera_info[MAIN_CAMERA].K_matrix, 
        camera_info[SECONDARY_CAMERA].K_matrix, 
        pts1, pts2,
        name
    )
    
    print(f"Configuration {name}: {n_in_front} points in front of both cameras")
    
    if n_in_front > max_in_front:
        max_in_front = n_in_front
        best_config = (R, t)
        best_pts_3d = pts_3d_valid

# Use the best configuration
R_correct, t_correct = best_config
print(f"\nSelected best configuration with {max_in_front} points in front of both cameras")
print("Rotation matrix:")
print(R_correct)
print("Translation vector:")
print(t_correct)

# Triangulate the 3D points using the correct configuration
P1 = np.dot(camera_info[MAIN_CAMERA].K_matrix, np.hstack((np.eye(3), np.zeros((3, 1)))))
P2 = np.dot(camera_info[SECONDARY_CAMERA].K_matrix, np.hstack((R_correct, t_correct.reshape(3, 1))))

pts_3d = best_pts_3d

# Convert 3D points to homogeneous coordinates (4 x N)
pts_3d_hom = np.vstack((pts_3d.T, np.ones((1, pts_3d.shape[0]))))  # Shape: (4, N)

# Project the 3D points into the secondary camera image plane using P2
pts_2d_hom = P2 @ pts_3d_hom  # Matrix multiplication (3x4) @ (4xN) -> (3xN)

# Convert from homogeneous to pixel coordinates
pts_2d = (pts_2d_hom[:2] / pts_2d_hom[2]).T  # Shape: (N, 2)

# Plot the reprojected points
plt.figure()
plt.scatter(pts_2d[:, 0], 1080 - pts_2d[:, 1], s=1, c='black')  # Flip y-axis for visualization
plt.xlim(0, 1920)
plt.ylim(0, 1080)
plt.title("Reprojected 3D Points in Image Coordinates")
plt.show()


correspondences = [correspondences[i] + (pts_3d[i],) for i in range(len(correspondences))]

# Create point cloud for the 3D points
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(pts_3d)
point_cloud.paint_uniform_color([0.0, 0.0, 1.0])  # Blue points

# Create visualizations for the cameras
# Camera 1 (Main) - at origin with identity rotation
cam1_frame = create_camera_visualization(np.eye(3), np.zeros(3), size=0.1, name="main")

# Camera 2 (Secondary) - positioned according to R_correct and t_correct
# Camera 2 position is -R_correct.T @ t_correct
cam2_position = -R_correct.T @ t_correct
# The rotation for camera 2 in world coordinates
cam2_rotation = R_correct.T
cam2_frame = create_camera_visualization(cam2_rotation, cam2_position, size=0.1, name="secondary")

# Create a visualizer for the final result
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Final 3D Reconstruction with Camera Poses")

# Add geometries to the visualizer
vis.add_geometry(point_cloud)
# vis.add_geometry(cam1_frame)
# vis.add_geometry(cam2_frame)

# Set initial viewpoint
ctr = vis.get_view_control()
ctr.set_lookat(np.mean(pts_3d, axis=0))
ctr.set_front([0, 0, -1])
ctr.set_up([0, -1, 0])
ctr.set_zoom(0.8)

# Run the visualizer
vis.run()
vis.destroy_window()

plt.show()