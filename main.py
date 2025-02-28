import json
import numpy as np
from scipy.interpolate import CubicSpline
from matplotlib import pyplot as plt
import cv2 as cv

from camera_info import Camera_info
import pandas as pd

MAIN_CAMERA = 0
SECONDARY_CAMERA = 1

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
            splines_i.append((spline_x, spline_y))
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
    df.to_csv('detections.csv', index=False)
    print("Done")
    return df, splines, contiguous, camera_info

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
    return frame_indices * alpha + beta

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
    
def estimate_time_shift(F, matches_cam1, matches_cam2, velocities_cam2):
    """
    Estimates the time shift beta_ik between two cameras.
    
    F: Fundamental matrix (3x3)
    matches_cam1: Nx2 array of 2D points in camera 1
    matches_cam2: Nx2 array of corresponding 2D points in camera 2
    velocities_cam2: Nx2 array of UAV velocities in camera 2 (image-plane velocity)
    
    Returns: Estimated time shifts (beta_ik) for each correspondence
    """
    beta_values = []
    
    for x1, x2, v2 in zip(matches_cam1, matches_cam2, velocities_cam2):
        x1_h = np.append(x1, 1)
        x2_h = np.append(x2, 1)
        v2_h = np.append(v2, 0)  # Assume velocity is 2D, so add zero for homogeneity
        
        numerator = np.dot(x2_h.T, np.dot(F, x1_h))
        denominator = np.dot(v2_h.T, np.dot(F, x1_h))
        
        if np.abs(denominator) > 1e-6:  
            beta_ik = -numerator / denominator
            beta_values.append(beta_ik)
       
    
    return np.median(beta_values)
    
with open('drone-tracking-datasets/dataset1/cameras.txt', 'r') as f:
    cameras = f.read().strip().split()    
cameras = cameras[2::3]

df, splines, contiguous, camera_info = load_dataframe(cameras)

for i, (spline_x, spline_y) in enumerate(splines[MAIN_CAMERA]):
    start_frame, end_frame = contiguous[MAIN_CAMERA][i]
    
    #compute global tame for each frame in the interval
    time_stamps = compute_global_time(np.arange(start_frame, end_frame), camera_info[MAIN_CAMERA].fps)
    plt.plot(spline_x(time_stamps), 1080-spline_y(time_stamps))
    plt.xlim(0, 1920)
    plt.ylim(0, 1080)
    
correspondences = [] # List of (x1, y1), (x2, y2), (v1, v2)
det = df[df['cam_id'] == SECONDARY_CAMERA][['frame_id', 'detection_x', 'detection_y', 'velocity_x', 'velocity_y']].values

for frame in det:
    if frame[1] == 0.0 and frame[2] == 0.0:
        continue
    global_ts = compute_global_time(frame[0], camera_info[SECONDARY_CAMERA].fps)
    
    # find the spline of cam0 that contains the global_ts
    found = False
    for spline_x, spline_y in splines[MAIN_CAMERA]:
        if spline_x(global_ts) is not None:
            found = True
            correspondences.append(((frame[1], frame[2]), (float(spline_x(global_ts)), float(spline_y(global_ts))), (frame[3],frame[4])))
            break
    if not found:
        print("No spline found for frame", frame[0])
        continue

correspondences_full = correspondences.copy()
    
# Initial estimation of the fundamental matrix
F, mask = cv.findFundamentalMat(np.array([x for x, _, _ in correspondences]), np.array([y for _, y, _ in correspondences]), cv.FM_RANSAC)

# Iteratively refine the fundamental matrix and time shift
F_diffs = []
beta_diffs = []
inliers_counts = []
beta_prev = 0

iteration = 0
consecutive_convergence = 0

while iteration < 100:    
    # Keep only inliers
    correspondences = [correspondences[i] for i in range(len(correspondences)) if mask[i] == 1 and correspondences[i][2][0] is not None and correspondences[i][2][1] is not None]
    
    beta = estimate_time_shift(F, 
                               np.array([x for x, _, _ in correspondences]), 
                               np.array([y for _, y, _ in correspondences]), 
                               np.array([v for _, _, v in correspondences]))
    
    matches_cam1 = np.array([x for x, _, _ in correspondences])
    matches_cam2 = np.array([y for _, y, _ in correspondences]) + beta * np.array([v for _, _, v in correspondences])
    
    F_new, mask = cv.findFundamentalMat(matches_cam1, matches_cam2, cv.FM_RANSAC)
    
    F_diff = np.linalg.norm(F_new - F)
    F_diffs.append(F_diff)
    
    beta_diff = np.abs(beta - beta_prev)
    beta_diffs.append(beta_diff)
    beta_prev = beta
    
    inliers_count = np.sum(mask)
    inliers_counts.append(inliers_count)
    
    F = F_new
    
    if F_diff < 1e-6 and beta_diff < 1e-6:
        consecutive_convergence += 1
    else:
        consecutive_convergence = 0
    if consecutive_convergence >= 5:
        break
    
    iteration += 1
E = camera_info[SECONDARY_CAMERA].K_matrix.T @ F @ camera_info[MAIN_CAMERA].K_matrix

print("Beta:", beta)

ret, R, t = cv.decomposeEssentialMat(E)
print("Rotation:\n", R)
print("Translation:\n", t)

# Triangulate points
points_3d = []
P1 = camera_info[MAIN_CAMERA].K_matrix @ np.hstack((np.eye(3), np.zeros((3, 1))))
P2 = camera_info[SECONDARY_CAMERA].K_matrix @ np.hstack((R, t))

for x1, x2, _ in correspondences_full:
    point_3d = cv.triangulatePoints(P1, P2, x1, x2)
    point_3d /= point_3d[3]
    points_3d.append(point_3d[:3])
    
points_3d = np.array(points_3d)

# Plot 3D points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D points')

plt.figure()
plt.plot(inliers_counts, marker='o')
plt.xlabel('Iteration')
plt.ylabel('Inliers count')
plt.title('Behavior of the inliers count')
plt.grid(True)


plt.figure()
plt.plot(F_diffs, marker='o', label='F differences')
plt.plot(beta_diffs, marker='x', label='Beta differences')
plt.xlabel('Iteration')
plt.ylabel('Difference')
plt.title('Behavior of the differences between refinements of F and Beta')
plt.legend()
plt.grid(True)

plt.show()
print("Done")    