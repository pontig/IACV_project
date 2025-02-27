import json
import numpy as np
from scipy.interpolate import CubicSpline
from matplotlib import pyplot as plt
import cv2 as cv

from camera_info import Camera_info

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
    
    
with open('drone-tracking-datasets/dataset1/cameras.txt', 'r') as f:
    cameras = f.read().strip().split()    
cameras = cameras[2::3]

cameras_info = []
detections = []
contiguous = []
splines = []

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
    plt.show()
    splines.append(splines_i)
    
    cameras_info.append(camera_info_i)
    detections.append(detections_i)
    contiguous.append(contiguous_i)

del detections_i, frame_indices_i, contiguous_i, splines_i, timestamps_j, spline_x, spline_y, start_frame, end_frame, camera_info_i

# for i, (spline_x, spline_y) in enumerate(splines[0]):
#     start_frame, end_frame = contiguous[0][i]
    
#     #compute global tame for each frame in the interval
#     time_stamps = compute_global_time(np.arange(start_frame, end_frame), cameras_info[0].fps)
#     plt.plot(spline_x(time_stamps), 1080-spline_y(time_stamps))
#     plt.xlim(0, 1920)
#     plt.ylim(0, 1080)
    
# plt.show()

correspondences = []

for frame in detections[1]:
    if frame[1] == 0.0 and frame[2] == 0.0:
        continue
    global_ts = compute_global_time(frame[0], cameras_info[1].fps)
    
    # find the spline of cam0 that contains the global_ts
    found = False
    for spline_x, spline_y in splines[0]:
        if spline_x(global_ts) is not None:
            found = True
            correspondences.append(((frame[1], frame[2]), (float(spline_x(global_ts)), float(spline_y(global_ts)))))        
            break
    if not found:
        print("No spline found for frame", frame[0])
        continue
    
F, mask = cv.findFundamentalMat(np.array([x for x, _ in correspondences]), np.array([y for _, y in correspondences]), cv.FM_RANSAC)

# Keep only inliers
correspondences = [correspondences[i] for i in range(len(correspondences)) if mask[i] == 1]

print("Done")    