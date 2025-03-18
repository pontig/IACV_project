import json
import numpy as np
from scipy.interpolate import CubicSpline, make_interp_spline
from matplotlib import pyplot as plt
import cv2 as cv
import pandas as pd
import open3d as o3d

from camera_info import Camera_info
from global_fn import *

def interpolate_trajectory(detections, time_stamps):
    """
    Interpolate the UAV's 2D trajectory using cubic splines.
    
    detections: List of (frame_id, x, y) for a camera
    time_stamps: Corresponding frame timestamps
    
    Returns: Spline function for interpolation
    """
    detections = np.array(detections)
    spline_x = make_interp_spline(time_stamps, detections[:, 1], k=3)
    spline_y = make_interp_spline(time_stamps, detections[:, 2], k=3)

    
    return spline_x, spline_y

def load_detections(camera_id, dataset_no):
    """
    Get detections for a camera, reading the file detections/cam{camera_id}.txt
    
    camera_id: Camera id
    
    Returns: List of (frame_id, x, y)
    """
    detections = []
    with open(f'drone-tracking-datasets/dataset{dataset_no}/detections/cam{camera_id}.txt', 'r') as f:
        for line in f:
            frame_id, x, y = map(float, line.strip().split())
            # if frame_id > 946:
            #     detections.append((frame_id - 946, x, y))
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
       
def load_dataframe(cameras, dataset_no):
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
    plt.figure(figsize=(19, 10))
    for i, camera in enumerate(cameras):
        camera_info_i = get_camera_info(camera)
        detections_i = load_detections(i, dataset_no)
        
        new_camera_matrix, roi = cv.getOptimalNewCameraMatrix(camera_info_i.K_matrix, camera_info_i.distCoeff, camera_info_i.resolution, 0, camera_info_i.resolution)
        dst = cv.undistortPoints(np.array([[x, y] for _, x, y in detections_i if x != 0.0 and y != 0.0]).reshape(-1, 1, 2), camera_info_i.K_matrix, camera_info_i.distCoeff, P=new_camera_matrix)
        cnt = 0
        for j in range(len(detections_i)):
            if detections_i[j][1] != 0.0 and detections_i[j][2] != 0.0:
                detections_i[j] = (detections_i[j][0], dst[cnt][0][0], dst[cnt][0][1])
                cnt += 1
        image = cv.imread(f'drone-tracking-datasets/dataset{dataset_no}/cam{i}.jpg')
        image = cv.undistort(image, camera_info_i.K_matrix, camera_info_i.distCoeff, None, new_camera_matrix)
        
        camera_info_i.K_matrix = new_camera_matrix
        camera_info_i.distCoeff = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        
        xx, yy, ww, hh = roi
        image = image[yy:yy+hh, xx:xx+ww]
        # Scatter plot of detections on the image
        for frame_id, x, y in detections_i:
            if x != 0.0 and y != 0.0:
                cv.circle(image, (int(x), int(y)), 2, (0, 255, 0), -1)
        
        cv.imwrite(f'plots/detections_on_image_camera_{i}.png', image)
        
        # Scatter plot of detections_i
        plt.scatter([x for _, x, _ in detections_i], [-y for _, _, y in detections_i], label=f'Camera {i}', s=1)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'Detections for Camera {i}')
        plt.xlim(0, camera_info_i.resolution[0])
        plt.ylim(-camera_info_i.resolution[1], 0)

        plt.savefig(f'plots/detections_camera_{i}.png')
        plt.clf()
        
        # detections_i = [(frame_id, 
        #          x * camera_info_i.K_matrix[0, 0] + camera_info_i.K_matrix[0, 2] if x != 0.0 else 0.0, 
        #          y * camera_info_i.K_matrix[1, 1] + camera_info_i.K_matrix[1, 2] if y != 0.0 else 0.0) 
        #         for frame_id, x, y in detections_i]
        frame_indices_i = np.array([frame_id for frame_id, _, _ in detections_i])
        contiguous_i = find_contiguous_regions(detections_i)
        splines_i = []
        for start_frame, end_frame in contiguous_i:
            timestamps_j = compute_global_time(frame_indices_i[int(start_frame-1):int(end_frame-1)], alphas[i], betas[i])
            if len(timestamps_j) > 3:
                spline_x, spline_y = interpolate_trajectory(detections_i[int(start_frame-1):int(end_frame-1)], timestamps_j)
                splines_i.append((spline_x, spline_y, timestamps_j))
        splines.append(splines_i)
        camera_info.append(camera_info_i)
        contiguous.append(contiguous_i)
        
        # Plot splines_i
        for spline_x, spline_y, timestamps_j in splines_i:
            ts_dense = timestamps_j
            x_dense = spline_x(ts_dense)
            y_dense = spline_y(ts_dense)
            plt.plot(x_dense, -y_dense, label=f'Times: {timestamps_j[0]:.2f} - {timestamps_j[-1]:.2f}')
        
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'Splines for Camera {i}')
        plt.xlim(0, camera_info_i.resolution[0])
        plt.ylim(-camera_info_i.resolution[1], 0)
        plt.savefig(f'plots/splines_camera_{i}.png')
        plt.clf()
        
        for frame_id, x, y in detections_i:
            global_ts = compute_global_time(np.array([frame_id]), alphas[i], betas[i])[0]
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
    df.to_csv('detections.csv', index=False)
    print("Done")
    return df, splines, contiguous, camera_info


def find_max_overlapping(df, contiguous):
    """
    Find the pair of cameras with the maximum overlapping time.
    
    df: DataFrame with detections
    contiguous: List of contiguous time intervals for each camera
    
    Returns: Camera ids with maximum overlapping time
    """
    max_overlap = 0
    prim, sec = None, None
    
    for i in range(len(contiguous)):
        for j in range(len(contiguous)):
            if i == j:
                continue
            overlap = 0
            for ts in df[df['cam_id'] == i]['global_ts']:
                for start, end in contiguous[j]:
                    if start <= ts <= end:
                        overlap += 1
                        break
            if overlap > max_overlap:
                max_overlap = overlap
                prim, sec = i, j
    return prim, sec, max_overlap
                        