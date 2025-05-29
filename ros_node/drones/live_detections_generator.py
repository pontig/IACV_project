import os
import logging
from datetime import datetime
import sys
import numpy as np
import pandas as pd

# from global_fn import *
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import rosbag2_py
from rclpy.serialization import serialize_message

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


def compute_global_time(frame_indices, alpha, beta = 0):
    """
    Convert frame indices to global timestamps and update min/max global time.
    
    frame_indices: List or array of frame indices
    alpha: Time scale factor (frame rate correction)
    beta: Time offset (initial shift)
    
    Returns: Global timestamps
    """
    global_times = frame_indices * alpha + beta
    return global_times
    
    
def main():
    rclpy.init()
    node = Node("live_detections_generator")
    node.get_logger().info("Starting live detections generator")
    logging.info("Starting live detections generator")
    camera_poses = []

    # # Load camera info
    # logging.info("Loading camera info")
    # with open(f"drone-tracking-datasets/dataset{DATASET_NO}/cameras.txt", 'r') as f:
    #     cameras = f.read().strip().split()    
    # cameras = cameras[2::3]
    # logging.info(f"Loaded {len(cameras)} cameras")

    # Load dataframe and splines
    logging.info("Loading dataframe and splines")
    df = pd.read_csv(f"detections.csv")

    if DATASET_NO == 4:
        alpha = np.array([
            [1.0000, 0.4983, 0.5005, 0.4999, 0.5000, 0.8342, 0.4171],
            [2.0069, 1.0000, 1.0044, 1.0033, 1.0034, 1.6741, 0.8370],
            [1.9981, 0.9956, 1.0000, 0.9989, 0.9989, 1.6667, 0.8334],
            [2.0003, 0.9967, 1.0011, 1.0000, 1.0001, 1.6686, 0.8349],
            [2.0002, 0.9966, 1.0011, 0.9999, 1.0000, 1.6685, 0.8343],
            [1.1988, 0.5973, 0.6000, 0.5993, 0.5993, 1.0000, 0.5000],
            [2.3975, 1.1947, 1.2000, 1.1978, 1.1986, 2.0000, 1.0000]
        ])

        beta = np.array([
            [0.00,     1156.29, 1128.78, 1219.88,  889.37, 3018.11, -1562.26],
            [-2320.60, 0.00,    -32.64,    59.77, -270.82, 1082.34, -2529.59],
            [-2255.38, 32.50,     0.00,    92.38, -238.21, 1136.75, -2502.62],
            [-2440.13, -59.56,  -92.46,     0.00, -330.57,  982.64, -2587.35],
            [-1778.91, 269.91,  238.46,   330.57,    0.00, 1534.20, -2304.50],
            [-3618.11, -646.52, -682.02, -588.87, -919.51,    0.00, -3071.00],
            [3745.56, 3022.07, 3003.04, 3099.07, 2762.22, 6142.00,     0.00]
        ])

    df = df[(df['detection_x'] != 0.0) & (df['detection_y'] != 0.0)]

    for i in range(0, 7):
        df_camera = df[df['cam_id'] == i]
        frame_ids = df_camera['frame_id'].values
        global_ts = compute_global_time(frame_ids, alpha[i, 0], beta[i, 0])
        df.loc[df['cam_id'] == i, 'global_ts'] = global_ts
        
    df = df.sort_values(by=['global_ts'])
    df.to_csv('detections.csv', index=False)

    # Convert dataframe to a sorted list of rows and write to bag
    df_sorted = df.sort_values(by=['global_ts'])
    # Publisher for detections
    publisher = node.create_publisher(Float32MultiArray, '/detections', 10)

    # Publish each detection as a message
    for _, row in df_sorted.iterrows():
        msg = Float32MultiArray()
        msg.data = [row['global_ts']/59.940060, float(row['cam_id']), row['detection_x'], row['detection_y']] # Timestamp in seconds
        logging.info(f"Publishing detection: {msg.data}")
        publisher.publish(msg)
        rclpy.spin_once(node, timeout_sec=0.01)  # Allow ROS to process events

    node.get_logger().info("All detections published.")
    rclpy.shutdown()