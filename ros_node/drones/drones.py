import rclpy
from rclpy.node import Node
import std_msgs
from std_msgs.msg import Float32MultiArray
from scipy.interpolate import make_interp_spline
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import threading
import time
from collections import namedtuple

from geometry_msgs.msg import TransformStamped
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
import sensor_msgs_py.point_cloud2 as pc2

import matplotlib
from sensor_msgs.msg import PointCloud2, PointField
from scipy.spatial.transform import Rotation as Rot
matplotlib.use('Agg')  # Use non-GUI backend for matplotlib

class MainNode(Node):
    def __init__(self, camera_calibrations):
        super().__init__('detections_subscriber')
        
        # PARAMETERS
        self.min_points_for_spline = 4
        self.max_time_gap = .5
        self.cameras_to_ignore = [0] # Camera 3 is the one with the most noise, so we ignore it for now
        self.min_correspondences = 500
        self.min_accettable_f = 0.8
        self.possible_values = range(0, 7)
        self.mode = "CALIBRATION"  
        self.message_count = 0

        # ATTRIBUTES INITIALIZATION
        self.camera_calibrations = {cam['camera_id']: cam for cam in camera_calibrations}
        self.point_correspondences = {value: {v: [] for v in self.possible_values} for value in self.possible_values} # Pair of corresponding plints: ((spline_x, spline_y), (pt_x, pt_y), timestamp)
        self.data_lists = {value: [] for value in self.possible_values} # TODO: remove this, we only need rectified data
        self.rectified_data_lists = {value: [[]] for value in self.possible_values}
        self.splines = {value: [] for value in self.possible_values} # Each spline is a tuple (spline_x, spline_y, time_points, data_indices)
        self.localized_cameras = set()  # Cameras that have been localized
        self.last_3d_spline = None 
        self.camera_poses = [None] * len(self.possible_values)  # Array of (t, R) tuples for each camera
        self.trajectory = []  # List of 3D points for the trajectory
        self.trajectory_timestamps = []  # List of timestamps for the trajectory points
        
        # PUBLISHERS AND SUBSCRIBERS
        self.tf_broadcaster = StaticTransformBroadcaster(self)
        
        self.image_publishers = {}
        self.live_detections = {}
        self.cv_bridge = CvBridge()
        for cam_id in self.possible_values:
            topic_name = f'/camera_{cam_id}/rectified_image'
            self.image_publishers[cam_id] = self.create_publisher(Image, topic_name, 10)
            resolution = self.camera_calibrations[cam_id]['resolution']
            self.live_detections[cam_id] = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)  # Initialize with zeros
            blank_image = np.ones((resolution[1], resolution[0], 3), dtype=np.uint8) * 255
            image_msg = self.cv_bridge.cv2_to_imgmsg(blank_image, encoding='bgr8')
            self.image_publishers[cam_id].publish(image_msg)

        self.pointcloud_publisher = self.create_publisher(PointCloud2, '/live_trajectory', 10)
        # self.splines_publisher = self.create_publisher(PointCloud2, '/splines', 10)
        
        self.detection_sub = self.create_subscription(Float32MultiArray,'/detections',self.main_loop,10)

        # Pre-compute rectification maps for all cameras
        self.rectification_maps = {}
        self.rectified_camera_matrices = {}
        self._precompute_rectified_calib_matrices()

        # RESET RVIZ
        for value in self.possible_values:
            self.sendTfStaticTransform(value, np.zeros(3), np.eye(3)) 
        self.publish_pointcloud([])
        
        self.get_logger().info("Node initialized successfully")
        
    def publish_pointcloud(self, points, publisher=None):
        """
        Publish a list of 3D points as a PointCloud2 message.
        :param points: List of (x, y, z) tuples
        """
        header = self.get_clock().now().to_msg()
        pc_header = self.get_clock().now().to_msg()
        msg = pc2.create_cloud_xyz32(
            std_msgs.msg.Header(
                stamp=self.get_clock().now().to_msg(),
                frame_id='map'
            ),
            points
        )
        if publisher is None:
            self.pointcloud_publisher.publish(msg)
        else:
            publisher.publish(msg)
        
    def sendTfStaticTransform(self, camera_id, translation, rotation):
        """Send the inverse of the given transform for the camera to the TF2 broadcaster (ROS convention)"""
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = 'map'
        transform.child_frame_id = f'camera_{camera_id}'

        # Invert rotation and translation
        rot_matrix = rotation
        inv_rot = rot_matrix.T
        inv_trans = -inv_rot @ np.array(translation).reshape(3)

        transform.transform.translation.x = float(inv_trans[0])
        transform.transform.translation.y = float(inv_trans[1])
        transform.transform.translation.z = float(inv_trans[2])

        # Convert rotation matrix to quaternion (scipy returns [x, y, z, w])
        quat = Rot.from_matrix(inv_rot).as_quat()
        transform.transform.rotation.x = quat[0]
        transform.transform.rotation.y = quat[1]
        transform.transform.rotation.z = quat[2]
        transform.transform.rotation.w = quat[3]

        self.get_logger().info(f"Inverse Tf published: camera_{camera_id}")

        self.tf_broadcaster.sendTransform(transform)

    def to_normalized_camera_coord(self, pts, K, distcoeff):
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
        pts_normalized = cv.undistortPoints(pts, K, distcoeff)
        
        # Convert from homogeneous coordinates to 2D
        pts_normalized = pts_normalized.reshape(-1, 2)
        
        return pts_normalized

    def main_loop(self, msg):
        if len(msg.data) < 4:
            return
            
        element_1 = int(msg.data[1])  # camera_id
        
        if element_1 == 3 or element_1 == 0:
            return
        
        if element_1 not in self.data_lists:
            self.get_logger().warn(f'Value {element_1} not in possible values')
            return
        
        self.message_count += 1
        
        if 330 < msg.data[0] < 330.5:
            threading.Thread(target=self.plotall, daemon=True).start()

        # Create raw data point
        raw_point = (msg.data[0], msg.data[1], msg.data[2], msg.data[3])
        
        # Rectify the point immediately upon arrival
        rectified_point = raw_point # Points are already in rectified coordinates
        
        # Publish the rectified image for this camera with a white dot at the rectified point
        if element_1 in self.image_publishers:
            image = self.live_detections[element_1].copy()
            cv.circle(image, (int(rectified_point[2]), int(rectified_point[3])), 5, (255, 255, 255), -1)
            # Convert to ROS Image message
            image_msg = self.cv_bridge.cv2_to_imgmsg(image, encoding='bgr8')
            # Publish the image
            self.image_publishers[element_1].publish(image_msg)
            self.live_detections[element_1] = image  # Update the live image for this camera
        else:
            self.get_logger().warn(f'No publisher for camera {element_1} - skipping image publishing')
        
        
        # First detection from this camera
        if len(self.data_lists[element_1]) == 0:
            self.get_logger().info(f'New camera detected: {element_1}')
        
        self.data_lists[element_1].append(raw_point)            
        
        # Check if we need to start a new time group
        current_camera_data = self.rectified_data_lists[element_1]
        
        if self.mode == "TRACKING" and element_1 in self.localized_cameras:
            # Get the last 5 points (flattened) from all groups for this camera
            all_points = self.data_lists[element_1]
            if len(all_points) >= 5:
                last_points = all_points[-5:]
                times = [p[0] for p in last_points]
                if np.max(np.diff(times)) < self.max_time_gap:
                    rectified_x = [p[2] for p in last_points]
                    rectified_y = [p[3] for p in last_points]
                    spline_x = make_interp_spline(times, rectified_x, k=min(3, len(times)-1))
                    spline_y = make_interp_spline(times, rectified_y, k=min(3, len(times)-1))
                    tmp_interpolations = []
                    for other_camera in self.possible_values:
                        if other_camera == element_1 or not self.rectified_data_lists[other_camera]:
                            continue
                        # Get last rectified point for this other camera
                        other_points = self.data_lists[other_camera]
                        if not other_points:
                            continue
                        last_other_point = other_points[-1]
                        t_other = last_other_point[0]
                        if times[-3] <= t_other <= times[-1]:
                            interp_x = float(spline_x(t_other))
                            interp_y = float(spline_y(t_other))
                            tmp_interpolations.append({
                                "camera": other_camera,
                                "time": t_other,
                                "interp": (interp_x, interp_y),
                                "other_point": (last_other_point[2], last_other_point[3])
                            })
                    # tmp_interpolations now contains the interpolated points for each other camera
                    if len(tmp_interpolations) > 0:
                        new_triangulated_points = []
                        for interp in tmp_interpolations:
                            if interp['camera'] in self.localized_cameras:
                                K1 = self.rectified_camera_matrices[element_1]
                                K2 = self.rectified_camera_matrices[interp['camera']]

                                P1 = K1 @ np.hstack([self.camera_poses[element_1][1], self.camera_poses[element_1][0].reshape(-1, 1)])
                                P2 = K2 @ np.hstack([self.camera_poses[interp['camera']][1], self.camera_poses[interp['camera']][0].reshape(-1, 1)])
                                
                                pt = cv.triangulatePoints(P1, P2, (raw_point[2], raw_point[3]), interp['other_point'])
                                pt /= pt[3]
                                pt = pt[:3].flatten()
                                new_triangulated_points.append((pt[0], pt[1], pt[2]))
                        if new_triangulated_points:
                            avg_triangulated_point = np.mean(new_triangulated_points, axis=0)
                            self.trajectory.append(avg_triangulated_point)
                            self.trajectory_timestamps.append(raw_point[0])
                            self.publish_pointcloud(self.trajectory)
            if self.message_count % 300 == 0:
                self.get_logger().info(f'Here {self.message_count}')
                self.compute_3d_splines()
                            
        
        # If the current camera has no data yet, or if the last group is empty
        if not current_camera_data or not current_camera_data[-1]:
            current_camera_data[-1].append(rectified_point)
        else:
            # Compare timestamps (index 0) to decide if points are close in time
            last_timestamp = current_camera_data[-1][-1][0]  # timestamp of last point in last group
            current_timestamp = rectified_point[0]
            
            # If timestamps are close (within your threshold), add to current group
            if abs(current_timestamp - last_timestamp) < self.max_time_gap:
                current_camera_data[-1].append(rectified_point)
            else:
                # Start a new time group
                points = current_camera_data[-1]
                current_camera_data.append([rectified_point])
                
                # Create a new spline
                if len(points) >= self.min_points_for_spline:
                    times = [p[0] for p in points]
                    rectified_x = [p[2] for p in points]  # rectified x
                    rectified_y = [p[3] for p in points]  # rectified y
                    spline_x = make_interp_spline(times, rectified_x, k=3)
                    spline_y = make_interp_spline(times, rectified_y, k=3)
                    self.splines[element_1].append((spline_x, spline_y, times, list(range(len(points)))))
                    if self.mode == "CALIBRATION":
                        threading.Thread(
                            target=self.do_first_step,
                            args=(element_1, times, spline_x, spline_y),
                            daemon=True
                        ).start()   
            
    def _precompute_rectified_calib_matrices(self):
        """Pre-compute rectification maps for all cameras - most efficient for dense rectification"""
        for camera_id, calib in self.camera_calibrations.items():
            if camera_id in self.possible_values:
                K = np.array(calib['K-matrix'])
                D = np.array(calib['distCoeff'])
                resolution = calib['resolution']  # (width, height)
                
                # Compute optimal camera matrix for rectification
                new_K, roi = cv.getOptimalNewCameraMatrix(K, D, resolution, 1, resolution)
                self.rectified_camera_matrices[camera_id] = new_K
                
                self.get_logger().info(f'Pre-computed rectified K for camera {camera_id}') 
                    
    def do_first_step(self, element_1, times, spline_x, spline_y):
        for other_camera in self.possible_values:
            if other_camera == element_1 or not self.rectified_data_lists[other_camera]:
                continue
            for other_points in reversed(self.rectified_data_lists[other_camera]):
                for point in reversed(other_points):
                    if point[0] < times[0]:
                        break
                    if times[0] <= point[0] <= times[-1]:
                        spline_point = (float(spline_x(point[0])), float(spline_y(point[0])))
                        self.point_correspondences[element_1][other_camera].append((spline_point, (point[2], point[3]), point[0]))
                        
            # Try computing the Fundamental matrix 
            if self.mode == "CALIBRATION" and len(self.point_correspondences[element_1][other_camera]) > self.min_correspondences and other_camera not in self.cameras_to_ignore and element_1 not in self.cameras_to_ignore:
                
                spline_points = np.array([p[0] for p in self.point_correspondences[element_1][other_camera]], dtype=np.float32)
                other_points = np.array([p[1] for p in self.point_correspondences[element_1][other_camera]], dtype=np.float32)
                
                K1 = self.rectified_camera_matrices[element_1] 
                K2 = self.rectified_camera_matrices[other_camera]
                
                # Find Fundamental Matrix
                F, mask = cv.findFundamentalMat(spline_points, other_points, cv.RANSAC)
                
                if F is not None and np.sum(mask) / len(mask) > self.min_accettable_f:
                    self.mode = "TRACKING"                                    
                    self.get_logger().info(f'Found valid Fundamental matrix between cameras {element_1} and {other_camera} with {len(mask)} correspondences')
                    self.get_logger().info(f'Beginnning localization mode...')
                    self.localized_cameras.add(other_camera)
                    self.localized_cameras.add(element_1)
                    
                    # Convert to Essential Matrix
                    E = K2.T @ F @ K1
                    
                    # Normalize points for pose recovery
                    pts1_norm = self.to_normalized_camera_coord(spline_points, K1, np.zeros(5))
                    pts2_norm = self.to_normalized_camera_coord(other_points, K2, np.zeros(5))

                    # Recover pose using normalized points
                    _, R, t, pose_mask = cv.recoverPose(E, pts1_norm, pts2_norm)
                    
                    self.camera_poses[element_1] = (np.zeros(3), np.eye(3))  # Initialize camera pose for element_1
                    self.camera_poses[other_camera] = (t.flatten(), R)  # Store pose for other camera
                    
                    # Create projection matrices for triangulation
                    P1 = K1 @ np.hstack([np.eye(3), np.zeros((3, 1))])
                    P2 = K2 @ np.hstack([R, t.reshape(-1, 1)])
                    
                    # Triangulate using ORIGINAL image coordinates (not normalized)
                    points_4d = cv.triangulatePoints(P1, P2, spline_points.T, other_points.T)
                    points_3d = (points_4d[:3] / points_4d[3]).T
                    triangulated_points = points_3d
                    
                    triangulated_timestamps = [p[2] for p in self.point_correspondences[element_1][other_camera]]
                    
                    CameraInfo = namedtuple('CameraInfo', ['K_matrix', 'distCoeff', 'resolution'])
                    # # Plot some statistics
                    # dx_stamps = np.diff(triangulated_timestamps)
                    # plt.figure(figsize=(19, 10))
                    # plt.plot(dx_stamps, marker='o')
                    # plt.yscale('log')
                    # plt.grid(True, which='both', linestyle='--', linewidth=0.5)
                    # plt.title(f'Timestamp Differences (dx_stamps) between cameras {element_1} and {other_camera}')
                    # plt.xlabel('Index')
                    # plt.ylabel('Time Difference (log scale)')
                    # plt.tight_layout()
                    # plt.savefig(f'dx_stamps_{element_1}_{other_camera}.png')
                    # plt.close()
                    
                    # camera_info_1 = CameraInfo(
                    #     K_matrix=self.rectified_camera_matrices[element_1],
                    #     distCoeff=np.zeros(5),  # Assuming no distortion for rectified points
                    #     resolution=self.camera_calibrations[element_1]['resolution']
                    # )
                    # camera_info_2 = CameraInfo(
                    #     K_matrix=self.rectified_camera_matrices[other_camera],
                    #     distCoeff=np.zeros(5),  # Assuming no distortion for rectified points
                    #     resolution=self.camera_calibrations[other_camera]['resolution']
                    # )
                    
                    # self.plot_reprojection_analysis(
                    #     triangulated_points, spline_points, 
                    #     np.eye(3), np.zeros(3),
                    #     camera_info_1, element_1, "Reprojection Analysis for Camera {element_1}")
                    # self.plot_reprojection_analysis(
                    #     triangulated_points, other_points, 
                    #     R, t,
                    #     camera_info_2, other_camera, "Reprojection Analysis for Camera {other_camera}")
                    
                    self.sendTfStaticTransform(element_1, np.zeros(3), np.eye(3))
                    self.sendTfStaticTransform(other_camera, t, R)
                    
                    self.trajectory.extend(triangulated_points.tolist())
                    self.publish_pointcloud(triangulated_points)
                    self.get_logger().info("Published first cloud")
                    
                    # 3D SPLINES GEN
                    # Convert to the format (x, y, z, timestamp)
                    self.trajectory_timestamps = [p[2] for p in self.point_correspondences[element_1][other_camera]]
                    self.compute_3d_splines()
                    
    def compute_3d_splines(self):
        trajectory_np = np.array(self.trajectory)
        points_3d_with_timestamps = np.vstack((trajectory_np.T, self.trajectory_timestamps))
        points_3d_with_timestamps = points_3d_with_timestamps.T  # Shape (N, 4) where N is number of points
        splines_3d_points = []
        this_spline = []
        
        for t in points_3d_with_timestamps:
            self.get_logger().info(f"t= {t[3]}")

        if len(points_3d_with_timestamps) == 0:
            # Handle empty case appropriately
            self.get_logger().warn("No triangulated points to generate splines from.")
        else:
            for i, current_point in enumerate(points_3d_with_timestamps[:-1]):
                next_point = points_3d_with_timestamps[i + 1]
                this_spline.append(points_3d_with_timestamps[i])

                # Check if the time difference exceeds the threshold (using same variable names as first method)
                if abs(next_point[3] - current_point[3]) >= self.max_time_gap: 
                    if len(this_spline) >= self.min_points_for_spline:  # Check minimum points
                        splines_3d_points.append(this_spline)
                    this_spline = []

            # Append the last triangulated point to the current spline
            this_spline.append(points_3d_with_timestamps[-1])
            if len(this_spline) >= self.min_points_for_spline:  # Check minimum points for last spline
                splines_3d_points.append(this_spline)

        # Filter out any empty splines or splines with less than required points 
        splines_3d_points = [spline for spline in splines_3d_points if spline and len(spline) >= self.min_points_for_spline]

        splines_3d = []
        for spline_points in splines_3d_points:
            spline_points_i = np.array(spline_points)
            ts = spline_points_i[:, 3]
            if not np.all(np.diff(ts) > 0):
                spline_points_i = spline_points_i[::-1]
            ts = spline_points_i[:, 3]  # Use timestamps from triangulated points

            
            if len(ts) < self.min_points_for_spline:  # Use consistent variable name
                continue  # Skip if not enough points for spline fitting
            
            spline_x = make_interp_spline(ts, spline_points_i[:, 0], k=3)
            spline_y = make_interp_spline(ts, spline_points_i[:, 1], k=3)
            spline_z = make_interp_spline(ts, spline_points_i[:, 2], k=3)
            
            splines_3d.append((spline_x, spline_y, spline_z, ts))  
        
        self.get_logger().info(f'Found {len(splines_3d)} 3D splines')
        self.last_3d_spline = splines_3d[-1]                                   
        
        # # Sample 100 points for each 3D spline and publish as a point cloud
        # all_sampled_points = []
        # for spline_x, spline_y, spline_z, ts in splines_3d:
        #     t_min, t_max = min(ts), max(ts)
        #     if t_max - t_min < 1e-6:
        #         continue  # Avoid degenerate splines
        #     t_samples = np.linspace(t_min, t_max, 100)
        #     x_samples = spline_x(t_samples)
        #     y_samples = spline_y(t_samples)
        #     z_samples = spline_z(t_samples)
        #     sampled_points = np.stack([x_samples, y_samples, z_samples], axis=1)
        #     all_sampled_points.extend(sampled_points.tolist())
        # self.publish_pointcloud(all_sampled_points, self.splines_publisher)
        
        self.localize_remaining_cameras(splines_3d, CameraInfo=namedtuple('CameraInfo', ['K_matrix', 'distCoeff', 'resolution']))
                    
    def localize_remaining_cameras(self, splines_3d, CameraInfo):
        """
        Attempt to localize all cameras that are not yet localized using 3D-2D correspondences and PnP.
        :param splines_3d: List of tuples (spline_x, spline_y, spline_z, ts) representing 3D splines.
        :param CameraInfo: Namedtuple with camera calibration info.
        """
        for camera_id in self.possible_values:
            if camera_id in self.localized_cameras:
                continue
            corresp_3d2d_this_camera = []
            for group in self.rectified_data_lists[camera_id]:
                if not group:
                    continue
                for point in group:
                    # Find spline that contains this point in time
                    for spline_x, spline_y, spline_z, ts in splines_3d:
                        if ts[0] <= point[0] <= ts[-1]:
                            # Point is within the time range of this spline
                            x_3d = float(spline_x(point[0]))
                            y_3d = float(spline_y(point[0]))
                            z_3d = float(spline_z(point[0]))
                            corresp_3d2d_this_camera.append(((x_3d, y_3d, z_3d, point[0]), (point[2], point[3]), point[0]))
                            break  # No need to check other splines

            if len(corresp_3d2d_this_camera) > self.min_correspondences * .5:
                rvec, tvec, inliers = None, None, None
                object_points = np.array([p[0][:3] for p in corresp_3d2d_this_camera], dtype=np.float32)
                image_points = np.array([p[1] for p in corresp_3d2d_this_camera], dtype=np.float32)
                # Ensure shapes are correct for solvePnPRansac
                _, rvec, tvec, inliers = cv.solvePnPRansac(
                    object_points,
                    image_points,
                    self.rectified_camera_matrices[camera_id],
                    np.zeros(5), 
                    confidence=0.99, reprojectionError=8.0
                )

                if inliers is not None and np.sum(inliers) / len(inliers) > self.min_accettable_f:
                    self.get_logger().info(f'Found valid pose for camera {camera_id} with {len(inliers)} correspondences over {len(corresp_3d2d_this_camera)} points')

                    # Convert rvec to rotation matrix
                    R, _ = cv.Rodrigues(rvec)

                    self.camera_poses[camera_id] = (tvec.flatten(), R)  # Store pose for this camera

                    # Send static transform for this camera
                    self.sendTfStaticTransform(camera_id, tvec.flatten(), R)

                    camera_info = CameraInfo(
                        K_matrix=self.rectified_camera_matrices[camera_id],
                        distCoeff=np.zeros(5),  # Assuming no distortion for rectified points
                        resolution=self.camera_calibrations[camera_id]['resolution']
                    )                                                                                 

                    self.plot_reprojection_analysis(
                        object_points,
                        image_points,
                        R, tvec.flatten(),
                        camera_info, camera_id                                                            
                    )

                    self.localized_cameras.add(camera_id)
                else:
                    self.get_logger().warn(f'Discarded pose for camera {camera_id} due to insufficient inliers: {np.sum(inliers) if inliers is not None else 0} / {len(corresp_3d2d_this_camera)}')
            else:
                self.get_logger().info(f'Insufficient points for solvePnPRansac for camera {camera_id}: {len(corresp_3d2d_this_camera)} points')
    
    def plotall(self):
        
        for camera_id, splines in self.splines.items():
            plt.figure(figsize=(19, 10))
            plt.title(f'Splines for Camera {camera_id}')
            for spline_x, spline_y, time_points, data_indices in splines:
                plt.plot(spline_x(time_points), -spline_y(time_points))
            plt.xlabel('Rectified X')
            plt.ylabel('Rectified Y')
            plt.xlim(0, self.camera_calibrations[camera_id]['resolution'][0])
            plt.ylim(-self.camera_calibrations[camera_id]['resolution'][1], 0)
            plt.savefig(f'spline_camera_{camera_id}.png')            
            
        for camera_id, data_points in self.rectified_data_lists.items():
            plt.figure(figsize=(19, 10))
            plt.title(f'Rectified Points for Camera {camera_id}')
            
            for group in data_points:
                if not group:
                    continue
                x_coords = [p[2] for p in group]
                y_coords = [-p[3] for p in group]
                plt.scatter(x_coords, y_coords, s=1, label=f'Group {data_points.index(group)}')
            
            plt.xlabel('Rectified X')
            plt.ylabel('Rectified Y')
            plt.xlim(0, self.camera_calibrations[camera_id]['resolution'][0])
            plt.ylim(-self.camera_calibrations[camera_id]['resolution'][1], 0)
            plt.savefig(f'scatter_camera_{camera_id}.png')
    
    def plot_reprojection_analysis(
        self, points_3d, original_points_2d, 
        R, t, camera_info, camera_id, title=None):
        """Plot reprojected 2D points for a single camera."""
        # Reproject points        
        reprojected_points, _ = cv.projectPoints(
            points_3d, R, t,
            camera_info.K_matrix, camera_info.distCoeff
        )
        reprojected_points = reprojected_points.reshape(-1, 2)

        plt.figure(figsize=(19, 10))
        plt.title(f"Reprojection Analysis for Camera {camera_id}")
        if title:
            plt.title(title)
        # Plot reprojected points
        plt.scatter(reprojected_points[:, 0], -reprojected_points[:, 1], c='r', label='Reprojected Points', s=1)
        plt.scatter(
            original_points_2d[:, 0], -original_points_2d[:, 1],
            c='b', label='Original Points', alpha=0.5, s=1
        )
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.xlim(0, camera_info.resolution[0])
        plt.ylim(-camera_info.resolution[1], 0)

        plt.tight_layout()
        plt.savefig(f"reprojection_analysis_camera_{camera_id}.png")
        self.get_logger().info(f"Reprojection analysis saved for camera {camera_id}")



def main(args=None):
    # Example camera calibration data structure
    camera_calibrations = [
        {
            "camera_id": 0,
            "comment":["Templete for providing camera information.",
                    "The path of this file should be included in the 'config.json' file under 'path_cameras'",
                    "K-matrix should be a 3*3 matrix",
                    "distCoeff should be a vector of [k1,k2,p1,p2[,k3]]"],

            "K-matrix":[[874.4721846047786, 0.0, 970.2688358898922], [0.0, 894.1080937815644, 531.2757796052425], [0.0, 0.0, 1.0]],

            "distCoeff":[-0.260720634999793, 0.07494782427852716, -0.00013631462898833923, 0.00017484761775924765, -0.00906247784302948],
                
            "fps":59.940060,

            "resolution":[1920,1080]

        },
        {
            "camera_id": 1,
            "comment":["Templete for providing camera information.",
                    "The path of this file should be included in the 'config.json' file under 'path_cameras'",
                    "K-matrix should be a 3*3 matrix",
                    "distCoeff should be a vector of [k1,k2,p1,p2[,k3]]"],

            "K-matrix":[[1918.685, 0.0, 965.581],
                        [0.0, 1914.767898726004, 550.550],
                        [0.0, 0.0, 1.0]],

            "distCoeff":[0.029412094519, -0.059510767134,  0.001856271031,  0.002269135198,
        0.276425764319],
                
            "fps":29.838692,

            "resolution":[1920,1080]

        },
        {
            "camera_id": 2,
            "comment":["Templete for providing camera information.",
                    "The path of this file should be included in the 'config.json' file under 'path_cameras'",
                    "K-matrix should be a 3*3 matrix",
                    "distCoeff should be a vector of [k1,k2,p1,p2[,k3]]"],

            "K-matrix":[[1569.2133510353567, 0.0, 956.6414016207103], [0.0, 1572.0082312735778, 533.1129574904855], [0.0, 0.0, 1.0]],

            "distCoeff":[0.1888803002995303, -0.6268259695581048, -0.0014010104725158184, 0.0003932341772251142, 0.6061741343069857],
                
            "fps":30.0,

            "resolution":[1920,1080]

        },
        {
            "camera_id": 3,
            "comment":["Templete for providing camera information.",
                    "The path of this file should be included in the 'config.json' file under 'path_cameras'",
                    "K-matrix should be a 3*3 matrix",
                    "distCoeff should be a vector of [k1,k2,p1,p2[,k3]]"],

            "K-matrix":[[3264.997613980756, 0.0, 1871.0028562370057], [0.0, 3245.0203739800636, 1002.0844321863265], [0.0, 0.0, 1.0]],

            "distCoeff":[0.18419266153533212, -0.7557852934198697, -0.0013645512787729467, -0.002783809883555535, 0.9397964121421092],

            "fps":30,

            "resolution":[3840,2160]

        },
        {
            "camera_id": 4,
            "comment":["Templete for providing camera information.",
                    "The path of this file should be included in the 'config.json' file under 'path_cameras'",
                    "K-matrix should be a 3*3 matrix",
                    "distCoeff should be a vector of [k1,k2,p1,p2[,k3]]"],

            "K-matrix":[[1545.425401191011, 0.0, 971.158336917263], 
                [0.0, 1545.96703364831, 535.682080735544], 
                [0.0, 0.0, 1.0]],

            "distCoeff":[-0.011232359677,  0.045931232417,  0.000263868094, -0.001253638454, -0.151703077572],

            "fps":29.970030,

            "resolution":[1920,1080]

        },
        {
            "camera_id": 5,
            "comment":["Templete for providing camera information.",
                    "The path of this file should be included in the 'config.json' file under 'path_cameras'",
                    "K-matrix should be a 3*3 matrix",
                    "distCoeff should be a vector of [k1,k2,p1,p2[,k3]]"],

            "K-matrix":[[1506.6612, 0,  966.6623],
                        [0, 1505.2822,  530.2608],
                        [0, 0,  1]],

            "distCoeff":[-0.006673507597820779, 0.007775663251591633, 0, 0],
                
            "fps":50,

            "resolution":[1920,1080]

        },
        {
            "camera_id": 6,
            "comment":["Templete for providing camera information.",
                    "The path of this file should be included in the 'config.json' file under 'path_cameras'",
                    "K-matrix should be a 3*3 matrix",
                    "distCoeff should be a vector of [k1,k2,p1,p2[,k3]]"],

            "K-matrix":[[1176.899407018602, 0.0, 702.661410507918], [0.0, 1572.936467868805, 532.023006642381], [0.0, 0.0, 1.0]],

            "distCoeff":[-0.104075713538,  0.141342139932, -0.000084551089,
                -0.000391293455, -0.068036995013],
                
            "fps":25,

            "resolution":[1440,1080]

        }


    ]
    
    rclpy.init(args=args)
    node = MainNode(camera_calibrations)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()