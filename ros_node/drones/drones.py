import rclpy
from rclpy.node import Node
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

import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for matplotlib

class MainNode(Node):
    def __init__(self, camera_calibrations):
        super().__init__('detections_subscriber')
        
        self.compiled_cameras = 0
        self.possible_values = range(0, 7)
        self.mode = "CALIBRATION"  

        # Store camera calibration data
        # Expected format: [{'camera_id': 0, 'K-matrix': K, 'distCoeff': D, 'resolution': (w,h), ...}, ...]
        self.camera_calibrations = {cam['camera_id']: cam for cam in camera_calibrations}

        # Point correspondences between pairs of cameras (square matrix)
        self.point_correspondences = {value: {v: [] for v in self.possible_values} for value in self.possible_values} # First index is camera with spline, second index is camera with point correspondences. Array contains pairs of corresponding points
        
        # Create one image publisher for each possible_value

        self.image_publishers = {}
        self.cv_bridge = CvBridge()
        for cam_id in self.possible_values:
            topic_name = f'/camera_{cam_id}/rectified_image'
            self.image_publishers[cam_id] = self.create_publisher(Image, topic_name, 10)
            # Publish a blank image as initialization
            resolution = self.camera_calibrations[cam_id]['resolution']
            blank_image = np.ones((resolution[1], resolution[0], 3), dtype=np.uint8) * 255
            image_msg = self.cv_bridge.cv2_to_imgmsg(blank_image, encoding='bgr8')
            self.image_publishers[cam_id].publish(image_msg)


        # Pre-compute rectification maps for all cameras
        self.rectification_maps = {}
        self.rectified_camera_matrices = {}
        self._precompute_rectification_maps()

        # Alternative: Pre-compute inverse distortion polynomials (faster for sparse points)
        self.use_polynomial_undistortion = True
        self.undistortion_polys = {}
        if self.use_polynomial_undistortion:
            self._precompute_undistortion_polynomials()

        # Store raw and rectified data points for each camera
        self.data_lists = {value: [] for value in self.possible_values}
        self.rectified_data_lists = {value: [[]] for value in self.possible_values}

        # Store ALL splines for each camera
        self.splines = {value: [] for value in self.possible_values} # Each spline is a tuple (spline_x, spline_y, time_points, data_indices)
        
        # Configuration
        self.min_points_for_spline = 4
        self.max_time_gap = .5
        self.max_spline_time_span = 0.2
        self.camera_to_ignore = 3 # Camera 3 is the one with the most noise, so we ignore it for now
        self.min_correspondences = 40

        self.detection_sub = self.create_subscription(
            Float32MultiArray,
            '/detections',
            self.listener_callback,
            10
        )

    def listener_callback(self, msg):
        if len(msg.data) < 4:
            return
            
        element_1 = int(msg.data[1])  # camera_id
        
        # if element_1 != 5 and element_1 != 4:
        #     return
        
        if element_1 not in self.data_lists:
            self.get_logger().warn(f'Value {element_1} not in possible values')
            return
        
        if 330 < msg.data[0] < 331:
            threading.Thread(target=self.plotall, daemon=True).start()

        # Create raw data point
        raw_point = (msg.data[0], msg.data[1], msg.data[2], msg.data[3])
        
        # Rectify the point immediately upon arrival
        rectified_coords = self.rectify_point_polynomial(element_1, (msg.data[2], msg.data[3]))
        rectified_point = (raw_point[0], raw_point[1], rectified_coords[0], rectified_coords[1])
        
        # Publish the rectified image for this camera with a white dot at the rectified point
        if element_1 in self.image_publishers:
            # Create a black image with the camera resolution
            resolution = self.camera_calibrations[element_1]['resolution']
            image = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)
            # Draw a white dot at the rectified point
            cv.circle(image, (int(rectified_coords[0]), int(rectified_coords[1])), 5, (255, 255, 255), -1)
            # Convert to ROS Image message
            image_msg = self.cv_bridge.cv2_to_imgmsg(image, encoding='bgr8')
            # Publish the image
            self.image_publishers[element_1].publish(image_msg)
        else:
            self.get_logger().warn(f'No publisher for camera {element_1} - skipping image publishing')
        
        
        # First detection from this camera
        if len(self.data_lists[element_1]) == 0:
            self.compiled_cameras += 1
            self.get_logger().info(f'New camera detected: {element_1}')
        
        # Store both raw and rectified data
        # Store both raw and rectified data
        self.data_lists[element_1].append(raw_point)
        
        # Check if we need to start a new time group
        current_camera_data = self.rectified_data_lists[element_1]
        
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
                    
                    # Launch correspondence finding in a separate thread (non-blocking)
                    def find_correspondences(element_1, times, spline_x, spline_y):
                        for other_camera in self.possible_values:
                            if other_camera == element_1 or not self.rectified_data_lists[other_camera]:
                                continue
                            for other_points in reversed(self.rectified_data_lists[other_camera]):
                                for point in reversed(other_points):
                                    if point[0] < times[0]:
                                        break
                                    if times[0] <= point[0] <= times[-1]:
                                        spline_point = (float(spline_x(point[0])), float(spline_y(point[0])))
                                        self.point_correspondences[element_1][other_camera].append((spline_point, (point[2], point[3])))
                                        
                            # Try computing the Fundamental matrix 
                            if len(self.point_correspondences[element_1][other_camera]) > self.min_correspondences and other_camera != self.camera_to_ignore and element_1 != self.camera_to_ignore:
                                
                                spline_points = np.array([p[0] for p in self.point_correspondences[element_1][other_camera]], dtype=np.float32)
                                other_points = np.array([p[1] for p in self.point_correspondences[element_1][other_camera]], dtype=np.float32)
                                
                                F, mask = cv.findFundamentalMat(spline_points, other_points, cv.RANSAC, 4.0, 0.99)
                                
                                self.get_logger().info(f'Inliers found: {np.sum(mask)} over {len(mask)} points between cameras {element_1} and {other_camera}')
                    threading.Thread(
                        target=find_correspondences,
                        args=(element_1, times, spline_x, spline_y),
                        daemon=True
                    ).start()         
                            

    def _precompute_undistortion_polynomials(self):
        """Pre-compute polynomial approximations for undistortion - faster for sparse points"""
        for camera_id, calib in self.camera_calibrations.items():
            if camera_id in self.possible_values:
                K = np.array(calib['K-matrix'])
                D = np.array(calib['distCoeff'])
                resolution = calib['resolution']
                
                # Create a grid of distorted points
                grid_density = 20  # Adjust based on accuracy needs
                x_grid = np.linspace(0, resolution[0]-1, grid_density)
                y_grid = np.linspace(0, resolution[1]-1, grid_density)
                xx, yy = np.meshgrid(x_grid, y_grid)
                
                # Distorted points (input)
                distorted_points = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float32)
                
                # Undistort using OpenCV
                undistorted_points = cv.undistortPoints(
                    distorted_points.reshape(-1, 1, 2), K, D, None, K
                ).reshape(-1, 2)
                
                # Fit polynomial mapping from distorted to undistorted coordinates
                
                # Use RBF for smooth interpolation (alternative: griddata)
                rbf_x = RBFInterpolator(distorted_points, undistorted_points[:, 0], 
                                      kernel='thin_plate_spline', smoothing=0.1)
                rbf_y = RBFInterpolator(distorted_points, undistorted_points[:, 1], 
                                      kernel='thin_plate_spline', smoothing=0.1)
                
                self.undistortion_polys[camera_id] = (rbf_x, rbf_y)
                self.get_logger().info(f'Pre-computed undistortion polynomials for camera {camera_id}')

    def _precompute_rectification_maps(self):
        """Pre-compute rectification maps for all cameras - most efficient for dense rectification"""
        for camera_id, calib in self.camera_calibrations.items():
            if camera_id in self.possible_values:
                K = np.array(calib['K-matrix'])
                D = np.array(calib['distCoeff'])
                resolution = calib['resolution']  # (width, height)
                
                # Compute optimal camera matrix for rectification
                new_K, roi = cv.getOptimalNewCameraMatrix(K, D, resolution, 1, resolution)
                
                # Pre-compute rectification maps
                map1, map2 = cv.initUndistortRectifyMap(K, D, None, new_K, resolution, cv.CV_32FC1)
                
                self.rectification_maps[camera_id] = (map1, map2)
                self.rectified_camera_matrices[camera_id] = new_K
                
                self.get_logger().info(f'Pre-computed rectification maps for camera {camera_id}')
                
    def rectify_point_fast(self, camera_id, point):
        """Fastest rectification method using pre-computed maps"""
        if camera_id not in self.rectification_maps:
            return point  # Return original if no calibration available
            
        map1, map2 = self.rectification_maps[camera_id]
        x, y = point
        
        # Bounds checking
        if x < 0 or y < 0 or x >= map1.shape[1] or y >= map1.shape[0]:
            return point
            
        # Bilinear interpolation in the pre-computed maps
        x_int, y_int = int(x), int(y)
        x_frac, y_frac = x - x_int, y - y_int
        
        # Handle edge cases
        x_int = min(x_int, map1.shape[1] - 2)
        y_int = min(y_int, map1.shape[0] - 2)
        
        # Bilinear interpolation
        def bilinear_interp(map_array, x_int, y_int, x_frac, y_frac):
            top_left = map_array[y_int, x_int]
            top_right = map_array[y_int, x_int + 1]
            bottom_left = map_array[y_int + 1, x_int]
            bottom_right = map_array[y_int + 1, x_int + 1]
            
            top = top_left * (1 - x_frac) + top_right * x_frac
            bottom = bottom_left * (1 - x_frac) + bottom_right * x_frac
            return top * (1 - y_frac) + bottom * y_frac
        
        rectified_x = bilinear_interp(map1, x_int, y_int, x_frac, y_frac)
        rectified_y = bilinear_interp(map2, x_int, y_int, x_frac, y_frac)
        
        return (rectified_x, rectified_y)
    
    def rectify_point_polynomial(self, camera_id, point):
        """Alternative rectification using pre-computed polynomials - good for sparse points"""
        if camera_id not in self.undistortion_polys:
            return point
            
        rbf_x, rbf_y = self.undistortion_polys[camera_id]
        x, y = point
        
        try:
            # Evaluate the RBF interpolators
            rectified_x = float(rbf_x(np.array([[x, y]])))
            rectified_y = float(rbf_y(np.array([[x, y]])))
            return (rectified_x, rectified_y)
        except:
            return point  # Fallback to original point if interpolation fails

    def rectify_point_opencv(self, camera_id, point):
        """Standard OpenCV rectification - slower but most accurate"""
        if camera_id not in self.camera_calibrations:
            return point
            
        calib = self.camera_calibrations[camera_id]
        K = np.array(calib['K-matrix'])
        D = np.array(calib['distCoeff'])
        
        # Convert to the format expected by OpenCV
        point_array = np.array([[[point[0], point[1]]]], dtype=np.float32)
        
        # Undistort the point
        rectified_point = cv.undistortPoints(point_array, K, D, None, K)
        
        return (float(rectified_point[0, 0, 0]), float(rectified_point[0, 0, 1]))

    
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