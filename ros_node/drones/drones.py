import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from scipy.interpolate import make_interp_spline
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

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
        self.rectified_data_lists = {value: [] for value in self.possible_values}
        
        # Store ALL splines for each camera
        self.splines = {value: [] for value in self.possible_values} # Each spline is a tuple (spline_x, spline_y, time_points, data_indices)
        
        # Configuration
        self.min_points_for_spline = 4
        self.max_time_gap = 0.1
        self.max_spline_time_span = 0.2

        self.detection_sub = self.create_subscription(
            Float32MultiArray,
            '/detections',
            self.listener_callback,
            10
        )

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
                from scipy.interpolate import RBFInterpolator
                
                # Use RBF for smooth interpolation (alternative: griddata)
                rbf_x = RBFInterpolator(distorted_points, undistorted_points[:, 0], 
                                      kernel='thin_plate_spline', smoothing=0.1)
                rbf_y = RBFInterpolator(distorted_points, undistorted_points[:, 1], 
                                      kernel='thin_plate_spline', smoothing=0.1)
                
                self.undistortion_polys[camera_id] = (rbf_x, rbf_y)
                self.get_logger().info(f'Pre-computed undistortion polynomials for camera {camera_id}')

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

    def listener_callback(self, msg):
        if len(msg.data) < 4:
            return
            
        element_1 = int(msg.data[1])  # camera_id
        if element_1 not in self.data_lists:
            self.get_logger().warn(f'Value {element_1} not in possible values')
            return
        
        if 20000 < msg.data[0] < 20010:
            self.plotall()

        # Create raw data point
        raw_point = (msg.data[0]/59.940060, msg.data[1], msg.data[2], msg.data[3])
        
        # Rectify the point immediately upon arrival
        rectified_coords = self.rectify_point_polynomial(element_1, (msg.data[2], msg.data[3]))
        rectified_point = (raw_point[0], raw_point[1], rectified_coords[0], rectified_coords[1])
        
        # First detection from this camera
        if len(self.data_lists[element_1]) == 0:
            self.compiled_cameras += 1
            self.get_logger().info(f'New camera detected: {element_1}')
        
        # Store both raw and rectified data
        self.data_lists[element_1].append(raw_point)
        self.rectified_data_lists[element_1].append(rectified_point)
        
        # Process splines using rectified coordinates
        threading.Thread(target=self._process_new_point, args=(element_1, rectified_point), daemon=True).start()
        
        if self.mode == "CALIBRATION":
        
            for main_camera in self.point_correspondences:
                for other_camera in self.point_correspondences[main_camera]:
                    if other_camera == element_1:
                        continue
                    if len(self.point_correspondences[main_camera][other_camera]) > 1000:
                        # self.get_logger().warn(f'Point correspondences for {main_camera} and {other_camera} exceeded 1000 points, consider clearing or optimizing')
                        if main_camera != 3 and other_camera != 3:
                            pts_main = np.array([p[0][2:4] for p in self.point_correspondences[main_camera][other_camera]], dtype=np.float32)
                            pts_secondary = np.array([p[1] for p in self.point_correspondences[main_camera][other_camera]], dtype=np.float32)
                            
                            F, mask = cv.findFundamentalMat(pts_main, pts_secondary, cv.FM_RANSAC, 0.1, 0.99)
                            E = self.rectified_camera_matrices[main_camera].T @ F @ self.rectified_camera_matrices[other_camera]
                            
                            
                            self.get_logger().info(f'Fundamental matrix found between cameras {main_camera} and {other_camera}: {F}')
                            
                            pts_main_normalized = cv.undistortPoints(pts_main, self.rectified_camera_matrices[main_camera], None)
                            pts_secondary_normalized = cv.undistortPoints(pts_secondary, self.rectified_camera_matrices[other_camera], None)
                            
                            self.get_logger().info(f'Inliers found: {np.sum(mask)} over {len(mask)} points')
                            _, R, t, mask, triangulated_points = cv.recoverPose(E, pts_main_normalized, pts_secondary_normalized, 
                                                                                cameraMatrix=np.eye(3),
                                                                                distanceThresh=100.0)
                            
                            self.get_logger().info(f'Inliers found: {np.sum(mask)/255} over {len(mask)} points')
                            if np.sum(mask)/len(mask) < 0.5:
                                self.get_logger().warn(f'Insufficient inliers found between cameras {main_camera} and {other_camera}: {np.sum(mask)/len(mask)}')
                            #     continue
                            # self.mode = "TRACKING"

    def _process_new_point(self, camera_id, new_point):
        """Process new rectified point for spline creation/update"""
        rectified_points = self.rectified_data_lists[camera_id]
        current_time = new_point[0]
        
        if len(rectified_points) < self.min_points_for_spline:
            return
        
        # Check if we can update the most recent spline
        if self.splines[camera_id]:
            last_spline_info = self.splines[camera_id][-1]
            last_spline_time = last_spline_info[2][-1]
            
            if current_time - last_spline_time < self.max_time_gap:
                self._update_current_spline(camera_id, rectified_points)
                return
        
        self._try_create_new_spline(camera_id, rectified_points)

    def _update_current_spline(self, camera_id, data_points):
        """Update the most recent spline with new rectified point"""
        if not self.splines[camera_id]:
            return
            
        spline_x, spline_y, time_points, data_indices = self.splines[camera_id][-1]
        new_data_index = len(data_points) - 1
        updated_indices = data_indices + [new_data_index]
        spline_points = [data_points[i] for i in updated_indices]
        
        try:
            t = [p[0] for p in spline_points]
            x = [p[2] for p in spline_points]  # rectified x
            y = [p[3] for p in spline_points]  # rectified y
            
            if len(set(t)) >= self.min_points_for_spline:
                k = min(3, len(t) - 1)
                new_spline_x = make_interp_spline(t, x, k=k)
                new_spline_y = make_interp_spline(t, y, k=k)
                self.splines[camera_id][-1] = (new_spline_x, new_spline_y, t, updated_indices)
                # NEW: Find correspondences after updating spline
                self._find_correspondences_for_spline(camera_id, len(self.splines[camera_id]) - 1)
                
        except Exception as e:
            self.get_logger().warn(f'Failed to update spline for camera {camera_id}: {e}')
            self._try_create_new_spline(camera_id, data_points)

    def _try_create_new_spline(self, camera_id, data_points):
        """Create new spline using rectified points"""
        recent_count = min(self.min_points_for_spline, len(data_points))
        recent_points = data_points[-recent_count:]
        recent_indices = list(range(len(data_points) - recent_count, len(data_points)))
        
        time_span = recent_points[-1][0] - recent_points[0][0]
        if time_span > self.max_spline_time_span:
            for i in range(len(recent_points) - self.min_points_for_spline + 1):
                subset_points = recent_points[i:]
                subset_indices = recent_indices[i:]
                subset_time_span = subset_points[-1][0] - subset_points[0][0]
                
                if subset_time_span <= self.max_spline_time_span and len(subset_points) >= self.min_points_for_spline:
                    recent_points = subset_points
                    recent_indices = subset_indices
                    break
            else:
                return
        
        try:
            t = [p[0] for p in recent_points]
            x = [p[2] for p in recent_points]  # rectified x
            y = [p[3] for p in recent_points]  # rectified y
            
            if len(set(t)) >= self.min_points_for_spline:
                k = min(3, len(t) - 1)
                spline_x = make_interp_spline(t, x, k=k)
                spline_y = make_interp_spline(t, y, k=k)
                self.splines[camera_id].append((spline_x, spline_y, t, recent_indices))
                self._find_correspondences_for_spline(camera_id, len(self.splines[camera_id]) - 1)
                
        except Exception as e:
            self.get_logger().warn(f'Failed to create new spline for camera {camera_id}: {e}')

    def _find_correspondences_for_spline(self, spline_camera_id, spline_index):
        """Find point correspondences for a specific spline in other cameras' data"""
        if spline_index >= len(self.splines[spline_camera_id]):
            return
            
        spline_x, spline_y, time_points, data_indices = self.splines[spline_camera_id][spline_index]
        spline_start_time = min(time_points)
        spline_end_time = max(time_points)
        
        # Search in all other cameras
        for other_camera_id in self.possible_values:
            if other_camera_id == spline_camera_id or other_camera_id not in self.rectified_data_lists:
                continue
                
            other_camera_data = self.rectified_data_lists[other_camera_id]
            
            # Find points within the spline's time range
            for point in other_camera_data:
                point_time = point[0]
                if spline_start_time <= point_time <= spline_end_time:
                    # Evaluate the spline at the point's time
                    spline_point = (spline_x(point_time), spline_y(point_time))
                    
                    # Check if this correspondence already exists to avoid duplicates
                    if not self._correspondence_exists(other_camera_id, spline_camera_id, point, spline_point):
                        self.point_correspondences[other_camera_id][spline_camera_id].append((point, spline_point))
                        self.get_logger().info(f'Point correspondence length: {len(self.point_correspondences[other_camera_id][spline_camera_id])} between cameras {other_camera_id} and {spline_camera_id}')
                                               
    def _correspondence_exists(self, camera_id, spline_camera_id, point, spline_point):
        """Check if a correspondence already exists to avoid duplicates"""
        existing_correspondences = self.point_correspondences[camera_id][spline_camera_id]
        
        # Check for duplicate based on time and position (with small tolerance for floating point comparison)
        tolerance = 1e-6
        for existing_pair in existing_correspondences:
            existing_point, existing_spline_point = existing_pair
            if (abs(existing_point[0] - point[0]) < tolerance and  # time
                abs(existing_point[2] - point[2]) < tolerance and  # rectified x
                abs(existing_point[3] - point[3]) < tolerance and  # rectified y
                abs(existing_spline_point[0] - spline_point[0]) < tolerance and  # spline x
                abs(existing_spline_point[1] - spline_point[1]) < tolerance):    # spline y
                return True
        return False
    
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
            x_coords = [p[2] for p in data_points]
            y_coords = [-p[3] for p in data_points]
            plt.scatter(x_coords, y_coords, s=1, label='Rectified Points')
            plt.xlabel('Rectified X')
            plt.ylabel('Rectified Y')
            plt.xlim(0, self.camera_calibrations[camera_id]['resolution'][0])
            plt.ylim(-self.camera_calibrations[camera_id]['resolution'][1], 0)
            plt.savefig(f'scatter_camera_{camera_id}.png')

    def _try_create_new_spline(self, camera_id, data_points):
        """Create new spline using rectified points"""
        recent_count = min(self.min_points_for_spline, len(data_points))
        recent_points = data_points[-recent_count:]
        recent_indices = list(range(len(data_points) - recent_count, len(data_points)))
        
        time_span = recent_points[-1][0] - recent_points[0][0]
        if time_span > self.max_spline_time_span:
            for i in range(len(recent_points) - self.min_points_for_spline + 1):
                subset_points = recent_points[i:]
                subset_indices = recent_indices[i:]
                subset_time_span = subset_points[-1][0] - subset_points[0][0]
                
                if subset_time_span <= self.max_spline_time_span and len(subset_points) >= self.min_points_for_spline:
                    recent_points = subset_points
                    recent_indices = subset_indices
                    break
            else:
                return
        
        try:
            t = [p[0] for p in recent_points]
            x = [p[2] for p in recent_points]  # rectified x
            y = [p[3] for p in recent_points]  # rectified y
            
            if len(set(t)) >= self.min_points_for_spline:
                k = min(3, len(t) - 1)
                spline_x = make_interp_spline(t, x, k=k)
                spline_y = make_interp_spline(t, y, k=k)
                self.splines[camera_id].append((spline_x, spline_y, t, recent_indices))
                
        except Exception as e:
            self.get_logger().warn(f'Failed to create new spline for camera {camera_id}: {e}')

    def benchmark_rectification_methods(self, camera_id, test_points, iterations=1000):
        """Benchmark different rectification methods"""

        if camera_id not in self.camera_calibrations:
            return
            
        methods = {
            'fast_maps': self.rectify_point_fast,
            'polynomial': self.rectify_point_polynomial,
            'opencv': self.rectify_point_opencv
        }
        
        results = {}
        
        for method_name, method_func in methods.items():
            start_time = time.time()
            for _ in range(iterations):
                for point in test_points:
                    method_func(camera_id, point)
            end_time = time.time()
            
            total_time = end_time - start_time
            results[method_name] = {
                'total_time': total_time,
                'avg_time_per_point': total_time / (iterations * len(test_points)),
                'points_per_second': (iterations * len(test_points)) / total_time
            }
            
        return results


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