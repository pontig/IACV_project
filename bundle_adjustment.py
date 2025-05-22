import numpy as np
import cv2 as cv
from scipy.optimize import least_squares
from scipy.interpolate import BSpline
from scipy.spatial.distance import cdist

def bundle_adjust_camera_pose(splines_3d, camera_detections, camera_K, camera_dist_coeffs, 
                            initial_rvec=None, initial_tvec=None, optimize_splines=False,
                            kinetic_energy_weight=1.0, smoothness_weight=0.1):
    """
    Perform bundle adjustment to refine camera pose and optionally spline coefficients.
    
    Parameters:
    -----------
    splines_3d : list of tuples
        List of (spline_x, spline_y, spline_z, ts) where each spline_* is a scipy.interpolate.BSpline
        object and ts is the array of timestamps where the spline is defined
    
    camera_detections : ndarray
        Array of shape (N, 3) where each row is [timestamp, x, y] with timestamp being the global
        timestamp of the detection and (x, y) being the pixel coordinates of the detection
    
    camera_K : ndarray
        3x3 camera intrinsic matrix
    
    camera_dist_coeffs : ndarray
        Camera distortion coefficients
    
    initial_rvec : ndarray, optional
        Initial rotation vector (Rodrigues format) for the camera pose. If None, assumed as zeros.
    
    initial_tvec : ndarray, optional
        Initial translation vector for the camera pose. If None, assumed as zeros.
    
    optimize_splines : bool, optional
        If True, optimize spline coefficients along with camera pose. Default is False.
    
    kinetic_energy_weight : float, optional
        Weight for the kinetic energy term (velocity regularization). Default is 1.0.
    
    smoothness_weight : float, optional
        Weight for spline smoothness regularization. Default is 0.1.
    
    Returns:
    --------
    dict
        Dictionary containing optimized parameters:
        - 'rvec': Optimized rotation vector
        - 'tvec': Optimized translation vector
        - 'R': Optimized rotation matrix
        - 'reprojection_error': Final mean reprojection error
        - 'inliers': Number of inliers used in optimization
        - 'total_points': Total number of correspondences
        - 'optimized_splines': Optimized splines (if optimize_splines=True)
    """
    
    # Initialize camera pose if not provided
    if initial_rvec is None:
        initial_rvec = np.zeros(3)
    if initial_tvec is None:
        initial_tvec = np.zeros(3)
        
    # Ensure proper shape for rvec
    if initial_rvec.ndim == 1:
        initial_rvec = initial_rvec.reshape(3, 1)
    
    if not optimize_splines:
        # Use original implementation when not optimizing splines
        return _bundle_adjust_camera_only(splines_3d, camera_detections, camera_K, 
                                        camera_dist_coeffs, initial_rvec, initial_tvec)
    else:
        # Use enhanced implementation with spline optimization
        return _bundle_adjust_camera_and_splines(splines_3d, camera_detections, camera_K, 
                                               camera_dist_coeffs, initial_rvec, initial_tvec,
                                               kinetic_energy_weight, smoothness_weight)


def _bundle_adjust_camera_only(splines_3d, camera_detections, camera_K, camera_dist_coeffs, 
                              initial_rvec, initial_tvec):
    """Original bundle adjustment implementation - camera pose only."""
    
    # Generate correspondences between 3D points and 2D detections
    correspondences = []
    
    for detection in camera_detections:
        global_ts = detection[0]
        pixel_x, pixel_y = detection[1], detection[2]
        
        # Find the appropriate spline for this timestamp
        for spline_x, spline_y, spline_z, ts in splines_3d:
            if np.min(ts) <= global_ts <= np.max(ts):
                # Evaluate spline at this timestamp
                x3d = float(spline_x(global_ts))
                y3d = float(spline_y(global_ts))
                z3d = float(spline_z(global_ts))
                
                correspondences.append({
                    'point3d': np.array([x3d, y3d, z3d]),
                    'point2d': np.array([pixel_x, pixel_y]),
                    'timestamp': global_ts
                })
                break
    
    if not correspondences:
        raise ValueError("No correspondences found between 3D splines and camera detections")
    
    # Define the reprojection error function for optimization
    def compute_residuals(params):
        # Extract rotation and translation parameters
        rvec = params[0:3]
        tvec = params[3:6]
        
        residuals = []
        for corr in correspondences:
            # Project 3D point to camera image plane
            point3d = corr['point3d'].reshape(1, 3)
            projected_point, _ = cv.projectPoints(point3d, rvec, tvec, camera_K, camera_dist_coeffs)
            projected_point = projected_point.reshape(-1)
            
            # Calculate residual (reprojection error)
            error = corr['point2d'] - projected_point
            residuals.extend(error)
        
        return np.array(residuals)
    
    # Initial parameters (rotation vector and translation vector)
    initial_params = np.concatenate([initial_rvec.flatten(), initial_tvec.flatten()])
    
    # Set optimization boundaries for parameters
    rvec_bounds = (initial_rvec.flatten() - np.deg2rad(90), initial_rvec.flatten() + np.deg2rad(90))  # +-90 degrees
    tvec_bounds = (initial_tvec.flatten() - 1, initial_tvec.flatten() + 1)  # +-1 unit
    bounds = (np.concatenate([rvec_bounds[0], tvec_bounds[0]]),
              np.concatenate([rvec_bounds[1], tvec_bounds[1]]))
    
    # Perform optimization using Levenberg-Marquardt algorithm with bounds
    print(f"Starting bundle adjustment with {len(correspondences)} correspondences...")
    result = least_squares(
        compute_residuals,
        initial_params,
        bounds=bounds,
        method='trf',  # Trust Region Reflective algorithm supports bounds
        ftol=1e-8,
        xtol=1e-8,
        verbose=0
    )
    
    # Extract optimized parameters
    optimized_rvec = result.x[0:3]
    optimized_tvec = result.x[3:6]
    
    # Convert rotation vector to rotation matrix
    optimized_R, _ = cv.Rodrigues(optimized_rvec)
    
    # Compute final reprojection error statistics
    final_residuals = compute_residuals(result.x)
    residuals_squared = np.square(final_residuals).reshape(-1, 2).sum(axis=1)
    mean_reprojection_error = np.sqrt(np.mean(residuals_squared))
    
    # Count inliers (points with reprojection error below threshold)
    inlier_threshold = 12.0  # pixels
    inliers = np.sum(np.sqrt(residuals_squared) < inlier_threshold)
    
    print(f"Done.")
    
    return {
        'rvec': optimized_rvec,
        'tvec': optimized_tvec,
        'R': optimized_R,
        'reprojection_error': mean_reprojection_error,
        'inliers': inliers,
        'total_points': len(correspondences),
    }


def _bundle_adjust_camera_and_splines(splines_3d, camera_detections, camera_K, camera_dist_coeffs,
                                    initial_rvec, initial_tvec, kinetic_energy_weight, smoothness_weight):
    """Enhanced bundle adjustment with spline coefficient optimization - iterative approach."""
    
    # Generate correspondences and group by spline
    spline_correspondences = {}  # spline_index -> list of correspondences
    spline_info = []
    
    for i, (spline_x, spline_y, spline_z, ts) in enumerate(splines_3d):
        spline_info.append({
            'spline_x': spline_x,
            'spline_y': spline_y, 
            'spline_z': spline_z,
            'ts': ts,
            'index': i
        })
        spline_correspondences[i] = []
    
    for detection in camera_detections:
        global_ts = detection[0]
        pixel_x, pixel_y = detection[1], detection[2]
        
        # Find the appropriate spline for this timestamp
        for spline_idx, info in enumerate(spline_info):
            if np.min(info['ts']) <= global_ts <= np.max(info['ts']):
                spline_correspondences[spline_idx].append({
                    'point2d': np.array([pixel_x, pixel_y]),
                    'timestamp': global_ts,
                    'spline_index': spline_idx
                })
                break
    
    # Remove splines with no correspondences
    active_splines = {k: v for k, v in spline_correspondences.items() if len(v) > 0}
    
    if not active_splines:
        raise ValueError("No correspondences found between 3D splines and camera detections")
    
    print(f"Starting iterative bundle adjustment with {len(active_splines)} splines...")
    
    # Initialize current camera pose
    current_rvec = initial_rvec.flatten()
    current_tvec = initial_tvec.flatten()
    current_splines = [info for info in spline_info]
    
    # Iterative optimization: alternate between camera pose and individual splines
    max_iterations = 5
    for iteration in range(max_iterations):
        print(f"Iteration {iteration + 1}/{max_iterations}")
        
        # Step 1: Optimize camera pose with current splines
        current_rvec, current_tvec = _optimize_camera_pose_only(
            current_splines, active_splines, camera_K, camera_dist_coeffs,
            current_rvec, current_tvec
        )
        
        # Step 2: Optimize each spline individually with current camera pose
        for spline_idx in active_splines.keys():
            if len(active_splines[spline_idx]) < 3:  # Need minimum points for spline optimization
                continue
                
            current_splines[spline_idx] = _optimize_single_spline(
                current_splines[spline_idx], active_splines[spline_idx],
                camera_K, camera_dist_coeffs, current_rvec, current_tvec,
                kinetic_energy_weight, smoothness_weight
            )
    
    # Final evaluation
    final_correspondences = []
    for spline_corrs in active_splines.values():
        final_correspondences.extend(spline_corrs)
    
    # Convert rotation vector to rotation matrix
    optimized_R, _ = cv.Rodrigues(current_rvec)
    
    # Compute final reprojection error
    reprojection_errors = []
    for corr in final_correspondences:
        spline_idx = corr['spline_index']
        timestamp = corr['timestamp']
        
        info = current_splines[spline_idx]
        x3d = float(info['spline_x'](timestamp))
        y3d = float(info['spline_y'](timestamp))
        z3d = float(info['spline_z'](timestamp))
        point3d = np.array([x3d, y3d, z3d])
        
        projected_point, _ = cv.projectPoints(point3d.reshape(1, 3), current_rvec, current_tvec,
                                            camera_K, camera_dist_coeffs)
        projected_point = projected_point.reshape(-1)
        
        error = np.linalg.norm(corr['point2d'] - projected_point)
        reprojection_errors.append(error)
    
    mean_reprojection_error = np.mean(reprojection_errors)
    inlier_threshold = 12.0
    inliers = np.sum(np.array(reprojection_errors) < inlier_threshold)
    
    # Prepare output splines
    optimized_splines = []
    for i, info in enumerate(current_splines):
        optimized_splines.append((info['spline_x'], info['spline_y'], info['spline_z'], info['ts']))
    
    print(f"Done. Iterative optimization completed.")
    
    return {
        'rvec': current_rvec,
        'tvec': current_tvec,
        'R': optimized_R,
        'reprojection_error': mean_reprojection_error,
        'inliers': inliers,
        'total_points': len(final_correspondences),
        'optimized_splines': optimized_splines
    }


def _optimize_camera_pose_only(spline_info, spline_correspondences, camera_K, camera_dist_coeffs,
                              initial_rvec, initial_tvec):
    """Optimize only camera pose with fixed splines."""
    
    def compute_camera_residuals(camera_params):
        rvec = camera_params[0:3]
        tvec = camera_params[3:6]
        
        residuals = []
        for spline_idx, correspondences in spline_correspondences.items():
            info = spline_info[spline_idx]
            
            for corr in correspondences:
                timestamp = corr['timestamp']
                
                # Evaluate current spline at timestamp
                x3d = float(info['spline_x'](timestamp))
                y3d = float(info['spline_y'](timestamp))
                z3d = float(info['spline_z'](timestamp))
                point3d = np.array([x3d, y3d, z3d])
                
                # Project to image
                projected_point, _ = cv.projectPoints(point3d.reshape(1, 3), rvec, tvec,
                                                    camera_K, camera_dist_coeffs)
                projected_point = projected_point.reshape(-1)
                
                # Reprojection error
                error = corr['point2d'] - projected_point
                residuals.extend(error)
        
        return np.array(residuals)
    
    initial_camera_params = np.concatenate([initial_rvec, initial_tvec])
    
    result = least_squares(
        compute_camera_residuals,
        initial_camera_params,
        method='lm',
        ftol=1e-8,
        xtol=1e-8,
        verbose=0
    )
    
    return result.x[0:3], result.x[3:6]


def _optimize_single_spline(spline_info, correspondences, camera_K, camera_dist_coeffs,
                           rvec, tvec, kinetic_energy_weight, smoothness_weight):
    """Optimize a single spline with fixed camera pose."""
    
    def create_spline_from_coeffs(original_spline, new_coeffs):
        return BSpline(original_spline.t, new_coeffs, original_spline.k)
    
    # Get initial coefficients
    initial_coeffs_x = spline_info['spline_x'].c
    initial_coeffs_y = spline_info['spline_y'].c
    initial_coeffs_z = spline_info['spline_z'].c
    
    n_coeffs = len(initial_coeffs_x)
    
    def compute_spline_residuals(spline_params):
        # Extract coefficients for each dimension
        coeffs_x = spline_params[0:n_coeffs]
        coeffs_y = spline_params[n_coeffs:2*n_coeffs]
        coeffs_z = spline_params[2*n_coeffs:3*n_coeffs]
        
        # Create updated splines
        spline_x = create_spline_from_coeffs(spline_info['spline_x'], coeffs_x)
        spline_y = create_spline_from_coeffs(spline_info['spline_y'], coeffs_y)
        spline_z = create_spline_from_coeffs(spline_info['spline_z'], coeffs_z)
        
        residuals = []
        points_3d = []
        timestamps = []
        
        # 1. Reprojection residuals
        for corr in correspondences:
            timestamp = corr['timestamp']
            
            # Evaluate spline at timestamp
            x3d = float(spline_x(timestamp))
            y3d = float(spline_y(timestamp))
            z3d = float(spline_z(timestamp))
            point3d = np.array([x3d, y3d, z3d])
            
            # Project to image
            projected_point, _ = cv.projectPoints(point3d.reshape(1, 3), rvec, tvec,
                                                camera_K, camera_dist_coeffs)
            projected_point = projected_point.reshape(-1)
            
            # Reprojection error
            error = corr['point2d'] - projected_point
            residuals.extend(error)
            
            points_3d.append(point3d)
            timestamps.append(timestamp)
        
        # 2. Kinetic energy residuals
        if kinetic_energy_weight > 0 and len(points_3d) > 1:
            sorted_indices = np.argsort(timestamps)
            sorted_points = np.array(points_3d)[sorted_indices]
            sorted_timestamps = np.array(timestamps)[sorted_indices]
            
            for i in range(len(sorted_points) - 1):
                dt = sorted_timestamps[i + 1] - sorted_timestamps[i]
                if dt > 0:
                    velocity = (sorted_points[i + 1] - sorted_points[i]) / dt
                    velocity_magnitude = np.linalg.norm(velocity)
                    residuals.append(kinetic_energy_weight * velocity_magnitude)
        
        # 3. Smoothness residuals
        if smoothness_weight > 0:
            for coeffs in [coeffs_x, coeffs_y, coeffs_z]:
                if len(coeffs) > 2:
                    second_derivatives = coeffs[2:] - 2 * coeffs[1:-1] + coeffs[:-2]
                    residuals.extend(smoothness_weight * second_derivatives)
        
        return np.array(residuals)
    
    # Initial parameters for this spline
    initial_spline_params = np.concatenate([initial_coeffs_x, initial_coeffs_y, initial_coeffs_z])
    
    # Optimize this spline
    result = least_squares(
        compute_spline_residuals,
        initial_spline_params,
        method='lm',
        ftol=1e-8,
        xtol=1e-8,
        verbose=0
    )
    
    # Extract optimized coefficients and create new splines
    optimized_coeffs_x = result.x[0:n_coeffs]
    optimized_coeffs_y = result.x[n_coeffs:2*n_coeffs]
    optimized_coeffs_z = result.x[2*n_coeffs:3*n_coeffs]
    
    new_spline_x = create_spline_from_coeffs(spline_info['spline_x'], optimized_coeffs_x)
    new_spline_y = create_spline_from_coeffs(spline_info['spline_y'], optimized_coeffs_y)
    new_spline_z = create_spline_from_coeffs(spline_info['spline_z'], optimized_coeffs_z)
    
    # Return updated spline info
    return {
        'spline_x': new_spline_x,
        'spline_y': new_spline_y,
        'spline_z': new_spline_z,
        'ts': spline_info['ts'],
        'index': spline_info['index']
    }