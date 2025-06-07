import numpy as np
import cv2 as cv
from scipy.optimize import least_squares
from scipy.interpolate import BSpline
from scipy.spatial.distance import cdist

def bundle_adjust_camera_pose(splines_3d, camera_detections, camera_K, camera_dist_coeffs, 
                            initial_rvec=None, initial_tvec=None):
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
    

    return _bundle_adjust_camera_only(splines_3d, camera_detections, camera_K, 
                                        camera_dist_coeffs, initial_rvec, initial_tvec)



def _bundle_adjust_camera_only(splines_3d, camera_detections, camera_K, camera_dist_coeffs, 
                              initial_rvec, initial_tvec):
    
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
