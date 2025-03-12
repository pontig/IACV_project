import numpy as np
import scipy.optimize as opt
import pandas as pd
from scipy.interpolate import BSpline, splev

from global_fn import compute_global_time

def prepare_data(dataframe):
    """
    Prepare the data for optimization by grouping by camera.
    
    Parameters:
    -----------
    dataframe : pandas.DataFrame
        DataFrame containing detection data with cam_id, frame_id, detection_x, detection_y.
        
    Returns:
    --------
    camera_data : list of DataFrames
        List of DataFrames, one for each camera.
    """
    num_cameras = dataframe['cam_id'].nunique()
    present_cameras = dataframe['cam_id'].unique()
    camera_data = [dataframe[dataframe['cam_id'] == i].copy() for i in present_cameras]
    return camera_data, present_cameras

def evaluate_bspline_at_time(spline, t):
    """
    Evaluate a 3D B-spline at time t.
    
    Parameters:
    -----------
    spline : tuple
        Tuple of (x_spline, y_spline, z_spline, time_range)
        Each *_spline is a tuple of (t, c, k) for BSpline
    t : float
        Time to evaluate spline at
        
    Returns:
    --------
    point : np.array
        3D point [x, y, z] if t is within spline's time range, None otherwise
    """
    if t >= spline[3][0] and t <= spline[3][1]:
        x = splev(t, spline[0], der=0)
        y = splev(t, spline[1], der=0)
        z = splev(t, spline[2], der=0)
        
        return np.array([x, y, z])
    return None

def compose_projection_matrix(R, t, K):
    """
    Compose the projection matrix P from R, t, and K.
    
    Parameters:
    -----------
    R : np.array
        3x3 rotation matrix
    t : np.array
        3x1 translation vector
    K : np.array
        3x3 camera calibration matrix
        
    Returns:
    --------
    P : np.array
        3x4 projection matrix
    """
    # Create [R|t] extrinsic matrix (3x4)
    Rt = np.column_stack((R, t))
    
    # P = K[R|t]
    P = np.dot(K, Rt)
    
    return P

def reprojection_error_splines_only(spline_params, camera_data, camera_indices, num_cameras, num_splines, 
                                   spline_data, time_bounds, calibration_matrices, alphas, betas, poses):
    """
    Compute reprojection error for all cameras when optimizing only spline parameters.
    
    Parameters:
    -----------
    spline_params : np.array
        Flattened array of spline coefficients
    
    camera_data, camera_indices, num_cameras, num_splines, spline_data, time_bounds, calibration_matrices:
        Same as in reprojection_error
        
    alphas : np.array
        Fixed alpha values for each camera
        
    betas : np.array
        Fixed beta values for each camera
        
    poses : list of tuples
        Fixed camera poses (R, t)
    
    Returns:
    --------
    errors : np.array
        Array of reprojection errors for all points
    """
    # Create projection matrices from fixed poses
    Ps = [compose_projection_matrix(R, t, K) for (R, t), K in zip(poses, calibration_matrices)]
    
    # Extract spline coefficients and reconstruct B-splines
    spline_param_idx = 0
    splines = []
    
    for i in range(num_splines):
        # Get knots and degree for this spline
        x_knots, x_degree = spline_data[i][0]
        y_knots, y_degree = spline_data[i][1]
        z_knots, z_degree = spline_data[i][2]
        
        # Number of coefficients for each dimension
        n_coefs_x = len(x_knots) - x_degree - 1
        n_coefs_y = len(y_knots) - y_degree - 1
        n_coefs_z = len(z_knots) - z_degree - 1
        
        # Extract coefficients for each dimension
        x_coefs = spline_params[spline_param_idx:spline_param_idx + n_coefs_x]
        spline_param_idx += n_coefs_x
        
        y_coefs = spline_params[spline_param_idx:spline_param_idx + n_coefs_y]
        spline_param_idx += n_coefs_y
        
        z_coefs = spline_params[spline_param_idx:spline_param_idx + n_coefs_z]
        spline_param_idx += n_coefs_z
        
        # Create B-spline representation for each dimension
        x_spline = (x_knots, x_coefs, x_degree)
        y_spline = (y_knots, y_coefs, y_degree)
        z_spline = (z_knots, z_coefs, z_degree)
        
        splines.append((x_spline, y_spline, z_spline, time_bounds[i]))
    
    # Compute reprojection errors
    all_errors = []
    
    for cam_idx, df in enumerate(camera_data):
        frames = df['frame_id'].values
        detections = df[['detection_x', 'detection_y']].values
        
        for j, frame in enumerate(frames):
            tss = compute_global_time(frame, alphas[cam_idx], betas[cam_idx])
            
            # Find the right spline for this time
            point_3d = None
            for spline in splines:
                point_3d = evaluate_bspline_at_time(spline, tss)
                if point_3d is not None:
                    break
            
            if point_3d is None:
                continue  # Skip if no spline covers this time
                
            # Project 3D point to camera
            point_3d_homogeneous = np.append(point_3d, 1)
            projected_point = np.dot(Ps[cam_idx], point_3d_homogeneous)
            
            # Convert to image coordinates by dividing by third component
            if projected_point[2] == 0:
                # Handle division by zero
                continue
                
            projected_point = projected_point[:2] / projected_point[2]
            
            # Compute error
            error = np.linalg.norm(detections[j] - projected_point)
            all_errors.append(error)
    
    return np.array(all_errors)

def reprojection_error_full(params, camera_data, camera_indices, num_cameras, 
                           num_splines, spline_data, time_bounds, calibration_matrices,
                           spline_params=None):
    """
    Compute reprojection error for all cameras.
    
    Parameters:
    -----------
    params : np.array
        Flattened array of parameters:
        - alphas: num_cameras values
        - betas: num_cameras values
        - Rs_ts: num_cameras * 12 values (3x3 rotation matrices and 3x1 translation vectors flattened)
        
    spline_params : np.array or None
        If provided, use these spline parameters instead of extracting from params
        
    All other parameters are the same as in reprojection_error
        
    Returns:
    --------
    errors : np.array
        Array of reprojection errors for all points
    """
    # Extract parameters
    alpha_start = 0
    beta_start = num_cameras
    Rt_start = 2 * num_cameras
    
    alphas = params[alpha_start:beta_start]
    betas = params[beta_start:Rt_start]
    
    # Reshape Rs and ts
    Rs = []
    ts = []
    for i in range(num_cameras):
        # Extract flattened R (9 values) and t (3 values)
        Rt_flat = params[Rt_start + i*12:Rt_start + (i+1)*12]
        R = Rt_flat[:9].reshape(3, 3)
        t = Rt_flat[9:12].reshape(3, 1)
        
        Rs.append(R)
        ts.append(t)
    
    # If spline_params is None, extract from params
    if spline_params is None:
        spline_start = Rt_start + 12 * num_cameras
        spline_params = params[spline_start:]
    
    # Extract spline coefficients and reconstruct B-splines
    spline_param_idx = 0
    
    splines = []
    for i in range(num_splines):
        # Get knots and degree for this spline
        x_knots, x_degree = spline_data[i][0]
        y_knots, y_degree = spline_data[i][1]
        z_knots, z_degree = spline_data[i][2]
        
        # Number of coefficients for each dimension
        n_coefs_x = len(x_knots) - x_degree - 1
        n_coefs_y = len(y_knots) - y_degree - 1
        n_coefs_z = len(z_knots) - z_degree - 1
        
        # Extract coefficients for each dimension
        x_coefs = spline_params[spline_param_idx:spline_param_idx + n_coefs_x]
        spline_param_idx += n_coefs_x
        
        y_coefs = spline_params[spline_param_idx:spline_param_idx + n_coefs_y]
        spline_param_idx += n_coefs_y
        
        z_coefs = spline_params[spline_param_idx:spline_param_idx + n_coefs_z]
        spline_param_idx += n_coefs_z
        
        # Create B-spline representation for each dimension
        x_spline = (x_knots, x_coefs, x_degree)
        y_spline = (y_knots, y_coefs, y_degree)
        z_spline = (z_knots, z_coefs, z_degree)
        
        splines.append((x_spline, y_spline, z_spline, time_bounds[i]))
    
    # Compute reprojection errors
    all_errors = []
    
    for cam_idx, df in enumerate(camera_data):
        # Get actual camera ID
        actual_cam_id = camera_indices[cam_idx]
        frames = df['frame_id'].values
        detections = df[['detection_x', 'detection_y']].values
        
        # Compose projection matrix for this camera
        P = compose_projection_matrix(Rs[cam_idx], ts[cam_idx], calibration_matrices[cam_idx])
        
        for j, frame in enumerate(frames):
            tss = compute_global_time(frame, alphas[cam_idx], betas[cam_idx])
            
            # Find the right spline for this time
            point_3d = None
            for spline in splines:
                point_3d = evaluate_bspline_at_time(spline, tss)
                if point_3d is not None:
                    break
            
            if point_3d is None:
                continue  # Skip if no spline covers this time
                
            # Project 3D point to camera
            point_3d_homogeneous = np.append(point_3d, 1)
            projected_point = np.dot(P, point_3d_homogeneous)
            
            # Convert to image coordinates by dividing by third component
            if projected_point[2] == 0:
                # Handle division by zero
                continue
                
            projected_point = projected_point[:2] / projected_point[2]
            
            # Compute error
            error = np.linalg.norm(detections[j] - projected_point)
            all_errors.append(error)
    
    return np.array(all_errors)

def minimize_reprojection_error(dataframe, calibration_matrices, initial_alphas=None, initial_betas=None, 
                             initial_poses=None, initial_splines=None):
    """
    Perform bundle adjustment to optimize camera parameters and B-spline coefficients.
    Uses an iterative approach to first optimize spline parameters, then all parameters.
    
    Parameters:
    -----------
    dataframe : pandas.DataFrame
        DataFrame with columns for cam_id, frame_id, detection_x, detection_y
        
    calibration_matrices : list of np.array
        List of 3x3 camera calibration matrices (K) for each camera
        
    initial_alphas : np.array or None
        Initial alpha values for each camera
        
    initial_betas : np.array or None
        Initial beta values for each camera
        
    initial_poses : list of tuples or None
        Initial camera poses as list of (R, t) tuples, where:
        - R is a 3x3 rotation matrix
        - t is a 3x1 translation vector
        
    initial_splines : list of tuples or None
        Initial B-splines in the format [(t_x, c_x, k_x), (t_y, c_y, k_y), (t_z, c_z, k_z), time_range]
        Where t_* are knots, c_* are coefficients, k_* are degrees, and time_range is (min_t, max_t)
        
    Returns:
    --------
    result : dict
        Dictionary containing optimized parameters and optimization results
    """
    # Prepare data by camera
    camera_data, camera_indices = prepare_data(dataframe)
    num_cameras = len(camera_data)
    
    # Verify that we have calibration matrices for all cameras
    if len(calibration_matrices) != num_cameras:
        raise ValueError(f"Expected {num_cameras} calibration matrices, got {len(calibration_matrices)}")
    
    # Initialize parameters if not provided
    if initial_alphas is None:
        initial_alphas = np.ones(num_cameras)
    
    if initial_betas is None:
        initial_betas = np.zeros(num_cameras)
    
    if initial_poses is None:
        # Initialize with identity rotations and zero translations
        initial_poses = []
        for i in range(num_cameras):
            R = np.eye(3)
            t = np.zeros((3, 1))
            initial_poses.append((R, t))
    
    # Determine number of splines and extract their data
    if initial_splines is None:
        raise ValueError("Initial B-splines must be provided")
    
    num_splines = len(initial_splines)
    spline_coeffs = []
    spline_data = []  # Store knots and degrees
    time_bounds = []  # Store time ranges
    
    for spline in initial_splines:
        # Extract components
        x_spline, y_spline, z_spline, time_range = spline
        
        # Extract and store time bounds
        time_bounds.append(time_range)
        
        # Extract knots, coefficients, and degrees for each dimension
        t_x, c_x, k_x = x_spline.t, x_spline.c, x_spline.k
        t_y, c_y, k_y = y_spline.t, y_spline.c, y_spline.k
        t_z, c_z, k_z = z_spline.t, z_spline.c, z_spline.k
        
        # Store knots and degrees for reconstruction
        spline_data.append([(t_x, k_x), (t_y, k_y), (t_z, k_z)])
        
        # Append coefficients to flatten into parameter vector
        spline_coeffs.extend(c_x)
        spline_coeffs.extend(c_y)
        spline_coeffs.extend(c_z)
    
    initial_spline_params = np.array(spline_coeffs)
    
    # Flatten camera parameters into a single array for later use
    pose_params = []
    for R, t in initial_poses:
        pose_params.extend(R.flatten())
        pose_params.extend(t.flatten())
    
    camera_params = np.concatenate([
        initial_alphas,
        initial_betas,
        np.array(pose_params)
    ])
    
    print("Stage 1: Optimizing spline parameters with fixed camera parameters...")
    
    # First optimization: fix camera parameters, optimize splines
    spline_result = opt.least_squares(
        reprojection_error_splines_only, 
        initial_spline_params, 
        args=(camera_data, camera_indices, num_cameras, num_splines, spline_data, 
              time_bounds, calibration_matrices, initial_alphas, initial_betas, initial_poses),
        method='trf',
        verbose=2
    )
    
    # Get optimized spline parameters
    optimized_spline_params = spline_result.x
    
    print("Stage 2: Optimizing all parameters together...")
    
    # Second optimization: optimize all parameters together, starting from the results of first stage
    initial_full_params = np.concatenate([camera_params, optimized_spline_params])
    
    full_result = opt.least_squares(
        reprojection_error_full, 
        initial_full_params, 
        args=(camera_data, camera_indices, num_cameras, num_splines, spline_data, 
              time_bounds, calibration_matrices),
        method='trf',
        verbose=2
    )
    
    # Extract optimized parameters
    params = full_result.x
    alpha_start = 0
    beta_start = num_cameras
    Rt_start = 2 * num_cameras
    spline_start = Rt_start + 12 * num_cameras
    
    optimized_alphas = params[alpha_start:beta_start]
    optimized_betas = params[beta_start:Rt_start]
    
    # Extract optimized camera poses
    optimized_poses = []
    for i in range(num_cameras):
        Rt_flat = params[Rt_start + i*12:Rt_start + (i+1)*12]
        R = Rt_flat[:9].reshape(3, 3)
        t = Rt_flat[9:12].reshape(3, 1)
        optimized_poses.append((R, t))
    
    # Extract spline parameters and reconstruct B-splines
    optimized_spline_params = params[spline_start:]
    spline_param_idx = 0
    
    optimized_splines = []
    for i in range(num_splines):
        # Get knots and degree for this spline
        x_knots, x_degree = spline_data[i][0]
        y_knots, y_degree = spline_data[i][1]
        z_knots, z_degree = spline_data[i][2]
        
        # Number of coefficients for each dimension
        n_coefs_x = len(x_knots) - x_degree - 1
        n_coefs_y = len(y_knots) - y_degree - 1
        n_coefs_z = len(z_knots) - z_degree - 1
        
        # Extract coefficients for each dimension
        x_coefs = optimized_spline_params[spline_param_idx:spline_param_idx + n_coefs_x]
        spline_param_idx += n_coefs_x
        
        y_coefs = optimized_spline_params[spline_param_idx:spline_param_idx + n_coefs_y]
        spline_param_idx += n_coefs_y
        
        z_coefs = optimized_spline_params[spline_param_idx:spline_param_idx + n_coefs_z]
        spline_param_idx += n_coefs_z
        
        # Create actual B-spline objects for each dimension
        x_spline = BSpline(x_knots, x_coefs, x_degree)
        y_spline = BSpline(y_knots, y_coefs, y_degree)
        z_spline = BSpline(z_knots, z_coefs, z_degree)
        
        optimized_splines.append((x_spline, y_spline, z_spline, time_bounds[i]))
    
    # Return results
    return {
        'alphas': optimized_alphas,
        'betas': optimized_betas,
        'poses': optimized_poses,  # List of (R, t) tuples
        'splines': optimized_splines,
        'spline_optimization_result': spline_result,
        'full_optimization_result': full_result
    }