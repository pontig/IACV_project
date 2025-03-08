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

def reprojection_error(params, camera_data, camera_indices, num_cameras, 
                      num_splines, spline_data, time_bounds):
    """
    Compute reprojection error for all cameras.
    
    Parameters:
    -----------
    params : np.array
        Flattened array of parameters:
        - alphas: num_cameras values
        - betas: num_cameras values
        - Ps: num_cameras * 12 values (3x4 projection matrices flattened)
        - spline_coeffs: coefficients for each B-spline
        
    camera_data : list of DataFrames
        List of DataFrames, one for each camera with detection data.
        
    camera_indices : np.array
        Array mapping camera data indices to actual camera IDs.
    
    num_cameras : int
        Number of cameras
        
    num_splines : int
        Number of spline segments
        
    spline_data : list of tuples
        List containing (knots, degrees) for each spline in each dimension
        
    time_bounds : list of tuples
        List of (min_time, max_time) for each spline
        
    Returns:
    --------
    errors : np.array
        Array of reprojection errors for all points
    """
    # Extract parameters
    alpha_start = 0
    beta_start = num_cameras
    P_start = 2 * num_cameras
    
    alphas = params[alpha_start:beta_start]
    betas = params[beta_start:P_start]
    
    # Reshape Ps into 3x4 matrices
    Ps = []
    for i in range(num_cameras):
        P_flat = params[P_start + i*12:P_start + (i+1)*12]
        P = P_flat.reshape(3, 4)
        Ps.append(P)
    
    # Extract spline coefficients and reconstruct B-splines
    spline_start = P_start + 12 * num_cameras
    spline_param_idx = spline_start
    
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
        x_coefs = params[spline_param_idx:spline_param_idx + n_coefs_x]
        spline_param_idx += n_coefs_x
        
        y_coefs = params[spline_param_idx:spline_param_idx + n_coefs_y]
        spline_param_idx += n_coefs_y
        
        z_coefs = params[spline_param_idx:spline_param_idx + n_coefs_z]
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

def minimize_reprojection_error(dataframe, initial_alphas=None, initial_betas=None, 
                             initial_Ps=None, initial_splines=None):
    """
    Perform bundle adjustment to optimize camera parameters and B-spline coefficients.
    
    Parameters:
    -----------
    dataframe : pandas.DataFrame
        DataFrame with columns for cam_id, frame_id, detection_x, detection_y
        
    initial_alphas : np.array or None
        Initial alpha values for each camera
        
    initial_betas : np.array or None
        Initial beta values for each camera
        
    initial_Ps : list of np.array or None
        Initial projection matrices for each camera
        
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
    
    # Initialize parameters if not provided
    if initial_alphas is None:
        initial_alphas = np.ones(num_cameras)
    
    if initial_betas is None:
        initial_betas = np.zeros(num_cameras)
    
    if initial_Ps is None:
        # Initialize with identity-like projection matrices
        initial_Ps = []
        for i in range(num_cameras):
            P = np.zeros((3, 4))
            P[:3, :3] = np.eye(3)
            initial_Ps.append(P)
    
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
    
    spline_coeffs = np.array(spline_coeffs)
    
    # Flatten all parameters into a single array
    initial_params = np.concatenate([
        initial_alphas,
        initial_betas,
        np.array([P.flatten() for P in initial_Ps]).flatten(),
        spline_coeffs
    ])
    
    # Run optimization
    result = opt.least_squares(
        reprojection_error, 
        initial_params, 
        args=(camera_data, camera_indices, num_cameras, num_splines, spline_data, time_bounds),
        method='trf',  # Trust Region Reflective algorithm
        verbose=2
    )
    
    # Extract optimized parameters
    params = result.x
    alpha_start = 0
    beta_start = num_cameras
    P_start = 2 * num_cameras
    
    optimized_alphas = params[alpha_start:beta_start]
    optimized_betas = params[beta_start:P_start]
    
    optimized_Ps = []
    for i in range(num_cameras):
        P_flat = params[P_start + i*12:P_start + (i+1)*12]
        P = P_flat.reshape(3, 4)
        optimized_Ps.append(P)
    
    # Extract spline coefficients and reconstruct B-splines
    spline_start = P_start + 12 * num_cameras
    spline_param_idx = spline_start
    
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
        x_coefs = params[spline_param_idx:spline_param_idx + n_coefs_x]
        spline_param_idx += n_coefs_x
        
        y_coefs = params[spline_param_idx:spline_param_idx + n_coefs_y]
        spline_param_idx += n_coefs_y
        
        z_coefs = params[spline_param_idx:spline_param_idx + n_coefs_z]
        spline_param_idx += n_coefs_z
        
        # Create B-spline representation for each dimension
        x_spline = (x_knots, x_coefs, x_degree)
        y_spline = (y_knots, y_coefs, y_degree)
        z_spline = (z_knots, z_coefs, z_degree)
        
        optimized_splines.append((x_spline, y_spline, z_spline, time_bounds[i]))
    
    # Return results
    return {
        'alphas': optimized_alphas,
        'betas': optimized_betas,
        'Ps': optimized_Ps,
        'splines': optimized_splines,
        'optimization_result': result
    }
