import numpy as np
import cv2 as cv
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R


def bundle_adjust_camera_pose(splines_3d, camera_detections, camera_K, camera_dist_coeffs, initial_rvec=None, initial_tvec=None):
    """
    Perform bundle adjustment to refine camera pose based on 3D splines and timestamped camera detections.
    
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
    
    Returns:
    --------
    dict
        Dictionary containing optimized camera pose parameters:
        - 'rvec': Optimized rotation vector
        - 'tvec': Optimized translation vector
        - 'R': Optimized rotation matrix
        - 'reprojection_error': Final mean reprojection error
        - 'inliers': Number of inliers used in optimization
        - 'total_points': Total number of correspondences
    """
    # Initialize camera pose if not provided
    if initial_rvec is None:
        initial_rvec = np.zeros(3)
    if initial_tvec is None:
        initial_tvec = np.zeros(3)
        
    # Check if rvec is in the right format
    if initial_rvec.shape != (3,1):
        raise ValueError("Initial rotation vector must be of shape (3,1)")
    
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
    
    # Perform optimization using Levenberg-Marquardt algorithm
    print(f"Starting bundle adjustment with {len(correspondences)} correspondences...")
    result = least_squares(
        compute_residuals,
        initial_params,
        method='lm',
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
    
    # print(f"Bundle adjustment completed:")
    # print(f"  Initial rotation vector: {initial_rvec.flatten()}")
    # print(f"  Optimized rotation vector: {optimized_rvec}")
    # print(f"  Initial translation vector: {initial_tvec.flatten()}")
    # print(f"  Optimized translation vector: {optimized_tvec}")
    # print(f"  Mean reprojection error: {mean_reprojection_error:.4f} pixels")
    # print(f"  Inliers: {inliers}/{len(correspondences)} ({inliers/len(correspondences)*100:.2f}%)")
    
    return {
        'rvec': optimized_rvec,
        'tvec': optimized_tvec,
        'R': optimized_R,
        'reprojection_error': mean_reprojection_error,
        'inliers': inliers,
        'total_points': len(correspondences),
        # 'correspondences': correspondences
    }


def visualize_bundle_adjustment_results(image, correspondences, camera_K, camera_dist_coeffs, 
                                        initial_rvec, initial_tvec, 
                                        optimized_rvec, optimized_tvec):
    """
    Visualize the results of bundle adjustment by showing original detections, 
    initial reprojections, and optimized reprojections.
    
    Parameters:
    -----------
    image : ndarray
        Camera image to draw on
    correspondences : list
        List of correspondence dictionaries
    camera_K, camera_dist_coeffs : camera parameters
    initial_rvec, initial_tvec : initial camera pose
    optimized_rvec, optimized_tvec : optimized camera pose
    
    Returns:
    --------
    ndarray
        Visualization image
    """
    import cv2 as cv
    import numpy as np
    
    # Create a copy of the image (or a blank one if None is provided)
    if image is None:
        vis_image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    else:
        vis_image = image.copy()
    
    for corr in correspondences:
        # Original detection point (green)
        pt_orig = tuple(map(int, corr['point2d']))
        cv.circle(vis_image, pt_orig, 5, (0, 255, 0), -1)
        
        # Initial reprojection (red)
        point3d = corr['point3d'].reshape(1, 3)
        initial_proj, _ = cv.projectPoints(point3d, initial_rvec, initial_tvec, camera_K, camera_dist_coeffs)
        pt_init = tuple(map(int, initial_proj.reshape(-1)))
        cv.circle(vis_image, pt_init, 3, (0, 0, 255), -1)
        
        # Optimized reprojection (blue)
        optimized_proj, _ = cv.projectPoints(point3d, optimized_rvec, optimized_tvec, camera_K, camera_dist_coeffs)
        pt_opt = tuple(map(int, optimized_proj.reshape(-1)))
        cv.circle(vis_image, pt_opt, 3, (255, 0, 0), -1)
        
        # Draw lines connecting the points
        cv.line(vis_image, pt_orig, pt_init, (0, 0, 255), 1)  # Original to initial (red)
        cv.line(vis_image, pt_orig, pt_opt, (255, 0, 0), 1)   # Original to optimized (blue)
    
    # Add legend
    cv.putText(vis_image, "Original Detections (Green)", (30, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv.putText(vis_image, "Initial Reprojections (Red)", (30, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv.putText(vis_image, "Optimized Reprojections (Blue)", (30, 90), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    cv.imshow("Bundle Adjustment Visualization", vis_image)
    cv.waitKey(0)
    
    return vis_image


def iterative_bundle_adjustment(splines_3d, camera_detections, camera_K, camera_dist_coeffs, 
                               initial_rvec=None, initial_tvec=None, max_iterations=5, 
                               inlier_threshold=5.0):
    """
    Perform iterative bundle adjustment with outlier rejection to improve robustness.
    
    Parameters:
    -----------
    splines_3d, camera_detections, camera_K, camera_dist_coeffs : Same as bundle_adjust_camera_pose
    initial_rvec, initial_tvec : Initial camera pose parameters
    max_iterations : Maximum number of refinement iterations
    inlier_threshold : Threshold for considering a point as an inlier (in pixels)
    
    Returns:
    --------
    dict
        Same as bundle_adjust_camera_pose with additional iteration statistics
    """
    # Initialize camera pose if not provided
    if initial_rvec is None:
        initial_rvec = np.zeros(3)
    if initial_tvec is None:
        initial_tvec = np.zeros(3)
    
    current_rvec = initial_rvec.copy()
    current_tvec = initial_tvec.copy()
    
    iteration_stats = []
    
    # Generate initial correspondences between 3D points and 2D detections
    all_correspondences = []
    
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
                
                all_correspondences.append({
                    'point3d': np.array([x3d, y3d, z3d]),
                    'point2d': np.array([pixel_x, pixel_y]),
                    'timestamp': global_ts
                })
                break
    
    if not all_correspondences:
        raise ValueError("No correspondences found between 3D splines and camera detections")
    
    print(f"Starting iterative bundle adjustment with {len(all_correspondences)} correspondences")
    
    # Iterative refinement
    active_correspondences = all_correspondences
    
    for iteration in range(max_iterations):
        print(f"\nIteration {iteration+1}/{max_iterations}")
        print(f"Using {len(active_correspondences)} correspondences")
        
        # Define the reprojection error function for this iteration
        def compute_residuals(params):
            rvec = params[0:3]
            tvec = params[3:6]
            
            residuals = []
            for corr in active_correspondences:
                point3d = corr['point3d'].reshape(1, 3)
                projected_point, _ = cv.projectPoints(point3d, rvec, tvec, camera_K, camera_dist_coeffs)
                projected_point = projected_point.reshape(-1)
                
                error = corr['point2d'] - projected_point
                residuals.extend(error)
            
            return np.array(residuals)
        
        # Optimize camera pose parameters
        initial_params = np.concatenate([current_rvec.flatten(), current_tvec.flatten()])
        
        result = least_squares(
            compute_residuals,
            initial_params,
            method='lm',
            ftol=1e-8,
            xtol=1e-8,
            verbose=0
        )
        
        current_rvec = result.x[0:3]
        current_tvec = result.x[3:6]
        
        # Evaluate reprojection errors for all points
        all_errors = []
        for corr in all_correspondences:
            point3d = corr['point3d'].reshape(1, 3)
            projected_point, _ = cv.projectPoints(point3d, current_rvec, current_tvec, camera_K, camera_dist_coeffs)
            projected_point = projected_point.reshape(-1)
            
            error = np.linalg.norm(corr['point2d'] - projected_point)
            all_errors.append(error)
        
        all_errors = np.array(all_errors)
        
        # Filter inliers for next iteration
        inlier_mask = all_errors < inlier_threshold
        active_correspondences = [corr for i, corr in enumerate(all_correspondences) if inlier_mask[i]]
        
        # Compute statistics
        mean_error = np.mean(all_errors)
        median_error = np.median(all_errors)
        num_inliers = np.sum(inlier_mask)
        inlier_ratio = num_inliers / len(all_correspondences)
        
        iteration_stats.append({
            'iteration': iteration + 1,
            'mean_error': mean_error,
            'median_error': median_error,
            'num_inliers': num_inliers,
            'inlier_ratio': inlier_ratio,
            'rvec': current_rvec.copy(),
            'tvec': current_tvec.copy()
        })
        
        print(f"  Mean error: {mean_error:.4f} pixels")
        print(f"  Median error: {median_error:.4f} pixels")
        print(f"  Inliers: {num_inliers}/{len(all_correspondences)} ({inlier_ratio*100:.2f}%)")
        
        # Stop if inlier ratio is not improving significantly
        if iteration > 0:
            if iteration_stats[-1]['inlier_ratio'] - iteration_stats[-2]['inlier_ratio'] < 0.01:
                print("Inlier ratio not improving significantly. Stopping iterations.")
                break
    
    # Final optimization with all inliers
    optimized_R, _ = cv.Rodrigues(current_rvec)
    
    return {
        'rvec': current_rvec,
        'tvec': current_tvec,
        'R': optimized_R,
        'reprojection_error': iteration_stats[-1]['mean_error'],
        'inliers': iteration_stats[-1]['num_inliers'],
        'total_points': len(all_correspondences),
        'iteration_stats': iteration_stats,
        'correspondences': active_correspondences
    }


# Example usage
if __name__ == "__main__":
    # This is just an example showing how to use the function
    # You would need to replace these with actual data
    
    # Load spline data and camera detections
    from scipy.interpolate import make_interp_spline
    import matplotlib.pyplot as plt
    
    # Example spline creation (in a real scenario, these would come from your data)
    t = np.linspace(0, 10, 100)
    x = np.sin(t)
    y = np.cos(t)
    z = t / 10
    
    spline_x = make_interp_spline(t, x, k=3)
    spline_y = make_interp_spline(t, y, k=3)
    spline_z = make_interp_spline(t, z, k=3)
    
    splines_3d = [(spline_x, spline_y, spline_z, t)]
    
    # Example camera detections (timestamp, x, y)
    # In real case, these would come from your camera data
    num_points = 50
    sample_times = np.sort(np.random.choice(t, num_points, replace=False))
    
    # Create synthetic camera with some rotation and translation
    K = np.array([
        [1000, 0, 960],
        [0, 1000, 540],
        [0, 0, 1]
    ])
    dist_coeffs = np.zeros(5)
    
    true_rvec = np.array([0.1, -0.2, 0.05])
    true_tvec = np.array([0.5, 0.3, -2.0])
    
    # Generate synthetic detections
    detections = []
    for ts in sample_times:
        # Get 3D point from spline
        x3d = float(spline_x(ts))
        y3d = float(spline_y(ts))
        z3d = float(spline_z(ts))
        
        # Project to image
        pt3d = np.array([[x3d, y3d, z3d]])
        pt2d, _ = cv.projectPoints(pt3d, true_rvec, true_tvec, K, dist_coeffs)
        
        # Add some noise
        noise = np.random.normal(0, 2, 2)  # 2 pixel standard deviation
        pt2d = pt2d.reshape(-1) + noise
        
        detections.append([ts, pt2d[0], pt2d[1]])
    
    detections = np.array(detections)
    
    # Add some outliers (about 10%)
    num_outliers = len(detections) // 10
    outlier_indices = np.random.choice(len(detections), num_outliers, replace=False)
    for idx in outlier_indices:
        detections[idx, 1:] = detections[idx, 1:] + np.random.uniform(-100, 100, 2)
    
    # Initial guess for camera pose (slightly perturbed from true value)
    initial_rvec = true_rvec + np.random.normal(0, 0.1, 3)
    initial_tvec = true_tvec + np.random.normal(0, 0.2, 3)
    
    # Run bundle adjustment
    result = iterative_bundle_adjustment(
        splines_3d, 
        detections, 
        K, 
        dist_coeffs,
        initial_rvec=initial_rvec,
        initial_tvec=initial_tvec,
        max_iterations=5
    )
    
    visualize_bundle_adjustment_results(
        None,  # No image provided, just for demonstration
        result['correspondences'],
        K,
        dist_coeffs,
        initial_rvec,
        initial_tvec,
        result['rvec'],
        result['tvec']
    )
    
    # Print results
    print("\nFinal Results:")
    print(f"True rotation vector: {true_rvec}")
    print(f"Initial rotation vector: {initial_rvec}")
    print(f"Optimized rotation vector: {result['rvec']}")
    print(f"Rotation vector error: {np.linalg.norm(true_rvec - result['rvec'])}")
    
    print(f"\nTrue translation vector: {true_tvec}")
    print(f"Initial translation vector: {initial_tvec}")
    print(f"Optimized translation vector: {result['tvec']}")
    print(f"Translation vector error: {np.linalg.norm(true_tvec - result['tvec'])}")
    
    # Plot iteration statistics
    plt.figure(figsize=(12, 8))
    
    plt.subplot(211)
    iterations = [stat['iteration'] for stat in result['iteration_stats']]
    mean_errors = [stat['mean_error'] for stat in result['iteration_stats']]
    median_errors = [stat['median_error'] for stat in result['iteration_stats']]
    
    plt.plot(iterations, mean_errors, 'o-', label='Mean Error')
    plt.plot(iterations, median_errors, 's-', label='Median Error')
    plt.xlabel('Iteration')
    plt.ylabel('Reprojection Error (pixels)')
    plt.title('Error Reduction During Bundle Adjustment')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(212)
    inlier_ratios = [stat['inlier_ratio'] * 100 for stat in result['iteration_stats']]
    
    plt.plot(iterations, inlier_ratios, 'o-', color='green')
    plt.xlabel('Iteration')
    plt.ylabel('Inlier Percentage (%)')
    plt.title('Inlier Ratio During Bundle Adjustment')
    plt.grid(True)
    
    plt.tight_layout()
