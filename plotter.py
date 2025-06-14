import os
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

## PLOT FUNCTIONS FOR THE BETA SEARCH

def plot_refinement_process(beta_values_coarse, inliers_coarse, best_beta_coarse, max_inliers_coarse,
                            beta_values_fine, inliers_fine, best_beta_fine, max_inliers_fine,
                            beta_values_finer, inliers_finer, best_beta_finer, max_inliers_finer,
                            beta_values_finest, inliers_finest, best_beta_finest, max_inliers_finest,
                            main_camera, secondary_camera, dataset_no,
                            output_dir=""):
    """
    Plot the beta search refinement process
    
    Parameters:
    -----------
    beta_values_* : list
        Lists of beta values for each search level
    inliers_* : list
        Lists of inlier ratios for each search level
    best_beta_* : float
        Best beta value for each search level
    max_inliers_* : float
        Maximum inlier ratio for each search level
    main_camera : int
        ID of main camera
    secondary_camera : int
        ID of secondary camera
    dataset_no : int
        Dataset number
    """
    figsize = (28, 15)
    
    # Create a figure with 2 rows and 2 columns
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f'Beta Search Refinement Process (Cameras {main_camera}-{secondary_camera})', fontsize=20)

    # Coarse search plot
    axes[0, 0].plot(beta_values_coarse, inliers_coarse, 'b.-', markersize=3, label='Inliers')
    axes[0, 0].axvline(x=best_beta_coarse, color='r', linestyle='--', 
                      label=f'Best β={best_beta_coarse} (inliers={max_inliers_coarse:.4f})')
    axes[0, 0].set_title('Coarse Search')
    axes[0, 0].set_xlabel('Beta')
    axes[0, 0].set_ylabel('Inlier Ratio')
    axes[0, 0].set_ylim(-0.1, 1.1)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # Fine search plot
    axes[0, 1].plot(beta_values_fine, inliers_fine, 'r.-', markersize=3, label='Inliers')
    axes[0, 1].axvline(x=best_beta_fine, color='b', linestyle='--', 
                      label=f'Best β={best_beta_fine} (inliers={max_inliers_fine:.4f})')
    axes[0, 1].set_title(f'Fine Search (range={best_beta_coarse}±100)')
    axes[0, 1].set_xlabel('Beta')
    axes[0, 1].set_ylabel('Inlier Ratio')
    axes[0, 1].set_ylim(-0.1, 1.1)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # Finer search plot
    axes[1, 0].plot(beta_values_finer, inliers_finer, 'g.-', markersize=3, label='Inliers')
    axes[1, 0].axvline(x=best_beta_finer, color='b', linestyle='--', 
                      label=f'Best β={best_beta_finer} (inliers={max_inliers_finer:.4f})')
    axes[1, 0].set_title(f'Finer Search (range={best_beta_fine}±10)')
    axes[1, 0].set_xlabel('Beta')
    axes[1, 0].set_ylabel('Inlier Ratio')
    axes[1, 0].set_ylim(-0.1, 1.1)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    # Finest search plot
    axes[1, 1].plot(beta_values_finest, inliers_finest, 'm.-', markersize=5, label='Inliers')
    axes[1, 1].axvline(x=best_beta_finest, color='b', linestyle='--', 
                      label=f'Best β={best_beta_finest} (inliers={max_inliers_finest:.4f})')
    axes[1, 1].set_title(f'Finest Search (range={best_beta_finer}±1)')
    axes[1, 1].set_xlabel('Beta')
    axes[1, 1].set_ylabel('Inlier Ratio')
    axes[1, 1].set_ylim(-0.1, 1.1)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Create plots directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/inliers_vs_beta_refinement_cam{main_camera}-{secondary_camera}_ds{dataset_no}.png")

def plot_combined_results(beta_values_coarse, inliers_coarse, best_beta_coarse, max_inliers_coarse,
                          beta_values_fine, inliers_fine, best_beta_fine, max_inliers_fine,
                          beta_values_finer, inliers_finer, best_beta_finer, max_inliers_finer,
                          beta_values_finest, inliers_finest, best_beta_finest, max_inliers_finest,
                          main_camera, secondary_camera, dataset_no,
                          output_dir=""):
    """
    Plot combined results of beta search
    
    Parameters:
    -----------
    (same as plot_refinement_process)
    """
    figsize = (28, 15)
    
    # Combined visualization in one plot with different colors and transparency
    plt.figure(figsize=figsize)
    plt.plot(beta_values_coarse, inliers_coarse, 'b-', alpha=0.5, linewidth=1, label='Coarse')
    plt.plot(beta_values_fine, inliers_fine, 'r-', alpha=0.6, linewidth=1.5, label='Fine')
    plt.plot(beta_values_finer, inliers_finer, 'g-', alpha=0.7, linewidth=2, label='Finer')
    plt.plot(beta_values_finest, inliers_finest, 'm-', alpha=1.0, linewidth=2.5, label='Finest')

    # Mark the best beta for each refinement level
    plt.scatter(best_beta_coarse, max_inliers_coarse, c='blue', marker='*', s=200, 
               label=f'Best β (Coarse)={best_beta_coarse}')
    plt.scatter(best_beta_fine, max_inliers_fine, c='red', marker='*', s=200, 
               label=f'Best β (Fine)={best_beta_fine}')
    plt.scatter(best_beta_finer, max_inliers_finer, c='green', marker='*', s=200, 
               label=f'Best β (Finer)={best_beta_finer}')
    plt.scatter(best_beta_finest, max_inliers_finest, c='magenta', marker='*', s=300, 
               label=f'Best β (Finest)={best_beta_finest}')

    plt.title(f'Multi-level Beta Search Refinement (Cameras {main_camera}-{secondary_camera})', fontsize=18)
    plt.xlabel('Beta', fontsize=14)
    plt.ylabel('Inlier Ratio', fontsize=14)
    plt.ylim(-0.1, 1.1)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    # Create plots directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/inliers_vs_beta_combined_cam{main_camera}-{secondary_camera}_ds{dataset_no}.png")
    plt.close()

## PLOT FUNCTIONS FOR THE ACTUAL DRONE TRAJECTORY RECONSTRUCTION

def plot_triangulated_points(triangulated_points, main_camera, secondary_camera, output_dir="plots"):
    """Plot triangulated 3D points and save the figure."""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(triangulated_points[:, 0], triangulated_points[:, 1], triangulated_points[:, 2], c='r', marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"Triangulated Points for Cameras {main_camera}-{secondary_camera}")
    plt.savefig(f"{output_dir}/triangulated_points_{main_camera}_{secondary_camera}.png")

def plot_reprojection_analysis(
    splines_3d, original_points_2d, 
    R, t, camera_info, camera_id, output_dir="plots", title=None, initial=False
):
    """Plot reprojected 2D points for a single camera."""
    # Reproject points
    
    points_3d = []
    
    for spline in splines_3d:
        spline_x, spline_y, spline_z, tss = spline
        timestamps_linspace = tss
        for tt in timestamps_linspace:
            points_3d.append([spline_x(tt), spline_y(tt), spline_z(tt)])
    points_3d = np.array(points_3d)
    points_3d = points_3d.reshape(-1, 3)
    
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
    
    before_or_after = "_before_ba_" if initial else ""
    
    plt.savefig(f"{output_dir}/reprojection_analysis_camera_{camera_id}{before_or_after}.png")

def plot_3d_splines_from_functions(splines_3d, main_camera_id, secondary_camera_id, iteration_num, output_dir="plots", title=None):
    """
    Plot 3D splines given as a list of (spline_x, spline_y, spline_z, timestamps).

    Parameters
    ----------
    splines_3d : list of tuples
        Each tuple is (spline_x, spline_y, spline_z, timestamps), where spline_x/y/z are callable and timestamps is an array.
    main_camera_id : int or str
        Identifier for the main camera.
    secondary_camera_id : int or str
        Identifier for the secondary camera.
    output_dir : str
        Directory to save the plot.
    title : str, optional
        Custom plot title.
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    for spline_x, spline_y, spline_z, ts in splines_3d:
        ax.plot(
            spline_x(ts),
            spline_y(ts),
            spline_z(ts),
            marker='o',
            markersize=1
        )
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plot_title = title if title else f"3D Splines for Cameras {main_camera_id}-{secondary_camera_id}"
    ax.set_title(plot_title)
    plt.savefig(f"{output_dir}/3d_splines_it{iteration_num}_{main_camera_id}_{secondary_camera_id}.png")
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

def plot_spline_extension(tpns_to_add_to_3d, splines_3d, main_camera, new_camera, iteration_num, plot_output_dir="plots"):
    """
    Plots 3D splines and triangulated points for a given camera pair and saves the figure.
    Parameters
    ----------
    tpns_to_add_to_3d : np.ndarray
        Array of shape (N, 3) containing the 3D coordinates of triangulated points to be plotted in blue.
    splines_3d : list of tuples
        List where each element is a tuple (spline_x, spline_y, spline_z, ts), representing the spline functions
        for x, y, z coordinates and the parameter values to evaluate the splines.
    main_camera : str or int
        Identifier for the main camera in the pair.
    new_camera : str or int
        Identifier for the new camera in the pair.
    plot_output_dir : str, optional
        Directory where the output plot will be saved. Default is "plots".
    Saves

    """
    # Plot splines in red and triangulated_points in blue
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot triangulated_points in blue
    ax.scatter(
        tpns_to_add_to_3d[:, 0], 
        tpns_to_add_to_3d[:, 1], 
        tpns_to_add_to_3d[:, 2], 
        c='blue', s=1, label=f"Triangulated Points for pair main-new {main_camera}-{new_camera}"
    )

    # Plot splines in red
    for spline_x, spline_y, spline_z, ts in splines_3d:
        ax.plot(
            spline_x(ts), 
            spline_y(ts), 
            spline_z(ts), 
            c='red'
        )

    # Set plot labels and title
    ax.set_title("3D Points and Splines Visualization")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    
    plt.savefig(f"{plot_output_dir}/Extension_to_splines_it{iteration_num}_{main_camera}_{new_camera}.png")

