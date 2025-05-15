import os
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

def plot_refinement_process(beta_values_coarse, inliers_coarse, best_beta_coarse, max_inliers_coarse,
                            beta_values_fine, inliers_fine, best_beta_fine, max_inliers_fine,
                            beta_values_finer, inliers_finer, best_beta_finer, max_inliers_finer,
                            beta_values_finest, inliers_finest, best_beta_finest, max_inliers_finest,
                            main_camera, secondary_camera, dataset_no):
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
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/inliers_vs_beta_refinement_cam{main_camera}-{secondary_camera}_ds{dataset_no}.png", dpi=300)

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
    R, t, camera_info, camera_id, output_dir="plots", title=None
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
    plt.savefig(f"{output_dir}/reprojection_analysis_camera_{camera_id}.png")


def plot_3d_splines(triangulated_points, correspondences, main_camera_id, secondary_camera_id, output_dir="plots"):
    """Plot 3D splines based on the triangulated points and time-continuity."""
    splines_3d_points = []
    this_spline = []

    for i, current_corr in enumerate(correspondences[:-1]):
        if i >= len(triangulated_points):
            break
        next_corr = correspondences[i + 1]
        this_spline.append(triangulated_points[i])

        if next_corr[2] - current_corr[2] >= 5:
            splines_3d_points.append(this_spline)
            this_spline = []

    this_spline.append(triangulated_points[-1])
    splines_3d_points.append(this_spline)

    # Filter splines
    splines_3d_points = [spline for spline in splines_3d_points if spline and len(spline) >= 4]

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    for spline in splines_3d_points:
        spline = np.array(spline)
        ax.plot(spline[:, 0], spline[:, 1], spline[:, 2], marker='o', markersize=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"3D Splines for Cameras {main_camera_id}-{secondary_camera_id}")
    plt.savefig(f"{output_dir}/3d_splines_{main_camera_id}_{secondary_camera_id}.png", dpi=300)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

