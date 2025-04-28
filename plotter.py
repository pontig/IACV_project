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
    ax.set_title('Triangulated Points')
    plt.savefig(f"{output_dir}/triangulated_points_{main_camera}_{secondary_camera}.png", dpi=300)


def plot_reprojection_analysis(
    triangulated_points, correspondences, 
    R, t, main_camera_info, secondary_camera_info, 
    main_camera_id, secondary_camera_id, output_dir="plots"
):
    """Plot reprojected 2D points for both main and secondary cameras."""
    # Reproject onto main camera
    main_reprojected_points, _ = cv.projectPoints(
        triangulated_points, np.zeros((3, 1)), np.zeros((3, 1)),
        main_camera_info.K_matrix, main_camera_info.distCoeff
    )
    main_reprojected_points = main_reprojected_points.reshape(-1, 2)

    # Reproject onto secondary camera
    secondary_reprojected_points, _ = cv.projectPoints(
        triangulated_points, R, t,
        secondary_camera_info.K_matrix, secondary_camera_info.distCoeff
    )
    secondary_reprojected_points = secondary_reprojected_points.reshape(-1, 2)

    fig, axes = plt.subplots(2, 1, figsize=(10, 20))
    fig.suptitle(f"Reprojection Analysis for Cameras {main_camera_id} and {secondary_camera_id}", fontsize=20)

    # Main camera
    axes[0].scatter(main_reprojected_points[:, 0], -main_reprojected_points[:, 1], c='r', label='Reprojected Points', s=1)
    axes[0].scatter(
        [x[0] for x, _, _ in correspondences], 
        [-x[1] for x, _, _ in correspondences], 
        c='b', label='Original Points', alpha=0.5, s=1
    )
    axes[0].set_title(f"Main Camera {main_camera_id}")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")
    axes[0].legend()
    axes[0].set_xlim(0, main_camera_info.resolution[0])
    axes[0].set_ylim(-main_camera_info.resolution[1], 0)
    axes[0].grid(True, alpha=0.3)

    # Secondary camera
    axes[1].scatter(secondary_reprojected_points[:, 0], -secondary_reprojected_points[:, 1], c='r', label='Reprojected Points', s=1)
    axes[1].scatter(
        [y[0] for _, y, _ in correspondences], 
        [-y[1] for _, y, _ in correspondences], 
        c='b', label='Original Points', alpha=0.5, s=1
    )
    axes[1].set_title(f"Secondary Camera {secondary_camera_id}")
    axes[1].set_xlabel("X")
    axes[1].set_ylabel("Y")
    axes[1].legend()
    axes[1].set_xlim(0, secondary_camera_info.resolution[0])
    axes[1].set_ylim(-secondary_camera_info.resolution[1], 0)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(f"{output_dir}/reprojection_analysis_cameras_{main_camera_id}_{secondary_camera_id}.png", dpi=300)


def plot_3d_splines(triangulated_points, correspondences, main_camera_id, secondary_camera_id, output_dir="plots"):
    """Plot 3D splines based on the triangulated points and time-continuity."""
    splines_3d_points = []
    this_spline = []

    for i, current_corr in enumerate(correspondences[:-1]):
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
    ax.set_title('3D Splines of Triangulated Points')
    plt.savefig(f"{output_dir}/3d_splines_{main_camera_id}_{secondary_camera_id}.png", dpi=300)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

