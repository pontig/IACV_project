
# Essential matrix
E = camera_info[secondary_camera].K_matrix.T @ F @ camera_info[main_camera].K_matrix
print(f"Estimated essential matrix:\n {E/E[2, 2]}")


_, E, R, t, mask = cv.recoverPose(
    np.array([x for x, _, _, _ in correspondences]),
    np.array([y for _, y, _, _ in correspondences]),
    camera_info[main_camera].K_matrix, camera_info[main_camera].distCoeff,
    camera_info[secondary_camera].K_matrix, camera_info[secondary_camera].distCoeff,
    cv.RANSAC, 0.999, 2
)
print(E/E[2, 2])
print(np.sum(mask))
P1 = np.dot(camera_info[main_camera].K_matrix, np.hstack((np.eye(3), np.zeros((3, 1)))))
P2 = np.dot(camera_info[secondary_camera].K_matrix, np.hstack((R, t)))
# pts_camera_coord_1 = to_normalized_camera_coord(np.array([x for x, _, _, _ in correspondences]), camera_info[main_camera].K_matrix, camera_info[main_camera].distCoeff, np.eye(3), np.zeros((3,1)))
# pts_camera_coord_2 = to_normalized_camera_coord(np.array([y for _, y, _, _ in correspondences]), camera_info[secondary_camera].K_matrix, camera_info[secondary_camera].distCoeff, R, t)
pts_camera_coord_1 = np.array([x for x, _, _, _ in correspondences])
pts_camera_coord_2 = np.array([y for _, y, _, _ in correspondences])

fig, ax = plt.subplots(2, 2, figsize=(15, 7))

# Plot points in camera coordinates for main camera
ax[0, 0].scatter(pts_camera_coord_1[:, 0], -pts_camera_coord_1[:, 1], c='r', marker='o', s=1)
ax[0, 0].set_title('Main Camera Coordinates')
ax[0, 0].set_xlabel('X')
ax[0, 0].set_ylabel('Y')
ax[0, 0].axis('equal')

# Plot points in camera coordinates for secondary camera
ax[0, 1].scatter(pts_camera_coord_2[:, 0], -pts_camera_coord_2[:, 1], c='b', marker='o', s=1)
ax[0, 1].set_title('Secondary Camera Coordinates')
ax[0, 1].set_xlabel('X')
ax[0, 1].set_ylabel('Y')
ax[0, 1].axis('equal')

# Triangulate points
pts_3d = cv.triangulatePoints(P1, P2, pts_camera_coord_1.T, pts_camera_coord_2.T)
pts_3d /= pts_3d[3]

# Re-project points onto the image planes
# Convert rotation matrices to rotation vectors using Rodrigues
rvec_main, _ = cv.Rodrigues(np.eye(3))
rvec_secondary, _ = cv.Rodrigues(R)

# Project 3D points to 2D image plane
pts_2d_main = cv.projectPoints(pts_3d.T[:, :3], rvec_main, np.zeros(3), camera_info[main_camera].K_matrix, camera_info[main_camera].distCoeff)[0]
pts_2d_secondary = cv.projectPoints(pts_3d.T[:, :3], rvec_secondary, t, camera_info[secondary_camera].K_matrix, camera_info[secondary_camera].distCoeff)[0]
pts_2d_main = pts_2d_main.reshape(-1, 2)
pts_2d_secondary = pts_2d_secondary.reshape(-1, 2)

# Calculate reprojection errors
reprojection_error_main = np.linalg.norm(pts_camera_coord_1 - pts_2d_main, axis=1)
reprojection_error_secondary = np.linalg.norm(pts_camera_coord_2 - pts_2d_secondary, axis=1)

# Plot re-projected points for main camera
scatter_main = ax[1, 0].scatter(pts_2d_main[:, 0], -pts_2d_main[:, 1], c=reprojection_error_main, cmap='viridis', marker='o', s=1)
ax[1, 0].set_title('Main Camera Reprojections')
ax[1, 0].set_xlabel('X')
ax[1, 0].set_ylabel('Y')
ax[1, 0].axis('equal')
fig.colorbar(scatter_main, ax=ax[1, 0], label='Reprojection Error')

# Plot re-projected points for secondary camera
scatter_secondary = ax[1, 1].scatter(pts_2d_secondary[:, 0], -pts_2d_secondary[:, 1], c=reprojection_error_secondary, cmap='viridis', marker='o', s=1)
ax[1, 1].set_title('Secondary Camera Reprojections')
ax[1, 1].set_xlabel('X')
ax[1, 1].set_ylabel('Y')
ax[1, 1].axis('equal')
fig.colorbar(scatter_secondary, ax=ax[1, 1], label='Reprojection Error')
fig.tight_layout()

fig.savefig('plots/reprojection_with_error.png')

# Plot 3D points
fig_3d = plt.figure(figsize=(10, 10))
ax_3d = fig_3d.add_subplot(111, projection='3d')
ax_3d.scatter(pts_3d[0], pts_3d[1], pts_3d[2], c='g', marker='o', s=1)
ax_3d.set_title('3D Points')
ax_3d.set_xlabel('X')
ax_3d.set_ylabel('Y')
ax_3d.set_zlabel('Z')

# plt.show()