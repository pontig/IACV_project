import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import os

DATASET_NO = 3
# Read the data
df = pd.read_csv(f"beta_results_dataset{DATASET_NO}.csv")

# Get unique cameras
cameras = sorted(list(set(df['main_camera'].tolist() + df['secondary_camera'].tolist())))
n_cameras = len(cameras)

# Create inlier ratio matrix (rows: secondary cameras, columns: main cameras)
inlier_matrix = np.zeros((n_cameras, n_cameras))

# Fill the matrix with inlier ratios (no symmetry assumption)
for _, row in df.iterrows():
    main_idx = cameras.index(row['main_camera'])
    sec_idx = cameras.index(row['secondary_camera'])
    
    # Only fill the actual pair (secondary as row, main as column)
    inlier_matrix[sec_idx, main_idx] = row['inlier_ratio']

# Set diagonal to 1 (camera with itself)
np.fill_diagonal(inlier_matrix, 1.0)

# Algorithm: Find camera processing order
def find_camera_processing_order(inlier_matrix, cameras):
    """
    Find the order cameras should be processed:
    1. Start with the camera pair with highest inlier ratio
    2. Add cameras by decreasing inlier ratio with already processed cameras
    """
    processed = set()
    processing_order = []
    used_pairs = set()
    
    # Find the pair with maximum inlier ratio
    max_ratio = 0
    best_pair = None
    
    for i in range(len(cameras)):
        for j in range(len(cameras)):
            if i != j and inlier_matrix[i, j] > 0 and inlier_matrix[i, j] > max_ratio:
                max_ratio = inlier_matrix[i, j]
                best_pair = (i, j)  # (secondary_idx, main_idx)
    
    # Add the best pair to processed cameras
    sec_idx, main_idx = best_pair
    processed.add(sec_idx)
    processed.add(main_idx)
    processing_order.extend([sec_idx, main_idx])
    used_pairs.add((sec_idx, main_idx))
    
    print(f"Starting with camera pair (secondary: {cameras[sec_idx]}, main: {cameras[main_idx]}) with ratio {max_ratio:.4f}")
    
    # Process remaining cameras
    while len(processed) < len(cameras):
        best_ratio = -1
        best_camera = None
        best_connection = None
        
        # Find unprocessed camera with highest inlier ratio to any processed camera
        for unprocessed in range(len(cameras)):
            if unprocessed in processed:
                continue
                
            for processed_cam in processed:
                # Check both directions since matrix is not symmetric
                ratio1 = inlier_matrix[unprocessed, processed_cam] if inlier_matrix[unprocessed, processed_cam] > 0 else -1
                ratio2 = inlier_matrix[processed_cam, unprocessed] if inlier_matrix[processed_cam, unprocessed] > 0 else -1
                
                if ratio1 > best_ratio:
                    best_ratio = ratio1
                    best_camera = unprocessed
                    best_connection = processed_cam
                
                if ratio2 > best_ratio:
                    best_ratio = ratio2
                    best_camera = unprocessed
                    best_connection = processed_cam
        
        if best_camera is not None:
            processed.add(best_camera)
            processing_order.append(best_camera)
            # Add both possible pair directions to used_pairs
            used_pairs.add((best_camera, best_connection))
            # used_pairs.add((best_connection, best_camera))
            print(f"Adding camera {cameras[best_camera]} (ratio {best_ratio:.4f} with camera {cameras[best_connection]})")
    
    return processing_order, used_pairs

# Find processing order
processing_order, used_pairs = find_camera_processing_order(inlier_matrix, cameras)

# Create output directory if it doesn't exist
os.makedirs("documentation/imgs", exist_ok=True)

# Plot 1: Complete inlier ratio heatmap
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

im = ax.imshow(inlier_matrix, cmap='viridis', aspect='auto')
ax.set_title('Camera Pair Inlier Ratios', fontsize=14, fontweight='bold')
ax.set_xlabel('Main Camera ID')
ax.set_ylabel('Secondary Camera ID')
ax.set_xticks(range(n_cameras))
ax.set_yticks(range(n_cameras))
ax.set_xticklabels(cameras)
ax.set_yticklabels(cameras)

# Add text annotations for inlier ratios
for i in range(n_cameras):
    for j in range(n_cameras):
        if i != j and inlier_matrix[i, j] > 0:  # Only show non-zero values
            text = f'{inlier_matrix[i, j]:.3f}'
            ax.text(j, i, text, ha='center', va='center', 
                    color='white' if inlier_matrix[i, j] < 0.9 else 'black', 
                    fontsize=8, fontweight='bold')

plt.colorbar(im, ax=ax, label='Inlier Ratio')
plt.tight_layout()
plt.savefig('documentation/imgs/camera_inlier_ratios.png',  bbox_inches='tight')
plt.close()

# Plot 2: Processing order heatmap
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

# Create a custom colormap for the processing visualization
colors = ['lightgray', 'darkgreen']  # ignored, used
cmap_custom = plt.cm.colors.ListedColormap(colors)

# Create binary matrix for visualization
binary_matrix = np.zeros_like(inlier_matrix)
for i in range(n_cameras):
    for j in range(n_cameras):
        if i == j:
            binary_matrix[i, j] = 0  # Diagonal
        elif (i, j) in used_pairs and inlier_matrix[i, j] > 0:
            binary_matrix[i, j] = 1  # Used pair
        else:
            binary_matrix[i, j] = 0  # Ignored pair

im = ax.imshow(binary_matrix, cmap=cmap_custom, aspect='auto', vmin=0, vmax=1)
ax.set_title('Camera Processing Order\n(Green: Used pairs, Gray: Ignored pairs)', 
              fontsize=14, fontweight='bold')
ax.set_xlabel('Main Camera ID')
ax.set_ylabel('Secondary Camera ID')
ax.set_xticks(range(n_cameras))
ax.set_yticks(range(n_cameras))
ax.set_xticklabels(cameras)
ax.set_yticklabels(cameras)

# Add text annotations showing inlier ratios with different styling
for i in range(n_cameras):
    for j in range(n_cameras):
        if i != j and inlier_matrix[i, j] > 0:  # Only show non-zero values
            if (i, j) in used_pairs:
                # Used pairs: bold white text
                text = f'{inlier_matrix[i, j]:.3f}'
                ax.text(j, i, text, ha='center', va='center', 
                        color='white', fontsize=9, fontweight='bold')
            else:
                # Ignored pairs: smaller gray text
                text = f'{inlier_matrix[i, j]:.3f}'
                ax.text(j, i, text, ha='center', va='center', 
                        color='darkgray', fontsize=7, style='italic')

plt.tight_layout()
plt.savefig('documentation/imgs/camera_processing_order.png',  bbox_inches='tight')
plt.close()

# Print processing summary
print(f"\nProcessing Summary:")
print(f"Processing order: {[cameras[i] for i in processing_order]}")
print(f"Total pairs used: {len([p for p in used_pairs if inlier_matrix[p[0], p[1]] > 0])}")
print(f"Total available pairs: {np.count_nonzero(inlier_matrix) - n_cameras}")  # Exclude diagonal

print("\nPlots saved to:")
print("- documentation/imgs/camera_inlier_ratios.png")
print("- documentation/imgs/camera_processing_order.png")