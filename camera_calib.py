import cv2
import numpy as np
import json
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

output_dir = "plots/intrinsic_calibration"
Path(output_dir).mkdir(parents=True, exist_ok=True)

class CameraCalibrator:
    def __init__(self, checkerboard_size=(9, 6), square_size=1.0, camera_name="camera"):
        """
        Initialize camera calibrator
        
        Args:
            checkerboard_size: (width, height) - number of inner corners
            square_size: size of checkerboard squares in real world units
            camera_name: name of the camera for identification
        """
        self.checkerboard_size = checkerboard_size
        self.square_size = square_size
        self.camera_name = camera_name
        
        # Prepare object points (3D points in real world space)
        self.objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
        self.objp *= square_size
        
        # Arrays to store object points and image points
        self.objpoints = []  # 3d points in real world space
        self.imgpoints = []  # 2d points in image plane
        
    def find_checkerboard_corners(self, image_folder):
        """
        Find checkerboard corners in all images in the folder
        
        Args:
            image_folder: path to folder containing checkerboard images
        """
        # Supported image extensions
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
        
        image_files = []
        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(image_folder, ext)))
            image_files.extend(glob.glob(os.path.join(image_folder, ext.upper())))
        
        if not image_files:
            raise ValueError(f"No image files found in {image_folder}")
        
        print(f"Found {len(image_files)} images")
        successful_detections = 0
        first_detection_written = False
        
        for fname in image_files:
            img = cv2.imread(fname)
            if img is None:
                print(f"Warning: Could not read image {fname}")
                continue
                
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, self.checkerboard_size, None)
            
            # If found, add object points, image points
            if ret:
                self.objpoints.append(self.objp)
                
                # Refine corner positions
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                self.imgpoints.append(corners2)
                successful_detections += 1
                print(f"✓ Detected corners in {os.path.basename(fname)}")

                # Save the first detected image with corners overlaid
                if not first_detection_written:
                    img_drawn = cv2.drawChessboardCorners(img.copy(), self.checkerboard_size, corners2, ret)
                    output_path = os.path.join(output_dir, f"{self.camera_name}_first_detection.png")
                    cv2.imwrite(output_path, img_drawn)
                    print(f"✓ First detection with overlay saved to {output_path}")
                    first_detection_written = True
            else:
                print(f"✗ Could not detect corners in {os.path.basename(fname)}")
        
        print(f"\nSuccessfully detected corners in {successful_detections}/{len(image_files)} images")
        
        if successful_detections < 3:
            raise ValueError("Need at least 3 successful corner detections for calibration")
        
        return gray.shape[::-1]  # Return image size (width, height)
    
    def calibrate_camera(self, image_size):
        """
        Perform camera calibration
        
        Args:
            image_size: (width, height) of images
            
        Returns:
            calibration results dictionary
        """
        print("\nPerforming camera calibration...")
        
        # Calibrate camera
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, image_size, None, None
        )
        
        if not ret:
            raise RuntimeError("Camera calibration failed")
        
        # Calculate reprojection error
        total_error = 0
        for i in range(len(self.objpoints)):
            imgpoints2, _ = cv2.projectPoints(self.objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(self.imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            total_error += error
        
        mean_error = total_error / len(self.objpoints)
        
        # Extract camera parameters
        fx, fy = mtx[0, 0], mtx[1, 1]
        cx, cy = mtx[0, 2], mtx[1, 2]
        
        results = {
            'K_matrix': mtx.tolist(),
            'distortion_coefficients': dist.flatten().tolist(),
            'reprojection_error': mean_error,
            'num_images_used': len(self.objpoints),
            'focal_length': {'fx': fx, 'fy': fy},
            'principal_point': {'cx': cx, 'cy': cy},
            'resolution': list(image_size)
        }
        
        print(f"✓ Calibration successful!")
        print(f"  Mean reprojection error: {mean_error:.4f} pixels")
        print(f"  Images used: {len(self.objpoints)}")
        
        return results
    
    def load_reference_calibration(self, json_file):
        """
        Load reference calibration from JSON file
        
        Args:
            json_file: path to JSON file with reference calibration
            
        Returns:
            reference calibration dictionary
        """
        with open(json_file, 'r') as f:
            ref_data = json.load(f)
        
        return {
            'K_matrix': ref_data['K-matrix'],
            'distortion_coefficients': ref_data['distCoeff'],
            'resolution': ref_data['resolution'],
            'fps': ref_data.get('fps', None)
        }
    
    def compare_calibrations(self, computed, reference):
        """
        Compare computed calibration with reference
        
        Args:
            computed: computed calibration results
            reference: reference calibration from JSON
            
        Returns:
            comparison results dictionary
        """
        print("\n" + "="*50)
        print("CALIBRATION COMPARISON")
        print("="*50)
        
        # Convert to numpy arrays for easier computation
        K_computed = np.array(computed['K_matrix'])
        K_reference = np.array(reference['K_matrix'])
        dist_computed = np.array(computed['distortion_coefficients'])
        dist_reference = np.array(reference['distortion_coefficients'])
        
        # Extract parameters
        fx_comp, fy_comp = K_computed[0, 0], K_computed[1, 1]
        cx_comp, cy_comp = K_computed[0, 2], K_computed[1, 2]
        
        fx_ref, fy_ref = K_reference[0, 0], K_reference[1, 1]
        cx_ref, cy_ref = K_reference[0, 2], K_reference[1, 2]
        
        # Calculate differences
        fx_diff = abs(fx_comp - fx_ref)
        fy_diff = abs(fy_comp - fy_ref)
        cx_diff = abs(cx_comp - cx_ref)
        cy_diff = abs(cy_comp - cy_ref)
        
        fx_pct = (fx_diff / fx_ref) * 100
        fy_pct = (fy_diff / fy_ref) * 100
        cx_pct = (cx_diff / cx_ref) * 100
        cy_pct = (cy_diff / cy_ref) * 100
        
        # Distortion coefficient differences
        min_len = min(len(dist_computed), len(dist_reference))
        dist_diffs = np.abs(dist_computed[:min_len] - dist_reference[:min_len])
        
        print(f"INTRINSIC PARAMETERS:")
        print(f"  Focal Length X:")
        print(f"    Computed: {fx_comp:.3f}")
        print(f"    Reference: {fx_ref:.3f}")
        print(f"    Difference: {fx_diff:.3f} ({fx_pct:.2f}%)")
        
        print(f"  Focal Length Y:")
        print(f"    Computed: {fy_comp:.3f}")
        print(f"    Reference: {fy_ref:.3f}")
        print(f"    Difference: {fy_diff:.3f} ({fy_pct:.2f}%)")
        
        print(f"  Principal Point X:")
        print(f"    Computed: {cx_comp:.3f}")
        print(f"    Reference: {cx_ref:.3f}")
        print(f"    Difference: {cx_diff:.3f} ({cx_pct:.2f}%)")
        
        print(f"  Principal Point Y:")
        print(f"    Computed: {cy_comp:.3f}")
        print(f"    Reference: {cy_ref:.3f}")
        print(f"    Difference: {cy_diff:.3f} ({cy_pct:.2f}%)")
        
        print(f"\nDISTORTION COEFFICIENTS:")
        coeff_names = ['k1', 'k2', 'p1', 'p2', 'k3']
        for i in range(min_len):
            name = coeff_names[i] if i < len(coeff_names) else f'coeff_{i}'
            print(f"  {name}:")
            print(f"    Computed: {dist_computed[i]:.6f}")
            print(f"    Reference: {dist_reference[i]:.6f}")
            print(f"    Difference: {dist_diffs[i]:.6f}")
        
        print(f"\nRESOLUTION:")
        print(f"  Computed: {computed['resolution']}")
        print(f"  Reference: {reference['resolution']}")
        
        # Summary statistics
        comparison_results = {
            'focal_length_errors': {'fx': fx_diff, 'fy': fy_diff, 'fx_pct': fx_pct, 'fy_pct': fy_pct},
            'principal_point_errors': {'cx': cx_diff, 'cy': cy_diff, 'cx_pct': cx_pct, 'cy_pct': cy_pct},
            'distortion_errors': dist_diffs.tolist(),
            'reprojection_error': computed['reprojection_error'],
            'K_matrix_diff': np.abs(K_computed - K_reference).tolist(),
            'computed': computed,
            'reference': reference
        }
        
        return comparison_results
    
    def plot_comparison(self, comparison_results):
        """
        Create visualization plots for comparison
        
        Args:
            comparison_results: results from compare_calibrations
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Camera Calibration Comparison', fontsize=16, fontweight='bold')
        
        # 1. Intrinsic Parameters Comparison
        ax1 = axes[0, 0]
        params = ['fx', 'fy', 'cx', 'cy']
        computed_vals = [
            comparison_results['computed']['focal_length']['fx'],
            comparison_results['computed']['focal_length']['fy'],
            comparison_results['computed']['principal_point']['cx'],
            comparison_results['computed']['principal_point']['cy']
        ]
        reference_vals = [
            comparison_results['reference']['K_matrix'][0][0],
            comparison_results['reference']['K_matrix'][1][1],
            comparison_results['reference']['K_matrix'][0][2],
            comparison_results['reference']['K_matrix'][1][2]
        ]
        
        x = np.arange(len(params))
        width = 0.35
        
        ax1.bar(x - width/2, computed_vals, width, label='Computed', alpha=0.8)
        ax1.bar(x + width/2, reference_vals, width, label='Reference', alpha=0.8)
        ax1.set_xlabel('Parameters')
        ax1.set_ylabel('Values (px)')
        ax1.set_title('Intrinsic Parameters Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(params)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Percentage Errors
        ax2 = axes[0, 1]
        errors = [
            comparison_results['focal_length_errors']['fx_pct'],
            comparison_results['focal_length_errors']['fy_pct'],
            comparison_results['principal_point_errors']['cx_pct'],
            comparison_results['principal_point_errors']['cy_pct']
        ]
        
        bars = ax2.bar(params, errors, color=['red' if e > 5 else 'orange' if e > 1 else 'green' for e in errors])
        ax2.set_xlabel('Parameters')
        ax2.set_ylabel('Percentage Error (%)')
        ax2.set_title('Percentage Errors in Intrinsic Parameters')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, error in zip(bars, errors):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{error:.2f}%', ha='center', va='bottom')
        
        # 3. Distortion Coefficients Comparison
        ax3 = axes[1, 0]
        dist_computed = comparison_results['computed']['distortion_coefficients']
        dist_reference = comparison_results['reference']['distortion_coefficients']
        min_len = min(len(dist_computed), len(dist_reference))
        
        coeff_names = ['k1', 'k2', 'p1', 'p2', 'k3'][:min_len]
        x = np.arange(len(coeff_names))
        
        ax3.bar(x - width/2, dist_computed[:min_len], width, label='Computed', alpha=0.8)
        ax3.bar(x + width/2, dist_reference[:min_len], width, label='Reference', alpha=0.8)
        ax3.set_xlabel('Distortion Coefficients')
        ax3.set_ylabel('Values')
        ax3.set_title('Distortion Coefficients Comparison')
        ax3.set_xticks(x)
        ax3.set_xticklabels(coeff_names)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. K-matrix Heatmap Difference
        ax4 = axes[1, 1]
        K_diff = np.array(comparison_results['K_matrix_diff'])
        im = ax4.imshow(K_diff, cmap='Reds', aspect='equal')
        ax4.set_title('K-matrix Absolute Differences (px)')
        
        # Add text annotations
        for i in range(3):
            for j in range(3):
                text = ax4.text(j, i, f'{K_diff[i, j]:.2f}',
                               ha="center", va="center", color="black", fontweight='bold')
        
        ax4.set_xticks([0, 1, 2])
        ax4.set_yticks([0, 1, 2])
        ax4.set_xticklabels(['col 0', 'col 1', 'col 2'])
        ax4.set_yticklabels(['row 0', 'row 1', 'row 2'])
        
        plt.colorbar(im, ax=ax4, shrink=0.6)
        plt.tight_layout()
        
        plt.savefig(f"{output_dir}/{self.camera_name}_calibration_summary.png")
        print(f"✓ Summary statistics plot saved to {output_dir}/{self.camera_name}_calibration_summary.png")
        plt.close(fig)
        
        # Additional summary plot
        # self._plot_summary_statistics(comparison_results)
    
    def _plot_summary_statistics(self, comparison_results):
        """
        Create a summary statistics plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Summary table
        ax1.axis('tight')
        ax1.axis('off')
        
        summary_data = [
            ['Metric', 'Value'],
            ['Reprojection Error', f"{comparison_results['reprojection_error']:.4f} pixels"],
            ['Images Used', f"{comparison_results['computed']['num_images_used']}"],
            ['Focal Length Error (fx)', f"{comparison_results['focal_length_errors']['fx']:.3f} ({comparison_results['focal_length_errors']['fx_pct']:.2f}%)"],
            ['Focal Length Error (fy)', f"{comparison_results['focal_length_errors']['fy']:.3f} ({comparison_results['focal_length_errors']['fy_pct']:.2f}%)"],
            ['Principal Point Error (cx)', f"{comparison_results['principal_point_errors']['cx']:.3f} ({comparison_results['principal_point_errors']['cx_pct']:.2f}%)"],
            ['Principal Point Error (cy)', f"{comparison_results['principal_point_errors']['cy']:.3f} ({comparison_results['principal_point_errors']['cy_pct']:.2f}%)"],
            ['Max Distortion Error', f"{max(comparison_results['distortion_errors']):.6f}"]
        ]
        
        table = ax1.table(cellText=summary_data, cellLoc='left', loc='center', colWidths=[0.4, 0.6])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2)
        
        # Style the header
        for i in range(2):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax1.set_title('Calibration Summary', fontsize=14, fontweight='bold', pad=20)
        
        # Error distribution
        all_errors = (
            list(comparison_results['focal_length_errors'].values())[:2] +  # fx, fy only
            list(comparison_results['principal_point_errors'].values())[:2] +  # cx, cy only
            comparison_results['distortion_errors']
        )
        
        ax2.hist(all_errors, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.set_xlabel('Error Magnitude')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Error Distribution')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()



def main():
    """
    Main function to run camera calibration and comparison for multiple cameras
    """
    # List of (camera_name, checkerboard_size)
    camera_configs = [
        ("gopro3", (7, 6)),
        ("iphone6", (5, 9)),
        ("mate7", (7, 6)),
        ("mate10", (7, 6)),
        ("mi9", (7, 6)),
        ("p20pro", (7, 6)),
        ("sony5n_1440x1080", (7, 6)),
        ("sony5n_1920x1080", (7, 6)),
        ("sony5100", (7, 6)),
        ("sonyG", (7, 6)),        
    ]
    SQUARE_SIZE = 1.0  # Size of checkerboard squares in your units

    for camera_name, CHECKERBOARD_SIZE in camera_configs:
        print(f"\n=== Processing camera: {camera_name} ===")
        IMAGE_FOLDER = f"drone-tracking-datasets/calibration/{camera_name}/calibration_images"
        REFERENCE_JSON = f"drone-tracking-datasets/calibration/{camera_name}/{camera_name}.json"
        try:
            # Initialize calibrator
            calibrator = CameraCalibrator(CHECKERBOARD_SIZE, SQUARE_SIZE, camera_name)

            # Check if paths exist
            if not os.path.exists(IMAGE_FOLDER):
                print(f"Error: Image folder '{IMAGE_FOLDER}' not found.")
                print("Please create the folder and add checkerboard images.")
                continue

            if not os.path.exists(REFERENCE_JSON):
                print(f"Error: Reference JSON file '{REFERENCE_JSON}' not found.")
                print("Please provide the reference calibration file.")
                continue

            # Find checkerboard corners
            image_size = calibrator.find_checkerboard_corners(IMAGE_FOLDER)

            # Perform calibration
            computed_calibration = calibrator.calibrate_camera(image_size)

            # Load reference calibration
            reference_calibration = calibrator.load_reference_calibration(REFERENCE_JSON)

            # Compare calibrations
            comparison_results = calibrator.compare_calibrations(computed_calibration, reference_calibration)

            # Create visualizations
            calibrator.plot_comparison(comparison_results)

            # Save results to JSON
            output_file = f"{output_dir}/calibration_comparison_results_{camera_name}.json"
            with open(output_file, 'w') as f:
                json.dump(comparison_results, f, indent=2)
            print(f"\n✓ Results saved to {output_file}")

        except Exception as e:
            print(f"Error processing camera '{camera_name}': {e}")
            continue

if __name__ == "__main__":
    main()