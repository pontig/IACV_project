import cv2
import numpy as np

def rectify_image(image_path, camera_matrix, dist_coeffs):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load image.")
        return

    # Get the image size
    h, w = image.shape[:2]

    # Compute the optimal new camera matrix
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 0, (w, h))

    # Undistort the image
    rectified_image = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)

    # Crop the image based on the ROI
    x, y, w, h = roi
    rectified_image = rectified_image[y:y+h, x:x+w]
    
    # Scale the image by half
    # rectified_image = cv2.resize(rectified_image, (w // 2, h // 2))

    # Show the rectified image
    cv2.imshow('Rectified Image', rectified_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    # Example camera matrix and distortion coefficients
    camera_matrix = np.array([[874.4721846047786, 0.0, 970.2688358898922], [0.0, 894.1080937815644, 531.2757796052425], [0.0, 0.0, 1.0]], dtype=np.float32)
    dist_coeffs = np.array([-0.260720634999793, 0.07494782427852716, -0.00013631462898833923, 0.00017484761775924765, -0.00906247784302948], dtype=np.float32)

    # Path to the input image
    image_path = 'drone-tracking-datasets/dataset4/cam0.jpg'

    rectify_image(image_path, camera_matrix, dist_coeffs)