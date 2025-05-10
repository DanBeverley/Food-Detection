import cv2
import numpy as np
import os
import argparse
import yaml
import logging

logger = logging.getLogger(__name__)

def calibrate_camera(image_dir, corners_width, corners_height, square_size, output_file=None):
    """
    Calibrates a camera using images of a checkerboard pattern.

    This function processes a directory of images, detects checkerboard corners,
    and then uses these detections to compute the camera's intrinsic matrix
    (fx, fy, cx, cy) and distortion coefficients.

    The intrinsic parameters are essential for accurately converting 2D image points
    to 3D world points, which is fundamental for tasks like volume estimation
    from depth data that is aligned with RGB images.

    Args:
        image_dir (str): Path to the directory containing checkerboard images.
        corners_width (int): Number of inner corners along the width of the checkerboard.
                             (e.g., for a 10x7 board, this is 9).
        corners_height (int): Number of inner corners along the height of the checkerboard.
                              (e.g., for a 10x7 board, this is 6).
        square_size (float): The side length of a square on the checkerboard, in real-world units
                             (e.g., mm, cm, meters). Ensure consistency with desired output units.
                             If your depth data is in mm, using mm here is convenient.
        output_file (str, optional): Path to save the calibration results (camera matrix
                                     and distortion coefficients) in YAML format. 
                                     Defaults to None (only prints to console).

    Returns:
        tuple: A tuple containing:
            - ret (bool): True if calibration was successful, False otherwise.
            - camera_matrix (np.ndarray): The 3x3 camera intrinsic matrix.
            - dist_coeffs (np.ndarray): The distortion coefficients (k1, k2, p1, p2, k3, ...).
            - rvecs (list of np.ndarray): Rotation vectors for each calibration image.
            - tvecs (list of np.ndarray): Translation vectors for each calibration image.
        Returns (None, None, None, None, None) if calibration fails or no valid images are found.
    """
    # Termination criteria for corner refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(corners_width-1,corners_height-1,0)
    objp = np.zeros((corners_height * corners_width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:corners_width, 0:corners_height].T.reshape(-1, 2)
    objp = objp * square_size  # Scale by square size

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    
    images_processed = 0
    images_with_corners = 0

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    image_files = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir)
                   if fname.lower().endswith(valid_extensions)]

    if not image_files:
        logger.error(f"No images found in directory: {image_dir}")
        return None, None, None, None, None

    logger.info(f"Found {len(image_files)} potential images in {image_dir}.")

    gray_shape = None

    for fname in image_files:
        images_processed += 1
        img = cv2.imread(fname)
        if img is None:
            logger.warning(f"Failed to load image: {fname}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if gray_shape is None:
            gray_shape = gray.shape[::-1] # width, height
        elif gray_shape != gray.shape[::-1]:
            logger.warning(f"Image {fname} has different dimensions {gray.shape[::-1]} than first image {gray_shape}. Skipping.")
            continue

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (corners_width, corners_height), None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            images_with_corners += 1
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners (optional, for visualization)
            # cv2.drawChessboardCorners(img, (corners_width, corners_height), corners2, ret)
            # cv2.imshow('img', img)
            # cv2.waitKey(500)
            logger.debug(f"Checkerboard corners found in {fname}")
        else:
            logger.warning(f"Checkerboard corners NOT found in {fname}")

    # cv2.destroyAllWindows()

    if images_with_corners == 0:
        logger.error("No checkerboard corners found in any of the provided images. Calibration cannot proceed.")
        return None, None, None, None, None
    
    if gray_shape is None:
        logger.error("Could not determine image dimensions (no valid images loaded).")
        return None, None, None, None, None

    logger.info(f"Processed {images_processed} images. Found corners in {images_with_corners} images.")
    logger.info(f"Image dimensions for calibration (W x H): {gray_shape[0]} x {gray_shape[1]}")

    # Calibrate the camera
    # gray_shape must be (width, height) for cv2.calibrateCamera
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray_shape, None, None)

    if ret:
        logger.info("Camera calibration successful.")
        fx = camera_matrix[0,0]
        fy = camera_matrix[1,1]
        cx = camera_matrix[0,2]
        cy = camera_matrix[1,2]
        logger.info(f"  Image Dims (W, H): {gray_shape[0]}, {gray_shape[1]}")
        logger.info(f"  Focal Length (fx): {fx:.4f} pixels")
        logger.info(f"  Focal Length (fy): {fy:.4f} pixels")
        logger.info(f"  Principal Point (cx): {cx:.4f} pixels")
        logger.info(f"  Principal Point (cy): {cy:.4f} pixels")
        logger.info(f"Camera Matrix (K):\n{camera_matrix}")
        logger.info(f"Distortion Coefficients (k1, k2, p1, p2, k3, ...):\n{dist_coeffs.ravel()}")

        # Calculate reprojection error
        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error += error
        reprojection_error = mean_error / len(objpoints)
        logger.info(f"Total mean reprojection error: {reprojection_error:.4f} pixels")

        if output_file:
            calibration_data = {
                'image_width': gray_shape[0],
                'image_height': gray_shape[1],
                'camera_matrix': {
                    'fx': fx,
                    'fy': fy,
                    'cx': cx,
                    'cy': cy,
                    'data': camera_matrix.tolist() # For easy serialization
                },
                'distortion_coefficients': {
                    'k1': dist_coeffs.ravel()[0] if len(dist_coeffs.ravel()) > 0 else 0.0,
                    'k2': dist_coeffs.ravel()[1] if len(dist_coeffs.ravel()) > 1 else 0.0,
                    'p1': dist_coeffs.ravel()[2] if len(dist_coeffs.ravel()) > 2 else 0.0,
                    'p2': dist_coeffs.ravel()[3] if len(dist_coeffs.ravel()) > 3 else 0.0,
                    'k3': dist_coeffs.ravel()[4] if len(dist_coeffs.ravel()) > 4 else 0.0,
                    'data': dist_coeffs.ravel().tolist() # For easy serialization
                },
                'avg_reprojection_error_pixels': reprojection_error
            }
            try:
                with open(output_file, 'w') as f:
                    yaml.dump(calibration_data, f, indent=4, sort_keys=False)
                logger.info(f"Calibration results saved to: {output_file}")
            except Exception as e:
                logger.error(f"Error saving calibration results to {output_file}: {e}")
        return ret, camera_matrix, dist_coeffs, rvecs, tvecs
    else:
        logger.error("Camera calibration failed.")
        return False, None, None, None, None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calibrate camera using checkerboard images.")
    parser.add_argument('--image_dir', type=str, required=True, 
                        help='Directory containing checkerboard images.')
    parser.add_argument('--corners_width', type=int, required=True, 
                        help='Number of inner corners along the width of the checkerboard (e.g., 9 for a 10x7 board).')
    parser.add_argument('--corners_height', type=int, required=True, 
                        help='Number of inner corners along the height of the checkerboard (e.g., 6 for a 10x7 board).')
    parser.add_argument('--square_size', type=float, required=True, 
                        help='Side length of a checkerboard square in real-world units (e.g., mm or cm).')
    parser.add_argument('--output_file', type=str, default=None, 
                        help='Optional. Path to save calibration results (e.g., calibration_results.yaml).')
    parser.add_argument('--log_level', type=str, default='INFO', 
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Set the logging level.')

    args = parser.parse_args()

    # Basic logging setup
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

    calibrate_camera(args.image_dir, 
                     args.corners_width, 
                     args.corners_height, 
                     args.square_size, 
                     args.output_file)
