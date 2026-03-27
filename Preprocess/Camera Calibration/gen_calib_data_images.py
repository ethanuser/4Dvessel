# This code generates a json containing camera parameters for individual checkerboard pattern images

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob
import os
import json
from scipy.spatial.transform import Rotation

# File Paths
IMAGES_PATH = 'Preprocess/Camera Calibration/calibrations/'
# SCALING_FACTOR = 0.01567125 # real chessboard scale
# SCALING_FACTOR = 15.67125 # real chessboard x 1000
# SCALING_FACTOR = 0.1567125 # real chessboard x 10, best for real
SCALING_FACTOR = 0.667125 # real chessboard x 10, best for real
# SCALING_FACTOR = 1.567125 # for beaded synthetic vessel when scaled 0.02

# WIDTH_OF_BOX = 800 #px for 1080p
WIDTH_OF_BOX = 1200 #px for 2160p
USE_ORTHOGRAPHIC_VIEW = False
USE_BIRDS_EYE_VIEW = False

# Load configuration
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..\\config.json')
with open(config_path, 'r') as f:
    config = json.load(f)

# Chessboard dimensions
number_of_squares_X = 8  # Number of chessboard squares along the x-axis
number_of_squares_Y = 6  # Number of chessboard squares along the y-axis
nX = number_of_squares_X - 1  # Number of interior corners along x-axis
nY = number_of_squares_Y - 1  # Number   of interior corners along y-axis

# Define flip operations for each camera new checkerboards
#2160p
# flip_lr = [1, 1, 0, 1, 0, 0, 0, 1, 1]
# flip_up = [0, 0, 1, 0, 0, 1, 0, 0, 0]

flip_lr = [0, 0, 1, 1, 1, 1, 1, 0]
flip_up = [1, 1, 0, 1, 1, 0, 1, 0]

def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def visualize_camera_poses(extrinsics_list, objp, n):
    fig = plt.figure(figsize=(14, 7))

    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Original Camera Poses")

    colors = plt.cm.jet(np.linspace(0, 1, len(objp)))

    for i in range(len(objp)):
        ax.scatter(objp[i, 0], objp[i, 1], objp[i, 2], c=[colors[i]], marker='.')
        ax.text(objp[i, 0], objp[i, 1], objp[i, 2], f'{i+1}', color='red')

    for idx, (rvecs, tvecs) in enumerate(extrinsics_list):
        if idx % n == 0:
            R, _ = cv2.Rodrigues(rvecs)
            camera_position = -np.dot(R.T, tvecs).flatten()
            vector_length = 5 * SCALING_FACTOR

            ax.scatter(camera_position[0], camera_position[1], camera_position[2], c='b', marker='.')
            ax.text(camera_position[0], camera_position[1], camera_position[2], f'{idx+1}', color='blue')

            cam_axes_transformed = R
            cam_axes_transformed[1, :] = -cam_axes_transformed[1, :]
            cam_axes_transformed[2, :] = -cam_axes_transformed[2, :]
            ax.quiver(camera_position[0], camera_position[1], camera_position[2],
                       cam_axes_transformed[0, 0], cam_axes_transformed[0, 1], cam_axes_transformed[0, 2],
                       color='r', length=vector_length)
            ax.quiver(camera_position[0], camera_position[1], camera_position[2],
                       cam_axes_transformed[1, 0], cam_axes_transformed[1, 1], cam_axes_transformed[1, 2],
                       color='g', length=vector_length)
            ax.plot([camera_position[0], camera_position[0] + cam_axes_transformed[2, 0] * -5 * vector_length],
                [camera_position[1], camera_position[1] + cam_axes_transformed[2, 1] * -5 * vector_length],
                [camera_position[2], camera_position[2] + cam_axes_transformed[2, 2] * -5 * vector_length], color='b')


    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    set_axes_equal(ax)

    if USE_BIRDS_EYE_VIEW:
        ax.view_init(elev = 90, roll = 0, azim = 90)
    if USE_ORTHOGRAPHIC_VIEW:
        ax.set_proj_type('ortho')
    else:
        ax.set_proj_type('persp', focal_length=0.2)

    plt.show()
    
def apply_flip_operations(corners, lr, up):
    corners_reshaped = corners.reshape((nX, nY, 2))
    if lr:
        corners_reshaped = np.fliplr(corners_reshaped)
    if up:
        corners_reshaped = np.flipud(corners_reshaped)
    return corners_reshaped.reshape((-1, 1, 2))

def save_camera_parameters(camera_params):
    with open(f'{IMAGES_PATH}camera_parameters.json', 'w') as json_file:
        json.dump(camera_params, json_file, indent=4)
    print("Camera parameters saved to camera_parameters.json")

def find_bounding_boxes():
    bounds = []
    image_files = sorted([f for f in os.listdir(IMAGES_PATH) if f.endswith('.png')])

    if len(image_files) == 0:
        print(f"Error: No images found in {IMAGES_PATH}")
        return bounds

    first_image = cv2.imread(os.path.join(IMAGES_PATH, image_files[0]))
    if first_image is None:
        print(f"Error: Could not read the first image file")
        return bounds

    frame_height, frame_width = first_image.shape[:2]

    for idx, filename in enumerate(image_files):
        if idx < len(image_files):
                image_path = os.path.join(IMAGES_PATH, image_files[idx])
                view = cv2.imread(image_path)

                if view is None:
                    print(f"Error: Could not read image file {image_path}")
                    continue

                gray = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
                success, corners = cv2.findChessboardCorners(gray, (nY, nX), flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_EXHAUSTIVE)
                if success:
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    corners2 = apply_flip_operations(corners2, flip_lr[idx], flip_up[idx])
                    center = corners2[int(0.5 * nX * nY)].ravel() + 1
                    center = (center[0], center[1])
                    x_top = int(max(center[0] - round(0.5 * WIDTH_OF_BOX), 0))
                    y_top = int(max(center[1] - round(0.5 * WIDTH_OF_BOX), 0))
                    x_bot = int(min(center[0] + round(0.5 * WIDTH_OF_BOX), gray.shape[1]))
                    y_bot = int(min(center[1] + round(0.5 * WIDTH_OF_BOX), gray.shape[0]))

                    bounding_box = (x_top, y_top, x_bot, y_bot)
                    bounds.append(bounding_box)
                else:
                    bounds.append((0, 0, frame_width, frame_height))
    return bounds

def main():
    image_files = sorted(glob.glob(f'{IMAGES_PATH}*.png'))
    extrinsics_list = []

    # Prepare object points (checkerboard corners in 3D space)
    objp = np.zeros((nX * nY, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nY, 0:nX].T.reshape(-1, 2) * SCALING_FACTOR

    # Create output directory for labeled checkerboards if it doesn't exist
    output_dir = os.path.join(IMAGES_PATH, 'labeled_checkerboards')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    bounds = find_bounding_boxes()
    print(bounds)
    
    objpoints = []
    imgpoints = []

    # Process each image file
    for idx, filename in enumerate(image_files):
        try:
            print(f'Working on file: {filename}.')
            image = cv2.imread(filename)
            if image is None:
                print(f'Error: Could not read image file {filename}')
                continue

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            success, corners = cv2.findChessboardCorners(gray, (nY, nX), flags=cv2.CALIB_CB_EXHAUSTIVE)

            if success:
                # Refine corner locations
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

                # Apply flip operations (if any)
                corners2 = apply_flip_operations(corners2, flip_lr[idx], flip_up[idx])

                # Annotate the checkerboard corners in the image
                for i in range(corners2.shape[0]):
                    cv2.putText(image, str(i+1), (int(corners2[i, 0, 0]), int(corners2[i, 0, 1])),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                # Draw and save the checkerboard corners on the image
                cv2.drawChessboardCorners(image, (nY, nX), corners2, success)
                new_filename = os.path.join(output_dir, os.path.basename(filename).replace('.png', '_drawn_corners.png'))
                print(f'Saving file name = {new_filename}')
                cv2.imwrite(new_filename, image)

                objpoints.append(objp)
                imgpoints.append(corners2)
            else:
                print(f'Failed to process image: {filename}')
        except Exception as e:
            print(f'Failed to process image: {filename}. Error: {e}')

    if objpoints and imgpoints:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
        flags = cv2.CALIB_TILTED_MODEL + cv2.CALIB_RATIONAL_MODEL
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None, flags=flags,criteria=criteria)
        center = np.mean(objp, axis=0)
        print(f"Center: {center}")
        
        extrinsics_list = list(zip(rvecs, tvecs))

        def validate_calibration(objpoints, imgpoints, camera_matrix, dist_coeffs, rvecs, tvecs):
            mean_error = 0
            for i in range(len(objpoints)):
                imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
                error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
                mean_error += error
                print(f"Image {i+1} error: {error}")
            print(f"Mean reprojection error: {mean_error/len(objpoints)}")

        validate_calibration(objpoints, imgpoints, camera_matrix, dist_coeffs, rvecs, tvecs)

        visualize_camera_poses(extrinsics_list, objp, n=1)

        camera_params = {
            "camera_matrix": camera_matrix.tolist(),
            "distortion_coefficients": dist_coeffs.tolist()
        }

        for idx, (rvec, tvec) in enumerate(extrinsics_list):
            R0, _ = cv2.Rodrigues(rvec)
            R0 = R0.T

            transform_matrix = np.eye(4)
            T = -np.dot(R0, tvec.flatten())

            # Center all translation vectors so that cameras n  point toward the origin
            T += [-center[0], -center[1], center[2]]
            direction = -T
            direction /= np.linalg.norm(direction)
            
            # New z-axis points towards the center
            new_z = direction
            
            # Project the original x-axis onto the plane perpendicular to new_z
            orig_x = R0[:, 0]
            proj_x = orig_x - np.dot(orig_x, new_z) * new_z
            proj_x /= np.linalg.norm(proj_x)
            
            # Calculate new y-axis to complete the orthonormal basis
            new_y = np.cross(new_z, proj_x)
            
            # Construct the new rotation matrix
            R = np.vstack((proj_x, new_y, new_z))
            
            T[0] = -T[0]
            T[1] = -T[1]
            transform_matrix[:3, 3] = T

            R[1, :] = -R[1, :]
            R[2, :] = -R[2, :]

            R[:, 0] = -R[:, 0]
            R[:, 1] = -R[:, 1]

            transform_matrix[:3, :3] = R.T

            camera_params[f"Camera {idx + 1}"] = {
                "top_left": bounds[idx][:2],
                "bottom_right": bounds[idx][2:],
                "transform_matrix": transform_matrix.tolist()
            }

        save_camera_parameters(camera_params)
    else:
        print('No checkerboards were found in the images!')


if __name__ == "__main__":
    main()