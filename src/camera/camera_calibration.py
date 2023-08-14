# Copyright (c) 2023 tc-haung
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import os
import numpy as np
import cv2 as cv
import glob
from typing import List, Tuple
import matplotlib.pyplot as plt

CHESSBOARD_ROW = 8
CHESSBOARD_COLUMN = 11
CHESSBOARD_GRID_SIZE = 20  # Unit: mm
MINIMUM_IMAGE_NUM = 20


def create_chessboard_corners_coordinate():
    # prepare object points, like
    # array([[0., 0., 0.],
    # [1., 0., 0.],
    # [2., 0., 0.],
    # [0., 1., 0.],
    # [1., 1., 0.],
    # [2., 1., 0.],
    # ...])
    objp = np.zeros((CHESSBOARD_ROW * CHESSBOARD_COLUMN, 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_COLUMN,
                           0:CHESSBOARD_ROW].T.reshape(-1, 2)
    objp *= CHESSBOARD_GRID_SIZE
    return objp


def read_gray_image(image_path: str):
    img = cv.imread(image_path)
    # TODO
    # assert img != None, f"Can't read the image ({image_path})"
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return gray


def find_chessboard_corners(image_gray):
    # The function attempts to determine whether the input image is a view of the chessboard pattern and locate the internal chessboard corners.
    # The function returns a non-zero value if all of the corners are found and they are placed in a certain order (row by row, left to right in every row).
    # Otherwise, if the function fails to find all the corners or reorder them, it returns 0.
    retval, corners = cv.findChessboardCorners(
        image_gray, (CHESSBOARD_COLUMN, CHESSBOARD_ROW), None)
    # TODO
    # ret, corners = cv2.findChessboardCorners(gray, (w, h),
    #                                                 flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FILTER_QUADS)

    if retval == 0:
        return None
    else:
        # Termination criteria
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # TODO
        # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, sys.float_info.epsilon)
        corners2 = cv.cornerSubPix(image_gray, corners, (11, 11), (-1, -1),
                                   criteria)
        return corners2


def get_image_size(image_paths: str):
    assert len(
        image_paths) >= 2, "The image path should contain more than 2 images."
    img_shape = cv.imread(image_paths[0]).shape

    for image_path in image_paths[1:]:
        assert img_shape == cv.imread(
            image_paths[0]
        ).shape, "All images should have the same size (height, width, channels)."

    return img_shape


def get_camera_calibration_coefficients(image_height, image_width, objpoints,
                                        imgpoints):
    retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, (image_width, image_height), None, None)

    imageSize = (image_width, image_height)
    newImgSize = (image_width, image_height)
    alpha = 1
    newcameramtx, validPixROI = cv.getOptimalNewCameraMatrix(
        cameraMatrix, distCoeffs, imageSize, alpha, newImgSize)

    return retval, cameraMatrix, distCoeffs, rvecs, tvecs, newcameramtx, validPixROI


def undistortion(image, cameraMatrix, distCoeffs, newcameramtx, validPixROI):
    # undistort
    dst = cv.undistort(image, cameraMatrix, distCoeffs, None, newcameramtx)

    # crop the image
    x, y, w, h = validPixROI
    return dst[y:y + h, x:x + w]


def save_processed_image(image, save_path, save_file_name):
    if not os.path.exists(save_path) or not os.path.isdir(save_path):
        os.makedirs(save_path)
    cv.imwrite(f"{save_path}/{save_file_name}", image)


def chessboard_image_camera_calibration_process(image_dir_path):
    image_paths = glob.glob(f"{image_dir_path}/*.jpg")
    assert len(
        image_paths
    ) >= MINIMUM_IMAGE_NUM, f"Require at least {MINIMUM_IMAGE_NUM} images to achieve good precision for this camera calibration method, but get {len(image_paths)} images."

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    success_found = 0

    for image_path in image_paths:
        image_gray = read_gray_image(image_path)
        chessboard_corners = find_chessboard_corners(image_gray)
        # TODO
        # if chessboard_corners != None:
        if True:
            objpoints.append(create_chessboard_corners_coordinate())
            imgpoints.append(chessboard_corners)
            success_found += 1

    assert success_found >= MINIMUM_IMAGE_NUM, f"Require at least {MINIMUM_IMAGE_NUM} success corner detection images to achieve good precision for this camera calibration method, but only {success_found} success detections."

    image_height, image_width, image_chanel_num = get_image_size(image_paths)

    retval, cameraMatrix, distCoeffs, rvecs, tvecs, newcameramtx, validPixROI = get_camera_calibration_coefficients(
        image_height, image_width, objpoints, imgpoints)

    return cameraMatrix, distCoeffs, rvecs, tvecs, newcameramtx, validPixROI


def undistortion_process(image_dir_path, image_save_dir_path,
                         image_save_name_prefix, cameraMatrix, distCoeffs,
                         newcameramtx, validPixROI):
    image_paths = glob.glob(f"{image_dir_path}/*.jpg")
    for i, image_path in enumerate(image_paths):
        image = cv.imread(image_path)
        image_undistortion = undistortion(image, cameraMatrix, distCoeffs,
                                          newcameramtx, validPixROI)
        save_file_name = f'{image_save_name_prefix}_{i}.jpg'
        save_processed_image(image_undistortion, image_save_dir_path,
                             save_file_name)
