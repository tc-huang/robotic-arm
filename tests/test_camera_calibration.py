# Copyright (c) 2023 tc-haung
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from src.camera import camera_calibration
import os


def test_camera_calibration():
    cameraMatrix, distCoeffs, rvecs, tvecs, newcameramtx, validPixROI = camera_calibration.chessboard_image_camera_calibration_process(
        "/tf/tests/test_images/chessboard")
    camera_calibration.undistortion_process(
        image_dir_path='/tf/tests/test_images/chessboard',
        image_save_dir_path='/tf/tests/test_images/chessboard_undistored',
        image_save_name_prefix="chessboard_undistored",
        cameraMatrix=cameraMatrix,
        distCoeffs=distCoeffs,
        newcameramtx=newcameramtx,
        validPixROI=validPixROI)
