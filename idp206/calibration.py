"""Calibration classes"""

import os
import re
from glob import glob
import numpy as np
import cv2 as cv

WIN_SIZE = (11, 11)
ZERO_ZONE = (-1, -1)
MAX_ITER = 30
EPSILON = 0.001
TERMINATION_CRITERIA = (
    cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, MAX_ITER, EPSILON
)

class Calibration:
    """ Base class for calibration """
    def __init__(self, dirpaths=None, prefix='', image_size=None, image_format='jpg', cached=''):

        if dirpaths:
            for dirpath in dirpaths:
                if not os.path.isdir(dirpath):
                    raise ValueError(f'Invalid directory path: {dirpath}')

        self._dirpaths = dirpaths
        self._prefix = prefix
        self._image_format = image_format
        self._camera_matrix = None
        self._dist_coefficients = None
        self._rvecs = None
        self._tvecs = None
        self._image_size = image_size
        self._image_points = None

        if cached:
            self.load(cached)

    @property
    def dirpaths(self):
        return self._dirpaths

    @property
    def prefix(self):
        return self._prefix

    @property
    def image_format(self):
        return self._image_format

    @property
    def camera_matrix(self):
        if self._camera_matrix is None:
            self.calibrate()
        return self._camera_matrix

    @property
    def dist_coefficients(self):
        if self._dist_coefficients is None:
            self.calibrate()
        return self._dist_coefficients

    @property
    def rvecs(self):
        if self._rvecs is None:
            self.calibrate()
        return self._rvecs

    @property
    def tvecs(self):
        if self._tvecs is None:
            self.calibrate()
        return self._tvecs

    @property
    def image_size(self):
        return self._image_size

    @property
    def image_points(self):
        if self._image_points is None:
            self._image_points = self._get_image_points()
        return self._image_points

    def _get_image_points(self):
        raise NotImplementedError

    def get_frame_paths(self):
        frame_paths = []
        for path in self.dirpaths:
            frame_paths.extend(glob(f'{path}/{self.prefix}*.{self.image_format}'))
        return frame_paths

    def calibrate(self, camera_matrix=None, distortion_coeff=None, show_results=False):
        raise NotImplementedError

    def save(self, path):
        """ Save the camera matrix and the distortion coefficients to given path/file. """
        file = cv.FileStorage(path, cv.FILE_STORAGE_WRITE)

        file.write("camera_matrix", self.camera_matrix)
        file.write("dist_coeff", self.dist_coefficients)
        file.write("image_points", np.array(self.image_points, dtype=np.float32))

        file.release()

    def load(self, path):
        """ Loads camera matrix and distortion coefficients. """
        file = cv.FileStorage(path, cv.FILE_STORAGE_READ)

        self._camera_matrix = file.getNode("camera_matrix").mat()
        self._dist_coefficients = file.getNode("dist_coeff").mat()
        self._image_points = file.getNode("image_points").mat()

        file.release()

        return self

    def undistort(self, frame, alpha):
        """ Return an undistorted frame """
        height, width = frame.shape[:2]
        newcameramtx, _ = cv.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coefficients, (width, height), alpha, (width, height))
        return cv.undistort(
            frame, self.camera_matrix, self.dist_coefficients, None, newcameramtx)

    def heat_map(self, path):
        """ Show heat map of all image points used for calibration """
        frame = cv.imread(path)
        for point in self.image_points:
            point = np.squeeze(point)
            x = int(point[0])
            y = int(point[1])
            frame[y:y+1, x:x+1] = (0, 0, 255)
        cv.imshow("Heatmap", frame)
        cv.waitKey(0)

class CharUcoCalibration(Calibration):
    def __init__(self, *args, board='', **kwargs):
        self.name = board

        self._board = self._get_board(board)

        self._aruco_dict = None
        self._corners_list = None
        self._ids_list = None
        super().__init__(*args, **kwargs)

    @property
    def board(self):
        return self._board

    @property
    def corners_list(self):
        if self._corners_list is None:
            self.process_frames()
        return self._corners_list

    @property
    def ids_list(self):
        if self._ids_list is None:
            self.process_frames()
        return self._ids_list

    def _get_image_points(self):
        image_points = []
        for pts in self.corners_list:
            for point in pts:
                image_points.append(point)
        return image_points

    def _get_board(self, pattern):
        match = re.match(r'(\d*)x(\d*)_(\d*)_(\d*)_(.*)', pattern)
        if match:
            squares_x = int(match.group(1))
            squares_y = int(match.group(2)) # Number of squares in X, Y
            square_size = float(match.group(3))
            marker_size = float(match.group(4))

            key = getattr(cv.aruco, match.group(5), None)
            if key:
                aruco_dict = cv.aruco.getPredefinedDictionary(key)
                return cv.aruco.CharucoBoard_create(
                    squares_x,
                    squares_y,
                    square_size,
                    marker_size,
                    aruco_dict
                )
        raise ValueError(f'Invalid CharUco Board: {pattern}')

    def process_frames(self, show_results=False):
        corners_list = []
        ids_list = []
        paths = self.get_frame_paths()
        for path in paths:
            frame = cv.imread(path)

            # Don't process if the image size is different from given
            _, *shape = frame.shape[::-1]
            if self.image_size != tuple(shape):
                continue

            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            corners, ids, _ = cv.aruco.detectMarkers(gray, self.board.dictionary)
            if ids is None:
                continue

            # https://docs.opencv.org/3.4/d9/d6a/group__aruco.html#gadcc5dc30c9ad33dcf839e84e8638dcd1
            _, charUco_corners, charUco_ids = cv.aruco.interpolateCornersCharuco(
                corners,
                ids,
                gray,
                self.board,
                None,
                None,
                None,
                None,
                0
            )

            if charUco_ids is None or len(charUco_corners) < 6:
                continue

            # Sub-pixel refining
            charUco_corners = cv.cornerSubPix(
                    gray, corners, WIN_SIZE, ZERO_ZONE, TERMINATION_CRITERIA)

            corners_list.append(charUco_corners)
            ids_list.append(charUco_ids)

            if not show_results:
                continue

            frame = cv.aruco.drawDetectedCornersCharuco(frame, charUco_corners, charUco_ids)
            cv.imshow('Calibration', frame)
            cv.waitKey(50)

        self._corners_list = corners_list
        self._ids_list = ids_list
        return corners_list, ids_list     
                
    def calibrate(self, camera_matrix=None, distortion_coeff=None, show_results=False):
        corners_list, ids_list = self.process_frames(show_results)
        args = (
            corners_list,
            ids_list,
            self.board,
            self.image_size,
            camera_matrix,
            distortion_coeff,
            None,
            None,
            cv.CALIB_TILTED_MODEL,

        )

        success, camera_matrix, distortion_coeff, rvecs, tvecs =\
            cv.aruco.calibrateCameraCharuco(*args)
        if success:
            self._camera_matrix = camera_matrix
            self._dist_coefficients = distortion_coeff
            self._rvecs = rvecs
            self._tvecs = tvecs

        return camera_matrix, distortion_coeff, rvecs, tvecs


class Chessboard:
    def __init__(self, squares_x, squares_y, square_size):
        self.squares_x = squares_x
        self.squares_y = squares_y
        self.square_size = square_size

    @property
    def size(self):
        return (self.squares_x, self.squares_y)


class ChessboardCalibration(Calibration):
    def __init__(self, *args, board='', **kwargs):
        self.name = board

        self._board = self._get_board(board)

        super().__init__(*args, **kwargs)

    @property
    def board(self):
        return self._board

    def _get_image_points(self):
        _, image_points = self.process_frame()
        return image_points

    def _get_board(self, pattern):
        match = re.match(r'(\d*)x(\d*)_(\d*)', pattern)
        if match:
            squares_x = int(match.group(1))
            squares_y = int(match.group(2))
            square_size = float(match.group(3))
            return Chessboard(squares_x, squares_y, square_size)
        raise ValueError(f'Invalid Chessboard: {pattern}')

    def get_objp(self):
        squares_x, squares_y = self.board.size
        square_size = self.board.square_size
        obj_points = np.zeros((squares_x*squares_y, 3), np.float32)
        obj_points[:, :2] = np.mgrid[0:squares_x, 0:squares_y].T.reshape(-1, 2)
        return obj_points * square_size

    def get_chessboard_corners(self, image,
        flags=cv.CALIB_CB_ADAPTIVE_THRESH\
            + cv.CALIB_CB_NORMALIZE_IMAGE, refined=True):

        patternfound, corners = cv.findChessboardCorners(image, self.board.size, flags)
        if patternfound:
            if refined:
                return cv.cornerSubPix(
                    image, corners, WIN_SIZE, ZERO_ZONE, TERMINATION_CRITERIA)
            return corners

    def process_frame(self, show_results=False):
        object_points = []
        image_points = []

        # All 3D points in a chessboard space
        objp = self.get_objp()

        for path in self.get_frame_paths():
            frame = cv.imread(path)
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            corners = self.get_chessboard_corners(gray)
            if corners is None:
                continue

            object_points.append(objp)
            image_points.append(corners)

            if not show_results:
                continue

            frame = cv.drawChessboardCorners(frame, self.board.size, corners, True)
            cv.imshow('Calibration', frame)
            cv.waitkey(500)
        
        return np.array(object_points, dtype=np.float32), np.array(image_points, dtype=np.float32)

    def calibrate(self, camera_matrix=None, distortion_coeff=None, show_results=False):
        object_points, image_points = self.process_frame(show_results=show_results)

        args = (
            object_points,
            image_points,
            self.image_size,
            camera_matrix,
            distortion_coeff,
            None,
            None,
            cv.CALIB_TILTED_MODEL
        )

        success, camera_matrix, distortion_coeff, rvecs, tvecs = cv.calibrateCamera(*args)
        if success:
            self._camera_matrix = camera_matrix
            self._dist_coefficients = distortion_coeff
            self._rvecs = rvecs
            self._tvecs = tvecs

        return camera_matrix, distortion_coeff, rvecs, tvecs
