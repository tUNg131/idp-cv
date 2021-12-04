"""Example"""

import cv2 as cv

from idp206.calibration import Calibration
from idp206.camera import Camera, CAMERA_INFO
from idp206.dummy import find_dummies

idpcam2 = Camera('idpcam2', *CAMERA_INFO.get('idpcam2'), local_port=8082,
    calibration=Calibration(cached='cam2_charuco_12nov.yaml'))

with idpcam2.open(ssh_tunnel=True) as cap:
    while cap.isOpened():
        _, frame = cap.read()

        dummy_img_points = find_dummies(frame)

        if len(dummy_img_points) > 0:
            for point in dummy_img_points:
                frame = cv.circle(
                                img=frame,
                                center=point,
                                radius=3,
                                color=(0, 0, 255),
                                thickness=(-1)
                            )

        cv.imshow('Example', frame)
        cv.waitKey(2)
    