"""Functions to detect dummies from a frame"""

import numpy as np
import cv2 as cv

PINK = np.array((180, 70, 235), dtype=np.uint8) #EB46B4
GREEN = np.array((70, 235, 180), dtype=np.uint8) #B4EB46

def get_mask(frame, color, deviation=20):
    return cv.inRange(frame, color - deviation, color + deviation)

mask = get_mask(cv.imread('cam2_pink.jpg'), PINK)

def clean_frame_Canny(frame, canny_max=300, canny_min=200):
    all_edges = cv.Canny(frame, canny_max, canny_min)
    dummy_edges = cv.bitwise_and(all_edges, mask)

    # Filter small dotted region
    nlabels, labels, stats, _ = cv.connectedComponentsWithStats(
        dummy_edges, None, None, None, 8, cv.CV_32S)
    sizes = stats[1:, -1] #get CC_STAT_AREA component
    filtered = np.zeros((labels.shape), np.uint8)

    for i in range(0, nlabels - 1):
        if sizes[i] >= 10:   #filter small dotted regions
            filtered[labels == i + 1] = 255

    kernel = np.ones((2,2),np.uint8)

    return cv.dilate(filtered, kernel, iterations=3)

def clean_frame_adaptive_threshold(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    threshold = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
        cv.THRESH_BINARY_INV,5,5)

    # Apply mask
    masked = cv.bitwise_and(threshold, mask)

    # Filter small dotted region
    nlabels, labels, stats, _ = cv.connectedComponentsWithStats(
        masked, None, None, None, 8, cv.CV_32S)
    sizes = stats[1:, -1] #get CC_STAT_AREA component
    filtered = np.zeros((labels.shape), np.uint8)

    for i in range(0, nlabels - 1):
        if sizes[i] >= 60:   #filter small dotted regions
            filtered[labels == i + 1] = 255

    blur = cv.medianBlur(filtered, 3)

    kernel = np.ones((2,2),np.uint8)
    dilated = cv.dilate(blur, kernel)

    return dilated

def clean_frame(frame):
    """ Get rid of a noises and unrelated area of the frame. """
    return clean_frame_Canny(frame)

def get_dummy_img_point(contour, principal_point=None, offset=-7):
    """ 
    Return an estimated position of the dummy from
    its contour if principal point is provided. 
    """

    M = cv.moments(contour)
    centroid = np.array([
        M['m10']/M['m00'],
        M['m01']/M['m00']
    ])

    if principal_point:
        estimated = offset * (centroid - principal_point) / \
            np.linalg.norm(centroid - principal_point) + centroid
        return estimated.astype(int)
    return centroid

def find_dummies(frame):
    """ Return an np array of found dummies in the frame"""
    cleaned = clean_frame(frame)
    
    contours, _ = cv.findContours(cleaned, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    dummy_img_points = [get_dummy_img_point(np.squeeze(cnt)) for cnt in contours]

    return np.array(dummy_img_points, dtype=np.float)
