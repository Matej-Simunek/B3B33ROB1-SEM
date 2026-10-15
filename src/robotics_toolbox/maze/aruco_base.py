"""ArUco detection on the maze base plate (drawing: docs/ROB1-2025-hw/drawings/base 01-02.pdf).

Base frame: origin in the plate corner, x to the right, y up in the drawing, units meters.
The plate is 125x125 mm with two 40x40 mm markers placed diagonally, 5 mm from the plate edges.
"""
from dataclasses import dataclass

import cv2
import numpy as np

ARUCO_DICT = cv2.aruco.DICT_4X4_50
PLATE_SIZE = 0.125
MARKER_SIZE = 0.040
MARKER_MARGIN = 0.005

# Marker centers in the base frame (marker ids are read from the first real image).
MARKER_CENTERS = {
    "upper_left": np.array([MARKER_MARGIN + MARKER_SIZE / 2, PLATE_SIZE - MARKER_MARGIN - MARKER_SIZE / 2]),
    "lower_right": np.array([PLATE_SIZE - MARKER_MARGIN - MARKER_SIZE / 2, MARKER_MARGIN + MARKER_SIZE / 2]),
}


@dataclass
class Marker:
    id: int
    corners: np.ndarray  # (4, 2) pixels, order TL, TR, BR, BL of the marker


def detect_markers(image: np.ndarray) -> list[Marker]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(ARUCO_DICT))
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return []
    return [Marker(int(i), c.reshape(4, 2)) for i, c in zip(ids.ravel(), corners)]


def draw_markers(image: np.ndarray, markers: list[Marker]) -> np.ndarray:
    out = image.copy()
    for m in markers:
        cv2.polylines(out, [m.corners.astype(np.int32)], True, (0, 255, 0), 2)
        cv2.putText(out, str(m.id), tuple(m.corners[0].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return out
