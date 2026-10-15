"""Detect markers, hoop and green piece in an image and save an annotated copy.
Usage: detect_all.py image.png"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2
import numpy as np

from robotics_toolbox.maze.aruco_base import detect_markers, draw_markers
from robotics_toolbox.maze.detect import detect_green, detect_hoop

path = Path(sys.argv[1])
image = cv2.imread(str(path))
out = draw_markers(image, detect_markers(image))
hoop = detect_hoop(image)
if hoop:
    cv2.circle(out, tuple(hoop.center.astype(int)), int(hoop.radius), (0, 0, 255), 3)
    cv2.drawMarker(out, tuple(hoop.center.astype(int)), (0, 0, 255), cv2.MARKER_CROSS, 30, 3)
    print("hoop", hoop.center.round(1), round(hoop.radius, 1))
green = detect_green(image)
if green:
    contour, rect = green
    cv2.drawContours(out, [contour], -1, (255, 0, 255), 3)
    cv2.drawContours(out, [np.intp(cv2.boxPoints(rect))], -1, (255, 255, 0), 2)
    print("green rect", rect)
cv2.imwrite(str(path.with_name(path.stem + "_detected.png")), out)
