"""Detect ArUco markers in an image and print their ids. Usage: detect_base.py image.png"""
import sys

import cv2

from robotics_toolbox.maze.aruco_base import detect_markers, draw_markers

image = cv2.imread(sys.argv[1])
markers = detect_markers(image)
for m in markers:
    print(m.id, m.corners.mean(axis=0))
out = sys.argv[1].rsplit(".", 1)[0] + "_markers.png"
cv2.imwrite(out, draw_markers(image, markers))
print(f"{len(markers)} markers, saved {out}")
