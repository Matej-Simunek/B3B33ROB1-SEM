"""Color-based detection of the hoop (blue ring) and the green puzzle piece."""
from dataclasses import dataclass

import cv2
import numpy as np

# HSV ranges (OpenCV: H 0-179). Tuned on data/images/01_hoop_above_base.png.
BLUE = ((90, 80, 40), (120, 255, 200))
GREEN = ((60, 60, 40), (95, 255, 220))


@dataclass
class Circle:
    center: np.ndarray  # pixels (x, y)
    radius: float


def _largest_contour(mask: np.ndarray, min_area: float):
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    return max(contours, key=cv2.contourArea) if contours else None


def detect_hoop(image: np.ndarray) -> Circle | None:
    """Circle through the blue ring of the hoop."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    contour = _largest_contour(cv2.inRange(hsv, *BLUE), min_area=500)
    if contour is None:
        return None
    (x, y), r = cv2.minEnclosingCircle(contour)
    return Circle(np.array([x, y]), float(r))


def detect_green(image: np.ndarray):
    """Largest green object: returns the contour and its min-area rectangle ((cx, cy), (w, h), angle)."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    contour = _largest_contour(cv2.inRange(hsv, *GREEN), min_area=500)
    if contour is None:
        return None
    return contour, cv2.minAreaRect(contour)
