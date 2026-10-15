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
    inliers: int = 0
    coverage: float = 1.0  # fraction of the circumference supported by blue pixels (1 = complete ring)


def _largest_contour(mask: np.ndarray, min_area: float):
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    return max(contours, key=cv2.contourArea) if contours else None


def _circle_from_3(p: np.ndarray):
    (ax, ay), (bx, by), (cx, cy) = p
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-9:
        return None
    ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) + (cx**2 + cy**2) * (ay - by)) / d
    uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) + (cx**2 + cy**2) * (bx - ax)) / d
    return np.array([ux, uy]), float(np.hypot(ax - ux, ay - uy))


def _refine_circle(points: np.ndarray):
    """Algebraic (Kasa) least-squares circle through the points."""
    A = np.column_stack([2 * points[:, 0], 2 * points[:, 1], np.ones(len(points))])
    b = (points**2).sum(axis=1)
    (cx, cy, c), *_ = np.linalg.lstsq(A, b, rcond=None)
    return np.array([cx, cy]), float(np.sqrt(max(c + cx**2 + cy**2, 0.0)))


def fit_circle_ransac(points: np.ndarray, r_range=(100.0, 300.0), tol=3.0, iterations=500, seed=0) -> Circle | None:
    """Robust circle through the ring pixels; other blue objects (handle, arrow, blobs) are outliers."""
    if len(points) < 50:
        return None
    rng = np.random.default_rng(seed)
    pts = points[rng.choice(len(points), min(len(points), 4000), replace=False)].astype(float)
    best, best_n = None, 0
    for _ in range(iterations):
        circle = _circle_from_3(pts[rng.choice(len(pts), 3, replace=False)])
        if circle is None or not (r_range[0] <= circle[1] <= r_range[1]):
            continue
        n = int(np.sum(np.abs(np.linalg.norm(pts - circle[0], axis=1) - circle[1]) < tol))
        if n > best_n:
            best, best_n = circle, n
    if best is None:
        return None
    for k in range(5):  # refine on the inliers, tightening the band
        inl = np.abs(np.linalg.norm(pts - best[0], axis=1) - best[1]) < (tol if k < 2 else 0.8 * tol)
        if inl.sum() < 20:
            break
        best = _refine_circle(pts[inl])
    inl = np.abs(np.linalg.norm(pts - best[0], axis=1) - best[1]) < 1.5 * tol
    angles = np.arctan2(pts[inl, 1] - best[0][1], pts[inl, 0] - best[0][0])
    coverage = float(np.mean(np.histogram(angles, bins=36, range=(-np.pi, np.pi))[0] > 0))
    return Circle(best[0], best[1], int(inl.sum()), coverage)


def detect_hoop(image: np.ndarray, r_range=(100.0, 300.0)) -> Circle | None:
    """Circle through the blue ring of the hoop (robust to other blue objects and a partly hidden ring)."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, *BLUE)
    edges = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))  # outlines: a blob adds few points
    ys, xs = np.nonzero(edges)
    circle = fit_circle_ransac(np.column_stack([xs, ys]), r_range, tol=5.0)
    return circle if circle is not None and circle.inliers >= 100 else None


def hoop_fully_visible(circle: Circle, image_size=(1920, 1200), min_coverage=0.85, margin=5.0) -> bool:
    """The ring is complete in the image (not cut by the border, not occluded)."""
    w, h = image_size
    x, y = circle.center
    r = circle.radius
    return circle.coverage >= min_coverage and r + margin < x < w - r - margin and r + margin < y < h - r - margin


def detect_green(image: np.ndarray):
    """Largest green object: returns the contour and its min-area rectangle ((cx, cy), (w, h), angle)."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    contour = _largest_contour(cv2.inRange(hsv, *GREEN), min_area=500)
    if contour is None:
        return None
    return contour, cv2.minAreaRect(contour)


def hoop_usable(circle: Circle | None, min_coverage=0.5) -> bool:
    """Enough of the ring is visible to trust the fitted circle (a half circle is enough, also at the border)."""
    return circle is not None and circle.coverage >= min_coverage
