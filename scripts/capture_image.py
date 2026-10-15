"""Grab one image from the CRS93 camera and save it to data/images (robot does not move)."""
import sys
from datetime import datetime
from pathlib import Path

import cv2

from robotics_toolbox.maze.camera import grab_image

out_dir = Path(__file__).resolve().parents[1] / "data" / "images"
out_dir.mkdir(parents=True, exist_ok=True)
name = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime("%Y%m%d_%H%M%S")
path = out_dir / f"{name}.png"
cv2.imwrite(str(path), grab_image())
print(f"saved {path}")
