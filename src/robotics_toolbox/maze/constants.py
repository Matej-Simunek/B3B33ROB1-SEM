"""Measured geometry of the CRS93 station (meters)."""
CAMERA_HEIGHT = 1.17  # camera above the table, measured
PLATFORM_HEIGHT = 0.033  # table to top of the base plate (foam + plate), measured
MARKER_EDGE = 0.040  # black square of the ArUco marker, measured
IMAGE_SIZE = (1920, 1200)  # width, height of camera-crs93
PUZZLE_RAISE = 0.034  # puzzle foot sits this much above the ArUco marker plane
PUZZLE_BASE_HEIGHT = PLATFORM_HEIGHT + PUZZLE_RAISE  # puzzle foot above the table (0.067 m)
