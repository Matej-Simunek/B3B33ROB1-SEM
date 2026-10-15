"""Hand-tuned correction for the real robot. Edit this file on the robot PC."""

# Every height sent to the robot is raised by this many meters, while the plan (floors, the 4 cm success check,
# the printed heights) still uses the planned heights. Use it when the hoop physically ends LOWER than planned
# (the arm sags a few cm): e.g. 0.02 = send the robot 2 cm higher. 0.0 = no correction.
Z_COMPENSATION = 0.0
