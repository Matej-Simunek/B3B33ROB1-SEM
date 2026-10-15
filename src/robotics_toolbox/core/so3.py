#!/usr/bin/env python
#
# Copyright (c) CTU -- All Rights Reserved
# Created on: 2023-07-4
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#

"""Module for representing 3D rotation."""

from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike


class SO3:
    """This class represents an SO3 rotations internally represented by rotation
    matrix."""

    def __init__(self, rotation_matrix: ArrayLike | None = None) -> None:
        """Creates a rotation transformation from rot_vector."""
        super().__init__()
        self.rot: np.ndarray = (
            np.asarray(rotation_matrix) if rotation_matrix is not None else np.eye(3)
        )

    @staticmethod
    def exp(rot_vector: ArrayLike) -> SO3:
        """Compute SO3 transformation from a given rotation vector, i.e. exponential
        representation of the rotation."""
        v = np.asarray(rot_vector)
        assert v.shape == (3,)
        t = SO3()
        # todo HW01: implement Rodrigues' formula, t.rot = ...
        theta = float(np.linalg.norm(v))
        
        if np.isclose(theta, 0.0):
            return t
        
        u = v / theta
        skew_mat = np.array([[0.0, -u[2], u[1]],
                          [u[2], 0.0, -u[0]],
                          [-u[1], u[0], 0.0]])
        
        t.rot = np.eye(3) + np.sin(theta) * skew_mat + (1 - np.cos(theta)) * (skew_mat @ skew_mat)
        return t

    def log(self) -> np.ndarray:
        """Compute rotation vector from this SO3"""
        # todo HW01: implement computation of rotation vector from this SO3
        R = self.rot
        cos_theta = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
        theta = np.arccos(cos_theta)

        if np.isclose(theta, 0.0):
            return np.zeros(3)

        if np.isclose(theta, np.pi):
            B = (R + np.eye(3)) / 2
            u = np.sqrt(np.clip(np.diag(B), 0.0, None))
            if u[0] > 0:
                u[1] = np.copysign(u[1], B[0, 1])
                u[2] = np.copysign(u[2], B[0, 2])
            elif u[1] > 0:
                u[2] = np.copysign(u[2], B[1, 2])

            return theta * u
        
        v = np.array([R[2,1] - R[1,2],
                      R[0,2] - R[2, 0],
                      R[1,0] - R[0, 1]])
        u = v / (2 * np.sin(theta))
        return theta * u

    def __mul__(self, other: SO3) -> SO3:
        """Compose two rotations, i.e., self * other"""
        # todo: HW01: implement composition of two rotation.
        res = SO3()
        res.rot = self.rot @ other.rot
        return res

    def inverse(self) -> SO3:
        """Return inverse of the transformation."""
        # todo: HW01: implement inverse, do not use np.linalg.inverse()
        res = SO3()
        res.rot = self.rot.T.copy()
        return res

    def act(self, vector: ArrayLike) -> np.ndarray:
        """Rotate given vector by this transformation."""
        v = np.asarray(vector)
        assert v.shape == (3,)
        return self.rot @ v

    def __eq__(self, other: SO3) -> bool:
        """Returns true if two transformations are almost equal."""
        return np.allclose(self.rot, other.rot)

    @staticmethod
    def rx(angle: float) -> SO3:
        """Return rotation matrix around x axis."""
        # todo: HW1opt: implement rx
        cos = np.cos(angle)
        sin = np.sin(angle)
        return SO3(np.array([[1, 0, 0],
                             [0, cos, -sin],
                             [0, sin, cos]]))

    @staticmethod
    def ry(angle: float) -> SO3:
        """Return rotation matrix around y axis."""
        # todo: HW1opt: implement ry
        cos = np.cos(angle)
        sin = np.sin(angle)
        return SO3(np.array([[cos, 0, sin],
                             [0,   1,    0],
                             [-sin, 0,  cos]]))

    @staticmethod
    def rz(angle: float) -> SO3:
        """Return rotation matrix around z axis."""
        # todo: HW1opt: implement rz
        cos = np.cos(angle)
        sin = np.sin(angle)
        return SO3(np.array([[cos, -sin, 0],
                             [sin,  cos, 0],
                             [  0,    0, 1]]))

    @staticmethod
    def from_quaternion(q: ArrayLike) -> SO3:
        """Compute rotation from quaternion in a form [qx, qy, qz, qw]."""
        # todo: HW1opt: implement from quaternion
        q = np.asarray(q, dtype=float)
        q = q / np.linalg.norm(q)  
                
        q_xyz = q[:3]
        q_w = np.clip(q[3], -1.0, 1.0)
                
        norm_q_xyz = np.linalg.norm(q_xyz)
                
        if np.isclose(norm_q_xyz, 0.0):
            return SO3()
                    
        theta = 2.0 * np.arccos(q_w)
        axis = q_xyz / norm_q_xyz
                
        return SO3.exp(theta * axis)


        #raise NotImplementedError("From quaternion needs to be implemented")

    def to_quaternion(self) -> np.ndarray:
        """Compute quaternion from self."""
        # todo: HW1opt: implement to quaternion
        R = self.rot
        tr = np.trace(R)
        
        if tr > 0:
            q_w = 0.5 * np.sqrt(1.0 + tr)
            
            mult = 1.0 / (4.0 * q_w)
            
            q_x = (R[2, 1] - R[1, 2]) * mult
            q_y = (R[0, 2] - R[2, 0]) * mult
            q_z = (R[1, 0] - R[0, 1]) * mult
            
        elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            q_x = 0.5 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            mult = 1.0 / (4.0 * q_x)
            
            q_w = (R[2, 1] - R[1, 2]) * mult
            q_y = (R[0, 1] + R[1, 0]) * mult
            q_z = (R[0, 2] + R[2, 0]) * mult
            
        elif R[1, 1] > R[2, 2]:
            q_y = 0.5 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            mult = 1.0 / (4.0 * q_y)
            
            q_w = (R[0, 2] - R[2, 0]) * mult
            q_x = (R[0, 1] + R[1, 0]) * mult
            q_z = (R[1, 2] + R[2, 1]) * mult
            
        else:
            q_z = 0.5 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            mult = 1.0 / (4.0 * q_z)
            
            q_w = (R[1, 0] - R[0, 1]) * mult
            q_x = (R[0, 2] + R[2, 0]) * mult
            q_y = (R[1, 2] + R[2, 1]) * mult
            
        return np.array([q_x, q_y, q_z, q_w])
        #raise NotImplementedError("To quaternion needs to be implemented")

    @staticmethod
    def from_angle_axis(angle: float, axis: ArrayLike) -> SO3:
        """Compute rotation from angle axis representation."""
        # todo: HW1opt: implement from angle axis
        axis = np.asarray(axis, dtype=float)
        norm = np.linalg.norm(axis)
        
        if np.isclose(norm, 0.0):
            return SO3()

        rot_vec = (axis / norm) * angle
        return SO3.exp(rot_vec)
        #raise NotImplementedError("Needs to be implemented")

    def to_angle_axis(self) -> tuple[float, np.ndarray]:
        """Compute angle axis representation from self."""
        # todo: HW1opt: implement to angle axis
        rot_vec = self.log()
        angle = float(np.linalg.norm(rot_vec))
        
        if np.isclose(angle, 0.0):
            return 0.0, np.ndarray([1.0, 0.0, 0.0])
        
        return angle, (rot_vec / angle)
        #raise NotImplementedError("Needs to be implemented")

    @staticmethod
    def from_euler_angles(angles: ArrayLike, seq: list[str]) -> SO3:
        """Compute rotation from euler angles defined by a given sequence.
        angles: is a three-dimensional array of angles
        seq: is a list of axis around which angles rotate, e.g. 'xyz', 'xzx', etc.
        """
        # todo: HW1opt: implement from euler angles
        angles = np.asarray(angles, dtype=float)
        axis_map = {'x': SO3.rx, 'y': SO3.ry, 'z': SO3.rz}

        res = SO3()
        for a, ax in zip(angles, seq):
            step_rot = axis_map[ax.lower()](a)
            res = res * step_rot

        return res
        
        #raise NotImplementedError("Needs to be implemented")

    def __hash__(self):
        return id(self)
