from __future__ import annotations

"""
Vector classes for 3D and 4D operations.

Provides Vec3 and Vec4 classes with arithmetic operations, conversions,
and utility methods for geometric calculations.
"""

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class Vec2:
    """Lightweight 2D vector with numeric operations and validation."""

    x: float
    y: float

    @classmethod
    def from_iterable(cls, values: Iterable[float]) -> "Vec2":
        vx, vy = list(values)
        return cls(float(vx), float(vy))

    @classmethod
    def from_polar(cls, r: float, theta: float) -> "Vec2":
        """Create Vec2 from polar coordinates (radius, angle in radians)."""
        return cls(r * math.cos(theta), r * math.sin(theta))

    def as_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)

    def to_numpy(self, dtype=float) -> np.ndarray:
        return np.array([self.x, self.y], dtype=dtype)

    def length(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y)

    def length_squared(self) -> float:
        """Return squared length (avoids sqrt for performance)."""
        return self.x * self.x + self.y * self.y

    def normalized(self) -> "Vec2":
        m = self.length()
        if m == 0:
            return Vec2(0.0, 0.0)
        return Vec2(self.x / m, self.y / m)

    def dot(self, other: "Vec2") -> float:
        return self.x * other.x + self.y * other.y

    def cross(self, other: "Vec2") -> float:
        """2D cross product returns scalar (z-component of 3D cross)."""
        return self.x * other.y - self.y * other.x

    def perpendicular(self) -> "Vec2":
        """Return perpendicular vector (90° counterclockwise rotation)."""
        return Vec2(-self.y, self.x)

    def angle(self) -> float:
        """Return angle in radians from positive x-axis."""
        return math.atan2(self.y, self.x)

    def angle_to(self, other: "Vec2") -> float:
        """Return angle in radians to another vector."""
        return math.atan2(self.cross(other), self.dot(other))

    def rotate(self, angle: float) -> "Vec2":
        """Rotate vector by angle in radians."""
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        return Vec2(
            self.x * cos_a - self.y * sin_a,
            self.x * sin_a + self.y * cos_a
        )

    def lerp(self, other: "Vec2", t: float) -> "Vec2":
        """Linear interpolation between this and other vector."""
        return Vec2(
            self.x + t * (other.x - self.x),
            self.y + t * (other.y - self.y)
        )

    def distance_to(self, other: "Vec2") -> float:
        """Distance to another vector."""
        dx = other.x - self.x
        dy = other.y - self.y
        return math.sqrt(dx * dx + dy * dy)

    def distance_squared_to(self, other: "Vec2") -> float:
        """Squared distance to another vector (avoids sqrt)."""
        dx = other.x - self.x
        dy = other.y - self.y
        return dx * dx + dy * dy

    def reflect(self, normal: "Vec2") -> "Vec2":
        """Reflect vector across a normal."""
        return self - 2 * self.dot(normal) * normal

    def project_onto(self, other: "Vec2") -> "Vec2":
        """Project this vector onto another vector."""
        other_len_sq = other.length_squared()
        if other_len_sq == 0:
            return Vec2(0.0, 0.0)
        return other * (self.dot(other) / other_len_sq)

    def __add__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> "Vec2":
        return Vec2(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar: float) -> "Vec2":
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> "Vec2":
        return Vec2(self.x / scalar, self.y / scalar)

    def __neg__(self) -> "Vec2":
        return Vec2(-self.x, -self.y)

    def __iter__(self):
        yield self.x
        yield self.y

    def __len__(self) -> int:
        return 2

    def __getitem__(self, idx: int) -> float:
        if idx == 0:
            return self.x
        if idx == 1:
            return self.y
        raise IndexError("Vec2 index out of range")

    def __setitem__(self, idx: int, value: float) -> None:
        if idx == 0:
            self.x = float(value)
        elif idx == 1:
            self.y = float(value)
        else:
            raise IndexError("Vec2 index out of range")


@dataclass
class Vec3:
    """Lightweight 3D vector with numeric operations and validation."""

    x: float
    y: float
    z: float

    @classmethod
    def from_iterable(cls, values: Iterable[float]) -> "Vec3":
        vx, vy, vz = list(values)
        return cls(float(vx), float(vy), float(vz))

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def to_numpy(self, dtype=float) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=dtype)

    def length(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def normalized(self) -> "Vec3":
        m = self.length()
        if m == 0:
            return Vec3(0.0, 0.0, 0.0)
        return Vec3(self.x / m, self.y / m, self.z / m)

    def dot(self, other: "Vec3") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: "Vec3") -> "Vec3":
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def cross_normalized(self, other: "Vec3") -> "Vec3":
        """Return the unit vector of the cross product self x other.

        If the cross product is the zero vector (parallel inputs), returns Vec3(0,0,0).
        """
        c = self.cross(other)
        m = c.length()
        if m == 0:
            return Vec3(0.0, 0.0, 0.0)
        return c / m

    def __add__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> "Vec3":
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar: float) -> "Vec3":
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> "Vec3":
        return Vec3(self.x / scalar, self.y / scalar, self.z / scalar)

    def __iter__(self):
        yield self.x
        yield self.y
        yield self.z

    def __len__(self) -> int:
        return 3

    def __getitem__(self, idx: int) -> float:
        if idx == 0:
            return self.x
        if idx == 1:
            return self.y
        if idx == 2:
            return self.z
        raise IndexError("Vec3 index out of range")


@dataclass
class Vec4:
    x: float
    y: float
    z: float
    w: float

    @classmethod
    def from_iterable(cls, values: Iterable[float]) -> "Vec4":
        vx, vy, vz, vw = list(values)
        return cls(float(vx), float(vy), float(vz), float(vw))

    def xyz(self) -> Vec3:
        return Vec3(self.x, self.y, self.z)

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x, self.y, self.z, self.w)