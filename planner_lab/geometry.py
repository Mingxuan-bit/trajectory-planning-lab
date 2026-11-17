from __future__ import annotations

import math
from dataclasses import dataclass


EPS = 1e-9


@dataclass(frozen=True)
class Point:
    x: float
    y: float

    def __add__(self, other: Point) -> Point:
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Point) -> Point:
        return Point(self.x - other.x, self.y - other.y)

    def __mul__(self, scale: float) -> Point:
        return Point(self.x * scale, self.y * scale)

    def __truediv__(self, scale: float) -> Point:
        return Point(self.x / scale, self.y / scale)


@dataclass(frozen=True)
class TimedPoint:
    x: float
    y: float
    t: float

    @property
    def point(self) -> Point:
        return Point(self.x, self.y)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def distance(a: Point, b: Point) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def norm(v: Point) -> float:
    return math.hypot(v.x, v.y)


def normalize(v: Point) -> Point:
    n = norm(v)
    if n < EPS:
        return Point(0.0, 0.0)
    return Point(v.x / n, v.y / n)


def dot(a: Point, b: Point) -> float:
    return a.x * b.x + a.y * b.y


def angle_of(v: Point) -> float:
    return math.atan2(v.y, v.x)


def angle_diff(a: float, b: float) -> float:
    d = (a - b + math.pi) % (2.0 * math.pi) - math.pi
    return d


def rotate(v: Point, angle: float) -> Point:
    c = math.cos(angle)
    s = math.sin(angle)
    return Point(c * v.x - s * v.y, s * v.x + c * v.y)


def lerp(a: Point, b: Point, u: float) -> Point:
    return Point(a.x + (b.x - a.x) * u, a.y + (b.y - a.y) * u)


def cubic_bezier(p0: Point, p1: Point, p2: Point, p3: Point, u: float) -> Point:
    q0 = lerp(p0, p1, u)
    q1 = lerp(p1, p2, u)
    q2 = lerp(p2, p3, u)
    r0 = lerp(q0, q1, u)
    r1 = lerp(q1, q2, u)
    return lerp(r0, r1, u)


def polyline_length(points: list[Point] | list[TimedPoint]) -> float:
    total = 0.0
    for a, b in zip(points, points[1:]):
        pa = a.point if isinstance(a, TimedPoint) else a
        pb = b.point if isinstance(b, TimedPoint) else b
        total += distance(pa, pb)
    return total


def path_smoothness(points: list[Point] | list[TimedPoint]) -> float:
    if len(points) < 3:
        return 0.0
    headings: list[float] = []
    plain = [p.point if isinstance(p, TimedPoint) else p for p in points]
    for a, b in zip(plain, plain[1:]):
        headings.append(angle_of(b - a))
    total = 0.0
    for h0, h1 in zip(headings, headings[1:]):
        total += angle_diff(h1, h0) ** 2
    return total / max(1, len(headings) - 1)


def resample_timed(points: list[Point], start_t: float, dt: float) -> list[TimedPoint]:
    return [TimedPoint(p.x, p.y, start_t + i * dt) for i, p in enumerate(points)]

