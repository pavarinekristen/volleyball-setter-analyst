"""The pinhole camera math shared by every calibrated measurement: a known real-world
size at some pixel size gives its depth; a known depth turns any pixel offset into meters.

Unlike the basketball's ``court.py``, there is no fixed court-relative origin here
(no reliable one is visible without the net square in frame). Every measurement is
relative to the player's own body at the moment of contact — which is what the
coaching cue "always set the ball in the same spot" actually means: the same spot
in front of your own face, not a fixed point on the floor.
"""

from __future__ import annotations

import math


def focal_px(frame_width: int, hfov_degrees: float) -> float:
    """The camera's focal length in pixels, from its horizontal field of view."""
    return (frame_width / 2) / math.tan(math.radians(hfov_degrees) / 2)


def depth_m(real_size_m: float, pixel_size: float, frame_width: int, hfov_degrees: float) -> float:
    """The depth (meters from the camera) of something ``real_size_m`` wide seen at
    ``pixel_size`` pixels."""
    return focal_px(frame_width, hfov_degrees) * real_size_m / pixel_size


def px_to_m(pixels: float, z_m: float, frame_width: int, hfov_degrees: float) -> float:
    """A pixel offset at depth ``z_m`` (meters from the camera), converted to meters.
    Works for horizontal or vertical offsets alike: the pinhole model treats both axes
    the same once the depth is known."""
    return pixels * z_m / focal_px(frame_width, hfov_degrees)


def height_m(y_px: float, floor_y_px: float, z_m: float, frame_width: int, hfov_degrees: float) -> float:
    """Height above the floor, in meters, of a point at pixel row ``y_px``, given the
    floor's own pixel row at the same depth (e.g. the player's ankle, in the same frame
    region, when both are roughly at ``z_m`` from the camera). Positive is up."""
    return px_to_m(floor_y_px - y_px, z_m, frame_width, hfov_degrees)
