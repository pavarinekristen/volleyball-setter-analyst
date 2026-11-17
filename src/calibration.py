"""Depth calibration, reconciled from up to three independent sources. Basketball
has one (the rim); volleyball has none that's always visible, so every source here
can fail gracefully and the result says how much to trust it.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import court


@dataclass
class Estimate:
    z_m: float | None
    source: str   # "player_height", "ball_diameter", or "net"


@dataclass
class Calibration:
    z_m: float                 # the depth actually used
    source: str                # which estimate it came from
    confidence: float          # 0-1: agreement between whichever sources were available
    estimates: dict[str, float]   # every source that produced a reading, for the record


def from_player(body_px: float, player_height_m: float, frame_width: int, hfov_degrees: float,
                *, shoulder_to_ankle: float = 0.78) -> float:
    """Depth from the player's own shoulder-to-ankle size in pixels, same technique
    as the basketball's court calibration."""
    return court.depth_m(shoulder_to_ankle * player_height_m, body_px, frame_width, hfov_degrees)


def from_ball(diameter_px: float, ball_diameter_m: float, frame_width: int, hfov_degrees: float) -> float:
    """Depth from the ball's own pixel size, ideally in the 2-3 frames just before
    contact (still in free flight, not yet blurred or occluded by the hands)."""
    return court.depth_m(ball_diameter_m, diameter_px, frame_width, hfov_degrees)


def from_net(net_width_px: float, net_width_m: float, frame_width: int, hfov_degrees: float) -> float:
    """Depth from the net's pixel span, only usable when the whole net is in frame."""
    return court.depth_m(net_width_m, net_width_px, frame_width, hfov_degrees)


def reconcile(*, player_z: float | None, ball_z: float | None, net_z: float | None) -> Calibration:
    """Player height is primary when available (it's sampled at the same depth as the
    contact, every time); the ball and net are cross-checks. Confidence drops with how
    much the available sources disagree, not just whether one is missing."""
    estimates = {k: v for k, v in (("player_height", player_z), ("ball_diameter", ball_z),
                                   ("net", net_z)) if v is not None}
    if not estimates:
        return Calibration(z_m=float("nan"), source="none", confidence=0.0, estimates={})

    source = "player_height" if "player_height" in estimates else next(iter(estimates))
    z = estimates[source]
    values = list(estimates.values())
    spread = (max(values) - min(values)) / z if len(values) > 1 else 0.0
    confidence = max(0.0, 1.0 - 2.0 * spread) if len(values) > 1 else 0.6   # one source alone: moderate trust
    return Calibration(z_m=z, source=source, confidence=round(confidence, 2), estimates=estimates)
