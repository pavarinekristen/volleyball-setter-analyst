"""Sets: found in the ball's own trajectory (not the pose, which is too similar across
a block, an attack and a set to gate on alone), then measured for height, form and
consistency against the player's own body, not a fixed target.

There is no "make or miss" here — see ``judge()`` in the basketball demo for that
idea. A set is scored for *quality*: how close to the player's own best reach, how
steady from one repetition to the next.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import court
from .skeleton import KPT_NAMES

J = {name: i for i, name in enumerate(KPT_NAMES)}


# ── pose, as arrays (identical technique to the basketball demo: generic geometry) ──

def keypoints(by_index: dict, n_frames: int, width: int, height: int) -> np.ndarray:
    """``(n_frames, 17, 2)`` in pixels; NaN where nobody is posed or a joint is missing."""
    out = np.full((n_frames, len(KPT_NAMES), 2), np.nan)
    for i in range(n_frames):
        persons = by_index.get(i)
        if not persons:
            continue
        kpts = np.asarray(persons[0].get("kpts_xy") or [], float)
        if kpts.shape != (len(KPT_NAMES), 2):
            continue
        missing = (kpts == 0).all(axis=1)
        kpts = kpts * [width, height]
        kpts[missing] = np.nan
        out[i] = kpts
    return out


def elbow_angle(kp: np.ndarray, side: str, *, min_projected: float = 0.0) -> np.ndarray:
    """The elbow angle per frame in the image, degrees (180 = straight). NaN where the
    arm points at the camera, same technique as the basketball demo."""
    s, e, w = (kp[:, J[f"{side}_{j}"]] for j in ("shoulder", "elbow", "wrist"))
    a, b = s - e, w - e
    la, lb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        angle = np.degrees(np.arccos(np.clip((a * b).sum(axis=1) / (la * lb), -1, 1)))
        if min_projected > 0:
            short = (la < min_projected * np.nanpercentile(la, 90)) | (
                lb < min_projected * np.nanpercentile(lb, 90))
            angle[short] = np.nan
    return angle


def shoulder_ratio(kp: np.ndarray) -> np.ndarray:
    """Shoulder width / torso length per frame: low side-on, high facing the camera."""
    shoulders = np.linalg.norm(kp[:, J["left_shoulder"]] - kp[:, J["right_shoulder"]], axis=1)
    mid_s = (kp[:, J["left_shoulder"]] + kp[:, J["right_shoulder"]]) / 2
    mid_h = (kp[:, J["left_hip"]] + kp[:, J["right_hip"]]) / 2
    with np.errstate(invalid="ignore", divide="ignore"):
        return _median(shoulders / np.linalg.norm(mid_s - mid_h, axis=1), 9)


def _median(values: np.ndarray, frames: int) -> np.ndarray:
    half = frames // 2
    out = np.full(len(values), np.nan)
    for i in range(len(values)):
        window = values[max(0, i - half):i + half + 1]
        if np.isfinite(window).any():
            out[i] = np.nanmedian(window)
    return out


def _line_y(kp: np.ndarray, name: str) -> np.ndarray:
    if name == "nose":
        return kp[:, J["nose"], 1]
    a, b = J[f"left_{name}"], J[f"right_{name}"]
    return np.nanmean(kp[:, [a, b], 1], axis=1)


# ── layer 1: coarse candidate windows, loose on purpose ─────────────────────────

def candidates_coarse(kp: np.ndarray, fps: float, *, armed_above: str, smooth_frames: int,
                      min_gap_seconds: float, pad_seconds: float) -> list[tuple[int, int]]:
    """Every moment the wrists rise above *armed_above*, padded into a window. No
    sustained-pose requirement (a set's contact is brief) and no "armed" reset
    (repeated touches can come quickly) — this is meant to over-trigger; layer 2
    is what actually finds and validates the contact inside each window."""
    line_y = _line_y(kp, armed_above)
    wrists = np.nanmax(kp[:, [J["left_wrist"], J["right_wrist"]], 1], axis=1)
    with np.errstate(invalid="ignore"):
        above = _median((wrists < line_y).astype(float), smooth_frames) > 0.5
    n = len(above)
    crossings = [i for i in range(1, n) if above[i] and not above[i - 1]]
    merged: list[int] = []
    for c in crossings:
        if merged and c - merged[-1] < min_gap_seconds * fps:
            continue
        merged.append(c)
    pad = int(pad_seconds * fps)
    windows = [[max(0, c - pad), min(n - 1, c + pad)] for c in merged]
    out: list[list[int]] = []
    for w in windows:
        if out and w[0] <= out[-1][1]:
            out[-1][1] = max(out[-1][1], w[1])
        else:
            out.append(w)
    return [(a, b) for a, b in out]


# ── layer 2: the contact instant, from the ball's own path ─────────────────────

def find_contact(a: int, b: int, x: np.ndarray, y: np.ndarray, d: np.ndarray, kp: np.ndarray, *,
                 proximity_balls: float, max_hold_frames: int) -> dict | None:
    """The sub-frame instant the ball's fall reverses into a rise (a peak in pixel y,
    found by fitting a parabola to three consecutive tracked frames), validated by how
    close it is to a wrist at that instant. ``None`` if no peak in the window passes."""
    frames = [f for f in range(a, b + 1) if np.isfinite(x[f]) and np.isfinite(y[f])]
    best = None
    for i in range(1, len(frames) - 1):
        f0, f, f1 = frames[i - 1], frames[i], frames[i + 1]
        if f - f0 != 1 or f1 - f != 1:
            continue   # need three frames in a row to fit the peak; a gap skips it
        y0, y1, y2 = y[f0], y[f], y[f1]
        if not (y1 >= y0 and y1 >= y2):
            continue   # not a local low point in space (the ball's lowest point, largest pixel y)
        denom = y0 - 2 * y1 + y2
        dt = float(np.clip(0.5 * (y0 - y2) / denom, -1, 1)) if denom else 0.0
        cf = f + dt
        size = d[f] if np.isfinite(d[f]) else np.nanmedian([d[g] for g in frames if np.isfinite(d[g])])
        if not np.isfinite(size) or size <= 0:
            continue
        wrists = kp[f, [J["left_wrist"], J["right_wrist"]], :2]
        if not np.isfinite(wrists).any():
            continue
        dist = np.hypot(wrists[:, 0] - x[f], wrists[:, 1] - y[f])
        near = float(np.nanmin(dist)) / size
        if near > proximity_balls:
            continue
        together = (float(np.hypot(*(wrists[0] - wrists[1]))) / size
                   if np.isfinite(wrists).all() else float("inf"))
        score = max(0.0, 1 - near / proximity_balls) * 0.7 + max(0.0, 1 - min(1.0, together / 3)) * 0.3
        if best is None or score > best["confidence"]:
            best = {"frame": cf, "contact_px_frame": f, "confidence": round(float(score), 2),
                    "near_balls": round(near, 2)}
    if best is None:
        return None
    # A long run of "ball basically at the wrist" around the contact reads as a carry
    # or a tracking snag, not a clean touch; halve confidence rather than discard it.
    f = best["contact_px_frame"]
    run = 0
    for g in range(max(a, f - max_hold_frames - 2), min(b, f + max_hold_frames + 2) + 1):
        if not np.isfinite(x[g]):
            run = 0
            continue
        wrists_g = kp[g, [J["left_wrist"], J["right_wrist"]], :2]
        size_g = d[g] if np.isfinite(d[g]) else best["near_balls"]
        if not np.isfinite(wrists_g).any() or not np.isfinite(size_g) or size_g <= 0:
            continue
        near_g = float(np.nanmin(np.hypot(wrists_g[:, 0] - x[g], wrists_g[:, 1] - y[g]))) / size_g
        run = run + 1 if near_g <= proximity_balls else 0
    if run > max_hold_frames:
        best["confidence"] = round(best["confidence"] * 0.5, 2)
        best["flag"] = "possible prolonged contact (carry or tracking snag)"
    return best


# ── calibration helpers: the player's own body as the ruler ────────────────────

def standing_reference(kp: np.ndarray, frame: int, fps: float, *, lookback_seconds: float = 2.0
                       ) -> tuple[float, float] | None:
    """``(ankle y px, shoulder-to-ankle px)`` from the window before *frame*, the same
    percentile-90 technique the basketball demo uses so a bent-knee gather doesn't
    read as shorter (and here, as closer)."""
    lo = max(0, frame - int(lookback_seconds * fps))
    seg = kp[lo:frame + 1]
    ankles = np.nanmean(seg[:, [J["left_ankle"], J["right_ankle"]]], axis=1)
    shoulders = np.nanmean(seg[:, [J["left_shoulder"], J["right_shoulder"]]], axis=1)
    size = np.linalg.norm(ankles - shoulders, axis=1)
    if not np.isfinite(size).any() or not np.isfinite(ankles[:, 1]).any():
        return None
    body_px = float(np.nanpercentile(size[np.isfinite(size)], 90))
    ankle_y = float(np.nanmedian(ankles[np.isfinite(ankles[:, 1]), 1]))
    return ankle_y, body_px


def session_ideal_reach(kp: np.ndarray, fps: float, z_m: float, frame_width: int, hfov_degrees: float,
                        *, min_elbow_deg: float = 160.0, frames: range | None = None) -> float | None:
    """The highest the player's wrist is ever seen with either arm near fully extended
    overhead, converted to a height above the floor. This stands in for "the ideal
    contact height": the player's own best reach, not a league-average number."""
    frames = frames or range(len(kp))
    best = None
    for f in frames:
        ref = standing_reference(kp, f, fps, lookback_seconds=2.0)
        if ref is None:
            continue
        ankle_y, _ = ref
        for side in ("left", "right"):
            angle = elbow_angle(kp[f:f + 1], side)[0]
            if not np.isfinite(angle) or angle < min_elbow_deg:
                continue
            wrist_y = kp[f, J[f"{side}_wrist"], 1]
            if not np.isfinite(wrist_y):
                continue
            h = court.height_m(wrist_y, ankle_y, z_m, frame_width, hfov_degrees)
            if best is None or h > best:
                best = h
    return best


# ── one set ──────────────────────────────────────────────────────────────────────

@dataclass
class Set:
    number: int
    window: tuple[int, int]
    contact_frame: float
    confidence: float
    reason: str = ""
    side_on: bool = True
    arm_extended_deg: float = float("nan")
    jumped: bool = False
    jump_height_m: float | None = None
    contact_height_m: float | None = None
    ideal_height_m: float | None = None
    height_deficit_m: float | None = None
    calibration_source: str = "none"
    calibration_confidence: float = 0.0
    lateral_m: float | None = None          # sideways from the player's own head, at contact
    height_above_head_m: float | None = None
    peak_height_post_m: float | None = None
    exit_angle_deg: float | None = None
    exit_direction: float | None = None     # +1 right / -1 left, as the camera sees it
    path: list[tuple[int, float, float]] = field(default_factory=list)

    @property
    def label(self) -> str:
        if self.height_deficit_m is None:
            return "?"
        return "LOW" if self.height_deficit_m > 0.15 else "GOOD"


def build_set(number: int, window: tuple[int, int], contact: dict, x: np.ndarray, y: np.ndarray,
             d: np.ndarray, kp: np.ndarray, fps: float, calib, *, ideal_height_m: float | None,
             frame_width: int, hfov_degrees: float, side_on_max_ratio: float,
             elbow_min_projected: float, jump_threshold_m: float, head_ref_window_seconds: float
             ) -> Set:
    """Everything measurable about one set, given its contact instant and the depth
    already reconciled by ``calibration.reconcile()``."""
    a, b = window
    f = contact["contact_px_frame"]
    ratio = shoulder_ratio(kp)
    lo, hi = max(a, f - int(0.2 * fps)), min(b, f + int(0.2 * fps))
    side_on = bool(np.nanmedian(ratio[lo:hi + 1]) < side_on_max_ratio) if np.isfinite(ratio[lo:hi + 1]).any() else True

    angles = {side: elbow_angle(kp, side, min_projected=elbow_min_projected)[f] for side in ("left", "right")}
    finite = [v for v in angles.values() if np.isfinite(v)]
    arm_extended = float(np.mean(finite)) if finite else float("nan")

    ref = standing_reference(kp, max(a - int(0.3 * fps), 0), fps, lookback_seconds=1.5)
    ankle_y, jump_height_m, jumped = None, None, False
    if ref is not None:
        ankle_y, _ = ref
        ankles_now = np.nanmean(kp[f, [J["left_ankle"], J["right_ankle"]]], axis=0)
        if np.isfinite(ankles_now[1]) and np.isfinite(calib.z_m):
            jump_height_m = court.px_to_m(ankle_y - ankles_now[1], calib.z_m, frame_width, hfov_degrees)
            jumped = jump_height_m > jump_threshold_m

    contact_height_m = (court.height_m(y[f], ankle_y, calib.z_m, frame_width, hfov_degrees)
                        if ankle_y is not None and np.isfinite(calib.z_m) else None)
    height_deficit_m = (ideal_height_m - contact_height_m
                        if ideal_height_m is not None and contact_height_m is not None else None)

    head_lo = max(0, a - int(head_ref_window_seconds * fps) - int(0.3 * fps))
    head_hi = max(head_lo + 1, a - int(0.3 * fps))
    head = np.nanmedian(kp[head_lo:head_hi, J["nose"]], axis=0) if head_hi > head_lo else np.array([np.nan, np.nan])
    lateral_m = height_above_head_m = None
    if np.isfinite(head).all() and np.isfinite(calib.z_m):
        lateral_m = court.px_to_m(x[f] - head[0], calib.z_m, frame_width, hfov_degrees)
        height_above_head_m = court.px_to_m(head[1] - y[f], calib.z_m, frame_width, hfov_degrees)

    path = [(g, float(x[g]), float(y[g])) for g in range(a, b + 1) if np.isfinite(x[g])]
    post = [(g, px, py) for g, px, py in path if g >= f]
    peak_height_post_m = (max(court.height_m(py, ankle_y, calib.z_m, frame_width, hfov_degrees)
                              for _, _, py in post)
                         if post and ankle_y is not None and np.isfinite(calib.z_m) else None)
    exit_angle_deg = exit_direction = None
    nxt = next(((g, px, py) for g, px, py in post if g > f), None)
    if nxt is not None:
        g1, x1, y1 = nxt
        dx, dy = x1 - x[f], y1 - y[f]   # dy < 0: rising
        exit_angle_deg = float(np.degrees(np.arctan2(abs(dx), max(1e-6, -dy))))
        exit_direction = float(np.sign(dx))

    reason = contact.get("flag", "")
    return Set(number=number, window=window, contact_frame=contact["frame"],
              confidence=contact["confidence"], reason=reason, side_on=side_on,
              arm_extended_deg=arm_extended, jumped=jumped, jump_height_m=jump_height_m,
              contact_height_m=contact_height_m, ideal_height_m=ideal_height_m,
              height_deficit_m=height_deficit_m, calibration_source=calib.source,
              calibration_confidence=calib.confidence, lateral_m=lateral_m,
              height_above_head_m=height_above_head_m, peak_height_post_m=peak_height_post_m,
              exit_angle_deg=exit_angle_deg, exit_direction=exit_direction, path=path)
