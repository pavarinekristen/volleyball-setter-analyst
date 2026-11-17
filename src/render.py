"""The export video (clip on top, consistency panel below), and plots.png / sequence.png."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn

from .ball import smoothed
from .skeleton import SKELETON, FACE_KPTS, edge_part

_PROGRESS_COLUMNS = (
    TextColumn("[cyan]rendering[/]"), BarColumn(), TaskProgressColumn(),
    TextColumn("{task.completed}/{task.total} frames"), TimeElapsedColumn(),
)

BG = (10, 10, 10)
FG = (247, 244, 242)       # BGR of #f2f4f7
MUTED = (153, 145, 139)    # BGR of #8b9199
GOOD_BGR = (102, 207, 81)  # #51cf66
BAD_BGR = (107, 107, 255)  # #ff6b6b
BALL_BGR = (43, 146, 255)  # #ff922b


def _p(v) -> tuple[int, int]:
    return int(round(v[0])), int(round(v[1]))


def _draw_skeleton(frame, pts: np.ndarray, *, colors, thick: int, radius: int) -> None:
    ok = np.isfinite(pts).all(axis=1)
    for a, b, _ in SKELETON:
        if a in FACE_KPTS or b in FACE_KPTS or not (ok[a] and ok[b]):
            continue
        cv2.line(frame, _p(pts[a]), _p(pts[b]), colors[edge_part(a, b)], thick, cv2.LINE_AA)
    for i in range(len(pts)):
        if ok[i] and i not in FACE_KPTS:
            cv2.circle(frame, _p(pts[i]), radius, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, _p(pts[i]), radius + 1, (20, 22, 26), 1, cv2.LINE_AA)


def draw_ball(frame, path, at, lo: int, k: int, r: int, thick: int) -> None:
    for (f0, x0, y0), (f1, x1, y1) in zip(path, path[1:]):
        if f1 - f0 > 6:
            continue
        fade = 0.25 + 0.75 * (f1 - lo) / max(1, k - lo)
        color = tuple(int(c * fade) for c in BALL_BGR)
        cv2.line(frame, (int(x0), int(y0)), (int(x1), int(y1)), color,
                 max(1, int(thick * fade)), cv2.LINE_AA)
    c = (int(at[0]), int(at[1]))
    cv2.circle(frame, c, r + 2, (20, 22, 26), 1, cv2.LINE_AA)
    cv2.circle(frame, c, r, BALL_BGR, max(2, thick - 1), cv2.LINE_AA)


def quality_color(deficit_m: float | None, good_m: float, bad_m: float) -> tuple[int, int, int]:
    """Green near the player's own ideal reach, red far below it; gray if unscored."""
    if deficit_m is None or not np.isfinite(deficit_m):
        return (140, 140, 140)
    t = float(np.clip((abs(deficit_m) - good_m) / max(1e-6, bad_m - good_m), 0, 1))
    good, bad = np.array(GOOD_BGR, float), np.array(BAD_BGR, float)
    return tuple(int(v) for v in (good * (1 - t) + bad * t))


def draw_contact(frame, center, r: int, color, thick: int) -> None:
    cv2.circle(frame, _p(center), r + int(2.5 * thick), color, thick, cv2.LINE_AA)
    cv2.circle(frame, _p(center), r, color, -1, cv2.LINE_AA)


class ClipOverlay:
    """The pose, the ball with its comet tail, and a marker lighting up at each contact,
    colored by how close it was to the player's own ideal reach."""

    def __init__(self, *, kp, track, vsets, n_frames: int, fps: float, sx: float, width: int, cfg):
        self.kp, self.sets, self.fps, self.sx, self.cfg = kp, vsets, fps, sx, cfg
        scale = width / 720
        self.thick = max(1, round(cfg.LINE_THICKNESS * scale))
        self.radius = max(1, round(cfg.POINT_RADIUS * scale))
        self.marker_frames = int(cfg.MARKER_SECONDS * fps)
        self.trail = int(cfg.BALL_TRAIL_SECONDS * fps)
        self.bx, self.by = smoothed(track, max_gap=int(cfg.BALL_FILL_SECONDS * fps))
        self.ball_r = max(3, int(track.diameter * sx / 2)) if np.isfinite(track.diameter) else 6

    def draw(self, frame, k: int) -> bool:
        sx, thick = self.sx, self.thick
        bx, by = self.bx, self.by
        pts = self.kp[k] * sx
        if np.isfinite(pts).any():
            _draw_skeleton(frame, pts, colors=self.cfg.SKELETON_COLORS, thick=thick, radius=self.radius)

        window = next((s for s in self.sets if s.window[0] <= k <= s.window[1]), None)
        drawn = False
        if window is not None and np.isfinite(bx[k]):
            lo = max(window.window[0], k - self.trail)
            path = [(f, bx[f] * sx, by[f] * sx) for f in range(lo, k + 1) if np.isfinite(bx[f])]
            draw_ball(frame, path, (bx[k] * sx, by[k] * sx), lo, k, self.ball_r, thick)
            drawn = True

        marker = next((s for s in self.sets
                       if 0 <= k - s.contact_frame <= self.marker_frames), None)
        if marker is not None and np.isfinite(bx[k]):
            color = quality_color(marker.height_deficit_m, self.cfg.HEIGHT_DEFICIT_GOOD_M,
                                  self.cfg.HEIGHT_DEFICIT_BAD_M)
            fade = 1 - (k - marker.contact_frame) / max(1, self.marker_frames)
            color = tuple(int(c * (0.4 + 0.6 * fade)) for c in color)
            draw_contact(frame, (bx[k] * sx, by[k] * sx), self.ball_r, color, thick)
        return drawn


class ConsistencyPanel:
    """Below the clip: the contact point scattered around the player's own head
    (lateral vs. height above head, in meters), colored by closeness to their own
    ideal reach, plus the running stats."""

    def __init__(self, vsets, width: int, height: int, *, cfg):
        self.sets, self.cfg = vsets, cfg
        self.w, self.h = width, height
        lateral = [abs(s.lateral_m) for s in vsets if s.lateral_m is not None]
        vertical = [abs(s.height_above_head_m) for s in vsets if s.height_above_head_m is not None]
        span = max([0.3] + lateral + vertical) * 1.4
        self.scale = min(width, height) * 0.42 / span
        self.cx, self.cy = width // 2, height // 2 + int(0.08 * height)
        base = np.zeros((height, width, 3), np.uint8)
        base[:] = BG
        for r_m in (0.1, 0.2, 0.3):
            cv2.circle(base, (self.cx, self.cy), int(r_m * self.scale), (40, 40, 40), 1, cv2.LINE_AA)
        cv2.line(base, (0, self.cy), (width, self.cy), (30, 30, 30), 1, cv2.LINE_AA)
        cv2.line(base, (self.cx, 0), (self.cx, height), (30, 30, 30), 1, cv2.LINE_AA)
        cv2.drawMarker(base, (self.cx, self.cy), FG, cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)
        cv2.putText(base, "contact point, relative to the player's own head", (16, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, MUTED, 1, cv2.LINE_AA)
        self.base = base

    def _decided(self, k: int) -> list:
        return [s for s in self.sets if s.contact_frame <= k]

    def render(self, k: int) -> np.ndarray:
        panel = self.base.copy()
        done = self._decided(k)
        for s in done:
            if s.lateral_m is None or s.height_above_head_m is None:
                continue
            p = (int(self.cx + s.lateral_m * self.scale), int(self.cy - s.height_above_head_m * self.scale))
            color = quality_color(s.height_deficit_m, self.cfg.HEIGHT_DEFICIT_GOOD_M,
                                  self.cfg.HEIGHT_DEFICIT_BAD_M)
            cv2.circle(panel, p, 9, (10, 10, 10), -1, cv2.LINE_AA)
            cv2.circle(panel, p, 7, color, -1, cv2.LINE_AA)

        heights = [s.contact_height_m for s in done if s.contact_height_m is not None]
        lines = [f"sets: {len(done)}/{len(self.sets)}"]
        if heights:
            lines.append(f"contact height: {np.mean(heights):.2f}m  (sd {np.std(heights):.2f}m)")
        deficits = [s.height_deficit_m for s in done if s.height_deficit_m is not None]
        if deficits:
            lines.append(f"avg below ideal reach: {np.mean(deficits):.2f}m")
        jumps = [s.jumped for s in done]
        if jumps:
            lines.append(f"jumped: {sum(jumps)}/{len(jumps)}")
        for i, line in enumerate(lines):
            cv2.putText(panel, line, (self.w - 10 - 320, 24 + i * 24), cv2.FONT_HERSHEY_SIMPLEX,
                       0.55, FG, 1, cv2.LINE_AA)
        return panel


def frame_size(info, canvas, cfg) -> tuple[int, int, int]:
    canvas = canvas or info
    video_h = min(int(cfg.EXPORT_HEIGHT) // 2, canvas.height) // 2 * 2
    return int(round(canvas.width * video_h / canvas.height / 2)) * 2, 2 * video_h, video_h


def render(info, *, kp, track, vsets, dst: Path, cfg, console, canvas=None) -> dict:
    canvas = canvas or info
    video_h = min(int(cfg.EXPORT_HEIGHT) // 2, canvas.height) // 2 * 2
    sx = video_h / info.height
    width = int(round(canvas.width * video_h / canvas.height / 2)) * 2
    out_h = 2 * video_h

    overlay = ClipOverlay(kp=kp, track=track, vsets=vsets, n_frames=info.n_frames, fps=info.fps,
                          sx=sx, width=width, cfg=cfg)
    panel = ConsistencyPanel(vsets, width, out_h - video_h, cfg=cfg)

    cap = cv2.VideoCapture(str(canvas.path))
    writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), info.fps, (width, out_h))
    ball_frames = 0
    with Progress(*_PROGRESS_COLUMNS, console=console, transient=True) as progress:
        task = progress.add_task("render", total=info.n_frames)
        for k in range(info.n_frames):
            ok, src = cap.read()
            if not ok:
                break
            frame = cv2.resize(src, (width, video_h), interpolation=cv2.INTER_AREA)
            ball_frames += overlay.draw(frame, k)
            out = np.vstack([frame, panel.render(k)])
            writer.write(out)
            progress.update(task, completed=k + 1)
    cap.release()
    writer.release()
    return {"width": width, "height": out_h, "video_height": video_h, "ball_frames_drawn": ball_frames}


# ── plots.png, sequence.png, arm.png ────────────────────────────────────────────

def plot_trajectories(vsets, fps: float, ankle_y_by_set: dict, calib_by_set: dict, frame_width: int,
                      hfov_degrees: float, dst: Path, *, cfg) -> None:
    """Left: every set's ball path, aligned on its own contact instant, height above
    the floor in meters. Right: the contact point scattered around the player's own
    head, same axes as the on-video panel."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from . import court

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 6.5), dpi=110)
    for s in vsets:
        ankle_y, calib = ankle_y_by_set.get(s.number), calib_by_set.get(s.number)
        if ankle_y is None or calib is None or not np.isfinite(calib.z_m) or not s.path:
            continue
        f, x, y = np.array(s.path).T
        h = [court.height_m(py, ankle_y, calib.z_m, frame_width, hfov_degrees) for py in y]
        t = (f - s.contact_frame) / fps
        color = quality_color(s.height_deficit_m, cfg.HEIGHT_DEFICIT_GOOD_M, cfg.HEIGHT_DEFICIT_BAD_M)
        color = tuple(c / 255 for c in color[::-1])
        ax.plot(t, h, "-", color=color, lw=1.6, alpha=0.85)
        ax.annotate(str(s.number), (t[-1], h[-1]), fontsize=8, color=color)
    ax.axvline(0, color="#868e96", lw=1, ls="--")
    if any(s.ideal_height_m is not None for s in vsets):
        ideal = next(s.ideal_height_m for s in vsets if s.ideal_height_m is not None)
        ax.axhline(ideal, color="#1c7ed6", lw=1, ls=":", label=f"player's ideal reach ({ideal:.2f}m)")
        ax.legend(loc="lower right", fontsize=8)
    ax.set_xlabel("seconds from contact")
    ax.set_ylabel("height above floor (m)")
    ax.set_title("Every set's ball path, aligned on contact", loc="left")

    for s in vsets:
        if s.lateral_m is None or s.height_above_head_m is None:
            continue
        color = quality_color(s.height_deficit_m, cfg.HEIGHT_DEFICIT_GOOD_M, cfg.HEIGHT_DEFICIT_BAD_M)
        color = tuple(c / 255 for c in color[::-1])
        ax2.plot(s.lateral_m, s.height_above_head_m, "o", color=color, ms=9)
        ax2.annotate(str(s.number), (s.lateral_m, s.height_above_head_m), fontsize=8,
                    xytext=(5, 5), textcoords="offset points")
    ax2.axhline(0, color="#868e96", lw=0.8)
    ax2.axvline(0, color="#868e96", lw=0.8)
    ax2.set_xlabel("sideways from the head (m)")
    ax2.set_ylabel("above the head (m)")
    ax2.set_title("Contact point, relative to the player", loc="left")
    ax2.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(dst)
    plt.close(fig)


def plot_sequence(vsets, dst: Path, *, cfg) -> None:
    """Contact height and sideways drift, set by set in order: is it drifting through
    the session, or just noisy around one spot?"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 7), dpi=110, sharex=True)
    numbers = [s.number for s in vsets]
    heights = [s.contact_height_m for s in vsets]
    colors = [tuple(c / 255 for c in quality_color(s.height_deficit_m, cfg.HEIGHT_DEFICIT_GOOD_M,
                                                   cfg.HEIGHT_DEFICIT_BAD_M)[::-1]) for s in vsets]
    ax.scatter(numbers, heights, c=colors, s=40, zorder=3)
    finite = [(n, h) for n, h in zip(numbers, heights) if h is not None]
    if finite:
        fn, fh = zip(*finite)
        ax.plot(fn, fh, "-", color="#adb5bd", lw=1, zorder=2)
    if any(s.ideal_height_m is not None for s in vsets):
        ideal = next(s.ideal_height_m for s in vsets if s.ideal_height_m is not None)
        ax.axhline(ideal, color="#1c7ed6", lw=1, ls=":", label=f"ideal reach ({ideal:.2f}m)")
        ax.legend(loc="lower right", fontsize=8)
    ax.set_ylabel("contact height (m)")
    ax.set_title("Contact height through the session", loc="left")

    lateral = [s.lateral_m for s in vsets]
    ax2.scatter(numbers, lateral, c=colors, s=40, zorder=3)
    finite2 = [(n, lv) for n, lv in zip(numbers, lateral) if lv is not None]
    if finite2:
        fn2, fl2 = zip(*finite2)
        ax2.plot(fn2, fl2, "-", color="#adb5bd", lw=1, zorder=2)
    ax2.axhline(0, color="#868e96", lw=0.8)
    ax2.set_xlabel("set number")
    ax2.set_ylabel("sideways from the head (m)")
    ax2.set_title("Sideways drift through the session", loc="left")

    fig.tight_layout()
    fig.savefig(dst)
    plt.close(fig)


def plot_arm(vsets, angle_left: np.ndarray, angle_right: np.ndarray, fps: float, dst: Path, *,
            cfg, before_seconds: float = 0.5, after_seconds: float = 0.3) -> None:
    """The elbow angle of each arm through every set, aligned on contact: are both
    arms reaching full extension at the same moment, every time?"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(12, 5.5), dpi=110, sharey=True)
    lo, hi = int(-before_seconds * fps), int(after_seconds * fps)
    offsets = np.arange(lo, hi + 1)
    for ax, angle, title in ((axl, angle_left, "left elbow"), (axr, angle_right, "right elbow")):
        for s in vsets:
            f = int(round(s.contact_frame))
            idx = f + offsets
            ok = (idx >= 0) & (idx < len(angle))
            curve = np.full(len(idx), np.nan)
            curve[ok] = angle[idx[ok]]
            color = tuple(c / 255 for c in quality_color(s.height_deficit_m, cfg.HEIGHT_DEFICIT_GOOD_M,
                                                         cfg.HEIGHT_DEFICIT_BAD_M)[::-1])
            ax.plot(offsets / fps, curve, color=color, lw=1.4, alpha=0.75)
        ax.axvline(0, color="#868e96", lw=1)
        ax.axhline(180, color="#adb5bd", lw=1, ls="--")
        ax.set_xlabel("seconds from contact")
        ax.set_title(title, loc="left")
    axl.set_ylabel("elbow angle (deg, 180 = straight)")
    fig.suptitle("Arm extension through every set (colored by contact height vs. ideal)")
    fig.tight_layout()
    fig.savefig(dst)
    plt.close(fig)
