"""Track a volleyball session: every set, its contact height against the player's
own ideal reach, and how consistent the contact point and trajectory are.

    python main.py                         # every clip in data/input/
    python main.py data/input/session.mov  # one clip
    python main.py -h                      # every flag

ViTPose finds the candidate windows; SAM 3.1 tracks the ball inside each one, and
the contact instant is found from the ball's own path, not the pose alone — a set,
a block and an attack all raise the hands the same way, but only a set's ball
reverses sharply right at the hands. See VOLLEYBALL_ADAPTATION_PLAN.md (in the
basketball/ folder next to this one) for the full design and its open questions.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from rich.console import Console
from rich.panel import Panel as RichPanel
from rich.table import Table

import config as cfg
from src import ball as ball_mod
from src import calibration, consistency, court, export, net as net_mod, render
from src import sets as sets_mod
from src import video
from src.env import load_api_key

console = Console()


def rule(step: int, title: str) -> None:
    console.rule(f"[bold cyan]{step}[/] {title}", align="left")


def kv(pairs: dict, title: str | None = None) -> None:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim", justify="right")
    table.add_column()
    for key, value in pairs.items():
        table.add_row(key, str(value))
    console.print(RichPanel(table, title=title, title_align="left", expand=False) if title else table)


def rel(path: Path) -> Path:
    try:
        return path.relative_to(cfg.PROJECT_DIR)
    except ValueError:
        return path


def stamp_now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def client(api_key: str):
    from openai import OpenAI

    return OpenAI(base_url=cfg.GATEWAY_BASE_URL, api_key=api_key,
                  timeout=cfg.REQUEST_TIMEOUT, max_retries=1)


def new_run_dir(kind: str) -> Path:
    stamp = datetime.now().strftime(cfg.RUN_STAMP_FORMAT)
    run_dir = cfg.OUTPUT_DIR / f"{stamp}-{kind}"
    n = 2
    while run_dir.exists():
        run_dir = cfg.OUTPUT_DIR / f"{stamp}-{kind}-{n}"
        n += 1
    run_dir.mkdir(parents=True)
    return run_dir


def cost_of(usage: dict | None) -> float:
    return float((usage or {}).get("cost") or 0)


# ── models ───────────────────────────────────────────────────────────────────

def pose_clip(api_key: str, mp4: Path, info: video.VideoInfo) -> dict:
    from src import pose

    video_b64, extra_body = pose.build_request(mp4, video_fps=cfg.POSE_FPS,
                                               max_frames=info.n_frames, precision=cfg.PRECISION)
    cache = pose.cache_path(mp4, cfg.CACHE_DIR, model=cfg.POSE_MODEL, extra_body=extra_body)
    cached = pose.load_cache(cache) if cfg.REUSE_CACHE else None
    if cached is not None:
        payload, usage, created = cached
        console.print(f"  [green]cache hit[/]: poses from [dim]{created}[/]; no gateway call")
    else:
        with console.status("[cyan]waiting on the gateway[/]; ViTPose on every frame…", spinner="dots"):
            payload, usage = pose.request_poses(client(api_key), model=cfg.POSE_MODEL,
                                                video_b64=video_b64, extra_body=extra_body)
        pose.save_cache(cache, payload, usage, stamp=stamp_now())
    frames, by_index = pose.unwrap(payload)
    dropped = pose.dedupe(frames, max_persons=cfg.MAX_PERSONS, iou=cfg.PERSON_IOU_THRESHOLD)
    return {"by_index": {f["index"]: f["persons"] for f in frames}, "dropped": dropped,
            "cost": 0.0 if cached is not None else cost_of(usage)}


def find_net(api_key: str, mp4: Path, info: video.VideoInfo) -> tuple[net_mod.Net | None, float]:
    """SAM 3.1 on one frame. Unlike the hoop, a miss here is not an error — the
    player's own height still calibrates the run."""
    seconds = cfg.NET_FRAME_SECONDS
    frame = info.n_frames // 2 if seconds is None else min(info.n_frames - 1, int(seconds * info.fps))
    cache = net_mod.cache_path(mp4, cfg.CACHE_DIR, model=cfg.SAM_MODEL, prompt=cfg.NET_PROMPT, frame=frame)
    blob = json.loads(cache.read_text()) if cfg.REUSE_CACHE and cache.is_file() else None
    cost = 0.0
    try:
        if blob is None:
            ok, png = cv2.imencode(".png", net_mod.read_frame(mp4, frame))
            with console.status("[cyan]waiting on the gateway[/]; SAM is looking for the net…", spinner="dots"):
                payload, usage = net_mod.request_segment(client(api_key), model=cfg.SAM_MODEL,
                                                          image_png=png.tobytes(), prompt=cfg.NET_PROMPT)
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps({"created": stamp_now(), "payload": payload, "usage": usage}))
            cost = cost_of(usage)
        else:
            payload = blob["payload"]
        return net_mod.locate(payload, frame=frame, width=info.width, height=info.height), cost
    except Exception as exc:
        console.print(f"  [yellow]net not found[/] ({exc}); calibrating from player height only")
        return None, cost


def track_window(api_key: str, mp4: Path, info: video.VideoInfo, start: int, length: int
                 ) -> tuple[dict, float]:
    cache = ball_mod.cache_path(mp4, cfg.CACHE_DIR, model=cfg.SAM_MODEL, prompt=cfg.BALL_PROMPT,
                                start=start, length=length, stride=cfg.BALL_STRIDE)
    blob = json.loads(cache.read_text()) if cfg.REUSE_CACHE and cache.is_file() else None
    cost = 0.0
    if blob is None:
        seg = cfg.CACHE_DIR / f"{mp4.stem}.win{start}_{length}.mp4"
        if not seg.is_file():
            video.segment(mp4, seg, start_frame=start, n_frames=length, fps=info.fps)
        payload, usage = ball_mod.request_track(
            client(api_key), model=cfg.SAM_MODEL,
            video_b64=base64.b64encode(seg.read_bytes()).decode("ascii"),
            prompt=cfg.BALL_PROMPT, stride=cfg.BALL_STRIDE)
        cache.write_text(json.dumps({"created": stamp_now(), "payload": payload, "usage": usage}))
        seg.unlink(missing_ok=True)
        cost = cost_of(usage)
    else:
        payload = blob["payload"]
    return ball_mod.detections(payload, frame_offset=start, width=info.width, height=info.height,
                               min_score=cfg.BALL_MIN_SCORE), cost


def track_contact(api_key: str, mp4: Path, info: video.VideoInfo, window: tuple[int, int],
                  kp: np.ndarray) -> tuple[ball_mod.BallTrack, dict | None, float, int]:
    """SAM-track the ball inside *window*, extending it if no contact is found yet,
    up to ``BALL_MAX_EXTENSIONS`` times."""
    a, b = window
    fps = info.fps
    cap = ball_mod.TRACK_MAX_FRAMES * cfg.BALL_STRIDE
    raw: dict[int, list] = {}
    windows: list[tuple[int, int]] = []
    cost, calls = 0.0, 0
    start, end = a, min(info.n_frames, b + 1)
    track, contact = None, None
    for _ in range(cfg.BALL_MAX_EXTENSIONS + 1):
        length = min(end - start, cap)
        if length <= 0:
            break
        found, c = track_window(api_key, mp4, info, start, length)
        raw.update(found)
        windows.append((start, length))
        cost += c
        calls += 1
        track = ball_mod.select(raw, info.n_frames, windows, size_tolerance=cfg.BALL_SIZE_TOLERANCE)
        contact = sets_mod.find_contact(a, min(b, start + length - 1), track.x, track.y, track.d, kp,
                                        proximity_balls=cfg.CONTACT_PROXIMITY_BALLS,
                                        max_hold_frames=cfg.CONTACT_MAX_HOLD_FRAMES)
        if contact is not None:
            break
        start, end = start + length, min(info.n_frames, start + length + int(cfg.BALL_EXTEND_SECONDS * fps))
    track = track or ball_mod.select(raw, info.n_frames, windows, size_tolerance=cfg.BALL_SIZE_TOLERANCE)
    track.shown = ball_mod.select(raw, info.n_frames, windows, local_size=True,
                                  size_tolerance=cfg.BALL_DRAW_SIZE_TOLERANCE)
    return track, contact, cost, calls


# ── the pipeline ─────────────────────────────────────────────────────────────

def run_clip(api_key: str) -> dict:
    t_start = time.perf_counter()

    rule(2, "Source video")
    src_info = video.probe_source(cfg.INPUT_VIDEO)
    kv({"path": rel(cfg.INPUT_VIDEO), "size": f"{cfg.INPUT_VIDEO.stat().st_size / 1e6:.1f} MB",
        "stored": f"{src_info['width']}x{src_info['height']}",
        "frames": f"{src_info['n_frames']} ({src_info['duration']:.1f}s)"})

    rule(3, "Convert to MP4")
    tonemap, reason = video.tonemap_filter(src_info, mode=cfg.TONEMAP, algorithm=cfg.TONEMAP_ALGORITHM)
    console.print(f"  tone-map: {'[green]on[/] — ' if tonemap else ''}{reason}")
    mp4 = video.cache_path(cfg.INPUT_VIDEO, cfg.CACHE_DIR, target_height=cfg.INFERENCE_HEIGHT,
                           trim_seconds=cfg.TRIM_SECONDS, crf=cfg.CONVERT_CRF, tonemap=tonemap)
    if mp4.is_file() and not cfg.FORCE_RECONVERT:
        console.print(f"  [green]cache hit[/]: [dim]{rel(mp4)}[/]")
    else:
        video.convert(cfg.INPUT_VIDEO, mp4, target_height=cfg.INFERENCE_HEIGHT,
                      trim_seconds=cfg.TRIM_SECONDS, crf=cfg.CONVERT_CRF,
                      total_frames=src_info["n_frames"], console=console, tonemap=tonemap)
    info = video.inspect(mp4)
    console.print(f"  [bold]{info}[/]  ({mp4.stat().st_size / 1e6:.1f} MB)")

    rule(4, "Pose (ViTPose)")
    posed = pose_clip(api_key, mp4, info)
    kp = sets_mod.keypoints(posed["by_index"], info.n_frames, info.width, info.height)
    windows = sets_mod.candidates_coarse(kp, info.fps, armed_above=cfg.SET_ARMED_ABOVE,
                                        smooth_frames=cfg.SET_SMOOTH_FRAMES,
                                        min_gap_seconds=cfg.SET_MIN_GAP_SECONDS,
                                        pad_seconds=cfg.SET_WINDOW_PAD_SECONDS)
    kv({"model": cfg.POSE_MODEL, "candidate windows": len(windows), "cost": f"${posed['cost']:.4f}"},
       title="pose (layer 1: loose candidates)")

    rule(5, "Net (SAM 3.1, optional)")
    net_obj, net_cost = find_net(api_key, mp4, info)
    if net_obj is not None:
        console.print(f"  net found: top at y={net_obj.top:.0f}px, width={net_obj.width:.0f}px "
                      f"(score {net_obj.score:.2f})")
    else:
        console.print("  [yellow]no net found or not visible[/]: calibrating from player height only")

    rule(6, "Ball + contact (SAM 3.1, layer 2)")
    raw_results = []
    ball_cost, ball_calls = 0.0, 0
    for n, window in enumerate(windows, 1):
        track, contact, cost, calls = track_contact(api_key, mp4, info, window, kp)
        ball_cost += cost
        ball_calls += calls
        if contact is None:
            console.print(f"  window {n}/{len(windows)} "
                          f"({window[0] / info.fps:.1f}-{window[1] / info.fps:.1f}s): "
                          "[dim]no contact found, discarded[/]")
            continue
        f = contact["contact_px_frame"]
        ref = sets_mod.standing_reference(kp, max(window[0] - int(0.3 * info.fps), 0), info.fps)
        player_z = (calibration.from_player(ref[1], cfg.PLAYER_HEIGHT_M, info.width, cfg.CAMERA_HFOV_DEGREES)
                   if ref is not None else None)
        d_vals = [track.d[g] for g in range(max(0, f - 5), f) if np.isfinite(track.d[g])]
        ball_z = (calibration.from_ball(float(np.median(d_vals)), cfg.BALL_DIAMETER_M, info.width,
                                        cfg.CAMERA_HFOV_DEGREES) if d_vals else None)
        net_z = (calibration.from_net(net_obj.width, cfg.NET_WIDTH_M, info.width, cfg.CAMERA_HFOV_DEGREES)
                if net_obj is not None else None)
        calib = calibration.reconcile(player_z=player_z, ball_z=ball_z, net_z=net_z)
        raw_results.append({"number": len(raw_results) + 1, "window": window, "contact": contact,
                            "track": track, "calib": calib, "ankle_y": ref[0] if ref else None})
        console.print(f"  window {n}/{len(windows)}: contact at {f / info.fps:.2f}s, "
                      f"confidence {contact['confidence']:.2f}, z={calib.z_m:.2f}m ({calib.source})")
    kv({"windows tried": len(windows), "contacts found": len(raw_results),
        "SAM calls": ball_calls, "cost": f"${net_cost + ball_cost:.4f}"}, title="SAM 3.1")

    rule(7, "Measuring: height, form, consistency")
    z_values = [r["calib"].z_m for r in raw_results if np.isfinite(r["calib"].z_m)]
    z_med = float(np.median(z_values)) if z_values else float("nan")
    ideal_height_m = (sets_mod.session_ideal_reach(kp, info.fps, z_med, info.width, cfg.CAMERA_HFOV_DEGREES)
                      if np.isfinite(z_med) else None)
    vsets = []
    ankle_y_by_set, calib_by_set = {}, {}
    for r in raw_results:
        s = sets_mod.build_set(r["number"], r["window"], r["contact"], r["track"].x, r["track"].y,
                               r["track"].d, kp, info.fps, r["calib"], ideal_height_m=ideal_height_m,
                               frame_width=info.width, hfov_degrees=cfg.CAMERA_HFOV_DEGREES,
                               side_on_max_ratio=cfg.SIDE_ON_MAX_SHOULDER_RATIO,
                               elbow_min_projected=cfg.ARM_ELBOW_MIN_PROJECTED,
                               jump_threshold_m=cfg.JUMP_THRESHOLD_M,
                               head_ref_window_seconds=cfg.HEAD_REF_WINDOW_SECONDS)
        vsets.append(s)
        ankle_y_by_set[s.number], calib_by_set[s.number] = r["ankle_y"], r["calib"]

    table = Table(box=None, pad_edge=False)
    for col in ("set", "at", "conf", "height (m)", "ideal", "deficit", "jumped", "lateral (m)", "arm"):
        table.add_column(col, justify="right" if col != "set" else "left")
    for s in vsets:
        table.add_row(str(s.number), f"{s.contact_frame / info.fps:.1f}s", f"{s.confidence:.2f}",
                      f"{s.contact_height_m:.2f}" if s.contact_height_m is not None else "—",
                      f"{s.ideal_height_m:.2f}" if s.ideal_height_m is not None else "—",
                      f"{s.height_deficit_m:+.2f}" if s.height_deficit_m is not None else "—",
                      "yes" if s.jumped else "no",
                      f"{s.lateral_m:+.2f}" if s.lateral_m is not None else "—",
                      f"{s.arm_extended_deg:.0f}°" if np.isfinite(s.arm_extended_deg) else "—")
    console.print(RichPanel(table, title=f"[bold]{len(vsets)}[/] sets", title_align="left", expand=False))

    summary = consistency.summarize(vsets, min_confidence=cfg.CONTACT_MIN_CONFIDENCE)
    kv({"contact height": f"{summary.contact_height_mean_m:.2f}m ± {summary.contact_height_std_m:.2f}m"
                          if summary.contact_height_mean_m is not None else "—",
        "below ideal reach": f"{summary.height_deficit_mean_m:+.2f}m ± {summary.height_deficit_std_m:.2f}m"
                             if summary.height_deficit_mean_m is not None else "—",
        "sideways spread": f"± {summary.lateral_std_m:.2f}m" if summary.lateral_std_m is not None else "—",
        "jump rate": f"{summary.jump_rate:.0%}" if summary.jump_rate is not None else "—"},
       title="session consistency")

    rule(8, "Render")
    run_dir = new_run_dir("set-consistency")
    stamp = run_dir.name
    canvas = info
    clip_h = min(int(cfg.EXPORT_HEIGHT) // 2, src_info["height"]) // 2 * 2
    if clip_h > info.height:
        big = video.cache_path(cfg.INPUT_VIDEO, cfg.CACHE_DIR, target_height=clip_h,
                               trim_seconds=cfg.TRIM_SECONDS, crf=cfg.CONVERT_CRF, tonemap=tonemap)
        if not big.is_file() or cfg.FORCE_RECONVERT:
            video.convert(cfg.INPUT_VIDEO, big, target_height=clip_h, trim_seconds=cfg.TRIM_SECONDS,
                          crf=cfg.CONVERT_CRF, total_frames=src_info["n_frames"], console=console,
                          tonemap=tonemap)
        canvas = video.inspect(big)
    raw_video = run_dir / "_raw.mp4"
    t0 = time.perf_counter()
    merged = _merge_tracks([r["track"] for r in raw_results], info.n_frames)
    stats = render.render(info, canvas=canvas, kp=kp, track=merged, vsets=vsets, dst=raw_video, cfg=cfg,
                          console=console)
    render_seconds = time.perf_counter() - t0
    out_video = run_dir / f"{cfg.INPUT_VIDEO.stem}_volleyball.mp4"
    video.encode_h264(raw_video, out_video, crf=cfg.OUTPUT_CRF)
    raw_video.unlink(missing_ok=True)
    console.print(f"  video  -> [dim]{rel(out_video)}[/] ({out_video.stat().st_size / 1e6:.1f} MB)")

    plots_png = run_dir / "plots.png"
    render.plot_trajectories(vsets, info.fps, ankle_y_by_set, calib_by_set, info.width,
                             cfg.CAMERA_HFOV_DEGREES, plots_png, cfg=cfg)
    console.print(f"  graph  -> [dim]{rel(plots_png)}[/]")
    sequence_png = run_dir / "sequence.png"
    render.plot_sequence(vsets, sequence_png, cfg=cfg)
    console.print(f"  graph  -> [dim]{rel(sequence_png)}[/]")
    arm_png = run_dir / "arm.png"
    angle_left = sets_mod.elbow_angle(kp, "left")
    angle_right = sets_mod.elbow_angle(kp, "right")
    render.plot_arm(vsets, angle_left, angle_right, info.fps, arm_png, cfg=cfg)
    console.print(f"  graph  -> [dim]{rel(arm_png)}[/]")

    for path in export.write(vsets, merged, info, run_dir, cfg.INPUT_VIDEO.stem, summary):
        console.print(f"  data   -> [dim]{rel(path)}[/]")

    total = time.perf_counter() - t_start
    cost = posed["cost"] + net_cost + ball_cost
    (run_dir / "run.json").write_text(json.dumps({
        "stamp": stamp, "total_seconds": round(total, 2), "input": {"path": str(cfg.INPUT_VIDEO)},
        "pose": {"model": cfg.POSE_MODEL},
        "net": None if net_obj is None else {"top": round(net_obj.top, 1), "left": round(net_obj.left, 1),
                                             "right": round(net_obj.right, 1), "score": round(net_obj.score, 2)},
        "ball": {"model": cfg.SAM_MODEL, "prompt": cfg.BALL_PROMPT, "calls": ball_calls},
        "court": {"hfov_degrees": cfg.CAMERA_HFOV_DEGREES, "player_height_m": cfg.PLAYER_HEIGHT_M,
                  "net_height_m": cfg.NET_HEIGHT_M},
        "ideal_height_m": ideal_height_m, "n_sets": len(vsets),
        "cost_usd": {"pose": round(posed["cost"], 6), "net": round(net_cost, 6),
                     "ball": round(ball_cost, 6), "total": round(cost, 6)},
        "render": {**stats, "seconds": round(render_seconds, 2)},
    }, indent=2))

    console.print()
    kv({"pose": f"${posed['cost']:.4f}", "net": f"${net_cost:.4f}", "ball": f"${ball_cost:.4f}",
        "total": f"[bold]${cost:.4f}[/]"}, title="cost")
    console.rule(f"[bold green]done[/] in {total:.1f}s", align="left")
    console.print(f"  [bold]{rel(run_dir)}/[/]\n")
    return {"sets": len(vsets), "cost": cost, "seconds": total, "output": run_dir}


def _merge_tracks(tracks: list[ball_mod.BallTrack], n_frames: int) -> ball_mod.BallTrack:
    """One track covering every set's window, for the video and the frame CSV."""
    if not tracks:
        return ball_mod.select({}, n_frames, [], size_tolerance=0.35)
    x, y, d = (np.full(n_frames, np.nan) for _ in range(3))
    covered = np.zeros(n_frames, bool)
    windows: list[tuple[int, int]] = []
    diameters = []
    for t in tracks:
        ok = np.isfinite(t.x)
        x[ok], y[ok], d[ok] = t.x[ok], t.y[ok], t.d[ok]
        covered |= t.covered
        windows += t.windows
        if np.isfinite(t.diameter):
            diameters.append(t.diameter)
    merged = ball_mod.BallTrack(x, y, d, covered, windows, float(np.median(diameters)) if diameters else float("nan"))
    shown_x, shown_y, shown_d = x.copy(), y.copy(), d.copy()
    merged.shown = ball_mod.BallTrack(shown_x, shown_y, shown_d, covered, windows, merged.diameter)
    return merged


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track a volleyball session: every set's contact height against the "
                    "player's own ideal reach, and how consistent it is. Flags override "
                    "config.py for this run only.")
    parser.add_argument("inputs", nargs="*", type=Path, metavar="CLIP|DIR",
                        help=f"clips, or folders of clips, to run (config: {rel(cfg.INPUT)})")
    parser.add_argument("--height", type=int, metavar="PX",
                        help=f"the output's total height (config: {cfg.EXPORT_HEIGHT})")
    parser.add_argument("--trim", type=float, metavar="SECONDS",
                        help="use only the first SECONDS of each clip, for a quick test")
    parser.add_argument("--player-height", type=float, metavar="M",
                        help=f"the player's height in meters (config: {cfg.PLAYER_HEIGHT_M:g})")
    parser.add_argument("--fov", type=float, metavar="DEG",
                        help=f"the camera's horizontal field of view (config: {cfg.CAMERA_HFOV_DEGREES:g})")
    parser.add_argument("--net-height", type=float, metavar="M",
                        help=f"net height: 2.43 men's, 2.24 women's, 2.35 mixed (config: {cfg.NET_HEIGHT_M:g})")
    parser.add_argument("--fresh", action="store_true", help="ignore every cached gateway reply")
    return parser.parse_args(argv)


def apply_args(args: argparse.Namespace) -> None:
    if args.height:
        cfg.EXPORT_HEIGHT = args.height
    if args.trim:
        cfg.TRIM_SECONDS = args.trim
    if args.player_height:
        cfg.PLAYER_HEIGHT_M = args.player_height
    if args.fov:
        cfg.CAMERA_HFOV_DEGREES = args.fov
    if args.net_height:
        cfg.NET_HEIGHT_M = args.net_height
    if args.fresh:
        cfg.REUSE_CACHE = False


def find_clips(inputs: list[Path]) -> list[Path]:
    clips: list[Path] = []
    for item in inputs:
        path = item.expanduser().resolve()
        if path.is_dir():
            clips += sorted(p for p in path.iterdir()
                            if p.is_file() and not p.name.startswith(".")
                            and p.suffix.lower() in cfg.VIDEO_EXTENSIONS)
        elif path.is_file():
            clips.append(path)
        else:
            console.print(f"[red]Not found:[/] {item}")
    return list(dict.fromkeys(clips))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    apply_args(args)
    clips = find_clips(args.inputs or [cfg.INPUT])
    console.print()
    console.rule("[bold]Volleyball: set height and consistency[/]", align="center")
    if not clips:
        console.print(f"[red]No clips to run.[/] Put a setting session video in "
                      f"{rel(cfg.INPUT)}/, or pass one: python main.py path/to/clip.mov")
        return 1

    rule(1, "API key")
    api_key, source = load_api_key(cfg.PROJECT_DIR)
    console.print(f"  key loaded from [green]{source}[/]")

    results: dict[str, dict | str] = {}
    for n, clip in enumerate(clips, 1):
        if len(clips) > 1:
            console.print()
            console.rule(f"[bold magenta]clip {n}/{len(clips)}[/] {clip.name}", align="left")
        cfg.INPUT_VIDEO = clip
        try:
            results[clip.name] = run_clip(api_key)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            console.print(f"\n[red]error on {clip.name}:[/] {exc}")
            results[clip.name] = str(exc).splitlines()[0][:100]

    if len(clips) > 1:
        table = Table(title="batch", box=None, pad_edge=False, title_justify="left")
        for col in ("clip", "sets", "cost", "output"):
            table.add_column(col)
        for name, r in results.items():
            if isinstance(r, str):
                table.add_row(name, "[red]failed[/]", "", "")
            else:
                table.add_row(name, str(r["sets"]), f"${r['cost']:.3f}", str(rel(r["output"])))
        console.print(table)
    return 0 if all(not isinstance(r, str) for r in results.values()) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]interrupted[/]")
        sys.exit(130)
    except Exception as exc:
        console.print(f"\n[red]error:[/] {exc}")
        sys.exit(1)
