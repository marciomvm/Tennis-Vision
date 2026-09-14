"""
tools/segment_points.py
------------------------
Cut the dead time out of a long recording before the real pipeline ever sees it.

Why this exists
----------------
A club session is one continuous take: an hour of tape for maybe fifteen minutes of
rallies, the rest being players walking to the baseline, picking up balls, arguing a
call. Every detector this project runs costs the same per frame whether the frame shows
a rally or an empty court - YOLO, TrackNet and the pose model do not know the difference,
so an hour of raw footage is paid for in full even when most of it is not tennis.

This finds the stretches with motion inside the calibrated court and drops everything
else, cheaply: frame differencing at reduced resolution, no model, no GPU. Measured on
one reference clip at roughly 150 frames/second on a single core - an hour of 30fps
footage costs about 12 minutes to scan, against the ~100 minutes the ball tracker alone
would need to run on the same hour.

What this is NOT
------------------
Not a rally detector, and not one output file per point. A rally is a burst of motion
per stroke with the ball's flight time as a lull in between, so the runs get merged with
a generous gap tolerance - otherwise a single point would shred into several files. This
is a coarse filter: it keeps whole rallies together, including the pauses inside them,
and drops the stretches with no motion at all for several seconds. If exact point
boundaries matter, run the real event detector on what survives here, which is now a
small fraction of the original recording.

Usage
-----
    python tools/segment_points.py session.mp4
    python tools/segment_points.py session.mp4 --calibration calibration/court_A.json
    python tools/segment_points.py session.mp4 --dry-run            # manifest only
    python tools/segment_points.py session.mp4 --plot               # + a debug PNG

Each retained window is written to <out-dir>/<video name>_NNN.mp4 (stream-copied by
default - fast, and may start a little EARLY because a stream copy can only cut at a
keyframe, never late), plus a manifest.json recording exactly what was kept, on what
basis, and the activity score at every cut - so a run that kept too little or too much
is auditable rather than a black box.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.activity_segments import (                        # noqa: E402
    DEFAULT_MIN_DURATION_S,
    DEFAULT_MIN_GAP_S,
    DEFAULT_PAD_S,
    auto_threshold,
    build_roi_mask,
    compute_activity_signal,
    find_segments,
    smooth,
)
from utils.court_calibration import CourtCalibration, find_calibration_for  # noqa: E402


def _resolve_calibration(video: str, explicit: str | None, disabled: bool):
    if disabled:
        return None, "disabled (--no-calibration)"
    if explicit:
        path = Path(explicit)
        if not path.exists():
            print(f"error: calibration not found: {explicit}", file=sys.stderr)
            sys.exit(2)
        return CourtCalibration.load(path), str(path)
    found = find_calibration_for(video)
    if found is None:
        return None, None
    return CourtCalibration.load(found), str(found)


def _format_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


def _progress_printer(total_hint: int, fps: float):
    start = time.time()

    def report(done: int, total: int):
        elapsed = time.time() - start
        rate = done / elapsed if elapsed > 0 else 0.0
        video_s = done / fps if fps else 0.0
        if total:
            eta = (total - done) / rate if rate > 0 else 0.0
            print(f"\r  scanning: {done}/{total} frames "
                  f"({100 * done / total:4.1f}%)  {rate:5.0f} fps  "
                  f"video time {_format_time(video_s)}  ETA {_format_time(eta)}   ",
                  end="", flush=True)
        else:
            print(f"\r  scanning: {done} frames  {rate:5.0f} fps  "
                  f"video time {_format_time(video_s)}   ", end="", flush=True)

    return report


def _save_plot(signal: np.ndarray, fps: float, threshold: float, segments, path: Path):
    """A debug PNG of the activity trace, no plotting library required."""
    width, height = 1600, 400
    pad = 40
    canvas = np.full((height, width, 3), 24, np.uint8)

    if len(signal) == 0:
        cv2.imwrite(str(path), canvas)
        return

    plot_w, plot_h = width - 2 * pad, height - 2 * pad
    y_max = max(float(signal.max()), threshold * 1.2, 1e-6)

    def to_xy(i: int, value: float) -> tuple[int, int]:
        x = pad + int(i / max(len(signal) - 1, 1) * plot_w)
        y = pad + plot_h - int(min(value, y_max) / y_max * plot_h)
        return x, y

    for seg in segments:
        x0, _ = to_xy(seg.start_frame, 0)
        x1, _ = to_xy(min(seg.end_frame, len(signal) - 1), 0)
        cv2.rectangle(canvas, (x0, pad), (x1, height - pad), (40, 80, 40), -1)

    step = max(1, len(signal) // plot_w)
    points = [to_xy(i, float(signal[i])) for i in range(0, len(signal), step)]
    for a, b in zip(points, points[1:]):
        cv2.line(canvas, a, b, (0, 210, 255), 1, cv2.LINE_AA)

    _, ty = to_xy(0, threshold)
    cv2.line(canvas, (pad, ty), (width - pad, ty), (120, 120, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"threshold {threshold:.3f}", (pad + 4, ty - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"activity signal, {len(signal)} frames, {len(segments)} "
                        f"segment(s) kept (green)", (pad, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), canvas)


def _extract_segment(video: str, seg, out_path: Path, reencode: bool) -> tuple[bool, str]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration = max(seg.duration_s, 1.0 / 30)
    if reencode:
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
              "-ss", f"{seg.start_s:.3f}", "-i", video, "-t", f"{duration:.3f}",
              "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
              "-c:a", "aac", str(out_path)]
    else:
        # -ss before -i seeks to the nearest keyframe AT OR BEFORE the requested start,
        # so a stream copy can only start a little EARLY, never late - the safe
        # direction given everything upstream already errs toward keeping too much
        # rather than too little.
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
              "-ss", f"{seg.start_s:.3f}", "-i", video, "-t", f"{duration:.3f}",
              "-c", "copy", str(out_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if not out_path.exists():
        return False, (result.stderr or "unknown ffmpeg error").strip()[:300]
    return True, ""


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="tennis-vision segment",
        description=("Cut the stretches with no motion out of a long recording, so the "
                     "real pipeline never has to process them."),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "notes:\n"
            "  This is a coarse pre-filter, not a rally detector. Segments keep whole\n"
            "  rallies together including the pauses inside them; if you need exact\n"
            "  point boundaries, run the real event detector on what survives here.\n"
        ))
    parser.add_argument("video", help="the long recording to segment")
    parser.add_argument("--calibration", default=None, metavar="FILE",
                        help="court calibration to restrict motion to (default: "
                             "calibration/<video name>.json if it exists)")
    parser.add_argument("--no-calibration", action="store_true",
                        help="use the whole frame instead of the calibrated court - "
                             "motion on a neighbouring court will count")
    parser.add_argument("--out-dir", default=None,
                        help="where to write the cut clips (default: "
                             "<video name>_points/ next to the input)")
    parser.add_argument("--downscale", type=int, default=4, metavar="N",
                        help="working resolution divisor for the scan (default 4)")
    parser.add_argument("--smooth-s", type=float, default=0.2, metavar="SECONDS",
                        help="moving-average window on the activity signal (default 0.2)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="activity threshold (default: chosen automatically from "
                             "the signal's own 10th/90th percentiles)")
    parser.add_argument("--min-gap-s", type=float, default=DEFAULT_MIN_GAP_S,
                        metavar="SECONDS",
                        help=f"idle time inside this long does not split a segment "
                             f"(default {DEFAULT_MIN_GAP_S})")
    parser.add_argument("--pad-s", type=float, default=DEFAULT_PAD_S, metavar="SECONDS",
                        help=f"context kept on each side of a segment (default "
                             f"{DEFAULT_PAD_S})")
    parser.add_argument("--min-duration-s", type=float, default=DEFAULT_MIN_DURATION_S,
                        metavar="SECONDS",
                        help=f"runs shorter than this, before padding, are dropped as "
                             f"noise (default {DEFAULT_MIN_DURATION_S})")
    parser.add_argument("--reencode", action="store_true",
                        help="re-encode each cut for a frame-accurate boundary instead "
                             "of a fast stream copy (slower; stream copy can only start "
                             "a little early, never late, which is the safe direction)")
    parser.add_argument("--dry-run", action="store_true",
                        help="compute the manifest and print the summary, but do not "
                             "extract any clips")
    parser.add_argument("--plot", action="store_true",
                        help="save <out-dir>/activity.png visualising the signal, the "
                             "threshold and which stretches were kept")
    args = parser.parse_args()

    if not Path(args.video).exists():
        print(f"error: video not found: {args.video}", file=sys.stderr)
        return 2
    if args.calibration and args.no_calibration:
        print("error: --calibration and --no-calibration contradict each other",
              file=sys.stderr)
        return 2
    if not args.dry_run and shutil.which("ffmpeg") is None:
        print("error: ffmpeg not found on PATH. Install it, or pass --dry-run to only "
              "compute the manifest.", file=sys.stderr)
        return 2

    calibration, calibration_source = _resolve_calibration(
        args.video, args.calibration, args.no_calibration)
    if calibration is None and not args.no_calibration:
        print("  no calibration found for this video - using the whole frame. Motion "
              "on a neighbouring court will count as activity here. Pass "
              "--calibration, or run 'tennis-vision calibrate' first, to restrict this "
              "to one court.")

    probe = cv2.VideoCapture(args.video)
    width = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(probe.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = probe.get(cv2.CAP_PROP_FPS) or 30.0
    probe.release()
    if not width or not height:
        print(f"error: could not read {args.video}", file=sys.stderr)
        return 2

    roi_polygon = None
    if calibration is not None:
        try:
            scaled = calibration.scaled_to((width, height))
            roi_polygon = scaled.roi(frame_size=(width, height))
        except ValueError as exc:
            print(f"  warning: could not build a court region from the calibration "
                 f"({exc}); using the whole frame instead")

    mask = build_roi_mask(roi_polygon, (width, height), args.downscale)

    print(f"{args.video}: {total_frames or '?'} frames at {width}x{height}, {fps:g}fps")
    if calibration_source:
        print(f"  restricted to the court in {calibration_source}")
    print(f"  scanning at 1/{args.downscale} resolution "
         f"({width // args.downscale}x{height // args.downscale})...")

    t0 = time.time()
    signal, measured_fps, _ = compute_activity_signal(
        args.video, roi_mask=mask, downscale=args.downscale,
        on_progress=_progress_printer(total_frames, fps))
    print()   # end the progress line
    scan_elapsed = time.time() - t0
    fps = measured_fps or fps

    smoothed = smooth(signal, window=max(1, int(round(args.smooth_s * fps))))
    threshold = args.threshold if args.threshold is not None else auto_threshold(smoothed)
    threshold_source = "explicit" if args.threshold is not None else "auto"

    segments = find_segments(
        smoothed, fps, threshold,
        min_gap_s=args.min_gap_s, pad_s=args.pad_s, min_duration_s=args.min_duration_s,
    )

    total_duration = len(signal) / fps if fps else 0.0
    retained = sum(s.duration_s for s in segments)
    print(f"\n  scan took {scan_elapsed:.1f}s ({len(signal) / max(scan_elapsed, 1e-6):.0f} "
         f"fps, single core, no GPU)")
    print(f"  activity: min={smoothed.min():.3f} p10={np.percentile(smoothed, 10):.3f} "
         f"median={np.median(smoothed):.3f} p90={np.percentile(smoothed, 90):.3f} "
         f"max={smoothed.max():.3f}")
    print(f"  threshold: {threshold:.3f} ({threshold_source})")
    print(f"\n  {len(segments)} segment(s) kept, {_format_time(retained)} of "
         f"{_format_time(total_duration)} "
         f"({100 * retained / total_duration if total_duration else 0:.0f}%)")
    for i, seg in enumerate(segments):
        print(f"    {i:3d}  {_format_time(seg.start_s):>8}  -  {_format_time(seg.end_s):>8}"
             f"   {seg.duration_s:5.1f}s   peak {seg.peak_activity:.2f}")

    stem = Path(args.video).stem
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.video).with_name(f"{stem}_points")

    manifest = {
        "video": str(args.video),
        "video_fps": round(fps, 3),
        "video_frames": len(signal),
        "video_duration_s": round(total_duration, 3),
        "calibration_source": calibration_source,
        "parameters": {
            "downscale": args.downscale,
            "smooth_s": args.smooth_s,
            "threshold": round(float(threshold), 4),
            "threshold_source": threshold_source,
            "min_gap_s": args.min_gap_s,
            "pad_s": args.pad_s,
            "min_duration_s": args.min_duration_s,
        },
        "activity_stats": {
            "min": round(float(smoothed.min()), 4) if len(smoothed) else None,
            "p10": round(float(np.percentile(smoothed, 10)), 4) if len(smoothed) else None,
            "median": round(float(np.median(smoothed)), 4) if len(smoothed) else None,
            "p90": round(float(np.percentile(smoothed, 90)), 4) if len(smoothed) else None,
            "max": round(float(smoothed.max()), 4) if len(smoothed) else None,
        },
        "retained_duration_s": round(retained, 3),
        "retained_fraction": round(retained / total_duration, 4) if total_duration else 0.0,
        "extraction_mode": "none (--dry-run)" if args.dry_run else (
            "reencode" if args.reencode else "stream_copy"),
        "segments": [],
    }

    if args.plot:
        out_dir.mkdir(parents=True, exist_ok=True)
        _save_plot(smoothed, fps, threshold, segments, out_dir / "activity.png")
        print(f"\n  wrote {out_dir / 'activity.png'}")

    if not segments:
        print("\n  nothing kept - nothing to extract. Check activity.png (--plot) or "
             "widen --threshold before assuming the recording is really empty.")
    elif args.dry_run:
        print(f"\n  --dry-run: not extracting. {len(segments)} clip(s) would be "
             f"written to {out_dir}/")
    else:
        print(f"\n  extracting {len(segments)} clip(s) to {out_dir}/ "
             f"({'re-encoding' if args.reencode else 'stream copy'})...")
        failures = 0
        for i, seg in enumerate(segments):
            name = f"{stem}_{i:03d}.mp4"
            ok, error = _extract_segment(args.video, seg, out_dir / name, args.reencode)
            record = seg.as_dict()
            record["index"] = i
            record["file"] = name if ok else None
            if not ok:
                failures += 1
                record["error"] = error
                print(f"    {i:3d}  FAILED: {error}")
            manifest["segments"].append(record)
        if failures:
            print(f"\n  {failures} of {len(segments)} clip(s) failed - see manifest.json")

    if args.dry_run or not segments:
        manifest["segments"] = [
            {**seg.as_dict(), "index": i, "file": None}
            for i, seg in enumerate(segments)
        ]

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n  wrote {manifest_path}")

    if segments and not args.dry_run:
        print(f"\n  next: run the real pipeline on each clip, e.g.\n"
             f"    tennis-vision analyze {out_dir / (stem + '_000.mp4')}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
