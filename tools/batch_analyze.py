"""
tools/batch_analyze.py
-----------------------
Run the real pipeline over every clip tools/segment_points.py produced, and combine
what came back into one report.

Why this exists
----------------
segment_points cuts a long recording into a manifest of short clips. Analysing 166 of
them by hand - one `tennis-vision analyze` per clip, one summary.json to open each time
- does not scale. This runs the pipeline on every clip in the manifest, one at a time,
and writes both the full per-clip data and a combined report.

One calibration for every clip
-------------------------------
The clips share a camera position with the recording they were cut from (segmentation
does not move the camera), so pass ONE calibration and it applies to all of them:

    tennis-vision segment  session.mp4 --court-calibration calibration/court_A.json
    python tools/batch_analyze.py session_points/ --court-calibration calibration/court_A.json

Auto-discovery by video file name, which the pipeline normally uses, will NOT find it on
its own here: a clip is named session_000.mp4, not session.mp4, so calibration/session.json
would never match it. --court-calibration says so explicitly rather than silently falling
back to the keypoint model on every clip.

What is NOT safe to combine across clips, and why
----------------------------------------------------
Player identity does not carry over. `utils.player_selection.select_two_players` numbers
the two selected people 1 and 2 by whichever has the LOWER internal tracker id in THAT
clip - not by which side of the net they are on, not by anything physical. ByteTrack
starts counting fresh every time main.py runs, so "Player 1" in clip 017 and "Player 1"
in clip 093 have no relationship: summing "Player 1's shots" across clips could just as
easily be summing shots from two different physical people. This report therefore never
produces a combined per-player figure. It sums total shots (both players together, which
needs no identity), and it keeps the per-clip P1/P2 breakdown available for someone
looking at ONE clip, where the labels are meaningful for the duration of that clip.

Speeds are combined by weighting each clip's average by its own shot count, not by
averaging the per-clip averages unweighted - a clip with one shot must not count as much
as a clip with twenty.

`near_camera_pid` in each row is the closest this report gets to identity, and it is
still only a WITHIN-CLIP fact: which of that clip's two ids sat, on median, closer to
the camera. It is not carried between clips and it does not survive a change of ends -
real tennis swaps which physical person is near partway through a match, and nothing
here detects that happening. Telling the two players apart for good, across a change of
ends, needs to look at what they look like (visual re-identification), which this tool
does not attempt.

A combined video
-----------------
--combine-video (implies --with-video) stitches every successfully rendered clip into
one file, in the manifest's chronological order, via ffmpeg's concat filter - a
re-encoding join rather than a stream-copy one, so clips do not need byte-identical
codec parameters to combine cleanly.

Usage
-----
    python tools/batch_analyze.py session_points/ --court-calibration calibration/court_A.json
    python tools/batch_analyze.py session_points/manifest.json --limit 5 --dry-run
    python tools/batch_analyze.py session_points/ --with-video   # render each clip too
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── resolving the input: a manifest.json, or a directory of clips ─────────────


def _resolve_clips(input_path: Path) -> tuple[list[Path], dict | None]:
    """
    The clips to process, and the segment_points manifest they came from if there is
    one - its start_s/end_s let a result be traced back to when in the original
    recording it happened, which a bare list of file names cannot.
    """
    if input_path.is_file() and input_path.suffix == ".json":
        manifest_path, base = input_path, input_path.parent
    elif input_path.is_dir() and (input_path / "manifest.json").exists():
        manifest_path, base = input_path / "manifest.json", input_path
    else:
        manifest_path, base = None, input_path

    if manifest_path is not None:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        clips = []
        for seg in manifest.get("segments", []):
            if seg.get("file"):
                clips.append(base / seg["file"])
        return clips, manifest

    if not input_path.is_dir():
        return [], None
    return sorted(input_path.glob("*.mp4")), None


def _segment_context(manifest: dict | None, clip: Path) -> dict:
    """start_s/end_s in the ORIGINAL recording, from the manifest, for traceability."""
    if manifest is None:
        return {}
    for seg in manifest.get("segments", []):
        if seg.get("file") and Path(seg["file"]).name == clip.name:
            return {"source_start_s": seg.get("start_s"), "source_end_s": seg.get("end_s")}
    return {}


# ── running one clip ────────────────────────────────────────────────────────


def _write_clip_config(base_config: str, clip_dir: Path) -> Path:
    """
    A tiny config that sends this clip's stats and logs into its OWN folder.

    main.py has no CLI flag for the stats directory - only io.output_stats_dir in the
    config file - and it defaults to the single shared output/stats/, timestamped to the
    second. Sequential runs are unlikely to collide there, but "unlikely" is not a
    property a report that feeds a spreadsheet should depend on: giving every clip its
    own folder makes finding "the" summary for clip N a listdir, not a race.
    """
    clip_dir.mkdir(parents=True, exist_ok=True)
    override = clip_dir / "_config.yaml"
    override.write_text(
        "io:\n"
        f"  output_stats_dir: \"{clip_dir.as_posix()}\"\n"
        f"  log_dir: \"{clip_dir.as_posix()}\"\n",
        encoding="utf-8",
    )
    return override


def _run_one_clip(
    clip: Path, clip_dir: Path, base_config: str, calibration: str | None,
    no_calibration: bool, with_video: bool, max_frames: int,
) -> tuple[bool, str]:
    override_config = _write_clip_config(base_config, clip_dir)
    forwarded = [
        sys.executable, "main.py",
        "--input", str(clip),
        "--config", str(override_config),
        "--no-stubs",
    ]
    if not with_video:
        forwarded.append("--no-video")
    else:
        forwarded += ["--output", str(clip_dir / "rendered.avi")]
    if max_frames:
        forwarded += ["--max-frames", str(max_frames)]
    if calibration:
        forwarded += ["--court-calibration", calibration]
    elif no_calibration:
        forwarded.append("--no-court-calibration")

    result = subprocess.run(forwarded, capture_output=True, text=True, timeout=1800)
    if result.returncode != 0:
        tail = (result.stderr or result.stdout or "unknown error").strip()
        return False, tail[-800:]
    return True, ""


def _latest_summary(clip_dir: Path) -> dict | None:
    summaries = sorted(clip_dir.glob("summary_*.json"))
    if not summaries:
        return None
    return json.loads(summaries[-1].read_text(encoding="utf-8"))


# ── combining every rendered clip into one video ────────────────────────────


def build_concat_command(video_paths: list[Path], out_path: Path) -> list[str]:
    """
    The ffmpeg command that joins `video_paths`, in order, into `out_path`.

    Pure and separate from running it, so the command itself - the part a flag-ordering
    mistake would break - is checkable without spawning ffmpeg or needing real video
    files on disk.

    Uses the CONCAT FILTER (`concat=n=...:v=1:a=0`, decode-and-re-encode), not the
    concat DEMUXER (`-f concat`, stream copy). The demuxer is faster but requires every
    input to share identical codec parameters, which nothing here guarantees - clips
    come from `main.py` runs that could in principle fall back from XVID to MJPG per
    run (utils/video_utils.save_video tries a second codec if the first fails to open).
    The filter re-encodes regardless of what it was handed, at the cost of the encoding
    time, which is small next to what producing these clips already cost.
    """
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    for path in video_paths:
        cmd += ["-i", str(path)]
    n = len(video_paths)
    streams = "".join(f"[{i}:v]" for i in range(n))
    cmd += ["-filter_complex", f"{streams}concat=n={n}:v=1:a=0[outv]",
           "-map", "[outv]",
           "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", str(out_path)]
    return cmd


def combine_videos(video_paths: list[Path], out_path: Path) -> tuple[bool, str]:
    """Run build_concat_command and report whether out_path came out the other end."""
    if not video_paths:
        return False, "no rendered clips to combine"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(build_concat_command(video_paths, out_path),
                            capture_output=True, text=True)
    if not out_path.exists() or out_path.stat().st_size == 0:
        return False, (result.stderr or "unknown ffmpeg error").strip()[-800:]
    return True, ""


def _clip_record(clip: Path, summary: dict, context: dict) -> dict:
    """The fields worth a row in the combined report, pulled out of one clip's full
    summary.json - see aggregate_results for what is and is not safe to combine."""
    p1 = int(summary.get("total_shots_p1", 0) or 0)
    p2 = int(summary.get("total_shots_p2", 0) or 0)
    shot_types = (summary.get("shot_classification") or {}).get("types", {})
    return {
        "clip": clip.name,
        **context,
        "total_shots": p1 + p2,
        "shots_p1_this_clip_only": p1,
        "shots_p2_this_clip_only": p2,
        "avg_shot_speed_p1_kmh": summary.get("avg_shot_speed_p1_kmh"),
        "avg_shot_speed_p2_kmh": summary.get("avg_shot_speed_p2_kmh"),
        "shot_types": shot_types,
        "court_calibrated": summary.get("court_calibrated"),
        "court_source": summary.get("court_source"),
        "court_line_support": summary.get("court_line_support"),
        "player_selection_status": (summary.get("player_selection") or {}).get("status"),
        # Which id sat closer to the camera in THIS clip - not a claim about which
        # physical person that is over the whole match. See the module docstring.
        "near_camera_pid": (summary.get("player_selection") or {}).get("near_camera_pid"),
        "ball_coverage": (summary.get("ball") or {}).get("coverage"),
        "shot_speed_3d_mean_kmh": (summary.get("shot_speed_3d_kmh") or {}).get("mean"),
        "shot_speed_3d_segments": (summary.get("shot_speed_3d_kmh") or {}).get("segments"),
        "warning": summary.get("warning"),
    }


# ── combining clips: the pure, testable part ────────────────────────────────


def aggregate_results(records: list[dict]) -> dict:
    """
    Combine per-clip records into session-wide totals, refusing to fabricate the one
    figure that cannot honestly be produced: a per-player total across clips. See the
    module docstring for why player identity does not carry over between clips.

    Args:
        records: successful clips' entries, in the shape _clip_record produces.

    Returns:
        A dict with total_shots (safe: identity-independent), a shot type breakdown
        (summed across clips), a shot-count-weighted average speed, the true min/max of
        3-D shot speeds seen anywhere, and lists of clips flagged for court or player
        problems - never a combined per-player number.
    """
    if not records:
        return {
            "clips": 0, "total_shots": 0, "shot_types": {},
            "avg_shot_speed_kmh": None, "shot_speed_3d": None,
            "clips_with_court_problems": [], "clips_with_player_problems": [],
            "note": "no successful clips to aggregate",
        }

    total_shots = sum(r["total_shots"] for r in records)

    shot_types: dict[str, int] = {}
    for r in records:
        for name, count in (r.get("shot_types") or {}).items():
            shot_types[name] = shot_types.get(name, 0) + int(count)

    # Weighted by each clip's own shot count, not an unweighted mean of per-clip
    # averages - a clip with one shot must not outweigh a clip with twenty.
    speed_weighted_sum = 0.0
    speed_weight = 0
    for r in records:
        for count_key, speed_key in (("shots_p1_this_clip_only", "avg_shot_speed_p1_kmh"),
                                     ("shots_p2_this_clip_only", "avg_shot_speed_p2_kmh")):
            count = r.get(count_key) or 0
            speed = r.get(speed_key)
            if count and speed:
                speed_weighted_sum += speed * count
                speed_weight += count
    avg_speed = round(speed_weighted_sum / speed_weight, 1) if speed_weight else None

    speed_3d_weighted_sum = 0.0
    speed_3d_weight = 0
    speed_3d_values = []
    for r in records:
        n = r.get("shot_speed_3d_segments") or 0
        mean = r.get("shot_speed_3d_mean_kmh")
        if n and mean:
            speed_3d_weighted_sum += mean * n
            speed_3d_weight += n
            speed_3d_values.append(mean)
    shot_speed_3d = None
    if speed_3d_weight:
        shot_speed_3d = {
            "weighted_mean_kmh": round(speed_3d_weighted_sum / speed_3d_weight, 1),
            "segments": speed_3d_weight,
            "clip_mean_min_kmh": round(min(speed_3d_values), 1),
            "clip_mean_max_kmh": round(max(speed_3d_values), 1),
        }

    court_problems = [r["clip"] for r in records if r.get("court_calibrated") is False]
    player_problems = [r["clip"] for r in records
                       if r.get("player_selection_status") not in (None, "ok")]

    return {
        "clips": len(records),
        "total_shots": total_shots,
        "shot_types": dict(sorted(shot_types.items(), key=lambda kv: -kv[1])),
        "avg_shot_speed_kmh": avg_speed,
        "shot_speed_3d": shot_speed_3d,
        "clips_with_court_problems": court_problems,
        "clips_with_player_problems": player_problems,
        "note": (
            "total_shots and avg_shot_speed_kmh combine both players per clip and are "
            "safe to read as session totals. There is deliberately no combined "
            "per-player figure: 'Player 1' is renumbered independently in every clip "
            "by internal tracker id, not by court side, so it does not name the same "
            "person across clips. Open batch_report.csv for the per-clip P1/P2 "
            "breakdown, valid only within each row's own clip."
        ),
    }


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="tennis-vision batch-analyze",
        description=("Run the pipeline over every clip a segmentation manifest "
                     "produced, and combine the results into one report."),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "note:\n"
            "  Player identity does not carry across clips - see the module docstring.\n"
            "  The report sums total shots (safe) and never a combined per-player\n"
            "  figure (not safe).\n"
        ))
    parser.add_argument("input", help="a segment_points manifest.json, or a directory "
                                      "containing one (or a plain directory of .mp4 "
                                      "clips, if no manifest exists)")
    parser.add_argument("--court-calibration", metavar="FILE", default=None,
                        help="applied to EVERY clip - they share one camera position. "
                             "Auto-discovery by clip file name will not find it, since "
                             "a clip is not named like the recording it came from")
    parser.add_argument("--no-court-calibration", action="store_true",
                        help="acknowledge running every clip without a calibration "
                             "(falls back to the keypoint model per clip)")
    parser.add_argument("--config", default="configs/config.yaml",
                        help="base pipeline config (default configs/config.yaml)")
    parser.add_argument("--out-dir", default=None,
                        help="where per-clip folders and the combined report go "
                             "(default: <input>_analysis/ next to the input)")
    parser.add_argument("--with-video", action="store_true",
                        help="render each clip's annotated video too (slow - off by "
                             "default, since a batch run is about the numbers)")
    parser.add_argument("--combine-video", action="store_true",
                        help="stitch every rendered clip into one video, in "
                             "chronological order, written to combined_analysis.mp4 - "
                             "implies --with-video")
    parser.add_argument("--max-frames", type=int, default=0, metavar="N",
                        help="cap every clip to its first N frames - a fast pass over "
                             "the whole batch before committing to the full run")
    parser.add_argument("--limit", type=int, default=0, metavar="N",
                        help="process only the first N clips")
    parser.add_argument("--skip-existing", action="store_true",
                        help="skip a clip whose folder already has a summary.json - "
                             "resume an interrupted batch without redoing finished work")
    parser.add_argument("--dry-run", action="store_true",
                        help="list what would be processed and exit")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"error: not found: {args.input}", file=sys.stderr)
        return 2
    if args.court_calibration and args.no_court_calibration:
        print("error: --court-calibration and --no-court-calibration contradict "
              "each other", file=sys.stderr)
        return 2
    if args.court_calibration and not Path(args.court_calibration).exists():
        print(f"error: calibration not found: {args.court_calibration}", file=sys.stderr)
        return 2
    if args.combine_video and not args.with_video:
        print("  --combine-video implies --with-video (there is nothing to stitch "
             "together without a rendered clip per video) - enabling it")
        args.with_video = True
    if args.combine_video and not args.dry_run and shutil.which("ffmpeg") is None:
        print("error: ffmpeg not found on PATH - required for --combine-video",
              file=sys.stderr)
        return 2

    clips, manifest = _resolve_clips(input_path)
    if not clips:
        print(f"error: no clips found under {args.input} (looked for manifest.json "
             f"and *.mp4)", file=sys.stderr)
        return 2
    if args.limit:
        clips = clips[:args.limit]

    if not args.court_calibration and not args.no_court_calibration:
        print("  no --court-calibration given: each clip will fall back to the "
             "keypoint model, which is exactly the case hand-placed calibration exists "
             "for on this kind of footage. Pass --court-calibration, or "
             "--no-court-calibration to proceed without one deliberately.")

    base = manifest.get("video") if manifest else str(input_path)
    out_dir = Path(args.out_dir) if args.out_dir else Path(f"{Path(base).stem}_analysis")

    print(f"{len(clips)} clip(s) from {args.input}")
    print(f"  output: {out_dir}/")
    if args.dry_run:
        for c in clips:
            print(f"    {c}")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    failures: list[dict] = []
    t0 = time.time()

    for i, clip in enumerate(clips):
        clip_dir = out_dir / clip.stem
        if args.skip_existing and list(clip_dir.glob("summary_*.json")):
            print(f"  [{i + 1}/{len(clips)}] {clip.name}: skipped (already done)")
            summary = _latest_summary(clip_dir)
            if summary:
                records.append(_clip_record(clip, summary, _segment_context(manifest, clip)))
            continue

        elapsed = time.time() - t0
        eta = (elapsed / i * (len(clips) - i)) if i else 0
        print(f"  [{i + 1}/{len(clips)}] {clip.name}...  "
             f"(elapsed {elapsed / 60:.1f}m, ETA {eta / 60:.1f}m)", end="", flush=True)

        ok, error = _run_one_clip(
            clip, clip_dir, args.config, args.court_calibration,
            args.no_court_calibration, args.with_video, args.max_frames,
        )
        if not ok:
            print("  FAILED")
            failures.append({"clip": clip.name, "error": error})
            continue

        summary = _latest_summary(clip_dir)
        if summary is None:
            print("  FAILED (no summary written)")
            failures.append({"clip": clip.name, "error": "pipeline exited 0 but wrote "
                                                          "no summary_*.json"})
            continue

        record = _clip_record(clip, summary, _segment_context(manifest, clip))
        records.append(record)
        print(f"  {record['total_shots']} shot(s)"
             + (f", court {'OK' if record['court_calibrated'] else 'FAILED'}"
                if record["court_calibrated"] is not None else ""))

    aggregate = aggregate_results(records)

    combined_video: dict = {"attempted": False}
    if args.combine_video:
        # In the same order records were appended, which is the order `clips` was
        # walked in - the manifest's own chronological order, preserved because only
        # successes are appended and never reordered.
        rendered = [out_dir / Path(r["clip"]).stem / "rendered.avi" for r in records]
        rendered = [p for p in rendered if p.exists()]
        combined_path = out_dir / "combined_analysis.mp4"
        print(f"\n  combining {len(rendered)} rendered clip(s) into "
             f"{combined_path.name}...")
        ok, error = combine_videos(rendered, combined_path)
        combined_video = {
            "attempted": True, "clips_combined": len(rendered),
            "path": str(combined_path) if ok else None,
            "error": None if ok else error,
        }
        if ok:
            print(f"  wrote {combined_path}")
        else:
            print(f"  FAILED to combine: {error}")

    report = {
        "input": str(args.input),
        "clips_total": len(clips),
        "clips_succeeded": len(records),
        "clips_failed": len(failures),
        "calibration": args.court_calibration,
        "elapsed_s": round(time.time() - t0, 1),
        "aggregate": aggregate,
        "combined_video": combined_video,
        "failures": failures,
        "clips": records,
    }
    (out_dir / "batch_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")

    if records:
        fields = list(records[0].keys())
        with open(out_dir / "batch_report.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for r in records:
                row = dict(r)
                row["shot_types"] = json.dumps(row["shot_types"])
                writer.writerow(row)

    print(f"\n{len(records)}/{len(clips)} clip(s) succeeded, {len(failures)} failed, "
         f"{(time.time() - t0) / 60:.1f} minutes")
    print(f"  total shots (both players, all clips): {aggregate['total_shots']}")
    if aggregate["shot_types"]:
        print("  " + ", ".join(f"{k}: {v}" for k, v in aggregate["shot_types"].items()))
    if aggregate["avg_shot_speed_kmh"]:
        print(f"  shot-count-weighted average speed: {aggregate['avg_shot_speed_kmh']} km/h")
    if aggregate["clips_with_court_problems"]:
        print(f"  {len(aggregate['clips_with_court_problems'])} clip(s) had a failed "
             f"court fit - see batch_report.json")
    if failures:
        print(f"  {len(failures)} clip(s) failed outright - see batch_report.json")
    print(f"\n  {aggregate['note']}")
    if combined_video.get("path"):
        print(f"\n  combined video: {combined_video['path']}")
    print(f"\n  wrote {out_dir / 'batch_report.json'}")
    print(f"  wrote {out_dir / 'batch_report.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
