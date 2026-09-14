"""
tools/calibrate_lens.py
------------------------
Measure one camera's lens distortion from a checkerboard, so a later change can correct
it in the positions the pipeline already computes.

Why this exists
----------------
A homography assumes straight lines project to straight lines, and the wide action
cameras this project has been pointed at so far do not honour that: a painted line that
is straight on the court visibly bows in the frame. Fitting distortion to the court's
own lines was tried and did not converge to a physical lens model - a court gives at
most nine lines, seen from one fixed angle, and that is not enough independent
information to separate "the lens bent this" from "the corner was placed here". A
checkerboard, filmed from many angles with the SAME camera and the SAME settings used
for the tennis footage, gives that information.

This produces the calibration and checks it is trustworthy. It does not apply it to
anything - see utils/lens_calibration.py's module docstring for why that is
deliberately a separate, later change.

Filming the checkerboard
--------------------------
Use a real checkerboard (printed, or shown on a large flat screen) with a KNOWN grid of
inner corners - the corners where four squares meet, not the outer edge, and not
counting squares. A common size is 9x6 inner corners (a 10x7-square board). Standard
sizes work fine with the default: pass --pattern COLSxROWS if using a different one.

Film it with the SAME camera, SAME lens setting (zoom, wide/linear mode if the camera
offers one) and SAME resolution as the tennis footage - a calibration measures one
specific optical setup, not "this camera in general". Move the board through the frame
so it visibly covers the corners and edges as well as the centre - distortion is
smallest in the middle and largest at the edges, which is exactly the part a few
centred photos would fail to measure at all. Tilt it at a few different angles too,
not only face-on.

Usage
-----
    python tools/calibrate_lens.py checkerboard.mp4 --pattern 9x6
    python tools/calibrate_lens.py checkerboard_photos/ --pattern 9x6 --square-size 25
    python tools/calibrate_lens.py checkerboard.mp4 --pattern 9x6 --model fisheye
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.lens_calibration import (                         # noqa: E402
    MIN_VIEWS,
    MODEL_FISHEYE,
    MODEL_STANDARD,
    BoardSpec,
    LensCalibration,
    coverage_span,
    fit_fisheye,
    fit_standard,
    find_board_corners,
    reprojection_error_per_view,
)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def _parse_pattern(text: str) -> tuple[int, int]:
    try:
        cols, rows = (int(part) for part in text.lower().split("x"))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected COLSxROWS (e.g. 9x6), got {text!r}") from None
    if cols < 3 or rows < 3:
        raise argparse.ArgumentTypeError("a board needs at least 3x3 inner corners")
    return (cols, rows)


def _iter_video_frames(path: str, stride: int):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"could not open video: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index % stride == 0:
                yield index, total, frame
            index += 1
    finally:
        cap.release()


def _iter_image_files(directory: Path):
    """
    Vanilla cv2.imread returns None for a file it cannot decode - empty, truncated, not
    really an image despite its extension. That is not safe to rely on in THIS
    process, though: ultralytics (already a dependency, for the player detector)
    monkey-patches cv2.imread at import time to raise cv2.error instead of returning
    None on exactly that input. Whether this tool sees the patched or the vanilla
    function depends on whether something else already imported ultralytics first in
    the same run - an import-order accident, not a difference in the file itself - so
    both outcomes are treated as "unreadable, skip it" rather than trusting either
    contract alone.
    """
    files = sorted(
        p for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    for i, path in enumerate(files):
        try:
            frame = cv2.imread(str(path))
        except cv2.error:
            frame = None
        if frame is not None:
            yield i, len(files), frame, path.name


def _collect_views(args, board: BoardSpec):
    """
    Walk the input, keeping up to `args.max_views` frames/images where the board was
    found, and the (width, height) they were found at.

    Detection is the slow part of this tool, not the fit - a `--stride` that is too
    small pays for a corner search on many near-duplicate frames of a slowly-moved
    board for no extra information, so progress is printed as it happens rather than
    only at the end.
    """
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None
    obj_template = board.object_points()

    input_path = Path(args.input)
    is_video = input_path.is_file()
    source = (_iter_video_frames(args.input, args.stride) if is_video
             else _iter_image_files(input_path))

    scanned = found = 0
    for item in source:
        if is_video:
            index, total, frame = item
            label = f"frame {index}"
        else:
            index, total, frame, name = item
            label = name
        scanned += 1
        if image_size is None:
            image_size = (frame.shape[1], frame.shape[0])
        elif (frame.shape[1], frame.shape[0]) != image_size:
            print(f"  skipping {label}: {frame.shape[1]}x{frame.shape[0]} does not "
                 f"match the first frame's {image_size[0]}x{image_size[1]} - a "
                 f"calibration needs one consistent resolution throughout")
            continue

        corners = find_board_corners(frame, board)
        if corners is not None:
            found += 1
            object_points.append(obj_template.astype(np.float32))
            image_points.append(corners.astype(np.float32))
            print(f"\r  scanned {scanned}, found the board in {found} "
                 f"(need {MIN_VIEWS}+)...  ", end="", flush=True)
        if found >= args.max_views:
            break
    print()
    return object_points, image_points, image_size, scanned, found


def _fit_and_report(object_points, image_points, image_size, requested_model: str):
    """
    Fits whichever model(s) were requested, prints both RMS figures when both were
    tried, and returns (model_name, FitResult, other_model_rms) - the third value is
    the model NOT chosen, kept for audit in the saved calibration rather than
    discarded once it has served the comparison that picked the winner.
    """
    results = {}
    if requested_model in ("auto", MODEL_STANDARD):
        try:
            results[MODEL_STANDARD] = fit_standard(object_points, image_points, image_size)
        except cv2.error as exc:
            print(f"  standard model: FAILED to fit ({exc})")
    if requested_model in ("auto", MODEL_FISHEYE):
        try:
            results[MODEL_FISHEYE] = fit_fisheye(object_points, image_points, image_size)
        except cv2.error as exc:
            print(f"  fisheye model:  FAILED to fit ({exc})")

    for name, fit in results.items():
        print(f"  {name:<10} rms {fit.rms:6.3f} px  ({len(object_points)} views)")

    if not results:
        return None, None, None
    if requested_model != "auto":
        return requested_model, results[requested_model], None

    winner = min(results, key=lambda name: results[name].rms)
    other_rms = next((r.rms for n, r in results.items() if n != winner), None)
    print(f"  -> {winner} fits better"
         + (f" ({results[winner].rms:.3f} vs {other_rms:.3f} px)" if other_rms else ""))
    return winner, results[winner], other_rms


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="tennis-vision calibrate-lens",
        description=("Measure a camera's lens distortion from a checkerboard, fitting "
                     "both the standard and fisheye models and reporting which one "
                     "actually fits this camera."),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "note:\n"
            "  Produces the calibration only - it is not applied to anything by this\n"
            "  tool. Film the checkerboard with the SAME camera, lens setting and\n"
            "  resolution as the footage you intend to correct, and move it so it\n"
            "  covers the edges and corners of the frame, not just the centre.\n"
        ))
    parser.add_argument("input", help="a checkerboard video, or a directory of "
                                      "checkerboard photos")
    parser.add_argument("--pattern", type=_parse_pattern, required=True, metavar="COLSxROWS",
                        help="inner corners of the board, e.g. 9x6 (where the board "
                             "has 10x7 squares) - the corners where four squares meet, "
                             "not the outer edge")
    parser.add_argument("--square-size", type=float, default=1.0, metavar="UNITS",
                        help="real-world size of one square, any consistent unit "
                             "(default 1.0). Only affects the intrinsics' absolute "
                             "scale, not the distortion this tool exists to measure")
    parser.add_argument("--stride", type=int, default=10, metavar="N",
                        help="video only: check every Nth frame (default 10) - "
                             "consecutive frames of a slowly moved board are nearly "
                             "identical and not worth detecting corners in")
    parser.add_argument("--max-views", type=int, default=40, metavar="N",
                        help="stop once this many good views are found (default 40) - "
                             "more views cost fitting time for steadily diminishing "
                             "improvement past a few dozen")
    parser.add_argument("--min-views", type=int, default=MIN_VIEWS, metavar="N",
                        help=f"refuse to fit from fewer views than this (default "
                             f"{MIN_VIEWS})")
    parser.add_argument("--model", choices=["auto", MODEL_STANDARD, MODEL_FISHEYE],
                        default="auto",
                        help="which lens model to fit (default: fit both, keep "
                             "whichever measures a lower reprojection error)")
    parser.add_argument("--out", default=None,
                        help="where to write the calibration (default: "
                             "calibration/lens_<input name>.json)")
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"error: not found: {args.input}", file=sys.stderr)
        return 2

    board = BoardSpec(inner_corners=args.pattern, square_size=args.square_size)
    print(f"{args.input}: looking for a {args.pattern[0]}x{args.pattern[1]} "
         f"checkerboard...")

    object_points, image_points, image_size, scanned, found = _collect_views(args, board)

    if found < args.min_views:
        print(f"\nerror: found the board in only {found} of {scanned} frame(s)/image(s) "
             f"scanned - need at least {args.min_views}. Check --pattern matches the "
             f"real board (inner corners, not squares), that the board is fully "
             f"visible and reasonably sharp, and try a smaller --stride on video.",
             file=sys.stderr)
        return 2

    span = coverage_span(image_points, image_size)
    print(f"\n  {found} view(s) found, at {image_size[0]}x{image_size[1]}")
    print(f"  frame coverage: {span['x_span']:.0%} wide, {span['y_span']:.0%} tall"
         + ("" if span["wide_enough"] else "  (LOW - see the warning below)"))
    if not span["wide_enough"]:
        print("    The board stayed too close to one part of the frame. Distortion is "
             "smallest at the centre and largest at the edges, so a calibration fit "
             "only from central views under-measures exactly the part that matters "
             "most. Re-film moving the board out to the corners and edges before "
             "trusting this result.")

    print()
    model, fit, other_model_rms = _fit_and_report(
        object_points, image_points, image_size, args.model)
    if fit is None:
        print("\nerror: could not fit any model to these views", file=sys.stderr)
        return 2

    errors = reprojection_error_per_view(
        object_points, image_points, fit.camera_matrix, fit.dist_coeffs, model,
        fit.rvecs, fit.tvecs)
    worst = int(np.argmax(errors))
    print(f"\n  per-view error: median {sorted(errors)[len(errors) // 2]:.3f}px, "
         f"worst {errors[worst]:.3f}px (view {worst})")
    if errors[worst] > 3 * fit.rms and errors[worst] > 1.0:
        print(f"    view {worst} fits much worse than the rest - likely a mis-detected "
             f"corner or a blurred frame. Consider re-running with a different "
             f"--stride to avoid it, or filming a cleaner pass.")

    calibration = LensCalibration(
        model=model,
        camera_matrix=fit.camera_matrix,
        dist_coeffs=fit.dist_coeffs,
        image_size=image_size,
        rms_error=fit.rms,
        views_used=found,
        coverage=span,
        other_model_rms=other_model_rms,
        video=str(args.input),
        notes=f"{found} views, {model} model chosen "
             f"{'automatically' if args.model == 'auto' else '(forced by --model)'}",
    )

    out_path = Path(args.out) if args.out else Path("calibration") / f"lens_{Path(args.input).stem}.json"
    calibration.save(out_path)
    print(f"\n  wrote {out_path}")
    print(f"\n  This measures the lens - it is not applied to anything yet. Positions "
         f"the pipeline computes (ball, feet, court keypoints) are corrected in a "
         f"separate, later step.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
