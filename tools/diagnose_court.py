"""
tools/diagnose_court.py
───────────────────────
Say WHY a clip's court fit failed, instead of guessing from the headline score.

`court_line_support` is one number for nine line segments, and a failure has at
least three causes that the single number cannot tell apart:

  1. The keypoint model predicted the wrong quadrilateral. The fit is genuinely
     wrong and no threshold change helps.

  2. The keypoints are right, but the paint is WIDER than the test assumes.
     line_support_score samples the surface at a fixed offset_px=6 either side of
     the predicted line. A line more than ~12 px wide in frame - which happens on
     high-resolution video, or simply near the camera - means both "surface"
     samples land on paint too. Nothing is brighter than anything, and a correct
     fit scores near zero.

  3. The keypoints are right, but a shadow edge crosses the line. The test needs
     the line brighter than BOTH sides by margin=8 grey levels. Where a hard
     shadow boundary runs along or across a line, the sunlit side is brighter
     than the paint, so every sample there fails by construction.

Cases 2 and 3 are false negatives: the gate refuses a clip whose geometry was
fine. They call for different fixes than case 1, and telling them apart needs
per-line numbers and a picture.

Output per sampled frame:
  - a table of the nine named lines with their individual support
  - measured paint width at the sample points, against the 6 px the test assumes
  - an overlay PNG: predicted keypoints, the segments, and every sample point
    coloured by whether it counted as paint

Usage:
    python tools/diagnose_court.py clipe1.mp4
    python tools/diagnose_court.py clipe1.mp4 --frames 8 --out output/diag
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from court_line_detector import CourtLineDetector          # noqa: E402
from utils.court_validity import (                          # noqa: E402
    COURT_LINES,
    MIN_LINE_SUPPORT,
    line_support_score,
)

LINE_NAMES = (
    "left outer sideline",
    "right outer sideline",
    "far baseline",
    "near baseline",
    "left singles sideline",
    "right singles sideline",
    "far service line",
    "near service line",
    "centre service line",
)

OFFSET_PX = 6      # must match line_support_score
MARGIN = 8         # must match line_support_score
SAMPLES = 25
SEARCH_PX = 40     # how far perpendicular to look for the edge of a bright ridge

# Verdict per sample point. The first version of this tool reported a single
# "paint width" and saturated it at 3 * OFFSET_PX when no edge was found, then
# concluded from a saturated value that the paint must be wide. That conflated
# two opposite situations and produced a confidently wrong diagnosis:
#
#   WIDE - the point sits on a bright ridge whose edge is beyond OFFSET_PX. The
#          surface samples land on paint too, so the gate cannot work here even
#          though the keypoints may be perfect.
#   FLAT - the point sits on nothing. There is no ridge: the centre is no
#          brighter than its surroundings, so no edge exists to find. The
#          keypoints are not on a painted line at all.
#
# Both give "no edge within OFFSET_PX". Only FLAT means the prediction is wrong,
# and it is the common case, so defaulting to the WIDE reading inverted the
# conclusion on real footage.
NORMAL, WIDE, FLAT = "normal", "wide", "flat"


def sample_points(frame, kp, a, b):
    """
    Re-walk one segment the way line_support_score does, keeping the per-point
    verdict and a measured paint width that the scorer throws away.
    """
    gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    h, w = gray.shape

    def at(x, y):
        xi, yi = int(round(x)), int(round(y))
        return float(gray[yi, xi]) if 0 <= xi < w and 0 <= yi < h else None

    p0, p1 = kp[a], kp[b]
    seg = p1 - p0
    length = float(np.hypot(*seg))
    if length < 1.0:
        return [], []
    perp = np.array([-seg[1], seg[0]], dtype=np.float32) / length

    points, verdicts, widths = [], [], []
    for t in np.linspace(0.1, 0.9, SAMPLES):
        pt = p0 + seg * t
        centre = at(*pt)
        side_a = at(*(pt + perp * OFFSET_PX))
        side_b = at(*(pt - perp * OFFSET_PX))
        if centre is None or side_a is None or side_b is None:
            continue
        hit = centre > side_a + MARGIN and centre > side_b + MARGIN
        points.append((pt, hit))

        # Walk out far enough to distinguish a wide ridge from no ridge at all.
        edge = None
        profile = [centre]
        for d in np.arange(1.0, SEARCH_PX, 1.0):
            lo, hi = at(*(pt - perp * d)), at(*(pt + perp * d))
            if lo is None or hi is None:
                break
            profile += [lo, hi]
            if edge is None and lo < centre - MARGIN and hi < centre - MARGIN:
                edge = d

        if edge is not None:
            verdicts.append(NORMAL if edge <= OFFSET_PX else WIDE)
            widths.append(edge * 2)
        else:
            # No edge anywhere within SEARCH_PX. Is that because the ridge is
            # enormous, or because there is no ridge? A real line is the
            # brightest thing in its own neighbourhood; flat ground is not.
            contrast = centre - float(np.median(profile)) if profile else 0.0
            verdicts.append(WIDE if contrast > MARGIN else FLAT)
            if contrast > MARGIN:
                widths.append(SEARCH_PX * 2)
    return points, verdicts, widths


def main():
    # __doc__ is drawn with box characters and argparse writes help to a cp1252 console
    # on Windows, so --help died with a UnicodeEncodeError before printing anything.
    ap = argparse.ArgumentParser(
        description=("Say WHICH court line failed a clip's fit, and whether the "
                     "line-support gate can work on this footage at all."),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--frames", type=int, default=6, help="frames to sample (default 6)")
    ap.add_argument("--model", default="models/keypoints_model_geoaug.pth")
    ap.add_argument("--out", default="output/court_diagnosis")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if not total:
        print(f"error: no frames in {args.video}", file=sys.stderr)
        return 2
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    indices = np.linspace(0, total - 1, args.frames, dtype=int)
    frames = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if ok:
            frames.append((int(i), frame))
    cap.release()

    os.makedirs(args.out, exist_ok=True)
    detector = CourtLineDetector(args.model)

    print(f"\n{args.video}: {total} frames at {w}x{h}, sampling {len(frames)}")
    print(f"gate passes at line support >= {MIN_LINE_SUPPORT}\n")

    all_widths = []
    tally = {NORMAL: 0, WIDE: 0, FLAT: 0}
    for idx, frame in frames:
        kp_flat = detector.predict(frame)
        kp = np.asarray(kp_flat, dtype=np.float32).reshape(-1, 2)
        overall = line_support_score(frame, kp_flat)

        inside = sum(1 for x, y in kp if 0 <= x < w and 0 <= y < h)
        print(f"frame {idx}  overall {overall:.3f}  "
              f"{'PASS' if overall >= MIN_LINE_SUPPORT else 'FAIL'}  "
              f"({inside}/14 keypoints inside the frame)")

        canvas = frame.copy()
        for (a, b), name in zip(COURT_LINES, LINE_NAMES):
            per_line = line_support_score(frame, kp_flat, lines=((a, b),))
            pts, verdicts, widths = sample_points(frame, kp, a, b)
            all_widths += widths
            for v in verdicts:
                tally[v] += 1
            flat = sum(1 for v in verdicts if v == FLAT)
            wide = sum(1 for v in verdicts if v == WIDE)
            if verdicts and flat > len(verdicts) / 2:
                note = f"NOT ON PAINT ({flat}/{len(verdicts)} samples flat)"
            elif verdicts and wide > len(verdicts) / 2:
                note = f"wide paint ~{np.median(widths):.0f}px" if widths else "wide paint"
            elif widths:
                note = f"paint ~{np.median(widths):.0f}px"
            else:
                note = "no samples"
            print(f"    {name:<24} {per_line:5.2f}   {note}")

            cv2.line(canvas, tuple(kp[a].astype(int)), tuple(kp[b].astype(int)),
                     (255, 200, 0), 1)
            for pt, hit in pts:
                cv2.circle(canvas, tuple(pt.astype(int)), 3,
                           (0, 220, 0) if hit else (0, 0, 255), -1)

        for n, (x, y) in enumerate(kp):
            cv2.circle(canvas, (int(x), int(y)), 7, (255, 255, 255), 2)
            cv2.putText(canvas, str(n), (int(x) + 9, int(y) - 9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.putText(canvas, f"frame {idx}  support {overall:.3f}  green=paint red=miss",
                    (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        path = os.path.join(args.out, f"frame_{idx:06d}.png")
        cv2.imwrite(path, canvas)
        print(f"    -> {path}\n")

    total_pts = sum(tally.values())
    if not total_pts:
        print("no sample points; keypoints are probably outside the frame")
        return 1

    pct = {k: 100.0 * v / total_pts for k, v in tally.items()}
    print(f"\nsample points: {pct[NORMAL]:.0f}% on a normal line, "
          f"{pct[WIDE]:.0f}% on wide paint, {pct[FLAT]:.0f}% on no line at all")

    print("\nVERDICT")
    if pct[FLAT] > 50:
        print("  THE COURT PREDICTION IS WRONG.")
        print(f"  {pct[FLAT]:.0f}% of sampled points sit where there is no bright line")
        print("  at all - not wide paint, no paint. The keypoint model has placed")
        print("  the court somewhere it is not, and the low support score is a")
        print("  consequence of that rather than a threshold being too strict.")
        print("  Relaxing the gate would not help; it would let a wrong court through.")
    elif pct[WIDE] > 40:
        print("  THE GATE IS MIS-CALIBRATED FOR THIS FOOTAGE.")
        print(f"  {pct[WIDE]:.0f}% of sampled points sit on paint wider than the test's")
        print(f"  {OFFSET_PX}px sampling offset, so both 'surface' samples land on paint")
        print("  too. Check the overlays: if the keypoints follow the real lines, the")
        print("  geometry is fine and offset_px is the thing to change.")
    else:
        print("  Prediction looks sound on most lines; read the per-line rows above")
        print("  for the ones that failed.")

    print("\n  Confirm on the overlays either way: keypoints lying along the real")
    print("  court lines means the geometry is right whatever the score says.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
