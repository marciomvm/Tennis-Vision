"""
utils/activity_segments.py
---------------------------
Find the stretches of a long recording where something is happening, so the expensive
part of this pipeline never has to look at the rest.

Why this exists
----------------
A club session is filmed as one continuous take: an hour of tape for maybe fifteen
minutes of actual rallies, the remainder being players walking to the baseline, picking
up balls, adjusting strings, arguing a call. Every detector this project runs - YOLO,
TrackNet, the pose model - costs the same per frame whether the frame shows a rally or
an empty court, so an hour of raw footage is paid for in full even though most of it is
not tennis.

What this measures
-------------------
Frame-to-frame motion, restricted to the calibrated court (so a rally on the NEXT court
does not count), at a heavily reduced resolution. This is orders of magnitude cheaper
than any of the real detectors: no model, no GPU, one absdiff per frame. It cannot tell a
rally from a player jogging to the net, and it is not asked to - see the docstring on
`find_segments` for what the gap-merge and padding are actually doing about that.

What this does NOT do
----------------------
It does not find one video per point. A rally is not one continuous burst of motion: it
is a burst per stroke, with the ball's flight time as a lull in between, and treating
every gap as a boundary would shred a single point into several files. The merge
tolerance is deliberately generous, so this is a coarse filter that keeps whole rallies
together and drops the stretches with no motion at all for several seconds. If exact
point boundaries matter downstream, run the real event detector on the surviving
segments - which are now a small fraction of the original clip - rather than trying to
get them from motion alone.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# ── the pure decision logic - no file I/O, so it is testable on synthetic arrays ──────

# How long a gap of no motion is allowed to sit INSIDE one retained segment before it
# splits into two. Set well above a single stroke's ball-flight time (typically under a
# second even for a deep lob) and well below how long a player pauses between points
# (walking back, receiving a second ball) - not a tuned constant, just wide enough that
# a real rally's own lulls cannot trigger it.
DEFAULT_MIN_GAP_S = 2.0

# Context kept on each side of a retained segment. Generous on purpose: the cost of
# padding too much is a few extra seconds of dead footage fed to the real detectors
# downstream, and the cost of padding too little is silently losing the first or last
# stroke of a point, which is the kind of error this project does not accept elsewhere
# and should not accept here either.
DEFAULT_PAD_S = 1.5

# Below this duration a run is noise - a gust of wind on the net, a bird crossing the
# court, a single frame of encoder artefact - rather than a stroke. Applied to the RAW
# run before padding, so padding cannot launder a noise spike into something that looks
# long enough to keep.
DEFAULT_MIN_DURATION_S = 0.6


@dataclass(frozen=True)
class Segment:
    """One retained stretch of the recording, in both frames and seconds."""

    start_frame: int
    end_frame: int              # exclusive, like a Python slice
    start_s: float
    end_s: float
    peak_activity: float
    mean_activity: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    def as_dict(self) -> dict:
        return {
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_s": round(self.start_s, 3),
            "end_s": round(self.end_s, 3),
            "duration_s": round(self.duration_s, 3),
            "peak_activity": round(self.peak_activity, 4),
            "mean_activity": round(self.mean_activity, 4),
        }


def auto_threshold(signal: np.ndarray, low_pct: float = 10.0, high_pct: float = 90.0) -> float:
    """
    A threshold with no tuning, from the signal's own spread.

    The midpoint between two percentiles rather than one fixed number, because "how
    bright is motion" depends on the camera, the light and the court surface and has no
    universal value. This still needs the low percentile to be genuinely idle and the
    high one to be genuinely active - on a recording that is either always moving
    (someone practising serves non-stop) or never moving (nobody arrives), there is no
    threshold that will separate what was not filmed to be separated, and the caller
    should look at the reported percentiles rather than trust the number blindly.
    """
    if len(signal) == 0:
        return 0.0
    low = float(np.percentile(signal, low_pct))
    high = float(np.percentile(signal, high_pct))
    return (low + high) / 2.0


def find_segments(
    signal: np.ndarray,
    fps: float,
    threshold: float,
    min_gap_s: float = DEFAULT_MIN_GAP_S,
    pad_s: float = DEFAULT_PAD_S,
    min_duration_s: float = DEFAULT_MIN_DURATION_S,
) -> list[Segment]:
    """
    Turn a per-frame activity signal into the segments worth keeping.

    In order, and in this order deliberately:

      1. threshold into active / idle frames.
      2. merge active runs separated by less than `min_gap_s` of idle time - a real
         rally's own between-stroke lulls must not become a cut.
      3. drop runs shorter than `min_duration_s`, measured BEFORE padding - padding a
         noise spike does not make it a stroke, it only makes it a longer noise spike.
      4. pad the survivors by `pad_s` on each side and clip to the signal's bounds.

    Padding after the duration filter, not before, is what keeps a genuine short rally
    (say, a single serve practised alone) from being padded into passing a filter it
    should have failed, while still giving every real segment the context around it.

    Args:
        signal:     per-frame activity, one value per frame, already smoothed by the
                    caller - this function does no smoothing of its own.
        fps:        frames per second, to convert the second-based parameters to frames.
        threshold:  frames strictly above this count as active.
        min_gap_s, pad_s, min_duration_s: see the module docstring for what each guards
                    against.

    Returns:
        Segments in frame order, non-overlapping (padding that would overlap the next
        segment is capped rather than allowed to merge them silently - two segments
        that pad into contact are reported as adjacent, not fused, so a caller counting
        them is not surprised by a merge it never asked for).
    """
    if len(signal) == 0:
        return []

    active = signal > threshold
    raw_runs: list[tuple[int, int]] = []
    start = None
    for i, is_active in enumerate(active):
        if is_active and start is None:
            start = i
        elif not is_active and start is not None:
            raw_runs.append((start, i))
            start = None
    if start is not None:
        raw_runs.append((start, len(active)))

    if not raw_runs:
        return []

    # Step 2: merge runs whose GAP (not whose combined span) is inside the tolerance.
    min_gap_frames = min_gap_s * fps
    merged: list[list[int]] = [list(raw_runs[0])]
    for run_start, run_end in raw_runs[1:]:
        if run_start - merged[-1][1] <= min_gap_frames:
            merged[-1][1] = run_end
        else:
            merged.append([run_start, run_end])

    # Step 3: drop short RAW runs, before any padding is added.
    min_duration_frames = min_duration_s * fps
    survivors = [(a, b) for a, b in merged if (b - a) >= min_duration_frames]
    if not survivors:
        return []

    # Step 4: pad, clip to bounds, and cap against the neighbour so padding two
    # adjacent segments cannot silently overlap them. The cap is the MIDPOINT of the
    # gap between two raw (unpadded) survivors, shared by both sides - segment i's
    # right bound and segment i+1's left bound are the same number, so with enough
    # padding to reach it the two segments meet exactly there and never cross.
    # Capping each side independently against the neighbour's own raw edge (rather
    # than a shared midpoint) was tried first and let two segments each claim the
    # whole gap, overlapping in the middle - this is what that bug looked like.
    pad_frames = pad_s * fps
    total = len(signal)
    n = len(survivors)
    bounds = [0] + [
        (survivors[k][1] + survivors[k + 1][0]) // 2 for k in range(n - 1)
    ] + [total]

    segments: list[Segment] = []
    for idx, (a, b) in enumerate(survivors):
        lo, hi = bounds[idx], bounds[idx + 1]
        padded_a = max(lo, int(round(a - pad_frames)))
        padded_b = min(hi, int(round(b + pad_frames)))
        window = signal[a:b]
        segments.append(Segment(
            start_frame=padded_a,
            end_frame=padded_b,
            start_s=padded_a / fps,
            end_s=padded_b / fps,
            peak_activity=float(window.max()),
            mean_activity=float(window.mean()),
        ))
    return segments


def smooth(signal: np.ndarray, window: int) -> np.ndarray:
    """
    A short moving average, so single-frame noise cannot register as an active run.

    `window` is in frames, not seconds - the caller converts, since this function has no
    opinion about fps. A window of 1 or less is returned unchanged rather than raising:
    a caller building this from a `--smooth-s 0` override should get "no smoothing",
    not an error.
    """
    if window <= 1 or len(signal) == 0:
        return signal
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(signal, kernel, mode="same")


# ── the expensive part - streamed, one frame at a time, no big buffer ─────────────────

def build_roi_mask(roi_polygon, frame_size: tuple[int, int], downscale: int) -> np.ndarray:
    """
    The court region as a mask at the ANALYSIS resolution, not the video's own.

    Built once at the working size rather than built full-size and shrunk afterwards,
    because a polygon scaled down keeps its shape while a full-resolution mask shrunk
    by area-averaging would blur its edge into fractional values with no clean meaning
    for a boolean test.
    """
    width, height = frame_size
    small_w, small_h = max(1, width // downscale), max(1, height // downscale)
    mask = np.zeros((small_h, small_w), dtype=np.uint8)
    if roi_polygon is not None:
        scaled = np.asarray(roi_polygon, dtype=np.float64).reshape(-1, 1, 2) / downscale
        cv2.fillPoly(mask, [np.round(scaled).astype(np.int32)], 255)
    else:
        mask[:] = 255
    return mask


def compute_activity_signal(
    video_path: str,
    roi_mask: np.ndarray | None = None,
    downscale: int = 4,
    on_progress=None,
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """
    Stream a video and return one activity value per frame.

    Streamed deliberately: this reads one frame, downsamples it, differences it against
    the previous one, and discards it, so it costs a few hundred MB regardless of how
    long the video is - the opposite of `utils.video_utils.read_video`, which this tool
    exists partly to make less necessary by shrinking what that function is ever asked
    to load.

    `roi_mask`, if given, must already be sized for `downscale` - see `build_roi_mask`.
    Frames outside the mask do not affect the signal at all: a rally on the next court
    contributes nothing to it.

    Args:
        on_progress: optional callback(frames_done, frames_total_or_0). Called every
            few hundred frames, not every frame - the caller decides what "progress"
            means to print or log; this function has no opinion about that.

    Returns:
        (signal, fps, (width, height)) where signal[i] is 0.0 for frame 0 (there is no
        previous frame to compare it to) and the mean absolute grey-level difference
        from the previous frame for every frame after it.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"could not open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_hint = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    small_w, small_h = max(1, width // downscale), max(1, height // downscale)

    if roi_mask is not None and roi_mask.shape != (small_h, small_w):
        raise ValueError(
            f"roi_mask is {roi_mask.shape[1]}x{roi_mask.shape[0]}, expected "
            f"{small_w}x{small_h} at downscale={downscale}"
        )
    mask_bool = None if roi_mask is None else (roi_mask > 0)

    values: list[float] = []
    prev_gray = None
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            small = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            if prev_gray is None:
                values.append(0.0)
            else:
                diff = cv2.absdiff(gray, prev_gray)
                region = diff if mask_bool is None else diff[mask_bool]
                values.append(float(region.mean()) if region.size else 0.0)
            prev_gray = gray
            i += 1
            if on_progress is not None and i % 300 == 0:
                on_progress(i, total_hint)
    finally:
        cap.release()

    if on_progress is not None:
        on_progress(i, total_hint)
    return np.asarray(values, dtype=np.float64), float(fps), (width, height)
