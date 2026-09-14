"""
Cheap motion-based pre-segmentation: the decision logic that finds "something is
happening" windows, tested on synthetic signals rather than real video so the ordering
of merge -> duration filter -> pad is checkable exactly.
"""
import numpy as np
import pytest

from utils.activity_segments import (
    Segment,
    auto_threshold,
    build_roi_mask,
    compute_activity_signal,
    find_segments,
    smooth,
)

FPS = 30.0


def _signal(*runs: tuple[int, int], length: int = 300, active: float = 1.0,
           idle: float = 0.0) -> np.ndarray:
    """A synthetic activity trace: `idle` everywhere except inside each (start, end)."""
    s = np.full(length, idle, dtype=np.float64)
    for a, b in runs:
        s[a:b] = active
    return s


# ── find_segments: the ordering of merge, duration filter and pad ─────────────────────

def test_a_single_run_is_kept_and_padded():
    signal = _signal((100, 130), length=300)
    segments = find_segments(signal, FPS, threshold=0.5, min_gap_s=1.0, pad_s=1.0,
                             min_duration_s=0.5)
    assert len(segments) == 1
    seg = segments[0]
    assert seg.start_frame == 100 - int(1.0 * FPS)
    assert seg.end_frame == 130 + int(1.0 * FPS)


def test_nearby_runs_merge_across_the_gap():
    """A rally's own lull between strokes must not become a cut."""
    signal = _signal((100, 120), (135, 160), length=300)   # 15-frame gap = 0.5s at 30fps
    segments = find_segments(signal, FPS, threshold=0.5, min_gap_s=1.0, pad_s=0.0,
                             min_duration_s=0.1)
    assert len(segments) == 1
    assert segments[0].start_frame == 100
    assert segments[0].end_frame == 160


def test_a_gap_longer_than_tolerance_stays_two_segments():
    signal = _signal((100, 120), (220, 240), length=400)   # 100-frame gap
    segments = find_segments(signal, FPS, threshold=0.5, min_gap_s=1.0, pad_s=0.0,
                             min_duration_s=0.1)
    assert len(segments) == 2


def test_a_short_noise_spike_is_dropped():
    """Below min_duration, before padding - a bird crossing the court, not a stroke."""
    signal = _signal((100, 104), length=300)   # 4 frames = 0.13s at 30fps
    segments = find_segments(signal, FPS, threshold=0.5, min_gap_s=1.0, pad_s=2.0,
                             min_duration_s=0.5)
    assert segments == []


def test_padding_cannot_rescue_a_run_the_duration_filter_already_dropped():
    """
    Padding is applied AFTER the duration filter specifically so a big pad cannot launder
    a noise spike into something that looks long enough to keep.
    """
    signal = _signal((100, 103), length=300)
    generous_pad = find_segments(signal, FPS, threshold=0.5, min_gap_s=1.0, pad_s=5.0,
                                 min_duration_s=0.5)
    assert generous_pad == [], "a 3-frame spike must not survive just because pad_s is big"


def test_padding_is_clipped_to_the_signal_bounds():
    signal = _signal((0, 10), (290, 300), length=300)
    segments = find_segments(signal, FPS, threshold=0.5, min_gap_s=1.0, pad_s=2.0,
                             min_duration_s=0.1)
    assert segments[0].start_frame == 0
    assert segments[-1].end_frame == 300


def test_padding_does_not_silently_fuse_two_segments():
    """
    Two segments close enough that their padding would overlap are capped against each
    other and reported as adjacent, not merged - a caller counting segments should not
    be surprised by a merge it never asked min_gap_s for.
    """
    signal = _signal((100, 110), (150, 160), length=300)   # 40-frame gap
    segments = find_segments(signal, FPS, threshold=0.5, min_gap_s=1.0, pad_s=3.0,
                             min_duration_s=0.1)
    assert len(segments) == 2
    assert segments[0].end_frame <= segments[1].start_frame
    assert segments[0].end_frame == segments[1].start_frame, (
        "padding should meet exactly in the middle of the gap, not fall short of it"
    )


def test_no_activity_at_all_returns_nothing():
    signal = _signal(length=300)
    assert find_segments(signal, FPS, threshold=0.5) == []


def test_activity_for_the_whole_clip_returns_one_segment():
    signal = _signal((0, 300), length=300)
    segments = find_segments(signal, FPS, threshold=0.5, pad_s=1.0)
    assert len(segments) == 1
    assert segments[0].start_frame == 0
    assert segments[0].end_frame == 300


def test_empty_signal_does_not_crash():
    assert find_segments(np.array([]), FPS, threshold=0.5) == []


def test_segment_reports_peak_and_mean_from_the_unpadded_window():
    signal = _signal(length=300, idle=0.1)
    signal[100:110] = np.linspace(0.5, 2.0, 10)
    segments = find_segments(signal, FPS, threshold=0.3, pad_s=0.0, min_duration_s=0.1)
    assert segments[0].peak_activity == pytest.approx(2.0)
    assert segments[0].mean_activity == pytest.approx(float(np.linspace(0.5, 2.0, 10).mean()))


def test_segment_duration_and_dict_round_trip():
    seg = Segment(start_frame=30, end_frame=90, start_s=1.0, end_s=3.0,
                 peak_activity=1.5, mean_activity=0.8)
    assert seg.duration_s == pytest.approx(2.0)
    d = seg.as_dict()
    assert d["start_frame"] == 30 and d["end_frame"] == 90
    assert d["duration_s"] == 2.0


# ── auto_threshold ──────────────────────────────────────────────────────────────────

def test_auto_threshold_sits_between_idle_and_active_levels():
    signal = _signal((100, 200), length=1000, active=2.0, idle=0.2)
    t = auto_threshold(signal)
    assert 0.2 < t < 2.0


def test_auto_threshold_of_empty_signal_is_zero():
    assert auto_threshold(np.array([])) == 0.0


# ── smooth ───────────────────────────────────────────────────────────────────────────

def test_smooth_reduces_single_frame_spikes():
    signal = np.zeros(50)
    signal[25] = 10.0
    smoothed = smooth(signal, window=5)
    assert smoothed.max() < 10.0
    assert smoothed.sum() == pytest.approx(signal.sum(), rel=0.05)


def test_smooth_with_window_one_is_a_no_op():
    signal = np.array([1.0, 5.0, 2.0])
    assert np.array_equal(smooth(signal, window=1), signal)
    assert np.array_equal(smooth(signal, window=0), signal)


def test_smooth_of_empty_signal_does_not_crash():
    assert len(smooth(np.array([]), window=5)) == 0


# ── build_roi_mask ───────────────────────────────────────────────────────────────────

def test_roi_mask_is_sized_to_the_downscaled_frame():
    mask = build_roi_mask(None, frame_size=(1920, 1080), downscale=4)
    assert mask.shape == (270, 480)


def test_roi_mask_with_no_polygon_covers_everything():
    mask = build_roi_mask(None, frame_size=(400, 200), downscale=4)
    assert (mask > 0).all()


def test_roi_mask_polygon_is_scaled_down_not_the_source_pixels():
    # A polygon covering the left half of a 400x200 frame, at downscale=4 -> 100x50.
    polygon = np.array([[[0, 0]], [[200, 0]], [[200, 200]], [[0, 200]]], dtype=np.float64)
    mask = build_roi_mask(polygon, frame_size=(400, 200), downscale=4)
    assert mask.shape == (50, 100)
    assert mask[25, 10] > 0        # inside the left half
    assert mask[25, 90] == 0       # outside it


# ── compute_activity_signal: masking and streaming behaviour on tiny synthetic clips ──

def _write_clip(path, frames, fps=30.0):
    import cv2

    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()


def test_activity_signal_first_frame_is_zero(tmp_path):
    import cv2

    frames = [np.full((40, 40, 3), 50, np.uint8) for _ in range(5)]
    frames[3][:, :] = 200
    path = tmp_path / "clip.mp4"
    _write_clip(path, frames)

    signal, fps, size = compute_activity_signal(str(path), downscale=1)
    assert signal[0] == 0.0
    assert len(signal) == 5
    assert fps == pytest.approx(30.0, abs=1.0)
    assert size == (40, 40)


def test_activity_signal_ignores_motion_outside_the_mask(tmp_path):
    frames = []
    for i in range(4):
        f = np.full((40, 40, 3), 50, np.uint8)
        # Motion confined to the RIGHT half only, growing each frame.
        f[:, 20:] = 50 + i * 40
        frames.append(f)
    path = tmp_path / "clip.mp4"
    _write_clip(path, frames)

    left_half = np.zeros((40, 40), np.uint8)
    left_half[:, :20] = 255

    signal, _, _ = compute_activity_signal(str(path), roi_mask=left_half, downscale=1)
    # Near zero, not exactly zero: mp4v is a lossy codec, so even a "constant" region
    # picks up a little compression noise across frames. The real assertion is the
    # contrast against test_activity_signal_sees_motion_inside_the_mask, which uses the
    # identical frames with the mask flipped and sees signal two orders of magnitude
    # larger than this.
    assert signal[1:].max() < 2.0


def test_activity_signal_sees_motion_inside_the_mask(tmp_path):
    frames = []
    for i in range(4):
        f = np.full((40, 40, 3), 50, np.uint8)
        f[:, :20] = 50 + i * 40      # motion in the LEFT half this time
        frames.append(f)
    path = tmp_path / "clip.mp4"
    _write_clip(path, frames)

    left_half = np.zeros((40, 40), np.uint8)
    left_half[:, :20] = 255

    signal, _, _ = compute_activity_signal(str(path), roi_mask=left_half, downscale=1)
    assert signal[1] > 0.0


def test_activity_signal_rejects_a_wrongly_sized_mask(tmp_path):
    frames = [np.zeros((40, 40, 3), np.uint8) for _ in range(2)]
    path = tmp_path / "clip.mp4"
    _write_clip(path, frames)

    wrong = np.zeros((10, 10), np.uint8)
    with pytest.raises(ValueError, match="expected"):
        compute_activity_signal(str(path), roi_mask=wrong, downscale=1)


def test_activity_signal_reports_progress(tmp_path):
    frames = [np.full((20, 20, 3), i % 256, np.uint8) for i in range(700)]
    path = tmp_path / "clip.mp4"
    _write_clip(path, frames)

    calls = []
    compute_activity_signal(str(path), downscale=1, on_progress=lambda i, t: calls.append(i))
    assert calls[-1] == 700
    assert any(c == 300 for c in calls) or any(c == 600 for c in calls)


def test_a_missing_video_raises_rather_than_returning_nothing(tmp_path):
    with pytest.raises(ValueError, match="could not open"):
        compute_activity_signal(str(tmp_path / "does_not_exist.mp4"))
