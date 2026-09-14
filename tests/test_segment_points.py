"""
The --target-fps path of tools/segment_points.py: resample an out-of-range clip to
something the pipeline's own fps gate accepts, and verify the result was actually
achieved rather than assumed from ffmpeg not erroring.

`resolve_reencode` is tested as pure logic. `_probe_output_fps` is tested against a
mocked subprocess for the parsing edge cases, and then once against a real ffmpeg/ffprobe
round trip on an actual 60fps clip, because a flag-ordering mistake in the ffmpeg command
is exactly the kind of bug a mocked test cannot catch.
"""
import shutil
import subprocess

import cv2
import numpy as np
import pytest

from tools.segment_points import _extract_segment, _probe_output_fps, resolve_reencode
from utils.activity_segments import Segment

HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


# ── resolve_reencode: pure logic ────────────────────────────────────────────────────

def test_target_fps_forces_reencode_on():
    reencode, forced = resolve_reencode(target_fps=30.0, reencode=False)
    assert reencode is True
    assert forced is True


def test_target_fps_with_reencode_already_on_is_not_reported_as_forced():
    """Nothing changed, so there is nothing to print about."""
    reencode, forced = resolve_reencode(target_fps=30.0, reencode=True)
    assert reencode is True
    assert forced is False


def test_no_target_fps_leaves_reencode_untouched():
    assert resolve_reencode(target_fps=None, reencode=False) == (False, False)
    assert resolve_reencode(target_fps=None, reencode=True) == (True, False)


def test_target_fps_of_zero_does_not_force_anything():
    """argparse's `const=30.0` means bare --target-fps is never 0, but a defensive
    caller passing 0.0 explicitly should not force a re-encode for a no-op request."""
    assert resolve_reencode(target_fps=0.0, reencode=False) == (False, False)


# ── _probe_output_fps: parsing, against a mocked subprocess ────────────────────────

def _mock_ffprobe(monkeypatch, stdout: str):
    def fake_run(cmd, capture_output, text):
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")
    monkeypatch.setattr(subprocess, "run", fake_run)


def test_probe_parses_a_simple_fraction(monkeypatch, tmp_path):
    _mock_ffprobe(monkeypatch, "30/1\n")
    assert _probe_output_fps(tmp_path / "x.mp4") == pytest.approx(30.0)


def test_probe_parses_an_ntsc_style_fraction(monkeypatch, tmp_path):
    _mock_ffprobe(monkeypatch, "60000/1001\n")
    assert _probe_output_fps(tmp_path / "x.mp4") == pytest.approx(59.94, abs=0.01)


def test_probe_handles_zero_over_zero_without_crashing(monkeypatch, tmp_path):
    """ffprobe reports 0/0 when it cannot determine a rate - a file with no video
    stream, or one ffprobe could not parse. Must return None, not raise or return inf."""
    _mock_ffprobe(monkeypatch, "0/0\n")
    assert _probe_output_fps(tmp_path / "x.mp4") is None


def test_probe_handles_empty_output(monkeypatch, tmp_path):
    _mock_ffprobe(monkeypatch, "")
    assert _probe_output_fps(tmp_path / "x.mp4") is None


def test_probe_handles_garbage_output(monkeypatch, tmp_path):
    _mock_ffprobe(monkeypatch, "not a fraction\n")
    assert _probe_output_fps(tmp_path / "x.mp4") is None


# ── real ffmpeg round trip: the part a mock cannot catch ────────────────────────────

def _write_60fps_clip(path, seconds=1.0, size=(64, 48)):
    w, h = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 60.0, (w, h))
    for i in range(int(60 * seconds)):
        frame = np.full((h, w, 3), i % 256, np.uint8)
        writer.write(frame)
    writer.release()


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg/ffprobe not on PATH")
def test_extracting_with_target_fps_actually_changes_the_rate(tmp_path):
    """
    The test a flag-ordering mistake would fail. If -r/-fps_mode ever regress to being
    placed as an INPUT option instead of an output one, or the flag name changes in a
    future ffmpeg and is silently ignored, this is what catches it - a mocked subprocess
    call cannot, because it would still report ffmpeg exited 0.
    """
    source = tmp_path / "source_60fps.mp4"
    _write_60fps_clip(source, seconds=1.0)

    seg = Segment(start_frame=0, end_frame=60, start_s=0.0, end_s=1.0,
                 peak_activity=1.0, mean_activity=1.0)
    out = tmp_path / "out_30fps.mp4"
    ok, error = _extract_segment(str(source), seg, out, reencode=True, target_fps=30.0)

    assert ok, error
    actual = _probe_output_fps(out)
    assert actual == pytest.approx(30.0, abs=0.1)

    cap = cv2.VideoCapture(str(out))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    # 60fps source, 1s, resampled to 30fps -> close to 30 frames, not close to 60.
    assert 20 <= frame_count <= 35


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg/ffprobe not on PATH")
def test_extracting_without_target_fps_keeps_the_source_rate(tmp_path):
    """The control case: re-encoding alone, with no --target-fps, must not silently
    resample anything - the flag's absence has to mean "leave the rate alone"."""
    source = tmp_path / "source_60fps.mp4"
    _write_60fps_clip(source, seconds=1.0)

    seg = Segment(start_frame=0, end_frame=60, start_s=0.0, end_s=1.0,
                 peak_activity=1.0, mean_activity=1.0)
    out = tmp_path / "out_same_fps.mp4"
    ok, error = _extract_segment(str(source), seg, out, reencode=True, target_fps=None)

    assert ok, error
    actual = _probe_output_fps(out)
    assert actual == pytest.approx(60.0, abs=0.5)
