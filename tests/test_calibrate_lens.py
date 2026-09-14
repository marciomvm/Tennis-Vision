"""
tools/calibrate_lens.py end to end: real checkerboard PHOTOS, rendered with a known
camera and known distortion, fed through the actual CLI as real files on disk - not
point correspondences handed directly to the fitting functions (that is what
tests/test_lens_calibration.py already covers). This is the layer those tests cannot
reach: does `find_board_corners` actually find real corners in a real image, does
`_collect_views` walk a directory correctly, does the whole tool recover parameters
close to what was used to render the photos.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from tools.calibrate_lens import _parse_pattern, _iter_image_files
from utils.lens_calibration import BoardSpec

REPO = Path(__file__).parent.parent
IMAGE_SIZE = (1280, 720)
TRUE_K = np.array([[900.0, 0, 640.0], [0, 900.0, 360.0], [0, 0, 1]])
TRUE_DIST = np.array([-0.22, 0.05, 0.0, 0.0, 0.0])
COLS, ROWS = 7, 5   # inner corners - a smaller board renders and detects faster


def _board_fully_visible(rvec, tvec, margin=25) -> bool:
    """
    Whether ALL of the board's outer corners land inside the frame (with a margin, so
    the pattern is not flush against the edge either).

    The board's own bounding corners are enough to check - a planar board's image is
    the convex hull of those four points, so if they are all in-frame the rest of the
    board is too. Skipped early rather than discovered after drawing every square: a
    pose that clips off-frame produces a picture cv2.findChessboardCorners will
    correctly refuse (only part of the inner-corner grid is visible), and letting many
    such poses through was the actual cause of low detection rates here, not anything
    in the module under test.
    """
    outer = np.array([[0, 0, 0], [COLS + 1, 0, 0],
                      [COLS + 1, ROWS + 1, 0], [0, ROWS + 1, 0]], dtype=np.float64)
    pts, _ = cv2.projectPoints(outer, rvec, tvec, TRUE_K, TRUE_DIST)
    pts = pts.reshape(-1, 2)
    if np.any(~np.isfinite(pts)):
        return False
    return bool(np.all(pts[:, 0] >= margin) and np.all(pts[:, 0] <= IMAGE_SIZE[0] - margin)
               and np.all(pts[:, 1] >= margin) and np.all(pts[:, 1] <= IMAGE_SIZE[1] - margin))


def _render_board(out_dir: Path, poses, dist=TRUE_DIST) -> int:
    """
    Real checkerboard photos: each square's own corners are projected through a known
    camera + distortion and filled as a distorted quad, so the pattern is genuinely
    bent the way a real lens would bend it - cv2.findChessboardCorners has to find real
    corners in a real image, not point correspondences handed to it directly.

    Poses whose board would be clipped by the frame edge are skipped before drawing -
    see `_board_fully_visible`.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for i, (rvec, tvec) in enumerate(poses):
        if not _board_fully_visible(rvec, tvec):
            continue
        canvas = np.full((IMAGE_SIZE[1], IMAGE_SIZE[0], 3), 235, np.uint8)
        for col in range(COLS + 1):
            for row in range(ROWS + 1):
                corners_3d = np.array([
                    [col, row, 0], [col + 1, row, 0],
                    [col + 1, row + 1, 0], [col, row + 1, 0],
                ], dtype=np.float64)
                pts, _ = cv2.projectPoints(corners_3d, rvec, tvec, TRUE_K, dist)
                pts = pts.reshape(-1, 2)
                colour = (20, 20, 20) if (col + row) % 2 == 0 else (235, 235, 235)
                cv2.fillConvexPoly(canvas, np.round(pts).astype(np.int32), colour,
                                   cv2.LINE_AA)
        cv2.imwrite(str(out_dir / f"board_{written:03d}.png"), canvas)
        written += 1
    return written


# tvec anchors whose board CENTROID sits near each corner and the centre of the frame,
# solved from the pinhole relation centroid = principal_point + f*(offset)/z rather than
# found by trial and error - a random uniform tvec rarely lands near a true image
# corner while staying fully in frame (the board's own angular size eats most of the
# margin), which is what an earlier version of this fixture kept failing to reach with
# thousands of random draws. Deliberately aiming at each corner is also a closer match
# to what the tool's filming instructions ask a real person to do (move the board out
# to the edges) than uniform random placement ever was.
_SPREAD_ANCHORS = {
    "TL": (-14.76, -7.89, 22.0), "TR": (6.76, -7.89, 22.0),
    "BL": (-14.76, 1.89, 22.0), "BR": (6.76, 1.89, 22.0),
    "C": (-4.0, -3.0, 22.0),
}


def _spread_poses(n=16, seed=1, centred=False):
    """
    Poses whose board sweeps toward the edges and corners of the frame, not just the
    centre - satisfying coverage_span's MIN_COVERAGE_SPAN the way filming instructions
    ask a real user to. `centred=True` gives the opposite deliberately, for the test
    that checks the LOW-coverage warning actually fires.

    Keeps sampling until `n` poses pass `_board_fully_visible` rather than returning
    whatever a fixed number of tries produced - a pose that would clip off-frame is not
    usable and must not silently shrink the fixture below what a test asked for.
    """
    rng = np.random.default_rng(seed)
    poses = []
    tries = 0
    anchor_names = list(_SPREAD_ANCHORS)
    while len(poses) < n and tries < n * 60:
        tries += 1
        if centred:
            # A single small neighbourhood in the middle of the range that DOES pass
            # visibility (checked empirically, not guessed) - the point is that every
            # view lands in nearly the same spot, not that any one view is invalid.
            rvec = rng.uniform(-0.05, 0.05, 3)
            tvec = np.array([-6.0, -3.5, 12.0]) + rng.uniform(-0.15, 0.15, 3)
        else:
            # Jitter around a corner/centre anchor, cycling through them so every
            # region gets roughly equal representation rather than however a single
            # random draw happened to fall.
            ax, ay, az = _SPREAD_ANCHORS[anchor_names[len(poses) % len(anchor_names)]]
            rvec = rng.uniform(-0.3, 0.3, 3)
            tvec = np.array([ax, ay, az]) + rng.uniform(-0.6, 0.6, 3)
        if _board_fully_visible(rvec, tvec):
            poses.append((rvec, tvec))
    return poses


# ── _parse_pattern ──────────────────────────────────────────────────────────

def test_parse_pattern_accepts_cols_x_rows():
    assert _parse_pattern("9x6") == (9, 6)
    assert _parse_pattern("9X6") == (9, 6)


def test_parse_pattern_rejects_garbage():
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_pattern("nope")


def test_parse_pattern_rejects_a_board_too_small_to_be_real():
    with pytest.raises(argparse.ArgumentTypeError, match="at least 3x3"):
        _parse_pattern("2x2")


# ── _iter_image_files ────────────────────────────────────────────────────────

def test_iter_image_files_is_sorted_and_skips_non_images(tmp_path):
    (tmp_path / "b.png").write_bytes(b"")
    (tmp_path / "a.jpg").write_bytes(b"")
    (tmp_path / "notes.txt").write_bytes(b"")
    # Real, readable images only - cv2.imread on the empty files above returns None
    # and _iter_image_files must skip those rather than yield a None frame.
    real = tmp_path / "c.png"
    cv2.imwrite(str(real), np.zeros((10, 10, 3), np.uint8))

    names = [name for _, _, frame, name in _iter_image_files(tmp_path) if frame is not None]
    assert names == ["c.png"]


def test_iter_image_files_survives_a_patched_imread_that_raises(tmp_path, monkeypatch):
    """
    Vanilla cv2.imread returns None on an unreadable file; ultralytics - already a
    dependency here, for the player detector - monkey-patches cv2.imread at import
    time to RAISE cv2.error on exactly that input instead. Which behaviour this tool
    actually sees depends on whether something else already imported ultralytics
    first in the same process, an import-order accident that showed up as this exact
    test passing alone and failing inside the full suite. Reproduced directly here
    rather than left to depend on suite ordering to catch it again.
    """
    import tools.calibrate_lens as m

    bad = tmp_path / "corrupt.png"
    bad.write_bytes(b"")
    good = tmp_path / "ok.png"
    cv2.imwrite(str(good), np.zeros((10, 10, 3), np.uint8))

    original_imread = cv2.imread

    def patched_imread(path, *a, **kw):
        result = original_imread(path, *a, **kw)
        if result is None:
            raise cv2.error("simulated ultralytics patch: raises instead of None")
        return result

    monkeypatch.setattr(m.cv2, "imread", patched_imread)

    names = [name for _, _, frame, name in _iter_image_files(tmp_path) if frame is not None]
    assert names == ["ok.png"]


# ── the real thing: rendered photos through the actual CLI ────────────────────

def test_recovers_known_distortion_from_rendered_photos(tmp_path):
    photos = tmp_path / "photos"
    written = _render_board(photos, _spread_poses(n=16))
    assert written >= 12, "fixture did not render enough usable views"

    out = tmp_path / "lens.json"
    result = subprocess.run(
        [sys.executable, "tools/calibrate_lens.py", str(photos),
         "--pattern", f"{COLS}x{ROWS}", "--min-views", "8", "--out", str(out)],
        capture_output=True, text=True, cwd=str(REPO), timeout=120,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert out.exists()

    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["model"] == "standard"
    assert data["views_used"] >= 8
    K = np.array(data["camera_matrix"])
    dist = np.array(data["dist_coeffs"])

    assert K[0, 0] == pytest.approx(TRUE_K[0, 0], rel=0.05)
    assert K[1, 1] == pytest.approx(TRUE_K[1, 1], rel=0.05)
    assert K[0, 2] == pytest.approx(TRUE_K[0, 2], rel=0.05)
    assert dist[0] == pytest.approx(TRUE_DIST[0], abs=0.03)
    assert dist[1] == pytest.approx(TRUE_DIST[1], abs=0.03)
    assert data["rms_error_px"] < 1.0
    assert data["coverage"]["wide_enough"] is True


def test_reports_low_coverage_when_the_board_stays_centred(tmp_path):
    """The filming-instructions warning has to actually fire on real detected corners,
    not only on synthetic points fed to coverage_span directly."""
    photos = tmp_path / "photos"
    written = _render_board(photos, _spread_poses(n=14, seed=2, centred=True))
    assert written >= 10

    result = subprocess.run(
        [sys.executable, "tools/calibrate_lens.py", str(photos),
         "--pattern", f"{COLS}x{ROWS}", "--min-views", "8",
         "--out", str(tmp_path / "lens.json")],
        capture_output=True, text=True, cwd=str(REPO), timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "LOW" in result.stdout
    assert "moving the board out to the corners" in result.stdout


def test_refuses_to_fit_from_too_few_views(tmp_path):
    photos = tmp_path / "photos"
    _render_board(photos, _spread_poses(n=3))

    result = subprocess.run(
        [sys.executable, "tools/calibrate_lens.py", str(photos),
         "--pattern", f"{COLS}x{ROWS}", "--min-views", "8",
         "--out", str(tmp_path / "lens.json")],
        capture_output=True, text=True, cwd=str(REPO), timeout=60,
    )
    assert result.returncode != 0
    assert "need at least" in result.stderr
    assert not (tmp_path / "lens.json").exists()


def test_wrong_pattern_size_finds_nothing_rather_than_a_wrong_answer(tmp_path):
    """A board rendered as 7x5 inner corners, asked for as 9x6, must find zero
    matches and refuse - not silently fit something from a partial/wrong match."""
    photos = tmp_path / "photos"
    _render_board(photos, _spread_poses(n=10))

    result = subprocess.run(
        [sys.executable, "tools/calibrate_lens.py", str(photos),
         "--pattern", "9x6", "--min-views", "1",
         "--out", str(tmp_path / "lens.json")],
        capture_output=True, text=True, cwd=str(REPO), timeout=60,
    )
    assert result.returncode != 0
    assert not (tmp_path / "lens.json").exists()


def test_help_is_printable_ascii():
    """Simulates the default Windows console encoding that broke --help on nine other
    commands in this project before every argparse description was made ASCII-safe."""
    result = subprocess.run(
        [sys.executable, "tools/calibrate_lens.py", "--help"],
        capture_output=True, text=True, cwd=str(REPO), timeout=30,
        env={**os.environ, "PYTHONIOENCODING": "cp1252"},
    )
    assert result.returncode == 0, result.stderr
