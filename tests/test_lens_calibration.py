"""
utils/lens_calibration.py: fitting a camera's lens distortion from a checkerboard, and
applying it back to points.

Every fitting test here works from SYNTHETIC views: a known camera matrix and known
distortion coefficients project a checkerboard's corners into several poses, and the fit
must recover what was put in. That is deliberate - it tests the maths this module is
actually responsible for (calibrateCamera / fisheye.calibrate wrapped correctly, and
undistort_points inverting what a lens does) without depending on real corner detection
in a real photo, which is a separately well-tested piece of OpenCV. It is also what
caught two real bugs before this module ever touched a camera: OpenCV's Python bindings
moved the fisheye CALIB_* flags between `cv2.fisheye.CALIB_*` and top-level `cv2.CALIB_*`
across versions, and `cv2.fisheye.calibrate`/`projectPoints` want each view shaped
(1, N, 3) rather than the (N, 1, 3) `cv2.calibrateCamera` takes - passing the wrong one
does not raise a clear shape error, it fails deep inside OpenCV with an unrelated-looking
message. A test that only checked "does it run" on real images would not have caught
either without a lot of confused debugging against real footage.
"""
import numpy as np
import pytest
import cv2

from utils.lens_calibration import (
    MODEL_FISHEYE,
    MODEL_STANDARD,
    BoardSpec,
    LensCalibration,
    coverage_span,
    fit_fisheye,
    fit_standard,
    reprojection_error_per_view,
    undistort_points,
)

IMAGE_SIZE = (1920, 1080)
TRUE_K = np.array([[1400.0, 0, 960.0], [0, 1400.0, 540.0], [0, 0, 1]])
BOARD = BoardSpec(inner_corners=(9, 6), square_size=1.0)


def _synthesize(distort, n=18, seed=0):
    """n board poses, projected through TRUE_K + `distort`, kept inside the frame."""
    rng = np.random.default_rng(seed)
    obj_template = BOARD.object_points()
    object_points, image_points = [], []
    tries = 0
    while len(image_points) < n and tries < n * 30:
        tries += 1
        rvec = rng.uniform(-0.5, 0.5, 3)
        tvec = np.array([rng.uniform(-4, 4), rng.uniform(-2.5, 2.5), rng.uniform(9, 16)])
        pts = distort(obj_template, rvec, tvec)
        if pts is None or np.any(pts[:, 0] < 0) or np.any(pts[:, 0] > IMAGE_SIZE[0]) \
                or np.any(pts[:, 1] < 0) or np.any(pts[:, 1] > IMAGE_SIZE[1]):
            continue
        object_points.append(obj_template.astype(np.float32))
        image_points.append(pts.astype(np.float32))
    return object_points, image_points


def _project_standard(dist_coeffs):
    def project(obj, rvec, tvec):
        pts, _ = cv2.projectPoints(obj, rvec, tvec, TRUE_K, dist_coeffs)
        return pts.reshape(-1, 2)
    return project


def _project_fisheye(dist_coeffs):
    def project(obj, rvec, tvec):
        pts, _ = cv2.fisheye.projectPoints(
            obj.reshape(-1, 1, 3).astype(np.float64), rvec, tvec, TRUE_K, dist_coeffs)
        return pts.reshape(-1, 2)
    return project


# ── BoardSpec ────────────────────────────────────────────────────────────────

def test_object_points_shape_and_spacing():
    board = BoardSpec(inner_corners=(9, 6), square_size=2.5)
    pts = board.object_points()
    assert pts.shape == (54, 3)
    assert np.all(pts[:, 2] == 0)          # flat board, Z=0
    xs = sorted(set(pts[:, 0]))
    assert xs[1] - xs[0] == pytest.approx(2.5)   # square_size honoured


# ── fit_standard: recovers known parameters exactly from noise-free views ─────

def test_fit_standard_recovers_known_camera_and_distortion():
    dist_true = np.array([-0.28, 0.09, 0.001, -0.0005, -0.01])
    obj, img = _synthesize(_project_standard(dist_true))
    assert len(img) >= 10

    rms, K, dist, *_ = fit_standard(obj, img, IMAGE_SIZE)

    assert rms < 0.5
    assert K[0, 0] == pytest.approx(TRUE_K[0, 0], abs=15)
    assert K[1, 1] == pytest.approx(TRUE_K[1, 1], abs=15)
    assert K[0, 2] == pytest.approx(TRUE_K[0, 2], abs=15)
    np.testing.assert_allclose(dist, dist_true, atol=0.02)


def test_fit_standard_with_zero_distortion_recovers_zero():
    """The boring but important case: a genuinely rectilinear lens must not be handed
    fabricated distortion because the optimiser found some tiny non-zero minimum."""
    obj, img = _synthesize(_project_standard(np.zeros(5)))
    _, _, dist, *_ = fit_standard(obj, img, IMAGE_SIZE)
    np.testing.assert_allclose(dist, np.zeros(5), atol=0.01)


# ── fit_fisheye: same check, the model this project's cameras likely need ─────

def test_fit_fisheye_recovers_known_camera_and_distortion():
    dist_true = np.array([-0.05, 0.01, -0.02, 0.005])
    obj, img = _synthesize(_project_fisheye(dist_true))
    assert len(img) >= 10

    rms, K, dist, *_ = fit_fisheye(obj, img, IMAGE_SIZE)

    assert rms < 0.5
    assert K[0, 0] == pytest.approx(TRUE_K[0, 0], abs=15)
    np.testing.assert_allclose(dist, dist_true, atol=0.02)


# ── choosing between the two models is measured, not assumed ──────────────────

def test_the_correct_model_fits_visibly_better_than_the_wrong_one():
    """The whole reason both models are fit and compared rather than one assumed:
    on genuinely fisheye-distorted views, the standard model must not look almost as
    good - if it did, there would be nothing to measure this decision by."""
    dist_true = np.array([-0.05, 0.01, -0.02, 0.005])
    obj, img = _synthesize(_project_fisheye(dist_true))

    rms_right, _, _, *_ = fit_fisheye(obj, img, IMAGE_SIZE)
    rms_wrong, _, _, *_ = fit_standard(obj, img, IMAGE_SIZE)

    assert rms_right < rms_wrong
    assert rms_wrong > rms_right * 5   # not a close call


# ── undistort_points: inverts what the same lens model distorts ───────────────

def test_undistort_standard_round_trips_through_a_known_distortion():
    dist_true = np.array([-0.28, 0.09, 0.001, -0.0005, -0.01])
    ideal = np.array([[500.0, 300.0], [1400.0, 800.0], [960.0, 540.0], [100.0, 1000.0]])

    # Forward-distort: treat `ideal` as an undistorted ray direction (K^-1 applied),
    # then image that ray through the real lens.
    normalized = (ideal - [TRUE_K[0, 2], TRUE_K[1, 2]]) / [TRUE_K[0, 0], TRUE_K[1, 1]]
    rays = np.column_stack([normalized, np.ones(len(ideal))]).reshape(-1, 1, 3)
    distorted, _ = cv2.projectPoints(rays, np.zeros(3), np.zeros(3), TRUE_K, dist_true)
    distorted = distorted.reshape(-1, 2)

    assert not np.allclose(distorted, ideal, atol=1.0), "fixture is not exercising real distortion"

    recovered = undistort_points(distorted, TRUE_K, dist_true, MODEL_STANDARD)
    np.testing.assert_allclose(recovered, ideal, atol=0.5)


def test_undistort_of_an_undistorted_camera_is_a_near_no_op():
    points = np.array([[300.0, 200.0], [1600.0, 900.0]])
    result = undistort_points(points, TRUE_K, np.zeros(5), MODEL_STANDARD)
    np.testing.assert_allclose(result, points, atol=0.01)


def test_undistort_rejects_an_unknown_model():
    with pytest.raises(ValueError, match="unknown lens model"):
        undistort_points(np.array([[1.0, 1.0]]), TRUE_K, np.zeros(5), "wide-angle-guess")


# ── coverage_span: did the board actually sweep the frame? ────────────────────

def test_coverage_span_of_corners_confined_to_the_centre_is_small():
    centre = np.array([[900.0, 500.0], [1000.0, 550.0], [950.0, 520.0]])
    span = coverage_span([centre], IMAGE_SIZE)
    assert span["x_span"] < 0.1
    assert span["wide_enough"] is False


def test_coverage_span_across_the_whole_frame_is_wide_enough():
    corners = [np.array([[50.0, 50.0]]), np.array([[1870.0, 1030.0]])]
    span = coverage_span(corners, IMAGE_SIZE)
    assert span["wide_enough"] is True


def test_coverage_span_of_nothing_does_not_crash():
    span = coverage_span([], IMAGE_SIZE)
    assert span["wide_enough"] is False


# ── reprojection_error_per_view: per-view, not only the pooled figure ─────────

def test_per_view_error_flags_the_one_bad_view():
    dist_true = np.array([-0.28, 0.09, 0.001, -0.0005, -0.01])
    obj, img = _synthesize(_project_standard(dist_true), n=12)
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj, img, IMAGE_SIZE, None, None)

    # Corrupt one view's detected corners, as a mis-detected checkerboard corner would.
    img_corrupted = [a.copy() for a in img]
    img_corrupted[3][0] += 40.0

    errors = reprojection_error_per_view(
        obj, img_corrupted, K, dist, MODEL_STANDARD, rvecs, tvecs)
    assert len(errors) == len(obj)
    worst = max(range(len(errors)), key=lambda i: errors[i])
    assert worst == 3
    assert errors[3] > 2 * sorted(errors)[len(errors) // 2]   # well above the median


# ── LensCalibration: persistence and rescaling ─────────────────────────────────

def _calibration(**overrides) -> LensCalibration:
    defaults = dict(
        model=MODEL_STANDARD,
        camera_matrix=TRUE_K.copy(),
        dist_coeffs=np.array([-0.28, 0.09, 0.001, -0.0005, -0.01]),
        image_size=IMAGE_SIZE,
        rms_error=0.31,
        views_used=18,
        coverage={"x_span": 0.8, "y_span": 0.7, "wide_enough": True},
        video="checkerboard.mp4",
    )
    defaults.update(overrides)
    return LensCalibration(**defaults)


def test_round_trip_through_json(tmp_path):
    original = _calibration(other_model_rms=0.9, notes="test fixture")
    path = original.save(tmp_path / "lens.json")
    loaded = LensCalibration.load(path)

    assert loaded.model == original.model
    np.testing.assert_allclose(loaded.camera_matrix, original.camera_matrix)
    np.testing.assert_allclose(loaded.dist_coeffs, original.dist_coeffs)
    assert loaded.image_size == original.image_size
    assert loaded.rms_error == pytest.approx(original.rms_error)
    assert loaded.views_used == original.views_used
    assert loaded.other_model_rms == pytest.approx(0.9)


def test_saved_file_is_plain_readable_json(tmp_path):
    import json
    path = _calibration().save(tmp_path / "lens.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["version"] == 1
    assert data["model"] == MODEL_STANDARD
    assert len(data["camera_matrix"]) == 3


def test_a_future_version_is_refused(tmp_path):
    import json
    path = tmp_path / "lens.json"
    payload = _calibration().to_dict()
    payload["version"] = 99
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="version 99"):
        LensCalibration.load(path)


def test_undistort_method_matches_the_free_function():
    calibration = _calibration()
    points = np.array([[500.0, 300.0], [1400.0, 800.0]])
    via_method = calibration.undistort(points)
    via_function = undistort_points(
        points, calibration.camera_matrix, calibration.dist_coeffs, calibration.model)
    np.testing.assert_allclose(via_method, via_function)


def test_scaled_to_scales_the_camera_matrix_linearly():
    calibration = _calibration(image_size=(1920, 1080))
    half = calibration.scaled_to((960, 540))
    assert half.image_size == (960, 540)
    assert half.camera_matrix[0, 0] == pytest.approx(TRUE_K[0, 0] / 2)
    assert half.camera_matrix[0, 2] == pytest.approx(TRUE_K[0, 2] / 2)
    # Distortion coefficients are dimensionless - unaffected by resolution.
    np.testing.assert_allclose(half.dist_coeffs, calibration.dist_coeffs)


def test_scaled_to_the_same_size_is_identity():
    calibration = _calibration(image_size=(1920, 1080))
    assert calibration.scaled_to((1920, 1080)) is calibration


def test_scaled_to_a_different_aspect_ratio_is_refused():
    calibration = _calibration(image_size=(1920, 1080))
    with pytest.raises(ValueError, match="aspect ratio"):
        calibration.scaled_to((1920, 1440))


# ── FitResult: named fields, not just position ─────────────────────────────────

def test_fit_result_exposes_poses_by_name():
    dist_true = np.array([-0.28, 0.09, 0.001, -0.0005, -0.01])
    obj, img = _synthesize(_project_standard(dist_true), n=12)

    fit = fit_standard(obj, img, IMAGE_SIZE)

    assert fit.rms == fit[0]
    np.testing.assert_array_equal(fit.camera_matrix, fit[1])
    assert len(fit.rvecs) == len(obj)
    assert len(fit.tvecs) == len(obj)


def test_fit_result_poses_feed_directly_into_per_view_error():
    """The point of exposing rvecs/tvecs on the result: no separate call needed to
    get them before asking which view fit worst."""
    dist_true = np.array([-0.28, 0.09, 0.001, -0.0005, -0.01])
    obj, img = _synthesize(_project_standard(dist_true), n=12)
    fit = fit_standard(obj, img, IMAGE_SIZE)

    errors = reprojection_error_per_view(
        obj, img, fit.camera_matrix, fit.dist_coeffs, MODEL_STANDARD,
        fit.rvecs, fit.tvecs)
    assert len(errors) == len(obj)
    assert max(errors) < 1.0   # noise-free synthetic views, should all fit tightly
