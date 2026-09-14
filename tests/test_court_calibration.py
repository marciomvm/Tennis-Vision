"""
Hand-placed court geometry: the conventions it has to share with the rest of the
pipeline, the mistakes it has to refuse, and which detections it keeps.

The tests that matter most here are the convention ones. A calibration is a drop-in
replacement for the keypoint model's output, so if its index order or its metric model
drifts from what MiniCourt draws and court_validity scores, nothing crashes - the court
is simply mapped to the wrong place, confidently, on every run.
"""
import json

import cv2
import numpy as np
import pytest

from utils.court_calibration import (
    CLICK_ORDER,
    COURT_MODEL_M,
    COURT_SEGMENTS,
    KEYPOINT_LABELS,
    N_KEYPOINTS,
    CourtCalibration,
    centre_point,
    court_roi_polygon,
    default_calibration_path,
    derive_keypoints,
    filter_detections,
    find_calibration_for,
    foot_point,
    polygon_contains,
    reprojection_residuals,
    validate_geometry,
)

FRAME = (1920, 1080)

# One synthetic camera: a real court seen in perspective, with no lens distortion, so
# every derived quantity has an exact expected value.
_CORNERS = {0: (520.0, 300.0), 1: (1400.0, 300.0), 3: (1750.0, 980.0), 2: (170.0, 980.0)}


def _truth() -> np.ndarray:
    H, _ = cv2.findHomography(
        COURT_MODEL_M[[0, 1, 3, 2]].astype(np.float32),
        np.array([_CORNERS[i] for i in (0, 1, 3, 2)], dtype=np.float32), 0)
    return cv2.perspectiveTransform(
        COURT_MODEL_M.astype(np.float32).reshape(-1, 1, 2), H).reshape(-1, 2)


def _calibration(**kwargs) -> CourtCalibration:
    points, _ = derive_keypoints(dict(_CORNERS))
    defaults = dict(keypoints=points, frame_size=FRAME, clicked=dict(_CORNERS),
                    video="synthetic.mp4", frame_index=7)
    defaults.update(kwargs)
    return CourtCalibration(**defaults)


# ── conventions shared with the rest of the pipeline ──────────────────────────

def test_segments_match_the_validity_gate():
    """
    The segments drawn here and the segments scored by the gate must be the same nine.

    They are separate tuples in separate modules describing one court. If they diverge,
    the overlay a human verifies is not the geometry the gate measures.
    """
    from utils.court_validity import COURT_LINES
    assert COURT_SEGMENTS == COURT_LINES


def test_metric_model_is_the_court_mini_court_draws():
    """
    The metric model and MiniCourt's diagram must be the same court.

    MiniCourt builds its fourteen destination points from `constants` directly. This
    model builds a metric court from the same constants in what is meant to be the same
    index order, and a homography between two descriptions of one court is exact. A
    residual here means the orders disagree, which silently maps play to the wrong part
    of the diagram.
    """
    from mini_visual_court import MiniCourt

    mini = MiniCourt(np.zeros((720, 1280, 3), dtype=np.uint8))
    drawn = np.array(mini.get_court_drawing_keypoints(), dtype=np.float32).reshape(-1, 2)

    H, _ = cv2.findHomography(COURT_MODEL_M.astype(np.float32), drawn, 0)
    mapped = cv2.perspectiveTransform(
        COURT_MODEL_M.astype(np.float32).reshape(-1, 1, 2), H).reshape(-1, 2)
    assert np.abs(mapped - drawn).max() < 1.0


def test_click_order_covers_every_point_once():
    assert sorted(CLICK_ORDER) == list(range(N_KEYPOINTS))
    assert set(KEYPOINT_LABELS) == set(range(N_KEYPOINTS))


def test_outer_corners_come_first():
    """Four corners already determine a homography, so the preview is useful at once."""
    assert set(CLICK_ORDER[:4]) == {0, 1, 2, 3}


# ── deriving fourteen points from fewer ───────────────────────────────────────

def test_four_corners_reconstruct_the_whole_court():
    points, residuals = derive_keypoints(dict(_CORNERS))
    assert np.abs(points - _truth()).max() < 0.01
    assert max(residuals.values()) < 0.01


def test_placed_points_are_kept_exactly_not_smoothed():
    """
    A clicked point is where the corner really appears; a fitted one is where a pinhole
    camera would have put it. On a distorted lens only the first is true, so the fit must
    never overwrite a placement.
    """
    clicks = dict(_CORNERS)
    clicks[12] = (900.0, 555.0)          # deliberately off the perspective-correct spot
    points, residuals = derive_keypoints(clicks)
    assert tuple(points[12]) == (900.0, 555.0)
    assert residuals[12] > 1.0           # and the disagreement is reported, not hidden


def test_three_points_are_refused():
    with pytest.raises(ValueError, match="at least 4"):
        derive_keypoints({0: (1.0, 1.0), 1: (2.0, 1.0), 2: (1.0, 2.0)})


def test_collinear_points_are_refused():
    with pytest.raises(ValueError):
        derive_keypoints({0: (0.0, 0.0), 1: (10.0, 0.0),
                          2: (20.0, 0.0), 3: (30.0, 0.0)})


def test_residuals_are_zero_on_an_undistorted_camera():
    points, _ = derive_keypoints(dict(_CORNERS))
    assert reprojection_residuals(points).max() < 0.01


# ── refusing a court that is not a court ──────────────────────────────────────

def test_a_good_court_has_no_problems():
    points, _ = derive_keypoints(dict(_CORNERS))
    assert validate_geometry(points) == []


def test_halves_placed_the_wrong_way_round_are_caught():
    """The realistic mistake: starting at the near baseline instead of the far one."""
    points, _ = derive_keypoints(dict(_CORNERS))
    flipped = points.copy()
    flipped[[0, 1, 2, 3]] = points[[2, 3, 0, 1]]
    flipped[[4, 5]] = points[[5, 4]]
    flipped[[6, 7]] = points[[7, 6]]
    problems = validate_geometry(flipped)
    assert any("wrong way round" in p for p in problems)


def test_a_mirrored_court_is_caught():
    points, _ = derive_keypoints(dict(_CORNERS))
    mirrored = points.copy()
    mirrored[[0, 1]] = points[[1, 0]]
    mirrored[[2, 3]] = points[[3, 2]]
    mirrored[[4, 6]] = points[[6, 4]]
    mirrored[[5, 7]] = points[[7, 5]]
    assert validate_geometry(mirrored) != []


def test_a_centre_t_off_its_service_line_is_caught():
    points, _ = derive_keypoints(dict(_CORNERS))
    broken = points.copy()
    broken[12] = (points[8][0] - 200.0, points[12][1])
    assert any("centre T" in p for p in validate_geometry(broken))


def test_degenerate_corners_are_caught_without_crashing():
    points = np.zeros((N_KEYPOINTS, 2), dtype=np.float64)
    assert validate_geometry(points) != []


# ── the region of the image this court occupies ───────────────────────────────

def test_the_region_contains_the_court_it_was_built_from():
    points, _ = derive_keypoints(dict(_CORNERS))
    roi = court_roi_polygon(points, frame_size=FRAME)
    for corner in points:
        assert polygon_contains(roi, corner)


def test_the_region_excludes_the_next_court_along():
    """
    A court beyond the far baseline is what the margin is for. Four metres of run-back is
    inside; the next court, which starts well past that, is not.
    """
    points, _ = derive_keypoints(dict(_CORNERS))
    roi = court_roi_polygon(points, frame_size=FRAME)
    far_baseline_y = float(np.mean([points[0][1], points[1][1]]))
    centre_x = float(np.mean([points[0][0], points[1][0]]))
    assert polygon_contains(roi, (centre_x, far_baseline_y + 5))
    assert not polygon_contains(roi, (centre_x, far_baseline_y - 220))


def test_the_margin_is_metric_not_pixels():
    """
    Perspective makes the same metric margin a wide band near the camera and a thin one
    at the far end. A region that widened by a constant pixel count would reach the next
    court; this one must not.
    """
    points, _ = derive_keypoints(dict(_CORNERS))
    roi = court_roi_polygon(points, frame_size=FRAME)
    far_y = float(np.mean([points[0][1], points[1][1]]))
    near_y = float(np.mean([points[2][1], points[3][1]]))
    centre_x = float(np.mean([points[0][0], points[1][0]]))

    def reach(direction, start):
        distance = 0
        while distance < 2000 and polygon_contains(roi, (centre_x, start + direction * distance)):
            distance += 5
        return distance

    assert reach(1, near_y) > reach(-1, far_y) * 2


def test_a_bigger_margin_makes_a_bigger_region():
    points, _ = derive_keypoints(dict(_CORNERS))
    tight = court_roi_polygon(points, frame_size=FRAME, beside_m=0.5,
                              behind_far_m=0.5, behind_near_m=0.5)
    loose = court_roi_polygon(points, frame_size=FRAME, beside_m=4.0,
                              behind_far_m=4.0, behind_near_m=8.0)
    assert cv2.contourArea(loose) > cv2.contourArea(tight)


def test_an_absent_region_contains_everything():
    """`filter_detections` is given None when no region could be built, and must then
    keep every detection rather than silently discarding the clip."""
    assert polygon_contains(None, (0, 0))


# ── which detections belong to this court ─────────────────────────────────────

def _box(cx, cy, w=60, h=160):
    return [cx - w / 2, cy - h, cx + w / 2, cy]


def test_a_person_is_judged_by_their_feet():
    """
    A torso is a metre and a half above the court plane and projects a long way up the
    image at a low camera angle, so the far player's centre routinely lands outside a
    region their feet are well inside.
    """
    box = _box(900, 400)
    assert foot_point(box)[1] > centre_point(box)[1]
    assert foot_point(box)[0] == centre_point(box)[0]


def test_people_off_this_court_are_dropped_and_counted():
    points, _ = derive_keypoints(dict(_CORNERS))
    roi = court_roi_polygon(points, frame_size=FRAME)
    on_court = _box(960, 900)
    next_court = _box(960, 120)

    detections = [{1: on_court, 2: next_court}, {1: on_court, 2: next_court}]
    kept, report = filter_detections(detections, roi=roi)

    assert [set(frame) for frame in kept] == [{1}, {1}]
    assert report["boxes_removed"] == 2
    assert report["tracks_removed_entirely"] == 1
    assert report["tracks_kept"] == 1


def test_an_exclusion_zone_removes_what_the_region_kept():
    points, _ = derive_keypoints(dict(_CORNERS))
    roi = court_roi_polygon(points, frame_size=FRAME)
    umpire = _box(960, 900)
    assert polygon_contains(roi, foot_point(umpire))

    zone = [(900, 800), (1020, 800), (1020, 950), (900, 950)]
    kept, report = filter_detections([{5: umpire}], roi=roi, exclusions=[zone])
    assert kept == [{}]
    assert report["boxes_removed"] == 1


def test_a_track_seen_both_on_and_off_court_is_not_reported_as_removed():
    """Partial removal is a different event from a track that never belonged here."""
    points, _ = derive_keypoints(dict(_CORNERS))
    roi = court_roi_polygon(points, frame_size=FRAME)
    kept, report = filter_detections(
        [{1: _box(960, 900)}, {1: _box(960, 120)}], roi=roi)
    assert report["boxes_removed"] == 1
    assert report["tracks_removed_entirely"] == 0
    assert report["tracks_kept"] == 1


def test_filtering_nothing_is_a_no_op():
    detections = [{1: _box(960, 900)}, {}]
    kept, report = filter_detections(detections, roi=None, exclusions=[])
    assert kept == detections
    assert report["boxes_removed"] == 0


# ── persistence ───────────────────────────────────────────────────────────────

def test_round_trip_preserves_everything_the_pipeline_reads(tmp_path):
    original = _calibration(exclusions=[[(10.0, 20.0), (30.0, 20.0), (30.0, 40.0)]],
                            line_support=0.081, notes="hello")
    path = original.save(tmp_path / "clip.json")
    loaded = CourtCalibration.load(path)

    assert np.abs(loaded.keypoints - original.keypoints).max() < 0.01
    assert loaded.frame_size == FRAME
    assert loaded.clicked.keys() == original.clicked.keys()
    assert loaded.exclusions == [[(10.0, 20.0), (30.0, 20.0), (30.0, 40.0)]]
    assert loaded.video == "synthetic.mp4"
    assert loaded.frame_index == 7
    assert loaded.line_support == 0.081
    assert loaded.notes == "hello"
    assert loaded.source_path == str(path)


def test_saved_file_is_readable_json_a_human_can_edit(tmp_path):
    path = _calibration().save(tmp_path / "clip.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["version"] == 1
    assert len(data["keypoints"]) == N_KEYPOINTS
    assert all(len(p) == 2 for p in data["keypoints"])
    assert data["created_at"]


def test_a_future_version_is_refused_rather_than_guessed_at(tmp_path):
    path = tmp_path / "clip.json"
    payload = _calibration().to_dict()
    payload["version"] = 99
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="version 99"):
        CourtCalibration.load(path)


def test_flat_is_the_28_float_layout_every_other_module_speaks():
    flat = _calibration().flat()
    assert flat.shape == (N_KEYPOINTS * 2,)
    assert flat[0] == pytest.approx(_CORNERS[0][0])
    assert flat[1] == pytest.approx(_CORNERS[0][1])


def test_rescaling_to_the_same_camera_at_another_resolution():
    half = _calibration().scaled_to((960, 540))
    assert half.frame_size == (960, 540)
    assert half.keypoints[0][0] == pytest.approx(_CORNERS[0][0] / 2)
    assert half.exclusions == []


def test_rescaling_to_a_different_aspect_ratio_is_refused():
    """A different aspect ratio is a different crop or lens, so the court really has
    moved and rescaling would place it confidently in the wrong spot."""
    with pytest.raises(ValueError, match="aspect ratio"):
        _calibration().scaled_to((1920, 1440))


def test_rescaling_to_the_same_size_is_identity():
    original = _calibration()
    assert original.scaled_to(FRAME) is original


# ── discovery ─────────────────────────────────────────────────────────────────

def test_discovery_is_by_video_file_name(tmp_path):
    assert default_calibration_path("videos/clipe1.mp4", str(tmp_path)).name == "clipe1.json"
    assert find_calibration_for("videos/clipe1.mp4", str(tmp_path)) is None
    _calibration().save(tmp_path / "clipe1.json")
    assert find_calibration_for("videos/clipe1.mp4", str(tmp_path)) is not None


# ── the placement tool's state machine ────────────────────────────────────────

def test_placing_points_walks_the_click_order():
    from tools.calibrate_court import CalibrationState

    state = CalibrationState()
    assert state.target == CLICK_ORDER[0]
    state.place((100.0, 100.0))
    assert state.target == CLICK_ORDER[1]
    state.undo()
    assert state.target == CLICK_ORDER[0]
    assert state.clicks == {}


def test_skipping_a_point_leaves_it_to_the_fit():
    from tools.calibrate_court import CalibrationState

    state = CalibrationState()
    skipped = state.target
    state.skip()
    assert skipped not in state.clicks
    assert state.target != skipped


def test_a_zone_needs_three_corners():
    from tools.calibrate_court import CalibrationState

    state = CalibrationState()
    state.begin_zone()
    state.add_zone_vertex((1.0, 1.0))
    state.add_zone_vertex((2.0, 2.0))
    assert state.close_zone() is False
    assert state.exclusions == []

    state.begin_zone()
    for vertex in ((1.0, 1.0), (9.0, 1.0), (9.0, 9.0)):
        state.add_zone_vertex(vertex)
    assert state.close_zone() is True
    assert len(state.exclusions) == 1


# ── correcting a point after it is placed ─────────────────────────────────────

def _placed_state():
    from tools.calibrate_court import CalibrationState

    state = CalibrationState()
    for index in (0, 1, 3, 2):
        state.place(_CORNERS[index])
    return state


def test_a_point_can_be_grabbed_by_clicking_near_it():
    state = _placed_state()
    near = (_CORNERS[1][0] + 4, _CORNERS[1][1] - 3)
    assert state.point_near(near, radius=15) == 1
    assert state.point_near(near, radius=2) is None


def test_grabbing_picks_the_closest_of_two_points():
    """Two points within the grab radius is the normal case near the service T."""
    state = _placed_state()
    positions = {0: (0.0, 0.0), 1: (100.0, 0.0)}
    assert state.point_near((60.0, 0.0), radius=1000, positions=positions) == 1
    assert state.point_near((40.0, 0.0), radius=1000, positions=positions) == 0


def test_a_fitted_point_can_be_grabbed_too():
    """
    The fit's guess being nearly right is the common case, and nudging it is the whole
    reason drag exists. Only four points are placed here, so point 9 exists only because
    the homography put it there.
    """
    state = _placed_state()
    fitted = state.keypoints()[9]
    assert 9 not in state.clicks
    assert state.point_near(tuple(fitted), radius=5) == 9


def test_grabbing_a_point_does_not_move_it():
    """A click that merely lands on a point selects it. Snapping it to wherever the
    cursor sat would nudge it by up to the grab radius for free."""
    state = _placed_state()
    before = state.clicks[1]
    undo_depth = len(state.history)
    state.begin_drag(1)
    assert state.clicks[1] == before
    assert len(state.history) == undo_depth, "a grab with no movement is not an edit"
    state.end_drag()
    assert state.clicks[1] == before


def test_dragging_moves_the_point_and_undo_puts_it_back():
    state = _placed_state()
    before = state.clicks[1]
    state.begin_drag(1)
    state.drag_to((900.0, 400.0))
    state.drag_to((901.0, 401.0))
    state.end_drag()
    assert state.clicks[1] == (901.0, 401.0)

    state.undo()
    assert state.clicks[1] == before, "undo must restore the old position, not delete it"


def test_one_drag_is_one_undo():
    """A drag fires a mouse-move event per pixel. Recording each would make undo useless."""
    state = _placed_state()
    undo_depth = len(state.history)
    state.begin_drag(3)
    for step in range(30):
        state.drag_to((1000.0 + step, 500.0 + step))
    state.end_drag()
    assert len(state.history) == undo_depth + 1


def test_dragging_a_fitted_point_makes_it_a_measured_one():
    state = _placed_state()
    assert 12 not in state.clicks
    state.begin_drag(12)
    state.drag_to((1000.0, 500.0))
    state.end_drag()
    assert state.clicks[12] == (1000.0, 500.0)

    state.undo()
    assert 12 not in state.clicks, "undoing a point that had no previous value removes it"


def test_all_placed_is_only_true_with_fourteen():
    from tools.calibrate_court import CalibrationState

    state = _placed_state()
    assert not state.all_placed
    full = CalibrationState({i: (float(i) * 10, float(i) * 20) for i in range(14)})
    assert full.all_placed


def test_reopening_a_calibration_has_nothing_to_undo():
    from tools.calibrate_court import CalibrationState

    state = CalibrationState(dict(_CORNERS))
    state.undo()
    assert set(state.clicks) == set(_CORNERS), (
        "there is no previous session to undo into"
    )


# ── reaching a point that is outside the video ────────────────────────────────

def test_the_view_always_contains_the_whole_video():
    from tools.calibrate_court import view_rect

    x0, y0, x1, y1 = view_rect(FRAME, None)
    assert x0 <= 0 and y0 <= 0
    assert x1 >= FRAME[0] and y1 >= FRAME[1]


def test_the_view_grows_around_a_point_outside_the_frame():
    """
    A wide camera close to the baseline puts a near doubles corner past the edge of its
    own picture. A canvas that stopped at the video left that point unreachable: no pixel
    to click, and no drag that could get to it.
    """
    from tools.calibrate_court import view_rect

    points = np.array([[100.0, 100.0]] * N_KEYPOINTS)
    points[3] = (FRAME[0] + 70.0, 900.0)
    x0, y0, x1, y1 = view_rect(FRAME, points, pad=60)
    assert x1 >= FRAME[0] + 70 + 60
    assert x0 <= 0, "growing to the right must not crop the left"


def test_the_view_grows_to_the_left_and_top_too():
    from tools.calibrate_court import view_rect

    points = np.array([[100.0, 100.0]] * N_KEYPOINTS)
    points[0] = (-120.0, -80.0)
    x0, y0, x1, y1 = view_rect(FRAME, points, pad=60)
    assert x0 <= -180 and y0 <= -140
    assert x1 >= FRAME[0] and y1 >= FRAME[1]


def test_a_wild_point_does_not_shrink_the_court_to_nothing():
    """A nearly-degenerate fit can throw a point a long way off. Zooming out to include
    it would make every other point unclickable."""
    from tools.calibrate_court import view_rect

    points = np.array([[100.0, 100.0]] * N_KEYPOINTS)
    points[5] = (500_000.0, 500_000.0)
    _, _, x1, y1 = view_rect(FRAME, points)
    assert x1 < FRAME[0] * 4
    assert y1 < FRAME[1] * 4


def test_a_non_finite_point_does_not_break_the_view():
    from tools.calibrate_court import view_rect

    points = np.array([[100.0, 100.0]] * N_KEYPOINTS)
    points[2] = (np.nan, np.inf)
    rect = view_rect(FRAME, points)
    assert all(np.isfinite(rect))


def test_display_and_full_coordinates_round_trip():
    from tools.calibrate_court import to_display, to_full, view_rect, view_scale

    points = np.array([[100.0, 100.0]] * N_KEYPOINTS)
    points[3] = (FRAME[0] + 70.0, 900.0)
    rect = view_rect(FRAME, points)
    scale = view_scale(rect, (1400, 800))

    for original in ((0.0, 0.0), (960.0, 540.0), (FRAME[0] + 70.0, 900.0), (-50.0, -20.0)):
        there = to_display(original, rect, scale)
        back = to_full(there, rect, scale)
        assert abs(back[0] - original[0]) < 1.5 / scale
        assert abs(back[1] - original[1]) < 1.5 / scale


def test_the_scale_never_magnifies():
    """Upscaling a frame to fill a large screen would only blur it."""
    from tools.calibrate_court import view_scale

    assert view_scale((0, 0, 640, 360), (1920, 1080)) == 1.0
    assert view_scale((0, 0, 1920, 1080), (960, 540)) == pytest.approx(0.5)


def test_rect_contains_is_what_decides_the_view_must_grow():
    from tools.calibrate_court import rect_contains

    rect = (0.0, 0.0, 100.0, 100.0)
    assert rect_contains(rect, np.array([[10.0, 10.0], [90.0, 90.0]]))
    assert not rect_contains(rect, np.array([[10.0, 10.0], [110.0, 50.0]]))
    assert rect_contains(rect, None)
    assert not rect_contains(rect, np.array([[10.0, 10.0], [96.0, 50.0]]), margin=8)


# ── the keyboard reaches what the mouse cannot ────────────────────────────────

def test_arrow_nudge_moves_the_selected_point():
    state = _placed_state()
    state._set_target(1)
    before = state.clicks[1]
    assert state.nudge(3, -2) is True
    assert state.clicks[1] == (before[0] + 3, before[1] - 2)


def test_nudging_a_fitted_point_makes_it_a_measured_one():
    state = _placed_state()
    state._set_target(9)
    fitted = tuple(state.keypoints()[9])
    assert 9 not in state.clicks
    assert state.nudge(1, 0) is True
    assert state.clicks[9] == pytest.approx((fitted[0] + 1, fitted[1]))


def test_nudging_with_too_few_points_does_nothing():
    from tools.calibrate_court import CalibrationState

    state = CalibrationState({0: (10.0, 10.0)})
    assert state.nudge(1, 1) is False


def test_a_run_of_nudges_is_one_undo():
    """Holding an arrow key would otherwise fill the history with one-pixel steps."""
    state = _placed_state()
    state._set_target(1)
    before = state.clicks[1]
    depth = len(state.history)
    for _ in range(25):
        state.nudge(1, 0)
    assert len(state.history) == depth + 1
    state.undo()
    assert state.clicks[1] == before


def test_selecting_another_point_starts_a_new_undo_entry():
    state = _placed_state()
    state._set_target(1)
    state.nudge(1, 0)
    depth = len(state.history)
    state._set_target(2)
    state.nudge(1, 0)
    assert len(state.history) == depth + 1


def test_window_size_is_parsed_and_sanity_checked():
    import argparse

    from tools.calibrate_court import _parse_size

    assert _parse_size("1280x720") == (1280, 720)
    assert _parse_size("1280X720") == (1280, 720)
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_size("enormous")
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_size("100x100")


def test_the_window_is_fitted_to_something_plausible():
    from tools.calibrate_court import _screen_size

    width, height = _screen_size()
    assert width >= 320 and height >= 240
