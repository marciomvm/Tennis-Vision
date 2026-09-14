"""
utils/court_calibration.py
──────────────────────────
Hand-placed court geometry, for footage the keypoint model cannot read.

Why this module exists
----------------------
`CourtLineDetector` is a ResNet-50 regression head trained on broadcast tennis: a high,
centred, long-lens camera. Point it at a phone or an action camera clamped to a fence
post behind the baseline and it still returns fourteen tidy points - just not on the
court. `utils.court_validity` catches that and refuses the clip, which is the correct
behaviour and still leaves the user with nothing to run.

On a FIXED camera the court is not a per-frame inference problem at all. It does not
move, so fourteen points placed once describe every frame of every clip shot from that
position. This module stores those points, checks that they describe a court rather than
a mis-click, and reports the region of the image that court occupies.

That region is the second thing this module is for. Club footage routinely shows the
next court along, and a rally happening there is real tennis played by real people that
the detector is entirely right to find. Nothing in the image says which court is the one
being analysed - only the calibration does - so the filtering belongs here.

What this does NOT fix
----------------------
A homography assumes straight lines. Action cameras have barrel distortion, so on that
footage a straight segment between two correctly-placed corners still cuts across the
grass in the middle. The fourteen stored points are unaffected, since each is placed
where it actually appears, but any position mapped THROUGH the homography carries that
error. `reprojection_residuals` measures it, so the cost is visible rather than assumed.

Keypoint convention
-------------------
Identical to the one `mini_visual_court.MiniCourt` draws and `utils.court_validity`
scores, because a calibration is a drop-in replacement for the model's output and a
second convention would be a silent coordinate swap:

     0 -- 4 ----------- 6 -- 1      FAR baseline   (0,1 doubles corners)
     |    |             |    |      4,6  singles sideline ends
     |    8 --- 12 ---- 9    |      8,9  far service line, 12 far centre T
     |    |             |    |
     |    |             |    |      (net)
     |    |             |    |
     |   10 --- 13 --- 11    |      10,11 near service line, 13 near centre T
     |    |             |    |
     2 -- 5 ----------- 7 -- 3      NEAR baseline  (2,3 doubles corners)

"Far" is the half further from the camera, "near" the half closer to it, and left/right
are as seen on screen.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import constants

# ── The court, in metres ───────────────────────────────────────────────────────
# A real tennis court, in the same index order as the diagram above. This is the fixed
# half of the problem: the only unknown a calibration supplies is where this rectangle
# landed in one particular image.

COURT_WIDTH_M = float(constants.DOUBLE_LINE_WIDTH)               # 10.97
COURT_LENGTH_M = float(constants.HALF_COURT_LINE_HEIGHT) * 2.0   # 23.77
_ALLEY = float(constants.DOUBLE_ALLY_DIFFERENCE)                 # 1.37
_SVC_DEPTH = float(constants.NO_MANS_LAND_HEIGHT)                # 5.48, baseline to service line
_SVC_WIDTH = float(constants.SINGLE_LINE_WIDTH)                  # 8.23

COURT_MODEL_M: np.ndarray = np.array([
    (0.0,                     0.0),                            # 0  doubles corner, far left
    (COURT_WIDTH_M,           0.0),                            # 1  doubles corner, far right
    (0.0,                     COURT_LENGTH_M),                 # 2  doubles corner, near left
    (COURT_WIDTH_M,           COURT_LENGTH_M),                 # 3  doubles corner, near right
    (_ALLEY,                  0.0),                            # 4  singles sideline, far left
    (_ALLEY,                  COURT_LENGTH_M),                 # 5  singles sideline, near left
    (COURT_WIDTH_M - _ALLEY,  0.0),                            # 6  singles sideline, far right
    (COURT_WIDTH_M - _ALLEY,  COURT_LENGTH_M),                 # 7  singles sideline, near right
    (_ALLEY,                  _SVC_DEPTH),                     # 8  far service line, left
    (_ALLEY + _SVC_WIDTH,     _SVC_DEPTH),                     # 9  far service line, right
    (_ALLEY,                  COURT_LENGTH_M - _SVC_DEPTH),    # 10 near service line, left
    (_ALLEY + _SVC_WIDTH,     COURT_LENGTH_M - _SVC_DEPTH),    # 11 near service line, right
    (_ALLEY + _SVC_WIDTH / 2, _SVC_DEPTH),                     # 12 centre T, far
    (_ALLEY + _SVC_WIDTH / 2, COURT_LENGTH_M - _SVC_DEPTH),    # 13 centre T, near
], dtype=np.float64)

N_KEYPOINTS = len(COURT_MODEL_M)

# The order a human is asked to place them in: the four outer corners first, clockwise
# from the far left, because those four alone already determine a homography and the
# wireframe preview becomes useful immediately. Everything after them refines the fit.
CLICK_ORDER: tuple[int, ...] = (0, 1, 3, 2, 4, 6, 5, 7, 8, 9, 10, 11, 12, 13)

KEYPOINT_LABELS: dict[int, str] = {
    0:  "FAR baseline  x  LEFT doubles sideline   (far outer corner, left)",
    1:  "FAR baseline  x  RIGHT doubles sideline  (far outer corner, right)",
    2:  "NEAR baseline x  LEFT doubles sideline   (near outer corner, left)",
    3:  "NEAR baseline x  RIGHT doubles sideline  (near outer corner, right)",
    4:  "FAR baseline  x  LEFT singles sideline",
    5:  "NEAR baseline x  LEFT singles sideline",
    6:  "FAR baseline  x  RIGHT singles sideline",
    7:  "NEAR baseline x  RIGHT singles sideline",
    8:  "FAR service line  x  LEFT singles sideline",
    9:  "FAR service line  x  RIGHT singles sideline",
    10: "NEAR service line x  LEFT singles sideline",
    11: "NEAR service line x  RIGHT singles sideline",
    12: "FAR centre T   (far service line x centre service line)",
    13: "NEAR centre T  (near service line x centre service line)",
}

# Segments to draw, the same nine the validity gate scores.
COURT_SEGMENTS: tuple[tuple[int, int], ...] = (
    (0, 2), (1, 3), (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13),
)

MIN_CLICKS = 4
CALIBRATION_VERSION = 1
DEFAULT_CALIBRATION_DIR = "calibration"

# How far outside the painted court a person can stand and still be a player. Behind the
# near baseline is deliberately generous: a low camera behind the court sees the near
# player retreat a long way, and the same metric distance covers far more image there.
# These are margins on an inclusion test, not thresholds anything is measured against.
MARGIN_BESIDE_SIDELINE_M = 3.0
MARGIN_BEHIND_FAR_BASELINE_M = 4.0
MARGIN_BEHIND_NEAR_BASELINE_M = 8.0


# ── Geometry ───────────────────────────────────────────────────────────────────

def _as_pairs(keypoints) -> np.ndarray:
    kp = np.asarray(keypoints, dtype=np.float64).reshape(-1, 2)
    if kp.shape[0] != N_KEYPOINTS:
        raise ValueError(f"expected {N_KEYPOINTS} keypoints, got {kp.shape[0]}")
    return kp


def fit_court_homography(clicks: dict[int, tuple[float, float]]) -> np.ndarray:
    """
    Fit metres to image pixels from the points a human placed.

    Least squares over every supplied point, with no RANSAC. RANSAC exists to discard
    outliers produced by a detector, and these points were not produced by a detector:
    dropping one would silently ignore a human's measurement. A genuine mis-click shows
    up in `reprojection_residuals` instead, where it can be seen and corrected.
    """
    idx = sorted(clicks)
    if len(idx) < MIN_CLICKS:
        raise ValueError(f"need at least {MIN_CLICKS} points, got {len(idx)}")
    src = COURT_MODEL_M[idx].astype(np.float32).reshape(-1, 1, 2)
    dst = np.array([clicks[i] for i in idx], dtype=np.float32).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(src, dst, 0)
    if H is None:
        raise ValueError(
            "could not fit a court to those points - three or more of them are probably "
            "collinear, or two are the same point"
        )
    return H.astype(np.float64)


def derive_keypoints(
    clicks: dict[int, tuple[float, float]]
) -> tuple[np.ndarray, dict[int, float]]:
    """
    Complete the fourteen points from however many were placed by hand.

    Placed points are kept exactly as placed and the rest are filled in from the fit.
    That ordering matters on a distorted lens: a clicked point is where the corner really
    appears, while a fitted one is where a pinhole camera would have put it. Overwriting
    the first with the second would discard the only distortion-free information here.

    Returns (14x2 points, {index: fit residual in px, for each clicked point}).
    """
    H = fit_court_homography(clicks)
    fitted = cv2.perspectiveTransform(
        COURT_MODEL_M.astype(np.float32).reshape(-1, 1, 2), H.astype(np.float32)
    ).reshape(-1, 2).astype(np.float64)

    residuals = {
        i: float(np.hypot(fitted[i][0] - clicks[i][0], fitted[i][1] - clicks[i][1]))
        for i in clicks
    }
    points = fitted.copy()
    for i, (x, y) in clicks.items():
        points[i] = (float(x), float(y))
    return points, residuals


def reprojection_residuals(keypoints) -> np.ndarray:
    """
    Per-point distance, in px, between each keypoint and where a single homography would
    put it.

    This is the price of the pinhole assumption on this footage. On a rectilinear lens it
    is a couple of pixels and the homography is exact enough to ignore. On a wide action
    camera it grows toward the frame edges, and it bounds how accurately any position
    mapped through that homography - every mini-court dot, every distance, every speed -
    can possibly be placed.
    """
    kp = _as_pairs(keypoints)
    H = fit_court_homography({i: tuple(kp[i]) for i in range(N_KEYPOINTS)})
    fitted = cv2.perspectiveTransform(
        COURT_MODEL_M.astype(np.float32).reshape(-1, 1, 2), H.astype(np.float32)
    ).reshape(-1, 2)
    return np.hypot(fitted[:, 0] - kp[:, 0], fitted[:, 1] - kp[:, 1])


def validate_geometry(keypoints) -> list[str]:
    """
    Check that these fourteen points describe a court, and say what is wrong when they
    do not.

    This is not the line-support gate. That one asks whether the points sit on paint,
    which a human placing them by eye has already answered better than any brightness
    heuristic can. This asks the question a human CAN get wrong while looking straight at
    the court: whether the points went in in the right order. Clicking the near baseline
    first, or mirroring left and right, produces a court that is entirely self-consistent,
    passes every brightness test, and silently reverses the direction of play.

    Returns a list of problems, empty when the geometry is sound.
    """
    kp = _as_pairs(keypoints)
    problems: list[str] = []

    outer = kp[[0, 1, 3, 2]].astype(np.float32)  # clockwise from the far left corner
    if cv2.contourArea(outer) < 1.0:
        problems.append("the four outer corners enclose no area - they are collinear, "
                        "or two of them are the same point")
        return problems
    if not cv2.isContourConvex(outer):
        problems.append("the four outer corners do not form a convex quadrilateral, so "
                        "at least two of them were placed in the wrong order")

    far_y = float(np.mean([kp[0][1], kp[1][1], kp[4][1], kp[6][1]]))
    near_y = float(np.mean([kp[2][1], kp[3][1], kp[5][1], kp[7][1]]))
    if near_y <= far_y:
        problems.append("the NEAR baseline is not below the FAR baseline in the image, "
                        "so the two halves of the court were placed the wrong way round")

    svc_far_y = float(np.mean([kp[8][1], kp[9][1], kp[12][1]]))
    svc_near_y = float(np.mean([kp[10][1], kp[11][1], kp[13][1]]))
    if not (far_y < svc_far_y < svc_near_y < near_y):
        problems.append("the service lines are not between the two baselines in the "
                        "order far baseline, far service line, near service line, near "
                        "baseline")

    left_x = float(np.mean([kp[0][0], kp[2][0], kp[4][0], kp[5][0]]))
    right_x = float(np.mean([kp[1][0], kp[3][0], kp[6][0], kp[7][0]]))
    if left_x >= right_x:
        problems.append("the LEFT sideline is not left of the RIGHT sideline, so the "
                        "court was placed mirrored")

    for centre, a, b, name in ((12, 8, 9, "far"), (13, 10, 11, "near")):
        lo, hi = sorted((kp[a][0], kp[b][0]))
        if not (lo < kp[centre][0] < hi):
            problems.append(f"the {name} centre T is not between the two ends of the "
                            f"{name} service line")

    return problems


def _project(H: np.ndarray, xy) -> tuple[float, float, float] | None:
    v = H @ np.array([float(xy[0]), float(xy[1]), 1.0])
    if not np.all(np.isfinite(v)) or abs(v[2]) < 1e-9:
        return None
    return (v[0] / v[2], v[1] / v[2], v[2])


def _metric_to_image(keypoints) -> np.ndarray:
    kp = _as_pairs(keypoints)
    return fit_court_homography({i: tuple(kp[i]) for i in range(N_KEYPOINTS)})


def court_roi_polygon(
    keypoints,
    frame_size: tuple[int, int] | None = None,
    beside_m: float = MARGIN_BESIDE_SIDELINE_M,
    behind_far_m: float = MARGIN_BEHIND_FAR_BASELINE_M,
    behind_near_m: float = MARGIN_BEHIND_NEAR_BASELINE_M,
    samples_per_edge: int = 16,
) -> np.ndarray:
    """
    The image region this court occupies, widened by a playing margin in METRES.

    The margin is applied in court space rather than in pixels because perspective makes
    those two completely different shapes. Three metres beside the near sideline is a
    wide band at the bottom of the frame and a handful of pixels at the top, so a fixed
    pixel margin is far too tight near the camera and far too loose at the far end -
    which is exactly where the next court along sits.

    Each edge is sampled rather than only its corners, so the polygon follows the
    perspective. A sample that projects behind the camera's horizon - which happens when
    the far margin is pushed past the vanishing line on a low camera - is walked back
    toward the court until it lands somewhere real, rather than being dropped or allowed
    to wrap around to a nonsense coordinate.

    Returns an Nx1x2 int32 contour, ready for `cv2.pointPolygonTest` and `cv2.polylines`.
    """
    H = _metric_to_image(keypoints)
    W, L = COURT_WIDTH_M, COURT_LENGTH_M

    reference = _project(H, (W / 2, L / 2))
    if reference is None:
        raise ValueError("degenerate court geometry: its own centre does not project")
    ref_sign = 1.0 if reference[2] > 0 else -1.0

    if frame_size is not None:
        limit_x, limit_y = frame_size[0] * 10.0, frame_size[1] * 10.0
    else:
        limit_x = limit_y = 1e5

    def place(inner, outer):
        for scale in (1.0, 0.8, 0.6, 0.4, 0.2, 0.0):
            candidate = (inner[0] + (outer[0] - inner[0]) * scale,
                         inner[1] + (outer[1] - inner[1]) * scale)
            got = _project(H, candidate)
            if got is None:
                continue
            x, y, w = got
            if w * ref_sign <= 0:               # behind the horizon
                continue
            if abs(x) > limit_x or abs(y) > limit_y:
                continue
            return (x, y)
        return None

    inner_ring = [(0.0, 0.0), (W, 0.0), (W, L), (0.0, L)]
    outer_ring = [(-beside_m, -behind_far_m), (W + beside_m, -behind_far_m),
                  (W + beside_m, L + behind_near_m), (-beside_m, L + behind_near_m)]

    points: list[tuple[float, float]] = []
    for i in range(4):
        i_a, i_b = inner_ring[i], inner_ring[(i + 1) % 4]
        o_a, o_b = outer_ring[i], outer_ring[(i + 1) % 4]
        for t in np.linspace(0.0, 1.0, samples_per_edge, endpoint=False):
            inner = (i_a[0] + (i_b[0] - i_a[0]) * t, i_a[1] + (i_b[1] - i_a[1]) * t)
            outer = (o_a[0] + (o_b[0] - o_a[0]) * t, o_a[1] + (o_b[1] - o_a[1]) * t)
            placed = place(inner, outer)
            if placed is not None:
                points.append(placed)

    if len(points) < 3:
        raise ValueError("could not build a court region from these keypoints")

    # The convex hull, because the perspective image of a convex rectangle is convex.
    # Walking a sample back toward the court can otherwise leave a dent that makes the
    # polygon locally concave and the inside/outside test wrong right next to it.
    hull = cv2.convexHull(np.array(points, dtype=np.float32).reshape(-1, 1, 2))
    return np.round(hull).astype(np.int32)


def polygon_contains(polygon, point) -> bool:
    """True when `point` lies inside or on `polygon`. A missing polygon contains all."""
    if polygon is None:
        return True
    contour = np.asarray(polygon, dtype=np.int32).reshape(-1, 1, 2)
    if len(contour) < 3:
        return True
    return cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), False) >= 0


# ── Which detections belong to this court ──────────────────────────────────────

def foot_point(bbox) -> tuple[float, float]:
    """
    Where a person meets the ground: the bottom-centre of their box.

    A person is tested by their feet rather than their centre because feet are on the
    court plane and a torso is a metre and a half above it. At a low camera angle that
    difference projects a long way up the image, and the far player's centre routinely
    lands beyond the far baseline - outside a region their feet are well inside.
    """
    x1, _y1, x2, y2 = bbox
    return ((float(x1) + float(x2)) / 2.0, float(y2))


def centre_point(bbox) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0)


def filter_detections(
    detections: list[dict],
    roi=None,
    exclusions=(),
    point_of=foot_point,
) -> tuple[list[dict], dict]:
    """
    Drop detections that do not belong to the calibrated court.

    Two independent tests, because they answer different questions. `roi` is the court
    plus its playing margin and answers "is this person on this court". `exclusions` are
    regions a human marked as never relevant - the next court along, a bench, a walkway -
    and answer "is this somewhere we already know to ignore". A club video usually needs
    the first; one with a doubles match running alongside needs both.

    Returns (filtered detections, a report of what was removed). The report is returned
    rather than logged so the caller can put it in summary.json: a run that quietly
    discarded half of its detections must not look like a run that found nothing.
    """
    zones = [np.asarray(z, dtype=np.int32).reshape(-1, 1, 2) for z in (exclusions or [])]
    kept_frames: list[dict] = []
    removed_boxes = kept_boxes = 0
    removed_ids: set = set()
    kept_ids: set = set()

    for frame in detections:
        surviving = {}
        for track_id, bbox in frame.items():
            point = point_of(bbox)
            outside = roi is not None and not polygon_contains(roi, point)
            excluded = any(polygon_contains(zone, point) for zone in zones)
            if outside or excluded:
                removed_boxes += 1
                removed_ids.add(track_id)
                continue
            surviving[track_id] = bbox
            kept_ids.add(track_id)
            kept_boxes += 1
        kept_frames.append(surviving)

    return kept_frames, {
        "boxes_kept": kept_boxes,
        "boxes_removed": removed_boxes,
        "tracks_removed_entirely": len(removed_ids - kept_ids),
        "tracks_kept": len(kept_ids),
    }


# ── The stored calibration ─────────────────────────────────────────────────────

@dataclass(eq=False)
class CourtCalibration:
    """One camera position's court geometry, as placed by a human."""

    keypoints: np.ndarray                        # (14, 2)
    frame_size: tuple[int, int]                  # (width, height) the points were placed on
    clicked: dict[int, tuple[float, float]] = field(default_factory=dict)
    exclusions: list[list[tuple[float, float]]] = field(default_factory=list)
    video: str = ""
    frame_index: int = 0
    created_at: str = ""
    line_support: float | None = None
    notes: str = ""
    source_path: str = ""

    # ── conversions the pipeline needs ────────────────────────────────────────
    def flat(self) -> np.ndarray:
        """The 28-float layout every other module in this project speaks."""
        return _as_pairs(self.keypoints).reshape(-1).astype(np.float32)

    def scaled_to(self, frame_size: tuple[int, int]) -> "CourtCalibration":
        """
        The same calibration, rescaled to a different resolution.

        A fixed camera recording at 4K and at 1080p produces the same court in the same
        place, measured in different units. Rescaling is exact for that case and wrong
        for any other, so it is refused when the aspect ratio changes: a different aspect
        ratio means a different crop or a different lens setting, and the court is then
        genuinely somewhere else in the frame.
        """
        old_w, old_h = self.frame_size
        new_w, new_h = frame_size
        if (old_w, old_h) == (new_w, new_h):
            return self
        if not old_w or not old_h:
            raise ValueError("calibration does not record the frame size it was placed "
                             "on, so it cannot be rescaled")
        if abs((old_w / old_h) - (new_w / new_h)) > 0.01:
            raise ValueError(
                f"calibration was placed on a {old_w}x{old_h} frame and this clip is "
                f"{new_w}x{new_h}. The aspect ratio differs, so this is a different crop "
                f"or lens rather than the same view at another size - calibrate this "
                f"footage instead of rescaling."
            )
        sx, sy = new_w / old_w, new_h / old_h
        return CourtCalibration(
            keypoints=_as_pairs(self.keypoints) * np.array([sx, sy]),
            frame_size=(new_w, new_h),
            clicked={i: (x * sx, y * sy) for i, (x, y) in self.clicked.items()},
            exclusions=[[(x * sx, y * sy) for x, y in zone] for zone in self.exclusions],
            video=self.video,
            frame_index=self.frame_index,
            created_at=self.created_at,
            line_support=self.line_support,
            notes=self.notes,
            source_path=self.source_path,
        )

    def roi(self, frame_size=None, **margins) -> np.ndarray:
        return court_roi_polygon(self.keypoints,
                                 frame_size=frame_size or self.frame_size, **margins)

    def exclusion_contours(self) -> list[np.ndarray]:
        return [np.asarray(z, dtype=np.int32).reshape(-1, 1, 2) for z in self.exclusions]

    # ── persistence ───────────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        return {
            "version": CALIBRATION_VERSION,
            "video": self.video,
            "frame_index": int(self.frame_index),
            "frame_size": [int(self.frame_size[0]), int(self.frame_size[1])],
            "created_at": self.created_at or datetime.now().isoformat(timespec="seconds"),
            "line_support": (None if self.line_support is None
                             else round(float(self.line_support), 4)),
            "notes": self.notes,
            "keypoints": [[round(float(x), 2), round(float(y), 2)]
                          for x, y in _as_pairs(self.keypoints)],
            "clicked": {str(i): [round(float(x), 2), round(float(y), 2)]
                        for i, (x, y) in sorted(self.clicked.items())},
            "exclusions": [[[round(float(x), 1), round(float(y), 1)] for x, y in zone]
                           for zone in self.exclusions],
        }

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
            f.write("\n")
        self.source_path = str(path)
        return path

    @classmethod
    def from_dict(cls, data: dict, source_path: str = "") -> "CourtCalibration":
        version = int(data.get("version", 0))
        if version != CALIBRATION_VERSION:
            raise ValueError(
                f"calibration file is version {version}, this build reads version "
                f"{CALIBRATION_VERSION}"
            )
        size = data.get("frame_size") or [0, 0]
        return cls(
            keypoints=_as_pairs(data["keypoints"]),
            frame_size=(int(size[0]), int(size[1])),
            clicked={int(k): (float(v[0]), float(v[1]))
                     for k, v in (data.get("clicked") or {}).items()},
            exclusions=[[(float(x), float(y)) for x, y in zone]
                        for zone in (data.get("exclusions") or [])],
            video=data.get("video", ""),
            frame_index=int(data.get("frame_index", 0)),
            created_at=data.get("created_at", ""),
            line_support=data.get("line_support"),
            notes=data.get("notes", ""),
            source_path=source_path,
        )

    @classmethod
    def load(cls, path) -> "CourtCalibration":
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f), source_path=str(path))


def default_calibration_path(video_path, directory: str = DEFAULT_CALIBRATION_DIR) -> Path:
    """Where a calibration for this video is looked for, and written, by default."""
    return Path(directory) / f"{Path(video_path).stem}.json"


def find_calibration_for(video_path, directory: str = DEFAULT_CALIBRATION_DIR):
    """The calibration for this video, or None. Discovery is by video file name."""
    candidate = default_calibration_path(video_path, directory)
    return candidate if candidate.exists() else None


# ── Drawing ────────────────────────────────────────────────────────────────────

def draw_court(
    frame: np.ndarray,
    keypoints,
    colour: tuple[int, int, int] = (0, 220, 255),
    thickness: int = 2,
    numbered: bool = True,
) -> np.ndarray:
    """Draw the calibrated court wireframe onto one frame, in place."""
    kp = _as_pairs(keypoints)
    for a, b in COURT_SEGMENTS:
        cv2.line(frame, tuple(np.round(kp[a]).astype(int)),
                 tuple(np.round(kp[b]).astype(int)), colour, thickness, cv2.LINE_AA)
    for i, (x, y) in enumerate(kp):
        centre = (int(round(x)), int(round(y)))
        cv2.circle(frame, centre, thickness + 3, (0, 0, 0), -1)
        cv2.circle(frame, centre, thickness + 1, colour, -1)
        if numbered:
            cv2.putText(frame, str(i), (centre[0] + 8, centre[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def draw_court_on_video(frames: list[np.ndarray], keypoints, **kwargs) -> list[np.ndarray]:
    for frame in frames:
        draw_court(frame, keypoints, **kwargs)
    return frames


def draw_region(
    frame: np.ndarray,
    polygon,
    colour: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    contour = np.asarray(polygon, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(frame, [contour], True, colour, thickness, cv2.LINE_AA)
    return frame
