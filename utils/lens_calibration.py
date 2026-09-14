"""
utils/lens_calibration.py
--------------------------
Measure a camera's lens distortion from a checkerboard, so it can be corrected in
positions the pipeline already computes - ball, feet, keypoints - before the homography
that turns them into real-world coordinates.

Why this exists
----------------
A homography assumes straight lines project to straight lines. Every camera this
project has been pointed at so far is a wide action camera close to the court, and its
lens does not honour that assumption: a painted line that is straight on the court
visibly bows in the frame. `utils.court_calibration` already measures the COST of that
- on the reference clip, a single homography sits about 10px rms from the painted lines
it was fitted to - but it does not correct it, because fitting distortion to the
court's own lines was tried and did not converge to a physical lens model (the court
gives at most nine lines and their curvature confounds with where the corners were
placed). A checkerboard gives many known-straight lines, seen from many angles, which
is what a real calibration needs.

Two models, not one assumed
-----------------------------
OpenCV offers two lens models: the standard polynomial model (`cv2.calibrateCamera`,
radial terms k1-k3 plus tangential p1-p2) and the fisheye model
(`cv2.fisheye.calibrate`, terms k1-k4 under an equidistant projection). Which one
actually fits a given action camera is not something to assume - a "wide" lens is not
necessarily a true fisheye, and forcing the wrong model produces confidently wrong
coefficients with a deceptively small-looking reprojection error over the CENTRE of the
frame while being worse at the edges, exactly where the distortion this project cares
about is largest. This module fits both from the same detected corners and reports the
reprojection error of each, so the choice is measured rather than assumed.

What this does NOT do
-----------------------
It does not touch main.py or any position the pipeline already computes. Producing a
trustworthy set of lens coefficients and a tested way to apply them to a list of points
is this module's whole job. Deciding where in the pipeline to apply that - to the ball,
to player feet, to the court keypoints themselves, and confirming it actually reduces
the court's own reprojection error rather than moving it around - is a separate,
riskier change with a real "confidently wrong number in metres" failure mode, and
belongs in its own change once this module's output has been checked against real
footage from the camera it claims to describe.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np


class FitResult(NamedTuple):
    """
    What one calibration attempt produced. A NamedTuple rather than a plain tuple so a
    caller who only wants the headline numbers can unpack `rms, K, dist, *_ = ...`, and
    one who also wants per-view poses - to report which view fit worst, say - can take
    the whole thing and read `.rvecs` / `.tvecs` by name instead of by a position
    nobody remembers.
    """
    rms: float
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    rvecs: list
    tvecs: list

MODEL_STANDARD = "standard"
MODEL_FISHEYE = "fisheye"
CALIBRATION_VERSION = 1

# A calibration fitted from fewer views than this is not trustworthy: too few equations
# for the number of unknowns (a camera matrix plus 4-5 distortion terms), and prone to
# fitting noise in whichever few poses happened to be supplied rather than the lens
# itself. Not a hard OpenCV requirement, a practical one - cv2.calibrateCamera will
# happily return an answer from 3 views, just not a good one.
MIN_VIEWS = 10

# Below this fraction of the detections must be right-of-centre AND left-of-centre (and
# similarly top/bottom) for the board to be judged as having covered the frame, rather
# than being waved around in the middle of it where distortion is smallest and hardest
# to measure.
MIN_COVERAGE_SPAN = 0.5


@dataclass(frozen=True)
class BoardSpec:
    """The checkerboard being calibrated against."""

    inner_corners: tuple[int, int]   # (columns, rows) of INNER corners, not squares
    square_size: float = 1.0         # real-world units; only the RATIO to other views
                                      # matters for distortion, so the unit is free -
                                      # millimetres if you plan to trust the intrinsics'
                                      # absolute scale too, otherwise leave it at 1.0.

    def object_points(self) -> np.ndarray:
        """The board's own corners, in its own flat plane, Z=0."""
        cols, rows = self.inner_corners
        points = np.zeros((rows * cols, 3), np.float32)
        points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * self.square_size
        return points


def find_board_corners(
    image: np.ndarray, board: BoardSpec, refine: bool = True
) -> np.ndarray | None:
    """
    Locate the board's inner corners in one image, or None if it is not visible.

    Refined to sub-pixel accuracy by default: `cv2.findChessboardCorners` alone is
    accurate to about a pixel, and a calibration fitted from unrefined corners is
    measurably noisier for no reason - the refinement is cheap relative to everything
    else this module does.
    """
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    found, corners = cv2.findChessboardCorners(
        grey, board.inner_corners,
        cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK,
    )
    if not found:
        return None
    if refine:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(grey, corners, (11, 11), (-1, -1), criteria)
    return corners.reshape(-1, 2)


def coverage_span(all_corners: list[np.ndarray], image_size: tuple[int, int]) -> dict:
    """
    How much of the FRAME the detected boards actually covered, not how many detections
    there were - ten photos of a board waved around one corner of the frame measure
    that corner's distortion ten times and say nothing about the rest.

    Returns fractional bounding box of all corner centres: left/right/top/bottom in
    [0, 1], plus whether it clears MIN_COVERAGE_SPAN in each axis.
    """
    width, height = image_size
    centres = np.array([c.mean(axis=0) for c in all_corners])
    if len(centres) == 0:
        return {"x_span": 0.0, "y_span": 0.0, "wide_enough": False}
    x_span = (centres[:, 0].max() - centres[:, 0].min()) / width
    y_span = (centres[:, 1].max() - centres[:, 1].min()) / height
    return {
        "x_span": round(float(x_span), 3),
        "y_span": round(float(y_span), 3),
        "wide_enough": bool(x_span >= MIN_COVERAGE_SPAN and y_span >= MIN_COVERAGE_SPAN),
    }


# ── fitting the two models ──────────────────────────────────────────────────


def fit_standard(
    object_points: list[np.ndarray], image_points: list[np.ndarray],
    image_size: tuple[int, int],
) -> FitResult:
    """cv2.calibrateCamera: k1,k2,p1,p2,k3."""
    rms, camera_matrix, dist, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None,
    )
    return FitResult(float(rms), camera_matrix, dist.reshape(-1), list(rvecs), list(tvecs))


def _fisheye_flag(name: str) -> int:
    """
    The fisheye CALIB_* flags moved between `cv2.fisheye.CALIB_*` and top-level
    `cv2.CALIB_*` across OpenCV's Python binding versions (present under `cv2.fisheye`
    in 4.x, only under `cv2` directly in the 5.0 build this was developed against).
    Tried in that order so this keeps working either way rather than being pinned to
    whichever layout happened to be on the machine that wrote it.
    """
    if hasattr(cv2.fisheye, name):
        return getattr(cv2.fisheye, name)
    if hasattr(cv2, name):
        return getattr(cv2, name)
    raise AttributeError(f"neither cv2.fisheye nor cv2 exposes {name}")


def fit_fisheye(
    object_points: list[np.ndarray], image_points: list[np.ndarray],
    image_size: tuple[int, int],
) -> FitResult:
    """
    cv2.fisheye.calibrate: k1,k2,k3,k4 under an equidistant model.

    Raises the same errors OpenCV does (typically for a degenerate/near-planar view
    set) rather than swallowing them, since a caller comparing this against
    `fit_standard` needs to know a model could not be fit at all, not receive a
    fabricated result in its place.
    """
    # cv2.fisheye.calibrate wants each view shaped (1, N, 3) / (1, N, 2) - N in the
    # SECOND axis, not the first like cv2.calibrateCamera takes. Passing the other
    # convention does not raise a clear shape error; it fails deep inside with an
    # unrelated-looking "arithm_op ... Sizes of input arguments do not match".
    obj = [p.reshape(1, -1, 3).astype(np.float64) for p in object_points]
    img = [p.reshape(1, -1, 2).astype(np.float64) for p in image_points]
    camera_matrix = np.eye(3)
    dist = np.zeros(4)
    flags = _fisheye_flag("CALIB_RECOMPUTE_EXTRINSIC") | _fisheye_flag("CALIB_FIX_SKEW")
    rms, camera_matrix, dist, rvecs, tvecs = cv2.fisheye.calibrate(
        obj, img, image_size, camera_matrix, dist, flags=flags,
    )
    return FitResult(float(rms), camera_matrix, dist.reshape(-1), list(rvecs), list(tvecs))


# ── applying a calibration to points the pipeline already has ─────────────────


def undistort_points(
    points: np.ndarray, camera_matrix: np.ndarray, dist_coeffs: np.ndarray, model: str,
) -> np.ndarray:
    """
    Where each point in `points` (Nx2, pixel coordinates) would be if this lens had no
    distortion - still in PIXEL coordinates (re-projected through the same camera
    matrix), not normalised camera coordinates, so the result drops straight into code
    that already expects pixels.

    This is the one function code outside this module should call to actually correct
    a position. It does not know or care whether the points are ball detections, feet,
    or court keypoints - that is the caller's business.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    if model == MODEL_FISHEYE:
        out = cv2.fisheye.undistortPoints(pts, camera_matrix, dist_coeffs, P=camera_matrix)
    elif model == MODEL_STANDARD:
        out = cv2.undistortPoints(pts, camera_matrix, dist_coeffs, P=camera_matrix)
    else:
        raise ValueError(f"unknown lens model: {model!r}")
    return out.reshape(-1, 2)


def reprojection_error_per_view(
    object_points: list[np.ndarray], image_points: list[np.ndarray],
    camera_matrix: np.ndarray, dist_coeffs: np.ndarray, model: str,
    rvecs, tvecs,
) -> list[float]:
    """Per-VIEW rms error, not just the one pooled figure calibrateCamera returns -
    a single bad photo (motion blur, a corner mis-detected) can hide inside a low
    overall average and is worth being able to see and drop."""
    errors = []
    for obj, img, rvec, tvec in zip(object_points, image_points, rvecs, tvecs):
        if model == MODEL_FISHEYE:
            # (1, N, 3), same binding quirk as fit_fisheye - see the comment there.
            projected, _ = cv2.fisheye.projectPoints(
                obj.reshape(1, -1, 3).astype(np.float64), rvec, tvec,
                camera_matrix, dist_coeffs)
        else:
            projected, _ = cv2.projectPoints(obj, rvec, tvec, camera_matrix, dist_coeffs)
        projected = projected.reshape(-1, 2)
        diff = projected - img.reshape(-1, 2)
        errors.append(float(np.sqrt((diff ** 2).sum(axis=1).mean())))
    return errors


# ── the stored result ───────────────────────────────────────────────────────


@dataclass(eq=False)
class LensCalibration:
    """One camera's measured distortion, with enough provenance to trust or reject it."""

    model: str                         # MODEL_STANDARD or MODEL_FISHEYE
    camera_matrix: np.ndarray          # 3x3
    dist_coeffs: np.ndarray            # (4,) or (5,) depending on model
    image_size: tuple[int, int]
    rms_error: float
    views_used: int
    coverage: dict
    other_model_rms: float | None = None   # the model NOT chosen, for audit
    video: str = ""
    created_at: str = ""
    notes: str = ""
    source_path: str = ""

    def undistort(self, points: np.ndarray) -> np.ndarray:
        return undistort_points(points, self.camera_matrix, self.dist_coeffs, self.model)

    def scaled_to(self, image_size: tuple[int, int]) -> "LensCalibration":
        """
        The same lens, described for footage at a different resolution.

        The camera matrix scales linearly with resolution (fx, fy, cx, cy all move
        together); the distortion coefficients are dimensionless and do not change.
        Refused across a different aspect ratio for the same reason
        CourtCalibration.scaled_to refuses one: a different aspect ratio is a different
        crop or a different lens setting, not the same picture at another size.
        """
        old_w, old_h = self.image_size
        new_w, new_h = image_size
        if (old_w, old_h) == (new_w, new_h):
            return self
        if not old_w or not old_h:
            raise ValueError("calibration does not record the image size it was fit "
                             "on, so it cannot be rescaled")
        if abs((old_w / old_h) - (new_w / new_h)) > 0.01:
            raise ValueError(
                f"calibration was fit at {old_w}x{old_h} and this is {new_w}x{new_h} - "
                f"the aspect ratio differs, so this is a different crop or lens rather "
                f"than the same view at another size."
            )
        scale = new_w / old_w
        matrix = self.camera_matrix.copy()
        matrix[0, 0] *= scale   # fx
        matrix[1, 1] *= scale   # fy
        matrix[0, 2] *= scale   # cx
        matrix[1, 2] *= scale   # cy
        return LensCalibration(
            model=self.model, camera_matrix=matrix, dist_coeffs=self.dist_coeffs,
            image_size=(new_w, new_h), rms_error=self.rms_error,
            views_used=self.views_used, coverage=self.coverage,
            other_model_rms=self.other_model_rms, video=self.video,
            created_at=self.created_at, notes=self.notes, source_path=self.source_path,
        )

    def to_dict(self) -> dict:
        return {
            "version": CALIBRATION_VERSION,
            "model": self.model,
            "camera_matrix": self.camera_matrix.tolist(),
            "dist_coeffs": self.dist_coeffs.tolist(),
            "image_size": [int(self.image_size[0]), int(self.image_size[1])],
            "rms_error_px": round(float(self.rms_error), 4),
            "other_model_rms_px": (None if self.other_model_rms is None
                                   else round(float(self.other_model_rms), 4)),
            "views_used": int(self.views_used),
            "coverage": self.coverage,
            "video": self.video,
            "created_at": self.created_at or datetime.now().isoformat(timespec="seconds"),
            "notes": self.notes,
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
    def from_dict(cls, data: dict, source_path: str = "") -> "LensCalibration":
        version = int(data.get("version", 0))
        if version != CALIBRATION_VERSION:
            raise ValueError(
                f"lens calibration file is version {version}, this build reads "
                f"version {CALIBRATION_VERSION}"
            )
        size = data.get("image_size") or [0, 0]
        return cls(
            model=data["model"],
            camera_matrix=np.asarray(data["camera_matrix"], dtype=np.float64),
            dist_coeffs=np.asarray(data["dist_coeffs"], dtype=np.float64),
            image_size=(int(size[0]), int(size[1])),
            rms_error=float(data["rms_error_px"]),
            views_used=int(data.get("views_used", 0)),
            coverage=data.get("coverage", {}),
            other_model_rms=data.get("other_model_rms_px"),
            video=data.get("video", ""),
            created_at=data.get("created_at", ""),
            notes=data.get("notes", ""),
            source_path=source_path,
        )

    @classmethod
    def load(cls, path) -> "LensCalibration":
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f), source_path=str(path))
