"""
tools/calibrate_court.py
------------------------
Place the court by hand, once per camera position, for footage the keypoint model
cannot read.

Why this exists
---------------
The ResNet-50 keypoint model was trained on broadcast tennis. On a phone or an action
camera behind the baseline it returns fourteen tidy points that are not on the court,
the validity gate refuses the clip, and there is nothing further the user can do. On a
club video that is not a rare failure, it is the normal case.

A fixed camera makes this a much smaller problem than the model is solving. The court
does not move, so its position is a property of the CAMERA, not of the frame: fourteen
points placed once describe every frame of every video shot from that spot. Two minutes
of clicking replaces a model that cannot generalise to this footage, and the result is
reusable for every later recording from the same position.

The same clicks answer a second question the model never could. Club footage shows the
next court along, with real people playing real tennis on it, and no detector has any
way to know which court is the one being analysed. Marking the court says which one it
is, and an optional exclusion zone removes anything that is still in the way.

Placing versus adjusting
------------------------
Two modes, and the tool moves between them on its own rather than asking.

While points are still missing, a click drops the highlighted one and moves to the next.
Once a point exists, dragging it is how it gets corrected - grab it and move it, at any
time, whether it was placed by hand or filled in from the fit. Dragging a filled-in point
turns it into a placed one, which is usually what you want: the fit's guess was close,
and nudging it is faster than starting over.

Points outside the video
------------------------
A wide camera close to the baseline routinely puts a court corner outside its own frame.
The near doubles corners are the usual ones, and on the reference clip one of them lands
70 px past the right edge. There is no pixel to click and no drag that can reach it, so
the canvas is deliberately larger than the video: the picture sits inside a border, the
area around it is where the court carries on, and a point out there is placed and dragged
like any other. The view grows on its own whenever an edit puts a point outside it, and
the arrow keys move the selected point whether it is in the picture or not.

Nothing out there can be checked against paint, because there is no paint in the picture.
The fit's own guess for such a point is usually the best answer available, and Tab leaves
it at exactly that.

Usage
-----
    python tools/calibrate_court.py clipe1.mp4
    python tools/calibrate_court.py clipe1.mp4 --frame 1500
    python tools/calibrate_court.py clipe1.mp4 --out calibration/court_A.json
    python tools/calibrate_court.py clipe1.mp4 --max-size 1280x720

The pipeline then finds it automatically:

    python main.py --input clipe1.mp4
    tennis-vision analyze clipe1.mp4

or is pointed at one explicitly, which is how a single calibration is reused across
every clip from the same camera:

    tennis-vision analyze another_clip.mp4 --court-calibration calibration/court_A.json
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.court_calibration import (                       # noqa: E402
    CLICK_ORDER,
    COURT_MODEL_M,
    COURT_SEGMENTS,
    KEYPOINT_LABELS,
    MIN_CLICKS,
    N_KEYPOINTS,
    CourtCalibration,
    court_roi_polygon,
    default_calibration_path,
    derive_keypoints,
    reprojection_residuals,
    validate_geometry,
)
from utils.court_validity import MIN_LINE_SUPPORT, line_support_score   # noqa: E402

CONTROLS = """
controls:
  left click   place the highlighted point
  drag a point move it, whether you placed it or the fit did
  arrow keys   nudge the selected point one pixel, including outside the video
  n / p        next / previous point to place
  Tab          skip this point (it is filled in from the others)
  u            undo             c    clear every placement
  , / .        step 10 frames back / forward
  m            magnifier on / off
  x            start an exclusion zone; click its corners, x again to close it
  z            delete the last exclusion zone
  v            show / hide the court region
  s            save             q    quit without saving
"""

WINDOW = "Tennis-Vision court calibration"
FALLBACK_DISPLAY = (1400, 800)
MAGNIFIER_PX = 80          # source region side, in full-resolution pixels
MAGNIFIER_ZOOM = 4
GRAB_RADIUS = 15           # display px within which a click grabs a point instead of placing
VIEW_PAD = 60              # full-res px of breathing room around the outermost point

_PLACED = (0, 220, 255)    # amber: put here by a person
_FITTED = (170, 170, 170)  # grey: derived from the others
_TARGET = (0, 255, 0)      # green: the one a click would place
_HOVER = (255, 120, 255)   # pink: the one a drag would grab
_ZONE = (0, 0, 255)
_OUTSIDE = (38, 38, 38)    # the canvas beyond the edge of the video

# Arrow keys, as cv2.waitKeyEx reports them. The Windows and GTK/Qt backends disagree,
# so both sets are listened for rather than detected.
_ARROWS = {
    2424832: (-1, 0), 65361: (-1, 0),      # left
    2555904: (1, 0),  65363: (1, 0),       # right
    2490368: (0, -1), 65362: (0, -1),      # up
    2621440: (0, 1),  65364: (0, 1),       # down
}


class CalibrationState:
    """
    The placements, with no window attached.

    Separated from the GUI loop so that the parts that can be wrong - which point is
    next, what a drag moves, what undo restores, when the geometry is complete - are
    testable without opening a window.
    """

    def __init__(self, clicks: dict[int, tuple[float, float]] | None = None):
        self.clicks: dict[int, tuple[float, float]] = dict(clicks or {})
        self.skipped: set[int] = set()
        # (index, where it was before) so undo restores a move as well as removing a
        # placement. Starts empty even when reopening a file: there is nothing from a
        # previous session to undo.
        self.history: list[tuple[int, tuple[float, float] | None]] = []
        self.exclusions: list[list[tuple[float, float]]] = []
        self.pending_zone: list[tuple[float, float]] | None = None
        self.dragging: int | None = None
        self._drag_from: tuple[float, float] | None = None
        self._drag_moved = False
        self._nudging: int | None = None
        self.cursor = self._first_unplaced()

    # ── what to place next ────────────────────────────────────────────────────
    def _first_unplaced(self) -> int:
        """Position within CLICK_ORDER, not a keypoint index: the two differ."""
        for position, index in enumerate(CLICK_ORDER):
            if index not in self.clicks and index not in self.skipped:
                return position
        return 0

    @property
    def target(self) -> int:
        """The keypoint index the next click would place."""
        return CLICK_ORDER[self.cursor % len(CLICK_ORDER)]

    @property
    def all_placed(self) -> bool:
        return len(self.clicks) >= N_KEYPOINTS

    def _set_target(self, keypoint_index: int) -> None:
        self.cursor = CLICK_ORDER.index(keypoint_index)
        self._nudging = None

    def step(self, delta: int) -> None:
        self.cursor = (self.cursor + delta) % len(CLICK_ORDER)
        self._nudging = None

    def advance_to_next_unplaced(self) -> None:
        for offset in range(1, len(CLICK_ORDER) + 1):
            index = CLICK_ORDER[(self.cursor + offset) % len(CLICK_ORDER)]
            if index not in self.clicks and index not in self.skipped:
                self._set_target(index)
                return
        self.step(1)

    # ── edits ─────────────────────────────────────────────────────────────────
    def place(self, point: tuple[float, float]) -> None:
        index = self.target
        self.history.append((index, self.clicks.get(index)))
        self.clicks[index] = (float(point[0]), float(point[1]))
        self.skipped.discard(index)
        self._nudging = None
        self.advance_to_next_unplaced()

    def skip(self) -> None:
        index = self.target
        self.history.append((index, self.clicks.get(index)))
        self.skipped.add(index)
        self.clicks.pop(index, None)
        self._nudging = None
        self.advance_to_next_unplaced()

    def undo(self) -> None:
        if not self.history:
            return
        index, previous = self.history.pop()
        if previous is None:
            self.clicks.pop(index, None)
        else:
            self.clicks[index] = previous
        self.skipped.discard(index)
        self._set_target(index)

    def clear(self) -> None:
        self.clicks.clear()
        self.skipped.clear()
        self.history.clear()
        self.dragging = None
        self._nudging = None
        self.cursor = 0

    def nudge(self, dx: float, dy: float) -> bool:
        """
        Move the selected point by whole pixels.

        The keyboard reaches a point the mouse cannot: one outside the video, one under
        the banner, or one that wants a single pixel of adjustment a hand on a mouse will
        not give. A run of nudges on the same point collapses into one undo entry, since
        holding an arrow key would otherwise fill the history with one-pixel steps.
        """
        index = self.target
        current = self.clicks.get(index)
        if current is None:
            points = self.keypoints()
            if points is None:
                return False
            current = (float(points[index][0]), float(points[index][1]))
        if self._nudging != index:
            self.history.append((index, self.clicks.get(index)))
            self._nudging = index
        self.clicks[index] = (current[0] + dx, current[1] + dy)
        self.skipped.discard(index)
        return True

    # ── dragging ──────────────────────────────────────────────────────────────
    def point_near(self, point, radius: float, positions=None) -> int | None:
        """
        Which keypoint is close enough to `point` to be grabbed, if any.

        Fitted points count as well as placed ones. A point the fit put roughly right is
        the common case, and nudging it is the whole reason drag exists.
        """
        if positions is None:
            positions = self.keypoints()
            if positions is None:
                positions = {i: p for i, p in self.clicks.items()}
        candidates = (positions.items() if isinstance(positions, dict)
                      else enumerate(positions))
        best, best_distance = None, radius
        for index, (x, y) in candidates:
            distance = float(np.hypot(x - point[0], y - point[1]))
            if distance <= best_distance:
                best, best_distance = index, distance
        return best

    def begin_drag(self, index: int) -> None:
        """
        Grab a point without moving it.

        Nothing is recorded and nothing changes until the mouse actually moves, so a
        click that merely lands on a point selects it rather than nudging it by however
        far the cursor sat from its centre.
        """
        self.dragging = index
        self._drag_from = self.clicks.get(index)
        self._drag_moved = False
        self._set_target(index)

    def drag_to(self, point: tuple[float, float]) -> None:
        if self.dragging is None:
            return
        if not self._drag_moved:
            # One undo entry per drag, not one per mouse-move event.
            self.history.append((self.dragging, self._drag_from))
            self._drag_moved = True
        self.clicks[self.dragging] = (float(point[0]), float(point[1]))
        self.skipped.discard(self.dragging)

    def end_drag(self) -> None:
        self.dragging = None
        self._drag_from = None
        self._drag_moved = False

    # ── exclusion zones ───────────────────────────────────────────────────────
    def begin_zone(self) -> None:
        self.pending_zone = []

    def add_zone_vertex(self, point: tuple[float, float]) -> None:
        if self.pending_zone is not None:
            self.pending_zone.append((float(point[0]), float(point[1])))

    def close_zone(self) -> bool:
        """Finish the zone being drawn. A zone needs three corners to enclose anything."""
        if self.pending_zone is None:
            return False
        zone, self.pending_zone = self.pending_zone, None
        if len(zone) >= 3:
            self.exclusions.append(zone)
            return True
        return False

    def drop_last_zone(self) -> None:
        if self.exclusions:
            self.exclusions.pop()

    # ── derived geometry ──────────────────────────────────────────────────────
    @property
    def is_complete(self) -> bool:
        return len(self.clicks) >= MIN_CLICKS

    def keypoints(self) -> np.ndarray | None:
        if not self.is_complete:
            return None
        points, _ = derive_keypoints(self.clicks)
        return points


# ── the view: a canvas that can be bigger than the video ──────────────────────

def view_rect(frame_size, points, pad: float = VIEW_PAD):
    """
    The full-resolution rectangle the canvas shows.

    Always at least the whole video, and enough beyond it to hold every keypoint. A wide
    camera close to the baseline puts a near doubles corner outside its own frame, and a
    canvas that stopped at the video edge left that point unclickable and undraggable.

    Returns (x0, y0, x1, y1) in the video's own coordinates, so x0 and y0 go negative
    when the court extends past the top or left edge.
    """
    width, height = frame_size
    x0, y0, x1, y1 = 0.0, 0.0, float(width), float(height)
    if points is not None and len(points):
        array = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        finite = array[np.isfinite(array).all(axis=1)]
        if len(finite):
            # Clamped, because a nearly-degenerate fit can throw a point a screen-width
            # away, and zooming out to reach it would shrink the court to nothing.
            limit_x, limit_y = width * 1.5, height * 1.5
            xs = np.clip(finite[:, 0], -limit_x, width + limit_x)
            ys = np.clip(finite[:, 1], -limit_y, height + limit_y)
            x0 = min(x0, float(xs.min()) - pad)
            y0 = min(y0, float(ys.min()) - pad)
            x1 = max(x1, float(xs.max()) + pad)
            y1 = max(y1, float(ys.max()) + pad)
    return (x0, y0, x1, y1)


def view_scale(rect, max_size) -> float:
    x0, y0, x1, y1 = rect
    return min(max_size[0] / max(x1 - x0, 1.0), max_size[1] / max(y1 - y0, 1.0), 1.0)


def to_display(point, rect, scale) -> tuple[int, int]:
    return (int(round((point[0] - rect[0]) * scale)),
            int(round((point[1] - rect[1]) * scale)))


def to_full(point, rect, scale) -> tuple[float, float]:
    return (point[0] / scale + rect[0], point[1] / scale + rect[1])


def rect_contains(rect, points, margin: float = 0.0) -> bool:
    """Whether every point is inside the view, which is what decides when it grows."""
    if points is None or not len(points):
        return True
    array = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    array = array[np.isfinite(array).all(axis=1)]
    if not len(array):
        return True
    x0, y0, x1, y1 = rect
    return bool((array[:, 0] >= x0 + margin).all() and (array[:, 0] <= x1 - margin).all()
                and (array[:, 1] >= y0 + margin).all()
                and (array[:, 1] <= y1 - margin).all())


def _screen_size(fallback=FALLBACK_DISPLAY) -> tuple[int, int]:
    """
    How much room the window actually has.

    A canvas sized to a constant was born wider than the screen on a smaller display, and
    an AUTOSIZE window cannot be dragged back into view: the controls and the outermost
    court points sat off the edge with no way to reach them.
    """
    try:
        import ctypes

        user32 = ctypes.windll.user32          # Windows only; anything else falls through
        try:
            user32.SetProcessDPIAware()
        except Exception:                      # noqa: BLE001 - already set, or refused
            pass
        width, height = int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))
        if width > 320 and height > 240:
            # Leaving room for the title bar, the taskbar and a margin of sanity.
            return (int(width * 0.92), int(height * 0.88))
    except Exception:                          # noqa: BLE001 - no ctypes, or not Windows
        pass
    return fallback


def _compose(frame: np.ndarray, rect, scale) -> np.ndarray:
    """Paint the video into the canvas, leaving the area beyond its edges visible."""
    x0, y0, x1, y1 = rect
    canvas_w = max(1, int(round((x1 - x0) * scale)))
    canvas_h = max(1, int(round((y1 - y0) * scale)))
    canvas = np.full((canvas_h, canvas_w, 3), _OUTSIDE, np.uint8)

    height, width = frame.shape[:2]
    scaled = cv2.resize(frame, (max(1, int(round(width * scale))),
                                max(1, int(round(height * scale)))),
                        interpolation=cv2.INTER_AREA)
    ox, oy = int(round(-x0 * scale)), int(round(-y0 * scale))
    src_x, src_y = max(0, -ox), max(0, -oy)
    dst_x, dst_y = max(0, ox), max(0, oy)
    w = min(scaled.shape[1] - src_x, canvas_w - dst_x)
    h = min(scaled.shape[0] - src_y, canvas_h - dst_y)
    if w > 0 and h > 0:
        canvas[dst_y:dst_y + h, dst_x:dst_x + w] = scaled[src_y:src_y + h, src_x:src_x + w]
        # The border says where the picture stops and guesswork starts.
        cv2.rectangle(canvas, (dst_x, dst_y), (dst_x + w - 1, dst_y + h - 1),
                      (100, 100, 100), 1)
    return canvas


def _magnifier(frame: np.ndarray, centre: tuple[float, float]) -> np.ndarray:
    """
    A zoomed patch of the full-resolution frame, so a line can be hit exactly.

    Beyond the edge of the video it shows empty canvas rather than clamping to the
    nearest row of pixels, which would claim there is picture where there is none.
    """
    half = MAGNIFIER_PX // 2
    height, width = frame.shape[:2]
    x, y = int(round(centre[0])), int(round(centre[1]))
    patch = np.full((MAGNIFIER_PX, MAGNIFIER_PX, 3), _OUTSIDE, np.uint8)

    src_x0, src_y0 = max(0, x - half), max(0, y - half)
    src_x1, src_y1 = min(width, x + half), min(height, y + half)
    if src_x1 > src_x0 and src_y1 > src_y0:
        patch[src_y0 - (y - half):src_y1 - (y - half),
              src_x0 - (x - half):src_x1 - (x - half)] = frame[src_y0:src_y1, src_x0:src_x1]

    big = cv2.resize(patch, None, fx=MAGNIFIER_ZOOM, fy=MAGNIFIER_ZOOM,
                     interpolation=cv2.INTER_NEAREST)
    bh, bw = big.shape[:2]
    cv2.line(big, (bw // 2, 0), (bw // 2, bh), (0, 255, 0), 1)
    cv2.line(big, (0, bh // 2), (bw, bh // 2), (0, 255, 0), 1)
    cv2.rectangle(big, (0, 0), (bw - 1, bh - 1), (255, 255, 255), 1)
    return big


def _schematic(state: CalibrationState, width: int = 150, height: int = 300) -> np.ndarray:
    """
    A plan view of the court with the point being asked for picked out.

    The prompt names a point in words - "FAR service line x LEFT singles sideline" - and
    on unfamiliar footage that sentence is harder to resolve than a picture. This says
    the same thing as a diagram, and also shows at a glance how much is left to do.
    """
    pad = 14
    canvas = np.full((height, width, 3), 32, np.uint8)
    span = COURT_MODEL_M.max(axis=0) - COURT_MODEL_M.min(axis=0)
    scale = min((width - 2 * pad) / span[0], (height - 2 * pad) / span[1])
    offset = np.array([(width - span[0] * scale) / 2, (height - span[1] * scale) / 2])
    plan = COURT_MODEL_M * scale + offset

    for a, b in COURT_SEGMENTS:
        cv2.line(canvas, tuple(plan[a].astype(int)), tuple(plan[b].astype(int)),
                 (90, 90, 90), 1, cv2.LINE_AA)
    # The net, for orientation: it is not a keypoint but it is what tells a reader
    # which end of the diagram is which.
    mid = int(offset[1] + span[1] * scale / 2)
    cv2.line(canvas, (int(offset[0]), mid),
             (int(offset[0] + span[0] * scale), mid), (60, 60, 140), 1)

    for index, (x, y) in enumerate(plan):
        centre = (int(round(x)), int(round(y)))
        if index == state.target:
            cv2.circle(canvas, centre, 6, _TARGET, 1, cv2.LINE_AA)
            cv2.circle(canvas, centre, 3, _TARGET, -1, cv2.LINE_AA)
        elif index in state.clicks:
            cv2.circle(canvas, centre, 3, _PLACED, -1, cv2.LINE_AA)
        elif index in state.skipped:
            cv2.circle(canvas, centre, 3, (100, 100, 100), 1, cv2.LINE_AA)
        else:
            cv2.circle(canvas, centre, 3, _FITTED, 1, cv2.LINE_AA)

    cv2.putText(canvas, "FAR", (pad, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.34,
                (150, 150, 150), 1, cv2.LINE_AA)
    cv2.putText(canvas, "NEAR (camera)", (pad, height - 4), cv2.FONT_HERSHEY_SIMPLEX,
                0.34, (150, 150, 150), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (0, 0), (width - 1, height - 1), (110, 110, 110), 1)
    return canvas


def _banner(canvas: np.ndarray, lines: list[tuple[str, tuple[int, int, int]]]) -> None:
    pad, line_h = 10, 24
    height = pad * 2 + line_h * len(lines)
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (canvas.shape[1], height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, canvas, 0.35, 0, canvas)
    for i, (text, colour) in enumerate(lines):
        cv2.putText(canvas, text, (pad, pad + line_h * (i + 1) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1, cv2.LINE_AA)


def _render(
    background: np.ndarray,
    frame: np.ndarray,
    state: CalibrationState,
    rect,
    scale: float,
    cursor: tuple[int, int] | None,
    hovered: int | None,
    show_magnifier: bool,
    show_roi: bool,
    status: list[tuple[str, tuple[int, int, int]]],
) -> np.ndarray:
    canvas = background.copy()

    def place(point) -> tuple[int, int]:
        return to_display(point, rect, scale)

    points = state.keypoints()
    if points is not None:
        if show_roi:
            try:
                roi = court_roi_polygon(
                    points, frame_size=(frame.shape[1], frame.shape[0]))
                shifted = np.array([place(p[0]) for p in roi], np.int32).reshape(-1, 1, 2)
                cv2.polylines(canvas, [shifted], True, (0, 255, 0), 1, cv2.LINE_AA)
            except ValueError:
                pass
        for a, b in COURT_SEGMENTS:
            measured = a in state.clicks and b in state.clicks
            cv2.line(canvas, place(points[a]), place(points[b]),
                     _PLACED if measured else _FITTED, 1, cv2.LINE_AA)
        for index, point in enumerate(points):
            centre = place(point)
            if index in (state.dragging, hovered):
                colour, radius = _HOVER, 5
            elif index == state.target:
                colour, radius = _TARGET, 4
            elif index in state.clicks:
                colour, radius = _PLACED, 3
            else:
                colour, radius = _FITTED, 3
            cv2.circle(canvas, centre, radius + 1, (0, 0, 0), -1)
            cv2.circle(canvas, centre, radius, colour, -1)
            if index in (hovered, state.dragging, state.target):
                cv2.circle(canvas, centre, radius + 5, colour, 1, cv2.LINE_AA)
            cv2.putText(canvas, str(index), (centre[0] + 7, centre[1] - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        for index, point in state.clicks.items():
            centre = place(point)
            cv2.circle(canvas, centre, 4, _PLACED, -1)
            cv2.putText(canvas, str(index), (centre[0] + 7, centre[1] - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    for zone in state.exclusions:
        pts = np.array([place(p) for p in zone], np.int32).reshape(-1, 1, 2)
        shade = canvas.copy()
        cv2.fillPoly(shade, [pts], _ZONE)
        cv2.addWeighted(shade, 0.25, canvas, 0.75, 0, canvas)
        cv2.polylines(canvas, [pts], True, _ZONE, 2, cv2.LINE_AA)

    if state.pending_zone:
        pts = np.array([place(p) for p in state.pending_zone], np.int32)
        cv2.polylines(canvas, [pts.reshape(-1, 1, 2)], False, _ZONE, 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(canvas, tuple(p), 4, _ZONE, -1)

    # The crosshair marks where a click would LAND. It is hidden while the cursor is over
    # a point, because there a click grabs rather than places, and showing the placement
    # cursor would say the opposite.
    if (cursor is not None and state.pending_zone is None
            and hovered is None and state.dragging is None and not state.all_placed):
        cv2.drawMarker(canvas, cursor, _TARGET, cv2.MARKER_CROSS, 22, 1)

    _banner(canvas, status)

    if state.pending_zone is None:
        plan = _schematic(state)
        ph, pw = plan.shape[:2]
        top = 10 + len(status) * 24 + 14
        left = canvas.shape[1] - pw - 10
        # Skipped rather than clipped on a window too small to hold it: half a court
        # diagram says less than none, and the words in the banner still do the job.
        if top + ph <= canvas.shape[0] and left >= 0:
            canvas[top:top + ph, left:left + pw] = plan

    if show_magnifier and cursor is not None:
        inset = _magnifier(frame, to_full(cursor, rect, scale))
        ih, iw = inset.shape[:2]
        # Put it in whichever bottom corner the cursor is not in.
        x0 = canvas.shape[1] - iw - 10 if cursor[0] < canvas.shape[1] // 2 else 10
        y0 = canvas.shape[0] - ih - 10
        if y0 >= 0 and x0 >= 0:
            canvas[y0:y0 + ih, x0:x0 + iw] = inset
    return canvas


def _status_lines(state, frame_index, total, support, residual, hovered, outside) -> list:
    white, green, amber, red = ((255, 255, 255), (120, 255, 120),
                                (0, 200, 255), (0, 80, 255))

    if state.pending_zone is not None:
        head = (f"EXCLUSION ZONE: click its corners ({len(state.pending_zone)} so far), "
                f"x to close it, z to delete the last one", _ZONE)
    elif state.dragging is not None:
        head = (f"moving point {state.dragging}: {KEYPOINT_LABELS[state.dragging]}",
                _HOVER)
    elif hovered is not None:
        head = (f"point {hovered}: {KEYPOINT_LABELS[hovered]}  -  drag it to move it",
                _HOVER)
    elif state.all_placed:
        head = ("all 14 placed. Drag a point, or arrow-key the selected one, then s to "
                "save.", green)
    else:
        index = state.target
        head = (f"[{len(state.clicks)}/{N_KEYPOINTS}] click point {index}: "
                f"{KEYPOINT_LABELS[index]}", green)

    second = f"frame {frame_index}/{max(total - 1, 0)}"
    if support is not None:
        verdict = "passes" if support >= MIN_LINE_SUPPORT else "below the automatic gate"
        second += f"   |   line support {support:.3f} ({verdict})"
    if residual is not None:
        second += f"   |   lens error {residual:.1f}px"
    colour = white
    if support is not None and support < MIN_LINE_SUPPORT:
        colour = amber
    if residual is not None and residual > 10:
        colour = amber

    third = ("click place  |  drag to move  |  arrows nudge  |  n/p point  |  Tab skip  "
             "|  u undo  |  ,/. frame  |  x zone  |  v region  |  m zoom  |  s save",
             white)
    if not state.is_complete:
        third = (f"{MIN_CLICKS - len(state.clicks)} more point(s) before a court can be "
                 f"fitted. The four outer corners come first.", red)
    elif outside:
        names = ", ".join(str(i) for i in outside)
        third = (f"point(s) {names} lie outside the video. Place them on the grey border "
                 f"by eye, or press Tab to leave them to the fit - there is no paint out "
                 f"there to check against.", amber)
    return [head, (second, colour), third]


def _read_frame(cap: cv2.VideoCapture, index: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, frame = cap.read()
    return frame if ok else None


def run(video: str, frame_index: int, out_path: Path, existing, max_size):
    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if not total or not width:
        print(f"error: could not read {video}", file=sys.stderr)
        return 2

    state = CalibrationState(existing.clicked if existing else None)
    if existing:
        state.exclusions = [list(z) for z in existing.exclusions]
        if not existing.clicked:
            # An older file, or one edited by hand, may carry only the fourteen points.
            # Treat them all as placed rather than discarding the calibration.
            state.clicks = {i: (float(x), float(y))
                            for i, (x, y) in enumerate(existing.keypoints)}
        state.cursor = 0

    frame_index = int(np.clip(frame_index, 0, total - 1))
    frame = _read_frame(cap, frame_index)
    if frame is None:
        print(f"error: could not read frame {frame_index} of {video}", file=sys.stderr)
        return 2

    view: dict = {"rect": None, "scale": 1.0, "background": None}

    def refit_view():
        view["rect"] = view_rect((width, height), state.keypoints())
        view["scale"] = view_scale(view["rect"], max_size)
        view["background"] = _compose(frame, view["rect"], view["scale"])

    refit_view()
    cursor: list[tuple[int, int]] = [(0, 0)]
    show_magnifier, show_roi = [True], [True]
    cached: dict = {"support": None, "residual": None, "key": None}

    def on_mouse(event, x, y, _flags, _param):
        cursor[0] = (x, y)
        full = to_full((x, y), view["rect"], view["scale"])

        if state.pending_zone is not None:
            if event == cv2.EVENT_LBUTTONDOWN:
                state.add_zone_vertex(full)
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            index = state.point_near(full, GRAB_RADIUS / view["scale"])
            if index is not None:
                state.begin_drag(index)
            elif not state.all_placed:
                # Only when something is still missing. Once all fourteen exist, a click
                # on empty court would silently teleport whichever point happened to be
                # selected, which is the opposite of what the click meant.
                state.place(full)
        elif event == cv2.EVENT_MOUSEMOVE and state.dragging is not None:
            state.drag_to(full)
        elif event == cv2.EVENT_LBUTTONUP:
            state.end_drag()

    try:
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW, on_mouse)
    except cv2.error as exc:
        print("error: this build of OpenCV cannot open a window "
              f"({exc}). The calibration tool needs a desktop session.", file=sys.stderr)
        return 2

    print(f"\n{video}: {total} frames at {width}x{height}")
    print(f"window fitted to at most {max_size[0]}x{max_size[1]}")
    print(f"calibrating on frame {frame_index}; saving to {out_path}\n")

    while True:
        points = state.keypoints()

        # The view grows when an edit puts a point outside it, and only between edits:
        # doing it mid-drag would pull the picture out from under the mouse.
        if state.dragging is None and not rect_contains(view["rect"], points, margin=8):
            refit_view()

        # Line support is recomputed only when the placements settle. It blurs the whole
        # frame on every call, which is far too slow for one call per mouse-move event,
        # so a drag keeps the last figure until the button comes up.
        key_state = (tuple(sorted(state.clicks.items())), frame_index)
        if key_state != cached["key"] and state.dragging is None:
            cached["key"] = key_state
            if points is not None:
                cached["support"] = float(line_support_score(frame, points.reshape(-1)))
                cached["residual"] = float(reprojection_residuals(points).max())
            else:
                cached["support"] = cached["residual"] = None

        hovered = None
        if state.pending_zone is None and state.dragging is None:
            hovered = state.point_near(to_full(cursor[0], view["rect"], view["scale"]),
                                       GRAB_RADIUS / view["scale"])

        outside = []
        if points is not None:
            outside = [i for i, (px, py) in enumerate(points)
                       if not (0 <= px < width and 0 <= py < height)]

        canvas = _render(
            view["background"], frame, state, view["rect"], view["scale"],
            cursor[0], hovered, show_magnifier[0], show_roi[0],
            _status_lines(state, frame_index, total, cached["support"],
                          cached["residual"], hovered, outside),
        )
        cv2.imshow(WINDOW, canvas)
        raw = cv2.waitKeyEx(20)
        key = (raw & 0xFF) if raw != -1 else 255

        if raw in _ARROWS:
            dx, dy = _ARROWS[raw]
            state.nudge(dx, dy)
            continue
        if key in (ord("q"), 27):
            print("quit without saving")
            cv2.destroyAllWindows()
            cap.release()
            return 1
        if key == ord("n"):
            state.step(1)
        elif key == ord("p"):
            state.step(-1)
        elif key == 9:                      # Tab
            state.skip()
        elif key == ord("u"):
            state.undo()
        elif key == ord("c"):
            state.clear()
            refit_view()
        elif key == ord("m"):
            show_magnifier[0] = not show_magnifier[0]
        elif key == ord("v"):
            show_roi[0] = not show_roi[0]
        elif key == ord("x"):
            if state.pending_zone is None:
                state.begin_zone()
            elif not state.close_zone():
                print("  a zone needs at least three corners - discarded")
        elif key == ord("z"):
            state.drop_last_zone()
        elif key in (ord(","), ord(".")):
            step = -10 if key == ord(",") else 10
            moved = int(np.clip(frame_index + step, 0, total - 1))
            new_frame = _read_frame(cap, moved)
            if new_frame is not None:
                frame_index, frame = moved, new_frame
                refit_view()
        elif key == ord("s"):
            saved = _save(state, video, frame_index, (width, height),
                          cached["support"], out_path)
            if saved:
                cv2.destroyAllWindows()
                cap.release()
                return 0

    return 0


def _save(state, video, frame_index, frame_size, support, out_path) -> bool:
    if not state.is_complete:
        print(f"  cannot save: {MIN_CLICKS} points are needed and "
              f"{len(state.clicks)} are placed")
        return False

    points = state.keypoints()
    problems = validate_geometry(points)
    if problems:
        # Refused rather than warned. Every one of these is an ordering mistake that
        # produces a self-consistent court pointing the wrong way, and a calibration is
        # trusted by the pipeline in place of a gate - so a wrong one is silent.
        print("\n  NOT SAVED. These points do not describe a court:")
        for problem in problems:
            print(f"    - {problem}")
        print("  Drag the offending point to where it belongs, or press c to start "
              "again.\n")
        return False

    residuals = reprojection_residuals(points)
    calibration = CourtCalibration(
        keypoints=points,
        frame_size=frame_size,
        clicked=state.clicks,
        exclusions=state.exclusions,
        video=Path(video).name,
        frame_index=frame_index,
        line_support=support,
        notes=f"placed by hand on {len(state.clicks)} of {N_KEYPOINTS} points",
    )
    path = calibration.save(out_path)

    outside = [i for i, (x, y) in enumerate(points)
               if not (0 <= x < frame_size[0] and 0 <= y < frame_size[1])]

    print(f"\n  saved {path}")
    print(f"    points placed by hand : {len(state.clicks)} of {N_KEYPOINTS}")
    print(f"    exclusion zones       : {len(state.exclusions)}")
    if outside:
        print(f"    outside the video     : {', '.join(str(i) for i in outside)} "
              f"(the court leaves the frame there, so those cannot be checked "
              f"against paint)")
    if support is not None:
        verdict = ("above" if support >= MIN_LINE_SUPPORT else "below")
        print(f"    line support          : {support:.3f} "
              f"({verdict} the {MIN_LINE_SUPPORT} automatic gate)")
        if support < MIN_LINE_SUPPORT:
            print("      That is expected on a wide lens and is not by itself a "
                  "problem: the gate samples straight segments between the corners, "
                  "which a curved line does not follow. Your eyes on the overlay are "
                  "the better check, which is why the pipeline trusts a calibration "
                  "instead of re-running that gate.")
    print(f"    lens error            : median {np.median(residuals):.1f}px, "
          f"max {residuals.max():.1f}px")
    if residuals.max() > 10:
        print("      Above 10px the single homography is a visible compromise, which is "
              "lens distortion rather than a bad click. Positions mapped through it "
              "carry that error.")
    print(f"\n  the pipeline will now find this automatically:\n"
          f"    tennis-vision analyze {video}\n")
    return True


def _parse_size(text: str) -> tuple[int, int]:
    try:
        width, height = (int(part) for part in text.lower().replace("*", "x").split("x"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected WxH, got {text!r}") from None
    if width < 320 or height < 240:
        raise argparse.ArgumentTypeError("a window smaller than 320x240 is unusable")
    return (width, height)


def main() -> int:
    # Not description=__doc__. The module docstring is long, and argparse writes help to
    # a console that is cp1252 on Windows, so anything exotic in it kills --help with a
    # UnicodeEncodeError before a line is printed. The help text stays short and ASCII.
    parser = argparse.ArgumentParser(
        prog="tennis-vision calibrate",
        description=("Place a tennis court by hand, once per camera position, for "
                     "footage the keypoint model cannot read."),
        epilog=CONTROLS,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", help="video to calibrate against")
    parser.add_argument("--frame", type=int, default=None,
                        help="frame to place the points on (default: the middle of the "
                             "clip, which is more likely to show an unobstructed court "
                             "than frame 0)")
    parser.add_argument("--out", default=None,
                        help="where to write the calibration "
                             "(default: calibration/<video name>.json)")
    parser.add_argument("--fresh", action="store_true",
                        help="ignore an existing calibration for this video instead of "
                             "reopening it for refinement")
    parser.add_argument("--max-size", type=_parse_size, default=None, metavar="WxH",
                        help="largest window to open, e.g. 1280x720 (default: fitted to "
                             "the screen). The window cannot be resized once open, so "
                             "use this if it comes up off the edge of the display.")
    args = parser.parse_args()

    if not Path(args.video).exists():
        print(f"error: video not found: {args.video}", file=sys.stderr)
        return 2

    out_path = Path(args.out) if args.out else default_calibration_path(args.video)

    existing = None
    if not args.fresh and out_path.exists():
        try:
            existing = CourtCalibration.load(out_path)
            print(f"reopening {out_path} ({len(existing.clicked)} placed points, "
                  f"{len(existing.exclusions)} exclusion zones)")
        except (ValueError, KeyError, OSError) as exc:
            print(f"warning: could not read {out_path} ({exc}); starting fresh",
                  file=sys.stderr)

    frame_index = args.frame
    if frame_index is None:
        probe = cv2.VideoCapture(args.video)
        frame_index = int((probe.get(cv2.CAP_PROP_FRAME_COUNT) or 2) // 2)
        probe.release()
        if existing is not None:
            frame_index = existing.frame_index

    return run(args.video, frame_index, out_path, existing,
               args.max_size or _screen_size())


if __name__ == "__main__":
    sys.exit(main())
