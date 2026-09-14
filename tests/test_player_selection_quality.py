"""
tests/test_player_selection_quality.py
───────────────────────────────────────
Guards the player-selection sanity gate.

Why it exists: the court-validity gate stops a clip whose COURT was fitted to the crowd.
Nothing stopped a clip whose PLAYERS were. Measured across the nine evaluation clips
(`eval/player_selection_sanity.py`), `input_video_11` passes the court gate comfortably at
0.327 line support and still selects two tracks on the same side of the net, one present
for 40% of frames with a 199-frame hole in the middle. The pipeline reported confident
per-player numbers on it.

Singles is played across the net, so "both selected tracks are on the same half" is
decidable with no ground truth at all. That is the same kind of reasoning the rally
grammar and the court gate already use, and it is what these tests pin.

They also pin the serialisation, because the first working version of this gate wrote a
numpy.bool_ into summary.json, which json.dump refuses: the file was written as far as
that key and then truncated. A gate whose output cannot be saved reports nothing.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from utils.player_selection import (
    MIN_COVERAGE,
    SELECTION_DEGRADED,
    SELECTION_FAILED,
    SELECTION_OK,
    assess_selection,
)

NET_Y = 400.0
NEAR_FEET = 600.0    # below the net line in image space
FAR_FEET = 200.0     # above it


def frames(n=100, near=None, far=None):
    """n frames with player 1 near and player 2 far, present on the given frame indices."""
    near = range(n) if near is None else near
    far = range(n) if far is None else far
    out = []
    for i in range(n):
        frame = {}
        if i in near:
            frame[1] = [10.0, 100.0, 40.0, NEAR_FEET]
        if i in far:
            frame[2] = [10.0, 100.0, 40.0, FAR_FEET]
        out.append(frame)
    return out


# ── the healthy case ────────────────────────────────────────────────────────────

def test_two_players_tracked_throughout_is_ok():
    result = assess_selection(frames(), NET_Y, people_detected=14)

    assert result.status == SELECTION_OK
    assert result.is_ok
    assert result.opposite_sides
    assert result.coverage == (1.0, 1.0)
    assert result.longest_gaps == (0, 0)


def test_a_few_scattered_misses_are_still_ok():
    """Real tracking drops the odd frame. The gate must not fire on that."""
    present = [i for i in range(100) if i % 25]
    result = assess_selection(frames(near=present), NET_Y, people_detected=14)

    assert result.status == SELECTION_OK


# ── the decisive failure ────────────────────────────────────────────────────────

def test_both_tracks_on_the_same_side_is_a_failure():
    """
    input_video_11's signature. Singles is played across the net, so two tracks on one
    half cannot both be players, whatever their coverage looks like.
    """
    same_side = [{1: [10.0, 100.0, 40.0, NEAR_FEET],
                  2: [50.0, 100.0, 80.0, NEAR_FEET + 20]} for _ in range(100)]

    result = assess_selection(same_side, NET_Y, people_detected=40)

    assert result.status == SELECTION_FAILED
    assert not result.opposite_sides
    assert "same side of the net" in result.reason
    assert "not measurements" in result.reason


def test_same_side_outranks_good_coverage():
    """
    Perfect coverage must not rescue a selection that took the wrong people. The
    coverage checks are secondary; the geometry check is decisive.
    """
    same_side = [{1: [10.0, 100.0, 40.0, FAR_FEET],
                  2: [50.0, 100.0, 80.0, FAR_FEET - 20]} for _ in range(100)]

    result = assess_selection(same_side, NET_Y, people_detected=40)

    assert result.coverage == (1.0, 1.0)
    assert result.status == SELECTION_FAILED


# ── degraded, not failed ────────────────────────────────────────────────────────

def test_a_long_continuous_gap_is_degraded():
    """
    Right players, lost for a stretch. Shots played then cannot be attributed, so the
    counts under-count, which is a different and lesser problem than picking the wrong
    people.
    """
    present = list(range(0, 40)) + list(range(70, 100))
    result = assess_selection(frames(near=present), NET_Y, people_detected=14)

    assert result.status == SELECTION_DEGRADED
    assert result.opposite_sides
    assert result.longest_gaps[0] == 30
    assert "under-count" in result.reason


def test_low_coverage_is_degraded():
    present = list(range(0, 50))
    result = assess_selection(frames(near=present), NET_Y, people_detected=14)

    assert result.status == SELECTION_DEGRADED
    assert result.coverage[0] < MIN_COVERAGE


def test_a_missing_player_is_a_failure_not_a_gap():
    """One side never selected at all cannot be opposite sides, so it fails outright."""
    result = assess_selection(frames(far=[]), NET_Y, people_detected=14)

    assert result.status == SELECTION_FAILED


# ── the output has to survive being written ─────────────────────────────────────

def test_as_dict_is_json_serialisable_with_numpy_inputs():
    """
    The regression. Bounding boxes arrive as numpy scalars, so the comparison produced a
    numpy.bool_ and json.dump truncated summary.json at that key.
    """
    numpy_frames = [{1: [np.float32(10), np.float32(100), np.float32(40),
                         np.float32(NEAR_FEET)],
                     2: [np.float32(10), np.float32(100), np.float32(40),
                         np.float32(FAR_FEET)]} for _ in range(50)]

    payload = assess_selection(numpy_frames, np.float32(NET_Y),
                               people_detected=np.int64(12)).as_dict()

    text = json.dumps(payload)
    assert json.loads(text)["players_on_opposite_sides"] is True
    for key, value in payload.items():
        assert type(value) in (str, int, float, bool), f"{key} is {type(value)}"


def test_failed_and_degraded_payloads_carry_the_warning():
    same_side = [{1: [10.0, 100.0, 40.0, NEAR_FEET],
                  2: [50.0, 100.0, 80.0, NEAR_FEET]} for _ in range(50)]

    assert "warning" in assess_selection(same_side, NET_Y, 40).as_dict()
    assert "warning" not in assess_selection(frames(), NET_Y, 14).as_dict()


# ── never raises ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("selected", [[], [{}], [{} for _ in range(10)]])
def test_empty_input_does_not_raise(selected):
    """
    A clip where nobody was selected must report a failure, not crash the run. The gate
    exists to describe bad outcomes, so a bad outcome cannot be an exception path.
    """
    result = assess_selection(selected, NET_Y, people_detected=0)
    assert result.status == SELECTION_FAILED
    json.dumps(result.as_dict())


# ── near_pid: which id sat closer to the camera in THIS clip ───────────────────
#
# Not a claim about which physical person that is across a whole match - a change of
# ends swaps it mid-session, and nothing in this module knows that happened. See
# tools/batch_analyze.py's module docstring for the identity problem this does and does
# not help with.

def test_near_pid_names_whichever_id_sat_lower_in_the_image():
    """The default fixture already puts player 1 near (larger image y) and player 2
    far - the healthy, unambiguous case."""
    result = assess_selection(frames(), NET_Y, people_detected=14)
    assert result.near_pid == 1


def test_near_pid_flips_when_the_positions_do():
    """Player 2 near, player 1 far this time - near_pid must follow the DATA, not
    default to a fixed id."""
    swapped = [{1: [10.0, 100.0, 40.0, FAR_FEET],
               2: [10.0, 100.0, 40.0, NEAR_FEET]} for _ in range(100)]
    result = assess_selection(swapped, NET_Y, people_detected=14)
    assert result.near_pid == 2


def test_near_pid_is_none_when_a_player_was_never_seen():
    """No feet position at all for player 2 - there is nothing to compare, so this must
    say so rather than guess."""
    only_p1 = [{1: [10.0, 100.0, 40.0, NEAR_FEET]} for _ in range(50)]
    result = assess_selection(only_p1, NET_Y, people_detected=14)
    assert result.near_pid is None


def test_near_pid_serialises_as_a_plain_int_not_a_numpy_scalar():
    """The same failure mode the rest of this file guards against for opposite_sides:
    a numpy scalar here would truncate the summary mid-write."""
    result = assess_selection(frames(), NET_Y, people_detected=14)
    payload = result.as_dict()
    assert payload["near_camera_pid"] == 1
    assert type(payload["near_camera_pid"]) is int
    json.dumps(payload)


def test_near_pid_is_reported_even_when_selection_otherwise_failed():
    """Positional data is still worth reporting on a failed selection - the caller
    already sees `status` and can decide whether to trust it, so this should not go
    silent just because the gate as a whole did."""
    same_side_but_measurable = [
        {1: [10.0, 100.0, 40.0, NEAR_FEET], 2: [50.0, 100.0, 80.0, NEAR_FEET - 5]}
        for _ in range(100)
    ]
    result = assess_selection(same_side_but_measurable, NET_Y, people_detected=40)
    assert result.status == SELECTION_FAILED
    assert result.near_pid in (1, 2)
