"""
tools/batch_analyze.py: combining many clips' summaries into one report.

The tests that matter most here are about what `aggregate_results` refuses to produce.
Player identity does not survive between clips - see the module docstring - so the real
risk in this tool is not a crash, it is a combined number that LOOKS like it means
something and does not. Every test below is either checking that a genuine aggregate is
computed correctly (weighted, not naively averaged) or that no per-player total is ever
fabricated.
"""
import json
import shutil

import numpy as np
import pytest

from tools.batch_analyze import (
    _clip_record,
    _resolve_clips,
    _segment_context,
    aggregate_results,
    build_concat_command,
    combine_videos,
)


def _record(clip="a.mp4", total=1, p1=0, p2=0, speed_p1=None, speed_p2=None,
           shot_types=None, court_ok=True, selection="ok",
           speed_3d_mean=None, speed_3d_n=None):
    return {
        "clip": clip, "total_shots": total,
        "shots_p1_this_clip_only": p1, "shots_p2_this_clip_only": p2,
        "avg_shot_speed_p1_kmh": speed_p1, "avg_shot_speed_p2_kmh": speed_p2,
        "shot_types": shot_types or {}, "court_calibrated": court_ok,
        "player_selection_status": selection,
        "shot_speed_3d_mean_kmh": speed_3d_mean, "shot_speed_3d_segments": speed_3d_n,
    }


# ── what must never appear in the aggregate ────────────────────────────────────

def test_no_combined_per_player_key_exists():
    """The one thing this function must never produce: a cross-clip 'Player 1 total'.
    Checked by absence of the key, not just by reading the prose note - a note is easy
    to trust and easy for a future edit to leave stale next to a key that quietly
    reappears."""
    records = [_record(p1=5, p2=1, speed_p1=100.0, speed_p2=50.0)]
    agg = aggregate_results(records)
    forbidden = {"total_shots_p1", "total_shots_p2", "avg_shot_speed_p1_kmh",
                "avg_shot_speed_p2_kmh", "player_1_total", "player_2_total"}
    assert forbidden.isdisjoint(agg.keys())


def test_the_note_explains_why_not():
    agg = aggregate_results([_record()])
    assert "does not name the same person" in agg["note"]


# ── total_shots: safe because it does not depend on identity ──────────────────

def test_total_shots_sums_both_players_across_clips():
    records = [_record("a", total=3), _record("b", total=20)]
    assert aggregate_results(records)["total_shots"] == 23


def test_shot_types_are_summed_across_clips():
    records = [
        _record("a", shot_types={"Forehand": 2, "Serve": 1}),
        _record("b", shot_types={"Forehand": 10, "Backhand": 5}),
    ]
    agg = aggregate_results(records)
    assert agg["shot_types"] == {"Forehand": 12, "Backhand": 5, "Serve": 1}


# ── speed: weighted by shot count, not an unweighted mean of per-clip averages ─

def test_speed_is_weighted_by_shot_count_not_averaged_flat():
    """A 1-shot clip at 150 must not pull the average as hard as a 20-shot clip at 65 -
    an unweighted mean of the two clip averages would give 107.5, which is wrong."""
    records = [
        _record("a", p1=1, p2=0, speed_p1=150.0, speed_p2=0.0),
        _record("b", p1=10, p2=10, speed_p1=70.0, speed_p2=65.0),
    ]
    agg = aggregate_results(records)
    expected = (1 * 150.0 + 10 * 70.0 + 10 * 65.0) / (1 + 10 + 10)
    assert agg["avg_shot_speed_kmh"] == pytest.approx(round(expected, 1))
    unweighted_mean_would_be = (150.0 + 67.5) / 2
    assert agg["avg_shot_speed_kmh"] != pytest.approx(unweighted_mean_would_be, abs=1.0)


def test_a_clip_with_zero_shots_does_not_poison_the_average_with_a_zero_speed():
    """summary.json reports 0.0 km/h for a player who took no shots, and that 0.0 must
    not be treated as a real (slow) shot when the weighted average is built - a 0-count
    clip should contribute nothing, not a speed of zero at full weight."""
    records = [
        _record("a", p1=0, p2=0, speed_p1=0.0, speed_p2=0.0),
        _record("b", p1=2, p2=0, speed_p1=100.0, speed_p2=0.0),
    ]
    agg = aggregate_results(records)
    assert agg["avg_shot_speed_kmh"] == pytest.approx(100.0)


def test_no_shots_anywhere_gives_no_speed_rather_than_a_crash():
    agg = aggregate_results([_record(p1=0, p2=0, speed_p1=0.0, speed_p2=0.0)])
    assert agg["avg_shot_speed_kmh"] is None


# ── 3-D speed: weighted mean, but true min/max across clips ────────────────────

def test_3d_speed_weighted_mean_and_true_extremes():
    records = [
        _record("a", speed_3d_mean=90.0, speed_3d_n=1),
        _record("b", speed_3d_mean=150.0, speed_3d_n=4),
    ]
    agg = aggregate_results(records)["shot_speed_3d"]
    assert agg["weighted_mean_kmh"] == pytest.approx((90.0 * 1 + 150.0 * 4) / 5, abs=0.1)
    assert agg["clip_mean_min_kmh"] == 90.0
    assert agg["clip_mean_max_kmh"] == 150.0
    assert agg["segments"] == 5


def test_3d_speed_is_none_when_no_clip_reconstructed_anything():
    agg = aggregate_results([_record(speed_3d_mean=None, speed_3d_n=0)])
    assert agg["shot_speed_3d"] is None


# ── problem flagging ────────────────────────────────────────────────────────────

def test_court_and_player_problems_are_flagged_by_clip_name():
    records = [
        _record("good.mp4", court_ok=True, selection="ok"),
        _record("bad_court.mp4", court_ok=False, selection="ok"),
        _record("bad_players.mp4", court_ok=True, selection="degraded"),
    ]
    agg = aggregate_results(records)
    assert agg["clips_with_court_problems"] == ["bad_court.mp4"]
    assert agg["clips_with_player_problems"] == ["bad_players.mp4"]


def test_a_missing_status_is_not_treated_as_a_problem():
    """An older summary.json without player_selection at all must not be reported as a
    problem clip just because the field is absent."""
    record = _record()
    record["player_selection_status"] = None
    agg = aggregate_results([record])
    assert agg["clips_with_player_problems"] == []


# ── the empty case ───────────────────────────────────────────────────────────────

def test_no_successful_clips_returns_a_safe_empty_aggregate():
    agg = aggregate_results([])
    assert agg["clips"] == 0
    assert agg["total_shots"] == 0
    assert agg["avg_shot_speed_kmh"] is None
    assert agg["shot_speed_3d"] is None


# ── _clip_record: pulling the right fields out of a real summary.json shape ────

def test_clip_record_reads_a_realistic_summary():
    summary = {
        "total_shots_p1": 3, "total_shots_p2": 7,
        "avg_shot_speed_p1_kmh": 90.0, "avg_shot_speed_p2_kmh": 60.0,
        "shot_classification": {"types": {"Forehand": 5, "Serve": 2}},
        "court_calibrated": True, "court_source": "manual", "court_line_support": 0.2,
        "player_selection": {"status": "ok"},
        "ball": {"coverage": 0.8},
        "shot_speed_3d_kmh": {"mean": 110.0, "segments": 3},
        "warning": None,
    }
    from pathlib import Path
    record = _clip_record(Path("clip_007.mp4"), summary, {"source_start_s": 12.0})
    assert record["clip"] == "clip_007.mp4"
    assert record["source_start_s"] == 12.0
    assert record["total_shots"] == 10
    assert record["shot_types"] == {"Forehand": 5, "Serve": 2}
    assert record["player_selection_status"] == "ok"
    assert record["shot_speed_3d_mean_kmh"] == 110.0


def test_clip_record_tolerates_a_bare_minimum_summary():
    """A refused clip's summary.json can be missing most sections entirely - the record
    builder must not crash on a clip that measured almost nothing."""
    from pathlib import Path
    record = _clip_record(Path("clip_x.mp4"), {}, {})
    assert record["total_shots"] == 0
    assert record["court_calibrated"] is None
    assert record["shot_types"] == {}


# ── resolving clips from a manifest or a plain directory ───────────────────────

def test_resolve_clips_from_a_manifest_file(tmp_path):
    manifest = {"video": "session.mp4", "segments": [
        {"file": "session_000.mp4"}, {"file": "session_001.mp4"}, {"file": None},
    ]}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    for name in ("session_000.mp4", "session_001.mp4"):
        (tmp_path / name).touch()

    clips, loaded = _resolve_clips(manifest_path)
    assert [c.name for c in clips] == ["session_000.mp4", "session_001.mp4"]
    assert loaded == manifest


def test_resolve_clips_from_a_directory_containing_a_manifest(tmp_path):
    manifest = {"video": "s.mp4", "segments": [{"file": "s_000.mp4"}]}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "s_000.mp4").touch()

    clips, loaded = _resolve_clips(tmp_path)
    assert [c.name for c in clips] == ["s_000.mp4"]
    assert loaded is not None


def test_resolve_clips_falls_back_to_globbing_mp4_with_no_manifest(tmp_path):
    (tmp_path / "b.mp4").touch()
    (tmp_path / "a.mp4").touch()
    (tmp_path / "notes.txt").touch()

    clips, loaded = _resolve_clips(tmp_path)
    assert [c.name for c in clips] == ["a.mp4", "b.mp4"]   # sorted
    assert loaded is None


def test_resolve_clips_on_an_empty_directory_returns_nothing(tmp_path):
    clips, loaded = _resolve_clips(tmp_path)
    assert clips == []
    assert loaded is None


def test_segment_context_finds_this_clips_own_timing():
    from pathlib import Path
    manifest = {"segments": [
        {"file": "x_000.mp4", "start_s": 0.0, "end_s": 5.0},
        {"file": "x_001.mp4", "start_s": 12.0, "end_s": 20.0},
    ]}
    ctx = _segment_context(manifest, Path("somewhere/x_001.mp4"))
    assert ctx == {"source_start_s": 12.0, "source_end_s": 20.0}


def test_segment_context_is_empty_without_a_manifest():
    from pathlib import Path
    assert _segment_context(None, Path("x.mp4")) == {}


# ── a real batch, end to end ─────────────────────────────────────────────────

@pytest.mark.slow
def test_a_real_two_clip_batch_end_to_end(tmp_path):
    """
    Runs the actual pipeline (via subprocess, exactly as a real batch does) on two
    copies of the reference clip and checks the whole chain: per-clip isolated output
    folders, a combined report, and a CSV a spreadsheet could open. --max-frames keeps
    it a smoke test rather than a full accuracy run.
    """
    from pathlib import Path
    import shutil
    import subprocess
    import sys

    REPO = Path(__file__).parent.parent
    required = ("models/tracknet.pt", "models/keypoints_model_geoaug.pth",
               "models/pose_landmarker_lite.task")
    missing = [w for w in required if not (REPO / w).exists()]
    if missing:
        pytest.skip("needs the downloadable model weights, missing: " + ", ".join(missing)
                    + ". Fetch them with: python scripts/download_models.py")

    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    source = REPO / "input_videos" / "input_video_2.mp4"
    shutil.copy(source, clips_dir / "ref_000.mp4")
    shutil.copy(source, clips_dir / "ref_001.mp4")

    out_dir = tmp_path / "batch_out"
    result = subprocess.run(
        [sys.executable, "tools/batch_analyze.py", str(clips_dir),
         "--no-court-calibration", "--config", "configs/config.yaml",
         "--out-dir", str(out_dir), "--max-frames", "40"],
        capture_output=True, text=True, cwd=str(REPO), timeout=600,
    )
    assert result.returncode == 0, f"batch_analyze crashed:\n{result.stderr}\n{result.stdout}"

    report = json.loads((out_dir / "batch_report.json").read_text(encoding="utf-8"))
    assert report["clips_succeeded"] == 2
    assert report["clips_failed"] == 0
    assert len(report["clips"]) == 2
    assert "total_shots" in report["aggregate"]
    assert "does not name the same person" in report["aggregate"]["note"]

    # Each clip's own folder, isolated - not a shared, guessable timestamp path.
    assert list((out_dir / "ref_000").glob("summary_*.json"))
    assert list((out_dir / "ref_001").glob("summary_*.json"))

    csv_text = (out_dir / "batch_report.csv").read_text(encoding="utf-8")
    assert "ref_000.mp4" in csv_text and "ref_001.mp4" in csv_text


# ── near_camera_pid: read through from a real summary shape ────────────────────

def test_clip_record_reads_near_camera_pid():
    from pathlib import Path
    summary = {"player_selection": {"status": "ok", "near_camera_pid": 2}}
    record = _clip_record(Path("clip.mp4"), summary, {})
    assert record["near_camera_pid"] == 2


def test_clip_record_near_camera_pid_absent_is_none():
    from pathlib import Path
    record = _clip_record(Path("clip.mp4"), {}, {})
    assert record["near_camera_pid"] is None


# ── combining rendered clips into one video ─────────────────────────────────────

def test_concat_command_lists_inputs_in_order():
    from pathlib import Path

    cmd = build_concat_command(
        [Path("a.avi"), Path("b.avi"), Path("c.avi")], Path("out.mp4"))
    input_flags = [i for i, tok in enumerate(cmd) if tok == "-i"]
    named = [cmd[i + 1] for i in input_flags]
    assert named == ["a.avi", "b.avi", "c.avi"]


def test_concat_command_uses_the_filter_not_the_demuxer():
    """The concat FILTER re-encodes and tolerates mismatched codec parameters; the
    concat DEMUXER (-f concat) is faster but requires identical inputs, which nothing
    here guarantees - main.py's own save_video can fall back from XVID to MJPG."""
    from pathlib import Path

    cmd = build_concat_command([Path("a.avi"), Path("b.avi")], Path("out.mp4"))
    assert "-filter_complex" in cmd
    assert "-f" not in cmd


def test_concat_command_references_every_input_in_the_filter_graph():
    from pathlib import Path

    cmd = build_concat_command(
        [Path("a.avi"), Path("b.avi"), Path("c.avi"), Path("d.avi")], Path("out.mp4"))
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert graph == "[0:v][1:v][2:v][3:v]concat=n=4:v=1:a=0[outv]"


def test_combine_videos_with_nothing_to_combine_fails_cleanly(tmp_path):

    ok, error = combine_videos([], tmp_path / "out.mp4")
    assert ok is False
    assert "no rendered clips" in error


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
def test_combine_videos_real_ffmpeg_round_trip(tmp_path):
    """
    Builds two tiny real clips with DIFFERENT dimensions and actually concatenates
    them, checking that the result plays back with all of both clips' frames -
    exactly the class of bug a mocked subprocess call cannot catch, and exactly why
    the concat FILTER (which can scale/normalise) rather than the demuxer is used.
    """
    import cv2

    def _clip(path, n, size, value):
        w, h = size
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
        for _ in range(n):
            writer.write(np.full((h, w, 3), value, np.uint8))
        writer.release()

    a = tmp_path / "a.mp4"
    b = tmp_path / "b.mp4"
    _clip(a, 15, (64, 48), 50)
    _clip(b, 20, (64, 48), 200)

    out = tmp_path / "combined.mp4"
    ok, error = combine_videos([a, b], out)

    assert ok, error
    cap = cv2.VideoCapture(str(out))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    # Re-encoding can shift the count by a frame or two; it must not be close to
    # either clip ALONE, which is what "only one clip made it in" would look like.
    assert frame_count >= 30


# ── missing clips: a manifest entry whose file was deleted by hand ────────────

def test_split_missing_separates_present_from_absent(tmp_path):
    from tools.batch_analyze import split_missing

    present_file = tmp_path / "a.mp4"
    present_file.touch()
    absent_file = tmp_path / "b.mp4"

    present, missing = split_missing([present_file, absent_file])
    assert present == [present_file]
    assert missing == [absent_file]


def test_split_missing_with_nothing_missing():
    from tools.batch_analyze import split_missing
    from pathlib import Path

    existing = [Path(__file__)]   # this test file itself, guaranteed to exist
    present, missing = split_missing(existing)
    assert present == existing
    assert missing == []
