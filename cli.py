"""
cli.py
──────
Console entry point for the `tennis-vision` command.

Subcommands are thin wrappers over the existing modules rather than reimplementations,
so there is exactly one code path per capability and the CLI cannot drift from what
`python main.py` does.

    tennis-vision analyze clip.mp4 -o output/run.avi
    tennis-vision calibrate clip.mp4
    tennis-vision segment session.mp4
    tennis-vision batch-analyze session_points/ --court-calibration calibration/c.json
    tennis-vision download-models
    tennis-vision version
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

__version__ = "2.1.1"


def _cmd_analyze(argv: list[str]) -> int:
    """Run the full pipeline on one video."""
    parser = argparse.ArgumentParser(
        prog="tennis-vision analyze",
        description="Analyse a tennis video: ball, players, court, shots and stats.",
    )
    parser.add_argument("input", help="path to the input video")
    parser.add_argument("-o", "--output", default=None,
                        help="annotated output video path (default: from config)")
    parser.add_argument("-c", "--config", default="configs/config.yaml",
                        help="config YAML (use configs/dev.yaml to enable caching)")
    parser.add_argument("--no-stubs", action="store_true",
                        help="force fresh detection, ignoring any cached stubs")
    parser.add_argument("--max-frames", type=int, default=0, metavar="N",
                        help="process only the first N frames (0 = all) - quick check "
                             "on a long video before a full run")
    parser.add_argument("--fast", action="store_true",
                        help="single-frame court keypoints; faster, less camera-robust")
    parser.add_argument("--debug", action="store_true", help="verbose logging")
    parser.add_argument("--no-video", action="store_true",
                        help="skip the annotated video and 3-D viewer; CSV and summary "
                             "JSON are written either way. Rendering is the slowest "
                             "stage - skip it when batch-processing many clips and only "
                             "the numbers are wanted")
    parser.add_argument("--court-calibration", metavar="FILE", default=None,
                        help="hand-placed court geometry (tennis-vision calibrate). "
                             "Found automatically at calibration/<video name>.json, so "
                             "pass this only to reuse one calibration across clips from "
                             "the same camera position")
    parser.add_argument("--no-court-calibration", action="store_true",
                        help="ignore any calibration file and use the keypoint model")
    args = parser.parse_args(argv)

    if not Path(args.input).exists():
        print(f"error: input video not found: {args.input}", file=sys.stderr)
        return 2

    # main.main() reads sys.argv, so hand it the flags it expects rather than
    # duplicating the pipeline here.
    forwarded = ["main.py", "--input", args.input, "--config", args.config]
    if args.output:
        forwarded += ["--output", args.output]
    if args.max_frames:
        forwarded += ["--max-frames", str(args.max_frames)]
    if args.no_stubs:
        forwarded.append("--no-stubs")
    if args.fast:
        forwarded.append("--fast")
    if args.debug:
        forwarded.append("--debug")
    if args.no_video:
        forwarded.append("--no-video")
    if args.court_calibration:
        forwarded += ["--court-calibration", args.court_calibration]
    if args.no_court_calibration:
        forwarded.append("--no-court-calibration")

    import main as pipeline

    original_argv = sys.argv
    try:
        sys.argv = forwarded
        pipeline.main()
    finally:
        sys.argv = original_argv
    return 0


def _cmd_calibrate(argv: list[str]) -> int:
    """Place the court by hand for one camera position."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from tools import calibrate_court

    original_argv = sys.argv
    try:
        sys.argv = ["calibrate_court.py", *argv]
        return calibrate_court.main()
    finally:
        sys.argv = original_argv


def _cmd_segment(argv: list[str]) -> int:
    """Cut the dead time out of a long recording before analysing it."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from tools import segment_points

    original_argv = sys.argv
    try:
        sys.argv = ["segment_points.py", *argv]
        return segment_points.main()
    finally:
        sys.argv = original_argv


def _cmd_batch_analyze(argv: list[str]) -> int:
    """Run the pipeline over every clip a segmentation manifest produced."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from tools import batch_analyze

    original_argv = sys.argv
    try:
        sys.argv = ["batch_analyze.py", *argv]
        return batch_analyze.main()
    finally:
        sys.argv = original_argv


def _cmd_download_models(argv: list[str]) -> int:
    """Fetch model weights into models/."""
    argparse.ArgumentParser(
        prog="tennis-vision download-models",
        description="Download the model weights the pipeline needs.",
    ).parse_args(argv)

    sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
    import download_models

    return download_models.main()


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="tennis-vision",
        description="Measured, reproducible tennis video analysis from a single camera.",
        epilog="Run 'tennis-vision <command> --help' for command-specific options.",
    )
    parser.add_argument("command", nargs="?", default="help",
                        choices=["analyze", "calibrate", "segment", "batch-analyze",
                                 "download-models", "version", "help"],
                        help="what to do")

    # Dispatch off sys.argv BEFORE argparse sees it, so that a -h after a subcommand
    # reaches that subcommand's own parser. The epilog just above promises exactly this
    # and it did not work: argparse's top-level -h matched first, so
    # `tennis-vision analyze --help` printed this help and none of analyze's flags -
    # including the ones that are the only way to discover a feature exists.
    subcommands = {
        "analyze": _cmd_analyze,
        "calibrate": _cmd_calibrate,
        "segment": _cmd_segment,
        "batch-analyze": _cmd_batch_analyze,
        "download-models": _cmd_download_models,
    }
    argv = sys.argv[1:]
    if argv and argv[0] in subcommands:
        return subcommands[argv[0]](argv[1:])

    args, rest = parser.parse_known_args()
    if args.command in subcommands:
        return subcommands[args.command](rest)
    if args.command == "version":
        print(f"tennis-vision {__version__}")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
