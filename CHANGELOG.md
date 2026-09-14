# Changelog

All notable changes to Tennis-Vision are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) with one addition:
a **Measured and rejected** section per release. Approaches that were built, measured and
turned down are recorded with the number that turned them down, because a negative result
is expensive to produce and cheap to reuse.

Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added

- **Cut the dead time out of a long recording before analysing it.**
  `tennis-vision segment session.mp4` (`tools/segment_points.py`,
  `utils/activity_segments.py`) streams the video once, computes frame-differencing
  activity restricted to the calibrated court at reduced resolution, and keeps only the
  stretches with motion. No model, no GPU: about 150 frames/second on a single core, so
  an hour of 30fps footage costs roughly 12 minutes to scan against the ~100 minutes the
  ball tracker alone would need for the same hour.

  It is a coarse pre-filter, not a rally detector - a real rally is a burst of motion per
  stroke with the ball's flight time as a lull in between, so nearby bursts are merged
  with a generous gap tolerance rather than each becoming its own clip. Verified on the
  reference clip: default settings recovered 10 segments covering 68% of the footage,
  correctly separating a 16-second genuine pause from the rally either side of it.

  Each retained window is written as its own clip (stream-copied by default, `--reencode`
  for a frame-accurate cut) alongside a `manifest.json` recording every cut, the
  parameters used and the activity score at each boundary - a run that kept too little or
  too much is auditable rather than a black box. `--dry-run` computes the manifest without
  extracting anything; `--plot` saves a PNG of the activity trace against the threshold.

  26 tests added, all against synthetic signals rather than real video: the ordering of
  merge, duration-filter-before-padding, and the neighbour-overlap bug that ordering was
  written to prevent (two segments independently padding into the same gap and
  overlapping - fixed by capping both at the gap's midpoint instead of at each other's
  raw edge). 473 to 499.

### Added (2)

- **`tennis-vision segment --target-fps` resamples an out-of-range clip during the same
  cut.** `test.mp4` in the reference footage is a genuine constant 60fps source (verified
  with `ffprobe`: `r_frame_rate` and `avg_frame_rate` both exactly `60/1`), far outside
  the 23-31fps band every published accuracy number here was measured on. Bare
  `--target-fps` means 30; it implies `--reencode`, since a stream copy carries frames
  through unchanged and only a decode-and-re-encode pass can change their rate - forced
  on rather than refused, and reported when it happens rather than silently, matching
  `--fast`'s own precedent of one flag implying a config change in `main.py`.

  Every extracted clip's ACTUAL rate is verified afterwards with `ffprobe` against the
  file that was actually written, not assumed from ffmpeg exiting 0, and classified with
  the pipeline's own `utils.fps_support.assess_fps` - the existing gate, not a new
  threshold invented for this. Recorded per segment in `manifest.json` as `fps_actual`
  and `fps_support`.

  Verified against a real 12-second slice of `test.mp4`: the source was correctly
  classified unsupported at 60.0fps, the extracted clip landed at exactly 30/1 confirmed
  independently with a second, unrelated `ffprobe` call, and its frame count (229 over
  7.6s) matched 30fps exactly. 11 tests added, two of them a real ffmpeg round trip
  rather than a mocked one - a flag-ordering mistake in the `-r`/`-fps_mode` command
  would exit 0 and pass a mocked test while producing an unchanged 60fps file. 499 to
  510.

### Added (3)

- **`main.py --no-video` skips rendering the annotated video and the 3-D HTML viewer.**
  The CSV, summary JSON and 3-D scene JSON - already written before rendering starts -
  are unaffected. Rendering (bounding boxes, court wireframe, mini-court, per-frame
  overlays, then the video encode) is the slowest stage of the pipeline, and batch-
  processing many clips pays for it once per clip for videos nobody is going to watch.
  Off by default - every existing caller keeps rendering exactly as before. Forwarded
  through `tennis-vision analyze --no-video`. 2 tests added, one a real subprocess run
  confirming neither file is written while the summary still is.

- **`tennis-vision batch-analyze` runs the pipeline over every clip a segmentation
  manifest produced, and combines the results.** (`tools/batch_analyze.py`) One
  `--court-calibration` applies to every clip, since they share the camera position of
  the recording they were cut from - auto-discovery by clip file name will not find it
  on its own, because a clip is not named like the source recording. Defaults to
  `--no-video` (a batch run is about the numbers), with `--with-video` to opt back in,
  and gives each clip its own output folder rather than relying on the shared,
  second-resolution timestamp `main.py` otherwise writes stats under - avoiding a
  collision risk rather than treating it as merely unlikely.

  **Player identity does not survive between clips, and the report is built to never
  imply otherwise.** `utils.player_selection` numbers a clip's two people 1 and 2 by
  whichever has the lower internal tracker id in THAT run, not by court side - checked
  in this repository's own code before writing a line of the aggregator. Track ids
  restart fresh every pipeline run, so "Player 1" in one clip and "Player 1" in another
  have no relationship, and `aggregate_results` never produces a combined per-player
  figure as a result: it sums total shots (identity-independent) and shot-count-weights
  the average speed across clips (not an unweighted mean of per-clip averages, which
  would let a 1-shot clip outweigh a 20-shot one), and leaves the per-clip P1/P2
  breakdown in `batch_report.csv`, valid only within each row's own clip.

  Verified two ways: `aggregate_results` against synthetic records, checking a forbidden
  key set never appears rather than only reading the prose note next to it; and a real,
  unmocked run against 3 real clips from a genuine `asd_points/` batch a user produced
  independently while this was in progress, using the reference clip's calibration
  reused across a different recording from the same fixed camera - confirmed to be the
  same camera by overlaying that calibration on three widely-spaced frames of the second
  recording and finding the court lines still lined up. 21 tests added, one a real
  two-clip subprocess batch end to end. 510 to 533.

### Added (4)

- **A within-clip hint at identity, and a combined video.** A user running the batch
  tool against a real 40-minute session asked the sharpest possible question about the
  identity gap the previous release documented but did not touch: real tennis players
  change ends partway through a match, so even court side (near/far the camera) cannot
  name one physical person for a whole session, on top of the internal-tracker-id
  numbering already known not to.

  `player_selection.near_pid` (checked directly in `utils/player_selection.py` before
  building anything on it) is the SAME "is_bottom_half" distinction the selection logic
  already computes internally to tell players from spectators, just not previously
  written anywhere a consumer could read it. Exposed as `near_camera_pid` in
  `summary.json`'s `player_selection` section and in every row of
  `batch_analyze`'s `batch_report.csv`. It answers "which id was closer to the camera in
  THIS clip" and nothing more - it is not carried between clips, and a change of ends
  between two clips is exactly the case it cannot detect. The module docstring and the
  README are explicit that a real fix needs to look at what the two people look like
  (visual re-identification), which is not attempted here.

  `tennis-vision batch-analyze ... --combine-video` (implies `--with-video`) answers a
  second question the same user asked: whether there is any video output at all.
  Stitches every successfully rendered clip into one file, in the manifest's
  chronological order, via ffmpeg's concat FILTER rather than its demuxer - a
  re-encoding join that does not require the clips being joined to share identical
  codec parameters, unlike the faster stream-copy alternative. Watching the result is
  also the practical way to build a real identity map by eye: the render already boxes
  and labels "Player 1" / "Player 2" every frame, so which shirt is in which box, clip
  by clip, says more than any automatic labelling this pipeline currently attempts.

  Verified against real data: a 3-clip, `--max-frames 90` batch with `--combine-video`
  produced a video whose duration (9.000s) matched 3 x 90 frames at 30fps exactly, and a
  frame pulled from inside the third clip's span showed the real rendered overlay -
  player boxes, court wireframe, mini-court, shot panel - confirming the joined file is
  real analysed footage and not a placeholder. `near_camera_pid` read back as `1` for
  two temporally adjacent real clips and empty for a clip too short for real tracking,
  matching what the underlying player-selection status already said about that clip.

  12 tests added: 5 pinning `near_pid` in `utils/player_selection.py` (including that it
  is still reported on an otherwise-failed selection, and that it serialises as a plain
  int rather than a numpy scalar that would truncate the JSON write - the same failure
  mode `opposite_sides` was already guarded against), 7 for the video-combining path
  including a real ffmpeg round trip that concatenates two differently-sized synthetic
  clips and checks the frame count reflects both rather than only one making it in.
  533 to 545.

### Added (5)

- **A manifest entry whose clip was deleted no longer costs a full subprocess
  spin-up to discover.** Found on a real batch: a user deleted `asd_points/asd_000.mp4`
  (pre-match footage, not a rally) after `segment_points` had already written it into
  `manifest.json`. `split_missing` checks which listed clips still exist on disk once,
  before any of them run, rather than letting each absent one fail inside `main.py`
  after Python has started, arguments were parsed, and model loading had begun. 2 tests
  added. 545 to 547.

### Changed
### Changed

- **Hand-placed court geometry, for footage the keypoint model cannot read.**
  `tools/calibrate_court.py` places the fourteen court points once per camera position;
  `utils/court_calibration.py` stores them, refuses a set that does not describe a court,
  and the pipeline uses them in place of the model. Discovery is by video name
  (`calibration/<video name>.json`), with `--court-calibration FILE` to reuse one
  calibration across every clip from the same camera and `--no-court-calibration` to
  compare against the model it replaced.

  The court model is trained on broadcast tennis. On a phone or an action camera behind
  the baseline it returns a tidy quadrilateral that is not the court, the validity gate
  refuses the clip, and the user has nowhere to go. On a fixed camera the court does not
  move, so it is a property of the camera rather than of the frame.

  **Points are corrected by dragging them.** The first version placed points and had no
  way to move one: a mis-click could only be fixed by reselecting that point from the
  keyboard, which nobody guessed, so the tool read as all-or-nothing. Any point can now be
  grabbed and dragged at any time, including one the fit filled in - which is the common
  case, since the fit's guess is usually close and nudging it beats starting over.
  Grabbing does not move a point until the mouse does, a whole drag is one undo rather
  than one per mouse-move event, and undo restores a moved point rather than deleting it.
  The point under the cursor is highlighted and named, and a plan view of the court shows
  which point is being asked for, because "FAR service line x LEFT singles sideline" is
  harder to resolve on unfamiliar footage than a picture.

  **The canvas is larger than the video, because some court corners are not in it.** A
  wide camera close to the baseline routinely puts a near doubles corner outside its own
  frame - on the reference clip, 70px past the right edge. There was no pixel to click and
  no drag that could reach it, so that point could not be corrected at all. The video now
  sits inside a border with the court's continuation drawn around it, the view grows by
  itself when an edit puts a point outside it, and the arrow keys nudge the selected point
  a pixel at a time whether it is in the picture or not. The banner names any point that
  is outside and says that nothing out there can be checked against paint.

  **The window is fitted to the screen.** It was sized to a constant, so on a smaller
  display it opened wider than the desktop with the controls and the outermost points off
  the edge, and an `AUTOSIZE` window cannot be dragged back. `--max-size WxH` overrides
  the detected size.

- **The calibration also says which court is being analysed.** Club footage shows the next
  court along, and the players on it are real people the detector is right to find: on one
  3,600-frame clip it tracked fifteen people. Detections whose feet fall outside the court
  and its playing margin are dropped before player selection, and hand-drawn exclusion
  zones remove anything still in the way. The margin is in metres, not pixels, because
  perspective makes a fixed pixel margin far too tight near the camera and far too loose
  at the far end.

  Measured on 600 frames of amateur footage, everything else unchanged: court fit failed
  at 0.076 line support and now passes by placement; people reaching player selection 15
  to 2; player 1 coverage 19% with a 2,439-frame hole to 93%; players on opposite sides of
  the net no to yes; 3-D reconstruction refused to 13 segments of which 5 are shots.

### Changed

- **The line-support gate does not apply to a calibrated run**, and `summary.json` says so
  through a new `court_source` field. The gate exists because a regression head cannot
  report being out of distribution; a person who placed the points on the lines and
  checked the overlay has answered that with better evidence. The gate samples straight
  segments between corners, which a wide lens bends, so it scores a correct court low. The
  measurement is still taken and still published as `court_line_support`.

- **`people_detected` in `summary.json` is now counted before court filtering.** Reporting
  the post-filter number would make a clip crowded with a neighbouring court's match look
  like an empty one.

### Fixed

- `test_no_mkdir_forgets_its_parents` skipped a virtualenv called `venv` but not one
  called `.venv`, so it failed on every checkout using the dotted name - scanning
  site-packages and reporting third-party code as offenders.

- **`--help` crashed on nine of this project's own commands.** The house style rules a
  docstring off with box characters, and those scripts pass the docstring to argparse as
  its description; argparse writes help to a console that is cp1252 on a default Windows
  install, so `--help` raised `UnicodeEncodeError` before printing a line. Among them were
  four of the eval scripts the README tells people to run. A test now asserts that any
  docstring used as argparse help is ASCII.

- **A subcommand's own `--help` was unreachable.** `tennis-vision --help` ends with "Run
  'tennis-vision <command> --help' for command-specific options", and that did not work:
  the top-level parser's `-h` matched first, so `tennis-vision analyze --help` printed the
  top-level command list instead of analyze's flags. Dispatch now happens before argparse
  sees the arguments.

### Known limitation

A homography assumes straight lines and a wide action camera bends them. The fourteen
points are unaffected, being placed where the corners really appear, but anything mapped
through the homography carries the error. It is measured and published as
`court_calibration.lens_error_px` rather than assumed away: on the reference amateur clip
a single homography sits about 10px rms from the painted lines. Lens correction is not
implemented.

### Tests

61 added: the conventions the calibration shares with the mini-court and the validity
gate, the ordering mistakes it must refuse, the metric margin, which detections survive
the court region, what a drag moves and what undo restores, the view that reaches a point
outside the video, the arrow-key nudge and how it collapses into one undo, that argparse
help text is printable, and that a subcommand's help is reachable. 412 to 473.

---

## [2.1.1] - 2026-09-07

Two defects that broke the documented first-run path for every new user, and were
invisible to everyone who already had the project working. Found by cloning the published
tag and following the README Quickstart verbatim, which nobody had done.

### Fixed

- **The documented TrackNet fetch command did not work.** `gdown` 5 removed the `--id`
  flag, and the dependency floor was `>=4.7.1`, so a new user installed 6.x and the
  command the tool itself prints failed with `unrecognized arguments: --id`. It is the
  only manual step in the install, and it was a dead end. Corrected to the positional
  form in all four places it appears, and the floor is pinned to `gdown>=5` so the
  printed instruction and the installed tool cannot disagree again.

- **The pipeline crashed after finishing the analysis, without writing anything.**
  `save_stats` called `out.mkdir(exist_ok=True)`, which does not create parent
  directories. `output/` is gitignored, so it exists on every developer machine and on no
  user's. The documented command ran the full pipeline for six minutes, completed the
  analysis, and then died with `FileNotFoundError` on `output/stats`. Every `mkdir` in the
  tree now creates parents.

### Verified

The published Quickstart, run end to end from a clean clone of the tag with weights
downloaded fresh: 5 m 56 s, all five outputs written, and every figure identical to the
development machine (15 shots as 8 and 7, court line support 0.638, ball coverage 82%,
13 shot segments spanning 61.5 to 153.3 km/h with a mean of 91.6, serve speed correctly
refused). Output JSON is strictly valid with no NaN.

### Tests

3 added: the documented `gdown` command against the pinned floor, output directories
created from nothing, and a tree-wide check that no `mkdir` forgets its parents.
409 to 412.

---

## [2.1.0] - 2026-09-01

21 commits. A hardening release, not a feature release. Every change here either removes a
way the pipeline could report something it had not measured, or makes an existing
measurement checkable by someone else.

The headline is not a new capability. It is that the system was run against footage it had
never been tuned on, refused all fourteen clips, and was right to.

### Added

**Gates. Each one closes a route to a confident wrong number.**

- **Frame-rate support gate** (`utils/fps_support.py`). Every threshold in the event path
  is counted in frames and the classifier's two largest weights are velocities in pixels
  per frame, none of it normalised by rate, and the entire measured record sits between
  23.6 and 30 fps. Resampling the reference clip's ball track and re-running the real
  generators gives the bands: 15 fps loses 43% of events per second, 60 fps finds 64% more
  and swings the contact/bounce split from 14:14 to 11:35. Supported 23-31, partially
  supported 18-50, unsupported outside, reported in the log, `summary.json` and a banner on
  the rendered video. The clip still runs; the caveat travels with it.

- **Player-selection quality gate** (`utils/player_selection.assess_selection`). The court
  gate stops a clip whose COURT was fitted to the crowd. Nothing stopped a clip whose
  PLAYERS were. One evaluation clip passes the court gate at 0.327 line support and then
  selects two tracks on the same side of the net, one present for 40% of frames with a
  199-frame hole, and the run published confident per-player statistics. Singles is played
  across the net, so that is decidable with no ground truth. Reports `ok`, `degraded`
  (right players, lost for a stretch, so counts under-count) or `failed` (wrong people, so
  nothing per-player is a measurement).

- **3-D speed validity** (`utils/trajectory_3d.classify_segment_speed`). Only a segment
  that begins at a racket contact is a ball leaving a racket; one beginning at a bounce is
  the post-bounce leg travelling to the receiver. Segments are labelled `valid`,
  `plausible_but_uncertain`, `outlier` or `not_a_shot`, and a speed is published only for
  the first two.

- **Court-fit failure detail** (`assess_court_fit_detail`). A clip containing a good court
  segment and a clip that never shows a court both failed as "court fit failed". Only one
  of those is something a user can fix, so the failure now names which it is and how many
  frames were usable.

**Evidence in the output**

- `summary.json` gains `frame_rate_support`, `player_selection`, `court_detail`, `ball`,
  `rally_decoding` and `shot_classification` blocks, each carrying a plain-language reason.
- `trajectory3d_*.json` gains `speed_status`, `speed_status_reason` and `duration_s`.
- Rally-decoder overrides are surfaced. The decoder's own docstring warns that repeatedly
  overruling a confident classifier signals an upstream problem; on the reference clip it
  overrules at 90% and 94%, and that warning previously reached nobody.

**Measurement**

- `eval/speed_timing_sensitivity.py`, `eval/physics_evidence_rate.py`,
  `eval/player_selection_sanity.py`, `eval/heldout_benchmark.py`,
  `eval/heldout_segments.py`.
- **Held-out benchmark.** Fourteen clips cut from a region of source video that never
  influenced any threshold, model choice or feature selection, frozen before any clip was
  viewed, selected by an arithmetic rule stated in `datasets/heldout/manifest.json`.

**Infrastructure**

- `.github/workflows/ci.yml`: tests on Python 3.10 and 3.12, CPU torch, editable install,
  plus a narrow lint. The suite had only ever passed on one machine.
- `docs/public/TECHNICAL_OVERVIEW.md`, the pipeline stage by stage with the assumption and
  failure mode of each.
- `scripts/build_demo_pack.py`, demo assets assembled from a real run, with key frames
  chosen from the run's own event list.
- Measured runtime published: 5 m 51 s fresh for 19 s of 720p on a GTX 1050 Ti, 18.5x real
  time. No real-time claim is made.

### Fixed

**Numbers that were wrong**

- **The final contact of every clip was dropped.** `ShotClassifier.classify_shots` and
  `main.py`'s statistics loop both iterated `range(len - 1)`, so the published shot count
  was always exactly one below the number of contacts detected and drawn. Reference clip:
  14 published against 15 detected, now 15 and 15.
- **An unmappable position became the centre of the court**, and flowed into player
  distance and speed as though it had been observed. Now omitted and counted.
- **An unmapped opponent contributed a 0.0 km/h movement sample**, biasing that average
  downward exactly where tracking was worst. Now no sample is recorded.
- **A failed homography silently substituted the nearest-keypoint approximation**, the
  method this project describes as the old and wrong one. Now counted and reported.
- **Requesting the SAM 3D pose backend without its weights fell through to MediaPipe in
  silence.** The two measure 85.5% and 66.4% balanced on identical clips.

**Failures that could not be seen**

- **A truncated run poisoned the detection cache.** `--max-frames` cached its detections
  under the full clip's name, and the end-to-end smoke test runs that way, so running
  `pytest tests/` was enough to leave a 40-frame cache at a 570-frame clip's path.
  `tools/label_shots.py` had no length guard and would have seeded ground-truth labelling
  from the wrong input, silently.
- **The 3-D viewer emitted an absolute video path on non-Windows hosts.** `Path().name`
  resolves separators for the host OS only, so a Windows path on Linux came through whole
  and the video failed to load. Found by CI on its first run.
- **The stats panel printed the literal string "nan"** where a player had not hit yet.
- A `numpy.bool_` in the player-selection payload truncated `summary.json` mid-write.
- `get_center_of_bbox` was defined twice in `utils/bbox_utils.py`.
- The pre-commit hook ran `ruff --select=F --fix`, which on the re-export module
  `utils/__init__.py` would have deleted the re-exports and broken the package.
- `pytest.ini` was gitignored, so a fresh clone had no marker registration. Configuration
  moved into `pyproject.toml` with `--strict-markers`.
- Thirty comments in tracked files cited paths under the private `docs/` tree, so a reader
  following them found nothing.

### Changed

- **`shot_speed_3d_kmh` counts only contact-initiated segments.** Reference clip: was 25
  segments spanning 18-170 km/h with a mean of 80.4, now 13 shots spanning 62-153 km/h with
  a mean of 91.6 and 12 segments excluded and counted.
- **The physics shot layer is described as a validation filter, not a classifier.** Across
  9 clips and 81 shots it evidences 1 and rejects 20. Its rejection logic is unchanged:
  rarely firing positively is correct when volleys and smashes are rare.
- **`trajectory_3d.py`'s stated limits are reordered by measured size.** Event timing
  dominates at 12.5% mean and 28.0% worst, against 0.7% for contact height and 0.1% for
  ball localization. The list previously named the small terms and omitted the large one.

### Measured and rejected

- **Relabelling-only rally decoding.** Repairing an impossible ordering only by flipping a
  label measured worse than not decoding at all: false positives 7 to 10, no recall gain.
  Every repair pushes an event into the other class, which is only correct when each
  candidate is a real event, and on this clip 41% are not.
- **A deletion prior above 0.02.** Chosen as 0.15 at first, from a flat region on the
  reference clip. The 40-clip recall curve showed that clip saturates early: recall falls
  monotonically with the prior, so 0.15 gave up 2.6 points of recall for no additional
  coherence.
- **A minimum-speed threshold on 3-D segments.** The obvious fix for an 18 km/h "shot
  speed", and the wrong one. Three explanations were tested against the data first:
  endpoints outside the baseline (refuted, 18 of 25 segments have one, because a contact
  endpoint is the player's feet), never crossing the net (true, and not a defect), and
  aggregating legs that are not shots (the real cause).

### Known limitation, newly measured

- **A clip containing a camera cut is refused whole.** There is no shot-boundary detection
  and the pipeline assumes one continuous take. The held-out benchmark refused 14 of 14
  arbitrary broadcast windows for this reason. Trimmed to the rally inside them, 5 of 5 had
  a valid court and valid player selection. Automatic segmentation is deliberately not in
  this release.

### Benchmark summary

| | |
|---|---|
| Held-out, arbitrary broadcast windows | 14 clips, **0 numbers reported, 14 refused** |
| Held-out, the rally inside them | 5 analysable, court **5/5**, players **5/5**, ball median **62%** |
| Output integrity across all 14 | 0 crashes, 14/14 valid JSON, 0 fabricated positions, 0 silent fallbacks |
| Reference clip | 7/7 shot recall, 8 false positives, F1 0.636, unchanged |
| Tests | 244 to **383** |

---

## [2.0.0] - 2026-08-18

147 commits. Not a feature release: V1 already produced numbers, and this release is about
whether those numbers were true. Several were not, and the corrections are below.

Every figure names the script that produced it. Where something is unmeasured, it says so.

### Added

**Measurement infrastructure**

- `eval/ball_localization_accuracy.py`, ball position against hand-labelled ground truth,
  reporting detection rate and localization error as separate quantities. Sweeps the
  heatmap threshold and cluster size, and compares postprocessing modes.
- `eval/event_detection_on_real_detections.py`, contact and bounce detection running real
  TrackNet inference end to end, rather than on the dataset's labelled ball coordinates.
- `eval/event_recall_funnel.py`, attributes every missed contact to the pipeline stage
  that lost it, so the next piece of work is chosen by size rather than by guess. Also
  sweeps the candidate merge window under both linkage rules.
- `eval/serve_false_positive_check.py`, does the pipeline claim a serve on clips cut from
  mid-rally, which is most footage a user brings.
- `eval/pose_availability_at_contacts.py`, how often pose is usable at the moment a shot
  is struck, at the exact frame and within a small window.
- `eval/forehand_backhand_on_thetis.py`, the geometric rule against ground truth, with an
  oracle mode separating a wrong hand choice from a projection that cannot express the
  answer.
- `eval/extract_thetis_pose_features.py` and `eval/train_forehand_backhand.py`, named pose
  features and a classifier trained with subject-grouped splits, repeated cross-validation,
  balanced accuracy and feature ablations.
- `eval/validate_on_broadcast_images.py`, transfer test from indoor training data to real
  broadcast footage.
- `eval/swing_candidate_recall.py` and `eval/sam3d_occluded_arm_test.py`.

**Pipeline capability**

- 3-D free-flight reconstruction between detected contacts, and an interactive viewer
  written as a single self-contained HTML file with no external dependencies.
- Court validity gate based on line support: samples along every predicted court line and
  asks whether those pixels are brighter than the surface beside them.
- Physical serve detection: ball above the player's head **and** hitter at or behind a
  baseline, with a minimum separation between serves in seconds rather than frames.
- Physics-based volley, smash and lob classification from the event sequence.
- Trained hit versus bounce classifier on four named trajectory features.
- Optional SAM 3D Body pose backend, off by default, interface-compatible with the
  MediaPipe estimator.
- `scripts/download_thetis.py` and `scripts/download_sam3d_body.py`.
- `tools/LABELLING_GUIDE.md`, procedure for producing shot-type ground truth.

**Packaging**

- `pip install -e .` with a `tennis-vision` CLI, layered configuration, and regression tests
  asserting the built-in defaults match the shipped config exactly.

### Changed

- **Ball position is now the centroid of the largest connected heatmap response**, not the
  mean of every responding pixel. Median localization error 5.8px to 5.4px, 90th percentile
  20.2px to 18.0px. The tail improves more than the median, which is the signature of
  removing blended two-response frames rather than of general smoothing.
- **Candidate clustering is bounded, not transitive.** Recall 51.3% to 68.4% at unchanged
  precision.
- **Detection caches are keyed per video** and off by default.
- **Pose is requested at the contact frame and then its neighbours**, since the contact
  frame is the worst moment to ask: the player is fully extended, often side-on, occluded
  by their own racket arm and motion-blurred. Pose reaches 77% of contacts at the exact
  frame and 91% within four frames either side.
- **Shots labelled "Groundstroke" are now eligible for the pose upgrade.** That label means
  "a ground stroke, side unknown", which is exactly what pose resolves, and excluding it
  withheld 4 of 13 shots on the reference clip. Shots receiving a pose-based label went
  from 54% to 85%.
- **The 3-D viewer draws players**, fits the canvas instead of using a fifth of it, and
  separates arcs by weight so the one under examination is legible among seventeen.
- Evals now grade the pipeline's shipped output rather than an earlier stage.

### Fixed

- **Detection caches were keyed to nothing.** A single shared file with no record of which
  video wrote it, so analysing one clip and then another silently gave the second the
  first's ball positions. The file on disk held 40 frames from a smoke run and was being
  used to grade a 570-frame video.
- **Candidate clusters grew without bound.** Frames 0, 10, 20, 30 and 40 collapsed into one
  event at frame 20. The merge window had been set to match "the tolerance used throughout
  this sprint's eval scripts", which is a category error: an eval tolerance answers how
  close a detection must be to count as a match, a merge window answers how close two
  generator firings can be and still describe one physical event.
- **`pip install` produced a silently degraded pipeline.** Wheel users got the superseded
  court model, YOLO instead of TrackNet, no pose shots, missing classifier weights, caching
  enabled against documented behaviour, and two undeclared runtime dependencies. Everything
  ran and nothing errored.
- **Classifier weights were resolved against the working directory**, so they loaded only
  when the process happened to start from the repository root. Elsewhere the classifier
  returned None for every event and the pipeline fell back to a weaker heuristic silently.
- **`PlayerTracker` crashed on a cache miss** instead of detecting fresh.
- **Two serves were reported 18 frames apart**, which is 0.6s at 30fps and physically
  impossible.
- **A features/weights mismatch raised a bare `KeyError`** from deep inside classification.
  It now names the drift, and never defaults a missing feature to zero, which would return
  confident probabilities from an input the model never saw.
- **`save_video` was hardcoded to 24fps**, so every 30fps clip was written 25% slow and its
  clock drifted against the 3-D viewer's timeline.
- **The 3-D viewer's first scale computation threw on a null trig basis** at page init,
  leaving the canvas blank with no console error because the listener attaches later.

### Corrected numbers

Figures previously published that described something other than what a reader would
assume.

| Claim | Previously | Actually |
|---|---|---|
| Ball detection | "82.5% detection rate", read as accuracy | 88.6% of frames get a position; **42.5%** of visible-ball frames are within 5px |
| Event recall | "87.6%, production config" | 87.6% **given perfect ball positions**; **72.0%** running real detection |
| Shot-frame accuracy | "mean offset 4.9 frames, EXCELLENT" | 7/7 recall, mean offset 7.4 frames, precision measured separately |
| Court validity clips | 8/9 | unchanged, but an eval reported 5/9 until it was fixed to use per-frame keypoints |

### Measured and rejected

Each was implemented, measured and turned down. Full detail in the README.

| Approach | Result |
|---|---|
| Homography reprojection error as a court-validity signal | 1.40-1.88px on correct fits, 2.13px on a wrong one, 14/14 inliers every time |
| Heatmap threshold and cluster-size sweeps | every setting within noise of the shipped one |
| A larger or newer YOLO for player detection | detection saturated: 11-14 people found per frame where 2 are needed |
| RTS forward-backward smoothing | complete coverage and 1.5 points of recall for 4% worse median error |
| A chi-square outlier gate | median error 6.3px to 29.4px, and to 207.9px at tight tuning |
| A body-based swing generator | 58.5% of ordinary frames score at or above the median contact |
| The energy-ratio bounce signal | 65.7% on 1,034 labelled events |
| A trained forehand/backhand classifier | 76.3% indoors, **53.6% on broadcast** |
| Replacing MediaPipe with SAM 3D Body | **19 points worse** on identical clips |

The last one is worth reading in full. SAM 3D Body recovers the occluded racket arm that
MediaPipe drops on 44-58% of backhands, taking usable clips from 171 to 200 and class
balance from 55/45 to a perfect 100/100, and it made the classifier substantially worse.
The damage is confined to position and leaves motion untouched: the wrist-side feature
falls from 78.6% to chance. On occluded frames MediaPipe declines and SAM 3D infers the arm
from a body prior, which is anatomically plausible and still a guess about where the racket
is. MediaPipe's refusal was a quality filter, not only a loss.

### Known limitations

- Forehand versus backhand is unreliable: 54% on balanced ground truth. Three replacements
  were measured and none is better.
- Rally and groundstroke speeds are unvalidated. Serve speed is radar-validated at a 0.96
  mean ratio; rally speed has no ground truth.
- Roughly a quarter of contacts in a rally are missed, at 95.9% precision, so the shot
  count is an under-count rather than noise.
- Volley and smash have never been checked against labels.
- Ball height is modelled from contact anchors and gravity, not measured, and cannot be
  otherwise from broadcast camera geometry.
- Ground-level cameras fail; the validity gate flags them rather than reporting wrong
  numbers.
- Doubles and amateur footage are untested.

### Tests

83 to 213.

---

## [1.0.0] - 2025-08-25

First working version, preserved at the `v1.0.0` tag and the `v1-stable` branch.

### Added

- YOLOv8 player and ball detection with ByteTrack
- ResNet-50 court keypoint regression, 14 points
- Mini-court coordinate mapping and bird's-eye visualisation
- Rule-based shot classification
- Player movement and shot speed statistics
- Annotated video output
- Training scripts, contributing guidelines, MIT licence

### Superseded by 2.0.0

V1's reported figures were not wrong so much as differently defined from how they read.
Its ball detection rate was taken as an accuracy, and its event recall described a pipeline
with perfect ball positions rather than the real one. 2.0.0 publishes both numbers in each
case.

---

[2.0.0]: https://github.com/HarshTomar1234/Tennis-Vision/releases/tag/v2.0.0
[1.0.0]: https://github.com/HarshTomar1234/Tennis-Vision/releases/tag/v1.0.0
