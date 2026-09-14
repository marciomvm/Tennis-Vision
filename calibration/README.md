# Court calibrations

One file per camera position: the fourteen court points placed by hand, for footage the
keypoint model cannot read. See `utils/court_calibration.py` for why this exists and
`tools/calibrate_court.py` for how to make one.

```bash
tennis-vision calibrate my_clip.mp4        # writes calibration/my_clip.json
tennis-vision analyze   my_clip.mp4        # found automatically, by video name
```

Discovery is by video file name. A camera that does not move produces the same court in
every recording, so point later clips at an existing file instead of redoing the clicks:

```bash
tennis-vision analyze another_clip.mp4 --court-calibration calibration/my_clip.json
```

The same file also carries the exclusion zones that keep a neighbouring court's match out
of the analysis.

## The format

Plain JSON, and editable by hand if a single point is out by a few pixels.

| field | meaning |
|---|---|
| `version` | schema version; a file from a newer build is refused, not guessed at |
| `keypoints` | the fourteen `[x, y]` points, in the order drawn in `utils/court_calibration.py` |
| `clicked` | which of them a person placed, as opposed to filled in from the fit |
| `frame_size` | the resolution they were placed on; rescaled automatically to another size of the same aspect ratio, and refused across a different one |
| `exclusions` | image polygons to ignore entirely, for both players and the ball |
| `line_support` | what the automatic gate measured at the time, recorded but not applied |
| `video`, `frame_index`, `created_at`, `notes` | provenance |

## Committing these

They are small, they are specific to one camera, and they are worth keeping next to the
footage they describe. A calibration for a clip nobody else has is harmless to commit and
saves redoing it on another machine.
