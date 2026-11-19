# Architecture

The goal is one table: the court x, y of every on-court player and the ball, ten times a second,
from NBA broadcast video. Each stage is a function with tables in and tables out, and every
intermediate table is kept on the pipeline's `Result`, so any stage can be inspected in a
notebook. Only `detect.py` needs a GPU; `pipeline.analyze` (everything after the model passes) is
what the tests run, on synthetic clips with a known camera.

```
run(video)
  detect.detect_objects      boxes: frame, role (player, referee, ball, hoop), x1 y1 x2 y2, conf   cached
  detect.detect_keypoints    (frames, 18, 3) court landmarks                                         cached
  camera.estimate_motion     frame-to-frame camera motion, candidate cuts                            cached
analyze(...)
  camera.track_homographies  image -> court mapping for every frame
  camera.confirm_cuts        keep cuts where the mapping really jumps
  track.track_people         track ids for players and referees
  teams.assign_teams         team per track, by jersey colour
  positions.player_positions court positions per player per frame
  ball.ball_table            ball image track, holder, court position per frame
  output.tracking_table      the 10 Hz table
  diagnostics                one number per stage, plus a per-frame table
```

## Detection (`detect.py`)

One YOLO pass per frame at `imgsz` 1280 (broadcast balls are 15 to 30 px). Each class name maps
to a role through `detection.class_names`, so a model with other names only needs a config
change. The ball gets a lower confidence bar than people. Results are cached per video
(`video.cached`), keyed by the model and the settings, so re-running the pipeline after a code
change skips the GPU.

## Court mapping through camera motion (`court.py`, `camera.py`)

`court.py` holds the 18 landmarks at true NBA (or FIBA) positions in metres, in the order of the
public court-keypoint dataset. With 4 or more confident, non-collinear landmarks,
`fit_homography` fits the image -> court mapping with RANSAC and rejects fits whose median error
is over 1 m.

Keypoints are only visible now and then (a zoomed half-court view shows a few lines, a pan shows
none for a second), but a broadcast camera pans and zooms about a fixed point, so consecutive
frames are related by one homography. `estimate_motion` measures it from optical flow on the
background (player boxes and screen graphics masked out) and marks candidate cuts where the
brightness histogram jumps. `track_homographies` then:

1. drops mirror-image fits: a model that swaps the two halves of the court can still fit
   perfectly, but the mapping is the wrong way round (`court.orientation`); every real view of a
   game has the same orientation, so the one most fits share is the real one;
2. carries each frame's mapping forward through the motion; a keypoint fit that agrees with the
   carried mapping is blended in (`court.smoothing`), one that disagrees by more than
   `court.reject_m` metres is rejected, unless several fits in a row agree with each other
   (then the carried mapping had drifted and is reset);
3. fills backwards from each fit the same way and blends both directions by distance, for up to
   `court.max_flow_frames`.

`confirm_cuts` keeps only the cuts where the mapping jumps or is lost (camera flashes and
graphics swing the histogram without the camera moving). A shot that never shows 4 landmarks (a
close-up, a replay) gets no mapping, so no positions.

## Tracking (`track.py`)

ByteTrack predicts each track with a constant-velocity filter and matches by box overlap. In
screen pixels a pan makes everyone appear to sprint at the camera's speed and tracks break, so
each box is first mapped into the first frame of its shot through the accumulated camera motion:
the tracker only sees the players' own motion. At a cut the tracker restarts with fresh ids, so
an id is never handed to whoever stands on the same screen spot in the next shot. Players and
referees share one tracker; each track's role is a confidence-weighted vote over its life.

## Teams (`teams.py`)

For each track, torso crops sampled across its life; in each crop a 2-cluster KMeans separates
jersey from background (the cluster owning the crop corners); each track's median jersey colour
(LAB) is clustered into two teams. One decision per track, so no flicker.

## Player positions (`positions.py`)

1. `project`: the foot point (bottom centre of the box) through that frame's mapping; points
   off the court (bench, coaches) are dropped.
2. `relink`: a track that ends is joined to one that starts within `relink_max_gap_s`, on the
   same team, never on screen together, and reachable at `relink_max_speed_mps`. This mends
   breaks from occlusion, missed detections and cuts.
3. `smooth`: per player a 5-frame median (foot-point spikes when a box is cut off), gaps up to
   `max_fill_s` interpolated, a moving average over `smooth_window_s`, speed from the smoothed
   path.

## Ball (`ball.py`)

* Image track: per frame, the candidate that best continues the track (confidence weighted by
  distance from the last position); impossible jumps dropped; short gaps interpolated.
* Holder: the ball near one of a player's hand points, in box heights so it works at any zoom,
  and moving with them (a pass flying past a player in the picture is not a catch).
* Court position: a floor mapping is only right for points on the floor, and the ball rarely
  is. Held: the holder's position (through dribbles too). In flight: between release and catch
  the ball's horizontal velocity is constant, so the point under it moves in a straight line at
  constant speed, and interpolating between the two holders is its exact ground track; a shot
  goes release, rim, rebounder. Loose with nothing to aim at: the bottom of the ball box,
  projected.

## The 10 Hz table (`output.py`)

The per-frame positions are linearly resampled onto a uniform clock (`output.hz`, 10: one
sample per 0.1 s). A sample is left out when the observations around it are more than 0.2 s
apart, so gaps are never papered over. `coverage` reports players per sample, the share of
samples with all ten players, and the ball's share by how it was placed.

## Diagnostics (`diagnostics.py`)

`diagnostics.json` and `frames.csv`: court mapping share and source, keypoints per frame,
rejected and mirror-image fits, cuts; players detected vs placed per frame, tracks, relinks,
short tracks; ball detected and placed; dataset coverage. Read top to bottom: a stage can only be
as good as the ones before it.

## Training data (`training/`)

| Module | |
|---|---|
| `sources.py` | builds one dataset from several (`build`), with leakage checks, leave-one-dataset-out (`lodo_folds`) and grouped k-fold (`kfold_folds`) |
| `sportsmot.py` | basketball-only SportsMOT zip from the 23 GB download; frames to YOLO, named by game |
| `labels.py` | YOLO rows; missing classes filled from a model; geometry-checked court labels |
| `crops.py` | zoomed half-court crops of full-court keypoint labels |
| `evaluate.py` | per-frame detector scores, MOTA and IDF1, court mapping coverage and PCK, new vs current |
| `mining.py` | which frames of your own clips are worth labelling |
| `look.py` | label grids, to check a dataset by eye |

Leakage rules in `build`:

* every source's own train/valid/test split is ignored;
* images are tied when they share a video (file name without frame number and Roboflow's
  export suffix), a SportsMOT game, or look the same (64-bit difference hash within 6 bits),
  across datasets too; tied images form a component and every component is in one split;
* splits are drawn per source by component (a component bigger than twice a split's share
  trains, so one video can't be most of a test split);
* `role="test"` sources (clips kept back) pull every component they touch into test;
* `leakage_report` must show zeros, and also reports how many images were tied across datasets.

Scoring rules in the notebooks: each source is scored only on the classes it labels itself
(SportsMOT's referee, ball and hoop boxes, and the court pseudo-labels, came from the current
models); court frames held back for scoring are chosen before pseudo-labelling; the new model
replaces the current one only if it is at least as good on test images not tied to the
dataset the current model was trained on.
