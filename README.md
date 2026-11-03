# CourtVision

NBA broadcast video in; a **10 Hz player tracking dataset** out: the court x, y (metres) of every
on-court player and the ball, ten times a second.

```
broadcast video ─► detect players, referees, ball, hoop (every frame)
                ─► court keypoints ─► image-to-court mapping, carried through pans by camera motion
                ─► track ids (camera-compensated, restarted at cuts) ─► teams by jersey colour
                ─► player court positions (relinked, smoothed) ─► ball court position
                ─► resample to 10 Hz ─► tracking.parquet
```

| Column | |
|---|---|
| `sample`, `time_s`, `frame` | sample number, time on the 10 Hz clock, nearest video frame |
| `object`, `id`, `team` | `player` (player id, team) or `ball` (id -1) |
| `x_m`, `y_m`, `speed_mps` | court position (x along the sideline 0 to 28.65 m, y across 0 to 15.24 m), speed |
| `has_ball` | this player holds the ball |
| `source` | players: `tracked`; ball: `held` (at the holder), `flight` (ground line of a pass or shot), `floor` (loose) |

Samples where an object was not seen for more than 0.2 s are left out, not guessed. Every run also
writes per-frame tables, `diagnostics.json` (which stage is weak on this clip) and an annotated
video with a top-down minimap.

Starting point: [abdullahtarek/basketball_analysis](https://github.com/abdullahtarek/basketball_analysis).
What CourtVision does beyond it:

| Area | Reference repo | CourtVision |
|---|---|---|
| Court model | 28 × 15 m approximation | true NBA (or FIBA) landmark positions |
| Court mapping | per frame, unchecked | RANSAC with an error gate, carried through pans and zooms by camera motion, bad fits rejected |
| Tracking | ByteTrack on raw boxes | boxes warped into a camera-stable frame, tracker restarted at cuts, broken tracks relinked on the court |
| Ball on the court | projects the ball box (wrong whenever it is in the air) | holder's spot when held, straight ground line of a pass or shot, floor projection only when loose |
| Output | annotated video | uniform 10 Hz table plus diagnostics |
| Training data | one Roboflow set, random split | several datasets, split by video with near-duplicates tied across datasets, scored per dataset, leave-one-dataset-out cross-validation |

## Models and training data

Two YOLO models, trained in Colab (`notebooks/`), weights in `MyDrive/courtvision/models/`:

| Model | Notebook | Datasets |
|---|---|---|
| `player_detector.pt`: Player, Ref, Ball, Hoop | 01 | your Roboflow set (v17), Basketball Game Detection, Basketball Broadcast, SportsMOT basketball, your own labelled frames |
| `court_keypoints.pt`: 18 court landmarks | 02 | the public court-keypoint set (reloc2), plus geometry-checked labels on Basketball Game Detection and Basketball Broadcast frames, your own labelled frames |

How leakage between training and scoring is kept out (`courtvision/training/sources.py`):

* each dataset's own random split is ignored; images are tied by video (file names), by SportsMOT
  game, and by look (an image hash, across datasets), and tied images always share a split;
* clips you keep back for testing pull everything tied to them into test;
* each dataset is scored only on the classes it labels itself, and labels made by a model are
  never used to measure accuracy;
* frames held back for scoring are chosen before any model labels the rest;
* a new model replaces the current one only if it is at least as good on test images the current
  model cannot have seen.

## Quick start (Colab)

1. `notebooks/01_train_detector.ipynb`: detector, cross-validated across datasets.
2. `notebooks/02_train_court_keypoints.ipynb`: court keypoint model.
3. `notebooks/00_end_to_end_colab.ipynb`: run on a clip; shows diagnostics, the 10 Hz table, a top-down replay and the annotated video.
4. `notebooks/03_collect_frames_to_label.ipynb`: picks the frames of your clips the models get wrong and exports them with pre-labels for Roboflow; corrected, they feed back into 01 and 02.

Each notebook has a `BRANCH` setting in its clone cell for testing a pull request. Store the
Roboflow key in Colab **Secrets** as `ROBOFLOW_API_KEY` (and a Hugging Face token as `HF_TOKEN`
for SportsMOT); never paste keys into a cell.

## Quick start (local)

```bash
pip install -e ".[models,dev]"
# weights in models/: player_detector.pt, court_keypoints.pt
courtvision input_videos/game.mp4 --config configs/default.yaml     # -> output/game/tracking.parquet
pytest
```

## Layout

```
courtvision/
  config.py       all settings (configs/default.yaml overrides them)
  court.py        court landmarks in metres, homography fitting
  video.py        frame reading, per-video cache of model passes
  detect.py       the GPU passes: boxes and court keypoints
  camera.py       camera motion, cuts, a court mapping for every frame
  track.py        camera-compensated tracking of players and referees
  teams.py        team by jersey colour
  positions.py    court positions: project, relink, smooth, speed
  ball.py         ball: image track, holder, court position
  output.py       the 10 Hz table and its coverage
  diagnostics.py  per-stage numbers and a per-frame table
  render.py       annotated video with minimap
  pipeline.py     run (model passes, cached) and analyze (everything after, no GPU)
  training/       dataset build and cross-validation, labels, crops, scoring, frame mining
tests/            synthetic broadcast camera and clips with known answers
```

More detail: [docs/architecture.md](docs/architecture.md).
