# CourtVision

Basketball broadcast video in; court x, y for every on-court player and the ball, 10 times a
second, plus touch-by-touch events out.

The goal is to take videos of professional basketball games (NBA, NCAA) and use YOLO detection and
tracking to return player-tracking data and the box score and advanced stats built on it, from any video.

```
broadcast video ──► detect + track ──► teams ──► possession ──► passes / interceptions / shots
                         │                            │
                         └─► court keypoints ─► homography (tracked through pans)
                                                      │
                                                      ├─► players: court x, y (m), speed, distance
                                                      └─► ball: held / flight / floor ─► tracking_2d at 10 Hz
```

The 3D reconstruction (camera, ball height, player bodies) is still in the package, off by
default: `output.mode: 3d`.

Starting point: [abdullahtarek/basketball_analysis](https://github.com/abdullahtarek/basketball_analysis).
What CourtVision does beyond it:

| Area | Reference repo | CourtVision |
|---|---|---|
| Court model | 28 × 15 m approximation | True NBA (28.65 × 15.24 m) or FIBA spec, 3-point arc and corners, rim height |
| Homography | Per frame, no checks | RANSAC, reprojection-error gate, hold through dropouts, smoothing with cut detection |
| Camera | Floor plane only | Full camera (focal, pose) recovered per frame, so points off the floor can be projected |
| Ball | Highest-confidence box, linear fill | Continuity-aware selection, outlier rejection, gap-limited interpolation |
| Ball on the court | Projects the ball box (wrong whenever it's in the air) | Holder's spot when held, exact straight ground track of a pass or shot, floor projection only when loose |
| Output clock | Every video frame | Uniform 10 Hz `tracking_2d` (configurable), models still run on every frame |
| Training data | One Roboflow set, random split | Several sources (Roboflow, SportsMOT, own broadcast frames), split by video, scored per dataset and leave-one-dataset-out |
| Teams | Zero-shot CLIP per frame | Per-track decision (no flicker): jersey-colour clustering by default, CLIP optional |
| Possession | Per-frame nearest player | Hand-point distance scaled by player size, with hysteresis |
| Events | Passes, interceptions | + turnovers, shot attempts, makes, 2PT/3PT, shot distance, offensive/defensive rebounds |
| Data out | Annotated video | `tracking_2d.parquet`/`.csv`, `touches.parquet`/`.csv`, `events.csv`, `box_score.csv`, video, diagnostics |
| Memory | Whole video in RAM | Frames streamed; full games fit in Colab |
| Tests | None | Unit tests plus a synthetic end-to-end clip with a known camera |

## Quick start (Colab)

1. `notebooks/01_train_detector.ipynb` trains the player / ball / hoop detector on several datasets
   at once and scores it on each (per-frame player recall, ball precision / recall, MOTA / IDF1),
   with leave-one-dataset-out cross-validation.
2. `notebooks/02_train_court_keypoints.ipynb` trains the court keypoint model.
3. `notebooks/00_end_to_end_colab.ipynb` runs everything on a video and shows the 2D tracking, a top-down replay, the touches table and the annotated video.
4. `notebooks/03_collect_frames_to_label.ipynb` picks the frames of your own clips the models get wrong and
   exports them with pre-labels for Roboflow. Correct them, then retrain with 01 and 02, which merge them in.

All four save to `MyDrive/courtvision/`, so weights and runs survive Colab disconnects.
Put your Roboflow key in Colab **Secrets** as `ROBOFLOW_API_KEY` (and a Hugging Face read token as `HF_TOKEN` for SportsMOT).

## Quick start (local)

```bash
pip install -e ".[models,dev]"
# put weights in models/: player_detector.pt, court_keypoints.pt (optional ball_detector.pt)
courtvision run input_videos/game.mp4 --config configs/default.yaml
pytest
```

Outputs land in `output/<video name>/`.

## Outputs

| File | Contents |
|---|---|
| `tracking_2d.parquet`, `tracking_2d.csv` | **The main table.** Every on-court player and the ball, 10 samples per second: time, id, team, `x_m`, `y_m`, speed, who has the ball, and for the ball how its position was found (`held`, `flight`, `floor`) |
| `events.csv` | One row per event: `pass`, `interception`, `turnover`, `shot`, `made_shot`, `offensive_rebound`, `defensive_rebound`, with time, player, team, target, and for shots the location, distance and points |
| `touches.parquet`, `touches.csv` | One row per touch: player, how the touch started (catch, steal, rebound), the action (pass, shot, turnover) with outcome and target, dribbles, the handler's position, speed and nearest defender, every other player's id, position and speed at that moment, and a confidence score |
| `tracking.csv` | 3d mode: every player and the ball, every frame, ball with height |
| `box_score.csv` | Per player: touches, time of possession, passes, interceptions, turnovers, FGA/FGM, 3PA/3PM |
| `scene.json`, `play_3d.html` | 3d mode: per-frame 3D scene and animated three.js viewer |
| `<video>_annotated.mp4` | Broadcast overlay: IDs, team colours, ball, holder, possession %, passes, minimap |

## Layout

```
courtvision/
  config.py                 typed config, YAML overrides (configs/default.yaml)
  types.py                  shared data structures
  io/                       streaming video I/O, stage caching
  detection/tracker.py      YOLO + ByteTrack: players, refs, ball candidates, hoop
  teams/team_assigner.py    jersey-colour (default) or CLIP team assignment
  ball/interpolation.py     ball selection, outliers, interpolation
  possession/acquisition.py per-frame holder with hysteresis, possession segments
  events/passes.py          passes, interceptions, turnovers, rebounds
  events/shots.py           attempts, makes, shooter
  events/timeline.py        touch / event / tracking tables, box score
  court/geometry.py         NBA/FIBA court spec and 18 keypoints
  court/keypoints.py        YOLO pose court keypoint detector
  court/homography.py       robust homography, tracking through pans, camera recovery
  court/camera_motion.py    frame-to-frame camera motion (optical flow), cut detection
  detection/stitch.py       re-link player tracks split by the tracker
  diagnostics.py            per-stage health report (which stage is weak)
  tactical/                 image-to-court projection, top-down court rendering
  metrics/speed_distance.py smoothed speed and distance
  pose/                     2D body keypoints per track, lifted to 3D skeletons with the camera
  recon3d/                  3D ball from physics, scene export (JSON), three.js body viewer, Plotly
  viz/annotate.py           video overlays
  training/                 multi-dataset build with grouped splits and cross-dataset folds (sources.py),
                            pipeline-style scores and MOTA / IDF1 (evaluate.py), SportsMOT and COCO
                            converters, zoomed court crops, hard-frame mining
  tracking2d.py             court x, y for players and the ball on a fixed clock (main output)
  pipeline.py               orchestration
  cli.py                    `courtvision run ...`
notebooks/                  Colab: end-to-end, detector training, keypoint training, frame collection
tests/                      unit tests and synthetic end-to-end test
docs/architecture.md        how each stage works and its known limits
```

## Models

| Weights | Task | Classes / keypoints | Notebook |
|---|---|---|---|
| `player_detector.pt` | detect | Ball, Clock, Hoop, Overlay, Player, Ref, Scoreboard | 01 |
| `ball_detector.pt` (optional) | detect | Ball | 01 |
| `court_keypoints.pt` | pose | 18 court landmarks (see `court/geometry.py`) | 02 |

Class names are mapped in `configs/default.yaml` (`detection.class_names`), so a dataset with
different labels only needs a config change.
