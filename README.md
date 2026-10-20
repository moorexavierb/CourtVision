# CourtVision

Basketball game video in; tracking data, touch-by-touch events and 3D play reconstructions out.

The goal is to take videos of professional basketball games (NBA, NCAA) and use YOLO detection and tracking to return box score stats and advanced statistical analysis from any video.

```
broadcast video ──► detect + track ──► teams ──► possession ──► passes / interceptions / shots
                         │                                              │
                         └─► court keypoints ─► homography ─► camera ──►├─► court positions (m), speed, distance
                                                                        ├─► 3D ball (physics) ──────────────┐
                                                                        └─► body keypoints ─► 3D skeletons ─┴─► 3D scene, animated players
```

Starting point: [abdullahtarek/basketball_analysis](https://github.com/abdullahtarek/basketball_analysis).
What CourtVision does beyond it:

| Area | Reference repo | CourtVision |
|---|---|---|
| Court model | 28 × 15 m approximation | True NBA (28.65 × 15.24 m) or FIBA spec, 3-point arc and corners, rim height |
| Homography | Per frame, no checks | RANSAC, reprojection-error gate, hold through dropouts, smoothing with cut detection |
| Camera | Floor plane only | Full camera (focal, pose) recovered per frame, so points off the floor can be projected |
| Ball | Highest-confidence box, linear fill | Continuity-aware selection, outlier rejection, gap-limited interpolation |
| Ball height | None | Physics fit: parabolic flight (gravity) fitted to every frame of a pass or shot |
| Players in 3D | None | Animated bodies (gait from speed, stances, shots, passes), 3D skeletons lifted from video, rigged characters |
| Teams | Zero-shot CLIP per frame | Per-track decision (no flicker): jersey-colour clustering by default, CLIP optional |
| Possession | Per-frame nearest player | Hand-point distance scaled by player size, with hysteresis |
| Events | Passes, interceptions | + turnovers, shot attempts, makes, 2PT/3PT, shot distance, offensive/defensive rebounds |
| Data out | Annotated video | `touches.parquet`/`.csv` (main table), `events.csv`, `tracking.csv`, `box_score.csv`, `scene.json`, interactive 3D HTML, video |
| Memory | Whole video in RAM | Frames streamed; full games fit in Colab |
| Tests | None | Unit tests plus a synthetic end-to-end clip with a known camera |

## Quick start (Colab)

1. `notebooks/01_train_detector.ipynb` trains the player / ball / hoop detector.
2. `notebooks/02_train_court_keypoints.ipynb` trains the court keypoint model.
3. `notebooks/00_end_to_end_colab.ipynb` runs everything on a video and shows the tables, 3D play and annotated video.

All three save to `MyDrive/courtvision/`, so weights and runs survive Colab disconnects.
Put your Roboflow key in Colab **Secrets** as `ROBOFLOW_API_KEY`.

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
| `events.csv` | One row per event: `pass`, `interception`, `turnover`, `shot`, `made_shot`, `offensive_rebound`, `defensive_rebound`, with time, player, team, target, and for shots the location, distance and points |
| `touches.parquet`, `touches.csv` | **The main table.** One row per touch: player, how the touch started (catch, steal, rebound), the action (pass, shot, turnover) with outcome and target, dribbles, the handler's position, speed and nearest defender, every other player's id, position and speed at that moment, a confidence score, and `id_confidence` (how sure the player id is one person) |
| `tracking.csv` | Every player and the ball, every frame, in court metres (ball includes height), with speed and running distance |
| `box_score.csv` | Per player: touches, time of possession, passes, interceptions, turnovers, FGA/FGM, 3PA/3PM |
| `scene.json` | Per-frame 3D scene for any front-end (three.js, Unity, Blender) |
| `play_3d.html` | Animated 3D reconstruction with player bodies (three.js); open it in any browser |
| `<video>_annotated.mp4` | Broadcast overlay: IDs, team colours, ball, holder, possession %, passes, minimap |

## Layout

```
courtvision/
  config.py                 typed config, YAML overrides (configs/default.yaml)
  types.py                  shared data structures
  io/                       streaming video I/O, stage caching
  detection/tracker.py      YOLO: players, refs, ball candidates, hoop
  detection/tracking.py     ByteTrack in camera-stabilised coordinates, reset at cuts
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
  detection/stitch.py       re-link split tracks; at most 5 identities per team
  diagnostics.py            per-stage health report (which stage is weak)
  tactical/                 image-to-court projection, top-down court rendering
  metrics/speed_distance.py smoothed speed and distance
  pose/                     2D body keypoints per track, lifted to 3D skeletons with the camera
  recon3d/                  3D ball from physics, scene export (JSON), three.js body viewer, Plotly
  viz/annotate.py           video overlays
  pipeline.py               orchestration
  cli.py                    `courtvision run ...`
notebooks/                  Colab: end-to-end, detector training, keypoint training
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
