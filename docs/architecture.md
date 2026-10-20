# Architecture

Each stage is a plain function or small class with data in, data out. Only
`detection/` and `court/keypoints.py` need a GPU model; everything after them is
geometry and logic, covered by `tests/` without any weights
(`Pipeline.analyze` is the model-free half).

## 1. Detection and tracking (`detection/tracker.py`, `detection/tracking.py`)

One YOLO pass per frame finds every class; that pass is cached on its own
(`RawDetections`). The ball and hoop are not tracked: all ball candidates are
kept for the ball stage, and the most confident hoop box is kept per frame.
An optional second, ball-only model can supply ball candidates instead.

Players and referees are then tracked together by one ByteTrack
(`track_buffer` frames of memory, so a player screened in the paint keeps his
ID), with two changes for broadcast video:

* **Camera compensation.** Every box is mapped into the first frame of its
  shot through the accumulated camera motion (`court/camera_motion.py`, which
  now runs before tracking), so the Kalman prediction only has to follow the
  player's own motion, not the pan. Frames whose motion can't be estimated
  are treated as a still camera.
* **Cuts.** At a histogram cut the tracker restarts with fresh IDs, so an ID
  is never handed to whoever stands at the same screen spot in the next shot.
  Identities are joined across the cut in court space (below).

Each track's class (player or referee) is a confidence-weighted vote over its
life, so a player the detector briefly calls a referee keeps his track.
`tracking.camera_compensation` and `tracking.reset_on_cut` switch these off.

## 2. Court keypoints, homography, camera (`court/`)

A YOLO pose model predicts 18 landmarks. With 4+ confident, non-collinear
landmarks, `cv2.findHomography` (RANSAC) maps image pixels to court metres.
A fit is rejected if its median reprojection error is above 1 m.

Broadcast cameras pan and zoom constantly and keypoints are often missing, so
the mapping is tracked rather than held (`court/camera_motion.py`,
`track_homographies`). Background features (players, referees and learned
screen overlays masked out) are tracked with optical flow, and RANSAC gives
the frame-to-frame homography; a failed fit or a histogram jump marks a cut.
Each frame's mapping is predicted from the previous one through that motion.
A keypoint fit that agrees with the prediction is blended in
(`homography_smoothing` is the weight on the prediction); one that disagrees
by more than `reject_threshold_m` is dropped as a mislabel unless several fits
in a row agree with each other. Frames between fits are filled forward and
backward through the motion, weighted by distance, for up to
`max_flow_frames`. With `court.camera_motion: false` the old behaviour (hold
the last fit for `homography_hold_frames`) is used.

`camera_from_homography` recovers the focal length from the two orthogonality
constraints of a plane-induced homography (principal point at the image
centre, square pixels), then rotation and translation. Focal length is median
filtered over time. This turns the 2D floor mapping into a 3x4 camera, which
the 3D stage needs.

Limits: a shot that never shows 4 landmarks (a close-up, a replay) has no
court mapping, since there is nothing to carry it from. Adding more landmark types (rim, backboard corners,
restricted-area arc points) is the main lever for coverage.

## Track re-linking, identities and smoothing (`detection/stitch.py`, `tactical/projector.py`)

After court projection, a player track that ends is joined to one that starts
within `relink_max_gap_s` on the same team, never on screen together, and
reachable at `relink_max_speed_mps`.

Then `assign_identities` uses the fact that a team has five players on the
floor: tracks, in order of their first frame (as a stream would deliver them),
take one of at most `max_per_team` identities per team. A track joins a free
identity (its last track has ended) that it can reach from where that player
was last seen (`identity_max_speed_mps`, within `identity_max_gap_s`), or opens
a new one while the team has fewer than five. Link cost is distance over what
a player covers at a jog; opening a new identity costs the same as a jog-speed
link, so far-off tracks start new identities until all five are known. Tracks
that start together (after a cut) are matched jointly (Hungarian). A sixth
simultaneous track on a team is an "extra" (bench, coach, wrong team) and
keeps its own ID. Each identity gets a 0-1 confidence, the weakest margin in
its chain, which the touches table reports as `id_confidence`.

Court position can't tell two teammates apart when both were out of the
picture for a while; jersey numbers are the missing cue (see Roadmap).

Court positions are then median-filtered
(spikes from cut-off boxes), short gaps up to `max_fill_s` interpolated, and
smoothed over `smooth_window_s`. Everything downstream (3D, speed, touches)
uses the smoothed positions.

## Diagnostics (`diagnostics.py`)

Each run writes `diagnostics.json`, `diagnostics_frames.csv` and
`diagnostics.png`: court mapping coverage and source, keypoints seen and fit
error, rejected fits, cuts; track counts, short tracks, re-links, identities
per team, weak identities, feet mapped
off the court, frame-to-frame jumps; team colour separation; ball detection
and possession rates. Findings name the weakest stage and its fix.

## 3. Ball track (`ball/interpolation.py`)

Per frame, the chosen candidate maximises confidence × closeness to the last
accepted position, so a bright head or a light does not hijack the track.
Detections implying more than `max_speed_px_per_frame` are dropped. Gaps up
to `max_gap_frames` are linearly interpolated; longer gaps stay empty.

## 4. Teams (`teams/team_assigner.py`)

For each track, up to `samples_per_track` torso crops are taken across the
clip. A 2-cluster KMeans inside each crop separates jersey from background
(the cluster owning the crop corners is background). Each track's median
jersey colour (LAB) is then clustered into two teams. One decision per track
means no frame-to-frame flicker. When one team's jersey matches the floor,
switch to `method: clip` with two prompts.

## 5. Possession (`possession/acquisition.py`)

A player is a candidate holder when the ball is close to a "hand point" of
their box, measured as a fraction of box height so it works at any zoom.
Containment halves the distance. A change of holder needs `min_frames`
consecutive frames; losing the ball also needs `min_frames`, which bridges
dribbles.

## 6. Events (`events/`)

Consecutive possession segments with different holders become a pass (same
team), interception (other team, short gap), turnover (other team, long gap),
or rebound (a shot attempt happened in between). Shot attempts are the free
ball entering the zone above the rim; a make is the ball crossing the rim line
downward inside the rim. The shooter is the last holder before the attempt.
Shot location, distance and 2PT/3PT come from the shooter's court position at
release, using the real 3-point arc and corner geometry.

## 7. Court positions, speed, distance (`tactical/`, `metrics/`)

Players are projected by their foot point (bottom-centre of the box).
Positions are median-filtered before differencing; steps faster than
`max_speed_mps` are treated as noise.

## 8. 3D (`recon3d/`)

* Held ball: height is where the camera ray through the ball pixel passes
  closest to the holder's vertical axis.
* Free ball: each run of free-ball frames is fitted as a parabola under
  gravity (initial position and velocity, 6 parameters) by minimising
  reprojection error over all its frames. Runs that do not fit one parabola
  (bounce passes, deflections) are split and refit; implausible fits are
  discarded.
* Players stand on the floor (z = 0) at their projected position.

## 9. Player bodies (`pose/`, `recon3d/viewer.py`)

Optional (`pose.enabled`). A YOLO pose model finds COCO-17 body keypoints on
each frame; each detected person is matched to a player track by box overlap,
so skeletons inherit track IDs and teams. Lifting to 3D uses the recovered
camera: every joint lies on the ray through its pixel, so only its depth along
the ray is unknown. Depths are solved per player per frame by least squares
against bone-length priors (scaled to `player_height_m`), a reach limit around
the tracked court position, the floor, and the previous frame's depths, then
smoothed. On a synthetic broadcast camera joints land within about 5 cm.
Known limits: occluded or blurred limbs drop out (the viewer synthesizes
them); a limb pointing straight at the camera has an ambiguous depth; lifting
costs about 10 ms per player-frame, so it suits clips more than full games.
A learned lifter (MotionBERT) or SMPL body fitting can replace `lift_pose`
with the same inputs and outputs.

`scene.json` holds the per-frame scene, with `pose` (17 joints in court
metres) on players that have a skeleton. `play_3d.html` is a three.js page
with articulated bodies posed from that file: speed blends idle, walk and run
with a stride phase tied to distance covered; the defending team faces the
ball in a stance; shot and pass events drive the jump, release, contest, push
and catch; measured skeletons override the synthesized limbs. A rigged
Mixamo-style `.glb` (`output.character_glb`) replaces the mannequin and is
retargeted from the same skeleton. `output.viewer: plotly` keeps the older
dot view.

## Roadmap

* Jersey numbers: a small digit recogniser on torso crops of each track
  (back and front, sampled where the box is big and sharp), voted per track,
  then used as a hard constraint in `assign_identities` (same number + team =
  same identity, different number = never) and joined to the scraped roster
  for names. Needs labelled number crops from our own broadcast clips (the
  frame-mining notebook can collect them).
* Appearance re-ID embeddings (e.g. OSNet) as a softer cue between teammates
  when numbers aren't readable.
* Replay detection, so replays don't add duplicate touches.
* More court landmarks for close-up coverage.
* Learned 3D lifting or SMPL fitting for player bodies; hand and head detail.
* Shot outcome from hoop/net motion as a second signal.
* Full-game batching: split by camera cuts, run segments in parallel.
