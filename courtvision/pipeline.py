"""End-to-end orchestration: video in, 2D court tracking + events out (3D optional).

Stages (each cached to ``output.cache_dir`` so re-runs only redo what changed):

    1. detect + track          players, referees, ball candidates, hoop
    2. court keypoints         -> per-frame homography, carried through pans by camera motion
                                  (optical flow) -> per-frame camera
    3. ball cleaning           select, de-outlier, interpolate
    4. team assignment         one team per track
    5. possession              per-frame holder with hysteresis
    6. events                  passes, interceptions, turnovers, shots, rebounds
    7. court projection        metres (tracks re-linked, positions smoothed), speed and distance
    8. 2D ball                 the ball's floor position from held / flight / floor (tracking2d)
    9. 3D (output.mode: 3d)    ball height from physics, player skeletons (optional), scene export
   10. outputs                 tracking_2d (x, y for every player and the ball at output.tracking_fps),
                               touches, events, box score, annotated video, diagnostics;
                               in 3d mode also tracking.csv with ball height, scene.json, 3D HTML

Frames are streamed, never all held in memory, so a full game works on Colab.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .ball.interpolation import clean_ball_track
from .config import Config
from .court.geometry import get_court
from .court.homography import estimate_homographies, fit_homography_info, track_homographies
from .detection.stitch import apply_remap, stitch_tracks
from .diagnostics import diagnose, format_report, plot_diagnostics
from .events.passes import detect_passes
from .events.shots import detect_shots
from .events.timeline import annotate_shots, box_score, events_table, touches_table, tracking_table
from .io.cache import cache_key, cached
from .io.video import iter_frames, read_frames_at, video_info
from .possession.acquisition import (NO_POSSESSION, detect_possession, frame_holder, possession_segments,
                                     team_control)
from .recon3d.ball3d import cameras_for_frames, reconstruct_ball_3d
from .pose.lift import lift_poses
from .recon3d.scene import build_scene, save_scene_json, scene_to_plotly
from .recon3d.viewer import scene_to_threejs
from .metrics.speed_distance import speed_and_distance
from .tactical.projector import project_tracks, smooth_court_positions
from .teams.team_assigner import TeamAssigner
from .tracking2d import ball_court_track, coverage, tracking_2d_table
from .types import PipelineResult

log = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or Config()
        self.court = get_court(self.cfg.court.standard)
        self.team_colors_bgr: list[tuple[int, int, int]] = TeamAssigner.team_bgr(None)
        # track id -> {"jersey": ..., "name": ...}; fills those columns of the touches table when set
        self.player_info: dict[int, dict] = {}

    # ------------------------------------------------------------------ stages
    def detect(self, video: Path, fps: float, total: int):
        from .detection.tracker import ObjectTracker
        c = self.cfg
        tracker = ObjectTracker(c.models.detector, c.detection, c.models.device, c.models.ball_detector, fps)
        return tracker.run(iter_frames(video), total=total)

    def court_keypoints(self, video: Path, total: int):
        from .court.keypoints import CourtKeypointDetector
        det = CourtKeypointDetector(self.cfg.models.court_keypoints, self.cfg.models.device)
        return det.run(iter_frames(video), total=total)

    def camera_motion(self, video: Path, det, total: int):
        from .court.camera_motion import CameraMotionEstimator
        boxes = [[p["bbox"] for p in (*pl.values(), *rf.values())] for pl, rf in zip(det.players, det.referees)]
        return CameraMotionEstimator().run(iter_frames(video), boxes, total=total)[0]

    def homographies(self, video: Path, det, total: int):
        """Per-frame image->court homographies and calibration info (both empty-ish without a model)."""
        c = self.cfg
        n = len(det.players)
        if not (c.models.court_keypoints and Path(c.models.court_keypoints).exists()):
            log.warning("No court keypoint model; skipping court-space outputs (tactical view, speed, 3D).")
            return [None] * n, []
        kps = cached(c.output.cache_dir, cache_key(video, "keypoints", c.models.court_keypoints),
                     lambda: self.court_keypoints(video, total))
        if c.court.camera_motion:
            motion = cached(c.output.cache_dir, cache_key(video, "motion", c.models.detector, c.detection, "v1"),
                            lambda: self.camera_motion(video, det, total))
            return track_homographies(kps, motion, self.court, c.court.keypoint_conf, c.court.max_flow_frames,
                                      c.court.homography_smoothing, c.court.reject_threshold_m)
        Hs = estimate_homographies(kps, self.court, c.court.keypoint_conf,
                                   c.court.homography_hold_frames, c.court.homography_smoothing)
        calibration = []
        for k, H in zip(kps, Hs):
            fit, fi = fit_homography_info(k, self.court, c.court.keypoint_conf)
            calibration.append({"n_kp": fi["n_kp"], "kp_conf": fi["kp_conf"], "fit_err_m": fi["err_m"],
                                "fit": fi["reason"], "kp_rejected": False,
                                "source": "keypoints" if fit is not None else ("held" if H is not None else "none")})
        return Hs, calibration

    def estimate_poses(self, video: Path, players, total: int):
        from .pose.estimator import PoseEstimator
        p = self.cfg.pose
        est = PoseEstimator(p.model, p.imgsz, p.conf, p.batch_size, p.match_iou, self.cfg.models.device)
        return est.run(iter_frames(video), players, total=total)

    # --------------------------------------------------------------------- run
    def run(self, video_path: str | Path, output_dir: str | Path | None = None) -> PipelineResult:
        c = self.cfg
        video = Path(video_path)
        out_dir = Path(output_dir or c.output.dir) / video.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        fps, total, size = video_info(video)
        cache = c.output.cache_dir

        det = cached(cache, cache_key(video, "detect", c.models.detector, c.models.ball_detector, c.detection),
                     lambda: self.detect(video, fps, total))
        n = len(det.players)

        homographies, calibration = self.homographies(video, det, total)

        assigner = TeamAssigner(c.teams)
        needed = sorted({f for fs in assigner.sample_frames(det.players).values() for f in fs})
        team_of = assigner.assign(read_frames_at(video, needed), det.players)
        self.team_colors_bgr = TeamAssigner.team_bgr(assigner.team_colors)

        poses2d = None
        if c.pose.enabled and c.output.mode == "3d":
            poses2d = cached(cache, cache_key(video, "pose", c.pose.model, c.pose.imgsz, c.pose.conf,
                                              c.pose.match_iou, c.models.detector, c.detection),
                             lambda: self.estimate_poses(video, det.players, total))

        result = self.analyze(det, homographies, team_of, fps, size, poses2d, calibration)
        if c.output.write_diagnostics:
            result.diagnostics, result.diagnostics_frames = diagnose(
                det.players, det.ball_candidates, result, calibration, self.court,
                assigner.track_colors, assigner.team_colors, result.track_remap,
                video={"width": size[0], "height": size[1], "fps": round(fps, 2),
                       "bitrate_mbps": round(video.stat().st_size * 8 / max(total / fps, 1e-6) / 1e6, 2)
                       if total else None})
            log.info("\n%s", format_report(result.diagnostics))
        self.write_outputs(result, video, out_dir)
        return result

    def analyze(self, det, homographies, team_of: dict[int, int], fps: float,
                size: tuple[int, int], poses2d=None, calibration: list[dict] | None = None) -> PipelineResult:
        """Everything after the models: pure geometry and logic, unit-testable without a GPU."""
        c = self.cfg
        players, remap = det.players, {}
        court_pos = project_tracks(players, homographies, self.court)
        if c.tracking.relink and court_pos:
            remap = stitch_tracks(players, court_pos, team_of, fps, c.tracking.relink_max_gap_s,
                                  c.tracking.relink_max_speed_mps)
            if remap:
                players = apply_remap(players, remap)
                team_of = {t: v for t, v in team_of.items() if t not in remap}
                poses2d = apply_remap(poses2d, remap) if poses2d else poses2d
                court_pos = project_tracks(players, homographies, self.court)
        if c.tracking.smooth:
            court_pos = smooth_court_positions(court_pos, fps, c.tracking.max_fill_s, c.tracking.smooth_window_s)

        ball = clean_ball_track(det.ball_candidates, c.ball.max_speed_px_per_frame, c.ball.max_gap_frames)
        possessor = detect_possession(players, ball, c.possession.max_distance_ratio, c.possession.min_frames)
        segments = possession_segments(possessor, team_of)
        shots = detect_shots(ball, det.hoops, possessor, segments, c.events.shot_window_frames)
        passes = detect_passes(segments, c.events.interception_max_gap_frames,
                               shot_frames={e.frame for e in shots if e.type == "shot"})
        events = passes + shots

        annotate_shots(events, court_pos, self.court)
        speed, dist = speed_and_distance(court_pos, fps, c.metrics.speed_window_frames, c.metrics.max_speed_mps)
        contact = [frame_holder(p, b, c.possession.max_distance_ratio) for p, b in zip(players, ball)]
        ball_court, ball_src = ball_court_track(ball, possessor, court_pos, homographies, self.court, events, fps,
                                                contact=contact)
        ball_3d, pose_3d = [None] * len(ball), {}
        if c.output.mode == "3d":
            cams = cameras_for_frames(homographies, size)
            ball_3d = reconstruct_ball_3d(ball, possessor, cams, court_pos, fps)
            pose_3d = (lift_poses(poses2d, cams, court_pos, c.pose.player_height_m, c.pose.min_keypoint_conf,
                                  c.pose.smooth_window) if poses2d else {})

        return PipelineResult(fps=fps, frame_size=size, players=players, referees=det.referees, ball=ball,
                              hoops=det.hoops, team_of=team_of, possessor=possessor, possessions=segments,
                              events=events, homographies=homographies, court_positions=court_pos,
                              speed_kmh=speed, distance_m=dist, ball_3d=ball_3d,
                              pose_3d=pose_3d, calibration=calibration or [], track_remap=remap,
                              ball_court=ball_court, ball_court_source=ball_src)

    # ----------------------------------------------------------------- outputs
    def write_outputs(self, result: PipelineResult, video: Path, out_dir: Path) -> None:
        names = self.cfg.teams.team_names
        touches = touches_table(result.possessions, result, names, self.court, self.player_info)
        events = events_table(result.events, result.fps, names)
        touches.to_csv(out_dir / "touches.csv", index=False)
        try:
            touches.to_parquet(out_dir / "touches.parquet", index=False)
        except ImportError:
            log.warning("pyarrow not installed; touches written as CSV only.")
        events.to_csv(out_dir / "events.csv", index=False)
        box_score(events, touches).to_csv(out_dir / "box_score.csv", index=False)
        if result.court_positions:
            t2d = tracking_2d_table(result, result.ball_court, result.ball_court_source,
                                    self.cfg.output.tracking_fps)
            t2d["team"] = t2d["team"].map(lambda t: names[int(t)] if t in (0, 1) else None)
            t2d.to_csv(out_dir / "tracking_2d.csv", index=False)
            try:
                t2d.to_parquet(out_dir / "tracking_2d.parquet", index=False)
            except ImportError:
                pass
            log.info("2D coverage: %s", coverage(t2d))
        if result.court_positions and self.cfg.output.mode == "3d":
            tracking_table(result).to_csv(out_dir / "tracking.csv", index=False)
            scene = build_scene(result, self.court, names)
            save_scene_json(scene, out_dir / "scene.json")
            if self.cfg.output.write_3d_html:
                colors = tuple("#%02x%02x%02x" % tuple(reversed(c)) for c in self.team_colors_bgr)
                if self.cfg.output.viewer == "plotly":
                    scene_to_plotly(scene, self.court, out_dir / "play_3d.html", team_colors=colors)
                else:
                    scene_to_threejs(scene, out_dir / "play_3d.html", team_colors=colors,
                                     character_glb=self.cfg.output.character_glb)
        if result.diagnostics is not None:
            import json
            (out_dir / "diagnostics.json").write_text(json.dumps(result.diagnostics, indent=2, default=float))
            result.diagnostics_frames.to_csv(out_dir / "diagnostics_frames.csv", index=False)
            plot_diagnostics(result.diagnostics_frames, result.fps, out_dir / "diagnostics.png")
        if self.cfg.output.write_video:
            self.render_video(result, video, out_dir / f"{video.stem}_annotated.mp4")
        log.info("Outputs written to %s", out_dir)

    def render_video(self, result: PipelineResult, video: Path, path: Path) -> None:
        import cv2
        from .tactical.render import TacticalRenderer
        from .viz.annotate import draw_ball, draw_panel, draw_player, paste_minimap

        renderer = TacticalRenderer(self.court)
        control = team_control(result.possessor, result.team_of)
        names = self.cfg.teams.team_names
        w, h = result.frame_size
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), result.fps, (w, h))
        counts = {"pass": [0, 0], "interception": [0, 0]}
        events_by_frame: dict[int, list] = {}
        for e in result.events:
            events_by_frame.setdefault(e.frame, []).append(e)

        for f, frame in enumerate(iter_frames(video)):
            if f >= len(result.players):
                break
            for e in events_by_frame.get(f, []):
                if e.type in counts and e.team in (0, 1):
                    counts[e.type][e.team] += 1
            holder = result.possessor[f]
            for tid, p in result.players[f].items():
                team = result.team_of.get(tid, -1)
                color = self.team_colors_bgr[team] if team in (0, 1) else (128, 128, 128)
                draw_player(frame, p["bbox"], color, tid, has_ball=tid == holder)
            for tid, p in result.referees[f].items():
                draw_player(frame, p["bbox"], (0, 220, 255), tid, label="REF")
            if result.ball[f] is not None:
                draw_ball(frame, result.ball[f])

            seen = [t for t in control[: f + 1] if t in (0, 1)]
            share = [seen.count(i) / len(seen) * 100 if seen else 0 for i in (0, 1)]
            lines = [f"{names[i]}: ball {share[i]:.0f}%  P {counts['pass'][i]}  INT {counts['interception'][i]}"
                     for i in (0, 1)]
            ids = [tid for tid in result.players[f] if tid in result.speed_kmh]
            if holder != NO_POSSESSION and result.speed_kmh.get(holder, [None] * (f + 1))[f] is not None:
                lines.append(f"#{holder}: {result.speed_kmh[holder][f]:.1f} km/h, "
                             f"{result.distance_m[holder][f]:.0f} m")
            draw_panel(frame, lines)

            if result.homographies[f] is not None and ids:
                pos = {t: result.court_positions[t][f] for t in ids if result.court_positions[t][f] is not None}
                bc = result.ball_court[f] if f < len(result.ball_court) else None
                mini = renderer.render(pos, result.team_of, self.team_colors_bgr, ball=bc,
                                       possessor=holder if holder != NO_POSSESSION else None)
                paste_minimap(frame, mini)
            cv2.putText(frame, f"{f}", (w - 80, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            writer.write(frame)
        writer.release()
