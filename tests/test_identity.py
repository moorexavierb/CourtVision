"""Player identity: tracking through pans and cuts, class votes, and re-joining identities.

A broadcast pan moves every box on screen together. Tracking in
camera-stabilised coordinates must keep each player's id through a fast pan
with missed detections; a cut must reset ids; and identities must be re-joined
in court space after a player leaves the picture.
"""
import numpy as np

from courtvision.detection.stitch import assign_identities
from courtvision.detection.tracker import PLAYER, REFEREE
from courtvision.detection.tracking import track_people, warp_boxes
from courtvision.types import RawDetections


def shift(dx, dy=0.0):
    return np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1.0]])


def panning_people(n=90, missing=range(40, 46)):
    """Three players standing still on the floor while the camera whips left and right.
    Returns per-frame people and the frame-to-frame camera motion."""
    spots = [(400.0, 500.0), (700.0, 520.0), (1000.0, 480.0)]
    pan = [0.0] + [60.0 * np.sin(f / 6.0) for f in range(1, n)]  # px per frame, changes direction
    motion, people, x0 = [None], [], 0.0
    for f in range(n):
        if f:
            motion.append(shift(-pan[f]))
            x0 -= pan[f]
        ppl = []
        if f not in missing:
            for x, y in spots:
                ppl.append({"bbox": [x + x0 - 20, y - 120, x + x0 + 20, y], "conf": 0.9, "cls": PLAYER})
        people.append(ppl)
    return people, motion


def ids(tracks):
    return set().union(*[set(ft) for ft in tracks])


def test_warp_boxes():
    b = np.array([[10.0, 20.0, 30.0, 60.0]])
    assert np.allclose(warp_boxes(b, shift(5, -2)), [[15, 18, 35, 58]])
    assert np.allclose(warp_boxes(b, np.diag([2.0, 2.0, 1.0])), [[20, 40, 60, 120]])


def test_tracking_through_a_pan():
    people, motion = panning_people()
    players, refs = track_people(people, motion, fps=30.0)
    assert len(ids(players)) == 3 and not ids(refs)
    # Each id stays on the same player (same screen-y row, since the camera only pans sideways).
    rows = {}
    for ft in players:
        for tid, d in ft.items():
            rows.setdefault(tid, set()).add(round(d["bbox"][3]))
    assert all(len(r) == 1 for r in rows.values())

    plain, _ = track_people(people, motion, fps=30.0, compensate=False)
    assert len(ids(plain)) > 3  # screen-space tracking breaks on the same pan


def test_cut_resets_ids_and_class_vote():
    n, cut = 40, 20
    people = []
    for f in range(n):
        x = 300.0 if f < cut else 900.0  # after the cut, a different player stands somewhere else
        cls = REFEREE if f in (5, 6, 7) else PLAYER  # detector briefly calls the player a referee
        people.append([{"bbox": [x, 100, x + 40, 220], "conf": 0.9, "cls": cls},
                       {"bbox": [600, 100, 640, 220], "conf": 0.8, "cls": REFEREE}])
    players, refs = track_people(people, [None] + [np.eye(3)] * (n - 1), cuts={cut}, fps=30.0)
    before = ids(players[:cut])
    after = ids(players[cut:])
    assert len(before) == 1 and len(after) == 1 and not before & after
    assert min(after) > max(ids(players[:cut]) | ids(refs[:cut]))  # ids stay unique across the reset
    assert len(ids(refs[:cut])) == 1 and len(ids(refs[cut:])) == 1  # the referee is never a player


def frames_from(pos, n):
    return [{t: {"bbox": [0, 0, 1, 1], "conf": 0.9} for t, p in pos.items() if p[f] is not None} for f in range(n)]


def test_identities_rejoin_after_leaving_frame():
    fps, n = 30.0, 200
    # Player 1 runs off the side of the picture at frame 40 and comes back as track 9 at frame 130,
    # 6 m further on. Teammate 2 is on screen throughout; 3 is an opponent.
    pos = {1: [(10 + 0.05 * f, 5.0) if f < 40 else None for f in range(n)],
           9: [(18 + 0.02 * (f - 130), 5.0) if f >= 130 else None for f in range(n)],
           2: [(12.0, 10.0)] * n,
           3: [(11.0, 6.0)] * n}
    team_of = {1: 0, 9: 0, 2: 0, 3: 1}
    remap, conf, extras = assign_identities(frames_from(pos, n), pos, team_of, fps)
    assert remap == {9: 1} and not extras
    assert 0 < conf[1] < 1 and conf[2] == 1.0  # 6 m in 3 s: likely him, but a new player was possible


def test_identities_match_jointly_after_a_cut_and_cap_five():
    fps, n, cut = 30.0, 120, 60
    # Two teammates both restart after a cut (the tracker reset) at nearly unchanged court spots;
    # matching each new track to the nearest lost player must not cross them.
    pos = {1: [(5.0, 5.0) if f < cut else None for f in range(n)],
           2: [(9.0, 5.0) if f < cut else None for f in range(n)],
           11: [(9.2, 5.1) if f >= cut else None for f in range(n)],
           12: [(5.1, 5.2) if f >= cut else None for f in range(n)]}
    team_of = {1: 0, 2: 0, 11: 0, 12: 0}
    remap, _, _ = assign_identities(frames_from(pos, n), pos, team_of, fps)
    assert remap == {11: 2, 12: 1}

    # Six team-0 tracks on screen at once: one of them can't be a sixth player on the floor.
    pos6 = {t: [(2.0 * t, 5.0)] * n for t in range(1, 7)}
    _, _, extras = assign_identities(frames_from(pos6, n), pos6, {t: 0 for t in pos6}, fps)
    assert len(extras) == 1


def test_pipeline_tracks_raw_detections(tmp_path, monkeypatch):
    """pipe.run with a stubbed detector returning untracked boxes: tracking runs on camera motion."""
    from courtvision.config import Config
    from courtvision.io.video import save_video
    from courtvision.pipeline import Pipeline
    from test_camera_tracking import floor_texture, pan_sequence, render

    n = 40
    tex = floor_texture()
    frames = [render(P, tex) for P in pan_sequence(n)]
    video = tmp_path / "pan.mp4"
    save_video(frames, video, 30.0)
    people, motion = panning_people(n, missing=range(15, 19))
    raw = RawDetections(people=people, ball_candidates=[[]] * n, hoops=[None] * n)

    cfg = Config.load(None, output={"dir": str(tmp_path / "out"), "cache_dir": None, "write_video": False,
                                    "write_3d_html": False, "write_diagnostics": False})
    pipe = Pipeline(cfg)
    monkeypatch.setattr(pipe, "detect", lambda *a: raw)
    monkeypatch.setattr(pipe, "camera_motion", lambda *a: (motion, set()))
    result = pipe.run(video)
    assert len(ids(result.players)) == 3


def test_flash_is_not_a_cut():
    from courtvision.court.homography import confirm_cuts
    H = np.diag([0.01, 0.01, 1.0])
    other = H @ shift(-800)  # a different view: the same pixels land 8 m away on the court
    Hs = [H] * 10 + [other] * 10
    # Histogram jumps at 4 (a flash; the court view is unchanged) and at 10 (a real cut).
    assert confirm_cuts({4, 10}, Hs, (1920, 1080)) == {10}
    assert confirm_cuts({4, 10}, [None] * 20, (1920, 1080)) == {4, 10}
