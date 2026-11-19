"""Court geometry, homographies, and following the camera through a pan."""
import numpy as np

from courtvision.camera import confirm_cuts, estimate_motion, track_homographies
from courtvision.court import FIBA, FLIP_IDX, NBA, apply_h, fit_homography, landmarks_in_image
from conftest import floor_homography
from scene import floor_texture, keypoints_for, mapping_error, pan_sequence, render


def test_keypoints_and_flip():
    assert NBA.keypoints.shape == (18, 2) and FIBA.keypoints.shape == (18, 2)
    mirrored = NBA.keypoints.copy()
    mirrored[:, 0] = NBA.length - mirrored[:, 0]
    assert np.allclose(mirrored[FLIP_IDX], NBA.keypoints)


def test_homography_roundtrip_and_landmark_projection(broadcast_cam):
    P, *_, size = broadcast_cam
    H_true = floor_homography(P)
    img = apply_h(np.linalg.inv(H_true), NBA.keypoints)
    kps = np.column_stack([img, np.ones(18)])
    kps[[0, 5, 10], 2] = 0.0                       # some keypoints not found
    H, info = fit_homography(kps, NBA)
    assert info["reason"] == "ok" and info["n_kp"] == 15
    assert np.allclose(apply_h(H, img), NBA.keypoints, atol=1e-3)
    # The fitted mapping puts every landmark back where it is, including the three it never saw.
    lm = landmarks_in_image(H, NBA, size)
    inside = (img[:, 0] >= 0) & (img[:, 0] < size[0]) & (img[:, 1] >= 0) & (img[:, 1] < size[1])
    assert np.allclose(lm[inside, :2], img[inside], atol=0.5) and (lm[:, 2] == inside).all()


def test_homography_rejects_collinear_and_too_few():
    kps = np.zeros((18, 3))
    kps[:6] = np.column_stack([np.linspace(100, 600, 6), np.linspace(100, 600, 6), np.ones(6)])
    assert fit_homography(kps, NBA)[1]["reason"] == "collinear"
    assert fit_homography(np.zeros((18, 3)), NBA)[1]["reason"] == "too_few_keypoints"


def test_motion_matches_true_pan_and_cut_breaks_it():
    cams = pan_sequence(30)
    tex = floor_texture()
    frames = [render(P, tex) for P in cams]
    motion, cuts = estimate_motion(frames)
    assert motion[0] is None and all(m is not None for m in motion[1:]) and cuts == []
    for t in (5, 15, 29):
        true = np.linalg.inv(floor_homography(cams[t])) @ floor_homography(cams[t - 1])
        pts = np.array([[100, 300], [480, 400], [850, 500]], float)
        assert np.abs(apply_h(motion[t], pts) - apply_h(true, pts)).max() < 1.0
    cut = frames[:5] + [f * 0 + 40 for f in frames[5:10]]
    motion, cuts = estimate_motion(cut)
    assert motion[5] is None and cuts == [5]


def test_mapping_follows_pan_between_sparse_noisy_fits():
    rng = np.random.default_rng(1)
    n = 120
    cams = pan_sequence(n)
    tex = floor_texture()
    motion, _ = estimate_motion([render(P, tex) for P in cams])
    kps = np.zeros((n, 18, 3))
    for f in range(0, n, 25):                       # keypoints only every 25 frames, noisy
        kps[f] = keypoints_for(cams[f], noise=1.0, rng=rng)
    kps[60] = keypoints_for(cams[60])[FLIP_IDX]     # frame 60: left and right swapped
    Hs, calib = track_homographies(kps, motion, NBA)
    errs = [mapping_error(H, P) for H, P in zip(Hs, cams)]
    assert np.median(errs) < 0.25 and max(errs) < 0.8, (np.median(errs), max(errs))
    assert calib[60]["kp_rejected"] and calib[10]["source"] == "flow" and calib[0]["source"] == "keypoints"


def test_confirm_cuts_drops_flashes(broadcast_cam):
    P, *_, size = broadcast_cam
    H = floor_homography(P)
    moved = H @ np.array([[1, 0, 300], [0, 1, 0], [0, 0, 1.0]])   # the view jumps 300 px
    Hs = [H] * 5 + [moved] * 5
    assert confirm_cuts([3, 5], Hs, size) == [5]


def test_mirror_image_fit_at_the_start_of_a_shot_is_dropped():
    # No tracked mapping yet to compare with: the mirrored fit is caught by orientation alone.
    n = 60
    cams = pan_sequence(n)
    motion, _ = estimate_motion([render(P, floor_texture()) for P in cams])
    kps = np.zeros((n, 18, 3))
    for f in (0, 20, 40):
        kps[f] = keypoints_for(cams[f])
    kps[0] = kps[0][FLIP_IDX]
    Hs, calib = track_homographies(kps, motion, NBA)
    assert calib[0]["mirrored"] and not calib[20]["mirrored"]
    assert max(mapping_error(H, P) for H, P in zip(Hs, cams)) < 0.5
