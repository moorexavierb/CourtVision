"""Synthetic broadcast scenes for the tests: a textured court seen by a panning camera."""
import cv2
import numpy as np

from courtvision.court import NBA, apply_h
from conftest import floor_homography, look_at_camera

SIZE = (960, 540)
PX_PER_M, MARGIN = 20, 30.0


def floor_texture(seed=0):
    rng = np.random.default_rng(seed)
    w = int((NBA.length + 2 * MARGIN) * PX_PER_M)
    h = int((NBA.width + 2 * MARGIN) * PX_PER_M)
    noise = rng.integers(0, 255, (h // 8, w // 8, 3), dtype=np.uint8)
    tex = cv2.resize(noise, (w, h), interpolation=cv2.INTER_NEAREST)
    tex = cv2.GaussianBlur(tex, (5, 5), 0)
    for (x0, y0), (x1, y1) in [((0, 0), (NBA.length, 0)), ((0, NBA.width), (NBA.length, NBA.width)),
                               ((0, 0), (0, NBA.width)), ((NBA.length, 0), (NBA.length, NBA.width)),
                               ((NBA.length / 2, 0), (NBA.length / 2, NBA.width))]:
        p = lambda x, y: (int((x + MARGIN) * PX_PER_M), int((y + MARGIN) * PX_PER_M))
        cv2.line(tex, p(x0, y0), p(x1, y1), (255, 255, 255), 4)
    return tex


def tex_to_court():
    return np.array([[1 / PX_PER_M, 0, -MARGIN], [0, 1 / PX_PER_M, -MARGIN], [0, 0, 1.0]])


def pan_sequence(n=120):
    """Camera on the near sideline panning from the left half to the right half."""
    cams = []
    for f in range(n):
        tx = 7.0 + 14.0 * f / (n - 1)
        P, *_ = look_at_camera((NBA.length / 2, -12.0, 8.0), (tx, NBA.width / 2, 0.0), f=800.0, size=SIZE)
        cams.append(P)
    return cams


def render(P, tex):
    H_court_to_img = np.linalg.inv(floor_homography(P))
    img = cv2.warpPerspective(tex, H_court_to_img @ tex_to_court(), SIZE)
    cv2.rectangle(img, (20, 470), (330, 520), (30, 30, 30), -1)  # static scoreboard overlay
    cv2.putText(img, "LAL 88  SAS 91  Q4 2:31", (28, 503), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return img


def keypoints_for(P, conf=0.9, noise=0.0, rng=None):
    img = apply_h(np.linalg.inv(floor_homography(P)), NBA.keypoints)
    inside = (img[:, 0] > 0) & (img[:, 0] < SIZE[0]) & (img[:, 1] > 0) & (img[:, 1] < SIZE[1])
    if noise and rng is not None:
        img = img + rng.normal(0, noise, img.shape)
    return np.column_stack([img, np.where(inside, conf, 0.0)])


def mapping_error(H, P):
    """Median court-metre error over a grid of visible floor pixels."""
    us, vs = np.meshgrid(np.linspace(50, SIZE[0] - 50, 12), np.linspace(SIZE[1] * 0.45, SIZE[1] - 30, 6))
    pix = np.column_stack([us.ravel(), vs.ravel()])
    return float(np.median(np.linalg.norm(apply_h(H, pix) - apply_h(floor_homography(P), pix), axis=1)))


