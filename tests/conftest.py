import numpy as np
import pytest

from courtvision.court import NBA


def look_at_camera(center, target, f=1400.0, size=(1920, 1080)):
    """Synthetic broadcast camera: returns (P, K, R, t, size)."""
    center, target = np.asarray(center, float), np.asarray(target, float)
    z = target - center
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.vstack([x, y, z])
    t = -R @ center
    K = np.array([[f, 0, size[0] / 2], [0, f, size[1] / 2], [0, 0, 1]])
    return K @ np.column_stack([R, t]), K, R, t, size


@pytest.fixture
def broadcast_cam():
    # Mid-court, high in the stands on the near sideline, looking at the court centre.
    return look_at_camera(center=(NBA.length / 2, -14.0, 9.0), target=(NBA.length / 2, NBA.width / 2, 0.0))


def floor_homography(P):
    """Image -> court homography implied by a 3x4 camera."""
    G = P[:, [0, 1, 3]]
    H = np.linalg.inv(G)
    return H / H[2, 2]
