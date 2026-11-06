import numpy as np

from orchardscout.mapping import MapConfig, OccupancyGrid, TrunkConfig, TrunkMap


def test_ray_frees_cells_and_disk_marks_obstacle():
    g = OccupancyGrid(MapConfig(size_m=40.0))
    cam = np.array([0.0, 0.0])
    for _ in range(5):
        g.integrate(cam, np.array([[5.0, 0.0]]), np.array([False]), stop_short=np.array([True]),
                    occ_xy=np.array([[5.0, 0.0]]), occ_radius=0.25)
    st = g.state()
    i, j = g.to_cell(np.array([2.5, 0.0]))
    assert st[i, j] == g.FREE
    i, j = g.to_cell(np.array([5.0, 0.0]))
    assert st[i, j] == g.OCCUPIED
    i, j = g.to_cell(np.array([0.0, 5.0]))
    assert st[i, j] == g.UNKNOWN


def _column(x, y, n=200, r=0.1, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.uniform(-np.pi / 2, np.pi / 2, n)  # the visible half of the trunk
    return np.column_stack([x - r * np.cos(a), y + r * np.sin(a), rng.uniform(0.3, 1.4, n)])


def test_trunk_detection_needs_a_vertical_column():
    tm = TrunkMap(TrunkConfig())
    trunk = _column(3.0, 1.0)
    flat = trunk.copy()
    flat[:, 2] = 0.3 + 0.05 * np.random.default_rng(1).random(len(flat))  # a low lump
    dets = tm.detect(np.vstack([trunk, flat + [0.0, 3.0, 0.0]]))
    assert len(dets) == 1
    assert np.linalg.norm(dets[0] - [2.95, 1.0]) < 0.1


def test_landmarks_fuse_and_confirm():
    tm = TrunkMap(TrunkConfig(min_obs=3))
    for k in range(4):
        tm.add(np.array([[2.0 + 0.05 * k, 1.0]]), np.zeros(2))
    assert len(tm.xy) == 1 and len(tm.confirmed()) == 1
    tm.add(np.array([[6.0, 1.0]]), np.zeros(2))
    assert len(tm.xy) == 2 and len(tm.confirmed()) == 1


def test_translation_correction_points_back_to_the_landmarks():
    tm = TrunkMap(TrunkConfig(min_obs=1, correct_gain=1.0, snap_gain=1.0))
    lm = np.array([[2.0, 1.5], [4.0, 1.5], [6.0, 1.5], [3.0, -1.5], [5.0, -1.5]])
    tm.add(lm, np.zeros(2))
    ang, shift = tm.fit_correction(lm + [0.3, -0.2], np.zeros(2))
    assert ang == 0.0
    np.testing.assert_allclose(shift, [-0.3, 0.2], atol=1e-9)
