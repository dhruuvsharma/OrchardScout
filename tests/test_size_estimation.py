import numpy as np

from orchardscout.geometry import rot_z
from orchardscout.orchard import generate_layout
from orchardscout.size_estimation import estimate_orchard, find_lines, grid_orientation


def _rotated_trunks(layout, angle_deg, shift=(3.0, -7.0)):
    R = rot_z(np.deg2rad(angle_deg))[:2, :2]
    return layout.trunks[:, :2] @ R.T + np.array(shift)


def test_exact_trunks_give_the_true_layout_in_any_frame():
    for seed, size in [(0, "small"), (3, "medium"), (5, "large")]:
        lay = generate_layout(seed, size)
        gt = lay.ground_truth()
        for ang in (0.0, 23.0, 71.0):
            est = estimate_orchard(_rotated_trunks(lay, ang))
            assert est.n_rows == gt["n_rows"]
            assert abs(est.length_m - gt["length_m"]) < 0.5
            assert abs(est.width_m - gt["width_m"]) < 0.5
            assert abs(est.row_spacing_m - gt["row_spacing_m"]) < 0.2
            assert abs(est.tree_spacing_m - gt["tree_spacing_m"]) < 0.2


def test_grid_orientation_is_modulo_90():
    lay = generate_layout(1, "medium")
    for ang in (5.0, 40.0, 95.0):
        theta = grid_orientation(_rotated_trunks(lay, ang))
        assert min(abs(theta - ang % 90), 90 - abs(theta - ang % 90)) < 1.0


def test_ghost_detections_between_rows_do_not_add_rows():
    lay = generate_layout(2, "small")
    rng = np.random.default_rng(0)
    xmin, ymin, xmax, ymax = lay.bbox()
    ghosts = np.column_stack([rng.uniform(xmin, xmax, 15), rng.uniform(ymin, ymax, 15)])
    est = estimate_orchard(np.vstack([lay.trunks[:, :2], ghosts]))
    assert est.n_rows == lay.n_rows


def test_local_spacing_sets_the_width():
    lay = generate_layout(4, "small")
    est = estimate_orchard(lay.trunks[:, :2], local_row_spacing=[3.0] * 30)
    assert est.spacing_source == "local"
    assert abs(est.width_m - (lay.n_rows - 1) * 3.0) < 1e-9


def test_find_lines():
    v = np.concatenate([np.full(5, 0.0), np.full(5, 3.0), np.full(4, 6.1), [1.4]])
    lines = find_lines(v + np.random.default_rng(1).normal(0, 0.05, len(v)))
    assert [len(g) for g in lines] == [5, 5, 4]
