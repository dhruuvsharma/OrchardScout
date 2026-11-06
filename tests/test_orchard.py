import numpy as np
import pytest

from orchardscout.orchard import (
    SIZE_PRESETS,
    OrchardConfig,
    OrchardLayout,
    generate_layout,
    sample_start_pose,
    world_elements,
)


def test_same_seed_same_orchard():
    a, b = generate_layout(7, "medium"), generate_layout(7, "medium")
    np.testing.assert_array_equal(a.trunks, b.trunks)
    np.testing.assert_array_equal(a.canopies, b.canopies)
    assert generate_layout(8, "medium").trunks.shape != a.trunks.shape or not np.allclose(
        generate_layout(8, "medium").trunks, a.trunks
    )


@pytest.mark.parametrize("size", ["small", "medium", "large"])
def test_parameters_within_preset_ranges(size):
    cfg = OrchardConfig.preset(size)
    for seed in range(10):
        lay = generate_layout(seed, size)
        assert cfg.n_rows[0] <= lay.n_rows <= cfg.n_rows[1]
        assert cfg.row_length[0] <= lay.row_length <= cfg.row_length[1]
        assert cfg.row_spacing[0] <= lay.row_spacing <= cfg.row_spacing[1]
        assert cfg.tree_spacing[0] <= lay.tree_spacing <= cfg.tree_spacing[1]
        gt = lay.ground_truth()
        assert gt["n_rows"] == lay.n_rows  # missing trees never remove a whole row here
        assert gt["length_m"] <= lay.row_length + 1.0
        assert abs(gt["width_m"] - (lay.n_rows - 1) * lay.row_spacing) < 1.0


def test_presets_do_not_overlap_in_size():
    assert SIZE_PRESETS["small"]["row_length"][1] < SIZE_PRESETS["medium"]["row_length"][0]
    assert SIZE_PRESETS["medium"]["row_length"][1] < SIZE_PRESETS["large"]["row_length"][0]


def test_trunks_do_not_overlap_and_canopy_is_above_flight_height():
    for seed in range(10):
        lay = generate_layout(seed, "any")
        xy, r = lay.trunks[:, :2], lay.trunks[:, 2]
        d = np.hypot(*(xy[:, None, :] - xy[None, :, :]).transpose(2, 0, 1))
        np.fill_diagonal(d, np.inf)
        assert (d - r[:, None] - r[None, :]).min() > 0.5
        assert lay.min_canopy_z() > 1.5


def test_start_poses_are_clear():
    lay = generate_layout(3, "medium")
    rng = np.random.default_rng(0)
    xmin, ymin, xmax, ymax = lay.bbox()
    for mode in ("alley", "edge"):
        for _ in range(20):
            x, y, yaw = sample_start_pose(lay, rng, mode)
            assert lay.clearance(x, y) >= 0.8
            assert -np.pi <= yaw <= np.pi
            inside = xmin <= x <= xmax and ymin <= y <= ymax
            assert inside == (mode == "alley")


def test_json_roundtrip(tmp_path):
    lay = generate_layout(5, "small")
    lay.save_json(tmp_path / "layout.json")
    back = OrchardLayout.load_json(tmp_path / "layout.json")
    np.testing.assert_allclose(back.trunks, lay.trunks)
    assert back.ground_truth() == lay.ground_truth()


def test_world_elements_reference_all_textures():
    lay = generate_layout(1, "small")
    asset_xml, world_xml, assets = world_elements(lay)
    for name in assets:
        assert f'file="{name}"' in asset_xml
    assert world_xml.count("<geom") == 1 + lay.n_rows + lay.n_trees + len(lay.canopies)
