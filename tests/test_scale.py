import numpy as np

from orchardscout.orchard import generate_layout
from orchardscout.scale import GroundPlaneConfig, GroundPlaneScale
from orchardscout.sim import OrchardSim


def test_recovers_a_known_scale_from_the_ground_plane():
    sim = OrchardSim(generate_layout(4, "small"), seed=1)
    try:
        gps = GroundPlaneScale(
            sim.K, sim.cam_cfg.pitch_deg, GroundPlaneConfig(altitude=sim.drone_cfg.altitude)
        )
        for _ in range(5):
            sim.reset()
            gt = sim.depth_gt()
            s = gps.update(gt * 4.0)  # a network that thinks everything is 4x farther
            assert gps.last_fit.ok
            assert abs(s - 0.25) < 0.01
            assert gps.last_fit.tilt_deg < 3.0
    finally:
        sim.close()


def test_rejects_a_frame_without_ground():
    K = np.array([[300.0, 0, 319.5], [0, 300.0, 239.5], [0, 0, 1]])
    gps = GroundPlaneScale(K, 5.0)
    wall = np.full((480, 640), 2.0)  # a fronto-parallel wall fills the view
    assert np.isnan(gps.update(wall))
    assert not gps.last_fit.ok
