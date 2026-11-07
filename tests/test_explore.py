import numpy as np

from orchardscout.explore import ExploreConfig, Explorer
from orchardscout.mapping import MapConfig, OccupancyGrid


def _grid_with_free_box(x0, x1, y0, y1, size=60.0):
    g = OccupancyGrid(MapConfig(size_m=size))
    xs, ys = np.meshgrid(np.arange(x0, x1, 0.1), np.arange(y0, y1, 0.1))
    ij = g.to_cell(np.column_stack([xs.ravel(), ys.ravel()]))
    g.logodds[ij[:, 0], ij[:, 1]] = -2.0
    g.seen[ij[:, 0], ij[:, 1]] = True
    return g


def test_plan_goes_around_a_row_of_trunks_to_a_frontier():
    g = _grid_with_free_box(-5, 5, -5, 5)
    row = np.column_stack([np.arange(-5.0, 3.0, 1.5), np.zeros(6)])  # wall with a gap at x > 3
    ex = Explorer(ExploreConfig(relevance_radius=8.0))
    ex.start_xy = np.array([0.0, -3.0])
    plan = ex._make_plan(np.array([0.0, -3.0]), g, row, row)
    assert plan is not None
    # the path must stay clear of every trunk by the inflation margin
    d = np.linalg.norm(plan.path[:, None, :] - row[None, :, :], axis=2).min()
    assert d >= ex.cfg.inflate
    assert np.linalg.norm(plan.goal) > 3.0  # a frontier at the edge of the known box


def test_no_relevant_frontier_means_explored():
    g = _grid_with_free_box(-5, 5, -5, 5)
    far_tree = np.array([[40.0, 40.0]] * 3) + np.arange(3)[:, None]
    ex = Explorer(ExploreConfig(relevance_radius=3.0, final_spins=1))
    ex.row_spacing = 2.0  # spacing known: only frontiers near trees count
    ex.mode = ex.FOLLOW
    T = np.eye(4)
    T[:3, 2] = [1.0, 0.0, 0.0]
    v, w = ex.command(T, g, far_tree, far_tree)
    assert ex.mode == ex.SPIN and w > 0  # one last look around before finishing
    for _ in range(200):
        ex.command(T, g, far_tree, far_tree)
        if ex.done:
            break
    assert ex.done and ex.status == "explored" and ex.n_final_spins == 1


def test_frontiers_near_the_start_count_until_spacing_is_known():
    g = _grid_with_free_box(-5, 5, -5, 5)
    trees = np.array([[2.0, 2.7], [-1.0, 2.7], [2.0, -2.7], [-1.0, -2.7]])  # > 5 m from edges
    ex = Explorer(ExploreConfig(relevance_radius=1.0))
    ex.start_xy = np.zeros(2)
    assert ex._make_plan(np.zeros(2), g, trees, trees) is not None  # spacing unknown
    ex.row_spacing = 0.5
    assert ex._make_plan(np.zeros(2), g, trees, trees) is None  # only near-tree frontiers


def test_frontiers_count_farther_along_the_rows():
    g = _grid_with_free_box(-8, 8, -5, 5)
    trees = np.array([[-1.0, -1.5], [1.0, -1.5], [-1.0, 1.5], [1.0, 1.5]])
    ex = Explorer(ExploreConfig(relevance_radius=3.0, relevance_along=8.0))
    ex.start_xy = np.array([0.0, 0.0])
    ex.row_spacing = 0.5  # known: no start radius; radius stays 3 m
    # frontiers: 7 m from the trees at the x edges, 3.5 m at the y edges
    assert ex._make_plan(np.zeros(2), g, trees, trees) is None
    ex.row_dir = np.array([1.0, 0.0])  # rows along x: the x edges lie beyond the row ends
    plan = ex._make_plan(np.zeros(2), g, trees, trees)
    assert plan is not None and abs(plan.goal[0]) > 6.0
    ex.row_dir = np.array([0.0, 1.0])  # rows along y: now the y edges are row ends
    plan = ex._make_plan(np.zeros(2), g, trees, trees)
    assert plan is not None and abs(plan.goal[1]) > 4.0 and abs(plan.goal[0]) < 6.0


def test_leaving_an_inflated_obstacle_never_cuts_past_the_trunk():
    g = _grid_with_free_box(-5, 5, -5, 5)
    trunk = np.array([[0.3, 0.0], [0.3, 0.0], [0.3, 0.0]])  # just ahead of the drone
    ex = Explorer(ExploreConfig(relevance_radius=8.0))
    ex.start_xy = np.zeros(2)
    plan = ex._make_plan(np.array([0.0, 0.0]), g, trunk, trunk)
    assert plan is not None
    hop = np.linspace(plan.path[0], plan.path[1], 20)
    assert np.linalg.norm(hop - trunk[0], axis=1).min() >= 0.28  # never closer than now


def test_watchdog_ends_a_run_that_stays_in_one_place():
    g = _grid_with_free_box(-5, 5, -5, 5)
    ex = Explorer(ExploreConfig(stall_steps=50, relevance_radius=8.0))
    ex.mode = ex.FOLLOW
    T = np.eye(4)
    T[:3, 2] = [1.0, 0.0, 0.0]
    trees = np.array([[2.0, 2.7], [-1.0, 2.7], [2.0, -2.7]])
    for _ in range(60):  # pose never changes, as if every command were refused
        ex.command(T, g, trees, trees)
        if ex.done:
            break
    assert ex.done and ex.status == "stuck"


def test_safety_layer_backs_away_from_a_close_trunk():
    ex = Explorer()
    T = np.eye(4)
    T[:3, 2] = [1.0, 0.0, 0.0]  # facing +x
    v, w = ex._safety(T, np.array([[0.3, 0.0]]), np.array([1.0, 0.0]), 0.0)
    assert v[0] < 0  # moves backwards, away from the trunk ahead
    v, w = ex._safety(T, np.array([[0.8, 0.0]]), np.array([1.0, 0.0]), 0.2)
    assert v[0] == 0.0  # trunk within the look-ahead: stop
