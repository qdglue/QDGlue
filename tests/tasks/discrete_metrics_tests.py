import gymnasium
import numpy as np

from qdglue.metrics.grid_based_metrics import CVTMetricsFunctor, GridMetricsFunctor


def test_grid_based_metrics():
    feature_space = gymnasium.spaces.Box(low=0.0, high=1.0, shape=(2,))
    fitness_bounds = gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(1,))
    resolution = (2, 2)

    calculator = GridMetricsFunctor(
        feature_space=feature_space,
        fitness_bounds=fitness_bounds,
        resolution=resolution,
        num_points_ccdf=11,
    )

    metrics = calculator.get_metrics(
        fitnesses=np.array([0.3, 0.5, 0.751]),
        features=np.array([[0.55, 0.55], [0.55, 0.551], [0.3, 0.3]]),
    )

    assert metrics.max_fitness == 0.751
    assert metrics.coverage == 0.5
    assert metrics.qd_score_bound_norm == 3.251 / 2.0
    assert metrics.qd_score_original == 3.251
    assert np.all(
        metrics.ccdf
        == np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.25, 0.0, 0.0])
    )


def test_cvt_based_metrics():
    fitness_bounds = gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(1,))

    centroids = np.array(
        [
            [0.1, 0.1],
            [0.9, 0.9],
            [0.1, 0.9],
            [0.9, 0.1],
        ]
    )

    calculator = CVTMetricsFunctor(
        fitness_bounds=fitness_bounds,
        centroids=centroids,
        num_points_ccdf=11,
    )

    metrics = calculator.get_metrics(
        fitnesses=np.array([0.3, 0.5, 0.751]),
        features=np.array([[0.9, 0.9], [0.9, 0.91], [0.3, 0.3]]),
    )

    assert metrics.max_fitness == 0.751
    assert metrics.coverage == 0.5
    assert metrics.qd_score_bound_norm == 3.251 / 2.0
    assert metrics.qd_score_original == 3.251
    assert np.all(
        metrics.ccdf
        == np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.25, 0.0, 0.0])
    )
