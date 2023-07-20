import dataclasses
from abc import ABC, abstractmethod
from typing import Dict, Tuple

import gymnasium
import numpy as np

from qdglue.types import Feature, Fitness


@dataclasses.dataclass
class DiscreteArchiveMetrics:
    """Holds statistics about an archive."""

    # Proportion of cells in the archive that have an elite - always in the
    # range :math:`[0,1]`.
    coverage: float

    # QD score, i.e. sum of objective values of all elites in the archive.
    # all the fitnesses are normalized to [0,1] using the fitness_bounds
    # before computing the QD score.
    qd_score: float

    # Maximum objective value of the elites in the archive.
    max_fitness: float

    # Complementary Cumulative Distribution of Fitness from Vassiliades et al. (2018)
    ccdf: np.ndarray


class DiscreteArchiveMetricsCalculator(ABC):
    def __init__(
        self,
        fitness_bounds: gymnasium.spaces.Box,
        num_points_ccdf: int = 100,
    ):
        assert np.all(np.isfinite(fitness_bounds.low)) and np.all(
            np.isfinite(fitness_bounds.high)
        ), "Fitness bounds must be finite"
        assert (
            np.prod(fitness_bounds.shape) == 1
        ), "Only scalar fitnesses are supported for now"

        self.fitness_bounds = fitness_bounds
        self.num_points_ccdf = num_points_ccdf

    def get_metrics(
        self,
        fitnesses: Fitness,
        features: Feature,
    ) -> DiscreteArchiveMetrics:
        self._check_validity_features(features)
        self._check_validity_fitnesses(fitnesses)

        # Extracting the best fitnesses per cell
        # The empty cells do not appear in the best_fitnesses array
        best_fitnesses = self._extract_best_fitnesses_per_cell(fitnesses, features)
        best_fitnesses_normalized = (best_fitnesses - self.fitness_bounds.low) / (
            self.fitness_bounds.high - self.fitness_bounds.low
        )
        # TODO(looka): support case where fitness_bounds.high == +inf, and just apply the offset in that case?

        # Computing the metrics
        total_cells = self.num_cells

        coverage = np.size(best_fitnesses) / total_cells
        max_fitness = np.max(best_fitnesses)
        qd_score = np.sum(best_fitnesses_normalized).item()

        # Computing the Complementary Cumulative Distribution of Fitness
        ccdf = self._compute_ccdf(best_fitnesses)

        return DiscreteArchiveMetrics(
            coverage=coverage,
            qd_score=qd_score,
            max_fitness=max_fitness,
            ccdf=ccdf,
        )

    def _compute_ccdf(
        self,
        best_fitnesses: Fitness,
    ) -> np.ndarray:
        fitnesses_intermediate = np.linspace(
            start=self.fitness_bounds.low,
            stop=self.fitness_bounds.high,
            num=self.num_points_ccdf,
            endpoint=True,
        )

        list_coverages_higher_than_fit = []

        total_cells = self.num_cells

        for _fit in fitnesses_intermediate:
            coverage_higher_than_fit = np.sum(best_fitnesses >= _fit) / total_cells
            list_coverages_higher_than_fit.append(coverage_higher_than_fit)

        ccdf = np.asarray(list_coverages_higher_than_fit)

        return ccdf

    @property
    @abstractmethod
    def num_cells(self) -> int:
        """Number of cells in the grid."""

    @abstractmethod
    def _check_validity_features(self, features: Feature) -> None:
        """Check validity of features."""

    def _check_validity_fitnesses(self, fitnesses):
        assert np.all(np.min(fitnesses, axis=0) >= self.fitness_bounds.low) and np.all(
            np.max(fitnesses, axis=0) <= self.fitness_bounds.high
        ), "Fitnesses must be within the fitness bounds"

    @abstractmethod
    def _extract_best_fitnesses_per_cell(
        self,
        fitnesses: Fitness,
        features: Feature,
    ) -> Fitness:
        """
        Returns an array of the best fitnesses per cell.
        The empty cells do not appear in the returned array.
        """


class GridMetricsCalculator(DiscreteArchiveMetricsCalculator):
    def __init__(
        self,
        feature_space: gymnasium.spaces.Box,
        fitness_bounds: gymnasium.spaces.Box,
        resolution: Tuple[int, ...],
        num_points_ccdf: int = 100,
    ):
        super().__init__(fitness_bounds, num_points_ccdf)
        self.feature_space = feature_space

        assert np.all(np.isfinite(self.feature_space.low)) and np.all(
            np.isfinite(self.feature_space.high)
        ), "Feature space bounds must be finite"

        self.resolution = resolution

    def _extract_best_fitnesses_per_cell(
        self,
        fitnesses: Fitness,
        features: Feature,
    ) -> Fitness:
        """
        Returns an array of the best fitnesses per cell.
        The empty cells do not appear in the returned array.
        """

        # First, we need to convert the features to single-number indexes in the grid
        resolution_array = np.asarray(self.resolution, dtype=np.int32)

        features_multi_indexes = (
            resolution_array
            * (features - self.feature_space.low)
            / (self.feature_space.high - self.feature_space.low)
        )
        features_multi_indexes = np.floor(features_multi_indexes).astype(np.int32)

        features_indexes = np.ravel_multi_index(
            multi_index=features_multi_indexes.T, dims=self.resolution, mode="raise"
        )  # TODO check this line

        # Collecting the best fitnesses per feature index
        best_fitness_per_feature_index: Dict[int, Fitness] = dict()

        for index_feature, fitness in zip(features_indexes, fitnesses):
            if index_feature not in best_fitness_per_feature_index:
                best_fitness_per_feature_index[index_feature] = fitness
            else:
                best_fitness_per_feature_index[index_feature] = np.maximum(
                    best_fitness_per_feature_index[index_feature], fitness
                )

        best_fitnesses = np.asarray(list(best_fitness_per_feature_index.values()))

        return best_fitnesses

    @property
    def num_cells(self) -> int:
        return np.prod(self.resolution).astype(np.int32).item()

    def _check_validity_features(self, features: Feature) -> None:
        assert np.all(np.min(features, axis=0) >= self.feature_space.low) and np.all(
            np.max(features, axis=0) <= self.feature_space.high
        ), "Features must be within the feature space bounds"


class CVTMetricsCalculator(DiscreteArchiveMetricsCalculator):
    def __init__(
        self,
        fitness_bounds: gymnasium.spaces.Box,
        centroids: Feature,
        num_points_ccdf: int = 100,
    ):
        super().__init__(fitness_bounds, num_points_ccdf)

        self.centroids = centroids

    @property
    def num_cells(self) -> int:
        return self.centroids.shape[0]

    def _check_validity_features(self, features: Feature) -> None:
        pass

    def _get_index_closest_centroid(self, feature: Feature) -> int:
        distances = np.linalg.norm(self.centroids - feature, axis=1)
        return np.argmin(distances).item()

    def _extract_best_fitnesses_per_cell(
        self,
        fitnesses: Fitness,
        features: Feature,
    ) -> Fitness:
        """
        Returns an array of the best fitnesses per cell.
        The empty cells do not appear in the returned array.
        """

        best_fitness_per_centroid_index: Dict[int, Fitness] = dict()

        for feature, fitness in zip(features, fitnesses):
            index_closest_centroid = self._get_index_closest_centroid(feature)
            if index_closest_centroid not in best_fitness_per_centroid_index:
                best_fitness_per_centroid_index[index_closest_centroid] = fitness
            else:
                best_fitness_per_centroid_index[index_closest_centroid] = np.maximum(
                    best_fitness_per_centroid_index[index_closest_centroid], fitness
                )

        best_fitnesses = np.asarray(list(best_fitness_per_centroid_index.values()))

        return best_fitnesses


def example_grid_based_metrics():
    feature_space = gymnasium.spaces.Box(low=0.0, high=1.0, shape=(2,))
    fitness_bounds = gymnasium.spaces.Box(low=0.0, high=1.0, shape=(1,))
    resolution = (10, 10)

    calculator = GridMetricsCalculator(
        feature_space=feature_space,
        fitness_bounds=fitness_bounds,
        resolution=resolution,
    )

    metrics = calculator.get_metrics(
        fitnesses=np.array([0.3, 0.5, 0.7]),
        features=np.array([[0.55, 0.55], [0.55, 0.551], [0.3, 0.3]]),
    )

    print("Grid based metrics:", metrics)


def example_cvt_based_metrics():
    fitness_bounds = gymnasium.spaces.Box(low=0.0, high=1.0, shape=(1,))

    centroids = np.array(
        [
            [0.1, 0.1],
            [0.9, 0.9],
            [0.1, 0.9],
            [0.9, 0.1],
        ]
    )

    calculator = CVTMetricsCalculator(
        fitness_bounds=fitness_bounds,
        centroids=centroids,
    )

    metrics = calculator.get_metrics(
        fitnesses=np.array([0.3, 0.5, 0.7]),
        features=np.array([[0.9, 0.9], [0.9, 0.91], [0.3, 0.3]]),
    )

    print("CVT based metrics:", metrics)


if __name__ == "__main__":
    example_grid_based_metrics()
    example_cvt_based_metrics()
