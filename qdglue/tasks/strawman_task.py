"""Linear projection benchmark."""
import gymnasium as gymnasium
import numpy as np

from qdglue.tasks.qd_task import QDTask

# 2/(1+exp(-x)) -1


class StrawMan(QDTask):
    """Implementation of a strawman projection domain.

    * measures are random
    * fitness is just solution to a scrappy logarithmic function, so that our "happy graph" of fitness over time looks right.

    Args:
        parameter_space_dims (int): Dimensionality of each solution.
    """

    def __init__(self, parameter_space_dims):
        super().__init__()

        self._parameter_space_dims = parameter_space_dims
        self._tick = 0
        self._measure_space_dims = 2

    def evaluate(self, parameters: np.ndarray, random_key=None):
        """
        Args:
            parameters (np.ndarray): (batch_size, dim) batch of solutions.
            random_key (): unused JAX random key
        Returns:
            objective_batch (np.ndarray): (batch_size,) batch of objectives.
            objective_grad_batch (np.ndarray): (batch_size, solution_dim) batch of
                objective gradients.
            measures_batch (np.ndarray): (batch_size, 2) batch of measures.
            measures_grad_batch (np.ndarray): (batch_size, 2, solution_dim) batch of
                measure gradients.
        """

        batch_size = 1
        solution_batch = parameters
        dim = solution_batch.shape[1]

        objective_batch = []
        measures_batch = []

        measures = np.random.random(size=(batch_size, self._measure_space_dims))

        objectives = 2 / (1 + np.exp(-0.001 * self._tick)) - 1

        # print(self._tick,objectives)
        # print(measures)

        objective_batch.append(objectives)
        measures_batch.append(measures)

        self._tick += 1

        return (
            objective_batch,
            None,
            measures_batch,
            None,
        )

    @property
    def parameter_space_dims(self):
        """Dimensions of the parameter space."""
        return self._parameter_space_dims

    @property
    def objective_space_dims(self):
        """Dimensions of the objective space."""
        return 1

    @property
    def descriptor_space_dims(self):
        """Dimensions of the descriptor space.

        Always equal to 2.
        """
        return len(self._measure_space_dims)

    @property
    def descriptor_space_bounds(self):
        """Bounds of the descriptor space."""
        return self._measure_space_dims

    @property
    def parameter_type(self):
        # TODO(btjanaka): Return type?
        return "continuous"

    @property
    def parameter_space(self) -> gymnasium.spaces.Space:
        return gymnasium.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._parameter_space_dims,),
            dtype=np.float32,
        )
