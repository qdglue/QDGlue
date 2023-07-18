"""Linear projection benchmark."""
import numpy as np

from qd_glue.tasks.qd_task import QDTask


class LinearProjection(QDTask):
    """Implementation of the linear projection domain.

    Args:
        parameter_space_dims (int): Dimensionality of each solution.
        function (str): The name of the function to optimize. Supported options
            are: ["sphere"]
    """

    def __init__(self, parameter_space_dims, function):
        self._parameter_space_dims = parameter_space_dims
        max_bound = parameter_space_dims / 2 * 5.12
        self._measure_space_dims = [(-max_bound, max_bound),
                                    (-max_bound, max_bound)]

        # TODO(btjanaka): Add rastrigin support
        if function not in ["sphere"]:
            raise ValueError(f"Unsupported function `{function}`")

        self._function = function

    def evaluate(self, parameters):
        """Sphere function evaluation and measures for a batch of solutions.

        Args:
            parameters (np.ndarray): (batch_size, dim) batch of solutions.
        Returns:
            objective_batch (np.ndarray): (batch_size,) batch of objectives.
            objective_grad_batch (np.ndarray): (batch_size, solution_dim) batch of
                objective gradients.
            measures_batch (np.ndarray): (batch_size, 2) batch of measures.
            measures_grad_batch (np.ndarray): (batch_size, 2, solution_dim) batch of
                measure gradients.
        """
        # TODO(btjanaka): We don't actually use self here. Maybe we can use the
        # predefined properties on self to run some validation checks on the
        # inputs, e.g., for dimensionality.

        # TODO(btjanaka): Rename solution_batch to parameters if needed.
        solution_batch = parameters

        dim = solution_batch.shape[1]

        # Shift the Sphere function so that the optimal value is at x_i = 2.048.
        sphere_shift = 5.12 * 0.4

        # Normalize the objective to the range [0, 100] where 100 is optimal.
        best_obj = 0.0
        worst_obj = (-5.12 - sphere_shift)**2 * dim
        raw_obj = np.sum(np.square(solution_batch - sphere_shift), axis=1)
        objective_batch = (raw_obj - worst_obj) / (best_obj - worst_obj) * 100

        # Compute gradient of the objective.
        objective_grad_batch = -2 * (solution_batch - sphere_shift)

        # Calculate measures.
        clipped = solution_batch.copy()
        clip_mask = (clipped < -5.12) | (clipped > 5.12)
        clipped[clip_mask] = 5.12 / clipped[clip_mask]
        measures_batch = np.concatenate(
            (
                np.sum(clipped[:, :dim // 2], axis=1, keepdims=True),
                np.sum(clipped[:, dim // 2:], axis=1, keepdims=True),
            ),
            axis=1,
        )

        # Compute gradient of the measures.
        derivatives = np.ones(solution_batch.shape)
        derivatives[clip_mask] = -5.12 / np.square(solution_batch[clip_mask])

        mask_0 = np.concatenate((np.ones(dim // 2), np.zeros(dim - dim // 2)))
        mask_1 = np.concatenate((np.zeros(dim // 2), np.ones(dim - dim // 2)))

        d_measure0 = derivatives * mask_0
        d_measure1 = derivatives * mask_1

        measures_grad_batch = np.stack((d_measure0, d_measure1), axis=1)

        return (
            objective_batch,
            objective_grad_batch,
            measures_batch,
            measures_grad_batch,
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
