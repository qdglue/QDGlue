"""Provides QDTask."""
from abc import ABC, abstractmethod


class QDTask(ABC):
    """Abstract base class for QD tasks."""

    def __init__(self):
        pass
        # TODO(btjanaka): What code should we put this in init function, if any?
        #
        # TODO(btjanaka): How should we handle configuration here? Is it
        # something that should be done on a per-class basis?
        #
        # TODO(btjanaka): My recommendation is that we first create tasks, then
        # as we create more, it will become obvious what code and configuration
        # should be shared here.

    def evaluate(self, parameters):
        """Evaluates

        Args:
            parameters: A batch of parameters to evaluate.
        Returns:
            (batch_of_fitness, batch_of_descriptors)
        """
        # TODO(btjanaka): How should we handle extra data in the returns here?
        # There's often a lot more info that we can return from evaluate().
        # Perhaps we can do something like `info` as is done in OpenAI Gym.

    @property
    @abstractmethod
    def parameter_space_dims(self):
        """Dimensions of the parameter space."""

    @property
    @abstractmethod
    def objective_space_dims(self):
        """Dimensions of the objective space."""

    @property
    @abstractmethod
    def descriptor_space_dims(self):
        """Dimensions of the descriptor space."""

    @property
    @abstractmethod
    def parameter_type(self):
        """Binary or continuous"""
        # TODO(btjanaka): We should be more robust to different types of
        # solutions in the future; more to discuss here.
        #
        # TODO(btjanaka): Define the return type here.
