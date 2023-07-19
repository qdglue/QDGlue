"""Provides QDTask."""
import abc
from abc import ABC, abstractmethod
from typing import Tuple, Dict

import gymnasium as gymnasium
import jax as jax
import jax.numpy as jnp
import numpy as np
from qd_glue.types import RNGKey, Solution


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

    @abc.abstractmethod
    def evaluate(self,
                 parameters: Solution,
                 random_key: RNGKey = None,
                 ) -> Tuple[jnp.ndarray, jnp.ndarray, Dict[str, jnp.ndarray]]:
        """Evaluates

        Args:
            parameters: A batch of parameters to evaluate.
            random_key:
        Returns:
            (batch_of_fitness, batch_of_descriptors, batch of extra_data)
        """
        # TODO(Looka): added a random_key parameter (useful for JAX tasks
        #  requiring random numbers).

        # TODO(btjanaka): How should we handle extra data in the returns here?
        # There's often a lot more info that we can return from evaluate().
        # Perhaps we can do something like `info` as is done in OpenAI Gym.
        # (Looka) : completely agreed! changed the function signature and return type
        #
        # TODO(btjanaka): We should incorporate DQD domains here -- for example,
        # what argument should we pass in if we want to compute gradients in
        # this function? What should we return if we are computing gradients?


    @property
    @abstractmethod
    def parameter_space_dims(self):
        """Dimensions of the parameter space."""
        # TODO(btjanaka): What return values are we expecting here? E.g. Do we
        # want 1D solutions only here, in which case this is just an int?

    @property
    @abstractmethod
    def objective_space_dims(self) -> int:
        """Dimensions of the objective space."""

    @property
    @abstractmethod
    def descriptor_space_dims(self) -> int:
        """Dimensions of the descriptor space."""

    @property
    @abstractmethod
    def descriptor_space_bounds(self):
        """Bounds of the descriptor space."""
        # TODO(looka): make this return a gymnasium.spaces.Box?
        #  it would make it more convenient I believe.

    @property
    @abstractmethod
    def parameter_type(self):
        """Binary or continuous"""
        # TODO(btjanaka): We should be more robust to different types of
        # solutions in the future; more to discuss here.
        #
        # TODO(btjanaka): Define the return type here - string, enum, int? I
        # would probably go with an Enum since we have a limited set of choices.

    @property
    @abstractmethod
    def parameter_space(self) -> gymnasium.spaces.Space:
        ...

    def get_initial_parameters(self,
                               seed: int,
                               number_parameters: int = 1,
                               ) -> np.ndarray:
        # TODO(Looka): added this method to generate random individuals
        #  for a given task
        parameter_space = self.parameter_space
        parameter_space.seed(seed)
        initial_parameters = np.asarray([
            parameter_space.sample()
            for _ in range(number_parameters)
        ])
        return initial_parameters
