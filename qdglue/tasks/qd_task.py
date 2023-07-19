"""Provides QDTask."""
from abc import ABC, abstractmethod
from typing import Dict, Tuple

#  import jax as jax
#  import jax.numpy as jnp
#  import numpy as np
from numpy.typing import ArrayLike

# TODO(btjanaka): Since this is an abstract class, I think we should aim to make
# it independent of any framework. This way, anyone can easily install qdglue
# and import just this class. Think of someone who wants to design a task using
# PyTorch, and all they need from us is this class. They should not have to
# install JAX to develop their task.
#
# I will argue that we should make an exception for NumPy because it is so
# universal, but we can also aim to eliminate it as well.


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

    def evaluate(
        self,
        parameters: ArrayLike,
        random_key: ArrayLike = None,
    ) -> Tuple[ArrayLike, ArrayLike, Dict[str, ArrayLike]]:
        """Evaluates

        Args:
            parameters: A batch of parameters to evaluate.
            random_key:
        Returns:
            (batch_of_fitness, batch_of_descriptors, dict containing batch of extra_data)
        """
        # TODO(Looka): added a random_key parameter (useful for JAX tasks
        #  requiring random numbers).
        # btjanaka: We may want to generalize 'random_key' to something that works across all
        # frameworks; e.g., random state

        # TODO (btjanaka): Change the return type and inputs here to be "array-like"
        # and add the "return_type" parameter.

        # TODO(btjanaka): How should we handle extra data in the returns here?
        # There's often a lot more info that we can return from evaluate().
        # Perhaps we can do something like `info` as is done in OpenAI Gym.
        # (Looka) : completely agreed! changed the function signature and return type
        # btjanaka: LGTM!
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
    def objective_space_dims(self):
        """Dimensions of the objective space."""

    @property
    @abstractmethod
    def descriptor_space_dims(self):
        """Dimensions of the descriptor space."""

    @property
    @abstractmethod
    def descriptor_space_bounds(self):
        """Bounds of the descriptor space."""

    @property
    @abstractmethod
    def parameter_type(self):
        """Binary or continuous"""
        # TODO(btjanaka): We should be more robust to different types of
        # solutions in the future; more to discuss here.
        #
        # TODO(btjanaka): Define the return type here - string, enum, int? I
        # would probably go with an Enum since we have a limited set of choices.

    # @property
    # @abstractmethod
    # def parameter_space(self) -> gymnasium.spaces.Space:
    #     ...

    def get_initial_parameters(
        self,
        seed: int,
        number_parameters: int = 1,
    ) -> ArrayLike:
        # TODO(Looka): added this method to generate random individuals
        #  for a given task
        #
        # btjanaka: I think we should prefer to leave this code up to the
        # individual tasks. Each task tends to have its own definition of what a
        # good initial solution is; e.g., in linear projection I just want a
        # zero vector, but in deep RL tasks, I might want Xavier initialization.
        # Adding this code here also binds the QDTask class to a specific
        # framework, which I think we should try to avoid since it forces people
        # to install something just to get this class.
        #
        pass
        # btjanaka: I commented this out for now so that tests pass on CI. Can
        # restore later if we decide to do so.
        #  parameter_space = self.parameter_space
        #  parameter_space.seed(seed)
        #  initial_parameters = jnp.asarray(
        #      [parameter_space.sample() for _ in range(number_parameters)]
        #  )
        #  return initial_parameters
