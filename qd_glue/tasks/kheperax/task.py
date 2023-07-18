from typing import Tuple, Dict

import gymnasium as gymnasium
import jax.random
import numpy as np
from jax import numpy as jnp

from qd_glue.tasks.kheperax.environment import KheperaxEnvironment, KheperaxConfig
from qd_glue.tasks.qd_task import QDTask
from qd_glue.types import Solution, Info, RNGKey, Fitness, Descriptor


class KheperaxTask(QDTask):


    def __init__(self, random_key: RNGKey, config_kheperax=None):
        super().__init__()

        if config_kheperax is None:
            config_kheperax = KheperaxConfig.get_default()

        random_key, subkey = jax.random.split(random_key)
        env, policy_network, scoring_fn = KheperaxEnvironment.create_default_task(kheperax_config=config_kheperax, random_key=subkey)
        self._env = env
        self._policy_network = policy_network
        self._scoring_fn = scoring_fn

        self._descriptor_space_bounds = [(0., 1.),
                                         (0., 1.)]
        self._objective_space_dims = 1
        self._parameter_space_dims =

    def evaluate(self,
                 parameters: Solution,
                 random_key: RNGKey = None
                 ) -> Tuple[Fitness, Descriptor, Info]:

        random_key, subkey = jax.random.split(random_key)
        return self._scoring_fn(parameters, random_key)

    @property
    def parameter_space(self) -> gymnasium.spaces.Space:
        pass

    @property
    def parameter_space_dims(self) -> int:
        pass

    @property
    def objective_space_dims(self):
        return self._objective_space_dims

    @property
    def descriptor_space_dims(self):
        return len(self._descriptor_space_bounds)

    @property
    def descriptor_space_bounds(self):
        return self._descriptor_space_bounds

    @property
    def parameter_type(self):
        return "continuous"


