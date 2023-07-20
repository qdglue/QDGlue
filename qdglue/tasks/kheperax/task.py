from typing import Tuple, Union

import gymnasium as gymnasium
import jax.numpy as jnp
import jax.random
import numpy as np
from jax.flatten_util import ravel_pytree

from qdglue.tasks.kheperax.environment import KheperaxConfig, KheperaxEnvironment
from qdglue.tasks.qd_task import QDTask
from qdglue.types import Feature, Fitness, Info, RNGKey


class KheperaxTask(QDTask):
    def __init__(self, random_key: RNGKey, config_kheperax=None):
        super().__init__()

        if config_kheperax is None:
            config_kheperax = KheperaxConfig.get_default()

        random_key, subkey = jax.random.split(random_key)
        env, policy_network, scoring_fn = KheperaxEnvironment.create_default_task(
            kheperax_config=config_kheperax, random_key=subkey
        )
        self._env: KheperaxEnvironment = env
        self._policy_network = policy_network
        self._scoring_fn = scoring_fn

        self._descriptor_space_bounds = [(0.0, 1.0), (0.0, 1.0)]
        self._descriptor_space_bounds = jnp.array
        self._objective_space_dims = 1

        random_key, subkey = jax.random.split(random_key)

        fake_batch = jnp.zeros(shape=(1, env.observation_size))

        random_key, subkey = jax.random.split(random_key)
        example_init_parameters = self._policy_network.init(subkey, fake_batch)
        flattened_parameters, _array_to_pytree_fn = ravel_pytree(
            example_init_parameters
        )

        self._parameter_space_dims = len(flattened_parameters)
        self._array_to_pytree_fn = _array_to_pytree_fn

    def evaluate(
        self,
        params: Union[np.ndarray, jnp.ndarray],
        random_key: RNGKey = None,
    ) -> Tuple[Fitness, Feature, Info]:
        params = jnp.asarray(params)

        params_pytree = jax.vmap(self._array_to_pytree_fn)(params)

        random_key, subkey = jax.random.split(random_key)
        fitness, descriptor, info, _ = self._scoring_fn(params_pytree, random_key)

        # TODO: casting values depending on return_type as suggested by Bryon
        return np.asarray(fitness), np.asarray(descriptor), info

    @property
    def parameter_space(self) -> gymnasium.spaces.Space:
        return gymnasium.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._parameter_space_dims,),
            dtype=np.float32,
        )

    @property
    def parameter_space_dims(self) -> int:
        return self._parameter_space_dims

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
        # TODO(looka): I think we can remove this, if we keep the parameter space method.
        return "continuous"
