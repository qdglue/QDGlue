"""Defines some types used in QDax"""

from typing import Dict, Generic, TypeVar, Union

import jax
import jax.numpy as jnp
from typing_extensions import TypeAlias


# MDP types
Observation: TypeAlias = jnp.ndarray
Action: TypeAlias = jnp.ndarray
Reward: TypeAlias = jnp.ndarray
Done: TypeAlias = jnp.ndarray
EnvState: TypeAlias = brax.envs.State
Params: TypeAlias = ArrayTree

# Evolution types
StateDescriptor: TypeAlias = jnp.ndarray
Fitness: TypeAlias = jnp.ndarray
Genotype: TypeAlias = ArrayTree
Descriptor: TypeAlias = jnp.ndarray
Centroid: TypeAlias = jnp.ndarray

Skill: TypeAlias = jnp.ndarray

ExtraScores: TypeAlias = Dict[str, ArrayTree]

Mask: TypeAlias = jnp.ndarray

# Others
RNGKey: TypeAlias = jax.random.KeyArray
Metrics: TypeAlias = Dict[str, jnp.ndarray]