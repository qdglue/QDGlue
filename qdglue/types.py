"""Defines some types used in QDax"""

from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np
from chex import ArrayTree
from typing_extensions import TypeAlias

# Evolution types
Fitness: TypeAlias = np.ndarray
Feature: TypeAlias = np.ndarray
Centroid: TypeAlias = np.ndarray

Info: TypeAlias = Dict[str, ArrayTree]

# Others
RNGKey: TypeAlias = jax.random.KeyArray
