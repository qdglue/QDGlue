import enum

import numpy as np
import jax
import jax.numpy as jnp

class ArrayTypes(enum.Enum):
    NumpyArray = "Numpy"
    JaxArray = "JaxArray"

def numpy_array_from_jax_array(jax_array: jnp.ndarray) -> np.ndarray:
    return np.asarray(jax_array)

def jax_array_from_numpy_array(numpy_array: np.ndarray) -> jnp.ndarray:
    return jnp.asarray(numpy_array)

def cast(input_return, output_return):

