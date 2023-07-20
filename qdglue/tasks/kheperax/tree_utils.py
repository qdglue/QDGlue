import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree


def get_batch_size(tree):
    batch_size = jax.tree_leaves(tree)[0].shape[0]
    return batch_size


def get_index_pytree(tree, index):
    return jax.tree_map(lambda x: x[index], tree)


def flatten_pytree(tree):
    flatten_tree, _ = ravel_pytree(tree)
    return flatten_tree


def unflatten_pytree(array, example_tree):
    _, reconstruction_fn = ravel_pytree(example_tree)
    return reconstruction_fn(array)
