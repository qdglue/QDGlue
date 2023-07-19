"""Second version of qd_rng adapted to new QDTask interface.

Usage:
    python qd_rng_2.py --help
"""
import fire
import jax.random
import numpy as np
import tqdm

from qd_glue.tasks.kheperax.task import KheperaxTask
from qd_glue.tasks.linear_projection import LinearProjection
from qd_glue.tasks.knights_tour import KnightsTour
from qd_glue.tasks.strawman_task import StrawMan


def main(task: str = "linear_projection",
         iterations: int = 1000,
         batch_size: int = 32):
    if task == "linear_projection":
        task_instance = LinearProjection(
            parameter_space_dims=20,
            function="sphere",
        )
        random_key = None
    elif task == "knights_tour":
        task_instance = KnightsTour("vae")
        random_key = None
    elif task == "strawman":
        print("using strawman")
        task_instance = StrawMan(parameter_space_dims=10)
        random_key = None
    elif task == "kheperax":
        print("using kheperax")
        random_key = jax.random.PRNGKey(seed=42)
        random_key, subkey = jax.random.split(random_key)
        task_instance = KheperaxTask(random_key=subkey)
    else:
        raise ValueError(f"Unknown task `{task}`")

    for itr in tqdm.trange(iterations):
        parameters = task_instance.get_initial_parameters(seed=42, number_parameters=batch_size)

        if task_instance.parameter_type == "discrete":
            parameters = np.floor(parameters).astype(int)

        if random_key is not None:
            random_key, subkey = jax.random.split(random_key)
        else:
            subkey = None
        evaluations = task_instance.evaluate(parameters, random_key=subkey)

        # TODO: Do something with evaluations -- insert into an archive, etc.


if __name__ == "__main__":
    fire.Fire(main)
