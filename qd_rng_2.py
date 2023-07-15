"""Second version of qd_rng adapted to new QDTask interface."""
import fire
import numpy as np
import tqdm

from linear_projection import LinearProjection
from qd_task import QDTask


def main(task: str = "linear_projection",
         iterations: int = 1000,
         batch_size: int = 32):
    if task == "linear_projection":
        task_instance = LinearProjection(
            parameter_space_dims=20,
            function="sphere",
        )
    else:
        raise ValueError(f"Unknown task `{task}`")

    for itr in tqdm.trange(iterations):
        parameters = np.random.random(size=(batch_size,
                                            task_instance.parameter_space_dims))
        evaluations = task_instance.evaluate(parameters)

        # TODO: Do something with evaluations -- insert into an archive, etc.


if __name__ == "__main__":
    fire.Fire(main)
