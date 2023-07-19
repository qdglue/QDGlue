"""Benchmarks for the CVTArhive."""
import numpy as np

from qdglue.tasks.linear_projection import LinearProjection


def benchmark_evaluate(benchmark):
    task = LinearProjection(20, "sphere")

    # Note this is not really a good benchmark since it includes creating the
    # random inputs. It is just an example.
    benchmark(lambda: task.evaluate(np.random.random(size=(32, 20))))
