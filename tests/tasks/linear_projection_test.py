"""Tests for the LinearProjectionTask."""
from qdglue.tasks.linear_projection import LinearProjection


def test_descriptor_dims():
    task = LinearProjection(10, "sphere")
    assert task.descriptor_space_dims == 2
