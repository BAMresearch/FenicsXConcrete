import pathlib
import tempfile
import pytest
import dolfinx
from mpi4py import MPI
import numpy as np
from fenicsxconcrete.boundary_conditions.boundary import show_marked


def everywhere(x):
    return np.full(x[0].shape, True, dtype=bool)


def unit_interval():
    return dolfinx.mesh.create_unit_interval(MPI.COMM_SELF, 10)


def unit_cube():
    return dolfinx.mesh.create_unit_cube(MPI.COMM_SELF, 2, 2, 2)


def test_write_fig():
    with tempfile.NamedTemporaryFile(suffix=".png") as tf:
        domain = dolfinx.mesh.create_unit_square(MPI.COMM_SELF, 5, 5)
        show_marked(domain, everywhere, filename=tf.name)
        assert pathlib.Path(tf.name).exists()


@pytest.mark.parametrize("domain", [unit_interval(), unit_cube()])
def test_tdim(domain):
    with pytest.raises(NotImplementedError):
        show_marked(domain, everywhere)


if __name__ == "__main__":
    test_write_fig()
    test_tdim()
