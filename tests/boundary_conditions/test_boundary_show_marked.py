import pathlib
import tempfile

import dolfinx
import numpy as np
import pytest
from mpi4py import MPI

from fenicsxconcrete.boundary_conditions.boundary import show_marked


def everywhere(x: np.ndarray) -> np.ndarray:
    return np.full(x[0].shape, True, dtype=bool)


def unit_interval() -> dolfinx.mesh.Mesh:
    return dolfinx.mesh.create_unit_interval(MPI.COMM_SELF, 10)


def unit_cube() -> dolfinx.mesh.Mesh:
    return dolfinx.mesh.create_unit_cube(MPI.COMM_SELF, 2, 2, 2)


def test_write_fig() -> None:
    with tempfile.NamedTemporaryFile(suffix=".png") as tf:
        domain = dolfinx.mesh.create_unit_square(MPI.COMM_SELF, 5, 5)
        show_marked(domain, everywhere, filename=tf.name)
        assert pathlib.Path(tf.name).exists()


@pytest.mark.parametrize("domain", [unit_interval(), unit_cube()])
def test_tdim(domain: dolfinx.mesh.Mesh) -> None:
    with pytest.raises(NotImplementedError):
        show_marked(domain, everywhere)


if __name__ == "__main__":
    test_write_fig()
    test_tdim()
