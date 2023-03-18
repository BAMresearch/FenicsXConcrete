"""Test `set_log_levels` and document how logging might be controlled for
application codes"""

import pytest
import logging

import dolfinx
import ffcx
import ufl
from mpi4py import MPI

from fenicsxconcrete import set_log_levels
from fenicsxconcrete.boundary_conditions.bcs import BoundaryConditions


def test_set_log_levels():
    domain = dolfinx.mesh.create_unit_interval(MPI.COMM_SELF, 10)
    V = dolfinx.fem.FunctionSpace(domain, ("P", 1))
    bch = BoundaryConditions(domain, V)

    # src/fenicsxconcrete/__init__.py sets log level INFO by default
    bch.logger.info("Hello, World!")
    bch.logger.debug("Hello, World!")  # will not be printed
    assert bch.logger.getEffectiveLevel() == logging.INFO

    # application specific settings for fenicsxconcrete
    set_log_levels({bch.logger.name: logging.DEBUG})
    bch.logger.debug("Hello, Debugger!")
    assert bch.logger.getEffectiveLevel() == logging.DEBUG


def test_fenicsx_loggers():

    # application specific settings for FEniCSx
    ufl_logger = ufl.log.get_logger()

    # it seems per default the levels are
    # ufl: DEBUG (10)
    # ffcx: WARNIG (30)
    assert ffcx.logger.getEffectiveLevel() == logging.WARNING
    assert ufl_logger.getEffectiveLevel() == logging.DEBUG

    set_log_levels(
        {ffcx.logger.name: logging.ERROR, ufl_logger.name: logging.CRITICAL}
            )

    assert ffcx.logger.getEffectiveLevel() == logging.ERROR
    assert ufl_logger.getEffectiveLevel() == logging.CRITICAL

    # note that dolfinx.log.LogLevel has levels ERROR, OFF, WARNING and INFO
    # and that the integer values do not follow the convention of the stdlib
    # logging
    dfx_levels = [
        dolfinx.log.LogLevel.INFO,
        dolfinx.log.LogLevel.WARNING,
        dolfinx.log.LogLevel.OFF,
        dolfinx.log.LogLevel.ERROR,
    ]
    for lvl in dfx_levels:
        print(f"Log level {lvl} has value: {lvl.value}.")

    initial_level = dolfinx.log.get_log_level()
    assert initial_level.value == -1  # WARNING

    # dolfinx.log.set_log_level() only accepts dolfinx.log.LogLevel
    with pytest.raises(TypeError):
        dolfinx.log.set_log_level(-1)
    with pytest.raises(TypeError):
        dolfinx.log.set_log_level(logging.INFO)
    with pytest.raises(TypeError):
        dolfinx.log.set_log_level("INFO")

    dolfinx.log.set_log_level(dolfinx.log.LogLevel.ERROR)
    current_level = dolfinx.log.get_log_level()
    assert current_level.value == -2


if __name__ == "__main__":
    test_set_log_levels()
    test_fenicsx_loggers()
