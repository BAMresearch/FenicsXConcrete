"""Test `set_log_levels` and document how logging might be controlled for
application codes"""

import logging

import dolfinx
import ffcx
import pytest
import ufl
from mpi4py import MPI

from fenicsxconcrete import set_log_levels
from fenicsxconcrete.boundary_conditions.bcs import BoundaryConditions
from fenicsxconcrete.experimental_setup.cantilever_beam import CantileverBeam
from fenicsxconcrete.finite_element_problem.linear_elasticity import LinearElasticity


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
    """application specific settings for FEniCSx"""

    # ### ufl and ffcx
    ufl_logger = ufl.log.get_logger()
    # it seems per default the levels are
    # ufl: DEBUG (10)
    # ffcx: WARNIG (30)
    assert ffcx.logger.getEffectiveLevel() == logging.WARNING
    assert ufl_logger.getEffectiveLevel() == logging.DEBUG

    set_log_levels({ffcx.logger.name: logging.ERROR, ufl_logger.name: logging.CRITICAL})

    assert ffcx.logger.getEffectiveLevel() == logging.ERROR
    assert ufl_logger.getEffectiveLevel() == logging.CRITICAL

    # ### dolfinx
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

    # note that dolfinx.log.LogLevel has levels INFO, WARNING, ERROR and OFF
    # and that the integer values do not follow the convention of the stdlib
    # logging
    dfx_levels = [
        (dolfinx.log.LogLevel.INFO, 0),
        (dolfinx.log.LogLevel.WARNING, -1),
        (dolfinx.log.LogLevel.ERROR, -2),
        (dolfinx.log.LogLevel.OFF, -9),
    ]
    for lvl, value in dfx_levels:
        dolfinx.log.set_log_level(lvl)
        assert dolfinx.log.get_log_level().value == value


@pytest.mark.parametrize("level", [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL])
def test_fenicsxconcrete_example(level):
    """Shows how to set log levels for fenicsxconcrete example"""

    setup_parameters = CantileverBeam.default_parameters()

    # initialize with default parameters
    experiment = CantileverBeam(setup_parameters)
    default_param = LinearElasticity.default_parameters()[1]
    problem = LinearElasticity(experiment, default_param)

    for thing in [problem, experiment]:
        assert thing.logger.getEffectiveLevel() == logging.INFO

    # set log level of each class instance
    # using `set_log_levels`
    my_levels = {problem.logger.name: level}
    set_log_levels(my_levels)

    assert problem.logger.getEffectiveLevel() == level


if __name__ == "__main__":
    test_set_log_levels()
    test_fenicsx_loggers()
