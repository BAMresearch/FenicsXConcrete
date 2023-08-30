"""Test `set_log_levels` and document how logging might be controlled for
application codes"""

import logging

import dolfinx
import ffcx
import pytest
import ufl

from fenicsxconcrete import set_log_levels
from fenicsxconcrete.experimental_setup.tensile_beam import TensileBeam
from fenicsxconcrete.finite_element_problem.linear_elasticity import LinearElasticity


def test_fenicsx_loggers():
    """application specific settings for FEniCSx"""

    # ### ufl and ffcx
    ufl_logger = ufl.log.get_logger()
    # it seems per default the levels are
    # ufl: DEBUG (10)
    # ffcx: WARNIG (30)
    # but these are set to logging.WARNING per default by fenicsxconcrete
    assert ufl_logger.getEffectiveLevel() == logging.WARNING
    assert ffcx.logger.getEffectiveLevel() == logging.WARNING

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


def test_set_log_levels():
    default_p = TensileBeam.default_parameters()
    experiment = TensileBeam(default_p)
    param = LinearElasticity.default_parameters()[1]
    problem = LinearElasticity(experiment, param)

    # default level is logging.WARNING
    for obj in [experiment, problem]:
        assert obj.logger.getEffectiveLevel() == logging.WARNING

    # set level for each logger of package "fenicsxconcrete"
    set_log_levels({"fenicsxconcrete": logging.INFO})
    for obj in [experiment, problem]:
        assert obj.logger.getEffectiveLevel() == logging.INFO

    # or set log level individually
    set_log_levels({"fenicsxconcrete": logging.DEBUG, problem.logger.name: logging.ERROR})
    assert experiment.logger.getEffectiveLevel() == logging.DEBUG
    assert problem.logger.getEffectiveLevel() == logging.ERROR


if __name__ == "__main__":
    test_fenicsx_loggers()
    test_set_log_levels()
