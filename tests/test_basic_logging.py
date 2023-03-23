"""Test `set_log_levels` and document how logging might be controlled for
application codes"""

import logging

import dolfinx
import ffcx
import pytest
import ufl


def test_fenicsx_loggers():
    """application specific settings for FEniCSx"""

    # ### ufl and ffcx
    ufl_logger = ufl.log.get_logger()
    # it seems per default the levels are
    # ufl: DEBUG (10)
    # ffcx: WARNIG (30)
    assert ufl_logger.getEffectiveLevel() == logging.DEBUG
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


@pytest.mark.parametrize("level", [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL])
def test_fenicsxconcrete_example(level):
    """Shows how to set log levels for fenicsxconcrete example"""
    from fenicsxconcrete import set_log_levels

    # returns dict[str, logging.Logger]
    all_loggers = logging.root.manager.loggerDict
    names = [name for name in all_loggers.keys() if name.startswith("fenicsxconcrete")]

    def check_all(loglvl):
        for name in names:
            assert logging.getLogger(name).getEffectiveLevel() == loglvl

    # all loggers related to fenicsxconcrete should have level WARNING per default
    set_log_levels(dict.fromkeys(names, logging.WARNING))  # set default
    check_all(logging.WARNING)

    # if we want also loggers for each subpackage later we can add __init__.py
    # with contents logging.getLogger(__name__) for each subpackage
    # we can set the log level for a subpackage individually ...
    subpackage = "fenicsxconcrete.experimental_setup"
    my_level = {subpackage: logging.DEBUG}
    set_log_levels(my_level)
    # note that the logger with name "fenicsxconcrete.experimental_setup"
    # did not exist until the call to set_log_levels
    # experimental_setup/__init__.py is empty
    assert logging.getLogger(subpackage).getEffectiveLevel() == logging.DEBUG
    assert logging.getLogger("fenicsxconcrete").getEffectiveLevel() == logging.WARNING

    # ... or set the level for all at once
    # by using logging.root.manager.loggerDict to get possible loggers
    set_log_levels(dict.fromkeys(names, level))
    check_all(level)


if __name__ == "__main__":
    test_fenicsx_loggers()
