import logging

import dolfinx

# define mappings for log levels
_dolfinx_loglvl = {
    logging.DEBUG: dolfinx.log.LogLevel.INFO,
    logging.INFO: dolfinx.log.LogLevel.INFO,
    logging.WARNING: dolfinx.log.LogLevel.WARNING,
    logging.ERROR: dolfinx.log.LogLevel.ERROR,
    logging.CRITICAL: dolfinx.log.LogLevel.OFF,
}

# see http://gmsh.info/doc/texinfo/gmsh.html (General.Verbosity)
_gmsh_verbosity = {logging.DEBUG: 99, logging.INFO: 4, logging.WARNING: 2, logging.ERROR: 1, logging.CRITICAL: 0}

_supported_loggers = ["fenicsxconcrete", "ffcx", "UFL", "dolfinx", "gmsh"]


def set_log_levels(levels: dict = None) -> None:
    """Sets log levels for loggers.

    Args:
      levels: The names of the loggers and log level to be set.
        Supported names are fenicsxconcrete, ffcx, UFL, dolfix and gmsh.
        Supported levels are logging.DEBUG, logging.INFO, logging.WARNING,
        logging.ERROR and logging.CRITICAL.
    """
    default_level = logging.WARNING
    default_levels = dict.fromkeys(_supported_loggers, default_level)

    levels = levels or default_levels
    for k, v in levels.items():
        if k == "dolfinx":
            dolfinx.log.set_log_level(_dolfinx_loglvl[v])
        elif k == "gmsh":
            global _GMSH_VERBOSITY
            _GMSH_VERBOSITY = _gmsh_verbosity[v]
        else:
            logging.getLogger(k).setLevel(v)


set_log_levels()
