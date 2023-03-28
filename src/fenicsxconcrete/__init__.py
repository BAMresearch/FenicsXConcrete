import logging


def set_log_levels(levels: dict = None) -> None:
    """Sets log levels for loggers.

    Args:
      levels: The names of the loggers and log level to be set.
    """
    levels = levels or {"fenicsxconcrete": logging.WARNING}
    for k, v in levels.items():
        logging.getLogger(k).setLevel(v)


set_log_levels()
