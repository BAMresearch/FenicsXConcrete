import logging


class LogMixin(object):
    @property
    def logger(self):
        name = self.__class__.__module__
        return logging.getLogger(name)
