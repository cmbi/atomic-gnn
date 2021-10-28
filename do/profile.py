import logging
from time import time

_log = logging.getLogger(__name__)


def time_profile(function):
    def time_wrapper(*args, **kwargs):
        start_time = time()
        result = function(*args, **kwargs)
        end_time = time()

        _log.debug("{} took {} seconds".format(function.__name__, end_time - start_time))

        return result

    return time_wrapper
