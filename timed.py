#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 16:11:10 2019

@author: pi
"""

import logging
import time
from functools import wraps

logger = logging.getLogger(__name__)
# Misc logger setup so a debug log statement gets printed on stdout.
logger.setLevel("DEBUG")
handler = logging.StreamHandler()
log_format = "%(asctime)s %(levelname)s -- %(message)s"
formatter = logging.Formatter(log_format)
handler.setFormatter(formatter)
if len(logger.handlers) == 0:
    logger.addHandler(handler)


def timed(func):
    '''This decorator prints the execution time for the decorated function.
    '''
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger.debug("{} ran in {}s".format(func.__name__, round(end - start, 2)))
        return result

    return wrapper
