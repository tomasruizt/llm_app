from typing import Callable
import bugsnag
from decorator import decorator


@decorator
def notify_bugsnag(f: Callable, *args, **kwargs):
    """Will notify bugsnag if the function raises an exception"""
    try:
        return f(*args, **kwargs)
    except Exception as e:
        bugsnag.notify(e)
        raise
