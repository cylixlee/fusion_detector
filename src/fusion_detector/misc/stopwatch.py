import time
from typing import *

from . import console
from .utility import attribute_of

__all__ = ["StopwatchContext", "Timewatch"]

_TResult = TypeVar("_TResult")


class StopwatchContext(object):
    """A context-managed object for measuring operations' elapsed time.

    Usage::

        with StopwatchContext("task"):
            ...

    It uses `time.time()` to calculate the elapsed time of operations in
        the context, and prints it when exiting the context.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def __enter__(self) -> "StopwatchContext":
        console.message(f">>> Task {self.name} Started <<<")
        self.start = time.time()
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback) -> None:
        elapsed = time.time() - self.start
        if exception_value is not None:
            console.error(f">>> Task {self.name} {elapsed=}s <<<")
        else:
            console.success(f">>> Task {self.name} {elapsed=}s <<<")


def _timewatch_wrapper(name: str, f: Callable[..., _TResult]):
    """A simple wrapper function, applying time measurement to the original one."""

    def wrapper(*args, **kwargs):
        with StopwatchContext(name):
            return f(*args, **kwargs)

    return wrapper


class Timewatch(object):
    """A decorator class that adds time measurement to functions.

    Usage::
        @Stopwatch
        def func(...): ...

        or

        @Stopwatch("task name")
        def func(...): ...

    In the first use case, `stopwatch` will automatically retrieve the function name
        and take it as the task name to print. The second one will use the customized
        name passed in.
    """

    def __new__(cls, target: Any):
        if not isinstance(target, str) and callable(target):
            return _timewatch_wrapper(
                attribute_of(target, "__code__", "co_name"),
                target,
            )
        return super().__new__(cls)

    def __init__(self, name: str) -> None:
        """Receives a name to distinguish stopwatch tasks.

        It takes the decorated function's name if `name` is left None.
        """
        self.name = name

    def __call__(self, f: Callable[..., _TResult]):
        """Returns a wrapper function with time measurement applied."""
        return _timewatch_wrapper(self.name, f)
