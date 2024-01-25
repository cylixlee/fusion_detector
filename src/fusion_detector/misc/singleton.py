from typing import *

__all__ = ["SingletonConstructor"]


class SingletonConstructor(object):
    """A decorator class making one class definition singleton.

    It takes a class definition when decorates, returns a closure that wraps the 
        constructor of class.
    That closure ensures the real constructor is called once when the singleton instance 
        is not initialized; or simply returns **THE** instance otherwise.
    """

    def __new__(cls, class_definition: Any):
        instance = None

        def _singleton_constructor(*args, **kwargs):
            nonlocal instance
            if instance is None:
                instance = class_definition(*args, **kwargs)
            return instance
        return _singleton_constructor
