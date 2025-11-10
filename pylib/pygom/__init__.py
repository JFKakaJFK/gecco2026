from pygom._pygoblin import *  # type: ignore # noqa: F403
from pygom._pygoblin import __version__  # noqa: F401


def default_termination_callback():
    from ctypes import c_int, pythonapi

    pythonapi.PyErr_CheckSignals.restype = c_int
    pythonapi.PyErr_CheckSignals.argtypes = []
    return pythonapi.PyErr_CheckSignals() != 0
