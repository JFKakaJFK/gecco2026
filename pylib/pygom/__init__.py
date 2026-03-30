from pygom._pygoblin import *  # type: ignore # noqa: F403
from pygom._pygoblin import __version__  # noqa: F401


def default_termination_callback():
    from ctypes import c_int, pythonapi

    pythonapi.PyErr_CheckSignals.restype = c_int
    pythonapi.PyErr_CheckSignals.argtypes = []
    return pythonapi.PyErr_CheckSignals() != 0


def _reexport_modules(source_package, target_package):
    """
    Automatically exposes submodules to enable the following pattern:

    ```Python
    # with re-exporting
    from package.module.module import X, Y

    # without
    import package
    package.module.module.X
    ```
    """
    import sys
    import types

    for name in dir(source_package):
        obj = getattr(source_package, name)
        if isinstance(obj, types.ModuleType):
            export_name = f"{target_package}.{name}"
            sys.modules[export_name] = obj  # expose the submodule
            _reexport_modules(obj, export_name)  # handle nesting


_reexport_modules(_pygoblin, __name__)
