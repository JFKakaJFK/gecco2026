import itertools
import pathlib
from types import NoneType

import yaml
import json


class Config:
    """A helper to get both human readable and exececuteable output from configuration files."""

    def __init__(self, *args, type=None, **kwargs):
        self.type = type
        self.args = args
        self.kwargs = kwargs

    def __getitem__(self, *raw_args) -> "Config":
        """A list of alternatives to expand."""
        assert len(raw_args) > 0, "At least one value is required"
        if isinstance(raw_args, tuple) and len(raw_args) == 1:
            arg_tuple = raw_args[0]
            if isinstance(arg_tuple, (bool, str, int, float, NoneType)):
                args = [arg_tuple]
            else:
                args = arg_tuple
        else:
            args = raw_args
        return Config(type="list", *args)

    def __call__(self, *args, **kwargs) -> "Config":
        """A constructor or function to call."""
        return Config(type=self.type, *args, **kwargs)

    def __getattr__(self, name) -> "Config":
        """An object to instantiate"""
        if self.type is None:
            return Config(type=name)
        else:
            self.type = f"{self.type}.{name}"
            return self

    @staticmethod
    def expand(value):
        """Expands nested configuration and yields all combinations"""

        def expand_helper(v):
            if isinstance(v, list):
                for c in itertools.product(*[list(expand_helper(item)) for item in v]):
                    yield list(c)
            elif isinstance(v, tuple):
                for c in itertools.product(*[list(expand_helper(item)) for item in v]):
                    yield c
            elif isinstance(v, dict):
                keys = list(v.keys())
                for vs in itertools.product(*[list(expand_helper(v[k])) for k in keys]):
                    yield dict(zip(keys, vs))
            elif isinstance(v, Config):
                if v.type == "list":
                    for arg in v.args:
                        yield from expand_helper(arg)
                else:
                    for args, kwargs in itertools.product(
                        list(expand_helper(v.args)), list(expand_helper(v.kwargs))
                    ):
                        c = dict()
                        if len(args) > 0:
                            c["args"] = args
                        for key, value in kwargs.items():
                            c[key] = value
                        yield {v.type: c}
            else:
                # if not isinstance(v, (bool, str, int, float, NoneType)):
                #     print(f"Possibly unhandled type: {type(v)}")

                yield v

        yield from expand_helper(value)

    @staticmethod
    def eval(cfg, ctx=None):
        """Generator that instantiates each expansion"""
        if ctx is None:
            ctx = globals()

        def from_ctx(s):
            parts = s.split(".")
            obj = ctx.get(parts[0], None)
            for member in parts[1:]:
                if obj is not None:
                    obj = getattr(obj, member, None)
            return obj

        def eval_expansion(v):
            assert not isinstance(v, Config)

            if isinstance(v, list):
                return list(eval_expansion(e) for e in v)
            elif isinstance(v, tuple):
                return tuple(eval_expansion(e) for e in v)
            elif isinstance(v, dict):
                res = dict()
                for key, raw_kwargs in v.items():
                    fn = from_ctx(key)
                    if fn is not None and isinstance(raw_kwargs, dict):
                        kwargs = {
                            k: eval_expansion(value) for k, value in raw_kwargs.items()
                        }
                        args = kwargs.pop("args", [])

                        try:
                            return fn(*args, **kwargs)
                        except Exception as e:
                            print(e)
                            kwargs["args"] = args

                    # fail silently and just return a dict
                    res[key] = eval_expansion(raw_kwargs)
                return res
            else:
                # if not isinstance(v, (bool, str, int, float, NoneType)):
                #     print(f"Possibly unhandled type: {type(v)}")

                return v

        return eval_expansion(cfg)

    @staticmethod
    def load(path: pathlib.Path | str):
        with open(path, "rb") as f:
            # return yaml.safe_load(f)
            return json.load(f)

    @staticmethod
    def save(data, path: pathlib.Path | str):
        with open(path, "+w") as f:
            # yaml.dump(data, f)
            json.dump(data, f, indent=2)


c = Config()


if __name__ == "__main__":
    print(list(Config.expand(1)), "\n=> [1]\n")
    print(list(Config.expand(False)), "\n=> [False]\n")
    print(list(Config.expand([1])), "\n=> [[1]]\n")
    print(list(Config.expand([1, 2, 3, 4])), "\n=> [[1, 2, 3, 4]]\n")
    print(list(Config.expand((1, 2))), "\n=> [(1, 2)]\n")
    print(list(Config.expand([1, 2, c[3, 4]])), "\n=> [[1, 2, 3],[1, 2, 4]]\n")
    print(
        list(Config.expand({"hello": "world", "test": c[[1, c[2, 42]], "it"]})),
        "\n=> [{'hello': 'world', 'test': [1, 2]}, {'hello': 'world', 'test': [1, 42]}, {'hello': 'world', 'test': 'it'}]\n",
    )

    # s = c.Sphere(5)
    # print(list(Config.expand(s)))
    # print(list(Config.eval(s)))

    # s = c.Repeat(c.Sphere(5), 2)
    # # print(list(Config.expand(s)))
    # # print(list(Config.eval(s)))
    # # print("??", s)
    # se = list(Config.expand(s))[0]
    # print(se)
    # print(list(Config.eval(se)))
    # print(se)
    # print(Config.eval(se))

    cf = {"Repeat": {"args": ({"Sphere": {"args": (5,)}}, 2)}}
    cf = {
        "instance": {"OneMax": {"args": (10,)}},
        "method": {"DiscreteGOMEA": {}},
        "budget": {"Budget": {"max_evaluations": 1000000, "max_time_seconds": 10}},
        "logfile": "results/raw/OneMax/010/Library (MI, LT)/000.csv",
        "loginfo": [
            ("problem_name", "OneMax"),
            ("method_name", "Library (MI, LT)"),
            ("dims", "10"),
            ("run", "0"),
        ],
    }

    from pygom import *

    cf = {
        "instance": {"BenchmarkInstance": {"args": [{"OneMax": {"args": [10]}}]}},
        "method": {"DiscreteGOMEA": {}},
        "budget": {"Budget": {"max_evaluations": 1000000, "max_time_seconds": 10}},
        "logfile": "results/raw/OneMax/010/Library (MI, LT)/000.csv",
        "loginfo": [
            ["problem_name", "OneMax"],
            ["method_name", "Library (MI, LT)"],
            ["dims", "10"],
            ["run", "0"],
        ],
    }
    task = Config.eval(cf, ctx=globals())  # {**locals(), **globals()})
    print(task)
    Tracked.run(
        task["instance"],
        task["method"],
        task["budget"],
        TrackingOptions(task["logfile"], task["loginfo"]),
    )
