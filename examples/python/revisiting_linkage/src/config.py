# TODO this whole thing effectively is a DSL for python -> why not just use python directly?
# i.e. instead of directly evaluating given some ctx, make it a python file with imports and call that instead of using yaml or something else... (i.e. instead of writing a custom instantiaton function, write a custom serialization function that outputs python code)

import pathlib


class Config:
    """A helper to get both human readable and exececuteable output from configurations."""

    def __init__(self, type=None, attr=None, args=None, kwargs=None):
        self.type = type
        self.attr = attr
        self.args = args
        self.kwargs = kwargs

    def __getitem__(self, key) -> "Config":
        return Config(type=self, args=[key])

    def __call__(self, *args, **kwargs) -> "Config":
        return Config(type=self, args=list(args), kwargs=kwargs)

    def __getattr__(self, name) -> "Config":
        if self.type is None:
            return Config(type=name)
        else:
            return Config(
                attr=name,
                type=self,
            )


c = Config()


def extract(v):
    """Replaces `Config` objects with plain dictionaries"""
    if isinstance(v, Config):
        cfg = dict()
        if v.attr is not None:
            cfg["attr"] = v.attr
            cfg["type"] = extract(v.type)
        else:
            cfg["type"] = extract(v.type)
            if v.args is not None:
                cfg["args"] = extract(v.args)
            if v.kwargs is not None:
                cfg["kwargs"] = extract(v.kwargs)
        return cfg
    elif isinstance(v, (tuple, list)):
        return [extract(e) for e in v]
    elif isinstance(v, dict):
        return {k: extract(e) for k, e in v.items()}
    else:
        return v


def save_config(conf, path: pathlib.Path | str):
    import yaml

    # ensure the directory structure exists
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)

    # Config type objects are not serializeable, so they need to be replaced with plain dicts...
    serializeable = extract(conf)

    with open(path, "+w") as f:
        yaml.dump(serializeable, f)


def load_config(path: pathlib.Path | str):
    import yaml

    with open(path, "rb") as f:
        return yaml.safe_load(f)


def instantiate(conf, ctx=None):
    if ctx is None:
        ctx = globals()

    def eval_helper(v):
        if isinstance(v, (tuple, list)):
            return [eval_helper(e) for e in v]
        elif isinstance(v, dict):
            evaluated = {k: eval_helper(e) for k, e in v.items()}
            if isinstance(evaluated.get("attr", None), str) and isinstance(
                v.get("type", None), dict
            ):
                # attribute access (dict or member)
                if (
                    isinstance(evaluated["type"], dict)
                    and evaluated["attr"] in evaluated["type"]
                ):
                    return evaluated["type"][evaluated["attr"]]
                elif hasattr(evaluated["type"], evaluated["attr"]):
                    return getattr(evaluated["type"], evaluated["attr"])
            elif isinstance(evaluated.get("type", None), str):
                # type lookup
                if evaluated["type"] in ctx:
                    return ctx[evaluated["type"]]
                else:
                    print(
                        f"Possibly failed to evaluate '{evaluated['type']}':\nGot: {v}\n"
                    )
            elif "type" in evaluated and isinstance(evaluated.get("args"), list):
                if isinstance(evaluated.get("kwargs"), dict):
                    # constructor/member function call
                    try:
                        return evaluated["type"](
                            *evaluated["args"], **evaluated["kwargs"]
                        )
                    except Exception as e:
                        print(e)
                elif len(evaluated["args"]) == 1:
                    # indexing
                    return evaluated["type"][evaluated["args"][0]]

            return evaluated
        else:
            return v

    return eval_helper(extract(conf))


if __name__ == "__main__":
    import numpy as np

    conf = c.np.array([1, 2])

    assert (instantiate(conf) == np.array([1, 2])).all(), extract(conf)

    conf = c.np.random.default_rng(seed=42).integers(10)

    assert instantiate(conf) == np.random.default_rng(seed=42).integers(10), extract(
        conf
    )

    conf = c.np.random.default_rng(seed=42).integers(10, size=2)[1]
    expected = np.random.default_rng(seed=42).integers(10, size=2)[1]

    assert instantiate(conf) == expected, extract(conf)

    a_random_number = conf
    conf = dict(a_random_number=a_random_number, an_array=c.np.array([a_random_number]))

    assert str(instantiate(conf)) == str(
        dict(a_random_number=expected, an_array=np.array([expected]))
    ), extract(conf)
