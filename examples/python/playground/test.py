import re

import numpy as np
import pandas as pd
import sympy as sym


def lambdify_expression(e):
    """Converts a `sympy` compatible expression string into a function accepting a dataset `X`."""
    e = str(e)

    symbols = {x: sym.Symbol(x) for x in re.findall(r"(x\d+)", e)}
    expr = sym.sympify(e, locals=symbols)
    f = sym.lambdify(symbols.values(), expr, modules=[{"clip": np.clip}, "numpy"])

    def fn(X: np.ndarray):
        try:
            return f(*[X[:, int(s[1:])] for s in symbols.keys()])
        except Exception as e:
            print(e)
            return np.repeat(float("nan"), X.shape[0])

    return fn


eq = "(exp(x4) / x4) + x5 + x6"
eq = "x4 + x5 + x6"
# eq = "x5 + x6"

X = pd.read_csv("Daily_Demand_Forecasting_Orders.csv", delimiter=";")
print(",".join(map(str, X.values[:, -1].flat)))
# print(X.shape)

y_pred = lambdify_expression(eq)(X.values)

y_target = X.values[:, -1]

# print(y_pred)

# print(y_pred - y_target)

# print(np.sum((y_pred - y_target) ** 2) / X.shape[0])
