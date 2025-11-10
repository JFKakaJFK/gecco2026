import re

import numpy as np
import sympy as sym
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer

import pygom

# TODO SRBench complicance (ideally both new and old)
# TODO simplification & finetuning...


class SymbolicRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.imputer = None
        self.front = []
        self.model = None

    def get_params(self, deep=True):
        return self.kwargs

    def set_params(self, **kwargs):
        self.kwargs = kwargs

    def __lambdify_expression(self, e):
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

    def fit(self, X: np.ndarray, y: np.ndarray):
        if not isinstance(X, np.ndarray):
            X = X.to_numpy().astype(np.float64)

        if not isinstance(X, np.ndarray):
            y = y.to_numpy().astype(np.float64)

        # impute if needed
        self.imputer = IterativeImputer(
            max_iter=10,
            random_state=self.kwargs.get("random_state", None),
            sample_posterior=True,
        )
        if np.isnan(X).any():
            X = self.imputer.fit_transform(X)
        else:
            self.imputer.fit(X)

        alg = pygom.MixedGOMEA(
            population_options=pygom.PopulationOptions(
                **self.kwargs.get("population_kwargs", {})
            ),
            ims_options=pygom.IMSOptions(**self.kwargs.get("ims_kwargs", {})),
            rv_options=pygom.RvOptions(**self.kwargs.get("rv_kwargs", {})),
            discrete_model=vars(pygom)[
                self.kwargs.get("discrete_model", "LinkageTreeFOS")
            ](**self.kwargs.get("discrete_model_kwargs", {})),
            continuous_model=vars(pygom)[
                self.kwargs.get("continuous_model", "FullFOS")
            ](**self.kwargs.get("continuous_model_kwargs", {})),
            repr=self.kwargs.get("repr", "aos"),
        )
        budget_kwargs = self.kwargs.get("budget_kwargs", {})
        # if "termination_callback" not in budget_kwargs:
        #     budget_kwargs["termination_callback"] = pygom.default_termination_callback
        budget = pygom.Budget(**budget_kwargs)
        template = pygom.Template()
        for branching_factor, depth in self.kwargs.get("outputs", [(2, 4)]):
            template.add_output(pygom.TemplateNode.full_nary(branching_factor, depth))
        for branching_factor, depth in self.kwargs.get("subtrees", []):
            template.add_subtree(pygom.TemplateNode.full_nary(branching_factor, depth))
        str2op = {
            "+": pygom.OpAdd(),
            "-": pygom.OpSub(),
            "*": pygom.OpMul(),
            "/": pygom.OpDiv(),
            "sin": pygom.OpSin(),
            # TODO
        }
        ctx = pygom.GPContext(
            num_inputs=int(X.shape[1]),
            expression_template=template,
            operators=[
                str2op[op]
                for op in self.kwargs.get("operators", "+,-,*,/,sin".split(","))
            ],
            constant_representation=self.kwargs.get("constant_representation", "ercs"),
        )

        Y = y.reshape(-1, 1) if len(y.shape) == 1 else y

        linear_scaling = self.kwargs.get("linear_scaling", True)
        init = vars(pygom)[self.kwargs.get("init", "HalfHalfInit")]()
        problem = pygom.SRProblem(
            ctx,
            X,
            Y,
            objectives=self.kwargs.get("objectives", "mse"),
            linear_scaling=linear_scaling,
            init=init,
            constant_init_lower_bound=self.kwargs.get(
                "constant_init_lower_bound", float(-np.nanmax(Y))
            ),
            constant_init_upper_bound=self.kwargs.get(
                "constant_init_upper_bound", float(np.nanmax(Y))
            ),
        )

        seed = self.kwargs.get("random_state", self.kwargs.get("seed"))
        archive, _status = alg.run(problem, budget, seed=seed)

        def str2expr(e: str):
            # format is expr1,expr2,...
            return str(e).split(",")[0]

        # use best in first objective by default
        self.model = str2expr(problem.format_solution(archive.so_solution(0)))

        self.front = [
            str2expr(problem.format_solution(archive[i])) for i in range(archive.size())
        ]

    def predict(self, X: np.ndarray):
        assert self.model is not None

        if not isinstance(X, np.ndarray):
            X = X.to_numpy().astype(np.float64)

        # impute if needed
        if np.isnan(X).any():
            X = self.imputer.transform(X)

        y_pred = self.__lambdify_expression(self.model)(X)

        if len(y_pred.shape) < 2:
            y_pred = y_pred.reshape(-1, 1)

        return y_pred
