import numpy as np
import pygom.gp as gp
from pygom import AMaLGaM, BenchmarkInstance, Budget, Sphere
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


def sr_example():
    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    est = gp.SymbolicRegressor(
        linear_scaling=True,
        budget_kwargs=dict(
            max_time_seconds=5,
        ),
        ims_kwargs=dict(initial_population_size=512, max_num_populations=1),
        rv_kwargs=dict(enabled=False),  # disable rv optimization
        population_kwargs=dict(
            gradient_step_frequency=1  # do gradient optimization after every generation
        ),
        discrete_model_kwargs=dict(
            merge_continuous=False,
            num_continuous_bins=25,
            normalize_initial_linkage_bias=True,
        ),
    )

    est.fit(X_train, y_train)

    r2_train = r2_score(y_train, est.predict(X_train))
    r2_test = r2_score(y_test, est.predict(X_test))

    print("Best expression:", est.model)
    print("R2 train:", r2_train)
    print("R2 test:", r2_test)


def rv_example():
    sphere = BenchmarkInstance(Sphere(5))
    sphere.register_target([1e-8])
    sphere.set_initial_bounds(100.0, 110.0)

    budget = Budget(max_evaluations=10_000)

    alg = AMaLGaM()

    archive, status = alg.run(sphere, budget)

    print("Best solution:", archive[0].continuous_values())
    print("Best fitness:", archive[0].quality().objectives)
    print("Target value reached:", sphere.target_reached(archive))


def rv_custom_fitness():
    def sphere(x: np.ndarray):
        return np.sum(x**2), 0.0  # objective value, constraint value

    alg = AMaLGaM()

    # works, but only for AMaLGaM at this point
    # will probably be adapted to be more similar to scipy.optimize at some point
    x, objective_value, constraint_value = alg.run(
        sphere, bounds=[(-1.0, 1.0) for _ in range(5)]
    )
    print(x, objective_value, constraint_value)


if __name__ == "__main__":
    sr_example()
    rv_example()
    # rv_custom_fitness()
