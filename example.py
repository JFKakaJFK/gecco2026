import pygom.gp as gp
from pygom import BenchmarkInstance, Budget
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


def sr_example(linkage_model_kwargs: dict):
    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    est = gp.SymbolicRegressor(
        linear_scaling=True,
        budget_kwargs=dict(
            max_time_seconds=5,
        ),
        ims_kwargs=dict(initial_population_size=256, max_num_populations=1),
        discrete_model_kwargs=dict(
            merge_continuous=False,
            num_continuous_bins=25,
            **linkage_model_kwargs
        ),
    )

    est.fit(X_train, y_train)

    r2_train = r2_score(y_train, est.predict(X_train))
    r2_test = r2_score(y_test, est.predict(X_test))

    print("Best expression:", est.model)
    print("R2 train:", r2_train)
    print("R2 test:", r2_test)

if __name__ == "__main__":
    sr_example(linkage_model_kwargs = dict(metric="random"))
    sr_example(linkage_model_kwargs = dict(metric="mi"))
    sr_example(linkage_model_kwargs = dict(metric="mi", normalize_initial_linkage_bias=True))
    sr_example(linkage_model_kwargs = dict(metric="mi", intron_strategy="mark_only"))
    sr_example(linkage_model_kwargs=dict(metric="node_proximity"))
