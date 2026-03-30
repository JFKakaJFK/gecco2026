import pygom.gp as gp
from pmlb import fetch_data
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split


def run_experiment(X_train, X_test, y_train, y_test, gpu_accelerated, logpath):
    est = gp.SymbolicRegressor(
        gpu_accelerated=gpu_accelerated,
        linear_scaling=False,
        budget_kwargs=dict(
            max_time_seconds=30,
        ),
        ims_kwargs=dict(initial_population_size=512, max_num_populations=1),
        rv_kwargs=dict(enabled=False),  # disable rv optimization
        discrete_model_kwargs=dict(
            merge_continuous=False,
            num_continuous_bins=25,
            normalize_initial_linkage_bias=True,
        ),
        tracking_kwargs=dict(
            logpath=logpath,
        ),
    )

    est.fit(X_train, y_train)

    print("Regression finished")

    r2_train = r2_score(y_train, est.predict(X_train))
    r2_test = r2_score(y_test, est.predict(X_test))

    mse_train = mean_squared_error(y_train, est.predict(X_train))
    mse_test = mean_squared_error(y_test, est.predict(X_test))

    print("Best expression:", est.model)
    print("R2 train:", r2_train)
    print("R2 test:", r2_test)
    print(f"MSE train: {mse_train}")
    print(f"MSE test: {mse_test}")


def get_dataset():
    # X, y = fetch_data("529_pollen", return_X_y=True)
    X, y = load_diabetes(return_X_y=True)

    return train_test_split(X, y)


def main():
    print("Preparing dataset...")
    X_train, X_test, y_train, y_test = get_dataset()
    print("Dataset prepared!")

    print("Running accelerated experiment")
    run_experiment(X_train, X_test, y_train, y_test, True, "results/gasr.csv")
    print("Running non-accelerated experiment")
    run_experiment(X_train, X_test, y_train, y_test, False, "results/sr.csv")


if __name__ == "__main__":
    main()
