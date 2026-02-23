import numpy as np

from src.data import load_problem


def main():
    for problem in sorted(
        [
            #
            "Airfoil",
            "Bike Sharing",
            # "Breast Cancer",
            "Concrete Compressive Strength",
            "Dow Chemical",
            "Tower",
            # "Energy Cooling",
            # "Energy Heating",
            # "Yacht Hydrodynamics",
        ]
    ):
        (X, y), _ = load_problem(problem)

        y_mean = float(np.mean(y))
        y_var = float(np.var(y))
        X_ls = np.ones((X.shape[0], X.shape[1] + 1))
        X_ls[:, :-1] = X
        b = np.linalg.lstsq(X_ls, y)[0]
        y_ls = X_ls @ b
        r2_ls = 1 - np.mean((y_ls - y) ** 2) / y_var

        # print(f"LS weights: {b.tolist()}")

        print(
            f"{problem} & {X.shape[0]} & {X.shape[1]} & {y_mean:.1f} & {float(np.sqrt(y_var)):.1f} & {r2_ls:.2f}\\\\"  # & {r2_ls:.2f}
        )


if __name__ == "__main__":
    main()
