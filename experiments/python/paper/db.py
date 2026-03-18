import glob
import os
import pathlib

import duckdb
from tqdm import tqdm


def create_db(dir: pathlib.Path):
    print("Starting db creation...")

    db_path = dir / "all_results.duckdb"

    if os.path.exists(db_path):
        print("Removed previous database")
        os.remove(db_path)

    conn = duckdb.connect(db_path)

    # Create table
    # conn.execute(
    #     """
    # CREATE TABLE IF NOT EXISTS results (
    #     algorithm TEXT,
    #     dataset TEXT,
    #     total_time_seconds DOUBLE,
    #     expression TEXT,
    #     mse DOUBLE,
    #     evaluation UBIGINT,
    #     fold INTEGER,
    #     num_observations INTEGER,
    #     num_features INTEGER,
    #     population_size INTEGER,
    #     operator_set TEXT,
    #     template_depth INTEGER,
    #     run INTEGER,
    #     seed UBIGINT,
    # );
    # """
    # )

    conn.execute(
        """
    CREATE TABLE IF NOT EXISTS results (
        algorithm TEXT,
        dataset TEXT,
        total_time_seconds DOUBLE,
        expression TEXT,
        mse DOUBLE,
        evaluations UBIGINT,
        fold INTEGER,
        num_observations INTEGER,
        num_features INTEGER,
        population_size INTEGER,
        operator_set TEXT,
        template_depth INTEGER,
        run INTEGER,
        seed UBIGINT,
    );
    """
    )

    for csv_file in tqdm(
        glob.glob(f"{dir}/**/*.csv", recursive=True), leave=False, ascii=True
    ):
        _, _, algorithm, dataset = csv_file.split(os.sep)

        # conn.execute(
        #     f"""
        # INSERT INTO results
        # SELECT
        #     '{algorithm}' AS algorithm,
        #     '{dataset.split(".")[0]}' AS dataset,
        #     total_time_seconds,
        #     expression,
        #     mse,
        #     evaluations,
        #     fold,
        #     num_observations,
        #     num_features,
        #     population_size,
        #     operator_set,
        #     template_depth,
        #     iteration AS run,
        #     seed,
        # FROM read_csv_auto('{csv_file}')
        # """
        # )

        conn.execute(
            f"""
        INSERT INTO results
        SELECT
            '{algorithm}' AS algorithm,
            '{dataset.split(".")[0]}' AS dataset,
            total_time_seconds,
            expression,
            mse,
            evaluations,
            fold,
            num_observations,
            num_features,
            population_size,
            operator_set,
            template_depth,
            iteration AS run,
            seed,
        FROM read_csv_auto('{csv_file}')
        """
        )

    for db_file in tqdm(
        glob.glob(f"{dir}/**/*.duckdb", recursive=True), leave=False, ascii=True
    ):
        stuff = db_file.split(os.sep)

        if len(stuff) != 4:
            continue

        _, _, algorithm, dataset = stuff

        conn.execute(f"ATTACH '{db_file}' AS src")

        # conn.execute(f"""
        #     INSERT INTO results
        #     SELECT
        #         '{algorithm}' AS algorithm,
        #         '{dataset.split(".")[0]}' AS dataset,
        #         total_time_seconds,
        #         expressions AS expression,
        #         mse_train AS mse,
        #         evaluations,
        #         fold,
        #         num_observations,
        #         num_features,
        #         population_size,
        #         operator_set,
        #         template_depth,
        #         run,
        #         seed,
        #     FROM src.results
        # """)

        conn.execute(f"""
            INSERT INTO results
            SELECT
                '{algorithm}' AS algorithm,
                '{dataset.split(".")[0]}' AS dataset,
                total_time_seconds,
                expressions AS expression,
                mse_train AS mse,
                evaluations,
                fold,
                num_observations,
                num_features,
                population_size,
                operator_set,
                template_depth,
                run,
                seed,
            FROM src.results
        """)

        conn.execute("DETACH src")


def create_db_experiment_1(dir: pathlib.Path):
    print("Starting experiment_1 db creation...")

    db_path = dir / "all_results.duckdb"

    if os.path.exists(db_path):
        print("Removed previous database")
        os.remove(db_path)

    conn = duckdb.connect(db_path)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS results (
        algorithm TEXT,
        run INTEGER,
        mse DOUBLE,
    );
    """)

    # Old version: take the final-generation result per run
    csv_file = dir / "old_cpu_version.csv"
    conn.execute(f"""
        INSERT INTO results
        SELECT
            'GP-GOMEA (original)' AS algorithm,
            run,
            mse_train AS mse,
        FROM (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY run ORDER BY generation DESC) AS rn
            FROM read_csv_auto('{csv_file}')
        )
        WHERE rn = 1
    """)

    # New version: already stores the final result per run
    db_file = dir / "new_cpu_version_depth_6.duckdb"
    conn.execute(f"ATTACH '{db_file}' AS src")
    conn.execute("""
        INSERT INTO results
        SELECT
            'GP-GOMEA (new)' AS algorithm,
            run,
            mse_train AS mse,
        FROM src.results
    """)
    conn.execute("DETACH src")


if __name__ == "__main__":
    create_db(pathlib.Path("results/experiment_2"))
    create_db_experiment_1(pathlib.Path("results/experiment_1"))
