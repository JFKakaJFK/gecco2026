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
        glob.glob("results/**/*.csv", recursive=True), leave=False, ascii=True
    ):
        _, algorithm, dataset = csv_file.split(os.sep)

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
        glob.glob("results/**/*.duckdb", recursive=True), leave=False, ascii=True
    ):
        stuff = db_file.split(os.sep)

        if len(stuff) != 3:
            continue

        _, algorithm, dataset = stuff

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


if __name__ == "__main__":
    create_db(pathlib.Path("results"))
