import glob
import os
import pathlib

import duckdb
from tqdm import tqdm

DATASETS = ["daily_demand", "auto_mpg", "california_housing", "feynman"]


def create_db_experiment_1(dir: pathlib.Path):
    db_path = dir / "all_results.duckdb"

    if os.path.exists(db_path):
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


def create_db_experiment_2(dir: pathlib.Path):
    db_path = dir / "all_results.duckdb"

    if os.path.exists(db_path):
        os.remove(db_path)

    conn = duckdb.connect(db_path)

    conn.execute(
        """
    CREATE TABLE IF NOT EXISTS results (
        algorithm TEXT,
        dataset TEXT,
        total_time_seconds DOUBLE,
        expression TEXT,
        mse DOUBLE,
        mse_val DOUBLE,
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

    for csv_file in [
        f
        for f in glob.glob(f"{dir}/**/*.csv", recursive=True)
        if "backup" not in f.split(os.sep)
    ]:
        _, _, algorithm, dataset = csv_file.split(os.sep)

        conn.execute(
            f"""
        INSERT INTO results
        SELECT
            '{algorithm}' AS algorithm,
            '{dataset.split(".")[0]}' AS dataset,
            total_time_seconds,
            expression,
            mse,
            mse_val,
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

    for db_file in [
        f
        for f in glob.glob(f"{dir}/**/*.duckdb", recursive=True)
        if "backup" not in f.split(os.sep)
    ]:
        file = db_file.split(os.sep)

        algorithm = file[-2]
        dataset = file[-1].split(".")[0]

        if dataset not in DATASETS:
            print(f"Skipping: {file}")
            continue

        conn.execute(f"ATTACH '{db_file}' AS src")

        mse_col = "mse" if "gpu" in algorithm else "mse_train"

        conn.execute(f"""
            INSERT INTO results
            SELECT
                '{algorithm}' AS algorithm,
                '{dataset.split(".")[0]}' AS dataset,
                total_time_seconds,
                expressions AS expression,
                {mse_col} AS mse,
                mse_val,
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


def create_db_experiment_3(dir: pathlib.Path):
    db_path = dir / "all_results.duckdb"

    if os.path.exists(db_path):
        os.remove(db_path)

    conn = duckdb.connect(db_path)

    conn.execute("""
    CREATE TABLE IF NOT EXISTS results (
        log_config TEXT,
        dataset TEXT,
        total_time_seconds DOUBLE,
        old_mse DOUBLE,
        mse DOUBLE,
        mse_val DOUBLE,
        var_y DOUBLE,
        var_y_val DOUBLE,
        fold INTEGER,
        template_depth INTEGER,
        population_size INTEGER,
    );
    """)

    for db_file in [
        f
        for f in glob.glob(f"{dir}/**/*.duckdb", recursive=True)
        if "backup" not in f.split(os.sep)
    ]:
        parts = db_file.split(os.sep)
        log_config = parts[-2]
        dataset = parts[-1].split(".")[0]

        conn.execute(f"ATTACH '{db_file}' AS src")

        conn.execute(f"""
            INSERT INTO results
            SELECT
                '{log_config}' AS log_config,
                '{dataset}' AS dataset,
                total_time_seconds,
                old_mse,
                mse,
                mse_val,
                var_y,
                var_y_val,
                fold,
                template_depth,
                population_size,
            FROM src.results
            WHERE old_mse IS NOT NULL
        """)

        conn.execute("DETACH src")

    conn.execute("ALTER TABLE results ADD COLUMN nmse DOUBLE")
    conn.execute("UPDATE results SET nmse = old_mse / var_y")
    conn.execute("ALTER TABLE results ADD COLUMN nmse_final DOUBLE")
    conn.execute("UPDATE results SET nmse_final = mse / var_y WHERE mse IS NOT NULL")
    conn.execute("ALTER TABLE results ADD COLUMN nmse_val DOUBLE")
    conn.execute(
        "UPDATE results SET nmse_val = mse_val / var_y_val WHERE mse_val IS NOT NULL"
    )


def create_db_experiment_4(dir: pathlib.Path):
    db_path = dir / "all_results.duckdb"

    if os.path.exists(db_path):
        os.remove(db_path)

    conn = duckdb.connect(db_path)

    conn.execute(
        """
    CREATE TABLE IF NOT EXISTS results (
        device TEXT,
        dataset TEXT,
        total_time_seconds DOUBLE,
        expression TEXT,
        mse DOUBLE,
        mse_val DOUBLE,
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

    for db_file in [
        f
        for f in glob.glob(f"{dir}/**/*.duckdb", recursive=True)
        if "backup" not in f.split(os.sep) and "all_results" not in f
    ]:
        parts = db_file.split(os.sep)
        device_folder = parts[-2]  # "cpu_results" or "gpu_results"
        device = "gpu" if "gpu" in device_folder else "cpu"
        mse_col = "mse" if device == "gpu" else "mse_train"

        conn.execute(f"ATTACH '{db_file}' AS src")

        src_cols = {r[0] for r in conn.execute("DESCRIBE src.results").fetchall()}
        mse_val_expr = "mse_val" if "mse_val" in src_cols else "NULL::DOUBLE AS mse_val"

        conn.execute(f"""
            INSERT INTO results
            SELECT
                '{device}' AS device,
                dataset,
                total_time_seconds,
                expressions AS expression,
                {mse_col} AS mse,
                {mse_val_expr},
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


def create_db_experiment_5(dir: pathlib.Path):
    db_path = dir / "all_results.duckdb"

    if os.path.exists(db_path):
        os.remove(db_path)

    conn = duckdb.connect(db_path)

    conn.execute(
        """
    CREATE TABLE IF NOT EXISTS results (
        kernel TEXT,
        dataset TEXT,
        total_time_seconds DOUBLE,
        expression TEXT,
        mse DOUBLE,
        mse_val DOUBLE,
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

    for db_file in [f for f in glob.glob(f"{dir}/*.duckdb") if "all_results" not in f]:
        dataset = os.path.basename(db_file).split(".")[0]

        if dataset not in DATASETS:
            print(f"Skipping: {db_file}")
            continue

        conn.execute(f"ATTACH '{db_file}' AS src")

        conn.execute(f"""
            INSERT INTO results
            SELECT
                kernel,
                '{dataset}' AS dataset,
                total_time_seconds,
                expressions AS expression,
                mse,
                mse_val,
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


def create_db_experiment_6(dir: pathlib.Path):
    db_path = dir / "all_results.duckdb"

    if os.path.exists(db_path):
        os.remove(db_path)

    conn = duckdb.connect(db_path)

    conn.execute(
        """
    CREATE TABLE IF NOT EXISTS results (
        algorithm TEXT,
        cutoff TEXT,
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

    for db_file in [
        f
        for f in glob.glob(f"{dir}/**/*.duckdb", recursive=True)
        if "backup" not in f.split(os.sep)
    ]:
        file = db_file.split(os.sep)

        algorithm = file[-3]
        cutoff = file[-2]
        dataset = file[-1].split(".")[0]  # Remove .duckdb

        if dataset not in DATASETS:
            print(f"Skipping: {file}")
            continue

        conn.execute(f"ATTACH '{db_file}' AS src")

        mse_col = "mse" if "gpu" in algorithm else "mse_train"

        conn.execute(f"""
            INSERT INTO results
            SELECT
                '{algorithm}' AS algorithm,
                '{cutoff}' AS cutoff,
                '{dataset}' AS dataset,
                total_time_seconds,
                expressions AS expression,
                {mse_col} AS mse,
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
