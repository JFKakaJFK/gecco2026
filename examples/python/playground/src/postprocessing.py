import os
import pathlib
import shutil
from typing import Iterable

import duckdb
import numpy as np


def load_results(
    result_dirs: Iterable[pathlib.Path | str] | pathlib.Path | str,
    types: dict | None = None,
    parquet_dir: pathlib.Path | str | None = None,
    preprocess: bool = False,
    partition_cols: list[str] = ("method_name", "problem_name"),
) -> duckdb.DuckDBPyConnection:
    """Returns a view over all results, you should close the database."""
    if isinstance(result_dirs, (str, pathlib.Path)):
        result_dirs = [result_dirs]

    if types is None:
        types = dict()

    default_types = dict(
        status="VARCHAR",
        evaluations="UBIGINT",
        generation="UBIGINT",
        total_time_seconds="DOUBLE",
        alg_time_seconds="DOUBLE",
        eval_time_seconds="DOUBLE",
        seed="UBIGINT",
        discrete="USMALLINT[]",
        discrete_active="BOOL[]",
        continuous="DOUBLE[]",
        continuous_active="BOOL[]",
        objectives="DOUBLE[]",
        constraint_value="DOUBLE",
    )

    # sample first file and check if columns are not present to  avoid the binder exception
    columns_checked = False
    for d in result_dirs:
        dp = pathlib.Path(d)
        for fn in dp.rglob("*.csv"):
            with open(fn, "r") as f:
                try:
                    columns = f.readline().split(",")
                    for col in columns:
                        if col in default_types:
                            if col not in types:
                                types[col] = default_types[col]
                    columns_checked = True
                except Exception:
                    pass
            if columns_checked:
                break
        if columns_checked:
            break

    conn = duckdb.connect(":memory:")
    if parquet_dir is None:
        conn.sql(
            f"""
            SET preserve_insertion_order=false;
            CREATE OR REPLACE VIEW results AS SELECT * FROM
                read_csv(
                    {str([f"{d}/**/*.csv" for d in result_dirs])}, union_by_name=true,
                    types={{{", ".join(f"'{k}': '{v}'" for k, v in types.items())}}}
                )
            """
        )

    if parquet_dir is not None:
        pdir = pathlib.Path(parquet_dir)
        if preprocess or not pdir.exists():
            if pdir.exists():
                shutil.rmtree(parquet_dir)
            conn.sql(f"""
                COPY (
                    SELECT * FROM
                        read_csv(
                            {str([f"{d}/**/*.csv" for d in result_dirs])}, union_by_name=true,
                            types={{{", ".join(f"'{k}': '{v}'" for k, v in types.items())}}}
                        )
                    )
                TO '{str(parquet_dir)}' (FORMAT parquet, PARTITION_BY ({", ".join(partition_cols)}), OVERWRITE_OR_IGNORE, COMPRESSION zstd, COMPRESSION_LEVEL 1)
            """)
        hive_types = (
            f""", hive_types={{{", ".join(f"'{k}': '{types[k]}'" for k in partition_cols)}}}"""
            if all(k in types for k in partition_cols)
            else ""
        )
        conn.sql(
            f"""
            CREATE OR REPLACE VIEW results AS SELECT * FROM
                read_parquet(
                    '{f"{str(parquet_dir)}/**/*.parquet"}',
                    union_by_name=true,
                    hive_partitioning=true{hive_types}
                )
            """
        )
    return conn


def rliable_score_dict(
    conn: duckdb.DuckDBPyConnection, run_expr: str, normalized_value_expr: str
) -> tuple[dict, list[str]]:
    """Score matrices for each method, each of which is of size `(num_runs (runs + folds) x num_problems)` and the corresponding problem names"""
    normalized_score_dict = dict()
    problems = None

    for method, *_ in conn.sql(
        "SELECT DISTINCT(method_name) AS method FROM results"
    ).fetchall():
        # get final results for each run and problem
        df = (
            conn.execute(
                f"""
            SELECT {run_expr} AS run, problem_name AS problem, {normalized_value_expr} AS value FROM results
            WHERE status != 'Running' AND method_name = $1
        """,
                [method],
            )
            .df()
            .pivot(index="run", values="value", columns="problem")
        )

        if problems is None:
            problems = df.columns.tolist()
        else:
            assert len(df.columns) == len(problems)
        normalized_score_dict[method] = df.values

    assert problems is not None
    return normalized_score_dict, problems


def rliable_convergence_score_dict(
    conn: duckdb.DuckDBPyConnection, run_expr: str, normalized_value_expr: str
) -> tuple[dict, list[str], list[int]]:
    """
    Score matrices for each method, each of which is of size `(num_runs (runs + folds) x num_problems x max(generations))`, the corresponding problem names and the generations.

    If a run finished before `max(generations)`, the data for the remaining generations are repeats of the last generation
    """
    normalized_score_dict = dict()
    problems = None

    generations = [
        g
        for g, *_ in conn.execute("""
        SELECT DISTINCT(generation) AS generation FROM results
        ORDER BY generation
    """).fetchall()
    ]
    for method, *_ in conn.sql(
        "SELECT DISTINCT(method_name) AS method FROM results"
    ).fetchall():
        data = None
        for i, g in enumerate(generations):
            df = (
                conn.execute(
                    f"""
                SELECT {run_expr} AS run, problem_name AS problem, {normalized_value_expr} AS value FROM results
                WHERE generation = $2 AND method_name = $1
            """,
                    [method, g],
                )
                .df()
                .pivot(index="run", values="value", columns="problem")
            )

            if df.empty:
                assert data is not None
                data[:, :, i] = data[:, :, i - 1]

            if g == 0:
                if problems is None:
                    problems = df.columns.tolist()
                else:
                    assert len(df.columns) == len(problems)

            if data is None:
                data = np.empty((*df.values.shape, len(generations)))
            data[:, :, i] = df.values
        normalized_score_dict[method] = data

    assert problems is not None
    return normalized_score_dict, problems, generations
