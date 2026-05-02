import pathlib
import re

import duckdb
from tqdm import tqdm


def create_db(dir: pathlib.Path):
    print("Starting db creation...")

    db_path = dir / "experiments.duckdb"

    conn = duckdb.connect(db_path)

    # Create table
    conn.execute(
        """
    CREATE TABLE IF NOT EXISTS results (
        status TEXT,
        evaluations UBIGINT,
        generation UBIGINT,
        total_time_seconds DOUBLE,
        alg_time_seconds DOUBLE,
        eval_time_seconds DOUBLE,
        expressions TEXT,
        unresolved_expressions TEXT,
        subtrees TEXT[],
        mse DOUBLE,
        old_mse DOUBLE,
        dataset TEXT,
        fold INTEGER,
        num_observations INTEGER,
        num_features INTEGER,
        population_size INTEGER,
        operator_set TEXT,
        template_depth INTEGER,
        run INTEGER,
        kernel TEXT,
        seed UBIGINT,
        discrete USMALLINT[],
        discrete_active BOOL[],
        continuous DOUBLE[],
        continuous_active BOOL[],
        objectives DOUBLE[],
        constraint_value DOUBLE,
        var_y DOUBLE,
    );
    """
    )

    filename_re = re.compile(
        r"""
        (?P<dataset>[^-]+)
        -
        (?P<device>[^-]+)
        -pop(?P<pop>\d+)
        -obs(?P<obs>\d+)
        -feat(?P<feat>\d+)
        -depth(?P<depth>\d+)
        -op_(?P<op_set>[^-]+)
        -fold(?P<fold>\d+)
        -iter(?P<iter>\d+)
        \.csv
        """,
        re.VERBOSE,
    )

    subtree_col_re = re.compile(r"^subtree_(\d+)$")

    for csv_path in tqdm(dir.rglob("*.csv"), leave=False, ascii=True):
        match = filename_re.match(csv_path.name)
        if not match:
            print(f"Skipping unrecognized file: {csv_path.name}")
            continue

        col_names = [
            d[0]
            for d in conn.execute(
                f"SELECT * FROM read_csv_auto('{csv_path.as_posix()}') LIMIT 0"
            ).description
        ]

        subtree_cols = sorted(
            [c for c in col_names if subtree_col_re.match(c)],
            key=lambda c: int(subtree_col_re.match(c).group(1)),
        )
        subtrees_expr = f"[{', '.join(subtree_cols)}]" if subtree_cols else "[]::TEXT[]"
        unresolved_expr = (
            "unresolved_expressions"
            if "unresolved_expressions" in col_names
            else "NULL"
        )

        conn.execute(
            f"""
        INSERT INTO results
        SELECT
            status,
            evaluations,
            generation,
            total_time_seconds,
            alg_time_seconds,
            eval_time_seconds,
            expressions,
            {unresolved_expr} AS unresolved_expressions,
            {subtrees_expr} AS subtrees,
            mse,
            mse_train AS old_mse,
            dataset,
            fold,
            num_observations,
            num_features,
            population_size,
            operator_set,
            depth,
            iteration,
            kernel,
            seed,
            discrete,
            discrete_active,
            continuous,
            continuous_active,
            objectives,
            constraint_value,
            var_y
        FROM read_csv_auto('{csv_path.as_posix()}')
        """
        )

    conn.close()
