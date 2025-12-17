import duckdb 
from pathlib import Path

def create_db(results_dir: Path):
    db_path = results_dir / "experiments.duckdb"

    types = dict(
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

    conn = duckdb.connect(db_path)

    conn.execute("SET preserve_insertion_order=false;")

    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS results AS
        SELECT *
        FROM read_csv(
            '{results_dir}/*.csv',
            union_by_name=true,
            types={{{", ".join(f"'{k}': '{v}'" for k, v in types.items())}}}
        );
    """)

    conn.close()