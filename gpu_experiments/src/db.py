import duckdb
import pathlib
import re


def create_db(dir: pathlib.Path):
    db_path = dir / "experiments.duckdb"

    conn = duckdb.connect(db_path)

    # Create table
    conn.execute(
        """
    CREATE TABLE IF NOT EXISTS results (
        dataset TEXT,
        device TEXT,
        population_size INTEGER,
        num_observations INTEGER,
        fold INTEGER,
        iter INTEGER,
        evaluations INTEGER,
        generation INTEGER,
        total_time_seconds DOUBLE,
        alg_time_seconds DOUBLE,
        eval_time_seconds DOUBLE,
        seed BIGINT,
        expressions TEXT,
        mse_train DOUBLE,
    );
    """
    )

    filename_re = re.compile(
        r"""
        (?P<dataset>.+)
        -
        (?P<device>[^-]+)
        -pop(?P<pop>\d+)
        -obs(?P<obs>\d+)
        -fold(?P<fold>\d+)
        -iter(?P<iter>\d+)
        \.csv
        """,
        re.VERBOSE,
    )

    for csv_path in dir.glob("*.csv"):
        match = filename_re.match(csv_path.name)
        if not match:
            print(f"Skipping unrecognized file: {csv_path.name}")
            continue

        meta = match.groupdict()

        conn.execute(
            f"""
        INSERT INTO results
        SELECT
            '{meta["dataset"]}' AS dataset,
            '{meta["device"]}' AS device,
            {meta["pop"]} AS population_size,
            {meta["obs"]} AS num_observations,
            {meta["fold"]} AS fold,
            {meta["iter"]} AS iter,
            evaluations,
            generation,
            total_time_seconds,
            alg_time_seconds,
            eval_time_seconds,
            seed,
            expressions,
            mse_train,
        FROM read_csv_auto('{csv_path.as_posix()}')
        WHERE status = 'Converged'
        """
        )

    conn.close()
