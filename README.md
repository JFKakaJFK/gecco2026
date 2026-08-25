# Introns and Templates Matter: Rethinking Linkage in GP-GOMEA

This repository contains the source code for the paper ["Introns and Templates Matter: Rethinking Linkage in GP-GOMEA"](http://dx.doi.org/10.1145/3795095.3805139) ([arXiv preprint](https://arxiv.org/abs/2602.02311)) accepted at GECCO 2026.

> [!WARNING]
> This is the code for the paper only, a more user-friendly and complete (GP-)GOMEA library containing the contributions of this repository is currently under development.

## Installation

This project supports Linux and macOS and requires a recent C++ compiler with C++23 support. With the toolchain in place, the Python bindings can be installed using:

```bash
pip install git+https://github.com/JFKakaJFK/gecco2026.git@main#egg=pygom
```

### Installation From Source

To build from source, the Python package manager [`uv`](https://docs.astral.sh/uv/getting-started/installation/) is additionally required:

```bash
# get the code
git clone --branch main --depth 1 --single-branch github.com/JFKakaJFK/gecco2026.git
cd gecco2026
# autogenerate C++ bindings and install the python package
make bindings
```

Note that building from source is only recommended if the underlying C++ code is changed as well.

## Usage

Once installed, the Python bindings in the `pygom` package provide a SKLearn compatible Python interface for symbolic regression:

```python
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
```

## Paper Experiments & Results

The experiment source code for the paper and thesis can be found at `examples/python/revisiting_linkage`.

## License

This work is licensed under CC BY-NC-ND 4.0. To view a copy of this license, visit https://creativecommons.org/licenses/by-nc-nd/4.0/

## Citation

If you find this work useful, please cite the published version or the arXiv preprint:

```
@inproceedings{Koch2026,
  series = {GECCO ’26},
  title = {Introns and Templates Matter: Rethinking Linkage in GP-GOMEA},
  url = {http://dx.doi.org/10.1145/3795095.3805139},
  DOI = {10.1145/3795095.3805139},
  booktitle = {Proceedings of the Genetic and Evolutionary Computation Conference},
  publisher = {ACM},
  author = {Koch,  Johannes and Alderliesten,  Tanja and Bosman,  Peter},
  year = {2026},
  month = July,
  pages = {780–789},
  collection = {GECCO ’26}
}

@misc{https://doi.org/10.48550/arxiv.2602.02311,
  doi = {10.48550/ARXIV.2602.02311},
  url = {https://arxiv.org/abs/2602.02311},
  author = {Koch,  Johannes and Alderliesten,  Tanja and Bosman,  Peter A. N.},
  keywords = {Neural and Evolutionary Computing (cs.NE),  FOS: Computer and information sciences,  FOS: Computer and information sciences},
  title = {Introns and Templates Matter: Rethinking Linkage in GP-GOMEA},
  publisher = {arXiv},
  year = {2026},
  copyright = {Creative Commons Attribution Non Commercial No Derivatives 4.0 International}
}
```
