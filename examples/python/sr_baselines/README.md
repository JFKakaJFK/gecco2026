# SR Baseline results

To see if the SR performance is in line with GP-GOMEA, gpg, and MultiGPG, I set up a small experiment that uses the datasets from the [2025 SRBench paper](https://arxiv.org/abs/2505.03977v1) that compares the performance based on an evaluation limit.

![](./plots/100000evals.png)

![](./plots/500000evals.png)

For the other versions, the results are based on the scripts in `scripts` are provided and the code to run the version in this repository is in the `goblin` directory:

```bash
# in /examples/python/sr_baselines/goblin
uv run run.py # updates the results for this version, overrides old results and can take a while

uv run plot.py # updates the comparison plots
```

There are multiple GP-GOMEA implementations, that all differ in some aspects:

- [x] original version: https://github.com/marcovirgolin/GP-GOMEA
  - [x] latest branch of that version: https://ir.cwi.nl/pub/34425/paper_115.pdf
- [x] second version: https://github.com/marcovirgolin/gpg
  - [ ] SRBench 2025 submission: https://github.com/matigekunstintelligentie/MultiGPG

To ensure this version is not a performance regression, the different versions are compared on problems from the latest SRBench iteration. Due to Apple using ARM chips and conda not having wheels for some packages used by some of the versions, not all experiments were run on the same machine and slight modifications of the source code had to be made to support the experiment settings used:

- Linear scaling: Yes | No
- Evaluation budget: 100k, 500k - longer tests are needed here for anything involving constant optimization
- Operator set: +,-,*,/,sin
- Initialization: Half-Half (50% Grow, 50% Full)
- Population size: 1024, no IMS
- Linkage: LinkageTree without the full subset, Marco's linkage normalization (or intron awareness)
- Constants: 25 bins (or 100 in the original versions) and no constant optimization (well mostly...)
- Forced improvements: Yes (but the gpg line of versions has a differences from GOMEA here)

This still leaves a few differences between the versions, but setting this up completely fair would require quite a few source code modifications too many - the scripts in `scripts` are already hacky enough.

# TODOs

- [ ] longer experiment (2e6 evals) for variants with constant optimization
- [ ] Don't just silently ignore nan's
