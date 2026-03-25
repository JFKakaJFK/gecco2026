# Evolutionary Van Gogh

<p align="center">
    <img src="results/SimpleGA/evolution.gif" width="50%" height="auto" />
</p>

The provided project has both Python and C++ parts, where the algorithm is implemented C++ but experiments are run from Python. While this is the recommended setup, feel free to change this in any way you want (as long as your supervisor allows!).


# learning objectives
# - work with a C++ based python library
# - reconstruct image, the point is to have an optimization problem with interesting properties that is fun, not efficiency
#   - no changing the fitness function beyond the provided variables ()
# - conditionally inactive variables
# - linkage
# - mo

# i.e. you know what each decision variable represents and how variables interact, but assume the image that is being reconstructed is an unknown black-box

# deliverables: presentation & code, containing a total of 3 hypotheses you come up, implement and test
# - initialisation & variation: 2 hypotheses
# - mo: 1 hypothesis
# - proper experiments, >= 30 repeats, parameter tuning & statistical testing required (if you think that does not apply to your ideas, get the supervisor to agree on that!)


## Getting started

First install [`uv`](https://docs.astral.sh/uv/#installation), then run

```bash
# installs the dependencies and then runs the example
uv run main.py
```

See the root readme for instructions on how to modify the C++ code. Generally, after modifying the C++ code you have to regenerate the Python bindings and recompile. These steps should automatically make your C++ code available from Python, for example:

```C++
// lib/include/goblin/examples/voronoi.h
class YourCustomCrossover : public goblin::classic::DiscreteCrossoverBase {
  bool crossover(Rng& rng,
                 InstanceBase& problem,
                 const SolutionBase& donor,
                 SolutionBase& offspring) const override final {
    // do whatever you want here
    return goblin::classic::UniformCrossover().crossover(rng, problem, donor, offspring);
  }
};
```

turns into

```python
from pygom.voronoi import YourCustomCrossover
```

To make this possible, so called bindings need to be generated for the C++ code which define the provided interface and handle converting between Python and C++ types:

```bash
# generates and builds the python bindings for the C++ code
make bindings
```
