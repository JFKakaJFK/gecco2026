import numpy as np
from pygom import *


# a bit awkward, but I haven't made a more pythonic version to do this yet so you can (ab)use the PyFunctionBase + BenchmarkInstance combination to specify the objective function + problem domain
class PyProblem(PyFunctionBase):
    def __init__(self, dims: int) -> None:
        super().__init__()

        self.dims = dims

    def num_objectives(self) -> int:  # overridable (pure virtual)
        return 3

    def num_discrete(self) -> int:  # overridable (pure virtual)
        return self.dims

    def num_continuous(self) -> int:  # overridable (pure virtual)
        return 0  # no continuous variables

    def eval(self, solution: SolutionBase):  # overridable (pure virtual)
        """Evaluates and updates the solution fitness ('quality') in-place"""

        values_np = solution.discrete_values()
        # print(type(values_np)) # -> <class 'numpy.ndarray'>

        # then do whatever (either accessing single variables or the whole thing as np.ndarray)
        zero_max = sum(
            solution.discrete_values()[i] == 0 for i in range(self.num_discrete())
        )
        one_max = np.sum(values_np == 1)
        two_max = np.sum(values_np == 2)

        objectives = np.array([zero_max, one_max, two_max], dtype=np.float64)
        constraint_value = 1.0 if (values_np > 2).any() else 0.0

        return objectives, constraint_value


if __name__ == "__main__":
    problem = BenchmarkInstance(
        objectives=PyProblem(3),
        # either one number corresponding to the domain size (will use values [0,domain_size-1], you need to map that to whatever options you have)
        discrete_domain=4,
    )

    # Effectively a discrete version of https://doi.org/10.1016/j.swevo.2018.02.005
    gomea = MixedGOMEA(
        ims_options=IMSOptions(
            restart_stale_populations=True,  # restarts converged populations
            max_num_populations=1,  # for whatever reason using IMS throws a segmentation fault, so don't change this and ignore the warnings from nanobind ;)
            initial_population_size=50,
            initial_num_clusters=problem.num_objectives() + 1,
        ),
        discrete_model=LinkageTreeFOS(),
    )

    # all the standard things should work, you can also have a custom python callback as termination criterion
    budget = Budget(max_evaluations=10_000, max_generations=100, max_time_seconds=10)

    # or just archive, status = gomea.run(problem, budget=budget, seed=42)
    archive, status = Tracked.run(
        problem,
        gomea,
        budget=budget,
        config=TrackingOptions(
            logpath="build/stats.csv"  # this needs to be a string, not a pathlib.Path
            # you can use this to
            # - add run specific information (e.g. a method name for comparing different methods, or problem dimensionality or whatever)
            # - define when you want to log (e.g. every generation, every 1000 evaluations, or time-based)
        ),
        seed=42,
    )
    print(status)

    # turns out currently restarting/interleaving isn't that easy, but you can hack it together in the following ways:
    # Option A (recommended): make a custom initialization method for the 'BenchmarkInstance' that initializes e.g. based on the archive you got and use that for the second call to gomea and onwards
    # Option B: similar to A, but with the evaluation function and a fixed population size - the evaluation function should have write access to solutions, so you can just override the first population size solutions you get with whatever you want...

    print(f"Archive contains {archive.size()} solutions")
    for i in range(archive.size()):
        s = archive[i]
        print(
            f"{i}: {s.discrete_values()} @ {s.quality().objectives}/{s.quality().constraint_value}"
        )

    # for hypervolumes (not implemented yet) I'd recommend https://esa.github.io/pygmo2/tutorials/hypervolume.html (needs conda though)

    # somehow the fact that the C++ code (benchmarkinstance) holds a reference to a python object (PyProblem) creates a reference cycle that requires manual deletion
    # notably the issue only appears if a C++ object owns (a reference) to the python object, just calling with a callback is fine
    del problem
