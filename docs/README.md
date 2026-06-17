# GOMEA Library Knowledge Base

> [!INFO]
> Note: This knowledge base explains design decisions and how to work with and contribute to this library. For using the library, separate [Python documentation](TODO make pydocs) and [C++ documentation](TODO make docs) are available.

Overall, the scope of this project is to be a good starting point for internal and external users. For that, it should be reasonable easy to use and feature complete enough to encompass a wide range of use cases, from algorithm research and benchmarking to use cases such as survival analysis. The motivation for having one joint library for this is that with a bit of planning and collaboration, it should be possible to both get our work done more easily while at the same time maintaining a nice library. Ideally, the library shortens the time until running the first experiment.

TODO @JFKakaJFK architecture - instead of some big document explaining details, have a figure of the different components (InstanceBase, MethodBase, SolutionBase, QualityBase, FitnessBase, ArchiveBase, SolutionSetBase) and explain how they go together

- [Design philosophy](./design_philosophy.md)
- [Architecture](TODO @JFKakaJFK main interfaces & how the interactions should work)
- [Development Process](./development_process.md)
- [Working with Git](./working_with_git.md)
- [Getting started](./getting_started.md)
- [Working with CMake](TODO effective modern cmake + our cmake setup)
- [Wrapping Existing Code](TODO @JFKakaJFK how to for C++ and Python)
- [Automatic Python Bindings with litgen](pylib/README.md)
- [Writing Tests](./writing_tests.md)
- [Writing Documentation](./writing_documentation.md)
- [Debugging and Performance Optimization](TODO how to run py/c++/mixed debuggers, bisection, reducing issues to minimum viable reproductions; amdahls law, profiling with samply, memory profiling with e.g. massif, timing with hyperfine/timeit, C++ data structure footguns, do less work ~> approximate work ~> do it faster, parallelism (OpenMP/Taskflow/TBB), GPU accelleration (CUDA/Metal/OpenGL), LAPACK/BLAS/BLIS)
- [Publishing and Archiving Code](TODO @8uurg)
- [Technical Debt](./technical_debt.md)

## Frequently Asked Questions

> Who are these individuals?

## GOMEA in the Wild

Nothing to see here. Come back later?

## Publications

We are standing on the shoulders of giants. Or in this case, ideas and code from various publications are implemented or helped shape the library:

- TODO add all publications we build on, from within our group but also utils like Eigen, litgen, OpenRAND...
