# Continuous Integration (CI)

On the [release and development branches](TODO link to dev process section), every code change needs to pass the following conditions[^1]:

- The analysis/code style checks pass (i.e. tidy, fmt,...) lead to no code changes/warnings/errors
- The project builds without errors on all compilers we target (clang, gcc, MSVC? - several versions @8uurg)
- The C++ tests pass (no feature that worked before was broken)
- The python library builds fine
- The python tests (nonexistent) run fine
- The regression tests indicate no performance regressions
- The python documentation builds fine
- The C++ documentation builds fine

These tests are automated and also available locally via the `make ci`[^2] command. While running all steps takes a while, they should be run regularly also on working forks to ensure that the changes introduced do not break existing features.

[^1]: TODO extend the list & actually implement everything @8uurg / @JFKakaJFK
[^2]: TODO probably a shell command instead? After all, this should run with multiple compilers etc - the local version does not need to do everything, but instead should be a good trade-off between catching a lot and running fast enough to run it often...
