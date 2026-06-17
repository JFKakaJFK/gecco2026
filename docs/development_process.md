# Development Process

This document describes the current intended development and collaboration process. Overall, the idea is that it is sufficient for everyone to leave the code in a better state than it was before. In other words, the goal is to strike the right balance between making it easy to research and develop new features and long-term sustainability of the library.

## Collaboration structure

First, there exist two main versions of the code[^1] - a stable release branch and a working development branch. In addition, the actual work happens in forks:

- Stable main/release branch: This branch corresponds to the published version to use for third-party end-users or projects that do not directly involve library development. Published here means packages on PyPI and hosted documentation.
- Development branch: Tis is the branch students/contributors should target. Partial features are welcome and even encouraged to prevent large merge conflicts, however, the branch should still always be in a working state. When papers are published or new features are complete the stable release branch is updated from this branch.
- Working forks: Full copies of the repository owned by contributors, without any restrictions. This is the intended way to develop new features, fix bugs and so on.

## Process

The intended process looks as follows:

1. [Create a fork](./working_with_git.md#creating-a-fork)
2. Develop new feature(s)/fix bug(s)
3. [Pull and merge with the upstream development branch](./working_with_git.md#updating-a-fork) to ensure that the code is compatible with the development branch
4. Ensure the [continuous integration](./continuous_integration.md) tests pass
4. [Create a pull request (PR)](./working_with_git.md#creating-a-pull-request) 
5. Someone with the maintainer role has to accept the PR. 

## PR Merging Checklist

Accepting a PR means the maintainers accept responsibility for the added code. This might include fixing bugs later discovered or having to support the new feature at least until the next major version increment. Hence, the function of having PRs is to both make the responsibility handoff explicit and to prevent changes with high maintenance burdens. For each PR, a maintainer should check the following[^2]:

- [ ] New code does not break general assumptions/invariants. In other words, the proposed code needs to be idiomatic/fits into the library well enough from the perspective of maintainers such that they feel comfortable with maintaining the newly added code in the long-term! If this is not the case, the PR needs to be rejected/adapted _before_ it is merged into the development version. This should include:
  - [ ] Code intrusiveness: new features should affect the least code surface possible, in the most general way to avoid increasing the complexity of bug fixes and future feature developments
  - [ ] Code quality & style: maintainers need to feel comfortable about being able to follow and understand the code in case of future maintenance needs[^3][^4]
  - [ ] Code features used: new code should not make use of "bad parts of C++" unless necessary (e.g. raw pointers)[^5]
- [ ] Newly introduced code is appropriately tested. This should include unit tests (C++ and Python) and regression tests as appropriate.
- [ ] Newly introduced code is appropriately documented. This should include usage examples and references to preprints/publications if applicable. Check the documentation output for interface documentation, the code comments are not always picked up by all code generation tools used!
- [ ] The version increment is appropriate, see [versioning](#versioning).

Doing so should ensure that the long-term maintenance burden is feasible. Other than bug fixes or other code improvements such as e.g. refactoring to reduce technical debt, PRs should be complete and should not require immediate maintenance effort. After getting a PR accepted, PhD students become maintainers, at least for the duration that their work is related to the library.

## Code Releases

TODO describe the process & intention behind adding release tags to commits that are associated to e.g. code for papers or otherwise notable commits that don't require a deploy?

## Code Deploys

TODO we have no infrastructure as of now to create the python package using cibuildwheel or to host the documentation on e.g. readthedocs in place, so do that first and then document it...

## RFCs / Improvement Proposals

TODO The idea is to have some sort of process for maintainers to document and others to propose code changes (new features, support for X, refactors,...). RFCs are typically mini documents that include:

- some problem/goal
- why this matters
- how it could be achieved

For example:

- We all run computational experiments, hopefully now with effectively the same log output. To be more effective as a group, we should have the basic building blocks in place to run, process and analyze experiments
- The code for this tends to grow over the thesis/PhD project, often taking significant amounts of time per student. With a good starting point, this should give everyone days/weeks of time to work on the actual problem.
- A draft of how this could look like is at `examples/python/revisiting_linkage/src`

Ideally these documents are evolving until they are either implemented or rejected. Especially at the beginning, the function is to shelf and prioritize the many ideas we still have of what the library should ideally include. For student projects there is no need for RFCs - the project itself typically already has a problem statement with a similar function, but being able to write it down is always a good exercise.

## Quarterly Meetings

Collaboration requires communication, ideally before multiple people work on the same thing/break each others code. To this end, at least quarterly the people actively working on the library should meet, to synchronize and possibly discuss the following:

- Who's using the library, and for what? #GOMEA inside
- Announcing new projects and publications related to the library
- Discuss whatever comes up, open questions

Next proposed agenda (TODO note this down somewhere else):

- @JFKakaJFK: Design Philosophy
- How do we want to regression test locally (should be fast enough to run often on limited hardware) and for PRs (should be more thorough)?
  - what problems, settings, what is tested,...
- Utilities for running & analyzing experiments (especially if everyone gets the same log output it makes sense for us to make an utility library & document institutional knowledge on how to get access and use the available compute resources)
- This should still be research code for the most part - all limitations/quality check processes should have less cost than starting from an alternative version => We should sometimes ask if we are getting this balance right...
- RFCs & focus until next QM
  - feature request/improvement triage - if maintainers have extra time, what should be worked on first?

## Versioning

The project should use [semantic versioning (`MAJOR.MINOR.PATCH`)](TODO) such that major versions can introduce larger breaking changes, minor versions include compatible developments and patch versions are mainly used for bug fixes.

[^1]: TODO should there be a third "archive" branch for code that should get a release tag but is not merged? E.g. if there is a feature/student project that is "done" but without a merge back to the main branch? Or is the archive on constellation that place?
[^2]: TODO other things? @8uurg @thaleaschlender @JFKakaJFK
[^3]: @JFKakaJFK do as I say and not as I do right? Right? Admittedly a lot of the code should be split up into smaller parts and have better documentation and so on...
[^4]: TODO better define "code quality & style" @8uurg @thaleaschlender @JFKakaJFK
[^5]: TODO better define what sort of code we don't want to allow @8uurg @thaleaschlender @JFKakaJFK
