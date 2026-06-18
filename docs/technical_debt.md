# Technical Debt

Things that should be different, but are not - sometimes with good reasons, other times because it seemed like a good idea at the time. Whatever the reason, the current state is not the desired state and these gaps are documented here. Some of the issues here are likely good candidates for RFCs.

## Tests and Documentation

What the class does, each parameter and what goes on in the code on a high level (note on method top what is done? - I’m on the fence here; large methods clearly are hard to retrace and figure out what & why something is done but documentation generally isn’t always updated and tends to be out of date and it should not explain what (that’s what the code does), but why (so the things not in the code))

## `InstanceBase` evaluation API

Evaluate call should not take a set and indices but direct references/“eval items” (solution + parent + change set for partial evaluations) - having to construct a solution set just to enable batching across different solution origins seems wrong

## GP Context has too many functions

GP setup should split language from representation from evaluation from problems. I have a new design that should work for this, but chances are this will either be adopted incrementally or not at all (not wrong if it’s not needed!). Type/other checking can be added during post-order visits and IR can be emitted to unify different evaluation backends. Also custom evaluators (Cuda, metal,… might need custom code per problem so dispatching there might not be such a bad idea after all

## Premature optimisation

Lots of buffers are allocated once and re-used across methods in various classes - not a problem until you want to refactor the code and maybe more importantly it was never profiled if this actually helps (caches are only so big…)


## Performance

It looks like on different machines speed between the various versions varies - and clearly the whole flat genotype spiel does not pay off yet (testability wrt other domains & generalisation are nice though)

## Compile Times

Sure you can cache etc but clean builds should also be fast…

## Too much Eigen


The eigen library is convenient for math, but currently is in the main solution interface and everywhere. Sometimes the refs make optimisations impossible, other times it’s more of a “there’s no need to use Eigen intrinsics where plain C++ also does the job (span/mdspan) - don’t specialise if it’s not required is one of the ideas at the core of the rewrite after all to be general yet pragmatic”. Looking at the annoyingly faster operon this also is one of the things done better there

## Too many large functions!

The main Gomea implementation is a huge pile of technical debt. I don’t have a good model of how to fix it (to enable batching, we need to do things cluster-aware and in a way that splits variation, evaluation & acceptance -> lots of data dependencies on each and every component which makes it hard to disentangle). The one option I see is to make a “generation based gomea” in the sense that after every FOS step control is yielded back to the caller (and every so often a step includes an update of the linkage model) which would at least allow splitting the integrated gambit/mixed stuff from discrete gomea

## Naming things

No `*_values`, gomea solutions/parents are more offspring/backup etc; the main priority is that names are clear, then they should go well together with other names (cohesion) and finally they shouldn’t be excessively longer than required

## Experiment Running Utilities

Right now the experiment configuration utility is very much still evolving, and compared to Hyperopt, Optuna, OmniConf, hydra, config files (.yaml,.toml,.json,...), having config scripts I like the idea of "basically write an almost plain Python generator so that control flow is possible and the code to loop over experiment configs is close to the code needed to run a single experiment, and outputting human readable experiment configurations is nice for debugging purposes" - but, instead of a non-pyhton custom DSL in e.g. yaml the output should just be the plain python code in the first place...

## Experiments in the repository

It is very convenient and to some extent provides usage examples, but maybe also a metastasizing growth that should not be there?
