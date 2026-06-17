# Design Philosophy

The theoretical/technical reasoning behind some decisions. Mostly the reasoning is to impose as little restrictions as possible which leads to a very general yet abstract design, while defaults and optional escape hatches keep things practical where possible.

## Problem / algorithm separation

At the core is the problem instance & method design.
Instance, because the same problem might have different instances - could be different graphs for maxcut or folds for symbolic regression. Each instance has to provide its search space. For some problems, the initial starting point is also given, e.g. for BBOB - also, since algorithms don’t necessarily know the domain they might not know how to initialise the population so it is easier to provide that on the instance side.

For other algorithms, initialisation is part of the algorithm. Luckily, at the time of running a method on a problem instance all information about the instance and method are known - so while this might not always be nice, this separation between instance and method does not really prevent anything.

In other words, methods should be separated from the problem to be general, yet problem specific mechanisms should still be possible. The escape hatch for this are solution extensions - they add to the interface and are available on both sides of the method/instance boundary, allowing problem and method specialisations.

Since methods don’t know what problem they are solving[^1], they are not aware of special semantics - for example in GP with the ERC representation there is a 1:1 mapping between discrete and continuous variables and discrete crossover should also copy the corresponding continuous values. Either the method knows this and is specialised to the problem, or the problem provides possibly specialised behaviours (ie inherit discrete, add_random, gradient handling, solution logging/formatting, providing cache keys). Most of these behaviours come with reasonable defaults and only problems with special needs need to be aware of these escape hatches.

Beyond this, the problem interface aims to allow exploiting batched & partial evaluations, and the option to provide information about (conditionally) inactive variables. Notably, each evaluation must set which variables are active for the solution!

For behaviours that are optional and not tied to the problem instance or method, such as logging or caching, wrapping an instance is the go to way to insert functionality in-between. The wrapper needs to forward (and optionally override) ALL problem instance behaviour in case the actual, wrapped, instance overrides any defaults! Similarly, some special algorithm behaviour required testing if a problem has a certain type, requiring yet another escape hatch in the form of an unwrap method that yields the innermost problem instance. To make this bearable, a default wrapper is provided that can be extended. Updates to the instance interface require default forwarding in the default wrapper!

On the method side, the main requirement is that methods can be run on instances. The results are returned in the form of an archive, where the instance again defines what the archiving criteria are.

## Solutions, Quality and Fitness

Solutions are an important part of the instance/method separation - after all, solutions are information that flows across the interface boundary. At the same time as the general behaviour of providing access to decision variable values is clear, the actual underlying memory layout is not. For example, solutions could be stored as individual arrays of variables (array of structs) or as parts of larger matrices (structs of arrays). Again, methods should not have to concern themselves with this unless that is desired (depending on the memory access patterns of an algorithm one or the other memory layout might be more favourable). To enable this, automatic implicit (!) conversion between solution representations is supported, and a second type that represents sets of solutions is introduced. Each solution set implementation provides a list/dynamic array like interface and can freely choose how solutions are stored, with extensions being the escape hatch to enable attaching information beyond the standard solution interface.

The fitness of solutions is another piece of information that needs to cross the instance/method boundary. However, since the methods  mainly are EAs, they generally only need to know which of two solutions is preferable instead of exact numbers. To keep the advantage of only requiring a relative ordering over solutions, the typical fitness is split up into a solution quality and a preference relation defining a partial order and distance over qualities. In typical fashion, the design choice here is to not place unnecessary restrictions. The main requirement is that the fitness is compatible with the quality used. Different components such as archives and methods might use different fitnesses, as long as the same quality type is used. The fitness, and with it the quality are defined by the problem instance. This means that methods are not able to construct qualities since they do not know the concrete type. Sometimes that is necessary, for this the instance exposes the fitness which in turn allows creating and comparing quality instances.

Notably, the lack of absolute comparisons prevents some operations. For example, it is not possible to construct the mean quality, but qualities can be sorted and the median can be determined much in the same way as k-means and k-medians. Likewise, for almost all operations necessary, exactly equivalent or semantically close workarounds are possible.

## Practical Considerations

While the language of choice for the library is C++, there should also be Python bindings. With the various approaches to bindings, most require defining an intermediate representation of the C++/Python interface that strongly resembles the original C++ code. To avoid this code duplication, litgen is used to generate the bindings automatically. The rough way this works is that litgen parses the C++ headers and then emits Python bindings for C++ declarations. This comes with some [caveats since the parser used by litgen differs from the compiler](pylib/README.md) but it mostly just works.

To allow dynamically calling the library code from Python, we need to resort to runtime polymorphism (virtual classes) instead of compile time polymorphism (templates/concepts) in most cases. This comes with some overhead as indirections and vtables are unavoidable. Nonetheless this arguably is a pragmatic approach. To make this possible, C++ 17 is the first sufficient standard, however, since C++20/23 should be broadly supported soon and those standards do come with some conveniences (eg std::span, std::print) the minimum C++ version supported is C++23 as of now.

At this point, the code is split up into the core library (lib), benchmarking functionality (bench), special domains (eg gp) and methods. Generally library code must not depend on other code, benchmarking code only depends on library code, domain code should be method independent and method code can pull in any dependencies.

Testing methods generally is hard. Individual components and operators with clear behaviour can be tested individually, but to ensure the overall optimization performance is up to par the best way is to compare to a trusted baseline on several benchmarks. Practically, these tests need to be thorough enough to allow confidence in the implementation while at the same time they should run quickly to allow fast iteration.

Ideally formatting, linting, regression tests and all examples are tested as much as possible and as is the path towards a CI/CD pipeline is open, that and extensive Python/C++ documentation are yet to happen.

## The Interface is evolving

Naturally, throughout the implementation new things arise - new domains, special cases requiring additional escape hatches or overly restrictive defaults (currently there is no way of accessing/setting the Gomea population directly). To minimise the friction, a decent design philosophy is to consider imposing as few restrictions as necessary to ease future changes. However, do not consider or implement plausible future use cases already. As in try to reduce what you do to its core instead of adding possibly useless layers of  abstraction.

[^1]: unless they ask via dynamic casting
