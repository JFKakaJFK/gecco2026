# Design Philosophy

The theoretical/technical reasoning behind some decisions. The reasoning is to impose as few restrictions as possible, which leads to a very general yet abstract design, while defaults and optional escape hatches keep things practical where possible.

## Problem-algorithm separation

At the core is the problem instance & method design.
Problem instance, because the same problem might have different instances - could be different graphs for maxcut or folds for symbolic regression. Each instance has to provide its search space. For some problems, the initial starting point is also given, e.g., for BBOB. Also, since algorithms don't necessarily know the domain, they might not know how to initialize the population, so it is easier to provide that on the instance side.

For other algorithms, initialization is part of the algorithm. Luckily, when running a method on a problem instance, all information about the instance and the method is known. While this may not always be convenient, the separation between instance and method does not prevent anything.

In other words, methods should be separated from the problem to be general, yet problem-specific mechanisms should still be possible. The escape hatch for this is solution extensions - they add to the interface and are available on both sides of the method/instance boundary, allowing problem and method specializations.

Since methods don't know what problem they are solving[^1], they are not aware of special semantics. For example, in GP with the ERC representation, there is a 1:1 mapping between discrete and continuous variables, and discrete crossover should also copy the corresponding continuous values. Either the method knows this and is specialized to the problem, or the problem provides possibly specialized behaviors (i.e., inherits discrete, add_random, gradient handling, solution logging/formatting, and providing cache keys). Most of these behaviors come with reasonable defaults, and only problems with special needs need to be aware of these escape hatches.

Beyond this, the problem interface aims to allow exploiting batched & partial evaluations, and the option to provide information about (conditionally) inactive variables. Notably, each evaluation must specify which variables are active for the solution!

For optional behaviors not tied to the problem instance or method, such as logging or caching, wrapping an instance is the go-to way to insert functionality between the instance and the problem. The wrapper needs to forward (and optionally override) ALL problem instance behavior in case the actual, wrapped, instance overrides any defaults! Similarly, some special algorithm behavior requires testing whether a problem has a certain type, which in turn requires yet another escape hatch in the form of an unwrap method that yields the innermost problem instance. To make this bearable, a default wrapper is provided that can be extended. Updates to the instance interface require default forwarding in the default wrapper!

On the method side, the main requirement is that methods can be run on instances. The results are returned as an archive, where the instance again defines the archiving criteria.

## Solutions, Quality, and Fitness

Solutions are an important part of the instance/method separation - after all, solutions are information that flows across the interface boundary. At the same time, the general behavior of providing access to decision variable values is clear, but the actual underlying memory layout is not. For example, solutions could be stored as individual arrays of variables (array of structs) or as parts of larger matrices (structs of arrays). Again, methods should not have to concern themselves with this unless that is desired (depending on an algorithm's memory access patterns, one or the other memory layout might be more favorable). To enable this, automatic implicit (!) conversion between solution representations is supported, and a second type that represents sets of solutions is introduced. Each solution set implementation provides a list/dynamic array-like interface and can freely choose how solutions are stored, with extensions serving as the escape hatch to attach information beyond the standard solution interface.

The fitness of solutions is another piece of information that needs to cross the instance/method boundary. However, since the methods are mainly EAs, they generally only need to know which of two solutions is preferable, rather than exact numbers. To retain the advantage of only requiring relative ordering among solutions, the typical fitness is split into a solution quality and a preference relation that defines a partial order and a distance over qualities. In typical fashion, the design choice here is not to place unnecessary restrictions. The main requirement is that the fitness is compatible with the quality used. Different components, such as archives and methods, might use different fitnesses, as long as the same quality type is used. The fitness, and with it the quality, is defined by the problem instance. Consequently, methods cannot construct qualities because they do not know the concrete type. Sometimes that is necessary; for this, the instance exposes the fitness, which in turn allows creating and comparing quality instances.

Notably, the lack of absolute comparisons prevents some operations. For example, it is not possible to compute the mean quality. Still, qualities can be sorted, and the median determined much the same way as k-means and k-medians are analogous. Likewise, for almost all necessary operations, exact or semantically close workarounds are possible.

## Practical Considerations

While the library's primary language is C++, there should also be Python bindings. With the various approaches to bindings, most require defining an intermediate representation of the C++/Python interface that strongly resembles the original C++ code. To avoid code duplication, litgen is used to automatically generate the bindings. The rough way this works is that litgen parses the C++ headers and then emits Python bindings for C++ declarations. There are some [caveats since the parser used by litgen differs from the compiler](pylib/README.md), but it mostly just works.

To allow dynamic invocation of library code from Python, we need to resort to runtime polymorphism (virtual classes) rather than compile-time polymorphism (templates/concepts) in most cases. This comes with some overhead as indirections and vtables are unavoidable. Nonetheless, this arguably is a pragmatic approach. To make this possible, C++17 is the first sufficient standard; however, since C++20/23 should be broadly supported soon and those standards do come with some conveniences (e.g., std::span, std::print), the minimum C++ version supported is C++23 as of now.

At this point, the code is split into the core library (lib), benchmarking functionality (bench), special domains (e.g., gp), and methods. Generally, library code must not depend on other code; benchmarking code only depends on library code; domain code should be method-independent, and method code can pull in any dependencies.

Testing methods generally is hard. Individual components and operators with clear behavior can be tested individually. However, to ensure the overall optimization performance is up to par, the best approach is to compare against a trusted baseline across several benchmarks. In practice, these tests need to be thorough enough to build confidence in the implementation while also running quickly to enable fast iteration.

Ideally, in a CI pipeline, formatting, linting, regression tests, and all examples are executed as much as possible to ensure nothing breaks, together with extensive Python/C++ documentation. However, this has yet to happen.

## The Interface is evolving

Naturally, throughout the implementation, new things arise - new domains, special cases requiring additional escape hatches or overly restrictive defaults (currently there is no way of accessing/setting the Gomea population directly). To minimize friction, a good design philosophy is to impose as few restrictions as necessary to ease future changes. However, do not consider or implement plausible future use cases already. Try to reduce what you do to its core, rather than adding potentially unnecessary layers of abstraction.

[^1]: unless they ask via dynamic casting
