#include <format>
#include <stdexcept>

// TODO add a macro for error handling and input handling to that
// a simple runtime assert to check cheap invariants even with optimizations
// - use often, profile and only remove if runtime cost is considerable
#ifndef __goblin_runtime_assert
#define __goblin_runtime_assert(x)                                                                                 \
    if (!(x)) {                                                                                                    \
        throw std::runtime_error(std::format("{}:{}:\n\t Runtime assertion `{}` failed", __FILE__, __LINE__, #x)); \
    }  // eats up semicolon
#endif
