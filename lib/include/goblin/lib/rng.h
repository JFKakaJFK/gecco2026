#pragma once
#ifndef _GOBLIN_LIB_RNG_H
#define _GOBLIN_LIB_RNG_H

#include <algorithm>
#include <concepts>
#include <csetjmp>
#include <numeric>
#include <optional>
#include <random>
#include <type_traits>
#include <vector>

#include <openrand/philox.h>
#include <openrand/squares.h>

#include "goblin/lib/types.h"

namespace goblin {
using Rng = openrand::Philox;
// using Rng = std::mt19937;

// OpenRAND and other rngs don't have the same API other than std::uniform_random_bit_generator, so a wrapper for the
// RNG creation allows changing PRNG
template <typename R = Rng>
inline R seeded_rng(u64 state, u32 ctr = 0) {
    // OpenRAND-like
    if constexpr (std::is_constructible_v<R, u64, u32>) {
        return R(state, ctr);
    } else {
        // std::random-like
        std::seed_seq seed{static_cast<u32>(state), ctr};
        R rng;
        rng.seed(seed);
        return rng;
    }
};
inline Rng seeded_rng(std::optional<u64> seed = std::nullopt) {
    if (seed.has_value()) {
        return seeded_rng(seed.value());
    } else {
        std::random_device rd;
        std::uniform_int_distribution<u64> seed_dist(0, std::numeric_limits<u64>::max());

        return seeded_rng<Rng>(seed_dist(rd));
    }
};

// TODO possibly use the OpenRAND provided sampling methods (decreases rng portability, but it looks like any rng can be
// wrapped with openrand::BaseRNG<T> and the sampling methods both look convenient and decently fast)

// TODO possibly profile & look at (for faster rn generation)
// - https://www.pcg-random.org/posts/bounded-rands.html
// - https://github.com/swiftlang/swift/pull/39143#issue-comment-box

inline std::vector<usize> permute(Rng& rng, usize n) {
    std::vector<usize> p(n);
    std::iota(p.begin(), p.end(), 0);
    std::shuffle(p.begin(), p.end(), rng);
    return p;
};
};  // namespace goblin

static_assert(std::uniform_random_bit_generator<goblin::Rng>);

#endif /* _GOBLIN_LIB_RNG_H */
