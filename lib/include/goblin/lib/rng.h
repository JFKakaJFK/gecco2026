#pragma once
#ifndef _GOBLIN_LIB_RNG_H
#define _GOBLIN_LIB_RNG_H

#include <algorithm>
#include <concepts>
#include <csetjmp>
#include <numeric>
#include <random>
#include <type_traits>
#include <vector>

#include <openrand/philox.h>

#include "goblin/lib/types.h"

namespace goblin {
using Philox = openrand::Philox;
using Rng = Philox;

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
