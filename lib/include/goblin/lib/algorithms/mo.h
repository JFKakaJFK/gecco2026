#pragma once
#ifndef _GOBLIN_LIB_ALGORITHMS_MO_H
#define _GOBLIN_LIB_ALGORITHMS_MO_H

#include <algorithm>
#include <limits>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>
#include <set>

#include "goblin/lib/assert.h"
#include "goblin/lib/ordering.h"

namespace goblin {

template <typename D>
inline std::tuple<std::vector<usize>, std::vector<std::set<usize>>> non_dominated_sorting(D&& cmp, usize pool_size) {
  std::vector<std::set<usize>> dominations(pool_size);
  std::vector<usize> domination_count(pool_size, 0);
  std::vector<usize> rank(pool_size);
  std::vector<std::set<usize>> fronts(1);

  for (usize i = 0; i < pool_size; i++) {
    for (usize j = 0; j < i; j++) {
      Ordering o = cmp(i, j);
      if (o == Ordering::Better) {
        dominations[i].insert(j);
        domination_count[j]++;
      } else if (o == Ordering::Worse) {
        dominations[j].insert(i);
        domination_count[i]++;
      }
    }
    if (domination_count[i] == 0) {
      rank[i] = 0;
      fronts.back().insert(i);
    }
  }

  while (!fronts.back().empty()) {
    std::set<usize> front;
    for (usize p : fronts.back()) {
      for (usize q : dominations[p]) {
        domination_count[q]--;
        if (domination_count[q] == 0) {
          rank[q] = fronts.size();
          front.insert(q);
        }
      }
    }
    fronts.push_back(front);
  }

  return std::make_tuple(rank, fronts);
};

};  // namespace goblin

#endif /* _GOBLIN_LIB_ALGORITHMS_MO_H */
