#pragma once
#ifndef _GOBLIN_GP_INIT_H
#define _GOBLIN_GP_INIT_H

#include <set>
#include <tuple>
#include <vector>

#include "goblin/lib/assert.h"
#include "goblin/lib/types.h"
#include "goblin/lib/rng.h"
#include "goblin/lib/init.h"
#include "goblin/gp/context.h"
#include "goblin/gp/instance.h"

namespace goblin {
class FullInit final : public DiscreteInitBase {
 public:
  Mat<DType> sample(Rng& rng, const InstanceBase& problem, usize count) const override final {
    try {
      const auto& ctx = dynamic_cast<const GPInstanceBase&>(problem).context();

      Mat<DType> dvals(count, problem.num_discrete());
      for (isize c = 0; c < dvals.cols(); c++) {
        std::vector<usize> idx2domain;

        bool is_leaf = ctx.children[c].empty();

        for (usize domain_value = 0; domain_value < problem.discrete_domain_sizes()(c); domain_value++) {
          DType value = ctx.domain2value(c, domain_value);
          if (is_leaf ? ctx.value_min_arity[value] == 0  // terminals only
                      : ctx.value_min_arity[value] > 0   // non-terminals only
          ) {
            idx2domain.push_back(domain_value);
          }
        }

        __goblin_runtime_assert(idx2domain.size() >= 1);

        std::uniform_int_distribution<DType> d(0, idx2domain.size() - 1);
        for (usize i = 0; i < count; i++) {
          dvals(i, c) = idx2domain[d(rng)];
        }
      }
      return dvals;

    } catch (const std::bad_cast& e) {
      __goblin_runtime_assert(false);
    }
  };
};

class HalfHalfInit final : public DiscreteInitBase {
 public:
  Mat<DType> sample(Rng& rng, const InstanceBase& problem, usize count) const override final {
    Mat<DType> dvals(count, problem.num_discrete());

    // Grow is "for each node sample from the full domain"
    // = Random uniform initialization per node
    usize grow_count = count / 2;
    dvals(Eigen::seq(0, grow_count - 1), Eigen::placeholders::all) = RandomDInit().sample(rng, problem, grow_count);

    usize full_count = count - grow_count;
    dvals(Eigen::seq(grow_count, count - 1), Eigen::placeholders::all) = FullInit().sample(rng, problem, full_count);

    return dvals;
  };
};

/// Complete init does not really work as intended for GP, since a lot of the diversity is distributed to inactive
/// individuals. This version aims to have the active and inactive values per variable be "probabilistically complete"
/// in isolation.
class RecursiveCompleteInit final : public DiscreteInitBase {
 public:
  Mat<DType> sample(Rng& rng, const InstanceBase& problem, usize count) const override final {
    try {
      const auto& ctx = dynamic_cast<const GPInstanceBase&>(problem).context();

      Mat<DType> dvals(count, problem.num_discrete());
      Arr2D<BType> dactive(count, problem.num_discrete());

      std::vector<usize> active_indices;
      active_indices.reserve(count);
      std::vector<usize> inactive_indices;
      inactive_indices.reserve(count);

      usize num_roots = ctx.output_roots.size() + ctx.subtree_roots.size();
      for (usize root_idx = 0; root_idx < num_roots; root_idx++) {
        usize root = root_idx < ctx.output_roots.size() ? ctx.output_roots[root_idx]
                                                        : ctx.subtree_roots[root_idx - ctx.output_roots.size()];
        std::vector<usize> queue{root};
        dactive.col(root) = true;

        while (!queue.empty()) {
          usize current = queue.back();
          queue.pop_back();

          active_indices.clear();
          inactive_indices.clear();
          for (usize i = 0; i < count; i++) {
            if (dactive(i, current)) {
              active_indices.push_back(i);
            } else {
              inactive_indices.push_back(i);
            }
          }

          // TODO is it better to maximize the number of active variables by sampling terminals only once?
          std::vector<DType> perm(problem.discrete_domain_sizes()(current));
          std::iota(perm.begin(), perm.end(), 0);

          usize i = perm.size();
          for (usize r : active_indices) {
            if (i >= perm.size()) {
              i = 0;
              std::shuffle(perm.begin(), perm.end(), rng);
            }

            dvals(r, current) = perm[i++];
          }
          for (usize r : inactive_indices) {
            if (i >= perm.size()) {
              i = 0;
              std::shuffle(perm.begin(), perm.end(), rng);
            }

            dvals(r, current) = perm[i++];
          }

          for (usize ci = 0; ci < ctx.children[current].size(); ci++) {
            usize c = ctx.children[current][ci];

            queue.push_back(c);
            for (usize i = 0; i < count; i++) {
              DType value = ctx.domain2value(current, dvals(i, current));
              // child is active if the parent (=current) is active
              // and the parent arity contains the child...
              dactive(i, c) = dactive(i, current) && ctx.value_max_arity[value] > ci;
            }
          }
        }
      }

      for (isize c = 0; c < dvals.cols(); c++) {
        std::vector<usize> idx2domain;

        bool is_leaf = ctx.children[c].empty();

        for (usize domain_value = 0; domain_value < problem.discrete_domain_sizes()(c); domain_value++) {
          DType value = ctx.domain2value(c, domain_value);
          if (is_leaf ? ctx.value_min_arity[value] == 0  // terminals only
                      : ctx.value_min_arity[value] > 0   // non-terminals only
          ) {
            idx2domain.push_back(domain_value);
          }
        }

        __goblin_runtime_assert(idx2domain.size() >= 1);

        std::uniform_int_distribution<DType> d(0, idx2domain.size() - 1);
        for (usize i = 0; i < count; i++) {
          dvals(i, c) = idx2domain[d(rng)];
        }
      }
      return dvals;

    } catch (const std::bad_cast& e) {
      return CompleteInit().sample(rng, problem, count);
    }
  };
};

// TODO ptc2

};  // namespace goblin

#endif /* _GOBLIN_GP_INIT_H */
