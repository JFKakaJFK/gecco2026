#pragma once
#ifndef _GOBLIN_GP_INIT_H
#define _GOBLIN_GP_INIT_H

#include <set>
#include <tuple>
#include <vector>
#include <random>
#include <stdexcept>

#include "goblin/lib/assert.h"
#include "goblin/lib/types.h"
#include "goblin/lib/rng.h"
#include "goblin/lib/init.h"
#include "goblin/gp/context.h"
#include "goblin/gp/instance.h"

namespace goblin {

class GrowInit final : public DiscreteInitBase {
 public:
  GrowInit(std::optional<double> p_terminal = std::nullopt, std::optional<double> p_constant = std::nullopt)
      : p_terminal(p_terminal), p_constant(p_constant) {};

  Mat<DType> sample(Rng& rng, const InstanceBase& problem, usize count) const override final {
    try {
      const auto& ctx = dynamic_cast<const GPInstanceBase&>(problem).context();

      Mat<DType> dvals(count, problem.num_discrete());
      for (isize c = 0; c < dvals.cols(); c++) {
        std::vector<usize> const_terminals;      // numeric constant symbols
        std::vector<usize> non_const_terminals;  // features/args/params
        std::vector<usize> non_terminals;        // operators/subfunctions

        usize domain_size = problem.discrete_domain_sizes()(c);
        if (domain_size == 0) {
          throw std::runtime_error("Variable domain must be non-empty.");
        }
        for (usize domain_value = 0; domain_value < domain_size; domain_value++) {
          DType value = ctx.domain2value(c, domain_value);
          if (ctx.value_min_arity[value] > 0) {
            non_terminals.push_back(domain_value);
          } else if (ctx.value_kind[value] == ValueKind::Constant) {
            const_terminals.push_back(domain_value);
          } else {
            non_const_terminals.push_back(domain_value);
          }
        }

        if (const_terminals.empty() && non_const_terminals.empty()) {
          throw std::runtime_error("Variable domain must include terminal symbols.");
        }

        double actual_p_terminal =
            p_terminal.value_or(1.0 - static_cast<double>(non_terminals.size()) / static_cast<double>(domain_size));
        double actual_p_constant = p_constant.value_or(
            static_cast<double>(const_terminals.size()) /
            (static_cast<double>(const_terminals.size()) + static_cast<double>(non_const_terminals.size())));
        if (const_terminals.empty()) {
          actual_p_constant = 0.0;
        }
        if (non_terminals.empty()) {
          actual_p_terminal = 1.0;
        }

        bool is_leaf = ctx.children[c].empty();
        std::uniform_real_distribution<double> U(0.0, 1.0);
        std::uniform_int_distribution<DType> ct_dist(0, const_terminals.size() - 1);
        std::uniform_int_distribution<DType> nct_dist(0, non_const_terminals.size() - 1);
        std::uniform_int_distribution<DType> nt_dist(0, non_terminals.size() - 1);

        for (usize i = 0; i < count; i++) {
          if (is_leaf || U(rng) < actual_p_terminal) {
            if (U(rng) < actual_p_constant) {
              dvals(i, c) = const_terminals[ct_dist(rng)];
            } else {
              dvals(i, c) = non_const_terminals[nct_dist(rng)];
            }
          } else {
            dvals(i, c) = non_terminals[nt_dist(rng)];
          }
        }
      }
      return dvals;

    } catch (const std::bad_cast& e) {
      __goblin_runtime_assert(false);
    }
  };

 private:
  std::optional<double> p_terminal{};
  std::optional<double> p_constant{};
};

class FullInit final : public DiscreteInitBase {
 public:
  FullInit(std::optional<double> p_constant = std::nullopt) : p_constant(p_constant) {};

  Mat<DType> sample(Rng& rng, const InstanceBase& problem, usize count) const override final {
    try {
      const auto& ctx = dynamic_cast<const GPInstanceBase&>(problem).context();

      Mat<DType> dvals(count, problem.num_discrete());
      for (isize c = 0; c < dvals.cols(); c++) {
        std::vector<usize> const_terminals;      // numeric constant symbols
        std::vector<usize> non_const_terminals;  // features/args/params
        std::vector<usize> non_terminals;        // operators/subfunctions

        usize domain_size = problem.discrete_domain_sizes()(c);
        if (domain_size == 0) {
          throw std::runtime_error("Variable domain must be non-empty.");
        }
        for (usize domain_value = 0; domain_value < domain_size; domain_value++) {
          DType value = ctx.domain2value(c, domain_value);
          if (ctx.value_min_arity[value] > 0) {
            non_terminals.push_back(domain_value);
          } else if (ctx.value_kind[value] == ValueKind::Constant) {
            const_terminals.push_back(domain_value);
          } else {
            non_const_terminals.push_back(domain_value);
          }
        }

        bool is_leaf = ctx.children[c].empty();
        if (is_leaf) {
          if (const_terminals.empty() && non_const_terminals.empty()) {
            throw std::runtime_error("Variable domain must include terminal symbols.");
          }

          double actual_p_constant = p_constant.value_or(
              static_cast<double>(const_terminals.size()) /
              (static_cast<double>(const_terminals.size()) + static_cast<double>(non_const_terminals.size())));
          if (const_terminals.empty()) {
            actual_p_constant = 0.0;
          }

          std::uniform_real_distribution<double> U(0.0, 1.0);
          std::uniform_int_distribution<DType> ct_dist(0, const_terminals.size() - 1);
          std::uniform_int_distribution<DType> nct_dist(0, non_const_terminals.size() - 1);

          for (usize i = 0; i < count; i++) {
            if (U(rng) < actual_p_constant) {
              dvals(i, c) = const_terminals[ct_dist(rng)];
            } else {
              dvals(i, c) = non_const_terminals[nct_dist(rng)];
            }
          }
        } else {
          if (non_terminals.empty()) {
            throw std::runtime_error("Variable domain must include non-terminal symbols for non-leaf nodes.");
          }

          std::uniform_int_distribution<DType> nt_dist(0, non_terminals.size() - 1);

          for (usize i = 0; i < count; i++) {
            dvals(i, c) = non_terminals[nt_dist(rng)];
          }
        }
      }
      return dvals;

    } catch (const std::bad_cast& e) {
      __goblin_runtime_assert(false);
    }
  };

 private:
  std::optional<double> p_constant{};
};

class HalfHalfInit final : public DiscreteInitBase {
 public:
  HalfHalfInit(std::optional<double> p_terminal = std::nullopt, std::optional<double> p_constant = std::nullopt)
      : grow(GrowInit(p_terminal, p_constant)), full(FullInit(p_constant)) {};

  Mat<DType> sample(Rng& rng, const InstanceBase& problem, usize count) const override final {
    Mat<DType> dvals(count, problem.num_discrete());

    // Grow is "for each node sample from the full domain"
    // = Random uniform initialization per node
    usize grow_count = count / 2;
    dvals(Eigen::seq(0, grow_count - 1), Eigen::placeholders::all) = grow.sample(rng, problem, grow_count);

    usize full_count = count - grow_count;
    dvals(Eigen::seq(grow_count, count - 1), Eigen::placeholders::all) = full.sample(rng, problem, full_count);

    return dvals;
  };

 private:
  GrowInit grow{};
  FullInit full{};
};

/// PTC2 as per https://cs.gmu.edu/~sean/papers/treecreation.pdf
class PTC2Init final : public DiscreteInitBase {
 public:
  PTC2Init(std::optional<double> p_constant = std::nullopt) : p_constant(p_constant) {};

  Mat<DType> sample(Rng& rng, const InstanceBase& problem, usize count) const override final {
    try {
      const auto& ctx = dynamic_cast<const GPInstanceBase&>(problem).context();

      Mat<DType> dvals(count, problem.num_discrete());

      std::vector<std::vector<usize>> non_terminals(dvals.cols());        // functions/subfunctions
      std::vector<std::vector<usize>> const_terminals(dvals.cols());      // numeric constant symbols
      std::vector<std::vector<usize>> non_const_terminals(dvals.cols());  // features/args/params

      for (usize i = 0; i < problem.num_discrete(); i++) {
        usize domain_size = problem.discrete_domain_sizes()(i);
        for (usize domain_value = 0; domain_value < domain_size; domain_value++) {
          DType value = ctx.domain2value(i, domain_value);
          if (ctx.value_min_arity[value] > 0) {
            non_terminals[i].push_back(domain_value);
          } else if (ctx.value_kind[value] == ValueKind::Constant) {
            const_terminals[i].push_back(domain_value);
          } else {
            non_const_terminals[i].push_back(domain_value);
          }
        }
        if (const_terminals[i].empty() && non_const_terminals[i].empty()) {
          throw std::runtime_error("Variable domain must include terminal symbols.");
        }
      }

      usize num_roots = ctx.output_roots.size() + ctx.subtree_roots.size();
      for (isize i = 0; i < dvals.rows(); i++) {
        for (usize root_idx = 0; root_idx < num_roots; root_idx++) {
          usize root = root_idx < ctx.output_roots.size() ? ctx.output_roots[root_idx]
                                                          : ctx.subtree_roots[root_idx - ctx.output_roots.size()];
          sample_tree(rng, problem, ctx, non_terminals, const_terminals, non_const_terminals, dvals.row(i), root);
        }
      }
      return dvals;

    } catch (const std::bad_cast& e) {
      __goblin_runtime_assert(false);
    }
  };

 private:
  void sample_terminal(Rng& rng,
                       const InstanceBase& problem,
                       const GPContext& ctx,
                       const std::vector<std::vector<usize>>& non_terminals,
                       const std::vector<std::vector<usize>>& const_terminals,
                       const std::vector<std::vector<usize>>& non_const_terminals,
                       RefS<Vec<DType>> values,
                       usize idx) const {
    // 1. determine if it is a constant or other terminal
    double actual_p_constant = p_constant.value_or(
        static_cast<double>(const_terminals[idx].size()) /
        (static_cast<double>(const_terminals[idx].size()) + static_cast<double>(non_const_terminals[idx].size())));
    if (const_terminals[idx].empty()) {
      actual_p_constant = 0.0;
    }

    // 2. sample and place the terminal
    std::uniform_real_distribution<double> U(0.0, 1.0);
    if (U(rng) < actual_p_constant) {
      std::uniform_int_distribution<usize> ct_dist(0, const_terminals[idx].size() - 1);
      values(idx) = const_terminals[idx][ct_dist(rng)];
    } else {
      std::uniform_int_distribution<usize> nct_dist(0, non_const_terminals[idx].size() - 1);
      values(idx) = non_const_terminals[idx][nct_dist(rng)];
    }

    // 3. enqueue any children to ensure all nodes get initialized
    for (auto c : ctx.children[idx]) {
      sample_tree(rng, problem, ctx, non_terminals, const_terminals, non_const_terminals, values, c);
    }
  }

  void sample_tree(Rng& rng,
                   const InstanceBase& problem,
                   const GPContext& ctx,
                   const std::vector<std::vector<usize>>& non_terminals,
                   const std::vector<std::vector<usize>>& const_terminals,
                   const std::vector<std::vector<usize>>& non_const_terminals,
                   RefS<Vec<DType>> values,
                   usize idx) const {
    std::uniform_int_distribution<usize> size_dist(1, ctx.nodes[idx].size());
    usize target_size = size_dist(rng);

    if (target_size < 2) {
      sample_terminal(rng, problem, ctx, non_terminals, const_terminals, non_const_terminals, values, idx);
    } else {
      std::vector<usize> holes{idx};
      usize current_size = 0;

      // more functions are needed to reach the target size
      while (!holes.empty() && holes.size() + current_size < target_size) {
        // 1. get a random hole
        std::swap(holes[std::uniform_int_distribution<usize>(0, holes.size() - 1)(rng)], holes.back());
        usize i = holes.back();
        holes.pop_back();

        // 2. fill the hole
        if (ctx.children[i].empty()) {
          sample_terminal(rng, problem, ctx, non_terminals, const_terminals, non_const_terminals, values, i);
        } else {
          if (non_terminals[i].empty()) {
            throw std::runtime_error("Variable domain must include non-terminal symbols.");
          }
          std::uniform_int_distribution<usize> nt_dist(0, non_terminals[i].size() - 1);
          values(i) = non_terminals[i][nt_dist(rng)];
        }
        current_size++;

        // 3. update the list of holes and fill intron subtrees recursively
        usize value = ctx.domain2value(i, values(i));
        usize arity = ctx.value_max_arity[value];
        for (usize j = 0; j < ctx.children[i].size(); j++) {
          if (j < arity) {
            holes.push_back(ctx.children[i][j]);
          } else {
            sample_tree(rng, problem, ctx, non_terminals, const_terminals, non_const_terminals, values,
                        ctx.children[i][j]);
          }
        }
      }

      // once the size has been reached, fill the remaining nodes with terminals
      for (usize i : holes) {
        sample_terminal(rng, problem, ctx, non_terminals, const_terminals, non_const_terminals, values, i);
      }
    }
  };

  std::optional<double> p_constant{};
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

      // for (isize c = 0; c < dvals.cols(); c++) {
      //   std::vector<usize> idx2domain;

      //   bool is_leaf = ctx.children[c].empty();

      //   for (usize domain_value = 0; domain_value < problem.discrete_domain_sizes()(c); domain_value++) {
      //     DType value = ctx.domain2value(c, domain_value);
      //     if (is_leaf ? ctx.value_min_arity[value] == 0  // terminals only
      //                 : ctx.value_min_arity[value] > 0   // non-terminals only
      //     ) {
      //       idx2domain.push_back(domain_value);
      //     }
      //   }

      //   __goblin_runtime_assert(idx2domain.size() >= 1);

      //   std::uniform_int_distribution<DType> d(0, idx2domain.size() - 1);
      //   for (usize i = 0; i < count; i++) {
      //     dvals(i, c) = idx2domain[d(rng)];
      //   }
      // }
      return dvals;

    } catch (const std::bad_cast& e) {
      return CompleteInit().sample(rng, problem, count);
    }
  };
};

/// Complete init does not really work as intended for GP, since a lot of the diversity is distributed to inactive
/// individuals. This version aims to have the active and inactive values per variable be "probabilistically complete"
/// in isolation.
class RecursiveCompleteInit2 final : public DiscreteInitBase {
 public:
  RecursiveCompleteInit2(std::optional<double> p_terminal = std::nullopt,
                         std::optional<double> p_constant = std::nullopt)
      : p_terminal(p_terminal), p_constant(p_constant) {};

  Mat<DType> sample(Rng& rng, const InstanceBase& problem, usize count) const override final {
    try {
      const auto& ctx = dynamic_cast<const GPInstanceBase&>(problem).context();

      Mat<DType> dvals(count, problem.num_discrete());
      Arr2D<BType> dactive(count, problem.num_discrete());

      std::vector<std::vector<DType>> non_terminals(dvals.cols());        // functions/subfunctions
      std::vector<std::vector<DType>> const_terminals(dvals.cols());      // numeric constant symbols
      std::vector<std::vector<DType>> non_const_terminals(dvals.cols());  // features/args/params

      for (usize i = 0; i < problem.num_discrete(); i++) {
        usize domain_size = problem.discrete_domain_sizes()(i);
        for (usize domain_value = 0; domain_value < domain_size; domain_value++) {
          DType value = ctx.domain2value(i, domain_value);
          if (ctx.value_min_arity[value] > 0) {
            non_terminals[i].push_back(domain_value);
          } else if (ctx.value_kind[value] == ValueKind::Constant) {
            const_terminals[i].push_back(domain_value);
          } else {
            non_const_terminals[i].push_back(domain_value);
          }
        }
        if (const_terminals[i].empty() && non_const_terminals[i].empty()) {
          throw std::runtime_error("Variable domain must include terminal symbols.");
        }
      }

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

          dvals(active_indices, current) = sample_nodes(rng, problem, non_terminals, const_terminals,
                                                        non_const_terminals, active_indices.size(), current);
          dvals(inactive_indices, current) = sample_nodes(rng, problem, non_terminals, const_terminals,
                                                          non_const_terminals, inactive_indices.size(), current);

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

      return dvals;
    } catch (const std::bad_cast& e) {
      return CompleteInit().sample(rng, problem, count);
    }
  };

 private:
  Vec<DType> sample_nodes(Rng& rng,
                          const InstanceBase& problem,
                          const std::vector<std::vector<DType>>& non_terminals,
                          const std::vector<std::vector<DType>>& const_terminals,
                          const std::vector<std::vector<DType>>& non_const_terminals,
                          usize total,
                          usize idx) const {
    usize domain_size = problem.discrete_domain_sizes()(idx);

    double actual_p_terminal =
        p_terminal.value_or(1.0 - static_cast<double>(non_terminals[idx].size()) / static_cast<double>(domain_size));
    double actual_p_constant = p_constant.value_or(
        static_cast<double>(const_terminals[idx].size()) /
        (static_cast<double>(const_terminals[idx].size()) + static_cast<double>(non_const_terminals[idx].size())));
    if (const_terminals[idx].empty()) {
      actual_p_constant = 0.0;
    }

    usize num_non_terminals =
        non_terminals[idx].empty() ? 0 : static_cast<usize>((1.0 - actual_p_terminal) * static_cast<double>(total));
    usize num_const_terminals = actual_p_constant * static_cast<double>(total - num_non_terminals);
    usize num_non_const_terminals = total - num_non_terminals - num_const_terminals;

    Vec<DType> values(total);

    sample_complete(rng, non_terminals[idx], values(Eigen::seqN(0, num_non_terminals)));
    sample_complete(rng, const_terminals[idx], values(Eigen::seqN(num_non_terminals, num_const_terminals)));
    sample_complete(rng, non_const_terminals[idx],
                    values(Eigen::seqN(num_non_terminals + num_const_terminals, num_non_const_terminals)));

    std::shuffle(values.begin(), values.end(), rng);

    return values;
  };

  void sample_complete(Rng& rng, const std::vector<DType>& pool, Ref<Vec<DType>> values) const {
    std::vector<DType> perm(pool.size());
    std::iota(perm.begin(), perm.end(), 0);

    usize i = perm.size();
    for (isize vi = 0; vi < values.size(); vi++) {
      if (i >= perm.size()) {
        i = 0;
        std::shuffle(perm.begin(), perm.end(), rng);
      }

      values(vi) = pool[perm[i++]];
    }
  };

  std::optional<double> p_terminal{};
  std::optional<double> p_constant{};
};

};  // namespace goblin

#endif /* _GOBLIN_GP_INIT_H */
