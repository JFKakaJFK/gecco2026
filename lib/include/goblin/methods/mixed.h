#pragma once
#ifndef _GOBLIN_MIXED_GOMEA_H
#define _GOBLIN_MIXED_GOMEA_H

#include <cstdint>
#include <limits>
#include <vector>
#include <algorithm>
#include <cassert>
#include <format>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <print>
#include <random>
#include <span>
#include <stdexcept>
#include <string_view>
#include <variant>
#include <sstream>

#include "goblin/lib/algorithms/subset_selection.h"
#include "goblin/lib/ims.h"
#include "goblin/lib/instance.h"
#include "goblin/lib/linkage_model.h"
#include "goblin/lib/method.h"
#include "goblin/bench/tracked.h"
#include "goblin/methods/continuous.h"

#ifndef NDEBUG
#define __assert_fitness_invariant(s) check_fitness_invariant(rng, s, std::format("{}:{}", __FILE__, __LINE__))
#define __assert_gom_backup_invariant() check_gom_backups(std::format("{}:{}", __FILE__, __LINE__))
#define __assert_invariants()            \
  __assert_fitness_invariant(solutions); \
  __assert_fitness_invariant(parents);   \
  __assert_fitness_invariant(donors);    \
  __assert_gom_backup_invariant()
#else
#define __assert_fitness_invariant(s)
#define __assert_gom_backup_invariant()
#define __assert_invariants()
#endif

namespace goblin {

inline std::tuple<std::vector<usize>, std::vector<std::vector<usize>>, std::vector<std::vector<usize>>>
create_and_register_clusters(Rng& rng,
                             const ArchiveBase& archive,
                             const FitnessBase& fitness,
                             const SolutionSetBase& solutions,
                             usize num_clusters,
                             usize donor_pool_size,
                             const SolutionSetBase& previous_solutions,
                             std::vector<usize>& previous_clusters) {
  assert(donor_pool_size > 0);
  std::vector<usize> solution_clusters(solutions.size(), 0);
  std::vector<std::vector<usize>> cluster_solutions(num_clusters);
  std::vector<std::vector<usize>> cluster_donors;

  // return single cluster with all solutions
  if (num_clusters < 2 || solutions.size() < num_clusters) {
    cluster_solutions[0].resize(solutions.size());
    std::iota(cluster_solutions[0].begin(), cluster_solutions[0].end(), 0);

    cluster_donors.resize(1);
    cluster_donors[0].resize(solutions.size());
    std::iota(cluster_donors[0].begin(), cluster_donors[0].end(), 0);

    return std::make_tuple(solution_clusters, cluster_solutions, cluster_donors);
  }

  usize num_objectives = fitness.num_objectives();

  // clustering here should also use relative comparisons, not absolute ones

  // 1. compute normalized pair-wise objective space distance within solutions
  //    Since only relative distances are used, the normalization uses the
  //    maximum difference for each objective
  Vec<CType> max_dist = Vec<CType>::Zero(num_objectives);
  Vec<CType> min_dist = Vec<CType>::Constant(num_objectives, std::numeric_limits<CType>::infinity());
  Array<usize> best_idx(num_objectives);
  for (usize o = 0; o < num_objectives; o++) {
    const auto& s = archive.so_solution(o).quality();

    for (usize i = 0; i < solutions.size(); i++) {
      auto dist = fitness.distance(solutions[i].quality(), s, o);
      if (!isna(dist)) {
        if (dist > max_dist(o)) {
          max_dist(o) = dist;
        }
        if (dist < min_dist(o)) {
          min_dist(o) = dist;
          best_idx(o) = i;
        }
      }
    }
  }
  assert(min_dist.array().isFinite().all());
  assert(max_dist.array().isFinite().all());

  Vec<CType> dist_norm = (max_dist - min_dist);
  Mat<CType> distance(solutions.size(), solutions.size());
  for (usize i = 0; i < solutions.size(); i++) {
    for (usize j = 0; j < i; j++) {
      CType d_ij(0.0);
      for (usize o = 0; o < num_objectives; o++) {
        auto dist = fitness.distance(solutions[i].quality(), solutions[j].quality(), o);
        d_ij += std::pow(dist / dist_norm(o), 2);
      }
      // ensure non-finite, i.e. replace infinite/na distances with 0.0
      // -> will never be selected during leader selection and
      //    will be assigned to efffectively random clusters
      distance(i, j) = isna(d_ij) ? CType(0.0) : std::sqrt(d_ij);
      distance(j, i) = distance(i, j);
    }
  }

  // 2. greedy scattered subset selection for leaders
  std::uniform_int_distribution<usize> random_objective(0, num_objectives - 1);
  auto [leaders, remaining] =
      greedy_scattered_subset_selection(distance, solutions.size(), num_clusters, best_idx(random_objective(rng)));

  // Depending on which paper you look at, K-means is used to improve the
  // cluster leader assignments here - but https://ir.cwi.nl/pub/23049/23049D.pdf (Chapter 4.6)
  // argues that that the k-means step does not necessarily help

  // 3. round robin cluster assignments (for clusters in random order, add
  // closest solution)

  for (usize k = 0; k < num_clusters; k++) {
    solution_clusters[leaders[k]] = k;
  }
  std::vector<usize> cluster_perm(num_clusters);
  std::iota(cluster_perm.begin(), cluster_perm.end(), 0);
  while (!remaining.empty()) {
    std::shuffle(cluster_perm.begin(), cluster_perm.end(), rng);

    for (auto k : cluster_perm) {
      usize closest_idx;
      distance(leaders[k], remaining).minCoeff(&closest_idx);

      solution_clusters[remaining[closest_idx]] = k;
      std::swap(remaining[closest_idx], remaining.back());
      remaining.pop_back();

      if (remaining.empty()) {
        break;
      }
    }
  }

  // 4. determine SO clusters and re-order clusters
  Mat<CType> min_dist_to_elite =
      Mat<CType>::Constant(num_clusters, num_objectives, std::numeric_limits<CType>::infinity());
  for (usize i = 0; i < solution_clusters.size(); i++) {
    for (usize o = 0; o < num_objectives; o++) {
      min_dist_to_elite(solution_clusters[i], o) =
          std::min(min_dist_to_elite(solution_clusters[i], o),
                   fitness.distance(archive.so_solution(o).quality(), solutions[i].quality(), o));
    }
  }
  std::vector<usize> remaining_clusters(num_clusters);
  std::iota(remaining_clusters.begin(), remaining_clusters.end(), 0);
  std::iota(cluster_perm.begin(), cluster_perm.end(), 0);  // [k_old] -> k_new
  for (usize o : permute(rng, num_objectives)) {
    // find cluster with best solution w.r.t. the current objective
    // and assign that cluster to the objective
    usize best = 0;
    for (usize i = 0; i < remaining_clusters.size(); i++) {
      if (min_dist_to_elite(remaining_clusters[i], o) < min_dist_to_elite(remaining_clusters[best], o)) {
        best = i;
      }
    }

    cluster_perm[remaining_clusters[best]] = o;
    std::swap(remaining_clusters[best], remaining_clusters.back());
    remaining_clusters.pop_back();
  }

  // 5. (if previous objectives + solution_cluster assignments are passed) -
  // perform cluster registration by minimizing maximum matched cluster distance
  // (= average distance between cluster solutions)
  if (!previous_clusters.empty() && remaining_clusters.size() > 1) {
    // assign remaining clusters to clusters [num_objectives,num_clusters) from
    // previous...

    // TODO this does extra work (we don't care about the SO clusters here
    // anymore)
    Mat<CType> generational_distance = Mat<CType>::Zero(num_clusters, num_clusters);
    for (usize i = 0; i < solutions.size(); i++) {
      for (usize j = 0; j < previous_solutions.size(); j++) {
        CType d_ij(0.0);
        for (usize o = 0; o < num_objectives; o++) {
          auto dist = fitness.distance(solutions[i].quality(), previous_solutions[j].quality(), o);
          d_ij += std::pow(dist / dist_norm(o), 2);
        }

        generational_distance(solution_clusters[i], previous_clusters[j]) += isna(d_ij) ? dist_norm.norm() : d_ij;
      }
    }

    // matching - consider all permutations and pick the one with smallest
    // distance sum
    // find best distances that when applied to new clusters matches old
    // clusters
    std::vector<usize> perm(remaining_clusters.size());
    std::iota(perm.begin(), perm.end(), 0);

    std::vector<usize> best_permutation = perm;
    CType best_cumulative_distance = 0.0;
    for (usize i = 0; i < remaining_clusters.size(); i++) {
      best_cumulative_distance += generational_distance(num_objectives + perm[i], num_objectives + i);
    }
    while (std::next_permutation(perm.begin(), perm.end())) {
      CType dist = 0.0;
      for (usize i = 0; i < remaining.size(); i++) {
        dist += generational_distance(num_objectives + perm[i], num_objectives + i);
      }
      if (dist < best_cumulative_distance) {
        best_cumulative_distance = dist;
        best_permutation = perm;
      }
    };

    for (usize i = 0; i < remaining_clusters.size(); i++) {
      cluster_perm[remaining_clusters[i]] = num_objectives + best_permutation[i];
    }
  } else {
    // if we don't do cluster registration, then we still need to make sure that each cluster is in cluster_perm
    for (usize i : remaining_clusters) {
      cluster_perm[i] = i;
    }
  }

  // 6. apply cluster permutation, i.e. apply permutation to solution_clusters
  for (usize i = 0; i < solution_clusters.size(); i++) {
    solution_clusters[i] = cluster_perm[solution_clusters[i]];
  }

  // fill reverse mapping
  for (usize k = 0; k < num_clusters; k++) {
    cluster_solutions[k].reserve(solutions.size() / num_clusters + 1);
  }
  for (usize i = 0; i < solution_clusters.size(); i++) {
    cluster_solutions[solution_clusters[i]].push_back(i);
  }

  // 7. assign donor indices -> closest donor pool size solutins to cluster
  std::vector<usize> indices(solutions.size());
  std::iota(indices.begin(), indices.end(), 0);

  cluster_donors.resize(num_clusters);
  for (usize i = 0; i < num_clusters; i++) {
    usize k = cluster_perm[i];
    std::sort(indices.begin(), indices.end(),
              [&](const auto& lhs, const auto& rhs) { return distance(leaders[i], lhs) < distance(leaders[i], rhs); });

    cluster_donors[k].insert(cluster_donors[k].end(), indices.begin(), indices.begin() + donor_pool_size);
  }

#ifndef NDEBUG
  for (usize i = 0; i < num_clusters; i++) {
    assert(cluster_donors[i].size() == std::min(donor_pool_size, solutions.size()) &&
           "All clusters are supposed to have the same, non-zero donor pool size.");
  }
#endif

  return std::make_tuple(solution_clusters, cluster_solutions, cluster_donors);
};

struct PopulationOptions {
  double donor_pool_size_multiplier = 2.0;
  std::optional<usize> max_nis = std::nullopt;
  bool forced_improvements = true;
  double target_continuous_to_discrete_balance = 1.0;
  bool sequential_gom = false;  // performs GOM sequentially per solution, incompatible with other mechanisms
  bool strict_elite_acceptance =
      false;  // should the single objective elite solutions accept only strict improvements or also neutral changes?

  double donor_search_proportion = 0.0;  // the fraction of solutions to consider before skipping an evaluation in case
                                         // of all subset variables being identical between the solution and donor

  // Coefficient mutation as per https://doi.org/10.1145/3520304.3534036
  double continuous_mutation_probability = 0.0;
  CType continuous_mutation_temperature = 0.1;
  CType continuous_mutation_decay_factor = 0.9;
  std::optional<usize> continuous_mutation_decay_patience = 5;

  bool mutate_before_gradient_step = true;
  usize gradient_step_frequency = 0;
  usize gradient_step_count = 10;
};

template <typename SolutionSet>
class Population {
 public:
  Population(InstanceBase& problem,
             ArchiveBase& global_archive,
             const LinkageModelBase& discrete_model,
             const LinkageModelBase& continuous_model,
             const RvSamplingModelBase& sampling_model,
             usize size,
             usize num_clusters,
             const PopulationOptions& options,
             const RvOptions& rv_options)
      : problem(problem),
        global_archive(global_archive),
        discrete_model(discrete_model.clone()),
        rv_state(rv_options, continuous_model, sampling_model),
        options(options),
        local_archive(global_archive.clone()),
        size(size),
        num_clusters(num_clusters),
        donor_pool_size(std::min(size,
                                 static_cast<usize>(options.donor_pool_size_multiplier * static_cast<double>(size) /
                                                    static_cast<double>(num_clusters)))),
        max_nis(options.max_nis.value_or(1 + static_cast<usize>(std::log10(size)))) {
    __goblin_runtime_assert(!isna(options.target_continuous_to_discrete_balance) &&
                            options.target_continuous_to_discrete_balance >= 0.0);

    __goblin_runtime_assert(!options.continuous_mutation_decay_patience.has_value() ||
                            options.continuous_mutation_decay_patience.value() > 0);
    __goblin_runtime_assert(0.0 <= options.continuous_mutation_probability &&
                            options.continuous_mutation_probability <= 1.0);
  };

  void restart() { solutions.clear(); };

  template <typename T>
  u64 perform_generation(Rng& rng, T should_terminate) {
    u64 evaluations = 0;
    bool is_discrete = problem.num_discrete() > 0;
    bool is_continuous = problem.num_continuous() > 0;
    __goblin_runtime_assert(is_discrete || is_continuous);

    // ======= initialization (if necessary) =======
    if (solutions.empty()) {
      evaluations += initialize(rng);

      if (should_terminate(evaluations)) {
        return evaluations;
      }
    }

    {  // ======= clustering =======
      bool perform_cluster_registration = is_continuous && rv_state.options.enabled;
      // the local archive is used since the elites from that should be in this
      // population
      if (!perform_cluster_registration) {
        solution_clusters.clear();
      }
      std::tie(solution_clusters, cluster_solutions, cluster_donors) = create_and_register_clusters(
          rng, *local_archive, problem.fitness(), solutions, num_clusters, donor_pool_size, donors, solution_clusters);
    }

    // after this, donors == parents == solutions holds
    for (usize i = 0; i < size; i++) {
      donors[i] = solutions[i];

      // if acceptance for elites is strict, find the current elite solution for each single-objective cluster
      if (options.strict_elite_acceptance) {
        usize k = solution_clusters[i];

        if (k < problem.num_objectives() &&
            problem.fitness().cmp(solutions[i].quality(), solutions[so_elite_idx[k]].quality(), std::nullopt) ==
                Ordering::Better) {
          so_elite_idx[k] = i;
        }
      }
    }
    __assert_gom_backup_invariant();

    // ======= state/model updates =======

    usize max_discrete_subset_count = 0;
    if (is_discrete) {
      // learn per cluster linkage models
      if (cluster_FOS.empty() || !discrete_model->is_static()) {
        cluster_FOS.clear();
        for (usize k = 0; k < num_clusters; k++) {
          cluster_FOS.push_back(discrete_model->subsets(rng, problem, solutions, cluster_solutions[k], std::nullopt));
        }
      }
      for (usize k = 0; k < num_clusters; k++) {
        max_discrete_subset_count = std::max(max_discrete_subset_count, cluster_FOS[k].size());
      }

      solution_changed.clear();
      solution_changed.resize(size, false);

      // and fix the subset order for each solution
      // (colwise, each column is contiguous and one step)
      if (static_cast<usize>(subset_orders.cols()) < max_discrete_subset_count || !discrete_model->is_static()) {
        subset_orders.resize(size, max_discrete_subset_count);

        // for(auto row: subset_orders.rowwise()){
        //     std::iota(row.begin(), row.begin() + max_discrete_subset_count, 0);
        // }
        for (usize i = 0; i < max_discrete_subset_count; i++) {
          subset_orders.col(i).array() = i;
        }
      }

      for (auto row : subset_orders.rowwise()) {
        std::shuffle(row.begin(), row.begin() + max_discrete_subset_count, rng);
      }
    }

    // ======= housekeeping before variation =======
    // update local archive, prepare changed flags

    local_archive->adapt();
    local_archive->reset_change_count();

    // ======= variation/evaluation/selection =======
    __assert_invariants();

    if (options.sequential_gom) {
      solutions_to_evaluate.resize(1);
      for (usize i = 0; i < size; i++) {
        usize k = solution_clusters[i];
        auto objective = k < problem.num_objectives() ? std::make_optional(k) : std::nullopt;

        perm.resize(cluster_donors[k].size());
        std::iota(perm.begin(), perm.end(), 0);

        usize max_donor_search_iterations = std::min(options.donor_search_proportion, 1.0) * perm.size();

        for (usize subset_idx = 0; subset_idx < max_discrete_subset_count; subset_idx++) {
          if (subset_idx < cluster_FOS[k].size()) {
            subsets[i] = &cluster_FOS[k][subset_orders(i, subset_idx)];

            usize perm_idx = 0;
            bool evaluation_needed, anything_changed;
            do {
              std::swap(perm[perm_idx], perm[std::uniform_int_distribution<usize>(perm_idx, perm.size() - 1)(rng)]);
              usize donor_idx = cluster_donors[k][perm[perm_idx++]];
              if (i == donor_idx) {
                continue;
              }

              std::tie(evaluation_needed, anything_changed) =
                  solutions[i].inherit(donors[donor_idx], *subsets[i], problem.always_inherit_continuous());

              if (evaluation_needed) {
                solutions_to_evaluate[0] = i;

                problem.evaluate_partial(rng, solutions, parents, subsets, solutions_to_evaluate);

                if (accept_and_update_archive(i, objective,
                                              /* strict */ false)) {
                  parents[i] = solutions[i];
                  solution_changed[i] = true;
                } else {
                  solutions[i] = parents[i];
                }
              } else if (anything_changed) {
                parents[i] = solutions[i];
              }

            } while (!evaluation_needed && perm_idx < max_donor_search_iterations);
          }
        }
      }
    } else {
      usize subset_idx = 0;
      std::uniform_real_distribution<double> U(0.0, 1.0);
      bool can_do_discrete_step;
      do {
        can_do_discrete_step = is_discrete && subset_idx < max_discrete_subset_count;
        bool do_discrete_step;
        if (!is_continuous || !rv_state.options.enabled) {
          do_discrete_step = can_do_discrete_step;
        } else if (!can_do_discrete_step) {
          do_discrete_step = false;
        } else {
          double actual_evaluation_balance =
              discrete_evaluations > 0.0
                  ? continuous_evaluations / discrete_evaluations
                  // since no discrete evaluations were performed we either flip a coin or force a first discrete step
                  // if a first continuous step was already done
                  : (continuous_evaluations > 0.0 ? 2.0 : 1.0) * options.target_continuous_to_discrete_balance;
          // maps the fraction to a percentage such that 0.5 is the target balance,
          // less means too many continuous evaluations, more means too many discrete evaluations
          double p_discrete = 0.5 * actual_evaluation_balance / options.target_continuous_to_discrete_balance;

          do_discrete_step = U(rng) < p_discrete;
        }

        u64 evals = 0;
        // we first do the continuous step - it might not do anything (not enough active variables or already
        // converged), so we still want to be able to do a discrete step instead
        if (is_continuous && rv_state.options.enabled && !do_discrete_step && !rv_state.converged()) {
          // RV-GOMEA uses the elite in the population (~= local archive) for forced improvements + adaptive variance
          // scalling (AVS) evals = rv_state.perform_generation(rng, global_archive, problem, solutions, parents,
          // solution_clusters, cluster_solutions);
          evals = rv_state.perform_generation(rng, *local_archive, problem, solutions, parents, solution_clusters,
                                              cluster_solutions);
          __assert_invariants();
          evaluations += evals;
          continuous_evaluations += evals;
        }

        if (do_discrete_step || (can_do_discrete_step && evals == 0)) {
          evals = discrete_gom_step(rng, subset_idx++);
          __assert_invariants();
          evaluations += evals;
          discrete_evaluations += evals;
        }

        if (is_continuous && options.continuous_mutation_probability > 0.0) {
          evaluations += continuous_mutation_step(rng);
          __assert_invariants();
        }

        if (should_terminate(evaluations).has_value()) {
          return evaluations;
        }
      } while (can_do_discrete_step);  // (subset_idx < max_discrete_subset_count);
    }

    if (is_continuous && options.gradient_step_frequency > 0 &&
        iterations_since_last_gradient_step++ % options.gradient_step_frequency == 0) {
      evaluations += gradient_step(rng);
      __assert_invariants();
    }

    if (options.forced_improvements && is_discrete) {
      evaluations += forced_improvements(rng, should_terminate, max_discrete_subset_count);
      __assert_invariants();

      // update no improvement stretches
      for (usize i = 0; i < size; i++) {
        if (problem.fitness().cmp(solutions[i].quality(), donors[i].quality(), std::nullopt) == Ordering::Better) {
          solution_nis[i] = 0;
        } else {
          solution_nis[i]++;
        }
      }
    }

    // ======= housekeeping after variation =======
    if (local_archive->change_count() > 0) {
      no_improvement_stretch = 0;
    } else {
      no_improvement_stretch++;
    }

    if (options.continuous_mutation_decay_patience.has_value() &&
        options.continuous_mutation_decay_patience.value() > 0 &&
        no_improvement_stretch % options.continuous_mutation_decay_patience.value() == 0) {
      options.continuous_mutation_temperature *= options.continuous_mutation_decay_factor;
    }
    no_evaluations_performed = evaluations == 0;

    return evaluations;
  };

  bool converged() {
    if (problem.num_objectives() == 1) {
      if (problem.num_continuous() == 0 && all_solutions_identical()) {
        return true;
      }
      // since we only have relative comparisons, this roughly is equal to the usual fitness variance == 0.0 condition
      if (problem.num_discrete() == 0 && (avg_dist_to_local_so_elite() == 0.0 || rv_state.converged())) {
        return true;
      }
    }

    return no_improvement_stretch >= max_nis && no_evaluations_performed;
  };

  bool all_solutions_identical() const {
    CType epsilon = 1e-10;
    const auto& s0 = solutions[0];
    for (usize i = 1; i < solutions.size(); i++) {
      const auto& si = solutions[i];

      // what is active differs or any active variable is different
      for (usize j = 0; j < problem.num_discrete(); j++) {
        if (s0.discrete_active()(j) != si.discrete_active()(j) ||
            (s0.discrete_active()(j) && (s0.discrete_values()(j) != si.discrete_values()(j)))) {
          return false;
        }
      }
      for (usize j = 0; j < problem.num_continuous(); j++) {
        if (s0.continuous_active()(j) != si.continuous_active()(j) ||
            (s0.continuous_active()(j) && std::abs(s0.continuous_values()(j) - si.continuous_values()(j)) > epsilon)) {
          return false;
        }
      }
    }
    return true;
  };

  CType avg_dist_to_global_so_elite() const {
    const auto& e = global_archive.so_solution(0);
    CType avg_dist = 0.0;
    for (usize i = 0; i < solutions.size(); i++) {
      avg_dist += problem.fitness().distance(e.quality(), solutions[i].quality(), 0);
    }
    return avg_dist / static_cast<CType>(solutions.size());
  };

  CType avg_dist_to_local_so_elite() const {
    const auto& e = local_archive->so_solution(0);
    CType avg_dist = 0.0;
    for (usize i = 0; i < solutions.size(); i++) {
      avg_dist += problem.fitness().distance(e.quality(), solutions[i].quality(), 0);
    }
    return avg_dist / static_cast<CType>(solutions.size());
  };

  /// For single objective optimization, this just is a roundabout way
  /// to return the elite to check if an IMS population should stop
  const ArchiveBase& archive() const { return *local_archive; };

 private:
  void check_gom_backups(std::string_view info) {
    assert(solutions.size() == parents.size());
    for (usize i = 0; i < solutions.size(); i++) {
      std::string s_discrete = log_helper(solutions[i].discrete_values(), /* escape = */ false, /* indent = */ false),
                  s_dactive = log_helper(solutions[i].discrete_active(), /* escape = */ false, /* indent = */ false),
                  s_continuous =
                      log_helper(solutions[i].continuous_values(), /* escape = */ false, /* indent = */ false),
                  s_cactive = log_helper(solutions[i].continuous_active(), /* escape = */ false, /* indent = */ false),
                  p_discrete = log_helper(parents[i].discrete_values(), /* escape = */ false, /* indent = */ false),
                  p_dactive = log_helper(parents[i].discrete_active(), /* escape = */ false, /* indent = */ false),
                  p_continuous = log_helper(parents[i].continuous_values(), /* escape = */ false, /* indent = */ false),
                  p_cactive = log_helper(parents[i].continuous_active(), /* escape = */ false, /* indent = */ false),
                  s_quality = problem.fitness().format(solutions[i].quality()),
                  p_quality = problem.fitness().format(parents[i].quality());

      // we expect an exact match - after all, the values should have been copied over without a re-evaluation...
      bool discrete_ok = s_discrete == p_discrete;
      bool dactive_ok = s_dactive == p_dactive;
      bool continuous_ok = s_continuous == p_continuous;
      bool cactive_ok = s_cactive == p_cactive;
      bool quality_ok = s_quality == p_quality;
      if (!(discrete_ok && dactive_ok && continuous_ok && cactive_ok && quality_ok)) {
        std::println("{}", info);
        std::println("GOM backup/parent does not match solution {}: ", i);
        std::println("Discrete: ({})", discrete_ok);
        std::println("  Solution: {}", s_discrete);
        std::println("  Parent:   {}", p_discrete);
        std::println("Discrete Active: ({})", dactive_ok);
        std::println("  Solution: {}", s_dactive);
        std::println("  Parent:   {}", p_dactive);
        std::println("Continuous: ({})", continuous_ok);
        std::println("  Solution: {}", s_continuous);
        std::println("  Parent:   {}", p_continuous);
        std::println("Continuous Active: ({})", cactive_ok);
        std::println("  Solution: {}", s_cactive);
        std::println("  Parent:   {}", p_cactive);
        std::println("Quality: ({})", quality_ok);
        std::println("  Solution: {}", s_quality);
        std::println("  Parent:   {}", p_quality);
        std::abort();
      }
    }
  };

  // checks that the fitness matches the solution
  void check_fitness_invariant(Rng& rng, SolutionSet& set, std::string_view info) {
    SolutionSet copy;
    for (usize i = 0; i < set.size(); i++) {
      copy.add(set[i]);
    }
    std::vector<usize> indices(copy.size());
    std::iota(indices.begin(), indices.end(), 0);
    problem.evaluate(rng, copy, indices);

    for (auto i : indices) {
      auto expected = copy[i].quality(), actual = set[i].quality();
      bool definitely_different =
          (expected.objectives.array().isFinite() != actual.objectives.array().isFinite()).any();
      if (expected.objectives.array().isFinite().all()) {
        definitely_different |= ((expected.objectives - actual.objectives).array().abs() >= 1e-6).any();
      }
      definitely_different |=
          isna(expected.constraint_value) != isna(actual.constraint_value) ||
          (!isna(expected.constraint_value) && std::abs(expected.constraint_value - actual.constraint_value) >= 1e-6);
      if (definitely_different) {
        std::println("{}", info);
        std::println("Fitness invariant violated at index {}: ", i);
        std::println("Expected: {} / '{}'", problem.format_solution(copy[i]), problem.fitness().format(expected));
        std::println("Actual:   {} / '{}'", problem.format_solution(set[i]), problem.fitness().format(actual));
        std::abort();
      }
    }
  };

  // Returns whether a solution should be accepted or not. The parameter `strict` determines if random walks in neutral
  // fitness landscape are allowed or not.
  bool accept_and_update_archive(usize idx, std::optional<usize> objective, bool strict) {
    Ordering o = problem.fitness().cmp(solutions[idx].quality(), parents[idx].quality(), objective);

    if (o == Ordering::Worse) {
      return false;
    }

    bool non_dominated = local_archive->update(solutions[idx], strict);

    if (options.strict_elite_acceptance && objective.has_value() && idx == so_elite_idx[objective.value()]) {
      return o == Ordering::Better;
    }

    // if strict: we want clear improvements,
    // i.e. better in SO or at least non-dominated in MO
    return !strict || o == Ordering::Better || (!objective.has_value() && non_dominated);
  };

  u64 initialize(Rng& rng) {
    // reset the local archive
    local_archive->clear();
    local_archive->unsync_all();
    local_archive->sync_with(global_archive);

    // populate solution, donor and backup buffers
    solutions_to_evaluate.resize(size);
    std::iota(solutions_to_evaluate.begin(), solutions_to_evaluate.end(), 0);

    solutions.reserve(size);
    problem.add_random(rng, solutions, size);
    assert(solutions.size() == size);
    problem.evaluate(rng, solutions, solutions_to_evaluate);

    donors.clear();
    donors.reserve(size);
    parents.clear();
    parents.reserve(size);
    assert(donors.size() == 0);
    assert(parents.size() == 0);
    for (usize i = 0; i < size; i++) {
      donors.add(solutions[i]);
      parents.add(solutions[i]);
      local_archive->update(solutions[i], true);
    }

    // other state
    solution_nis.clear();
    solution_nis.resize(size, 0);

    so_elite_idx.resize(num_clusters);

    no_improvement_stretch = 0;
    no_evaluations_performed = false;
    iterations_since_last_gradient_step = 0;

    solutions_to_improve.reserve(size);
    subsets.resize(size);
    perm.reserve(size);

    // This callback is needed to support learning the linkage
    // normalization matrix from https://arxiv.org/pdf/1904.02050
    // and to tell the linkage model about how many variables there are in case that was not set beforehand
    discrete_model->init(rng, problem, solutions, VariableSet::Discrete);

    discrete_evaluations = 0.0;
    continuous_evaluations = 0.0;

    return solutions_to_evaluate.size();
  };

  u64 discrete_gom_step(Rng& rng, usize subset_idx) {
    std::vector<usize> donor_pool;
    {
      usize max_donor_pool_size = cluster_donors[0].size();
      assert(max_donor_pool_size > 0);
      for (usize k = 1; k < num_clusters; k++) {
        assert(cluster_donors[k].size() > 0);
        max_donor_pool_size = std::max(max_donor_pool_size, cluster_donors[k].size());
      }
      donor_pool.resize(max_donor_pool_size);
      std::iota(donor_pool.begin(), donor_pool.end(), 0);
    }

    solutions_to_evaluate.clear();

    // TODO parallel?
    for (usize i = 0; i < solutions.size(); i++) {
      auto k = solution_clusters[i];
      auto fos_idx = subset_orders(i, subset_idx);

      // due to filtering/max_subset_size, some clusters might have more
      // subsets...
      if (fos_idx < cluster_FOS[k].size()) {
        subsets[i] = &cluster_FOS[k][fos_idx];
        assert(subsets[i]->discrete.size() > 0);

        // the library does donor search, so this also is added behind a flag to allow fair comparisons to the
        // reference version...
        usize max_donor_search_iterations = std::min(options.donor_search_proportion, 1.0) * cluster_donors[k].size();
        usize donor_idx, donor_pool_idx = 0;

        bool evaluation_needed, anything_changed;
        do {
          // do a partial Fisher-Yates shuffle
          std::swap(donor_pool[donor_pool_idx],
                    donor_pool[std::uniform_int_distribution<usize>(donor_pool_idx, donor_pool.size() - 1)(rng)]);

          donor_idx = donor_pool[donor_pool_idx++];
          if (donor_idx >= cluster_donors[k].size() || i == cluster_donors[k][donor_idx]) {
            continue;
          }

          std::tie(evaluation_needed, anything_changed) = solutions[i].inherit(
              donors[cluster_donors[k][donor_idx]], *subsets[i], problem.always_inherit_continuous());

          if (evaluation_needed) {  // parent will be updated during acceptance
            solutions_to_evaluate.push_back(i);
          } else if (anything_changed) {  // no acceptance, parent has to be updated now
            parents[i] = solutions[i];
          }
        } while (!evaluation_needed && donor_pool_idx < max_donor_search_iterations);
      }
    }

    if (solutions_to_evaluate.empty())
      return 0;

    problem.evaluate_partial(rng, solutions, parents, subsets, solutions_to_evaluate);

    // acceptance happens after one step for all solutions rather than the
    // default of all steps for one solution after the other
    std::shuffle(solutions_to_evaluate.begin(), solutions_to_evaluate.end(), rng);
    for (usize i : solutions_to_evaluate) {
      auto k = solution_clusters[i];
      std::optional<usize> objective = k < problem.fitness().num_objectives() ? std::make_optional(k) : std::nullopt;

      if (!accept_and_update_archive(i, objective,
                                     /* strict */ false)) {
        // solutions[i].reject(
        //     parents[i], problem.always_inherit_continuous(),
        //     std::nullopt);  // *subsets[i]); // TODO do the more granular update once the LS terms are handled better
        solutions[i] = parents[i];
      } else {
        solution_changed[i] = true;
        parents[i] = solutions[i];
      }
    }

    return solutions_to_evaluate.size();
  }

  // TODO so far this only does discrete FI, but it should be mixed as in the other version...
  template <typename T>
  u64 forced_improvements(Rng& rng, T& should_terminate, usize max_discrete_subset_count) {
    u64 evaluations = 0;

    solutions_to_improve.clear();
    for (usize i = 0; i < size; i++) {
      if (!solution_changed[i] || solution_nis[i] > max_nis) {
        solutions_to_improve.push_back(i);

        // ensure we use a different order
        std::shuffle(subset_orders.row(i).begin(), subset_orders.row(i).end(), rng);
      }
    }

    std::vector<usize> eval2improve_idx;
    eval2improve_idx.reserve(solutions_to_improve.size());

    usize subset_idx = 0;
    while (!solutions_to_improve.empty() && subset_idx < max_discrete_subset_count) {
      eval2improve_idx.clear();
      solutions_to_evaluate.clear();

      // TODO parallel?
      for (usize j = 0; j < solutions_to_improve.size(); j++) {
        auto i = solutions_to_improve[j];
        auto k = solution_clusters[i];

        auto fos_idx = subset_orders(i, subset_idx);

        // due to filtering/max_subset_size, some clusters might have more
        // subsets...
        if (fos_idx < cluster_FOS[k].size()) {
          subsets[i] = &cluster_FOS[k][fos_idx];

          auto [evaluation_needed, anything_changed] =
              solutions[i].inherit(k < problem.fitness().num_objectives() ? global_archive.so_solution(k)
                                                                          : global_archive.random_solution(rng),
                                   *subsets[i], problem.always_inherit_continuous());
          if (evaluation_needed) {  // parent will be updated during acceptance
            eval2improve_idx.push_back(j);
            solutions_to_evaluate.push_back(i);
          } else if (anything_changed) {  // no acceptance, so we need to update the parent
            parents[i] = solutions[i];
          }
        }
      }

      problem.evaluate_partial(rng, solutions, parents, subsets, solutions_to_evaluate);
      evaluations += solutions_to_evaluate.size();

      // accept in a random order and remove the improved solutions from the
      // worklist
      {
        perm.resize(solutions_to_evaluate.size());
        std::iota(perm.begin(), perm.end(), 0);
        std::shuffle(perm.begin(), perm.end(), rng);

        solutions_to_evaluate.clear();
        auto& indices_to_remove = solutions_to_evaluate;
        for (usize j : perm) {
          auto i = solutions_to_improve[eval2improve_idx[j]];

          auto k = solution_clusters[i];
          std::optional<usize> objective =
              k < problem.fitness().num_objectives() ? std::make_optional(k) : std::nullopt;

          if (!accept_and_update_archive(i, objective,
                                         /* strict */ true)) {
            // solutions[i].reject(parents[i], problem.always_inherit_continuous(),
            //                     std::nullopt);  // *subsets[i]); // TODO do the more granular update once the LS
            //                     terms
            //                                     // are handled better
            solutions[i] = parents[i];
          } else {
            solution_changed[i] = true;
            parents[i] = solutions[i];

            // this solution can stop forced improvements
            indices_to_remove.push_back(eval2improve_idx[j]);
          }
        }
        // sort indices in reverse - otherwise removing smaller indices might
        // cause invalid memory accesses when trying to remove a larger index
        // later on...
        std::sort(indices_to_remove.begin(), indices_to_remove.end(), std::greater{});
        for (usize i : indices_to_remove) {
          std::swap(solutions_to_improve[i], solutions_to_improve.back());
          solutions_to_improve.pop_back();
        }
      }

      subset_idx++;

      if (should_terminate(evaluations).has_value()) {
        return evaluations;
      }
    }

    // replace any remaining solutions with non-dominated solutions from the
    // archive

    // TODO parallel
    for (usize i : solutions_to_improve) {
      auto k = solution_clusters[i];
      solutions[i] =
          k < problem.fitness().num_objectives() ? global_archive.so_solution(k) : global_archive.random_solution(rng);
      parents[i] = solutions[i];
    }

    return evaluations;
  };

  template <typename S>
  void mutate_continuous(Rng& rng, S& solution, bool& any_active_changed, Subset& changed) const {
    thread_local static std::uniform_real_distribution<double> U(0.0, 1.0);
    thread_local static std::normal_distribution<double> N(0.0, 1.0);

    any_active_changed = false;
    if (options.continuous_mutation_probability > 0.0) {
      if (changed.continuous.empty()) {
        changed.continuous.reserve(solution.num_continuous());

        for (usize i = 0; i < solution.num_continuous(); i++) {
          if (U(rng) < options.continuous_mutation_probability) {
            solution.continuous_values()(i) +=
                N(rng) * options.continuous_mutation_temperature * solution.continuous_values()(i);
            changed.continuous.push_back(i);
            any_active_changed |= solution.continuous_active()(i);
          }
        }
      } else {
        for (usize i : changed.continuous) {
          if (U(rng) < options.continuous_mutation_probability) {
            solution.continuous_values()(i) +=
                N(rng) * options.continuous_mutation_temperature * solution.continuous_values()(i);
            any_active_changed |= solution.continuous_active()(i);
          }
        }
      }
    }
  }

  u64 continuous_mutation_step(Rng& rng) {
    solutions_to_evaluate.clear();
    assert(options.continuous_mutation_probability > 0.0);

    std::vector<Subset> _subsets(solutions.size());

    // TODO parallel?
    for (usize i = 0; i < solutions.size(); i++) {
      bool evaluation_needed;
      mutate_continuous(rng, solutions[i], evaluation_needed, _subsets[i]);
      subsets[i] = &_subsets[i];

      if (evaluation_needed) {
        solutions_to_evaluate.push_back(i);
      }
    }

    if (solutions_to_evaluate.empty())
      return 0;

    problem.evaluate_partial(rng, solutions, parents, subsets, solutions_to_evaluate);

    std::shuffle(solutions_to_evaluate.begin(), solutions_to_evaluate.end(), rng);
    for (usize i : solutions_to_evaluate) {
      auto k = solution_clusters[i];
      std::optional<usize> objective = k < problem.fitness().num_objectives() ? std::make_optional(k) : std::nullopt;

      if (!accept_and_update_archive(i, objective,
                                     /* strict */ false)) {
        // solutions[i].reject(
        //     parents[i], problem.always_inherit_continuous(),
        //     std::nullopt);  //  *subsets[i]); // TODO do the more granular update once the LS terms are handled
        //     better
        solutions[i] = parents[i];
      } else {
        // solution_changed[i] = true;
        parents[i] = solutions[i];
      }
    }

    return solutions_to_evaluate.size();
  }

  u64 gradient_step(Rng& rng) {
    solutions_to_evaluate.clear();

    // the mutation before doing a gradient step is decoupled from the other mutation operator
    // - i.e. it should be possible to enable randomization before the gradient optimization without having to also
    // enable the other mutation operator. To make this work, the continuous_mutation_probability is temporarily
    // overwritten
    auto backup_mutation_probability = options.continuous_mutation_probability;
    options.continuous_mutation_probability = 1.0;

    // TODO parallel?
    for (usize i = 0; i < solutions.size(); i++) {
      bool evaluation_needed;
      if (options.mutate_before_gradient_step) {
        Subset _;
        mutate_continuous(rng, solutions[i], evaluation_needed, _);
      } else {
        evaluation_needed = solutions[i].continuous_active().any();
      }

      if (evaluation_needed) {
        solutions_to_evaluate.push_back(i);
      }
    }

    options.continuous_mutation_probability = backup_mutation_probability;

    // If no solution has active continuous values, there is nothing more to do
    if (solutions_to_evaluate.empty())
      return 0;

    auto [changed_indices, evaluations] =
        problem.gradient_steps(rng, solutions, parents, solutions_to_evaluate, options.gradient_step_count);

    // acceptance is still needed since the gradient step isn't guaranteed to be an improvement - e.g. too large
    // steps can be regressions
    for (usize i : changed_indices) {
      auto k = solution_clusters[i];
      std::optional<usize> objective = k < problem.fitness().num_objectives() ? std::make_optional(k) : std::nullopt;

      if (!accept_and_update_archive(i, objective,
                                     /* strict */ false)) {
        // solutions[i].reject(parents[i], problem.always_inherit_continuous(), std::nullopt);
        solutions[i] = parents[i];
      } else {
        // solution_changed[i] = true;
        parents[i] = solutions[i];
      }
    }

    return evaluations;
  }

  // state requiring initialization

  InstanceBase& problem;
  ArchiveBase& global_archive;
  std::unique_ptr<LinkageModelBase> discrete_model;
  RvState rv_state;
  PopulationOptions options;
  std::unique_ptr<ArchiveBase> local_archive;

  // options

  usize size;
  usize num_clusters;
  usize donor_pool_size;
  usize max_nis;

  // state that is required across generations (absolutely needs to be stored)

  double discrete_evaluations = 0.0;
  double continuous_evaluations = 0.0;
  usize no_improvement_stretch;
  bool no_evaluations_performed;
  usize iterations_since_last_gradient_step;

  std::vector<usize> so_elite_idx;

  SolutionSet donors;  // previous population
  SolutionSet solutions;

  std::vector<usize> solution_nis;
  std::vector<usize> solution_clusters;               // solution -> cluster, previous values
                                                      // required for cluster registration
  std::vector<std::vector<usize>> cluster_solutions;  // cluster -> solutions

  // buffers that are re-used where possible

  SolutionSet parents;                             // gom backups
  std::vector<FOS> cluster_FOS;                    // cluster -> FOS
  std::vector<std::vector<usize>> cluster_donors;  // cluster -> donors
  std::vector<bool> solution_changed;
  std::vector<usize> solutions_to_evaluate;
  std::vector<usize> solutions_to_improve;
  std::vector<usize> perm;
  Mat<usize> subset_orders;            // per solution subset permutations
  std::vector<const Subset*> subsets;  // pointers because 1. we want to avoid copies and 2. the view
                                       // should be nullable
};

class MixedGOMEA : public MethodBase {
 public:
  MixedGOMEA(PopulationOptions population_options = PopulationOptions(),
             RvOptions rv_options = RvOptions(),
             IMSOptions ims_options = IMSOptions(),
             std::shared_ptr<LinkageModelBase> discrete_model = std::make_shared<LinkageTreeFOS>(),
             std::shared_ptr<LinkageModelBase> continuous_model = std::make_shared<FullFOS>(),
             std::shared_ptr<RvSamplingModelBase> sampling_model = std::make_shared<AMaLGaMSamplingModel>(),
             std::string repr = "aos")
      : population_options(population_options),
        rv_options(rv_options),
        ims_options(ims_options),
        ims_runner(std::nullopt),
        discrete_model(discrete_model),
        continuous_model(continuous_model),
        sampling_model(sampling_model),
        repr(repr) {};

  std::tuple<std::shared_ptr<ArchiveBase>, TerminationStatus> run(
      InstanceBase& problem,
      const Budget& budget,
      std::optional<u64> seed = std::nullopt,

      std::optional<usize> population_size = std::nullopt) override {
    // MAYBE add a check to ensure the desired population can hold the solution
    // type, but how? One option that DOES NOT WORK is to 1. get a solution of
    // the actual type with problem.add_random, then convert to the desired set
    // and back - if that works, everything is good. The problem is that all
    // solutionsets convert a `const SolutionSetBase&` to the internal storage
    // type, so if there is more information it is already lost earlier... => Is
    // there even a way to check before evaluation? If not, just ignore this and
    // let the first evaluation fail due to a type mismatch...
    if (repr == "soa_cols") {
      ims_runner = IMS<Population<SoASet<Eigen::ColMajor>>>(
          std::bind_front(&MixedGOMEA::create_population<SoASet<Eigen::ColMajor>>, this), ims_options);
      return std::get<IMS<Population<SoASet<Eigen::ColMajor>>>>(*ims_runner)
          .run(problem, budget, seed, population_size);
    } else if (repr == "soa_rows") {
      ims_runner = IMS<Population<SoASet<Eigen::RowMajor>>>(
          std::bind_front(&MixedGOMEA::create_population<SoASet<Eigen::RowMajor>>, this), ims_options);
      return std::get<IMS<Population<SoASet<Eigen::RowMajor>>>>(*ims_runner)
          .run(problem, budget, seed, population_size);
    } else {
      ims_runner = IMS<Population<AoSSet>>(std::bind_front(&MixedGOMEA::create_population<AoSSet>, this), ims_options);
      return std::get<IMS<Population<AoSSet>>>(*ims_runner).run(problem, budget, seed, population_size);
    }
  };

  std::optional<u64> current_generation() const override {
    if (!ims_runner.has_value()) {
      return std::nullopt;
    }
    return std::visit([](const auto& r) { return r.current_generation(); }, ims_runner.value());
  };

 private:
  template <typename SolutionSet>
  Population<SolutionSet> create_population(InstanceBase& problem,
                                            ArchiveBase& global_archive,
                                            usize size,
                                            usize num_clusters) {
    return Population<SolutionSet>(problem, global_archive, *discrete_model, *continuous_model, *sampling_model, size,
                                   num_clusters, population_options, rv_options);
  };

  PopulationOptions population_options;
  RvOptions rv_options;
  IMSOptions ims_options;
  std::optional<std::variant<IMS<Population<AoSSet>>,
                             IMS<Population<SoASet<Eigen::ColMajor>>>,
                             IMS<Population<SoASet<Eigen::RowMajor>>>>>
      ims_runner;
  std::shared_ptr<LinkageModelBase> discrete_model;
  std::shared_ptr<LinkageModelBase> continuous_model;
  std::shared_ptr<RvSamplingModelBase> sampling_model;
  std::string repr;
};
};  // namespace goblin

#endif /* _GOBLIN_MIXED_GOMEA_H */
