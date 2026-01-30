#pragma once
#ifndef _GOBLIN_LIB_IMS_H
#define _GOBLIN_LIB_IMS_H

#include <chrono>
#include <cstddef>
#include <functional>
#include <limits>
#include <optional>
#include <print>
#include <random>
#include <type_traits>
#include <utility>

#include "goblin/lib/archive.h"
#include "goblin/lib/budget.h"
#include "goblin/lib/instance.h"
#include "goblin/lib/method.h"
#include "goblin/lib/rng.h"
#include "goblin/bench/tracked.h"

namespace goblin {

struct IMSOptions {
  usize initial_population_size = 2;
  usize max_num_populations = 25;
  usize subgeneration_factor = 4;
  bool restart_stale_populations = false;
  bool stop_covered_populations = false;
  usize initial_num_clusters = 1;
  std::optional<usize> archive_capacity = 100;
  bool so_parameter_space_clustering = false;  // TODO remove or implement parameter space clustering
  usize additional_clusters_per_start = 1;
  std::optional<usize> generations_without_improvement_until_restart = std::nullopt;

  std::optional<std::string> population_logfile = std::nullopt;
  std::string population_log_resolution = "archive";
};

template <typename P>
class IMS final : public MethodBase {
  using C = std::function<
      P(InstanceBase& /* problem */, ArchiveBase& /* global_archive */, usize /* size */, usize /* num_clusters
                                                                                                 */
        )>;

 public:
  IMS(C create_population, IMSOptions options = IMSOptions())
      : create_population(create_population), options(options), total_generations(0) {};

  std::tuple<std::shared_ptr<ArchiveBase>, TerminationStatus> run(
      InstanceBase& problem,
      const Budget& budget,
      std::optional<u64> seed = std::nullopt,
      std::optional<usize> population_size = std::nullopt) override final {
    bool is_multi_objective = problem.num_objectives() > 1;
    IMSOptions opts = options;
    if (opts.initial_num_clusters == 0) {
      opts.initial_num_clusters = problem.num_objectives() + (is_multi_objective ? 1 : 0);
    }

    if (opts.initial_population_size <= opts.initial_num_clusters) {
      opts.initial_population_size *= opts.initial_num_clusters;
    }

    if (population_size.has_value()) {
      opts.initial_population_size = population_size.value();
      opts.max_num_populations = 1;
    }

    if (!opts.so_parameter_space_clustering && !is_multi_objective) {
      opts.additional_clusters_per_start = 0;
    }

    Rng rng = seeded_rng(seed);
    auto archive =
        opts.archive_capacity.has_value() && opts.archive_capacity.value() > 0
            ? std::static_pointer_cast<ArchiveBase>(std::make_shared<UnboundedArchive>(problem.archive_fitness()))
            : std::static_pointer_cast<ArchiveBase>(
                  std::make_shared<AdaptiveGridArchive>(problem.archive_fitness(), opts.archive_capacity.value()));

    std::vector<P> populations;
    populations.reserve(opts.max_num_populations);
    sizes.clear();
    sizes.reserve(opts.max_num_populations);
    generations.clear();
    generations.reserve(opts.max_num_populations);
    std::vector<bool> running;
    running.reserve(opts.max_num_populations);
    std::vector<usize> generations_since_last_improvement;
    generations_since_last_improvement.reserve(opts.max_num_populations);

    total_generations = 0;
    u64 evaluations = 0;
    std::chrono::high_resolution_clock::time_point t_start = std::chrono::high_resolution_clock::now();

    auto should_terminate = [&](usize additional_evaluations = 0,
                                bool check_external_criterion = false) -> std::optional<TerminationStatus> {
      std::chrono::high_resolution_clock::time_point now = std::chrono::high_resolution_clock::now();
      std::chrono::nanoseconds elapsed = now - t_start;

      auto status = check_external_criterion
                        ? budget.exhausted_or_external_criterion_met(total_generations,
                                                                     evaluations + additional_evaluations, elapsed)
                        : budget.exhausted(total_generations, evaluations + additional_evaluations, elapsed);
      if (status.has_value()) {
        // std::println("{}", format_as(status.value()));
        return status;
      }

      if (problem.target_reached(*archive)) {
        // std::println("{}", format_as(TerminationStatus::TargetReached));
        return std::make_optional(TerminationStatus::TargetReached);
      }

      return std::nullopt;
    };

    auto any_running = [&]() {
      for (auto r : running) {
        if (r) {
          return true;
        }
      }
      return false;
    };

    Vec<CType> avg_dist_to_so_elite(opts.max_num_populations);
    p_idx = 0;
    while (!should_terminate(/* additional_evaluations = */ 0, /* check_external_criterion = */ true) &&
           (populations.size() < opts.max_num_populations || opts.restart_stale_populations || any_running())) {
      // init/restart population if necessary
      if (p_idx >= populations.size()) {
        usize size = opts.initial_population_size * std::pow(2, p_idx);
        sizes.push_back(size);
        populations.push_back(create_population(
            problem, *archive, size, opts.initial_num_clusters + p_idx * opts.additional_clusters_per_start));
        generations.push_back(0);
        generations_since_last_improvement.push_back(0);
        running.push_back(true);
      } else if (!running[p_idx] && opts.restart_stale_populations &&
                 (p_idx == opts.max_num_populations - 1 || (!opts.stop_covered_populations && is_multi_objective))) {
        populations[p_idx].restart();
        generations[p_idx] = 0;
        generations_since_last_improvement[p_idx] = 0;
        running[p_idx] = true;
      }

      // do a step, optionally terminate this or smaller ones
      generations[p_idx]++;  // this needs to always be increased, no matter if we do
                             // a step or not
      if (running[p_idx]) {
        archive->reset_change_count();
        evaluations += populations[p_idx].perform_generation(rng, should_terminate);
        total_generations++;

        if (opts.population_logfile.has_value()) {
          AoSSet p;  // copy is needed because p needs to be non-const, and that is the case because logging for the sr
                     // problem at this point in time might do a test set evaluation...
          if (opts.population_log_resolution == "archive") {
            const auto& a = populations[p_idx].archive();
            for (usize i = 0; i < a.size(); i++) {
              p.add(a[i]);
            }
          } else if (opts.population_log_resolution == "population") {
            const auto& s = populations[p_idx].get_solutions();
            for (usize i = 0; i < s.size(); i++) {
              p.add(s[i]);
            }
          } else {
            throw std::runtime_error("Unknown population log resolution.");
          }

          debug_log(problem, opts.population_logfile.value(), "", "", p);
        }

        if (archive->change_count() > 0) {
          generations_since_last_improvement[p_idx] = 0;
        } else {
          generations_since_last_improvement[p_idx]++;
        }

        archive->adapt();

        if (!is_multi_objective || opts.stop_covered_populations) {
          for (usize j = 0; j < p_idx; j++) {
            if (populations[p_idx].archive().covers(populations[j].archive())) {
              running[j] = false;
            }
          }
        }
        if (!is_multi_objective) {  //  && problem.num_discrete() == 0) {  // continuous only
          // since we only have relative comparisons, this roughly is equal to the usual avg fitness of larger
          // population is better condition
          avg_dist_to_so_elite(p_idx) = populations[p_idx].avg_dist_to_global_so_elite();
          for (usize j = 0; j < p_idx; j++) {
            if (avg_dist_to_so_elite(j) > avg_dist_to_so_elite(p_idx)) {
              running[j] = false;
            }
          }
        }
        if (populations[p_idx].converged() ||
            (opts.restart_stale_populations &&
             generations_since_last_improvement[p_idx] >
                 opts.generations_without_improvement_until_restart.value_or(total_generations))) {
          running[p_idx] = false;
        }
      }

      // go to the next population to do a step with
      p_idx = generations[p_idx] % opts.subgeneration_factor == 0 ? (p_idx + 1) % opts.max_num_populations : 0;
    }

    // std::println(
    //     "{} && ({} || {} || {}) - G: {} / #P: {} of {}", !should_terminate(),
    //     populations.size() < opts.max_num_populations,
    //     opts.restart_stale_populations, any_running(), total_generations,
    //     populations.size(), opts.max_num_populations);

    // for (usize p_idx = 0; p_idx < archive->size(); p_idx++) {
    //   std::println("{}", problem.format_solution((*archive)[p_idx]));
    // }

    return std::make_tuple(archive, should_terminate().value_or(TerminationStatus::Converged));
  };

  std::optional<u64> current_generation() const override final { return total_generations; };

  std::optional<std::tuple<usize, u64>> current_population() const override {
    return std::make_tuple(sizes[p_idx], generations[p_idx]);
  };

 private:
  C create_population;
  IMSOptions options;
  // The whole reason run is not a static method -
  // generation reporting needs to be accessible (via an IMS instance)
  u64 total_generations;

  usize p_idx;
  std::vector<usize> sizes;
  std::vector<u64> generations;
};

};  // namespace goblin

#endif /* _GOBLIN_LIB_IMS_H */
