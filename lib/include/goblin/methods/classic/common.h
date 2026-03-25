#pragma once
#ifndef _GOBLIN_CLASSIC_COMMON_H
#define _GOBLIN_CLASSIC_COMMON_H

#include <cmath>
#include <numbers>
#include <memory>
#include <optional>
#include <tuple>
#include <utility>
#include <algorithm>
#include <random>
#include <numeric>
#include <stdexcept>
#include <iterator>

#include "goblin/lib/method.h"
#include "goblin/lib/algorithms/mo.h"
#include "goblin/lib/algorithms/subset_selection.h"
#include "goblin/gp/instance.h"

namespace goblin {
namespace classic {
class SelectionStrategyBase {
 public:
  virtual std::vector<usize> select(Rng& rng,
                                    const FitnessBase& fitness,
                                    const SolutionSetBase& solutions,
                                    usize target_size) const = 0;

  virtual ~SelectionStrategyBase() = default;
};

class TournamentSelection : public SelectionStrategyBase {
  usize tournament_size;
  bool with_replacement;

 public:
  TournamentSelection(usize tournament_size = 2, bool with_replacement = false)
      : tournament_size(tournament_size), with_replacement(with_replacement) {
    if (tournament_size < 2) {
      throw std::runtime_error("Tournament size must be greater than 1.");
    }
  };

  std::vector<usize> select(Rng& rng,
                            const FitnessBase& fitness,
                            const SolutionSetBase& solutions,
                            usize target_size) const override final {
    std::vector<usize> selection;
    selection.reserve(target_size);

    std::vector<usize> candidates(tournament_size);

    std::uniform_int_distribution<usize> U(0, solutions.size() - 1);

    std::vector<usize> perm;
    if (!with_replacement) {
      perm = permute(rng, solutions.size());
    }

    usize perm_idx = 0;
    while (selection.size() < target_size) {
      // fill candidate pool with/without replacement
      candidates.clear();
      while (candidates.size() < tournament_size) {
        if (with_replacement) {
          candidates.push_back(U(rng));
        } else {
          if (perm_idx >= perm.size()) {
            std::shuffle(perm.begin(), perm.end(), rng);
            perm_idx = 0;
          }
          candidates.push_back(perm[perm_idx++]);
        }
      }

      // add winner to selection
      std::sort(candidates.begin(), candidates.end(), [&solutions, &fitness](usize lhs, usize rhs) {
        return fitness.cmp(solutions[lhs].quality(), solutions[rhs].quality(), std::nullopt) == Ordering::Better;
      });
      selection.push_back(candidates[0]);
    }

    return selection;
  };
};

class TruncationSelection : public SelectionStrategyBase {
 public:
  std::vector<usize> select(Rng& rng,
                            const FitnessBase& fitness,
                            const SolutionSetBase& solutions,
                            usize target_size) const override final {
    std::vector<usize> selection;

    if (solutions.size() <= target_size) {
      selection.resize(solutions.size());
      std::iota(selection.begin(), selection.end(), 0);
      return selection;
    }

    if (fitness.num_objectives() < 2) {
      // single-objective: sort, then truncate
      selection.resize(solutions.size());
      std::iota(selection.begin(), selection.end(), 0);
      std::sort(selection.begin(), selection.end(), [&solutions, &fitness](usize lhs, usize rhs) {
        return fitness.cmp(solutions[lhs].quality(), solutions[rhs].quality(), std::nullopt) == Ordering::Better;
      });
      selection.resize(target_size);
    } else {
      // multi-objective: non-dominated sorting, then add/truncate fronts until the target size is reached
      auto [ranks, fronts] = non_dominated_sorting(
          [&](usize lhs, usize rhs) {
            return fitness.cmp(solutions[lhs].quality(), solutions[rhs].quality(), std::nullopt);
          },
          solutions.size());

      selection.reserve(target_size);
      for (auto& front : fronts) {
        if (selection.size() >= target_size) {
          break;
        }
        if (selection.size() + front.size() <= target_size) {
          for (usize i : front) {
            selection.push_back(i);
          }
        } else {
          // for the last front, select based on objective space diversity
          std::vector<usize> f2s(front.begin(), front.end());
          auto [selected, _] = greedy_scattered_subset_selection(
              [&](const usize lhs, const usize rhs) {
                return fitness.distance(solutions[f2s[lhs]].quality(), solutions[f2s[lhs]].quality(), std::nullopt);
              },
              /* pool_size = */ f2s.size(),
              /* target_size = */ target_size - selection.size(),
              /* initial = */ std::uniform_int_distribution<usize>(0, f2s.size() - 1)(rng));

          for (usize i : selected) {
            selection.push_back(f2s[i]);
          }
        }
      }
    }

    return selection;
  };
};

class pHVCSelection : public SelectionStrategyBase {
  std::unique_ptr<QualityBase> reference_point;

 public:
  pHVCSelection(std::unique_ptr<QualityBase> reference_point) : reference_point(std::move(reference_point)) {};

  std::vector<usize> select(Rng& rng,
                            const FitnessBase& fitness,
                            const SolutionSetBase& solutions,
                            usize target_size) const override final {
    std::vector<usize> selection;
    selection.reserve(target_size);

    if (solutions.size() <= target_size) {
      selection.resize(solutions.size());
      std::iota(selection.begin(), selection.end(), 0);
      return selection;
    }

    std::vector<usize> selection_pool(solutions.size());
    std::iota(selection_pool.begin(), selection_pool.end(), 0);

    std::vector<usize> best_per_objective(fitness.num_objectives(), 0);
    for (usize i = 1; i < solutions.size(); i++) {
      for (usize j = 0; j < fitness.num_objectives(); j++) {
        if (fitness.cmp(solutions[i].quality(), solutions[best_per_objective[j]].quality(), j) == Ordering::Better) {
          best_per_objective[j] = i;
        }
      }
    }
    std::vector<usize> worst_per_objective(fitness.num_objectives(), 0);
    for (usize i = 1; i < solutions.size(); i++) {
      for (usize j = 0; j < fitness.num_objectives(); j++) {
        bool worst_is_invalid = isna(fitness.distance(solutions[worst_per_objective[j]].quality(),
                                                      solutions[best_per_objective[j]].quality(), j));

        bool is_valid = !isna(fitness.distance(solutions[i].quality(), solutions[best_per_objective[j]].quality(), j));
        if (worst_is_invalid ||
            (is_valid &&
             fitness.cmp(solutions[i].quality(), solutions[worst_per_objective[j]].quality(), j) == Ordering::Worse)) {
          worst_per_objective[j] = i;
        }
      }
    }
    bool use_ref = false;
    std::vector<CType> o_norm(fitness.num_objectives());
    for (usize j = 0; j < fitness.num_objectives(); j++) {
      o_norm[j] =
          fitness.distance(solutions[best_per_objective[j]].quality(), solutions[worst_per_objective[j]].quality(), j);
      if (isna(o_norm[j]) || o_norm[j] <= 0.0) {
        use_ref = true;
      }
    }
    auto normed_dist_to_ref = [&](const QualityBase& point, const usize objective) {
      const auto& ref = solutions[worst_per_objective[objective]].quality();
      CType dist = fitness.distance(point, ref, objective) / o_norm[objective];
      bool point_is_worse = fitness.cmp(point, ref, objective) == Ordering::Worse;
      return point_is_worse ? -dist : dist;
    };

    // TODO instead of reference point, use worst in each dimension...
    // -> can't assemble, so call dispatch function...
    // TODO delete :/

    CType best_contribution;
    usize best_idx;
    while (selection.size() < target_size) {
      // find point with highest contribution
      best_idx = selection_pool.size();
      for (usize i = 0; i < selection_pool.size(); i++) {
        usize idx = selection_pool[i];
        CType contribution;
        if (use_ref) {
          // get pool without idx
          std::swap(selection_pool[i], selection_pool.back());
          selection_pool.pop_back();
          contribution = pHVC(solutions, fitness, selection_pool, solutions[idx].quality(), *reference_point);
          // restore pool
          selection_pool.push_back(selection_pool[i]);
          selection_pool[i] = idx;
        } else {
          contribution = pHVC_impl(
              [&](const usize s, const usize objective) {
                return normed_dist_to_ref(solutions[s < idx ? s : s + 1].quality(), objective);
              },
              [&](const usize objective) { return normed_dist_to_ref(solutions[idx].quality(), objective); },
              selection_pool.size() - 1, fitness.num_objectives());
        }

        if (best_idx >= selection_pool.size() || contribution > best_contribution) {
          best_idx = i;
          best_contribution = contribution;
        }
      }

      // add to selection and remove from pool
      selection.push_back(selection_pool[best_idx]);
      std::swap(selection_pool[best_idx], selection_pool.back());
      selection_pool.pop_back();
    }

    return selection;
  };
};

class EABase : public MethodBase {
  AoSSet population{};
  u64 generation{};

 protected:
  usize population_size{};

 public:
  EABase() = delete;
  EABase(usize population_size) : population_size(population_size) {};

  virtual u64 step(Rng& rng, InstanceBase& problem, SolutionSetBase& population, ArchiveBase& archive) const = 0;

  void set_population(const SolutionSetBase& population) {
    this->population_size = population.size();
    this->population.clear();
    for (usize i = 0; i < population.size(); i++) {
      this->population.add(population[i]);
    }
  }

  AoSSet get_population() { return population; }

  std::tuple<std::shared_ptr<ArchiveBase>, TerminationStatus> run(
      InstanceBase& problem,
      const Budget& budget,
      std::optional<u64> seed = std::nullopt,
      std::optional<usize> population_size = std::nullopt) override {
    usize n = population_size.value_or(this->population_size);

    generation = 0;
    u64 evaluations = n;
    std::chrono::high_resolution_clock::time_point t_start = std::chrono::high_resolution_clock::now();

    Rng rng = seeded_rng(seed);

    // create & evaluate initial population
    if (population.size() < n) {
      problem.add_random(rng, population, n - population.size());
    }

    std::vector<usize> solutions_to_evaluate;
    solutions_to_evaluate.reserve(n);
    auto worst = problem.archive_fitness().worst();
    for (usize i = 0; i < n; i++) {
      if (problem.fitness().cmp(population[i].quality(), *worst, std::nullopt) != Ordering::Better) {
        solutions_to_evaluate.push_back(i);
      }
    }
    problem.evaluate(rng, population, solutions_to_evaluate);

    auto archive = std::make_shared<UnboundedArchive>(problem.archive_fitness());
    for (usize i = 0; i < n; i++) {
      archive->update(population[i], false);
    }

    auto status = TerminationStatus::Running;
    while (true) {
      // check termination criterion
      auto s = budget.exhausted(generation, evaluations, std::chrono::high_resolution_clock::now() - t_start);
      if (s.has_value()) {
        status = s.value();
        break;
      }
      if (problem.target_reached(*archive)) {
        status = TerminationStatus::TargetReached;
        break;
      }

      evaluations += step(rng, problem, population, *archive);

      generation++;
    }

    return std::make_tuple(archive, status);
  };

  std::optional<u64> current_generation() const override { return generation; };
  std::optional<std::tuple<usize, u64>> current_population() const override {
    return std::make_tuple(population_size, generation);
  };
};

};  // namespace classic
};  // namespace goblin

#endif /* _GOBLIN_CLASSIC_COMMON_H */
