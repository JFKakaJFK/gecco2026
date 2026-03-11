#pragma once
#ifndef _GOBLIN_SIMPLE_GA_H
#define _GOBLIN_SIMPLE_GA_H

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
          [&solutions, &fitness, &selection](usize lhs, usize rhs) {
            return fitness.cmp(solutions[lhs].quality(), solutions[rhs].quality(), std::nullopt);
          },
          solutions.size());

      selection.reserve(target_size);
      for (auto& front : fronts) {
        for (usize i : front) {
          selection.push_back(i);
        }
      }
    }

    return selection;
  };
};

/// Strategy used to generate the crossover masks to exchange information between two parents
class DiscreteCrossoverStrategyBase {
 public:
  virtual void crossover_masks(Rng& rng,
                               const SolutionBase& parent1,
                               const SolutionBase& parent2,
                               Subset& mask1,
                               Subset& mask2) const = 0;
  virtual ~DiscreteCrossoverStrategyBase() = default;
};

class UniformCrossover : public DiscreteCrossoverStrategyBase {
  double p_crossover{};

 public:
  UniformCrossover(double p_crossover = 0.5) : p_crossover(p_crossover) {
    if (p_crossover <= 0.0 || 1.0 <= p_crossover) {
      throw std::runtime_error("Crossover probability must be in (0,1) to perform variation, not copying!");
    }
  };

  void crossover_masks(Rng& rng,
                       const SolutionBase& parent1,
                       const SolutionBase& parent2,
                       Subset& mask1,
                       Subset& mask2) const override final {
    std::uniform_real_distribution<double> U(0.0, 1.0);
    for (usize i = 0; i < parent1.num_discrete(); i++) {
      if (U(rng) < p_crossover) {
        mask1.discrete.push_back(i);
        mask2.discrete.push_back(i);
      }
    }
  };
};

class NPointCrossover : public DiscreteCrossoverStrategyBase {
  usize num_points{};

 public:
  NPointCrossover(usize num_points = 1) : num_points(num_points) {
    if (num_points < 1) {
      throw std::runtime_error("At least one crossover point is needed to perform variation, not copying!");
    }
  };

  void crossover_masks(Rng& rng,
                       const SolutionBase& parent1,
                       const SolutionBase& parent2,
                       Subset& mask1,
                       Subset& mask2) const override final {
    // here the two endpoints are excluded to ensure not all values come from the same parent
    auto points = permute(rng, parent1.num_discrete() - 1);
    if (num_points < parent1.num_discrete()) {
      points.resize(num_points);
    }
    std::sort(points.begin(), points.end());

    // swap the indices between every other set of points
    for (usize i = 0; i < points.size(); i += 2) {
      // + 1 since the first real crossover point is between index 0 and 1, not before index 0
      usize start = points[i] + 1;
      usize end = i + 1 < points.size() ? points[i + 1] + 1 : parent1.num_discrete();
      for (usize j = start; j < end; j++) {
        mask1.discrete.push_back(j);
        mask2.discrete.push_back(j);
      }
    }
  };
};

class SimpleGA : public MethodBase {
 private:
  // options
  std::shared_ptr<SelectionStrategyBase> selection_strategy;
  std::shared_ptr<DiscreteCrossoverStrategyBase> crossover_strategy;
  usize population_size{};
  double p_mutation{};
  bool steady_state{};

  // run state
  u64 generation{};

  // temporary buffers
  mutable AoSSet offspring;
  mutable AoSSet parents;
  mutable std::vector<usize> solutions_to_evaluate;
  mutable std::vector<Subset> subsets;
  mutable std::vector<const Subset*> subset_refs;

 public:
  SimpleGA(usize population_size = 100,
           double p_mutation = 0.1,
           std::shared_ptr<DiscreteCrossoverStrategyBase> crossover_strategy = std::make_shared<UniformCrossover>(),
           bool steady_state =
               true,  // steady_state vs generational: select from P + O or just from O after generating more offspring?
           std::shared_ptr<SelectionStrategyBase> selection_strategy = std::make_shared<TournamentSelection>(4))
      : selection_strategy(selection_strategy),
        crossover_strategy(crossover_strategy),
        population_size(population_size),
        p_mutation(p_mutation),
        steady_state(steady_state) {
    if (population_size % 2 != 0) {
      throw std::runtime_error("Population size must be even!");
    }
    if (p_mutation >= 1.0) {
      throw std::runtime_error("A mutation rate of 100% performs random search!");
    }
    if (auto p = dynamic_cast<TruncationSelection*>(&*selection_strategy); p != nullptr && !steady_state) {
      // generational: need to select population_size parents -> no selection pressure with truncation selection
      throw std::runtime_error("Truncation selection is not compatible with a generational replacement scheme!");
    }
  };

  /// Mutates each discrete decision variable with p_mutation and returns the set of changed indices
  Subset mutate(Rng& rng, InstanceBase& problem, SolutionBase& solution, bool& any_active_changed) const {
    std::uniform_real_distribution<double> U(0.0, 1.0);

    Subset changed_indices;
    if (p_mutation > 0.0) {
      for (usize j = 0; j < problem.num_discrete(); j++) {
        std::uniform_int_distribution<DType> D(0, problem.discrete_domain_sizes()(j) - 1);
        if (U(rng) < p_mutation) {
          auto v = solution.discrete_values()(j);
          while (solution.discrete_values()(j) == v) {
            v = D(rng);
          }
          solution.discrete_values()(j) = v;
          any_active_changed |= solution.discrete_active()(j);

          changed_indices.discrete.push_back(j);
        }
      }
    }

    return changed_indices;
  };

  u64 step(Rng& rng, InstanceBase& problem, SolutionSetBase& population, ArchiveBase& archive) const {
    if (problem.num_discrete() < 1) {
      return 0;
    }
    std::uniform_real_distribution<double> U(0.0, 1.0);
    usize n = population.size();
    if (n % 2 != 0) {
      throw std::runtime_error("Population size must be even!");
    }

    // house keeping
    offspring.clear();
    parents.clear();
    solutions_to_evaluate.clear();
    solutions_to_evaluate.reserve(n);

    // The problem interface supports partial evaluations and provides information
    // on which decision variables are active (or not) in case the problem can have
    // conditionally inactive variables.
    // To take full advantage of such problem settings, partial evaluations are performed
    // and only on offspring solutions where the active variables changed.
    // The crossover masks/subset of changed variables between offspring and parent need
    // to be passed to the evaluation call to support this.
    subsets.resize(n);
    subset_refs.resize(n);
    for (usize i = 0; i < n; i++) {
      subsets[i].discrete.reserve(problem.num_discrete());
      subset_refs[i] = &subsets[i];
    }

    // parent selection (if generational)
    std::vector<usize> parent_indices;
    if (!steady_state) {
      // generational selection from P
      parent_indices = selection_strategy->select(rng, problem.fitness(), population, n);
    } else {
      // steady state: every solution gets offspring
      parent_indices = permute(rng, population.size());
    }

    // variation
    for (usize i = 0, idx1, idx2; i < n; i += 2) {
      idx1 = i;
      idx2 = i + 1;

      // get the parents
      const auto& parent1 = population[parent_indices[i]];
      const auto& parent2 = population[parent_indices[i + 1]];

      // add the offspring to the population (and a copy of the parent to support partial evaluations)
      offspring.add(parent1);
      parents.add(parent1);

      offspring.add(parent2);
      parents.add(parent2);

      // get the crossover masks
      auto& mask1 = subsets[i];
      auto& mask2 = subsets[i + 1];
      mask1.discrete.clear();
      mask2.discrete.clear();

      crossover_strategy->crossover_masks(rng, parent1, parent2, mask1, mask2);

      // crossover
      auto o1_needs_evaluation = std::get<0>(problem.inherit_discrete(offspring[idx1], parent2, mask1));
      auto o2_needs_evaluation = std::get<0>(problem.inherit_discrete(offspring[idx2], parent1, mask2));

      // mutation
      auto mutated_indices1 = mutate(rng, problem, offspring[idx1], o1_needs_evaluation);
      auto mutated_indices2 = mutate(rng, problem, offspring[idx2], o2_needs_evaluation);

      if (o1_needs_evaluation) {
        solutions_to_evaluate.push_back(idx1);
        mask1 = mask1.merge(mutated_indices1);
      }
      if (o2_needs_evaluation) {
        solutions_to_evaluate.push_back(idx2);
        mask2 = mask2.merge(mutated_indices2);
      }
    }

    // evaluation
    problem.evaluate_partial(rng, offspring, parents, subset_refs, solutions_to_evaluate);

    // archive update
    for (usize i : solutions_to_evaluate) {
      archive.update(offspring[i], false);
    }

    // offspring selection (if steady-state)
    if (steady_state) {
      // steady state selection from P + O
      for (usize i = 0; i < n; i++) {
        offspring.add(population[i]);
      }
      auto selection = selection_strategy->select(rng, problem.fitness(), offspring, n);
      population.clear();
      for (auto i : selection) {
        population.add(offspring[i]);
      }
    } else {
      population.clear();
      for (usize i = 0; i < n; i++) {
        population.add(offspring[i]);
      }
    }

    return solutions_to_evaluate.size();
  };

  std::tuple<std::shared_ptr<ArchiveBase>, TerminationStatus> run(InstanceBase& problem,
                                                                  const Budget& budget,
                                                                  std::optional<u64> seed,
                                                                  std::optional<usize> population_size) override final {
    usize n = population_size.value_or(this->population_size);
    if (n % 2 != 0) {
      throw std::runtime_error("Population size must be even!");
    }

    generation = 0;
    u64 evaluations = n;
    std::chrono::high_resolution_clock::time_point t_start = std::chrono::high_resolution_clock::now();

    Rng rng = seeded_rng(seed);

    // create & evaluate initial population
    AoSSet population;
    problem.add_random(rng, population, n);

    solutions_to_evaluate.resize(n);
    std::iota(solutions_to_evaluate.begin(), solutions_to_evaluate.end(), 0);
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

  std::optional<u64> current_generation() const override final { return generation; };
  std::optional<std::tuple<usize, u64>> current_population() const override final {
    return std::make_tuple(population_size, generation);
  };
};

class DEStrategyBase {
 public:
  virtual std::tuple<Solution, bool> trial_vector(Rng& rng,
                                                  InstanceBase& problem,
                                                  const SolutionSetBase& population,
                                                  const ArchiveBase& archive,
                                                  usize idx,
                                                  Subset& subset) const = 0;
  virtual ~DEStrategyBase() = default;
};

class Rand1Bin : public DEStrategyBase {
  // strategy for selectig first vector
  enum R0Strategy : u8 {
    Random,        // uniformally randomly
    Best,          // best so far
    CurrentToBest  // best - current
  };

  enum ScaleStrategy : u8 {
    Constant,  // F is constant
    Dither,    // F_i ~ N(F, 1) (F shared amongst variables)
    Jitter     // F_i ~ N(F, 1) (F re-sampled per variable)
  };

  double F{};                  // scale factor / differential weight
  double Cr{};                 // crossover probability
  R0Strategy r0_strategy{};    // what is the base vector?
  ScaleStrategy F_strategy{};  // how is F selected?

 public:
  Rand1Bin(double F = 0.8, double Cr = 0.9, std::string base = "best", std::string scale = "dither") : F(F), Cr(Cr) {
    if (F <= 0.0 || F > 2.0) {
      throw std::runtime_error("F must be in (0, 2]");
    }
    if (Cr < 0.0 || Cr > 1.0) {
      throw std::runtime_error("Cr must be in [0, 1]");
    }

    if (base == "best") {
      r0_strategy = R0Strategy::Best;
    } else if (base == "current-to-best") {
      r0_strategy = R0Strategy::CurrentToBest;
    } else if (base == "random") {
      r0_strategy = R0Strategy::Random;
    } else {
      throw std::runtime_error("Unknown strategy for selecting base vector: '" + base + "'");
    }

    if (scale == "constant") {
      F_strategy = ScaleStrategy::Constant;
    } else if (scale == "dither") {
      F_strategy = ScaleStrategy::Dither;
    } else if (scale == "jitter") {
      F_strategy = ScaleStrategy::Jitter;
    } else {
      throw std::runtime_error("Unknown scaling strategy '" + scale + "'");
    }
  };

  std::tuple<Solution, bool> trial_vector(Rng& rng,
                                          InstanceBase& problem,
                                          const SolutionSetBase& population,
                                          const ArchiveBase& archive,
                                          usize idx,
                                          Subset& subset) const override final {
    const usize n = population.size();
    if (n < 4) {
      throw std::runtime_error("DE requires a population size >= 4!");
    }
    const usize D = problem.num_continuous();

    std::uniform_int_distribution<usize> P(0, n - 1);
    std::uniform_real_distribution<double> U(0.0, 1.0);

    usize r0, r1, r2;
    do {
      r1 = P(rng);
    } while (r1 == idx);
    do {
      r2 = P(rng);
    } while (r2 == idx || r1 == r2);
    if (r0_strategy == R0Strategy::Random) {
      do {
        r0 = P(rng);
      } while (r0 == idx || r0 == r1 || r0 == r2);
    }

    Solution o = population[idx];

    usize jrand = std::uniform_int_distribution<usize>(0, D - 1)(rng);

    auto trial = o.continuous_values();
    const auto x_r0 = (r0_strategy == R0Strategy::Random ? population[r0] : archive.so_solution(0)).continuous_values();
    const auto x_r1 = population[r1].continuous_values();
    const auto x_r2 = population[r2].continuous_values();

    const auto lb = problem.continuous_lower_bounds();
    const auto ub = problem.continuous_upper_bounds();

    bool any_active_changed = false;

    std::normal_distribution<CType> dF(F, 1.0);
    CType F_actual = F_strategy == ScaleStrategy::Dither ? dF(rng) : F;
    for (usize j = 0; j < D; j++) {
      if (j == jrand || U(rng) < Cr) {
        if (F_strategy == ScaleStrategy::Jitter) {
          F_actual = dF(rng);
        }
        if (r0_strategy == R0Strategy::CurrentToBest) {
          trial(j) = trial(j) + F_actual * (x_r0(j) - trial(j)) + F * (x_r1(j) - x_r2(j));
        } else {
          trial(j) = x_r0(j) + F_actual * (x_r1(j) - x_r2(j));
        }

        trial(j) = std::clamp(trial(j), lb(j), ub(j));

        subset.continuous.push_back(j);
        any_active_changed |= o.continuous_active()(j);
      }
    }

    return std::make_tuple(o, any_active_changed);
  };
};

class DifferentialEvolution : public MethodBase {
 private:
  // options
  std::shared_ptr<DEStrategyBase> strategy;
  usize population_size{};

  // run state
  u64 generation{};

  // temporary buffers
  mutable AoSSet offspring;
  mutable std::vector<usize> solutions_to_evaluate;
  mutable std::vector<Subset> subsets;
  mutable std::vector<const Subset*> subset_refs;

 public:
  DifferentialEvolution(usize population_size = 100,
                        std::shared_ptr<DEStrategyBase> strategy = std::make_shared<Rand1Bin>())
      : strategy(strategy), population_size(population_size) {
    if (population_size < 4) {
      throw std::runtime_error("DE requires a population size >= 4!");
    }
  };

  u64 step(Rng& rng, InstanceBase& problem, SolutionSetBase& population, ArchiveBase& archive) const {
    if (problem.num_continuous() < 1) {
      return 0;
    }
    std::uniform_real_distribution<double> U(0.0, 1.0);
    usize n = population.size();
    if (n < 4) {
      throw std::runtime_error("DE requires a population size >= 4!");
    }

    // housekeeping
    offspring.clear();
    solutions_to_evaluate.clear();
    solutions_to_evaluate.reserve(n);

    // The problem interface supports partial evaluations and provides information
    // on which decision variables are active (or not) in case the problem can have
    // conditionally inactive variables.
    // To take full advantage of such problem settings, partial evaluations are performed
    // and only on offspring solutions where the active variables changed.
    // The crossover masks/subset of changed variables between offspring and parent need
    // to be passed to the evaluation call to support this.
    subsets.resize(n);
    subset_refs.resize(n);
    for (usize i = 0; i < n; i++) {
      subsets[i].continuous.reserve(problem.num_continuous());
      subset_refs[i] = &subsets[i];
    }

    // create trial and mutant vectors
    for (usize i = 0; i < n; i++) {
      subsets[i].continuous.clear();
      auto [trial, any_active_changed] = strategy->trial_vector(rng, problem, population, archive, i, subsets[i]);
      offspring.add(trial);

      if (any_active_changed) {
        solutions_to_evaluate.push_back(i);
      }
    }

    // evaluation
    problem.evaluate_partial(rng, offspring, population, subset_refs, solutions_to_evaluate);

    // acceptance
    for (usize i : solutions_to_evaluate) {
      if (problem.fitness().cmp(offspring[i].quality(), population[i].quality(), std::nullopt) != Ordering::Worse) {
        population[i] = offspring[i];

        archive.update(offspring[i], false);
      }
    }

    return solutions_to_evaluate.size();
  };

  std::tuple<std::shared_ptr<ArchiveBase>, TerminationStatus> run(InstanceBase& problem,
                                                                  const Budget& budget,
                                                                  std::optional<u64> seed,
                                                                  std::optional<usize> population_size) override final {
    usize n = population_size.value_or(this->population_size);

    generation = 0;
    u64 evaluations = n;
    std::chrono::high_resolution_clock::time_point t_start = std::chrono::high_resolution_clock::now();

    Rng rng = seeded_rng(seed);

    // create & evaluate initial population
    AoSSet population;
    problem.add_random(rng, population, n);

    solutions_to_evaluate.resize(n);
    std::iota(solutions_to_evaluate.begin(), solutions_to_evaluate.end(), 0);
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

  std::optional<u64> current_generation() const override final { return generation; };
  std::optional<std::tuple<usize, u64>> current_population() const override final {
    return std::make_tuple(population_size, generation);
  };
};

class PSOTopologyBase {
 public:
  virtual std::vector<usize> neighbours(Rng& rng, const SolutionSetBase& population, usize idx) const = 0;
  virtual ~PSOTopologyBase() = default;
};

class RingTopology : public PSOTopologyBase {
  usize num_neighbours{};

 public:
  /// Considers a population of size N as ring where the ends wrap (nodes 0 and N-1 are neighbours). The number of
  /// neighbours determines the (symmetric) reach of each node around the ring, where `num_neighbours >= floor(N-1) / 2`
  /// corresponds to a fully connected star topology.
  RingTopology(usize num_neighbours = 2) : num_neighbours(num_neighbours) {
    if (num_neighbours < 1) {
      throw std::runtime_error("The neighbourhood must extend to at least 1 neighbour in each direction.");
    }
  };

  std::vector<usize> neighbours(Rng& rng, const SolutionSetBase& population, usize idx) const {
    const usize N = population.size();
    const usize n = std::min(2 * num_neighbours + 1, N);
    std::vector<usize> nbs;
    if (n >= N) {
      nbs.resize(N);
      std::iota(nbs.begin(), nbs.end(), 0);
      return nbs;
    }

    nbs.reserve(n);
    isize i = static_cast<isize>(idx) - num_neighbours;
    while (nbs.size() < n) {
      if (i < 0) {
        i += N;
      } else if (i >= N) {
        i -= N;
      }
      nbs.push_back(i);
      i++;
    }

    return nbs;
  };
};

class PSOState : public SolutionExtension<PSOState> {
  std::unique_ptr<QualityBase> _previous_best_quality{};

 public:
  Vec<CType> velocity{};
  Vec<CType> previous_best{};

  const QualityBase& previous_best_quality() const { return *_previous_best_quality; }
  QualityBase& previous_best_quality() { return *_previous_best_quality; }

  void assign_previous_best_quality(const QualityBase& quality) { _previous_best_quality = quality.clone(); }

  PSOState() = default;
  ~PSOState() = default;
  PSOState(const PSOState& other)
      : velocity(other.velocity),
        previous_best(other.previous_best),
        _previous_best_quality(other._previous_best_quality->clone()) {};
  PSOState(PSOState&& other)
      : velocity(std::move(other.velocity)),
        previous_best(std::move(other.previous_best)),
        _previous_best_quality(std::move(other._previous_best_quality)) {};
  PSOState& operator=(const PSOState& other) {
    if (&other != this) {
      velocity = other.velocity;
      previous_best = other.previous_best;
      _previous_best_quality = other._previous_best_quality->clone();
    }
    return *this;
  }
  PSOState& operator=(PSOState&& other) {
    if (&other != this) {
      velocity = std::move(other.velocity);
      previous_best = std::move(other.previous_best);
      _previous_best_quality = std::move(other._previous_best_quality);
    }
    return *this;
  }

  std::unique_ptr<SolutionExtensionBase> clone() const override final { return std::make_unique<PSOState>(*this); };
};

class PSO : public MethodBase {
 private:
  // options
  std::shared_ptr<PSOTopologyBase> topology;
  usize population_size{};
  double inertia{};
  double cognitive{};
  double social{};
  bool generational{};

  // run state
  u64 generation{};

  // temporary buffers
  mutable std::vector<usize>
      solutions_to_evaluate;  // mutable because the buffer does not contain persistent state, and there is little point
                              // in re-allocating the memory each iteration

 public:
  PSO(usize population_size = 25,
      double inertia = 0.729,
      double cognitive = 1.494,
      double social = 1.494,
      bool generational = false,
      std::shared_ptr<PSOTopologyBase> topology = std::make_shared<RingTopology>())
      : topology(topology),
        population_size(population_size),
        inertia(inertia),
        cognitive(cognitive),
        social(social),
        generational(generational) {
    if (population_size < 2) {
      throw std::runtime_error("PSO requires a population size >= 2!");
    }
  };

  u64 step(Rng& rng, InstanceBase& problem, SolutionSetBase& population, ArchiveBase& archive) const {
    if (problem.num_continuous() < 1) {
      return 0;
    }
    std::uniform_real_distribution<double> U(0.0, 1.0);
    const usize n = population.size();
    const usize D = problem.num_continuous();
    if (n < 2) {
      throw std::runtime_error("PSO requires a population size >= 2!");
    }

    // housekeeping
    solutions_to_evaluate.clear();
    solutions_to_evaluate.reserve(n);

    // update the previous best value
    for (usize i = 0; i < n; i++) {
      // initialize if not set
      if (!population[i].has_extension(PSOState::type_key())) {
        PSOState ext;
        ext.velocity.resize(D);
        for (usize j = 0; j < D; j++) {
          ext.velocity(j) = 0.0;
        };
        ext.previous_best = population[i].continuous_values();
        ext.assign_previous_best_quality(population[i].quality());
        population[i].get_or_insert_extension(ext);
      } else {
        auto& ext = population[i].extension<PSOState>();
        if (problem.fitness().cmp(population[i].quality(), ext.previous_best_quality(), std::nullopt) !=
            Ordering::Worse) {
          ext.assign_previous_best_quality(population[i].quality());
          ext.previous_best = population[i].continuous_values();
        }
      }
    }

    // update the positions of each particle
    Vec<CType> r0 = Vec<CType>::Zero(D);
    Vec<CType> r1 = Vec<CType>::Zero(D);
    for (usize i = 0; i < n; i++) {
      auto& ext = population[i].extension<PSOState>();

      // social update
      auto neighbours = topology->neighbours(rng, population, i);
      if (neighbours.size() < 1) {
        throw std::runtime_error("Invalid, empty neighbourhood");
      }
      usize best_nb_idx = 0;
      for (usize j = 1; j < neighbours.size(); j++) {
        if (problem.fitness().cmp(population[neighbours[j]].extension<PSOState>().previous_best_quality(),
                                  population[neighbours[best_nb_idx]].extension<PSOState>().previous_best_quality(),
                                  std::nullopt) != Ordering::Worse) {
          best_nb_idx = j;
        }
      }

      // velocity update
      auto x = population[i].continuous_values();
      const auto pb = ext.previous_best;
      const auto gb = population[neighbours[best_nb_idx]].extension<PSOState>().previous_best;

      for (usize j = 0; j < D; j++) {
        r0(j) = U(rng);
        r1(j) = U(rng);
      }

      ext.velocity = inertia * ext.velocity + cognitive * r0.cwiseProduct(pb - x) + social * r1.cwiseProduct(gb - x);

      // position update
      x += ext.velocity;

      // clamp to boundaries
      const auto lb = problem.continuous_lower_bounds();
      const auto ub = problem.continuous_upper_bounds();
      x = x.cwiseMax(lb).cwiseMin(ub);

      if (generational) {
        solutions_to_evaluate.push_back(i);
      } else {
        solutions_to_evaluate = {i};
        problem.evaluate(rng, population, solutions_to_evaluate);
        archive.update(population[i], false);
      }
    }

    if (generational) {
      // evaluation (no partial evaluations since the velocity updates all variables at once)
      problem.evaluate(rng, population, solutions_to_evaluate);

      for (usize i : solutions_to_evaluate) {
        archive.update(population[i], false);
      }
    }

    return solutions_to_evaluate.size();
  };

  std::tuple<std::shared_ptr<ArchiveBase>, TerminationStatus> run(InstanceBase& problem,
                                                                  const Budget& budget,
                                                                  std::optional<u64> seed,
                                                                  std::optional<usize> population_size) override final {
    usize n = population_size.value_or(this->population_size);

    generation = 0;
    u64 evaluations = n;
    std::chrono::high_resolution_clock::time_point t_start = std::chrono::high_resolution_clock::now();

    Rng rng = seeded_rng(seed);

    // create & evaluate initial population
    AoSSet population;
    problem.add_random(rng, population, n);

    solutions_to_evaluate.resize(n);
    std::iota(solutions_to_evaluate.begin(), solutions_to_evaluate.end(), 0);
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

  std::optional<u64> current_generation() const override final { return generation; };
  std::optional<std::tuple<usize, u64>> current_population() const override final {
    return std::make_tuple(population_size, generation);
  };
};

enum class ESStrategy : u8 {
  /// Single variance for all variables
  SingleVariance,
  /// Separate variance for all variables
  MultipleVariance,
  /// Full covariance matrix
  FullVariance,
  /// Directional variance for one arbitrary direction, single variance in all other directions
  DirectedVariance
};

class ESStrategyParameters : public SolutionExtension<ESStrategyParameters> {
 public:
  ESStrategy strategy{};
  Vec<CType> parameters{};

  std::unique_ptr<SolutionExtensionBase> clone() const override final {
    return std::make_unique<ESStrategyParameters>(*this);
  };
};

class ES : public MethodBase {
 private:
  // options
  usize population_size{};
  usize num_parents{};
  usize num_offspring{};
  double epsilon{};
  std::optional<double> tau{};
  std::optional<double> tau_i{};
  double beta{};
  bool steady_state{};
  ESStrategy strategy{};

  // run state
  u64 generation{};

  // temporary buffers
  mutable AoSSet offspring;
  mutable std::vector<usize> solutions_to_evaluate;
  mutable std::vector<usize> parent_pool;

  void init_strategy_params(Rng& rng, SolutionBase& solution) const {
    if (solution.has_extension(ESStrategyParameters::type_key())) {
      return;
    }

    ESStrategyParameters ext;
    ext.strategy = strategy;
    usize num_params;
    switch (strategy) {
      case ESStrategy::SingleVariance:
        num_params = 1;
        break;
      case ESStrategy::MultipleVariance:
        num_params = solution.num_continuous();
        break;
      case ESStrategy::FullVariance:
        num_params = ((solution.num_continuous() + 1) * solution.num_continuous()) / 2;
        break;
      case ESStrategy::DirectedVariance:
        num_params = solution.num_continuous() + 1;
        break;
      default:
        throw std::runtime_error("Strategy initialization not implemented!");
    }
    ext.parameters.resize(num_params);

    std::normal_distribution<CType> N(0.0, 1.0);
    for (usize i = 0; i < num_params; i++) {
      ext.parameters(i) = N(rng);
    }

    solution.get_or_insert_extension(ext);
  };

  Solution recombine_parents(Rng& rng, const SolutionSetBase& parents) const {
    parent_pool.resize(parents.size());
    const usize np = std::min(parent_pool.size(), num_parents);
    std::iota(parent_pool.begin(), parent_pool.end(), 0);
    std::shuffle(parent_pool.begin(), parent_pool.end(), rng);

    Solution combined = parents[parent_pool[0]];
    auto& ext = combined.extension<ESStrategyParameters>();

    for (usize i = 1; i < np; i++) {
      combined.continuous_values() += parents[parent_pool[i]].continuous_values();
      const auto& p_ext = parents[parent_pool[i]].extension<ESStrategyParameters>();
      if (p_ext.strategy != ext.strategy) {
        throw std::runtime_error("Cannot recombine between different strategy types!");
      }
      ext.parameters += p_ext.parameters;
    }
    combined.continuous_values() /= static_cast<CType>(np);
    ext.parameters /= static_cast<CType>(np);

    return combined;
  }

  Solution mutate_and_sample(Rng& rng, const SolutionBase& parent) const {
    const usize l = parent.num_continuous();
    std::normal_distribution<CType> N(0.0, 1.0);

    Solution o = parent;
    auto& ext = o.extension<ESStrategyParameters>();

    Vec<CType> z = Vec<CType>::Zero(parent.num_continuous());

    if (ext.strategy == ESStrategy::SingleVariance) {
      // parameter update
      CType t = tau.value_or(std::sqrt(1.0 / l));
      ext.parameters(0) = std::max(ext.parameters(0) * std::exp(N(rng) * t), epsilon);

      // sample
      for (usize i = 0; i < l; i++) {
        z(i) = N(rng) * ext.parameters(0);
      }
    } else if (ext.strategy == ESStrategy::MultipleVariance) {
      // parameter update
      CType t_shared = N(rng) * tau.value_or(std::sqrt(1.0 / (2.0 * l)));
      CType t_i = tau_i.value_or(std::sqrt(1.0 / (2.0 * std::sqrt(l))));
      for (usize i = 0; i < l; i++) {
        ext.parameters(i) = std::max(ext.parameters(i) * std::exp(t_shared + N(rng) * t_i), epsilon);
      }

      // sample
      for (usize i = 0; i < l; i++) {
        z(i) = N(rng) * ext.parameters(i);
      }
    } else if (ext.strategy == ESStrategy::FullVariance) {
      // parameter update
      CType t_shared = N(rng) * tau.value_or(std::sqrt(1.0 / (2.0 * l)));
      CType t_i = tau_i.value_or(std::sqrt(1.0 / (2.0 * std::sqrt(l))));

      // std dev update
      for (usize i = 0; i < l; i++) {
        ext.parameters(i) = std::max(ext.parameters(i) * std::exp(t_shared + N(rng) * t_i), epsilon);
      }

      // rotation update
      const auto pi = std::numbers::pi_v<CType>;
      for (usize i = l; i < ext.parameters.size(); i++) {
        ext.parameters(i) += N(rng) * beta;

        // keeps the angles meaningful (not really needed since sin/cos are periodic)
        if (ext.parameters(i) > pi) {
          ext.parameters(i) -= 2.0 * pi;
        } else if (ext.parameters(i) < -pi) {
          ext.parameters(i) += 2.0 * pi;
        }
      }

      // sample
      for (usize i = 0; i < l; i++) {
        z(i) = N(rng) * ext.parameters(i);
      }

      // apply rotations
      for (usize p = 1, p0; p <= l - 1; p++) {
        for (usize q = p + 1, q0; q <= l; q++) {
          // formula is 1-indexed in paper, but matrix indices are not
          usize j = ((2 * l - p) * (p + 1)) / 2 + q - 2 * l - 1;

          CType angle_j = ext.parameters(l + j);
          CType s = std::sin(angle_j);
          CType c = std::cos(angle_j);

          CType zp = z(p - 1), zq = z(q - 1);
          z(p - 1) = c * zp - s * zq;
          z(q - 1) = s * zp + c * zq;
        }
      }
    } else if (ext.strategy == ESStrategy::DirectedVariance) {
      // parameter update
      CType t_shared = N(rng) * tau.value_or(std::sqrt(1.0 / (2.0 * l)));
      CType t_i = tau_i.value_or(std::sqrt(1.0 / (2.0 * std::sqrt(l))));

      // std dev update
      for (usize i = 0; i < 2; i++) {
        ext.parameters(i) = std::max(ext.parameters(i) * std::exp(t_shared + N(rng) * t_i), epsilon);
      }

      // rotation update
      const auto pi = std::numbers::pi_v<CType>;
      for (usize i = 2; i < ext.parameters.size(); i++) {
        ext.parameters(i) += N(rng) * beta;

        // keeps the angles meaningful (not really needed since sin/cos are periodic)
        if (ext.parameters(i) > pi) {
          ext.parameters(i) -= 2.0 * pi;
        } else if (ext.parameters(i) < -pi) {
          ext.parameters(i) += 2.0 * pi;
        }
      }

      // sample
      z(0) = N(rng) * ext.parameters(1);
      for (usize i = 1; i < l; i++) {
        z(i) = N(rng) * ext.parameters(0);
      }

      // apply rotations
      for (usize i = 0; i < l - 1; i++) {
        CType angle_j = ext.parameters(2 + i);
        CType s = std::sin(angle_j);
        CType c = std::cos(angle_j);

        CType zp = z(i), zq = z(i + 1);
        z(i) = c * zp - s * zq;
        z(i + 1) = s * zp + c * zq;
      }
    } else {
      throw std::runtime_error("Strategy sampling not implemented!");
    }

    o.continuous_values() += z;

    return o;
  };

 public:
  ES(usize population_size = 15,  // mu
     usize num_parents = 1,       // rho
     usize num_offspring = 100,   // lambda
     // steady state (mu + lambda) vs generational (mu, lambda)
     bool steady_state = true,
     std::string strategy = "single",
     double epsilon = 1e-6,
     std::optional<double> tau = std::nullopt,
     std::optional<double> tau_i = std::nullopt,
     double beta = 0.0873  // ~ 5degrees
     )
      : population_size(population_size),
        num_parents(num_parents),
        num_offspring(num_offspring),
        epsilon(epsilon),
        tau(tau),
        tau_i(tau_i),
        beta(beta),
        steady_state(steady_state) {
    if (population_size < 1) {
      throw std::runtime_error("ES requires a population size >= 1!");
    }
    if (num_parents > population_size) {
      this->num_parents = population_size;
    }
    if (num_offspring < 1) {
      throw std::runtime_error("ES requires at least one offspring per solution!");
    }

    if (strategy == "single") {
      this->strategy = ESStrategy::SingleVariance;
    } else if (strategy == "multiple") {
      this->strategy = ESStrategy::MultipleVariance;
    } else if (strategy == "full") {
      this->strategy = ESStrategy::FullVariance;
    } else if (strategy == "directed") {
      this->strategy = ESStrategy::DirectedVariance;
    } else {
      throw std::runtime_error("Unknown ES strategy: '" + strategy + "'");
    }

    if (epsilon <= 0.0) {
      throw std::runtime_error("Epsilon must be > 0.0!");
    }
  };

  u64 step(Rng& rng, InstanceBase& problem, SolutionSetBase& population, ArchiveBase& archive) const {
    if (problem.num_continuous() < 1) {
      return 0;
    }
    std::uniform_real_distribution<double> U(0.0, 1.0);
    usize n = population.size();
    if (n < 1) {
      throw std::runtime_error("ES requires a population size >= 1!");
    }

    // housekeeping
    offspring.clear();
    solutions_to_evaluate.clear();
    solutions_to_evaluate.reserve(num_offspring);

    // initialize the strategy parameters if necessary
    for (usize i = 0; i < n; i++) {
      init_strategy_params(rng, population[i]);
    }

    // variation
    const auto lb = problem.continuous_lower_bounds();
    const auto ub = problem.continuous_upper_bounds();

    for (usize i = 0; i < num_offspring; i++) {
      if (num_parents > 1) {
        Solution parent = recombine_parents(rng, population);
        offspring.add(mutate_and_sample(rng, parent));
      } else {
        offspring.add(mutate_and_sample(rng, population[i % n]));
      }

      // boundary handling by clamping
      auto x = offspring[i].continuous_values();
      x = x.cwiseMax(lb).cwiseMin(ub);

      solutions_to_evaluate.push_back(i);
    }

    // evaluation & archive update
    problem.evaluate(rng, offspring, solutions_to_evaluate);
    for (usize i : solutions_to_evaluate) {
      archive.update(offspring[i], false);
    }

    // (optionally) O = O + P
    if (steady_state) {
      for (usize i = 0; i < population.size(); i++) {
        offspring.add(population[i]);
      }
    }

    // selection
    auto selection = TruncationSelection().select(rng, problem.fitness(), offspring, n);
    population.clear();
    for (usize i : selection) {
      population.add(offspring[i]);
    }

    return solutions_to_evaluate.size();
  };

  std::tuple<std::shared_ptr<ArchiveBase>, TerminationStatus> run(InstanceBase& problem,
                                                                  const Budget& budget,
                                                                  std::optional<u64> seed,
                                                                  std::optional<usize> population_size) override final {
    usize n = population_size.value_or(this->population_size);

    generation = 0;
    u64 evaluations = n;
    std::chrono::high_resolution_clock::time_point t_start = std::chrono::high_resolution_clock::now();

    Rng rng = seeded_rng(seed);

    // create & evaluate initial population
    AoSSet population;
    problem.add_random(rng, population, n);

    solutions_to_evaluate.resize(n);
    std::iota(solutions_to_evaluate.begin(), solutions_to_evaluate.end(), 0);
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

  std::optional<u64> current_generation() const override final { return generation; };
  std::optional<std::tuple<usize, u64>> current_population() const override final {
    return std::make_tuple(population_size, generation);
  };
};

};  // namespace classic

// TODO GP
//  - either use same GPContext
//  - or make an intermediary -> behaviour is what counts, so linearized postfix as common denominator?

/*

class GPCrossoverBase {
  virtual void crossover virtual ~GPCrossoverBase() = default;
};

class GPMutationBase {
  virtual ~GPMutationBase() = default;
};

class StandardGP : public MethodBase {
 private:
  std::shared_ptr<SelectionStrategyBase> selection_strategy;
  std::shared_ptr<GPCrossoverBase> crossover;
  std::shared_ptr<GPMutationBase> mutation;

  usize population_size{};

  double p_crossover {}
  double p_mutation{};

  u64 generation{};

 public:
  std::tuple<std::shared_ptr<ArchiveBase>, TerminationStatus> run(InstanceBase& problem,
                                                                  const Budget& budget,
                                                                  std::optional<u64> seed,
                                                                  std::optional<usize> population_size) {
     std::uniform_real_distribution<double> U(0.0, 1.0);

    usize n = population_size.value_or(this->population_size);

    Rng rng = seeded_rng(seed);

    auto archive = std::make_shared<UnboundedArchive>(problem.archive_fitness());

    AoSSet population;
    AoSSet offspring;
    AoSSet parents;
    problem.add_random(rng, population, n);

    return std::make_tuple(archive, status);
  };
  std::optional<u64> current_generation() const override final { return generation; };
  std::optional<std::tuple<usize, u64>> current_population() const override final {
    return std::make_tuple(population_size, generation);
  };
};

 */
}  // namespace goblin

#endif /* _GOBLIN_SIMPLE_GA_H */
