#pragma once
#ifndef _GOBLIN_SIMPLE_GA_H
#define _GOBLIN_SIMPLE_GA_H

#include <cmath>
#include <print>

#include "goblin/methods/classic/common.h"

namespace goblin {
namespace classic {
/// Strategy used to generate the crossover masks to exchange information between two parents
class DiscreteCrossoverBase {
 public:
  virtual Subset crossover_mask(Rng& rng,
                                InstanceBase& problem,
                                const SolutionBase& donor,
                                SolutionBase& offspring) const {
    throw std::runtime_error("Either implement this method or override the full crossover behaviour!");
  };

  virtual bool crossover(Rng& rng, InstanceBase& problem, const SolutionBase& donor, SolutionBase& offspring) const {
    auto mask = crossover_mask(rng, problem, donor, offspring);

    auto [evaluation_needed, _] = problem.inherit_discrete(offspring, donor, mask);

    return evaluation_needed;
  };

  virtual ~DiscreteCrossoverBase() = default;
};

class CombinedCrossover : public DiscreteCrossoverBase {
  std::vector<std::tuple<std::shared_ptr<DiscreteCrossoverBase>, double>> operators;

 public:
  CombinedCrossover() = delete;
  CombinedCrossover(std::vector<std::tuple<std::shared_ptr<DiscreteCrossoverBase>, double>>&& operators,
                    bool normalize = true)
      : operators(std::move(operators)) {
    if (this->operators.empty()) {
      throw std::runtime_error("At least one operator is required!");
    }

    if (normalize) {
      double norm = 0.0;
      for (auto& [_, probability] : operators) {
        norm += probability;
      }
      for (auto& [_, probability] : operators) {
        probability /= norm;
      }
    }
  };

  bool crossover(Rng& rng,
                 InstanceBase& problem,
                 const SolutionBase& donor,
                 SolutionBase& offspring) const override final {
    bool evaluation_needed = false;
    std::uniform_real_distribution<double> U(0.0, 1.0);
    double r = U(rng);
    for (auto& [op, probability] : operators) {
      if (r < probability) {
        evaluation_needed |= op->crossover(rng, problem, donor, offspring);
        break;
      } else {
        r -= probability;
      }
    }
    return evaluation_needed;
  };
};

class UniformCrossover : public DiscreteCrossoverBase {
  double p_crossover{};

 public:
  UniformCrossover(double p_crossover = 0.5) : p_crossover(p_crossover) {
    if (p_crossover <= 0.0 || 1.0 <= p_crossover) {
      throw std::runtime_error("Crossover probability must be in (0,1) to perform variation, not copying!");
    }
  };

  Subset crossover_mask(Rng& rng,
                        InstanceBase& problem,
                        const SolutionBase& donor,
                        SolutionBase& offspring) const override final {
    Subset mask;
    std::uniform_real_distribution<double> U(0.0, 1.0);
    for (usize i = 0; i < problem.num_discrete(); i++) {
      if (U(rng) < p_crossover) {
        mask.discrete.push_back(i);
      }
    }
    return mask;
  };
};

class NPointCrossover : public DiscreteCrossoverBase {
  usize num_points{};

 public:
  NPointCrossover(usize num_points = 1) : num_points(num_points) {
    if (num_points < 1) {
      throw std::runtime_error("At least one crossover point is needed to perform variation, not copying!");
    }
  };

  Subset crossover_mask(Rng& rng,
                        InstanceBase& problem,
                        const SolutionBase& donor,
                        SolutionBase& offspring) const override final {
    const usize l = problem.num_discrete();
    Subset mask;
    // here the two endpoints are excluded to ensure not all values come from the same parent
    auto points = permute(rng, l - 1);
    if (num_points < l) {
      points.resize(num_points);
    }
    std::sort(points.begin(), points.end());

    // swap the indices between every other set of points
    for (usize i = 0; i < points.size(); i += 2) {
      // + 1 since the first real crossover point is between index 0 and 1, not before index 0
      usize start = points[i] + 1;
      usize end = i + 1 < points.size() ? points[i + 1] + 1 : l;
      for (usize j = start; j < end; j++) {
        mask.discrete.push_back(j);
      }
    }
    return mask;
  };
};

class DiscreteMutationBase {
 public:
  virtual void mutate(Rng& rng, InstanceBase& problem, SolutionBase& offspring) const = 0;
  virtual ~DiscreteMutationBase() = default;
};

class CombinedMutation : public DiscreteMutationBase {
  std::vector<std::tuple<std::shared_ptr<DiscreteMutationBase>, double>> operators;

 public:
  CombinedMutation() = delete;
  CombinedMutation(std::vector<std::tuple<std::shared_ptr<DiscreteMutationBase>, double>>&& operators,
                   bool normalize = true)
      : operators(std::move(operators)) {
    if (this->operators.empty()) {
      throw std::runtime_error("At least one operator is required!");
    }

    if (normalize) {
      double norm = 0.0;
      for (auto& [_, probability] : operators) {
        norm += probability;
      }
      for (auto& [_, probability] : operators) {
        probability /= norm;
      }
    }
  };

  void mutate(Rng& rng, InstanceBase& problem, SolutionBase& offspring) const override final {
    std::uniform_real_distribution<double> U(0.0, 1.0);
    double r = U(rng);
    for (auto& [op, probability] : operators) {
      if (r < probability) {
        op->mutate(rng, problem, offspring);
        break;
      } else {
        r -= probability;
      }
    }
  };
};

class RandomMutation : public DiscreteMutationBase {
  std::optional<double> p_mutation;

 public:
  RandomMutation(std::optional<double> p_mutation = std::nullopt) : p_mutation(p_mutation) {
    if (p_mutation.value_or(0.0) >= 1.0) {
      throw std::runtime_error("A mutation rate of 100% performs random search!");
    }
  };

  void mutate(Rng& rng, InstanceBase& problem, SolutionBase& offspring) const override final {
    std::uniform_real_distribution<double> U(0.0, 1.0);

    const usize l = problem.num_discrete();
    const double p_mut = p_mutation.value_or(1.0 / static_cast<double>(l));

    if (p_mut > 0.0) {
      for (usize i = 0; i < l; i++) {
        const usize d_i = problem.discrete_domain_sizes()(i);
        std::uniform_int_distribution<usize> D(0, d_i - 1);
        if (U(rng) < p_mut && d_i > 1) {
          usize v = D(rng);
          if (offspring.discrete_values()(i) == v) {
            v = (v + 1) % d_i;
          }
          offspring.discrete_values()(i) = v;
        }
      }
    }
  }
};

/// A mutation operator that assumes an ordinal relationship for discrete variables
class LocalizedMutation : public DiscreteMutationBase {
  std::optional<double> p_mutation;
  double strength;
  bool wrap;

 public:
  LocalizedMutation(std::optional<double> p_mutation = std::nullopt, double strength = 0.05, bool wrap = false)
      : p_mutation(p_mutation), strength(strength), wrap(wrap) {
    if (p_mutation.value_or(0.0) >= 1.0) {
      throw std::runtime_error("A mutation rate of 100% performs random search!");
    }

    if (strength <= 0.0) {
      throw std::runtime_error("The mutation strength must be positive!");
    }
  };

  void mutate(Rng& rng, InstanceBase& problem, SolutionBase& offspring) const override final {
    std::uniform_real_distribution<double> U(0.0, 1.0);
    std::normal_distribution<double> N(0.0, 1.0);

    const usize l = problem.num_discrete();
    const double p_mut = p_mutation.value_or(1.0 / static_cast<double>(l));

    if (p_mut > 0.0) {
      for (usize i = 0; i < l; i++) {
        const usize d_i = problem.discrete_domain_sizes()(i);
        if (U(rng) < p_mut && d_i > 1) {
          double v = static_cast<double>(offspring.discrete_values()(i));
          v += N(rng) * strength * static_cast<double>(d_i);
          if (wrap) {
            if (v < 0.0) {  // wrap around by adding d_i * ceil(|v| / d_i)
              v += static_cast<double>(d_i) * std::ceil(-v / static_cast<double>(d_i));
              v = std::fmod(v, d_i);
            }
          } else {
            v = std::clamp(v, 0.0, static_cast<double>(d_i - 1));
          }

          offspring.discrete_values()(i) = static_cast<DType>(v);
        }
      }
    }
  }
};

class SimpleGA : public EABase {
 private:
  // options
  std::shared_ptr<SelectionStrategyBase> selection_strategy;
  std::shared_ptr<DiscreteCrossoverBase> crossover_strategy;
  std::shared_ptr<DiscreteMutationBase> mutation_strategy;
  bool steady_state{};

  // temporary buffers
  mutable AoSSet offspring;
  mutable std::vector<usize> solutions_to_evaluate;
  mutable std::vector<Subset> subsets;
  mutable std::vector<const Subset*> subset_refs;

 public:
  SimpleGA(usize population_size = 100,
           std::shared_ptr<DiscreteCrossoverBase> crossover = std::make_shared<UniformCrossover>(),
           std::shared_ptr<DiscreteMutationBase> mutation = std::make_shared<RandomMutation>(),
           bool steady_state =
               true,  // steady_state vs generational: select from P + O or just from O after generating more offspring?
           std::shared_ptr<SelectionStrategyBase> selection = std::make_shared<TournamentSelection>(4))
      : EABase(population_size),
        selection_strategy(selection),
        crossover_strategy(crossover),
        mutation_strategy(mutation),
        steady_state(steady_state) {
    if (!selection) {
      throw std::runtime_error("No selection provided!");
    }
    if (!crossover) {
      throw std::runtime_error("No crossover provided!");
    }
    if (!mutation) {
      throw std::runtime_error("No mutation provided!");
    }
    if (population_size % 2 != 0) {
      throw std::runtime_error("Population size must be even!");
    }
    if (auto p = dynamic_cast<TruncationSelection*>(&*selection_strategy); p != nullptr && !steady_state) {
      // generational: need to select population_size parents -> no selection pressure with truncation selection
      throw std::runtime_error("Truncation selection is not compatible with a generational replacement scheme!");
    }
  };

  void check_changes(const SolutionBase& parent,
                     const SolutionBase& offspring,
                     std::vector<usize>& changed_indices,
                     bool& evaluation_needed) const {
    changed_indices.clear();
    for (usize i = 0; i < parent.num_discrete(); i++) {
      // something changed if the values are different, but we only need to evaluate if at least one variable active in
      // the parent changed or the fitness should still be the same
      if (parent.discrete_values()(i) != offspring.discrete_values()(i)) {
        changed_indices.push_back(i);
        if (parent.discrete_active()(i)) {
          evaluation_needed = true;
        }
      }
    }
  };

  u64 step(Rng& rng, InstanceBase& problem, SolutionSetBase& population, ArchiveBase& archive) const override final {
    if (problem.num_discrete() < 1) {
      return 0;
    }
    std::uniform_real_distribution<double> U(0.0, 1.0);
    const usize n = population.size();
    if (n % 2 != 0) {
      throw std::runtime_error("Population size must be even!");
    }

    // The problem interface supports partial evaluations and provides information
    // on which decision variables are active (or not) in case the problem can have
    // conditionally inactive variables.
    // To take full advantage of such problem settings, partial evaluations are performed
    // and only on offspring solutions where the active variables changed.
    // To support this, the subset of changed variables between offspring and parent need
    // to be passed to the evaluation call.

    // variation
    subsets.resize(n);
    subset_refs.resize(n);
    offspring.clear();
    solutions_to_evaluate.clear();
    solutions_to_evaluate.reserve(n);
    std::vector<usize> parent_indices = permute(rng, n);
    for (usize i = 0; i < n; i++) {
      // copy to offspring
      offspring.add(population[i]);
      bool evaluation_needed = false;

      const auto& donor = population[parent_indices[i]];

      // perform crossover
      if (crossover_strategy) {
        evaluation_needed |= crossover_strategy->crossover(rng, problem, donor, offspring[i]);
      }

      // apply mutation
      if (mutation_strategy) {
        mutation_strategy->mutate(rng, problem, offspring[i]);
      }

      // check what changed to allow for partial evaluations & exploiting introns
      check_changes(population[i], offspring[i], subsets[i].discrete, evaluation_needed);
      if (evaluation_needed) {
        solutions_to_evaluate.push_back(i);
        subset_refs[i] = &subsets[i];
      }
    }

    // evaluation & archive update
    problem.evaluate_partial(rng, offspring, population, subset_refs, solutions_to_evaluate);
    for (usize i : solutions_to_evaluate) {
      archive.update(offspring[i], false);
    }

    // add offspring to selection pool (if steady-state)
    if (steady_state) {
      for (usize i = 0; i < n; i++) {
        offspring.add(population[i]);
      }
    }

    // selection
    auto selection = selection_strategy->select(rng, problem.fitness(), offspring, n);
    population.clear();
    for (auto i : selection) {
      population.add(offspring[i]);
    }

    return solutions_to_evaluate.size();
  };
};

};  // namespace classic
}  // namespace goblin

#endif /* _GOBLIN_SIMPLE_GA_H */
