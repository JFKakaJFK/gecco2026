#pragma once
#ifndef _GOBLIN_SIMPLE_GA_H
#define _GOBLIN_SIMPLE_GA_H

#include "goblin/methods/classic/common.h"

namespace goblin {
namespace classic {
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


class SimpleGA : public EABase {
 private:
  // options
  std::shared_ptr<SelectionStrategyBase> selection_strategy;
  std::shared_ptr<DiscreteCrossoverStrategyBase> crossover_strategy;
  double p_mutation{};
  bool steady_state{};

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
      : EABase(population_size), selection_strategy(selection_strategy),
        crossover_strategy(crossover_strategy),
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

  u64 step(Rng& rng, InstanceBase& problem, SolutionSetBase& population, ArchiveBase& archive) const override final {
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
      const auto& parent1 = population[parent_indices[idx1]];
      const auto& parent2 = population[parent_indices[idx2]];

      // add the offspring to the population (and a copy of the parent to support partial evaluations)
      offspring.add(parent1);
      parents.add(parent1);

      offspring.add(parent2);
      parents.add(parent2);

      // get the crossover masks
      auto& mask1 = subsets[idx1];
      auto& mask2 = subsets[idx2];
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
};

};  // namespace classic
}  // namespace goblin

#endif /* _GOBLIN_SIMPLE_GA_H */
