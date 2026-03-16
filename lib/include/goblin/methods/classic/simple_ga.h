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
  virtual std::tuple<Subset, Subset> crossover_masks(Rng& rng,
                                                     const SolutionBase& parent1,
                                                     const SolutionBase& parent2) const = 0;
  virtual ~DiscreteCrossoverBase() = default;
};

class UniformCrossover : public DiscreteCrossoverBase {
  double p_crossover{};

 public:
  UniformCrossover(double p_crossover = 0.5) : p_crossover(p_crossover) {
    if (p_crossover <= 0.0 || 1.0 <= p_crossover) {
      throw std::runtime_error("Crossover probability must be in (0,1) to perform variation, not copying!");
    }
  };

  std::tuple<Subset, Subset> crossover_masks(Rng& rng,
                                             const SolutionBase& parent1,
                                             const SolutionBase& parent2) const override final {
    Subset mask1, mask2;
    std::uniform_real_distribution<double> U(0.0, 1.0);
    for (usize i = 0; i < parent1.num_discrete(); i++) {
      if (U(rng) < p_crossover) {
        mask1.discrete.push_back(i);
        mask2.discrete.push_back(i);
      }
    }
    return std::make_tuple(mask1, mask2);
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

  std::tuple<Subset, Subset> crossover_masks(Rng& rng,
                                             const SolutionBase& parent1,
                                             const SolutionBase& parent2) const override final {
    Subset mask1, mask2;
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
    return std::make_tuple(mask1, mask2);
  };
};

class DiscreteMutationBase {
 public:
  virtual void mutate(Rng& rng, InstanceBase& problem, SolutionBase& offspring) const = 0;
  virtual ~DiscreteMutationBase() = default;
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
          if(wrap){
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

class MergeSplitMutation : public classic::DiscreteMutationBase {
  double p_merge;
  double noise;
  usize min_num_cells;

 public:
  MergeSplitMutation(usize min_num_cells, std::optional<double> p_mutation = std::nullopt, double p_merge = 0.5, double splitting_noise = 0.05) : p_merge(p_merge), noise(splitting_noise), min_num_cells(min_num_cells) {
    if (p_merge < 0.0 || 1.0 < p_merge) {
      throw std::runtime_error("The probability of merging must be in [0, 1]!");
    }

    if (splitting_noise <= 0.0) {
      throw std::runtime_error("The splitting noise must be positive!");
    }
  };

  void mutate(Rng& rng, InstanceBase& problem, SolutionBase& offspring) const override final {
      const usize VARS_PER_CELL = 6;
    const usize num_cells = problem.num_discrete() / VARS_PER_CELL;

    usize cell = std::uniform_int_distribution<usize>(0, num_cells - 1)(rng);
    usize random_cell = std::uniform_int_distribution<usize>(min_num_cells, num_cells - 1)(rng);

    std::uniform_real_distribution<double> U(0.0, 1.0);
    std::normal_distribution<double> N(0.0, 1.0);

    bool merge = U(rng) < p_merge;

    // loop until we find another active/inactive cell for merging/splitting into
    usize start = random_cell;
    while(merge != offspring.discrete_active()(random_cell * VARS_PER_CELL)){
        random_cell++;
        if(random_cell >= num_cells){
            random_cell = min_num_cells;
        }
        if(random_cell == start){
            break;
        }
    }
    usize offset = cell * VARS_PER_CELL;
    usize offset_random = random_cell * VARS_PER_CELL;

    auto x = offspring.discrete_values();
    if(merge){
        // disable the other cell
        x(offset_random) = false;
        // and linearly combine the cell information with random weights
        double w = U(rng);
        for(usize i = 1; i < VARS_PER_CELL; i++){
            x(offset + i) = w * static_cast<double>(x(offset + i)) + (1.0 - w) * static_cast<double>(x(offset_random + i));
        }
    } else /* split */ {
        // enable the other cell
        x(offset_random) = true;

        // add noise to both cell values
        double v;
        for(usize i = 1; i < VARS_PER_CELL; i++){
            const usize d_i = problem.discrete_domain_sizes()(offset + i);

            // get value, add noise & map back into the domain for the other cell
            v = static_cast<double>(x(offset + i));
            v += N(rng) * noise * static_cast<double>(d_i);
            if (v < 0.0) {  // wrap around by adding d_i * ceil(|v| / d_i)
              v += static_cast<double>(d_i) * std::ceil(-v / static_cast<double>(d_i));
            }
            v = std::fmod(v, d_i);
            x(offset_random + i) = static_cast<DType>(v);

            // get value, add noise & map back into the domain for this cell
            v = static_cast<double>(x(offset + i));
            v += N(rng) * noise * static_cast<double>(d_i);
            if (v < 0.0) {  // wrap around by adding d_i * ceil(|v| / d_i)
              v += static_cast<double>(d_i) * std::ceil(-v / static_cast<double>(d_i));
            }
            v = std::fmod(v, d_i);
            x(offset + i) = static_cast<DType>(v);
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
  mutable AoSSet parents;
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

  bool create_offspring(Rng& rng,
                        InstanceBase& problem,
                        const SolutionBase& parent,
                        const SolutionBase& donor,
                        const Subset& crossover_mask,
                        SolutionBase& offspring,
                        std::vector<usize>& changed_indices) const {
    // perform crossover with mask
    problem.inherit_discrete(offspring, donor, crossover_mask);

    // apply mutation
    mutation_strategy->mutate(rng, problem, offspring);

    // check what changed to support exploiting partial evaluations & knowledge about inactive variables
    bool needs_evaluation = false;
    changed_indices.clear();
    for (usize i = 0; i < problem.num_discrete(); i++) {
      if (parent.discrete_values()(i) != offspring.discrete_values()(i)) {
        changed_indices.push_back(i);
        if (parent.discrete_active()(i)) {
          needs_evaluation = true;
        }
      }
    }

    return needs_evaluation;
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
    // The crossover masks/subset of changed variables between offspring and parent need
    // to be passed to the evaluation call to support this.

    // variation
    subsets.resize(n);
    subset_refs.resize(n);
    offspring.clear();
    parents.clear();
    solutions_to_evaluate.clear();
    solutions_to_evaluate.reserve(n);
    std::vector<usize> parent_indices = permute(rng, n);
    for (usize i = 0; i < n; i += 2) {
      // get and copy parents
      const auto& parent1 = population[parent_indices[i]];
      const auto& parent2 = population[parent_indices[i + 1]];

      auto [mask1, mask2] = crossover_strategy->crossover_masks(rng, parent1, parent2);

      // add the offspring to the population (and a copy of the parent to support partial evaluations)
      offspring.add(parent1);
      parents.add(parent1);
      offspring.add(parent2);
      parents.add(parent2);

      if (create_offspring(rng, problem, parents[i], parent2, mask1, offspring[i], subsets[i].discrete)) {
        solutions_to_evaluate.push_back(i);
      }

      if (create_offspring(rng, problem, parents[i + 1], parent1, mask2, offspring[i + 1], subsets[i + 1].discrete)) {
        solutions_to_evaluate.push_back(i + 1);
      }
    }

    // evaluation & archive update
    for (usize i : solutions_to_evaluate) {
      subset_refs[i] = &subsets[i];
    }
    problem.evaluate_partial(rng, offspring, parents, subset_refs, solutions_to_evaluate);
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
