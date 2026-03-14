#pragma once
#ifndef _GOBLIN_CLASSIC_DE_H
#define _GOBLIN_CLASSIC_DE_H

#include "goblin/methods/classic/common.h"


namespace goblin {
namespace classic {

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
          trial(j) = trial(j) + F_actual * ((x_r0(j) - trial(j)) + (x_r1(j) - x_r2(j)));
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

class DE : public EABase {
 private:
  // options
  std::shared_ptr<DEStrategyBase> strategy;

  // temporary buffers
  mutable AoSSet offspring;
  mutable std::vector<usize> solutions_to_evaluate;
  mutable std::vector<Subset> subsets;
  mutable std::vector<const Subset*> subset_refs;

 public:
  DE(usize population_size = 100,
                        std::shared_ptr<DEStrategyBase> strategy = std::make_shared<Rand1Bin>())
      : EABase(population_size), strategy(strategy) {
    if (population_size < 4) {
      throw std::runtime_error("DE requires a population size >= 4!");
    }
  };

  u64 step(Rng& rng, InstanceBase& problem, SolutionSetBase& population, ArchiveBase& archive) const override final {
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
};

};
};

#endif /* _GOBLIN_CLASSIC_DE_H */
