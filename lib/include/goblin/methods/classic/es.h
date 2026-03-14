#pragma once
#ifndef _GOBLIN_ES_H
#define _GOBLIN_ES_H

#include "goblin/methods/classic/common.h"

namespace goblin {
namespace classic {

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

class ES : public EABase {
 private:
  // options
  usize num_parents{};
  usize num_offspring{};
  double epsilon{};
  std::optional<double> tau{};
  std::optional<double> tau_i{};
  double beta{};
  bool steady_state{};
  ESStrategy strategy{};

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
  ES(usize population_size = 8,  // mu
     usize num_parents = 1,      // rho
     usize num_offspring = 50,   // lambda
     // steady state (mu + lambda) vs generational (mu, lambda)
     bool steady_state = true,
     std::string strategy = "single",
     double epsilon = 1e-6,
     std::optional<double> tau = std::nullopt,
     std::optional<double> tau_i = std::nullopt,
     double beta = 0.0873  // ~ 5degrees
     )
      : EABase(population_size),
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

  u64 step(Rng& rng, InstanceBase& problem, SolutionSetBase& population, ArchiveBase& archive) const override final {
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
};

};  // namespace classic
}  // namespace goblin

#endif /* _GOBLIN_ES_H */
