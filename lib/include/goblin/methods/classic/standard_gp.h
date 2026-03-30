#pragma once
#ifndef _GOBLIN_STANDARD_GP_H
#define _GOBLIN_STANDARD_GP_H

#include "goblin/methods/classic/common.h"

namespace goblin {
namespace classic {

class GPVariationOperatorBase {
 public:
  virtual void apply(Rng& rng,
                     const InstanceBase& problem,
                     const GPContext& ctx,
                     const SolutionSetBase& population,
                     SolutionBase& offspring) const = 0;
  virtual ~GPVariationOperatorBase() = default;
};

class Chained : public GPVariationOperatorBase {
  std::vector<std::tuple<std::shared_ptr<GPVariationOperatorBase>, double>> operators;

 public:
  Chained() = delete;
  Chained(std::vector<std::tuple<std::shared_ptr<GPVariationOperatorBase>, double>>&& operators)
      : operators(std::move(operators)) {
    if (this->operators.empty()) {
      throw std::runtime_error("At least one operator is required!");
    }
  };

  void apply(Rng& rng,
             const InstanceBase& problem,
             const GPContext& ctx,
             const SolutionSetBase& population,
             SolutionBase& offspring) const override final {
    std::uniform_real_distribution<double> U(0.0, 1.0);
    for (auto& [op, probability] : operators) {
      if (U(rng) < probability) {
        op->apply(rng, problem, ctx, population, offspring);
      }
    }
  };
};

class SubtreeCrossover : public GPVariationOperatorBase {
 public:
  void apply(Rng& rng,
             const InstanceBase& problem,
             const GPContext& ctx,
             const SolutionSetBase& population,
             SolutionBase& offspring) const override final {
    // get donor
    const auto& donor = population[std::uniform_int_distribution<usize>(0, population.size() - 1)(rng)];
    auto donor_nodes = ctx.active_nodes(donor);

    // get random node
    auto nodes = ctx.active_nodes(offspring);
    usize node_idx = nodes[std::uniform_int_distribution<usize>(0, nodes.size() - 1)(rng)];

    // replace with random donor subtree
    std::shuffle(donor_nodes.begin(), donor_nodes.end(), rng);
    for (usize i : donor_nodes) {
      // this can fail due to tree shape constraints (max_depth/size)
      if (ctx.copy_tree(donor, i, offspring, node_idx)) {
        break;
      }
    }
  }
};

/// Replaces random subtree with a random subtree
class SubtreeMutation : public GPVariationOperatorBase {
 public:
  void apply(Rng& rng,
             const InstanceBase& problem,
             const GPContext& ctx,
             const SolutionSetBase& population,
             SolutionBase& offspring) const override final {
    // get random node index
    auto nodes = ctx.active_nodes(offspring);
    std::shuffle(nodes.begin(), nodes.end(), rng);

    // get random tree
    AoSSet s;
    problem.add_random(rng, s, 1);

    ctx.copy_tree(s[0], nodes[0], offspring, nodes[0]);
  };
};

/// Randomly mutates constants
class ConstantMutation : public GPVariationOperatorBase {
  double probability{};
  CType temperature{};

 public:
  ConstantMutation(double probability = 0.25, CType temperature = 0.25)
      : probability(probability), temperature(temperature) {};

  void apply(Rng& rng,
             const InstanceBase& problem,
             const GPContext& ctx,
             const SolutionSetBase& population,
             SolutionBase& offspring) const override final {
    std::uniform_real_distribution<double> U(0.0, 1.0);
    std::normal_distribution<CType> N(0.0, 1.0);

    // get random node index
    auto indices = ctx.active_constant_indices(offspring);

    for (usize ci : indices) {
      if (U(rng) < probability) {
        offspring.continuous_values()(ci) += N(rng) * offspring.continuous_values()(ci) * temperature;
      }
    }
  };
};

class StandardGP : public EABase {
 private:
  // options
  std::shared_ptr<SelectionStrategyBase> selection_strategy;
  std::shared_ptr<GPVariationOperatorBase> variation_operator;
  bool steady_state{};

  // temporary buffers
  mutable AoSSet offspring;
  mutable std::vector<usize> solutions_to_evaluate;

 public:
  StandardGP(usize population_size = 32,
             bool steady_state = true,  // steady_state vs generational: select from P + O or just from O after
                                        // generating more offspring?
             std::shared_ptr<GPVariationOperatorBase> variation_operator = std::shared_ptr<GPVariationOperatorBase>(),
             std::shared_ptr<SelectionStrategyBase> selection_strategy = std::make_shared<TournamentSelection>(2))
      : EABase(population_size),
        selection_strategy(selection_strategy),
        variation_operator(
            variation_operator != nullptr
                ? variation_operator
                : std::make_shared<Chained>(std::vector<std::tuple<std::shared_ptr<GPVariationOperatorBase>, double>>{
                      std::make_tuple(std::make_shared<SubtreeCrossover>(), 1.0),
                      std::make_tuple(std::make_shared<SubtreeMutation>(), 0.25),
                      std::make_tuple(std::make_shared<ConstantMutation>(), 0.25)})),
        steady_state(steady_state) {
    if (auto p = dynamic_cast<TruncationSelection*>(&*selection_strategy); p != nullptr && !steady_state) {
      // generational: need to select population_size solutions from population_size offspring -> no selection pressure
      // with truncation selection
      throw std::runtime_error("Truncation selection is not compatible with a generational replacement scheme!");
    }
  };

  u64 step(Rng& rng, InstanceBase& problem, SolutionSetBase& population, ArchiveBase& archive) const override final {
    if (auto p = dynamic_cast<const GPInstanceBase*>(&problem.unwrap()); p != nullptr) {
      const auto& ctx = p->context();
      const usize n = population.size();

      // variation
      offspring.clear();
      solutions_to_evaluate.clear();
      solutions_to_evaluate.reserve(n);
      for (usize i = 0; i < n; i++) {
        offspring.add(population[i]);
        variation_operator->apply(rng, problem, ctx, population, offspring[i]);
        solutions_to_evaluate.push_back(i);
      }

      // evaluation & archive update
      problem.evaluate(rng, offspring, solutions_to_evaluate);
      for (usize i : solutions_to_evaluate) {
        archive.update(offspring[i], false);
      }

      if (steady_state) {
        // steady state selection from P + O
        for (usize i = 0; i < n; i++) {
          offspring.add(population[i]);
        }
      }

      auto selection = selection_strategy->select(rng, problem.fitness(), offspring, n);
      population.clear();
      for (auto i : selection) {
        population.add(offspring[i]);
      }

      return solutions_to_evaluate.size();
    } else {
      throw std::runtime_error("Not a GP problem!");
    }
  };
};

};  // namespace classic
}  // namespace goblin

#endif /* _GOBLIN_STANDARD_GP_H */
