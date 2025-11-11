#pragma once
#include <cassert>
#include <memory>
#include <random>
#ifndef _GOBLIN_LIB_INIT_H
#define _GOBLIN_LIB_INIT_H

#include <variant>
#include <tuple>
#include <optional>

#include "goblin/lib/types.h"
#include "goblin/lib/rng.h"
#include "goblin/lib/solution.h"
#include "goblin/lib/instance.h"

namespace goblin {

class InitBase {
 public:
  virtual std::tuple<Mat<DType>, Mat<CType>> sample(Rng& rng, const InstanceBase& problem, usize count) const = 0;

  void add_random(Rng& rng, const InstanceBase& problem, SolutionSetBase& solutions, usize count) const {
    auto [dvals, cvals] = sample(rng, problem, count);

    assert(static_cast<usize>(dvals.rows()) == count);
    assert(static_cast<usize>(dvals.cols()) == problem.num_discrete());
    assert(static_cast<usize>(cvals.rows()) == count);
    assert(static_cast<usize>(cvals.cols()) == problem.num_continuous());

    for (usize i = 0; i < count; i++) {
      solutions.add(
          Solution(problem.archive_fitness().worst(),
                   problem.num_discrete() > 0 ? std::make_optional<Vec<DType>>(dvals.row(i)) : std::nullopt,
                   problem.num_continuous() > 0 ? std::make_optional<Vec<CType>>(cvals.row(i)) : std::nullopt));
    }
  };

  virtual ~InitBase() = default;
};

class DiscreteInitBase {
 public:
  virtual Mat<DType> sample(Rng& rng, const InstanceBase& problem, usize count) const = 0;

  virtual ~DiscreteInitBase() = default;
};

class ContinuousInitBase {
 public:
  virtual Mat<CType> sample(Rng& rng, const InstanceBase& problem, usize count) const = 0;

  virtual ~ContinuousInitBase() = default;
};

class RandomDInit final : public DiscreteInitBase {
 public:
  Mat<DType> sample(Rng& rng, const InstanceBase& problem, usize count) const override final {
    Mat<DType> dvals(count, problem.num_discrete());
    for (isize c = 0; c < dvals.cols(); c++) {
      std::uniform_int_distribution<DType> d(0, problem.discrete_domain_sizes()(c) - DType(1));
      for (usize i = 0; i < count; i++) {
        dvals(i, c) = d(rng);
      }
    }
    return dvals;
  };
};

class RandomCInit final : public ContinuousInitBase {
 public:
  Mat<CType> sample(Rng& rng, const InstanceBase& problem, usize count) const override final {
    Mat<CType> cvals(count, problem.num_continuous());
    for (isize c = 0; c < cvals.cols(); c++) {
      std::uniform_real_distribution<CType> d(problem.continuous_init_lower_bounds()(c),
                                              problem.continuous_init_upper_bounds()(c));
      for (usize i = 0; i < count; i++) {
        cvals(i, c) = d(rng);
      }
    }
    return cvals;
  };
};

class RandomInit final : public InitBase {
 public:
  std::tuple<Mat<DType>, Mat<CType>> sample(Rng& rng, const InstanceBase& problem, usize count) const override final {
    return std::make_tuple<Mat<DType>, Mat<CType>>(RandomDInit().sample(rng, problem, count),
                                                   RandomCInit().sample(rng, problem, count));
  };
};

class CombinedInit final : public InitBase {
 public:
  CombinedInit(std::optional<std::shared_ptr<DiscreteInitBase>> discrete = std::nullopt,
               std::optional<std::shared_ptr<ContinuousInitBase>> continuous = std::nullopt)
      : _discrete(discrete.value_or(std::make_shared<RandomDInit>())),
        _continuous(continuous.value_or(std::make_shared<RandomCInit>())) {};

  std::tuple<Mat<DType>, Mat<CType>> sample(Rng& rng, const InstanceBase& problem, usize count) const override final {
    return std::make_tuple<Mat<DType>, Mat<CType>>(_discrete->sample(rng, problem, count),
                                                   _continuous->sample(rng, problem, count));
  };

 private:
  std::shared_ptr<DiscreteInitBase> _discrete;
  std::shared_ptr<ContinuousInitBase> _continuous;
};

using AnyInit = std::variant<std::shared_ptr<InitBase>,
                             std::shared_ptr<DiscreteInitBase>,
                             std::shared_ptr<ContinuousInitBase>,
                             std::tuple<std::shared_ptr<DiscreteInitBase>, std::shared_ptr<ContinuousInitBase>>>;

inline std::shared_ptr<InitBase> from_any_init(AnyInit any_init) {
  if (std::holds_alternative<std::shared_ptr<InitBase>>(any_init)) {
    return std::get<std::shared_ptr<InitBase>>(any_init);
  } else if (std::holds_alternative<std::shared_ptr<DiscreteInitBase>>(any_init)) {
    return std::make_shared<CombinedInit>(std::get<std::shared_ptr<DiscreteInitBase>>(any_init), std::nullopt);
  } else if (std::holds_alternative<std::shared_ptr<ContinuousInitBase>>(any_init)) {
    return std::make_shared<CombinedInit>(std::nullopt, std::get<std::shared_ptr<ContinuousInitBase>>(any_init));
  } else if (std::holds_alternative<std::tuple<std::shared_ptr<DiscreteInitBase>, std::shared_ptr<ContinuousInitBase>>>(
                 any_init)) {
    auto [discrete, continuous] =
        std::get<std::tuple<std::shared_ptr<DiscreteInitBase>, std::shared_ptr<ContinuousInitBase>>>(any_init);
    return std::make_shared<CombinedInit>(discrete, continuous);
  } else {
    std::unreachable();
  }
};

class CompleteInit final : public DiscreteInitBase {
 public:
  Mat<DType> sample(Rng& rng, const InstanceBase& problem, usize count) const override final {
    Mat<DType> dvals(count, problem.num_discrete());
    for (isize c = 0; c < dvals.cols(); c++) {
      std::vector<DType> perm(problem.discrete_domain_sizes()(c));
      std::iota(perm.begin(), perm.end(), 0);

      usize i = perm.size();
      for (usize r = 0; r < count; r++) {
        if (i >= perm.size()) {
          i = 0;
          std::shuffle(perm.begin(), perm.end(), rng);
        }

        dvals(r, c) = perm[i++];
      }
    }
    return dvals;
  };
};

};  // namespace goblin

#endif /* _GOBLIN_LIB_INIT_H */
