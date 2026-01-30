#include <iostream>
#include <Eigen/Dense>
#include <memory>
#include <doctest/doctest.h>
#include <cassert>

#include "goblin/lib/fitness.h"
#include "goblin/methods/mixed.h"
#include "goblin/lib/init.h"

using namespace goblin;

#define D 5
#define NUM_SAMPLES 10
#define NOISE 0.1

// One of the scenarios where purely relative solution assessments are possible
// are noisy settings, where each quality corresponds to samples from a distribution
// - and then comparing the distributions directly in preference decisions might be
// the way to go

struct NoisyResult : public QualityBase {
  Vec<CType> samples;

  std::unique_ptr<QualityBase> clone() const override final { return std::make_unique<NoisyResult>(*this); }
};

class NoisyFitness : public ArchiveFitnessBase {
 public:
  usize num_objectives() const override final { return 1; };

  Ordering cmp(const QualityBase& lhs, const QualityBase& rhs, std::optional<usize> objective) const override final {
    const auto &ql = static_cast<const NoisyResult&>(lhs), qr = static_cast<const NoisyResult&>(lhs);
    const usize N = 1000;

    Rng rng = seeded_rng();
    std::uniform_int_distribution<u64> sample(0, NUM_SAMPLES - 1);

    usize lhs_better_count = 0, tie_count = 0;
    for (usize i = 0; i < N; i++) {
      CType lhs_mean = 0.0, rhs_mean = 0.0;
      for (isize j = 0; j < ql.samples.size(); j++) {
        lhs_mean += ql.samples(sample(rng));
      }
      for (isize j = 0; j < qr.samples.size(); j++) {
        rhs_mean += qr.samples(sample(rng));
      }
      lhs_mean /= ql.samples.size();
      rhs_mean /= qr.samples.size();
      CType delta = std::abs(lhs_mean - rhs_mean);
      CType magnitude = 0.01 * std::max(std::abs(lhs_mean), std::abs(rhs_mean));
      if (delta < magnitude) {
        tie_count++;
      } else if (lhs_mean < rhs_mean) {
        lhs_better_count++;
      }
    }

    const usize threshold = 0.6 * N;
    if (lhs_better_count > threshold) {
      return Ordering::Better;
    } else if (N - lhs_better_count - tie_count > threshold) {
      return Ordering::Worse;
    } else {
      return Ordering::Equal;
    }
  };

  CType distance(const QualityBase& lhs, const QualityBase& rhs, std::optional<usize> objective) const override final {
    const auto &ql = static_cast<const NoisyResult&>(lhs), qr = static_cast<const NoisyResult&>(lhs);

    return std::abs((ql.samples.sum() / static_cast<CType>(ql.samples.size())) -
                    (qr.samples.sum() / static_cast<CType>(qr.samples.size())));
  };

  void log_header(std::ostream& os) const override final { os << "sample_mean,"; }

  void log(std::ostream& os, const QualityBase& quality) const override final {
    const auto& q = static_cast<const NoisyResult&>(quality);
    os << q.samples.sum() / q.samples.size();
  };

  std::unique_ptr<QualityBase> worst() const override final {
    const CType inf = std::numeric_limits<CType>().infinity();
    auto q = std::make_unique<NoisyResult>();
    q->samples = Vec<CType>::Constant(NUM_SAMPLES, inf);
    return q;
  };
};

class NoisySphere : public InstanceBase {
 public:
  usize num_discrete() const override final { return 0; }
  CRef<Vec<DType>> discrete_domain_sizes() const override final {
    static Vec<DType> d;
    return d;
  }

  usize num_continuous() const override final { return D; }
  CRef<Vec<CType>> continuous_lower_bounds() const override final {
    static Vec<CType> lb = Vec<CType>::Constant(D, std::numeric_limits<CType>().infinity());
    return lb;
  }
  CRef<Vec<CType>> continuous_upper_bounds() const override final {
    static Vec<CType> ub = Vec<CType>::Constant(D, -std::numeric_limits<CType>().infinity());
    return ub;
  }

  CRef<Vec<CType>> continuous_init_lower_bounds() const override final {
    static Vec<CType> ilb = Vec<CType>::Constant(D, 100.0);
    return ilb;
  }
  CRef<Vec<CType>> continuous_init_upper_bounds() const override final {
    static Vec<CType> iub = Vec<CType>::Constant(D, 110.0);
    return iub;
  }

  void evaluate(Rng& rng, SolutionSetBase& solutions, const std::span<const usize>& indices) override final {
    Vec<CType> noise(D);
    std::uniform_real_distribution<CType> d(0.0, NOISE);
    for (auto idx : indices) {
      for (usize i = 0; i < NUM_SAMPLES; i++) {
        for (usize j = 0; j < D; j++) {
          noise(j) = d(rng);
        }

        auto& q = static_cast<NoisyResult&>(solutions[idx].quality());
        q.samples(i) = (solutions[idx].continuous_values() + noise).array().square().sum();
      }
    }
  }

  void add_random(Rng& rng, SolutionSetBase& solutions, usize count) const override final {
    RandomInit().add_random(rng, *this, solutions, count);
  }

  const FitnessBase& fitness() const override final { return fitness_; }

  const ArchiveFitnessBase& archive_fitness() const override final { return fitness_; }

  bool target_reached(const ArchiveBase& archive) const override {
    NoisyResult target;
    target.samples = Vec<CType>::Constant(1, NOISE);

    return archive.size() > 0 && fitness().cmp(archive[0].quality(), target, std::nullopt) != Ordering::Worse;
  };

 private:
  NoisyFitness fitness_{};
};

TEST_CASE("goblin::lib::custom_quality_and_fitness") {
  NoisySphere sphere{};

  Budget budget(/* max_evaluations = */ 10000);

  auto gomea = MixedGOMEA(PopulationOptions(), RvOptions{.max_nis = 100},
                          IMSOptions{
                              .initial_population_size = 10, .max_num_populations = 1
                              // .initial_population_size = 10,
                              // .subgeneration_factor = 8,
                          },
                          std::make_shared<LinkageTreeFOS>(), std::make_shared<FullFOS>());
  auto [front, _] = gomea.run(sphere, budget);

  REQUIRE(sphere.target_reached(*front));
}
