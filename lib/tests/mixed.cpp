#include <iostream>
#include <cassert>
#include <print>

#include "doctest/doctest.h"
#include <Eigen/Dense>

#include "goblin/bench/functions/combinators.h"
#include "goblin/bench/functions/discrete.h"
#include "goblin/bench/problem.h"
#include "goblin/bench/tracked.h"
#include "goblin/methods/library.h"
#include "goblin/methods/mixed.h"

using namespace std::chrono_literals;
using namespace goblin;

TEST_CASE("goblin::methods::mixed") {
  // BenchmarkInstance instance(
  //     std::vector<std::shared_ptr<ObjectiveBase>>{std::make_shared<OneMax>(5)});
  const usize REPEATS = 4;
  BenchmarkInstance instance(std::make_shared<Repeat>(std::make_shared<DeceptiveTrap>(5), REPEATS));
  double vtr = 5.0 * REPEATS;
  instance.register_target({vtr});

  Budget budget(/* max_evaluations = */ 100000, /* max_generations = */ 100);

  auto dgomea = DiscreteGOMEA();
  Tracked::run(instance, dgomea, budget, TrackingOptions("discrete.csv"));

  std::vector<IMSOptions> ims_options{
      IMSOptions(),
      // IMSOptions(/* initial_population_size */ 10, /* max_num_populations */ 1) // too flaky across compilers - this
      // is a test of some method options, not a performance baseline comparison...
  };
  std::vector<std::string> reprs{"aos", "soa_cols", "soa_rows"};
  std::vector<std::shared_ptr<LinkageModelBase>> linkage_models{// std::make_shared<UnivariateFOS>(),
                                                                std::make_shared<LinkageTreeFOS>(
                                                                    /* metric = */ "mi"),
                                                                std::make_shared<LinkageTreeFOS>(
                                                                    /* metric = */ "nmi"),
                                                                std::make_shared<LinkageTreeFOS>(
                                                                    /* metric = */ "mi",
                                                                    /* intron_strategy */ "any_active"),
                                                                std::make_shared<LinkageTreeFOS>(
                                                                    /* metric = */ "mi",
                                                                    /* intron_strategy */ "all_active"),
                                                                std::make_shared<LinkageTreeFOS>(
                                                                    /* metric = */ "mi",
                                                                    /* intron_strategy */ "mark_only")};
  for (auto repr : reprs) {
    usize i = 0;
    for (auto& lm : linkage_models) {
      for (auto& iopt : ims_options) {
        auto gomea = MixedGOMEA(PopulationOptions(), RvOptions(), iopt, lm, std::make_shared<FullFOS>(),
                                std::make_shared<AMaLGaMSamplingModel>(), repr);
        auto [front, status] =
            Tracked::run(instance, gomea, budget, TrackingOptions(repr + "_mixed.csv"), /* seed = */ 42);

        std::println("Status {} {}: {}", repr, i++, format_as(status));

        REQUIRE(front.empty() == false);
        REQUIRE_MESSAGE(front.so_solution(0).quality().objectives[0] == vtr, repr);
      }
    }
    // REQUIRE(false);
  }
}
