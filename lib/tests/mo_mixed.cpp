#include <cstdio>
#include <iostream>
#include <print>

#include "doctest/doctest.h"

#define EIGEN_MAX_ALIGN_BYTES 0
#define EIGEN_DONT_VECTORIZE
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
  const usize NDIMS = 20;
  BenchmarkInstance instance(
      std::vector<std::shared_ptr<ObjectiveBase>>{std::make_shared<OneMax>(NDIMS), std::make_shared<ZeroMax>(NDIMS)});
  instance.add_target_front_size(NDIMS + 1);

  Budget budget(/* max_evaluations = */ 100000, /* max_generations = */ 100);

  std::vector<std::string> reprs{"aos", "soa_cols", "soa_rows"};
  std::vector<std::shared_ptr<LinkageModelBase>> linkage_models{// std::make_shared<UnivariateFOS>(),
                                                                std::make_shared<LinkageTreeFOS>(
                                                                    /* metric = */ "mi"),
                                                                std::make_shared<LinkageTreeFOS>(
                                                                    /* metric = */ "nmi")};
  for (auto repr : reprs) {
    usize i = 0;
    for (auto& lm : linkage_models) {
      auto gomea = MixedGOMEA(PopulationOptions(), RvOptions(), IMSOptions(), lm, std::make_shared<FullFOS>(),
                              std::make_shared<AMaLGaMSamplingModel>(), repr);
      auto [front, status] = Tracked::run(instance, gomea, budget, TrackingOptions(repr + "_mo_mixed.csv"),
                                          /* seed = */ 42);

      std::println("Status {} {}: {}", repr, i++, format_as(status));

      REQUIRE(front->empty() == false);

      for (usize j = 0; j < front->size(); j++) {
        std::println("{}", instance.format_solution((*front)[j]));
      }

      REQUIRE(front->size() == NDIMS + 1);  // checks if the full pareto front was recovered
    }
    // REQUIRE(false);
  }
}
