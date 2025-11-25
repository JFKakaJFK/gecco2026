#include <cstdio>
#include <iostream>
#include <print>

#include "doctest/doctest.h"
#include <Eigen/Dense>

#include "goblin/bench/functions/combinators.h"
#include "goblin/bench/functions/discrete.h"
#include "goblin/bench/problem.h"
#include "goblin/bench/tracked.h"
#include "goblin/methods/library.h"
#include "goblin/methods/mo_binary.h"

using namespace std::chrono_literals;
using namespace goblin;

TEST_CASE("goblin::methods::mixed") {
  // BenchmarkInstance instance(
  //     std::vector<std::shared_ptr<ObjectiveBase>>{std::make_shared<OneMax>(5)});
  const usize NDIMS = 20;
  BenchmarkInstance instance(
      std::vector<std::shared_ptr<ObjectiveBase>>{std::make_shared<OneMax>(NDIMS), std::make_shared<ZeroMax>(NDIMS)});
  instance.register_target_archive_size(NDIMS + 1);

  Budget budget(/* max_evaluations = */ 100000, /* max_generations = */ 100);

  auto gomea = MOBinaryGOMEA();
  auto [front, status] = Tracked::run(instance, gomea, budget, TrackingOptions("mo_binary.csv"),
                                      /* seed = */ 42);

  std::println("Status: {}", format_as(status));

  REQUIRE(front.empty() == false);

  for (usize j = 0; j < front.size(); j++) {
    std::println("{}", instance.format_solution(front[j]));
  }

  REQUIRE(front.size() == NDIMS + 1);  // checks if the full pareto front was recovered
}
