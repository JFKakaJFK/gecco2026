#include <cstdio>
#include <iostream>
#include <print>

#include "doctest/doctest.h"
#include <Eigen/Dense>

#include "goblin/bench/functions/combinators.h"
#include "goblin/bench/functions/continuous.h"
#include "goblin/bench/problem.h"
#include "goblin/bench/tracked.h"
#include "goblin/methods/mo_rv_gomea.h"

using namespace std::chrono_literals;
using namespace goblin;

TEST_CASE("goblin::methods::mo_rv_gomea") {
  const usize FRONT_SIZE = 100;
  const CType TOL = 1e-5;  // Usually 1e-7, so this should succeed almost always

  auto sphere = std::make_shared<Sphere>(2);
  Vec<CType> t0(2);
  t0 << -1.0, -1.0;
  Vec<CType> t1(2);
  t1 << 1.0, 1.0;
  BenchmarkInstance instance(std::vector<std::shared_ptr<ObjectiveBase>>{
      std::make_shared<goblin::Translated>(sphere, t0), std::make_shared<goblin::Translated>(sphere, t1)});

  Budget budget(/* max_evaluations = */ 10'000);

  auto gomea = MoRvGOMEA();
  auto [front, status] = Tracked::run(instance, gomea, budget, TrackingOptions("mo_rv_gomea.csv"));

  std::println("Status: {}", format_as(status));

  REQUIRE(front->empty() == false);

  // Idea: use the D_{PF<-S} measure to ensure that for each point on the true PF (FRONT_SIZE samples) there is a member
  // in the approximation front within a distance of TOL
  Mat<CType> pareto_front(FRONT_SIZE, 2);
  {
    Vec<CType> s = (t1 - t0).array() / static_cast<double>(FRONT_SIZE - 1);
    for (usize i = 0; i < FRONT_SIZE; i++) {
      pareto_front.row(i) = t0 + static_cast<double>(i) * s;
    }
  }
  Vec<CType> min_dist_s2pf(FRONT_SIZE);

  for (usize j = 0; j < front->size(); j++) {
    const auto& s = (*front)[j];
    std::println("{} @ {}", instance.format_solution(s), instance.archive_fitness().format(s.quality()));

    for (usize i = 0; i < FRONT_SIZE; i++) {
      CType d = (s.continuous_values() - pareto_front.row(i).transpose()).squaredNorm();
      if (j == 0 || d < min_dist_s2pf(i)) {
        min_dist_s2pf(i) = d;
      }
    }
  }

  CType max_min_dist = min_dist_s2pf.minCoeff();
  std::println("Max distance to PF member: {}", max_min_dist);
  REQUIRE(max_min_dist <= TOL);

  // REQUIRE(status == TerminationStatus::TargetReached);
}
