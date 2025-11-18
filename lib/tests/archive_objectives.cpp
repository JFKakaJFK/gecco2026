#include <print>

#include "doctest/doctest.h"

#define EIGEN_MAX_ALIGN_BYTES 0
#define EIGEN_DONT_VECTORIZE
#include <Eigen/Dense>

#include "goblin/bench/tracked.h"
#include "goblin/gp/sr.h"
#include "goblin/methods/mixed.h"

using namespace std::chrono_literals;
using namespace goblin;

TEST_CASE("goblin::gp::archive_objectives") {
  auto tree = TemplateNode::full_nary(2, 3);
  Template tmplate;
  tmplate.add_output(tree);
  tmplate.add_output(tree);

  REQUIRE(tree.size() == 15);
  REQUIRE(tmplate.size() == 30);

  Arr2D<CType> X = Arr2D<CType>::Random(10, 3);
  Arr2D<CType> Y(X.rows(), 2);
  Y.col(0) = X.col(0) + X.col(1);
  Y.col(1) = X.col(1) * X.col(2);

  Arr2D<CType> X_test = Arr2D<CType>::Random(5, 3);
  Arr2D<CType> Y_test(X_test.rows(), 2);
  Y_test.col(0) = X_test.col(0) + X_test.col(1);
  Y_test.col(1) = X_test.col(1) * X_test.col(2);

  std::vector<std::shared_ptr<OperatorBase>> operators = {std::make_shared<OpAdd>(), std::make_shared<OpSub>(),
                                                          std::make_shared<OpMul>(), std::make_shared<OpDiv>(),
                                                          std::make_shared<OpSin>()};

  GPContext ctx(
      /* num_inputs = */ X.cols(), tmplate, operators);

  std::vector<CType> vtr = {1e-8, 6.0};  // best size for (x0 + x1), (x1 * x2) is 6

  std::vector<std::string> objectives{"nmse", "size"};
  for (usize objectives_to_optimize = 1; objectives_to_optimize <= objectives.size(); objectives_to_optimize++) {
    SRProblem srp(ctx, X, Y, X_test, Y_test, objectives, /* objectives_to_optimize = */ objectives_to_optimize,
                  /* linear_scaling */ true);
    srp.register_target(vtr);

    Budget budget(/* max_evaluations = */ 100000, /* max_generations = */ 100);
    auto gomea = MixedGOMEA(PopulationOptions(), RvOptions{.enabled=false},
                            IMSOptions(
                                /* initial_population_size = */ 256,
                                /* max_num_populations = */ 1),
                            std::make_shared<LinkageTreeFOS>(
                                /* metric = */ "mi",
                                /* intron_strategy */ "any_active"));

    auto [front, status] = Tracked::run(srp, gomea, budget, TrackingOptions("sr.csv"), /* seed = */ 42);

    std::println("Status {}: {}", format_as(status), srp.format_solution(front.so_solution(0)));

    REQUIRE(front.empty() == false);

    // ls values are re-computed, so there can be slight differences here, hence 10x
    REQUIRE_MESSAGE(front.so_solution(0).quality().objectives(0) <= vtr[0] * 10.0,
                    srp.format_solution(front.so_solution(0)));
    if (objectives_to_optimize > 1) {
      REQUIRE_MESSAGE(front.so_solution(1).quality().objectives(1) <= vtr[1],
                      srp.format_solution(front.so_solution(1)));
    }
  }
}
