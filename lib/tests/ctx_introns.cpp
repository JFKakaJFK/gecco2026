#include <print>

#include "doctest/doctest.h"
#include <Eigen/Dense>
#include <random>

#include "goblin/bench/tracked.h"
#include "goblin/gp/sr.h"
#include "goblin/methods/mixed.h"

using namespace std::chrono_literals;
using namespace goblin;

TEST_CASE("goblin::gp::ctx_introns") {
  // idea: generate and evaluate random solutions, then randomly mess with the introns and evaluate - if there are
  // introns that change the value, that's a problem...
  const usize N = 5000;

  std::vector<usize> indices(N);
  std::iota(indices.begin(), indices.end(), 0);

  auto tree = TemplateNode::full_nary(/* branching_factor = */ 3, /* depth */ 3);
  Template tmplate;
  tmplate.add_subtree(tree);
  tmplate.add_subtree(tree);
  tmplate.add_output(tree);

  Arr2D<CType> X = Arr2D<CType>::Random(10, 2);
  Arr2D<CType> Y = X(Eigen::placeholders::all, std::vector<int>{0});

  std::vector<std::shared_ptr<OperatorBase>> operators = {
      // clang-format off
      std::make_shared<OpAdd>(),
      std::make_shared<OpSub>(),
      std::make_shared<OpMul>(),
      std::make_shared<OpDiv>(),
      std::make_shared<OpSin>(),
      std::make_shared<OpCos>(),
      // std::make_shared<OpExp>(),
      // std::make_shared<OpLog>(),
      // std::make_shared<OpSquare>(),
      // std::make_shared<OpSqrt>(),
      // std::make_shared<OpPow>(),
      std::make_shared<OpAbs>(),
      std::make_shared<OpMin>(),
      std::make_shared<OpMax>()
      // clang-format on
  };

  std::vector<std::string> objectives{"mse"};
  std::vector<std::string> constant_reprs{"ercs", "pool"};

  for (usize es = 0; es <= 1; es++) {
    for (usize ls = 0; ls <= 1; ls++) {
      for (auto constant_representation : constant_reprs) {
        bool linear_scaling = ls > 0;
        bool enable_subfunctions = es > 0;

        GPContext ctx(
            /* num_inputs = */ X.cols(), tmplate, operators, /* num_parameters = */ 0, constant_representation,
            /* constant_pool_size = */ 10, enable_subfunctions);

        SRProblem srp(ctx, X, Y, std::nullopt, std::nullopt, objectives, /* objectives_to_optimize = */ std::nullopt,
                      linear_scaling);

        Rng rng(1, 0);
        AoSSet solutions;

        srp.add_random(rng, solutions, N);

        srp.evaluate(rng, solutions, indices);

        Quality worst = srp.archive_fitness().worst();

        bool at_least_one_valid = false;
        for (usize i = 0; i < N; i++) {
          // it's possible to generate semantically invalid expressions...
          if (!isna(solutions[i].quality().objectives(0))) {
            at_least_one_valid = true;
            Solution before = solutions[i];

            // resample all inactive discrete and continuous values
            for (usize j = 0; j < srp.num_discrete(); j++) {
              if (!solutions[i].discrete_active()(j)) {
                std::uniform_int_distribution<DType> d(0, srp.discrete_domain_sizes()(j) - 1);
                DType previous = solutions[i].discrete_values()(j);
                while (previous == solutions[i].discrete_values()(j)) {
                  solutions[i].discrete_values()(j) = d(rng);
                }
              }
            }
            for (usize j = 0; j < srp.num_continuous(); j++) {
              if (!solutions[i].continuous_active()(j)) {
                std::normal_distribution<CType> N(0.0, 1.0);
                CType previous = solutions[i].continuous_values()(j);
                while (previous == solutions[i].continuous_values()(j)) {
                  solutions[i].continuous_values()(j) = N(rng);
                }
              }
            }

            srp.evaluate(rng, solutions, std::vector<usize>{i});

            auto dist = srp.fitness().distance(before.quality(), solutions[i].quality(), std::nullopt);

            REQUIRE_MESSAGE(dist == doctest::Approx(0.0), srp.format_solution(before),
                            " != ", srp.format_solution(solutions[i]), "(", srp.InstanceBase::format_solution(before),
                            " != ", srp.InstanceBase::format_solution(solutions[i]), " / ",
                            srp.fitness().format(before.quality()),
                            " != ", srp.fitness().format(solutions[i].quality()), ")");
          }
          // else {
          //   std::println("{} @ {} / {}", srp.format_solution(solutions[i]),
          //   srp.fitness().format(solutions[i].quality()), format_as(srp.fitness().cmp(solutions[i].quality(), worst,
          //   std::nullopt)));
          // }
        }
        REQUIRE_MESSAGE(at_least_one_valid, "FNs=", enable_subfunctions, ", LS=", linear_scaling,
                        ", CRepr=", constant_representation);
      }
    }
  }
}
