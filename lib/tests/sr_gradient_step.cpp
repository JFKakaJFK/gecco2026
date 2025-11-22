#include <print>

#include "doctest/doctest.h"

#define EIGEN_DONT_VECTORIZE
#include <Eigen/Dense>

#include "goblin/gp/context.h"    // for GPContext
#include "goblin/gp/operator.h"   // for OpAdd, OpMul, OpSin
#include "goblin/gp/sr.h"         // for SRProblem
#include "goblin/gp/template.h"   // for Template, Templa...
#include "goblin/lib/archive.h"   // for UnboundedArchive
#include "goblin/lib/fitness.h"   // for Quality
#include "goblin/lib/rng.h"       // for Rng
#include "goblin/lib/solution.h"  // for Solution, AoSSet
#include "goblin/lib/types.h"     // for CType, Arr2D, Vec

using namespace goblin;

TEST_CASE("goblin::gp::sr_problem.gradient_steps") {
  CType vtr = 1e-8;

  auto tree = TemplateNode::full_nary(2, 3);
  Template tmplate;
  tmplate.add_output(tree);
  tmplate.add_output(tree);

  REQUIRE(tree.size() == 15);
  REQUIRE(tmplate.size() == 30);

  Arr2D<CType> X = Arr2D<CType>::Random(100, 1);
  Arr2D<CType> Y(X.rows(), 2);
  CType c0 = 2.72, c1 = 3.141;
  Y.col(0) = X.col(0).array() * c0 + c1;
  Y.col(1) = (X.col(0).array() * c0 + c1).sin();

  std::vector<std::shared_ptr<OperatorBase>> operators = {std::make_shared<OpAdd>(), std::make_shared<OpMul>(),
                                                          std::make_shared<OpSin>()};

  GPContext ctx(/* num_inputs = */ X.cols(), tmplate, operators, /* num_parameters = */ 0,
                /* constant_representation = */ "pool");

  for (usize ls = 0; ls <= 1; ls++) {
    bool linear_scaling = ls > 0;
    SRProblem srp(ctx, X, Y, std::nullopt, std::nullopt, {"nmse"}, std::nullopt, linear_scaling);
    srp.register_target({vtr});

    Rng rng(42, 0);
    AoSSet solutions, parents;
    std::vector<usize> indices{0};
    UnboundedArchive archive(srp.archive_fitness());

    Solution s(srp.archive_fitness().worst(), Vec<DType>::Zero(srp.num_discrete()),
               Vec<CType>::Zero(srp.num_continuous()));

    // x * c0 + c1
    auto n = ctx.output_roots[0];
    auto l = ctx.children[n][0];
    s.discrete_values()(n) = ctx.value2domain(n, ctx.op_idx2value[0]).value_or(1000);  // +
    s.discrete_values()(l) = ctx.value2domain(l, ctx.op_idx2value[1]).value_or(1000);  // *
    s.discrete_values()(ctx.children[l][0]) = 0;                                       // x0
    s.discrete_values()(ctx.children[l][1]) = 1;                                       // c0
    s.discrete_values()(ctx.children[n][1]) = 2;                                       // c1

    // sin(x * c0 + c1) - sin cannot be fully undone by linear scaling
    n = ctx.output_roots[1];
    l = ctx.children[n][0];
    auto ll = ctx.children[l][0];
    s.discrete_values()(n) = ctx.value2domain(n, ctx.op_idx2value[2]).value_or(1000);    // sin
    s.discrete_values()(l) = ctx.value2domain(l, ctx.op_idx2value[0]).value_or(1000);    // +
    s.discrete_values()(ll) = ctx.value2domain(ll, ctx.op_idx2value[1]).value_or(1000);  // *
    s.discrete_values()(ctx.children[ll][0]) = 0;                                        // x0
    s.discrete_values()(ctx.children[ll][1]) = 1;                                        // c0
    s.discrete_values()(ctx.children[l][1]) = 2;                                         // x1

    s.continuous_values()(0) = c0;
    s.continuous_values()(1) = c1;

    std::println("Outputs: {}", ctx.output_roots);
    std::println("Subfunctions: {}", ctx.subtree_roots);
    for (usize i = 0; i < ctx.num_discrete; i++) {
      std::println("{}: V = {}, C = {}", i, s.discrete_values()(i), ctx.children[i]);
      REQUIRE_MESSAGE(s.discrete_values()(i) < ctx.domain_sizes[i], i);
    }

    std::println("{}", srp.format_solution(s));

    Array<CType> params;
    Arr2D<CType> out = ctx.compute_outputs<CType>(s, X, params).value();
    for (isize col = 0; col < Y.cols(); col++) {
      for (isize row = 0; row < Y.rows(); row++) {
        REQUIRE_MESSAGE(Y(row, col) == doctest::Approx(out(row, col)), "X(", row, ", ", col, "): ", X(row, col));
      }
    }

    solutions.add(s);
    srp.evaluate(rng, solutions, indices);

    archive.update(solutions[0], false);
    std::println("{} @ {}", srp.format_solution(solutions[0]), srp.fitness().format(solutions[0].quality()));
    REQUIRE(srp.target_reached(archive));  // with the right values, this works
    archive.clear();  // but we want to test the gradient optimization, so we need to not have the solution in the
                      // archive...

    solutions[0].continuous_values()(0) = 123.4;  // definitely not 2.72
    solutions[0].continuous_values()(1) = -54.3;  // definitely not 3.141

    srp.evaluate(rng, solutions, indices);
    parents = solutions;

    archive.update(solutions[0], false);
    REQUIRE(!srp.target_reached(archive));  // the starting point is not good

    std::println("{}", srp.format_solution(solutions[0]));
    Quality prev = solutions[0].quality();
    srp.gradient_steps(rng, solutions, parents, indices, 1000);
    std::println("{}", srp.format_solution(solutions[0]));

    archive.update(solutions[0], false);
    REQUIRE_MESSAGE(
        srp.target_reached(archive), "Solution: ", srp.format_solution(solutions[0]),
        "\nBefore: ", srp.fitness().format(prev), "\nAfter: ",
        srp.fitness().format(solutions[0].quality()));  // but after gradient based optimization the vtr is reached
  }
}
