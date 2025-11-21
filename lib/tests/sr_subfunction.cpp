#include <print>

#include "doctest/doctest.h"
#include <Eigen/Dense>

#include "goblin/bench/tracked.h"
#include "goblin/gp/sr.h"
#include "goblin/methods/mixed.h"

using namespace std::chrono_literals;
using namespace goblin;

TEST_CASE("goblin::gp::ctx_subfunction") {
  auto tree = TemplateNode::full_nary(/* branching_factor = */ 2, /* depth */ 1);
  Template tmplate;
  tmplate.add_subtree(tree);
  tmplate.add_subtree(tree);
  tmplate.add_output(tree);

  Arr2D<CType> X = Arr2D<CType>::Random(10, 2);
  Arr2D<CType> Y(X.rows(), 1);
  Y.col(0) = X.col(1) - X.col(0);

  std::vector<std::shared_ptr<OperatorBase>> operators = {std::make_shared<OpSub>()};

  GPContext ctx(
      /* num_inputs = */ X.cols(), tmplate, operators, /* num_parameters = */ 0, /* constant_representation = */ "ercs",
      /* constant_pool_size = */ 0, /* enable_subfunctions = */ true);

  auto f = MOFitness(1);
  Solution s(f.worst(), Vec<DType>::Zero(ctx.num_discrete), Vec<CType>::Zero(ctx.num_continuous));

  // 0 -> x0
  // 1 -> x1
  // 2 -> constant
  // 3 -> arg[0] / fn[0] / op[0]

  // Arg[0] - Arg[1]
  s.discrete_values()(ctx.subtree_roots[0]) =
      ctx.value2domain(ctx.subtree_roots[0], ctx.op_idx2value[0]).value_or(1000);  // -
  s.discrete_values()(ctx.children[ctx.subtree_roots[0]][0]) = 3;                  // Arg[0]
  s.discrete_values()(ctx.children[ctx.subtree_roots[0]][1]) = 4;                  // Arg[1]

  // Fn[0](Arg[1], Arg[0])
  s.discrete_values()(ctx.subtree_roots[1]) = 5;                   // Fn[0]
  s.discrete_values()(ctx.children[ctx.subtree_roots[1]][0]) = 4;  // Arg[1]
  s.discrete_values()(ctx.children[ctx.subtree_roots[1]][1]) = 3;  // Arg[0]

  // Fn[1](x0, x1) = ... = x1 - x0
  s.discrete_values()(ctx.output_roots[0]) = 4;                   // Fn[1]
  s.discrete_values()(ctx.children[ctx.output_roots[0]][0]) = 0;  // x0
  s.discrete_values()(ctx.children[ctx.output_roots[0]][1]) = 1;  // x1

  std::println("Outputs: {}", ctx.output_roots);
  std::println("Subfunctions: {}", ctx.subtree_roots);
  for (usize i = 0; i < ctx.num_discrete; i++) {
    std::println("{}: V = {}, C = {}", i, s.discrete_values()(i), ctx.children[i]);
    REQUIRE_MESSAGE(s.discrete_values()(i) < ctx.domain_sizes[i], i);
  }

  Array<CType> params;
  Arr2D<CType> out = ctx.compute_outputs<CType>(s, X, params).value();

  std::println("{}", ctx.to_sympy(s));

  for (isize col = 0; col < Y.cols(); col++) {
    for (isize row = 0; row < Y.rows(); row++) {
      REQUIRE(Y(row, col) == doctest::Approx(out(row, col)));
    }
  }
}
