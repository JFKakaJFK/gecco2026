#include <print>
#include <iostream>

#define EIGEN_DONT_ALIGN
#define EIGEN_MAX_ALIGN_BYTES 0
#define EIGEN_DONT_VECTORIZE
#include "doctest/doctest.h"
#include <Eigen/Dense>

#include "goblin/bench/tracked.h"
#include "goblin/gp/sr.h"
#include "goblin/methods/mixed.h"

using namespace std::chrono_literals;
using namespace goblin;

TEST_CASE("goblin::gp::sr") {
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

  std::vector<std::vector<std::string>> objs = {{"mse"}, {"nmse"}, {"mse", "size"}};
  for (auto& obj : objs) {
    for (usize ls = 0; ls <= 1; ls++) {
      bool linear_scaling = ls > 0;
      SRProblem srp(ctx, X, Y, X_test, Y_test, obj, /* objectives_to_optimize = */ std::nullopt, linear_scaling);

      Rng rng = seeded_rng(42);

      AoSSet sset;

      Solution s(srp.archive_fitness().worst(), Vec<DType>::Zero(srp.num_discrete()),
                 Vec<CType>::Zero(srp.num_continuous()));
      sset.add(s);

      sset[0].discrete_values()(ctx.output_roots[0]) = ctx.op_idx2value[0];  // +
      sset[0].discrete_values()(ctx.children[ctx.output_roots[0]][0]) = 0;   // x0
      sset[0].discrete_values()(ctx.children[ctx.output_roots[0]][1]) = 1;   // x1

      sset[0].discrete_values()(ctx.output_roots[1]) = ctx.op_idx2value[2];  // *
      sset[0].discrete_values()(ctx.children[ctx.output_roots[1]][0]) = 1;   // x1
      sset[0].discrete_values()(ctx.children[ctx.output_roots[1]][1]) = 2;   // x2

      for (auto& rep : ctx.to_sympy(sset[0])) {
        std::println("{}", rep);
      }

      Arr2D<CType> buffer;
      Array<CType> params;
      usize expression_size;
      auto Y_pred = ctx.compute_outputs(buffer, sset[0], X, params, expression_size).value();

      REQUIRE(expression_size == 6);

      for (isize i = 0; i < Y.cols(); i++) {
        for (isize j = 0; j < Y.rows(); j++) {
          REQUIRE_MESSAGE(Y_pred(j, i) == doctest::Approx(Y(j, i)), j, i);
        }
      }

      std::vector<usize> indices{0};
      srp.evaluate(rng, sset, indices);

      REQUIRE(static_cast<const MOQuality&>(sset[0].quality()).objectives(0) == doctest::Approx(0.0));
      if (srp.num_objectives() > 1) {
        REQUIRE(static_cast<const MOQuality&>(sset[0].quality()).objectives(1) == 6.0);
      }

      sset[0].discrete_values()(ctx.output_roots[0]) = ctx.op_idx2value[0];  // +
      sset[0].discrete_values()(ctx.children[ctx.output_roots[0]][0]) = 1;   // x1
      sset[0].discrete_values()(ctx.children[ctx.output_roots[0]][1]) = 0;   // x0

      sset[0].discrete_values()(ctx.output_roots[1]) = ctx.op_idx2value[2];  // *
      sset[0].discrete_values()(ctx.children[ctx.output_roots[1]][0]) = 2;   // x2
      sset[0].discrete_values()(ctx.children[ctx.output_roots[1]][1]) = 1;   // x1

      for (auto& rep : ctx.to_sympy(sset[0])) {
        std::println("{}", rep);
      }

      Y_pred = ctx.compute_outputs(buffer, sset[0], X, params, expression_size).value();

      for (isize i = 0; i < Y.cols(); i++) {
        for (isize j = 0; j < Y.rows(); j++) {
          REQUIRE_MESSAGE(Y_pred(j, i) == doctest::Approx(Y(j, i)), j, i);
        }
      }

      srp.evaluate(rng, sset, indices);

      REQUIRE(static_cast<MOQuality&>(sset[0].quality()).objectives(0) == doctest::Approx(0.0));

      // ls (x0 + x1 + 42.0 -> scaled to x0 + x1)
      std::vector<CType> ls_intercepts = {42.0, 0.0};
      sset[0].discrete_values()(ctx.output_roots[0]) = ctx.op_idx2value[0];                   // +
      sset[0].discrete_values()(ctx.children[ctx.output_roots[0]][0]) = 1;                    // x1
      sset[0].discrete_values()(ctx.children[ctx.output_roots[0]][1]) = ctx.op_idx2value[0];  // +
      auto rc = ctx.children[ctx.output_roots[0]][1];
      sset[0].discrete_values()(ctx.children[rc][0]) = 0;  // x0
      sset[0].discrete_values()(ctx.children[rc][1]) =
          ctx.num_inputs;  // first constant/node is constant depending on repr
      sset[0].continuous_values()(ctx.children[rc][1]) = ls_intercepts[0];

      sset[0].discrete_values()(ctx.output_roots[1]) = ctx.op_idx2value[2];  // *
      sset[0].discrete_values()(ctx.children[ctx.output_roots[1]][0]) = 2;   // x2
      sset[0].discrete_values()(ctx.children[ctx.output_roots[1]][1]) = 1;   // x1

      for (auto& rep : ctx.to_sympy(sset[0])) {
        std::println("{}", rep);
      }

      Y_pred = ctx.compute_outputs(buffer, sset[0], X, params, expression_size).value();

      for (isize i = 0; i < Y.cols(); i++) {
        for (isize j = 0; j < Y.rows(); j++) {
          REQUIRE_MESSAGE(Y_pred(j, i) == doctest::Approx(Y(j, i) + ls_intercepts[i]), j, i);
        }
      }

      if (linear_scaling) {
        srp.evaluate(rng, sset, indices);

        REQUIRE(static_cast<MOQuality&>(sset[0].quality()).objectives(0) == doctest::Approx(0.0));
      }
    }
  }

  std::vector<std::string> constant_reps = {"ercs", "pool", "edges", "none"};
  for (auto& crep : constant_reps) {
    for (usize ic = 0; ic <= 2; ic++) {
      std::optional<bool> always_inherit_continuous = ic < 2 ? std::make_optional(ic > 0) : std::nullopt;
      GPContext pctx(
          /* num_inputs = */ X.cols(), tmplate, operators,
          /* num_parameters = */ 0, crep);

      CType vtr =
          crep == "edges" ? 1e-6 : 1e-8;  // edges is less numerically stable, but this is just to speed up the test...
      SRProblem srp(pctx, X, Y, X_test, Y_test, {"nmse"}, /* objectives_to_optimize = */ std::nullopt,
                    /* linear_scaling = */ true,
                    /* init = */ std::nullopt,
                    /* constant_init_lower_bound = */ -1.0,
                    /* constant_init_upper_bound = */ 1.0,
                    /* target_objectives = */ std::nullopt,
                    /* gradient_mode = */ "forward",
                    /* gradient_epsilon = */ 1e-5,
                    /* archive_epsilon = */ 0.0, always_inherit_continuous);
      srp.register_target({vtr});

      Budget budget(/* max_evaluations = */ 100000, /* max_generations = */ 100);

      std::vector<std::unique_ptr<LinkageModelBase>> models;
      for (usize i : srp.ctx.subtree_roots) {
        models.push_back(std::make_unique<LinkageTreeFOS>(
            /* metric = */ "mi",
            /* intron_strategy = */ "any_active",
            /* merge_continuous = */ true,
            /* num_continuous_bins = */ std::nullopt,
            /* filter_parent_threshold = */ 1e-6,
            /* filter_children_threshold = */ 1.0 - 1e-6,
            /* filter_root = */ true,
            /* max_subset_size = */ std::nullopt,
            /* normalize_initial_linkage_bias = */ true,
            /* subset = */ Subset{.discrete = ctx.nodes[i]}));
      }
      for (usize i : srp.ctx.output_roots) {
        models.push_back(std::make_unique<LinkageTreeFOS>(
            /* metric = */ "mi",
            /* intron_strategy = */ "any_active",
            /* merge_continuous = */ true,
            /* num_continuous_bins = */ std::nullopt,
            /* filter_parent_threshold = */ 1e-6,
            /* filter_children_threshold = */ 1.0 - 1e-6,
            /* filter_root = */ true,
            /* max_subset_size = */ std::nullopt,
            /* normalize_initial_linkage_bias = */ true,
            /* subset = */ Subset{.discrete = ctx.nodes[i]}));
      }

      auto gomea = MixedGOMEA(
          PopulationOptions{
              // .subset_logfile = "fos_stats.csv"
          },
          RvOptions{.enabled = false},
          IMSOptions(
              /* initial_population_size = */ 256,
              /* max_num_populations = */ 1),
          std::make_shared<CombinedFOS>(models));

      // auto [front, status] = Tracked::run(srp, gomea, budget, TrackingOptions("sr.csv"), /* seed = */ 42);
      auto [front, status] = gomea.run(srp, budget, /* seed = */ 42);

      std::println("Status {}: {}", format_as(status), srp.format_solution(front->so_solution(0)));

      REQUIRE(front->empty() == false);
      // ls values are re-computed, so there can be slight differences here, hence 10x
      REQUIRE(static_cast<const MOQuality&>(front->so_solution(0).quality()).objectives(0) <= vtr * 10.0);
    }
  }
}
