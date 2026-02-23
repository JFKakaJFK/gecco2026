#include "doctest/doctest.h"

#include "goblin/bench/tracked.h"
#include "goblin/ga-gp/ga_sr.h"
#include "goblin/gp/sr.h"
#include "goblin/methods/mixed.h"

using namespace goblin;

TEST_CASE("goblin::ga-gp::ga_sr") {
    auto tree = TemplateNode::full_nary(2, 3);
    Template tmplate;

    tmplate.add_output(tree);

    REQUIRE_EQ(tree.size(), 15);

    Arr2D<CType> X = Arr2D<CType>::Random(10, 3);
    Arr2D<CType> Y(X.rows(), 1);
    Y.col(0) = X.col(0) + X.col(1);

    std::vector<std::shared_ptr<OperatorBase>> operators = {
        std::make_shared<OpAdd>(), std::make_shared<OpSubGPU>(),
        std::make_shared<OpMul>(), std::make_shared<OpDiv>(),
    };

    GPContext ctx(X.cols(), tmplate, operators, 0, "ercs", 10, false, 15);
    GASRProblem gasrp(ctx, X, Y);

    Rng rng(1, 0);
    AoSSet sset;

    Solution s(
        gasrp.archive_fitness().worst(), 
        Vec<DType>::Zero(gasrp.num_discrete()),
        Vec<CType>::Zero(gasrp.num_continuous())
    );
    sset.add(s);
    sset.add(s);

    REQUIRE_EQ(sset.size(), 2);

    sset[0].discrete_values()(ctx.output_roots[0]) = ctx.op_idx2value[0];  // +
    sset[0].discrete_values()(ctx.children[ctx.output_roots[0]][0]) = 0;   // x0
    sset[0].discrete_values()(ctx.children[ctx.output_roots[0]][1]) = 1;   // x1

    sset[1].discrete_values()(ctx.output_roots[0]) = ctx.op_idx2value[2];  // *
    sset[1].discrete_values()(ctx.children[ctx.output_roots[0]][0]) = 1;   // x1
    sset[1].discrete_values()(ctx.children[ctx.output_roots[0]][1]) = 2;   // x2

    std::vector<usize> indices{0, 1};
    gasrp.evaluate(rng, sset, indices);

    CHECK_EQ(sset[0].quality_as<SRQuality>().objectives(0), doctest::Approx(0.0));
    CHECK_NE(sset[1].quality_as<SRQuality>().objectives(0), doctest::Approx(0.0));

    sset.clear();

    CType vtr = 1e-8;
    gasrp.register_target({vtr});
    Budget budget(/* max_evaluations = */ 100000, /* max_generations = */ 100);

    auto gomea = MixedGOMEA(
        PopulationOptions(), 
        RvOptions{.enabled = false},
        IMSOptions(/* initial_population_size = */ 512, /* max_num_populations = */ 1)
    );

    auto [front, status] = Tracked::run(gasrp, gomea, budget, TrackingOptions("sr.csv"), /* seed = */ 42);

    // std::println("Status {}: {}", format_as(status), gasrp1.format_solution(front.so_solution(0)));
    INFO("Status: ", format_as(status), "\t|\tSolution: ", gasrp.format_solution(front->so_solution(0)));

    CHECK_EQ(status, TerminationStatus::TargetReached);
    CHECK_EQ(front->empty(), false);
}
