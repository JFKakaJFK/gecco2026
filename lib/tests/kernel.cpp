#include "doctest/doctest.h"

#include "goblin/bench/tracked.h"
#include "goblin/ga-gp/ga_sr.h"
#include "goblin/methods/mixed.h"

using namespace goblin;

/*
 * This test is only used for kernel profiling.
 * That is why the max_generations has been set to 1.
*/

TEST_CASE("goblin::ga-gp::ga_sr") {
    auto tree = TemplateNode::full_nary(2, 4);
    Template tmplate;

    tmplate.add_output(tree);

    REQUIRE_EQ(tree.size(), 31);

    Arr2D<CType> X = Arr2D<CType>::Random(512, 20);
    Arr2D<CType> Y = Arr2D<CType>::Random(X.rows(), 1);
    Y.col(0) = (X.col(12) + X.col(19)) * (X.col(11) * X.col(7) + X.col(2) - X.col(17));

    std::vector<std::shared_ptr<OperatorBase>> operators = {
        std::make_shared<OpAdd>(), std::make_shared<OpSubGPU>(),
        std::make_shared<OpMul>(), std::make_shared<OpDiv>(),
    };

    GPContext ctx(X.cols(), tmplate, operators, 0, "ercs", 10, false, 15);
    GASRProblem gasrp(ctx, X, Y);

    CType vtr = 1e-8;
    gasrp.register_target({vtr});
    Budget budget(/* max_evaluations = */ 1'000'000, /* max_generations = */ 1);

    auto gomea = MixedGOMEA(
        PopulationOptions(), 
        RvOptions{.enabled = false},
        IMSOptions(/* initial_population_size = */ 256, /* max_num_populations = */ 1)
    );

    auto [front, status] = Tracked::run(gasrp, gomea, budget, TrackingOptions("sr.csv"), /* seed = */ 42);

    // std::println("Status {}: {}", format_as(status), gasrp1.format_solution(front.so_solution(0)));
    INFO("Status: ", format_as(status), "\t|\tSolution: ", gasrp.format_solution(front->so_solution(0)));

    CHECK_EQ(status, TerminationStatus::GenerationLimitReached);
}
