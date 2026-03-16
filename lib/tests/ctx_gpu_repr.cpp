#include "doctest/doctest.h"

#include "goblin/gp/context.h"
#include "goblin/gp/sr.h"
#include "goblin/gp/gpu_evaluation/types.h"

using namespace goblin;

TEST_CASE("goblin::gp::ctx_gpu_repr::single_output") {
    using namespace test;

    auto tree = TemplateNode::full_nary(2, 3);
    Template tmplate;
    tmplate.add_output(tree);

    Arr2D<CType> X = Arr2D<CType>::Random(10, 3);
    Arr2D<CType> Y = Arr2D<CType>::Random(X.rows(), 1);

    std::vector<std::shared_ptr<OperatorBase>> operators = {
        std::make_shared<OpAdd>(), 
        std::make_shared<OpSubGPU>(),
        std::make_shared<OpMul>(), 
        std::make_shared<OpDiv>()
    };

    GPContext ctx(X.cols(), tmplate, operators, 0, "ercs", 10, false, 15);
    std::vector<std::string> obj = {"mse"};
    SRProblem srp(ctx, X, Y, std::nullopt, std::nullopt, obj, std::nullopt, false);

    Solution s(srp.archive_fitness().worst(), Vec<DType>::Zero(srp.num_discrete()),
                 Vec<CType>::Zero(srp.num_continuous()));
    
    std::vector<float> node_type; 
    std::vector<float> node_value;
    usize size = 0;

    std::vector<float> expected_node_type;
    std::vector<float> expected_node_value;
    usize expected_size;

    auto expect = [&](std::vector<float> types, std::vector<float> values) {
        types.resize(ctx.max_expression_size, std::numeric_limits<float>::max());
        values.resize(ctx.max_expression_size, std::numeric_limits<float>::max());
        expected_node_type = std::move(types);
        expected_node_value = std::move(values);
    };

    SUBCASE("subcase_0") {
        REQUIRE(node_type.empty());
        REQUIRE(node_value.empty());
        REQUIRE(expected_node_type.empty());
        REQUIRE(expected_node_value.empty());

        REQUIRE_EQ(size, 0);

        auto root = ctx.output_roots[0];
        auto left = ctx.children[root][0];
        auto right = ctx.children[root][1];

        s.discrete_values()(root) = ctx.op_idx2value[0]; // +
        s.discrete_values()(left) = 0;                   // x0
        s.discrete_values()(right) = 1;                  // x1

        expect({I, I, O}, {Idx(0), Idx(1), Add});
        expected_size = 3;
    }

    SUBCASE("subcase_1") {
        REQUIRE(node_type.empty());
        REQUIRE(node_value.empty());
        REQUIRE(expected_node_type.empty());
        REQUIRE(expected_node_value.empty());

        REQUIRE_EQ(size, 0);

        auto root = ctx.output_roots[0];
        auto left = ctx.children[root][0];
        auto right  = ctx.children[root][1];
        auto right_left = ctx.children[right][0];
        auto right_right = ctx.children[right][1];

        s.discrete_values()(root) = ctx.op_idx2value[2]; // *
        s.discrete_values()(left) = ctx.op_idx2value[0]; // +
        s.discrete_values()(right) = ctx.op_idx2value[1]; // -

        // Left child: x0 + 3.2
        s.discrete_values()(ctx.children[left][0]) = 0; // x0
        s.discrete_values()(ctx.children[left][1]) = ctx.num_inputs;
        s.continuous_values()(ctx.children[left][1]) = 3.2;

        // Right child: x2 - (x1 / -1.5)
        s.discrete_values()(right_left) = 2; // x2
        s.discrete_values()(right_right) = ctx.op_idx2value[3]; // /
        s.discrete_values()(ctx.children[right_right][0]) = 1; // x1
        s.discrete_values()(ctx.children[right_right][1]) = ctx.num_inputs;
        s.continuous_values()(ctx.children[right_right][1]) = -1.5;

        expect(
            {I, C, O, I, I, C, O, O, O},
            {Idx(0), Val(3.2), Add, Idx(2), Idx(1), Val(-1.5), Div, Sub, Mul}
        );
        expected_size = 9;
    }

    ctx.gpu_nodes_post_order(s, node_type, node_value, size);

    CHECK_EQ(node_type.size(), expected_node_type.size());
    CHECK_EQ(node_value.size(), expected_node_value.size());
    CHECK_EQ(node_type.size(), node_value.size());
    CHECK_EQ(size, expected_size);

    for (usize i = 0; i < ctx.max_expression_size; i++) {
        INFO("Index: ", i);
        INFO("Type: ", node_type[i], "\t|\t Expected: ", expected_node_type[i]);
        INFO("Value: ", node_value[i], "\t|\t Expected: ", expected_node_value[i]);
        
        CHECK_EQ(node_type[i], expected_node_type[i]);
        CHECK_EQ(node_value[i], expected_node_value[i]);
    }
}