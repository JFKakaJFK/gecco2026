#include "doctest/doctest.h"

#include <print>

#include "goblin/gp/context.h"
#include "goblin/gp/sr.h"

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

TEST_CASE("goblin::gp::ctx_gpu_repr::multiple_output") {
    using namespace test;

    auto tree = TemplateNode::full_nary(2, 3);
    Template tmplate;
    tmplate.add_output(tree);
    tmplate.add_output(tree);

    Arr2D<CType> X = Arr2D<CType>::Random(10, 3);
    Arr2D<CType> Y = Arr2D<CType>::Random(X.rows(), 2);

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
        REQUIRE(types.size() == values.size());

        // Append output nodes
        expected_node_type.insert(
            expected_node_type.end(), types.begin(), types.end()
        );
        expected_node_value.insert(
            expected_node_value.end(), values.begin(), values.end()
        );

        const usize pad = ctx.max_expression_size - types.size();
        expected_node_type.insert(
            expected_node_type.end(), pad, std::numeric_limits<float>::max()
        );
        expected_node_value.insert(
            expected_node_value.end(), pad, std::numeric_limits<float>::max()
        );
    };


    SUBCASE("subcase_0") {
        REQUIRE(node_type.empty());
        REQUIRE(node_value.empty());
        REQUIRE(expected_node_type.empty());
        REQUIRE(expected_node_value.empty());

        REQUIRE_EQ(size, 0);

        // Output 0
        auto root0 = ctx.output_roots[0];
        auto left0 = ctx.children[root0][0];
        auto right0 = ctx.children[root0][1];

        s.discrete_values()(root0) = ctx.op_idx2value[0];  // +
        s.discrete_values()(left0) = 0;   // x0
        s.discrete_values()(right0) = 1;   // x1

        // Output 1
        auto root1 = ctx.output_roots[1];
        auto left1 = ctx.children[root1][0];
        auto right1 = ctx.children[root1][1];

        s.discrete_values()(root1) = ctx.op_idx2value[1];  // -
        s.discrete_values()(left1) = 1;   // x1
        s.discrete_values()(right1) = 0;   // x0

        expect({I, I, O}, {Idx(0), Idx(1), Add});
        expect({I, I, O}, {Idx(1), Idx(0), Sub});
        expected_size = 6;
    }

    SUBCASE("subcase_1") {
        REQUIRE(node_type.empty());
        REQUIRE(node_value.empty());
        REQUIRE(expected_node_type.empty());
        REQUIRE(expected_node_value.empty());

        REQUIRE_EQ(size, 0);

        // Output 0
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

        // Output 1
        auto root1 = ctx.output_roots[1];
        auto left1 = ctx.children[root1][0];
        auto right1 = ctx.children[root1][1];

        s.discrete_values()(root1) = ctx.op_idx2value[1];  // -
        s.discrete_values()(left1) = 1;   // x1
        s.discrete_values()(right1) = 0;   // x0

        expect(
            {I, C, O, I, I, C, O, O, O},
            {Idx(0), Val(3.2), Add, Idx(2), Idx(1), Val(-1.5), Div, Sub, Mul}
        );
        expect({I, I, O}, {Idx(1), Idx(0), Sub});
        expected_size = 12;
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

TEST_CASE("goblin::gp::context_subexpressions") {
    using namespace test;

    auto tree = TemplateNode::full_nary(2, 3);
    Template tmplate;
    tmplate.add_subtree(tree);
    tmplate.add_subtree(tree);
    tmplate.add_output(tree);

    Arr2D<CType> X = Arr2D<CType>::Random(10, 2);
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

        // Subexpression: Arg[0] - Arg[1]
        s.discrete_values()(ctx.subtree_roots[0]) =
            ctx.value2domain(ctx.subtree_roots[0], ctx.op_idx2value[0]).value_or(1000);  // -
        s.discrete_values()(ctx.children[ctx.subtree_roots[0]][0]) = 3;                  // Arg[0]
        s.discrete_values()(ctx.children[ctx.subtree_roots[0]][1]) = 4;                  // Arg[1]

        // Subfunction: Fn[0](Arg[1], Arg[0])
        s.discrete_values()(ctx.subtree_roots[1]) = 5;                   // Fn[0]
        s.discrete_values()(ctx.children[ctx.subtree_roots[1]][0]) = 4;  // Arg[1]
        s.discrete_values()(ctx.children[ctx.subtree_roots[1]][1]) = 3;  // Arg[0]

        // Main Tree: Fn[1](x0, x1) = ... = x1 - x0
        s.discrete_values()(ctx.output_roots[0]) = 4;                   // Fn[1]
        s.discrete_values()(ctx.children[ctx.output_roots[0]][0]) = 0;  // x0
        s.discrete_values()(ctx.children[ctx.output_roots[0]][1]) = 1;  // x1
        
        expect({I, I, O}, {Idx(0), Idx(1), Sub});
        expected_size = 3;
    }

    // SUBCASE("subcase_1") {
    //     REQUIRE(node_type.empty());
    //     REQUIRE(node_value.empty());
    //     REQUIRE(expected_node_type.empty());
    //     REQUIRE(expected_node_value.empty());

    //     auto root = ctx.output_roots[0];
    //     auto left = ctx.children[root][0];
    //     auto right  = ctx.children[root][1];
    //     auto right_left = ctx.children[right][0];
    //     auto right_right = ctx.children[right][1];

    //     sset[0].discrete_values()(root) = ctx.op_idx2value[2]; // *
    //     sset[0].discrete_values()(left) = ctx.op_idx2value[0]; // +
    //     sset[0].discrete_values()(right) = ctx.op_idx2value[1]; // -

    //     // Left child: x0 + 3.2
    //     sset[0].discrete_values()(ctx.children[left][0]) = 0; // x0
    //     sset[0].discrete_values()(ctx.children[left][1]) = ctx.num_inputs;
    //     sset[0].continuous_values()(ctx.children[left][1]) = 3.2;

    //     // Right child: x2 - (x1 / -1.5)
    //     sset[0].discrete_values()(right_left) = 2; // x2
    //     sset[0].discrete_values()(right_right) = ctx.op_idx2value[3]; // /
    //     sset[0].discrete_values()(ctx.children[right_right][0]) = 1; // x1
    //     sset[0].discrete_values()(ctx.children[right_right][1]) = ctx.num_inputs;
    //     sset[0].continuous_values()(ctx.children[right_right][1]) = -1.5;

    //     expect(
    //         {I, C, O, I, I, C, O, O, O},
    //         {Idx(0), Val(3.2), Add, Idx(2), Idx(1), Val(-1.5), Div, Sub, Mul}
    //     );
    // }

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
        // CHECK_EQ(expected_node_type.size(), size);
    }
}