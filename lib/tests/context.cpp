#include "doctest/doctest.h"

#include "goblin/gp/context.h"
#include "goblin/gp/sr.h"

using namespace goblin;

TEST_CASE("goblin::gp::context_single_output") {
    using namespace test;

    auto tree = TemplateNode::full_nary(2, 3);
    Template tmplate;
    tmplate.add_output(tree);

    usize size;

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

    AoSSet sset;

    Solution s(srp.archive_fitness().worst(), Vec<DType>::Zero(srp.num_discrete()),
                 Vec<CType>::Zero(srp.num_continuous()));
    sset.add(s);

    std::vector<float> node_type; 
    std::vector<float> node_value;

    std::vector<float> expected_node_type;
    std::vector<float> expected_node_value;

    auto expect = [&](std::vector<float> types, std::vector<float> values) {
        types.resize(ctx.num_discrete, std::numeric_limits<float>::max());
        values.resize(ctx.num_discrete, std::numeric_limits<float>::max());
        expected_node_type = std::move(types);
        expected_node_value = std::move(values);
    };

    SUBCASE("subcase_1") {
        REQUIRE(node_type.empty());
        REQUIRE(node_value.empty());
        REQUIRE(expected_node_type.empty());
        REQUIRE(expected_node_value.empty());

        auto root = ctx.output_roots[0];
        auto left = ctx.children[root][0];
        auto right = ctx.children[root][1];

        sset[0].discrete_values()(root) = ctx.op_idx2value[0];  // +
        sset[0].discrete_values()(left) = 0;   // x0
        sset[0].discrete_values()(right) = 1;   // x1

        expect({I, I, O}, {Idx(0), Idx(1), Add});
    }

    SUBCASE("subcase_1") {
        REQUIRE(node_type.empty());
        REQUIRE(node_value.empty());
        REQUIRE(expected_node_type.empty());
        REQUIRE(expected_node_value.empty());

        auto root = ctx.output_roots[0];
        auto left = ctx.children[root][0];
        auto right  = ctx.children[root][1];
        auto right_left = ctx.children[right][0];
        auto right_right = ctx.children[right][1];

        sset[0].discrete_values()(root) = ctx.op_idx2value[2]; // *
        sset[0].discrete_values()(left) = ctx.op_idx2value[0]; // +
        sset[0].discrete_values()(right) = ctx.op_idx2value[1]; // -

        // Left child: x0 + 3.2
        sset[0].discrete_values()(ctx.children[left][0]) = 0; // x0
        sset[0].discrete_values()(ctx.children[left][1]) = ctx.num_inputs;
        sset[0].continuous_values()(ctx.children[left][1]) = 3.2;

        // Right child: x2 - (x1 / -1.5)
        sset[0].discrete_values()(right_left) = 2; // x2
        sset[0].discrete_values()(right_right) = ctx.op_idx2value[3]; // /
        sset[0].discrete_values()(ctx.children[right_right][0]) = 1; // x1
        sset[0].discrete_values()(ctx.children[right_right][1]) = ctx.num_inputs;
        sset[0].continuous_values()(ctx.children[right_right][1]) = -1.5;

        expect(
            {I, C, O, I, I, C, O, O, O},
            {Idx(0), Val(3.2), Add, Idx(2), Idx(1), Val(-1.5), Div, Sub, Mul}
        );
    }

    ctx.gpu_nodes_post_order(sset[0], node_type, node_value, true, size);

    CHECK_EQ(node_type.size(), expected_node_type.size());
    CHECK_EQ(node_value.size(), expected_node_value.size());

    for (usize i = 0; i < ctx.num_discrete; i++) {
        INFO("Index: ", i, "\t|\t Type: ", node_type[i], "\t|\t Value: ", node_value[i]);
        CHECK_EQ(node_type[i], expected_node_type[i]);
        CHECK_EQ(node_value[i], expected_node_value[i]);
    }
}

TEST_CASE("goblin::gp::context_multiple_output") {
    using namespace test;

    auto tree = TemplateNode::full_nary(2, 3);
    Template tmplate;
    tmplate.add_output(tree);
    tmplate.add_output(tree);

    usize size;

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

    AoSSet sset;

    Solution s(srp.archive_fitness().worst(), Vec<DType>::Zero(srp.num_discrete()),
                 Vec<CType>::Zero(srp.num_continuous()));
    sset.add(s);

    std::vector<float> node_type; 
    std::vector<float> node_value;

    std::vector<float> expected_node_type;
    std::vector<float> expected_node_value;

    auto expect = [&](std::vector<float> types, std::vector<float> values) {
        REQUIRE(types.size() == values.size());

        // Append output nodes
        expected_node_type.insert(
            expected_node_type.end(), types.begin(), types.end()
        );
        expected_node_value.insert(
            expected_node_value.end(), values.begin(), values.end()
        );

        // Pad this output to stride = ctx.num_discrete
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

        // Output 0
        auto root0 = ctx.output_roots[0];
        auto left0 = ctx.children[root0][0];
        auto right0 = ctx.children[root0][1];

        sset[0].discrete_values()(root0) = ctx.op_idx2value[0];  // +
        sset[0].discrete_values()(left0) = 0;   // x0
        sset[0].discrete_values()(right0) = 1;   // x1

        // Output 1
        auto root1 = ctx.output_roots[1];
        auto left1 = ctx.children[root1][0];
        auto right1 = ctx.children[root1][1];

        sset[0].discrete_values()(root1) = ctx.op_idx2value[1];  // -
        sset[0].discrete_values()(left1) = 1;   // x1
        sset[0].discrete_values()(right1) = 0;   // x0

        expect({I, I, O}, {Idx(0), Idx(1), Add});
        expect({I, I, O}, {Idx(1), Idx(0), Sub});
    }

    SUBCASE("subcase_1") {
        REQUIRE(node_type.empty());
        REQUIRE(node_value.empty());
        REQUIRE(expected_node_type.empty());
        REQUIRE(expected_node_value.empty());

        // Output 0
        auto root = ctx.output_roots[0];
        auto left = ctx.children[root][0];
        auto right  = ctx.children[root][1];
        auto right_left = ctx.children[right][0];
        auto right_right = ctx.children[right][1];

        sset[0].discrete_values()(root) = ctx.op_idx2value[2]; // *
        sset[0].discrete_values()(left) = ctx.op_idx2value[0]; // +
        sset[0].discrete_values()(right) = ctx.op_idx2value[1]; // -

        // Left child: x0 + 3.2
        sset[0].discrete_values()(ctx.children[left][0]) = 0; // x0
        sset[0].discrete_values()(ctx.children[left][1]) = ctx.num_inputs;
        sset[0].continuous_values()(ctx.children[left][1]) = 3.2;

        // Right child: x2 - (x1 / -1.5)
        sset[0].discrete_values()(right_left) = 2; // x2
        sset[0].discrete_values()(right_right) = ctx.op_idx2value[3]; // /
        sset[0].discrete_values()(ctx.children[right_right][0]) = 1; // x1
        sset[0].discrete_values()(ctx.children[right_right][1]) = ctx.num_inputs;
        sset[0].continuous_values()(ctx.children[right_right][1]) = -1.5;

        // Output 1
        auto root1 = ctx.output_roots[1];
        auto left1 = ctx.children[root1][0];
        auto right1 = ctx.children[root1][1];

        sset[0].discrete_values()(root1) = ctx.op_idx2value[1];  // -
        sset[0].discrete_values()(left1) = 1;   // x1
        sset[0].discrete_values()(right1) = 0;   // x0

        expect(
            {I, C, O, I, I, C, O, O, O},
            {Idx(0), Val(3.2), Add, Idx(2), Idx(1), Val(-1.5), Div, Sub, Mul}
        );
        expect({I, I, O}, {Idx(1), Idx(0), Sub});
    }

    ctx.gpu_nodes_post_order(sset[0], node_type, node_value, true, size);

    CHECK_EQ(node_type.size(), expected_node_type.size());
    CHECK_EQ(node_value.size(), expected_node_value.size());

    for (usize i = 0; i < ctx.num_discrete; i++) {
        INFO("Index: ", i, "\t|\t Type: ", node_type[i], "\t|\t Value: ", node_value[i]);
        CHECK_EQ(node_type[i], expected_node_type[i]);
        CHECK_EQ(node_value[i], expected_node_value[i]);
    }
}

