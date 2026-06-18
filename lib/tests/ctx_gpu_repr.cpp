#include "doctest/doctest.h"

#include "goblin/gp/context.h"
#include "goblin/gp/gpu_evaluation/types.h"
#include "goblin/lib/fitness.h"

using namespace goblin;

TEST_CASE("goblin::gp::ctx_gpu_repr::simple_tree") {
    using namespace test;

    // Single binary tree with depth 3 (4 layers)
    auto tree = TemplateNode::full_nary(2, 3);
    Template tmplate;
    tmplate.add_output(tree);

    std::vector<std::shared_ptr<OperatorBase>> operators = {
        std::make_shared<OpSubGPU>(),
        std::make_shared<OpMul>(),
        std::make_shared<OpDiv>(),
        std::make_shared<OpAdd>(),
    };

    GPContext ctx(
        /* num_inputs = */ 3, tmplate, operators, /* num_parameters = */ 0,
        /* constant_representation = */ "ercs", /* constant_pool_size= */ 10, /* enable_subfunctions= */ false);

    auto f = MOFitness(/* num_objectives= */ 1);
    Solution s(f.worst(), Vec<DType>::Zero(ctx.num_discrete), Vec<CType>::Zero(ctx.num_continuous));

    auto root = ctx.output_roots[0];
    auto left = ctx.children[root][0];
    auto right = ctx.children[root][1];
    auto right_left = ctx.children[right][0];
    auto right_right = ctx.children[right][1];

    std::vector<u8> node_type;
    std::vector<float> node_value;
    usize size = 0;

    std::vector<u8> expected_node_type;
    std::vector<float> expected_node_value;
    usize expected_size;

    auto expect = [&](std::vector<u8> types, std::vector<float> values) {
        types.resize(ctx.max_expression_size, std::numeric_limits<u8>::max());
        values.resize(ctx.max_expression_size, std::numeric_limits<float>::max());
        expected_node_type = std::move(types);
        expected_node_value = std::move(values);
    };

    REQUIRE(node_type.empty());
    REQUIRE(node_value.empty());
    REQUIRE(expected_node_type.empty());
    REQUIRE(expected_node_value.empty());
    REQUIRE_EQ(size, 0);

    SUBCASE("x0 + x1") {
        s.discrete_values()(root) = ctx.op_idx2value[3];  // +
        s.discrete_values()(left) = 0;                    // x0
        s.discrete_values()(right) = 1;                   // x1

        expect({I, I, O}, {Idx(0), Idx(1), Add});
        expected_size = 3;
    }

    SUBCASE("(x0 + 3.2) * (x2 - (x1 / -2.5))") {
        // Root
        s.discrete_values()(root) = ctx.op_idx2value[1];  // *

        // Left child: x0 + 3.2
        s.discrete_values()(left) = ctx.op_idx2value[3];  // +
        s.discrete_values()(ctx.children[left][0]) = 0;   // x0
        s.discrete_values()(ctx.children[left][1]) = ctx.num_inputs;
        s.continuous_values()(ctx.children[left][1]) = 3.2;

        // Right child: x2 - (x1 / -1.5)
        s.discrete_values()(right) = ctx.op_idx2value[0];        // -
        s.discrete_values()(right_left) = 2;                     // x2
        s.discrete_values()(right_right) = ctx.op_idx2value[2];  // /
        s.discrete_values()(ctx.children[right_right][0]) = 1;   // x1
        s.discrete_values()(ctx.children[right_right][1]) = ctx.num_inputs;
        s.continuous_values()(ctx.children[right_right][1]) = -1.5;

        expect({I, C, O, I, I, C, O, O, O}, {Idx(0), Val(3.2), Add, Idx(2), Idx(1), Val(-1.5), Div, Sub, Mul});
        expected_size = 9;
    }

    ctx.to_gpu_representation(s, node_type, node_value, true, size);

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

TEST_CASE("goblin::gp::ctx_gpu_repr::multi_output") {
    using namespace test;

    // Binary tree with depth 1: root + 2 children = 3 nodes per output
    auto tree = TemplateNode::full_nary(2, 1);
    Template tmplate;
    tmplate.add_output(tree);
    tmplate.add_output(tree);

    std::vector<std::shared_ptr<OperatorBase>> operators = {
        std::make_shared<OpSubGPU>(),
        std::make_shared<OpAdd>(),
    };

    GPContext ctx(
        /* num_inputs = */ 2, tmplate, operators, /* num_parameters = */ 0,
        /* constant_representation = */ "ercs", /* constant_pool_size= */ 10, /* enable_subfunctions= */ false);

    auto f = MOFitness(/* num_objectives= */ 2);
    Solution s(f.worst(), Vec<DType>::Zero(ctx.num_discrete), Vec<CType>::Zero(ctx.num_continuous));

    auto out0_root = ctx.output_roots[0];
    auto out0_left = ctx.children[out0_root][0];
    auto out0_right = ctx.children[out0_root][1];

    auto out1_root = ctx.output_roots[1];
    auto out1_left = ctx.children[out1_root][0];
    auto out1_right = ctx.children[out1_root][1];

    std::vector<u8> node_type;
    std::vector<float> node_value;
    usize size = 0;

    std::vector<u8> expected_node_type;
    std::vector<float> expected_node_value;
    usize expected_size;

    // Appends one output's worth of expected data, padded to max_expression_size
    auto expect_output = [&](std::vector<u8> types, std::vector<float> values) {
        types.resize(ctx.max_expression_size, std::numeric_limits<u8>::max());
        values.resize(ctx.max_expression_size, std::numeric_limits<float>::max());
        expected_node_type.insert(expected_node_type.end(), types.begin(), types.end());
        expected_node_value.insert(expected_node_value.end(), values.begin(), values.end());
    };

    SUBCASE("x0 + x1 and x1 - x0") {
        // Output 0: x0 + x1
        s.discrete_values()(out0_root) = ctx.op_idx2value[1];  // +
        s.discrete_values()(out0_left) = 0;                    // x0
        s.discrete_values()(out0_right) = 1;                   // x1

        // Output 1: x1 - x0
        s.discrete_values()(out1_root) = ctx.op_idx2value[0];  // -
        s.discrete_values()(out1_left) = 1;                    // x1
        s.discrete_values()(out1_right) = 0;                   // x0

        expect_output({I, I, O}, {Idx(0), Idx(1), Add});
        expect_output({I, I, O}, {Idx(1), Idx(0), Sub});
        expected_size = 6;
    }

    SUBCASE("leaf output followed by operator output") {
        // Output 0: just x0 at root (children are inactive)
        s.discrete_values()(out0_root) = 0;  // x0

        // Output 1: x0 + x1
        s.discrete_values()(out1_root) = ctx.op_idx2value[1];  // +
        s.discrete_values()(out1_left) = 0;                    // x0
        s.discrete_values()(out1_right) = 1;                   // x1

        expect_output({I}, {Idx(0)});
        expect_output({I, I, O}, {Idx(0), Idx(1), Add});
        expected_size = 4;
    }

    ctx.to_gpu_representation(s, node_type, node_value, true, size);

    CHECK_EQ(node_type.size(), expected_node_type.size());
    CHECK_EQ(node_value.size(), expected_node_value.size());
    CHECK_EQ(node_type.size(), node_value.size());
    CHECK_EQ(size, expected_size);

    for (usize i = 0; i < 2 * ctx.max_expression_size; i++) {
        INFO("Index: ", i);
        INFO("Type: ", node_type[i], "\t|\t Expected: ", expected_node_type[i]);
        INFO("Value: ", node_value[i], "\t|\t Expected: ", expected_node_value[i]);

        CHECK_EQ(node_type[i], expected_node_type[i]);
        CHECK_EQ(node_value[i], expected_node_value[i]);
    }
}

TEST_CASE("goblin::gp::ctx_gpu_repr::subfunctions") {
    using namespace test;

    // Binary tree of depth 2 (3 layers)
    auto tree = TemplateNode::full_nary(2, 2);
    Template tmplate;
    tmplate.add_subtree(tree);
    tmplate.add_subtree(tree);
    tmplate.add_output(tree);

    std::vector<std::shared_ptr<OperatorBase>> operators = {std::make_shared<OpSubGPU>()};

    GPContext ctx(
        /* num_inputs = */ 2, tmplate, operators, /* num_parameters = */ 0,
        /* constant_representation = */ "ercs", /* constant_pool_size= */ 10, /* enable_subfunctions= */ true);

    auto f = MOFitness(1);
    Solution s(f.worst(), Vec<DType>::Zero(ctx.num_discrete), Vec<CType>::Zero(ctx.num_continuous));

    auto sub0_root = ctx.subtree_roots[0];
    auto sub0_left = ctx.children[sub0_root][0];
    auto sub0_right = ctx.children[sub0_root][1];

    auto sub1_root = ctx.subtree_roots[1];
    auto sub1_left = ctx.children[sub1_root][0];
    auto sub1_right = ctx.children[sub1_root][1];

    auto out_root = ctx.output_roots[0];
    auto out_left = ctx.children[out_root][0];
    auto out_right = ctx.children[out_root][1];

    std::vector<u8> node_type;
    std::vector<float> node_value;
    usize size = 0;

    std::vector<u8> expected_node_type;
    std::vector<float> expected_node_value;
    usize expected_size;

    auto expect = [&](std::vector<u8> types, std::vector<float> values) {
        types.resize(ctx.max_expression_size, std::numeric_limits<u8>::max());
        values.resize(ctx.max_expression_size, std::numeric_limits<float>::max());
        expected_node_type = std::move(types);
        expected_node_value = std::move(values);
    };

    REQUIRE(node_type.empty());
    REQUIRE(node_value.empty());
    REQUIRE(expected_node_type.empty());
    REQUIRE(expected_node_value.empty());
    REQUIRE_EQ(size, 0);

    SUBCASE("x1 - x0") {
        // 0 -> x0
        // 1 -> x1
        // 2 -> constant
        // 3 -> arg[0] / fn[0] / op[0]

        // Arg[0] - Arg[1]
        s.discrete_values()(sub0_root) = ctx.value2domain(sub0_root, ctx.op_idx2value[0]).value_or(1000);  // -
        s.discrete_values()(sub0_left) = 3;                                                                // Arg[0]
        s.discrete_values()(sub0_right) = 4;                                                               // Arg[1]

        // Fn[0](Arg[1], Arg[0])
        s.discrete_values()(sub1_root) = 5;   // Fn[0]
        s.discrete_values()(sub1_left) = 4;   // Arg[1]
        s.discrete_values()(sub1_right) = 3;  // Arg[0]

        // Fn[1](x0, x1) = ... = x1 - x0
        s.discrete_values()(out_root) = 4;   // Fn[1]
        s.discrete_values()(out_left) = 0;   // x0
        s.discrete_values()(out_right) = 1;  // x1

        expect({I, I, O}, {Idx(1), Idx(0), Sub});
        expected_size = 3;
    }

    SUBCASE("nested calls") {
        // Ouput = Fn[1](Fn[1](x0, x1), Fn[1](x0, x0)) = (x0 - x1) - (x1 - x0)
        // Fn[1] = Fn[0](Arg[0], Arg[1]) = Fn[0]
        // Fn[0] = Arg[0] - Arg[1]

        // Fn[0]
        s.discrete_values()(sub0_root) = ctx.value2domain(sub0_root, ctx.op_idx2value[0]).value_or(1000);  // -
        s.discrete_values()(sub0_left) = 3;                                                                // Arg[0]
        s.discrete_values()(sub0_right) = 4;                                                               // Arg[1]

        // Fn[1]
        s.discrete_values()(sub1_root) = 5;   // Fn[0]
        s.discrete_values()(sub1_left) = 3;   // Arg[0]
        s.discrete_values()(sub1_right) = 4;  // Arg[1]

        // Output
        s.discrete_values()(out_root) = 4;                    // Fn[1]
        s.discrete_values()(out_left) = 4;                    // Fn[1]
        s.discrete_values()(ctx.children[out_left][0]) = 0;   // x0
        s.discrete_values()(ctx.children[out_left][1]) = 1;   // x1
        s.discrete_values()(out_right) = 4;                   // Fn[1]
        s.discrete_values()(ctx.children[out_right][0]) = 1;  // x1
        s.discrete_values()(ctx.children[out_right][1]) = 0;  // x0

        expect({I, I, O, I, I, O, O}, {Idx(0), Idx(1), Sub, Idx(1), Idx(0), Sub, Sub});
        expected_size = 5;
    }

    ctx.to_gpu_representation(s, node_type, node_value, true, size);

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

TEST_CASE("goblin::gp::ctx_gpu_repr::subfunctions_multi_output") {
    using namespace test;

    // Binary tree of depth 2 (3 layers), 1 subtree, 2 outputs
    auto tree = TemplateNode::full_nary(2, 2);
    Template tmplate;
    tmplate.add_subtree(tree);
    tmplate.add_output(tree);
    tmplate.add_output(tree);

    std::vector<std::shared_ptr<OperatorBase>> operators = {std::make_shared<OpSubGPU>()};

    GPContext ctx(
        /* num_inputs = */ 2, tmplate, operators, /* num_parameters = */ 0,
        /* constant_representation = */ "ercs", /* constant_pool_size= */ 10, /* enable_subfunctions= */ true);

    auto f = MOFitness(/* num_objectives= */ 2);
    Solution s(f.worst(), Vec<DType>::Zero(ctx.num_discrete), Vec<CType>::Zero(ctx.num_continuous));

    auto sub_root = ctx.subtree_roots[0];
    auto sub_left = ctx.children[sub_root][0];
    auto sub_right = ctx.children[sub_root][1];

    auto out0_root = ctx.output_roots[0];
    auto out0_left = ctx.children[out0_root][0];
    auto out0_right = ctx.children[out0_root][1];

    auto out1_root = ctx.output_roots[1];
    auto out1_left = ctx.children[out1_root][0];
    auto out1_right = ctx.children[out1_root][1];

    std::vector<u8> node_type;
    std::vector<float> node_value;
    usize size = 0;

    std::vector<u8> expected_node_type;
    std::vector<float> expected_node_value;
    usize expected_size;

    auto expect_output = [&](std::vector<u8> types, std::vector<float> values) {
        types.resize(ctx.max_expression_size, std::numeric_limits<u8>::max());
        values.resize(ctx.max_expression_size, std::numeric_limits<float>::max());
        expected_node_type.insert(expected_node_type.end(), types.begin(), types.end());
        expected_node_value.insert(expected_node_value.end(), values.begin(), values.end());
    };

    REQUIRE(node_type.empty());
    REQUIRE(node_value.empty());
    REQUIRE(expected_node_type.empty());
    REQUIRE(expected_node_value.empty());
    REQUIRE_EQ(size, 0);

    // Fn[0] = Arg[0] - Arg[1] (shared setup for all subcases)
    // Subtree root domain (fn[0] excluded as cyclic): 0=x0, 1=x1, 2=const, 3=arg[0], 4=arg[1], 5=Sub
    s.discrete_values()(sub_root) = ctx.value2domain(sub_root, ctx.op_idx2value[0]).value_or(1000);  // -
    s.discrete_values()(sub_left) = 3;                                                               // Arg[0]
    s.discrete_values()(sub_right) = 4;                                                              // Arg[1]

    // Output root domain (args excluded): 0=x0, 1=x1, 2=const, 3=fn[0], 4=Sub

    SUBCASE("same subfunction, but with swapped arguments") {
        // Output 0: fn[0](x0, x1) -> ... -> x0 - x1
        s.discrete_values()(out0_root) = 3;   // Fn[0]
        s.discrete_values()(out0_left) = 0;   // x0  (= arg[0])
        s.discrete_values()(out0_right) = 1;  // x1  (= arg[1])

        // Output 1: fn[0](x1, x0) -> ... -> x1 - x0
        s.discrete_values()(out1_root) = 3;   // Fn[0]
        s.discrete_values()(out1_left) = 1;   // x1  (= arg[0])
        s.discrete_values()(out1_right) = 0;  // x0  (= arg[1])

        expect_output({I, I, O}, {Idx(0), Idx(1), Sub});
        expect_output({I, I, O}, {Idx(1), Idx(0), Sub});
        expected_size = 5;
    }

    SUBCASE("subfunction in one output, direct operator in other") {
        // Output 0: x0 - x1 without subfunction; call stack must stay clean for output 1
        s.discrete_values()(out0_root) = 4;   // Sub (direct)
        s.discrete_values()(out0_left) = 0;   // x0
        s.discrete_values()(out0_right) = 1;  // x1

        // Output 1: fn[0](x1, x0) -> x1 - x0
        s.discrete_values()(out1_root) = 3;   // Fn[0]
        s.discrete_values()(out1_left) = 1;   // x1  (= arg[0])
        s.discrete_values()(out1_right) = 0;  // x0  (= arg[1])

        expect_output({I, I, O}, {Idx(0), Idx(1), Sub});
        expect_output({I, I, O}, {Idx(1), Idx(0), Sub});
        expected_size = 6;
    }

    ctx.to_gpu_representation(s, node_type, node_value, true, size);

    CHECK_EQ(node_type.size(), expected_node_type.size());
    CHECK_EQ(node_value.size(), expected_node_value.size());
    CHECK_EQ(node_type.size(), node_value.size());
    CHECK_EQ(size, expected_size);

    for (usize i = 0; i < 2 * ctx.max_expression_size; i++) {
        INFO("Index: ", i);
        INFO("Type: ", node_type[i], "\t|\t Expected: ", expected_node_type[i]);
        INFO("Value: ", node_value[i], "\t|\t Expected: ", expected_node_value[i]);

        CHECK_EQ(node_type[i], expected_node_type[i]);
        CHECK_EQ(node_value[i], expected_node_value[i]);
    }
}