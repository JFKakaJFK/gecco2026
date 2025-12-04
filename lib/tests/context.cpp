#include "doctest/doctest.h"

#include "goblin/gp/context.h"
#include "goblin/gp/sr.h"

using namespace goblin;

TEST_CASE("goblin::gp::context") {
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

    SUBCASE("x0 + x1") {
        REQUIRE(node_type.empty());
        REQUIRE(node_value.empty());

        auto root = ctx.output_roots[0];
        auto left = ctx.children[root][0];
        auto right = ctx.children[root][1];

        sset[0].discrete_values()(root) = ctx.op_idx2value[0];  // +
        sset[0].discrete_values()(left) = 0;   // x0
        sset[0].discrete_values()(right) = 1;   // x1

        expect({I, I, O}, {Idx(1), Idx(0), Add});
    }

    SUBCASE("(x0 + 3.2) * (x2 - (x1 / -1.5))") {
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
            {C, I, O, I, O, C, I, O, O},
            {Val(-1.5f), Idx(1), Div, Idx(2), Sub, Val(3.2f), Idx(0), Add, Mul}
        );
    }

    ctx.to_gpu_repr(sset[0], node_type, node_value);

    CHECK_EQ(node_type.size(), expected_node_type.size());
    CHECK_EQ(node_value.size(), expected_node_value.size());

    for (usize i = 0; i < ctx.num_discrete; i++) {
        INFO("Index: ", i, "\t|\t Type: ", node_type[i], "\t|\t Value: ", node_value[i]);
        CHECK_EQ(node_type[i], expected_node_type[i]);
        CHECK_EQ(node_value[i], expected_node_value[i]);
    }
}