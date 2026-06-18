#include "doctest/doctest.h"

#include <cmath>
#include <numbers>
#include <ranges>
#include <utility>
#include <vector>

#include "goblin/gp/gpu_evaluation/evaluate.h"
#include "goblin/gp/gpu_evaluation/types.h"

#define E2 std::pow(e_v<float>, 2.0F)

using namespace goblin;

TEST_CASE("goblin::gp::gpu-evaluation::evaluate::compute_tree_output") {
    using namespace std::numbers;
    using namespace test;

    using u8 = std::uint8_t;

    struct TestCase {
        std::vector<float> X;
        std::vector<u8> type;
        std::vector<float> value;
        size_t num_datapoints;
        size_t datapoint_index;
        float expected;
    };

    // Instantiation of test cases
    std::vector<TestCase> test_cases = {
        //////////////////////////////////////////
        /// SINGLE SOLUTION | SINGLE DATAPOINT ///
        //////////////////////////////////////////

        // constant at root #0
        {{0}, {C}, {Val(10)}, 1, 0, 10},

        // input at root #1
        {{1, 2, 3, 4}, {I}, {Idx(0)}, 1, 0, 1},  // x0
        {{1, 2, 3, 4}, {I}, {Idx(1)}, 1, 0, 2},  // x1
        {{1, 2, 3, 4}, {I}, {Idx(2)}, 1, 0, 3},  // x2
        {{1, 2, 3, 4}, {I}, {Idx(3)}, 1, 0, 4},  // x3

        // operator at root #5
        {{2}, {I, C, O}, {Idx(0), Val(1), Add}, 1, 0, 3},  // x0 + 1
        {{9}, {I, C, O}, {Idx(0), Val(2), Sub}, 1, 0, 7},  // x0 - 2
        {{2}, {I, C, O}, {Idx(0), Val(4), Mul}, 1, 0, 8},  // x0 * 4
        {{8}, {I, C, O}, {Idx(0), Val(4), Div}, 1, 0, 2},  // x0 / 4
        {{pi}, {I, O}, {Idx(0), Sin}, 1, 0, 0},            // x0 = pi, sin(x0) = 0
        {{pi}, {I, O}, {Idx(0), Cos}, 1, 0, -1},           // x0 = pi, cos(x0) = -1
        {{2}, {I, O}, {Idx(0), Exp}, 1, 0, E2},            // x0 = 2, exp(x0) = e ^ 2
        {{E2}, {I, O}, {Idx(0), Log}, 1, 0, 2},            // x0 = e ^ 2, log(x0) = 2
        {{3}, {I, O}, {Idx(0), Square}, 1, 0, 9},          // x0 = 3, x0 ^ 2 = 9
        {{9}, {I, O}, {Idx(0), Sqrt}, 1, 0, 3},            // x0 = 9, sqrt(x0) = 3
        {{2}, {I, C, O}, {Idx(0), Val(3), Pow}, 1, 0, 8},  // x0 = 2, x0 ^ 3 = 8
        {{-2}, {I, O}, {Idx(0), Abs}, 1, 0, 2},            // x0 = - 2, |x0| = 2
        {{5}, {I, C, O}, {Idx(0), Val(2), Min}, 1, 0, 2},  // x0 = 5, min(x0, 2) = 2
        {{5}, {I, C, O}, {Idx(0), Val(2), Max}, 1, 0, 5},  // x0 = 5, max(x0, 2) = 5

        // nested operators #19
        {{1}, {C, C, I, O, O}, {Val(3), Val(5), Idx(0), Sub, Add}, 1, 0, 7},                         // 3 + (5 - x0)
        {{2, 3}, {C, I, O, I, I, O, O}, {Val(10), Idx(0), Div, Idx(1), Idx(0), Sub, Mul}, 1, 0, 5},  // (10 / x0) * (x1
                                                                                                     // - x0)
        // x0 = 3, x1 = 4, x2 = 2, sin(exp(log(max(x0 - 5, sqrt(x1) * pi))) / 2) = sin(pi / 2) = 1
        {{3, 4, 4},
         {I, C, O, I, O, C, O, O, O, O, I, O, O},
         {Idx(0), Val(5), Sub, Idx(1), Sqrt, Val(pi), Mul, Max, Log, Exp, Idx(2), Div, Sin},
         1,
         0,
         1},

        ////////////////////////////////////////////
        /// SINGLE SOLUTION | MULTIPLE DATAPOINT ///
        ////////////////////////////////////////////

        // input at root #22
        {{3, 2, 1, 0}, {I}, {Idx(0)}, 4, 0, 3},
        {{3, 2, 1, 0}, {I}, {Idx(0)}, 4, 1, 2},
        {{3, 2, 1, 0}, {I}, {Idx(0)}, 4, 2, 1},
        {{3, 2, 1, 0}, {I}, {Idx(0)}, 4, 3, 0},

        // operator at root #26
        {{0, 1, 2, 3}, {I, I, O}, {Idx(0), Idx(1), Add}, 2, 0, 2},          // x0 + x1
        {{0, 1, 2, 3}, {I, I, O}, {Idx(0), Idx(1), Add}, 2, 1, 4},          // x0 + x1
        {{0, 1, 2, 3}, {I, I, O}, {Idx(0), Idx(1), Sub}, 2, 0, -2},         // x0 - x1
        {{0, 1, 2, 3}, {I, I, O}, {Idx(0), Idx(1), Sub}, 2, 1, -2},         // x0 - x1
        {{0, 1, 2, 3}, {I, I, O}, {Idx(0), Idx(1), Mul}, 2, 0, 0},          // x0 * x1
        {{0, 1, 2, 3}, {I, I, O}, {Idx(0), Idx(1), Mul}, 2, 1, 3},          // x0 * x1
        {{0, 1, 2, 3}, {I, I, O}, {Idx(0), Idx(1), Div}, 2, 0, 0},          // x0 / x1
        {{0, 1, 2, 3}, {I, I, O}, {Idx(0), Idx(1), Div}, 2, 1, 1.0 / 3.0},  // x0 / x1
        {{0, 1, 2, 3}, {I, I, O}, {Idx(0), Idx(1), Pow}, 2, 0, 0},          // x0 ^ x1
        {{0, 1, 2, 3}, {I, I, O}, {Idx(0), Idx(1), Pow}, 2, 1, 1},          // x0 ^ x1
        {{0, 1, 2, 3}, {I, I, O}, {Idx(0), Idx(1), Min}, 2, 0, 0},          // min(x0, x1)
        {{0, 1, 2, 3}, {I, I, O}, {Idx(0), Idx(1), Min}, 2, 1, 1},          // min(x0, x1)
        {{0, 1, 2, 3}, {I, I, O}, {Idx(0), Idx(1), Max}, 2, 0, 2},          // max(x0, x1)
        {{0, 1, 2, 3}, {I, I, O}, {Idx(0), Idx(1), Max}, 2, 1, 3},          // max(x0, x1)

        // nested operators #40
        {{1, 2, 3, 6, 12, 18}, {I, I, C, O, O}, {Idx(1), Idx(0), Val(4), Mul, Sub}, 3, 0, 2},  // x1 - (x0 * 4)
        {{1, 2, 3, 6, 12, 18}, {I, I, C, O, O}, {Idx(1), Idx(0), Val(4), Mul, Sub}, 3, 1, 4},  // x1 - (x0 * 4)
        {{1, 2, 3, 6, 12, 18}, {I, I, C, O, O}, {Idx(1), Idx(0), Val(4), Mul, Sub}, 3, 2, 6},  // x1 - (x0 * 4)

        {{2, 3, 5, 3, 6, 9}, {I, I, C, O, O}, {Idx(0), Idx(1), Val(3), Div, Mul}, 3, 0, 2},   // x0 * (x1 / 3)
        {{2, 3, 5, 3, 6, 9}, {I, I, C, O, O}, {Idx(0), Idx(1), Val(3), Div, Mul}, 3, 1, 6},   // x0 * (x1 / 3)
        {{2, 3, 5, 3, 6, 9}, {I, I, C, O, O}, {Idx(0), Idx(1), Val(3), Div, Mul}, 3, 2, 15},  // x0 * (x1 / 3)

        {{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
         {I, I, O, C, I, O, O},
         {Idx(1), Idx(2), Mul, Val(10), Idx(0), Sub, Add},
         4,
         0,
         54},  // (x1 * x2) + (10 - x0)
        {{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
         {I, I, O, C, I, O, O},
         {Idx(1), Idx(2), Mul, Val(10), Idx(0), Sub, Add},
         4,
         1,
         68},  // (x1 * x2) + (10 - x0)
        {{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
         {I, I, O, C, I, O, O},
         {Idx(1), Idx(2), Mul, Val(10), Idx(0), Sub, Add},
         4,
         2,
         84},  // (x1 * x2) + (10 - x0)
        {{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
         {I, I, O, C, I, O, O},
         {Idx(1), Idx(2), Mul, Val(10), Idx(0), Sub, Add},
         4,
         3,
         102},  // (x1 * x2) + (10 - x0)
    };

    for (auto&& [i, tc] : std::views::enumerate(std::as_const(test_cases))) {
        INFO("\tTest Case: ", i);

        REQUIRE_EQ(tc.type.size(), tc.value.size());

        float result = test_compute_tree_output(tc.X, tc.type, tc.value, tc.num_datapoints, tc.datapoint_index,
                                                KernelVersion::SingleBlock);

        CHECK_EQ(result, doctest::Approx(tc.expected));
    }
}

TEST_CASE("goblin::gp::gpu-evaluation::evaluate::single_block") {
    using namespace test;

    struct TestCase {
        std::vector<float> X;
        std::vector<float> Y;
        std::vector<u8> type;
        std::vector<float> value;
        size_t num_solutions;
        size_t num_datapoints;
        std::vector<float> expected;
    };

    // Instantiation of test cases
    std::vector<TestCase> test_cases = {
        //////////////////////////////////////////
        /// SINGLE SOLUTION | SINGLE DATAPOINT ///
        //////////////////////////////////////////

        // (10 / x0) * (x1 - x0) = 5
        {{2, 3}, {5}, {C, I, O, I, I, O, O}, {Val(10), Idx(0), Div, Idx(1), Idx(0), Sub, Mul}, 1, 1, {0}},  // se = (5 -
                                                                                                            // 5)**2
        {{2, 3},
         {10},
         {C, I, O, I, I, O, O},
         {Val(10), Idx(0), Div, Idx(1), Idx(0), Sub, Mul},
         1,
         1,
         {25}},  // se = (5 - 10)**2
        {{2, 3}, {1}, {C, I, O, I, I, O, O}, {Val(10), Idx(0), Div, Idx(1), Idx(0), Sub, Mul}, 1, 1, {16}},  // se = (5
                                                                                                             // - 1)**2

        ////////////////////////////////////////////
        /// SINGLE SOLUTION | MULTIPLE DATAPOINT ///
        ////////////////////////////////////////////

        // x0, mse = (1 + 0 + 1 + 4) / 4 = 1.5
        {{3, 2, 1, 0}, {2, 2, 2, 2}, {I}, {Idx(0)}, 1, 4, {1.5}},

        // x0 + x1, mse = (4 + 0) / 2 = 2
        {{0, 1, 2, 3}, {4, 4}, {I, I, O}, {Idx(0), Idx(1), Add}, 1, 2, {2}},

        // (x1 * x2) + (10 - x0), tree outputs = {54, 68, 84, 102}, mse = (16 + 4 + 36 + 64) / 4 = 30
        {{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
         {50, 70, 90, 110},
         {I, I, O, C, I, O, O},
         {Idx(1), Idx(2), Mul, Val(10), Idx(0), Sub, Add},
         1,
         4,
         {30}},

        ////////////////////////////////////////////
        /// MULTIPLE SOLUTION | SINGLE DATAPOINT ///
        ////////////////////////////////////////////

        // c0 = 2, 4, 6, mse = {1, 1, 9}
        {{0}, {3}, {C, C, C}, {Val(2), Val(4), Val(6)}, 3, 1, {1, 1, 9}},

        // x0 = 1, 2, 3, mse = {4, 1, 0}
        {{1, 2, 3}, {3}, {I, I, I}, {Idx(0), Idx(1), Idx(2)}, 3, 1, {4, 1, 0}},

        // x0 op x1, op = +, -, *, /, tree outputs = {6, 2, 8, 2}, mse = {0, 16, 4, 16}
        {{4, 2},
         {6},
         {I, I, O, I, I, O, I, I, O, I, I, O},
         {Idx(0), Idx(1), Add, Idx(0), Idx(1), Sub, Idx(0), Idx(1), Mul, Idx(0), Idx(1), Div},
         4,
         1,
         {0, 16, 4, 16}},  // x0 op x1, op = +, -, *, /

        //////////////////////////////////////////////
        /// MULTIPLE SOLUTION | MULTIPLE DATAPOINT ///
        //////////////////////////////////////////////

        // s0 = x0, s1 = x1, mse = {(1 + 4) / 2, (1 + 16) / 2} = {2.5, 8.5}
        {{1, 2, 3, 4}, {2, 0}, {I, I}, {Idx(0), Idx(1)}, 2, 2, {2.5, 8.5}},

        // s0 = c0 + x0, s1 = x0 - c1, mse = {(1 + 25) / 2, (25 + 1) / 2} = {13, 13}
        {{2, 7}, {5, 6}, {C, I, O, I, C, O}, {Val(4), Idx(0), Add, Idx(0), Val(2), Sub}, 2, 2, {13, 13}},

        // s0 = c0 + (x1 * (x2 / c1)), s1 = ((x0 * x2) - c1) / c0, s2 = x0 / ((x2 + c0) * c1)
        {{1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12},  // 3 inputs, 4 datapoints
         {5, 10, 20, 35},
         {
             C, I, I, C, O, O, O,  // Solution 1
             I, I, O, C, O, C, O,  // Solution 2
             I, I, C, O, C, O, O,  // Solution 3
         },
         {
             Val(1.2), Idx(1), Idx(2),   Val(-3.14), Div,        Mul,    Add,  // Solution 1
             Idx(0),   Idx(2), Mul,      Val(-4.5),  Sub,        Val(5), Div,  // Solution 2
             Idx(0),   Idx(2), Val(0.1), Add,        Val(0.25f), Mul,    Div,  // Solution 3
         },
         3,
         4,
         {
             1965.5847,  // (32.6136 + 336.875 + 1741.39 + 5751.46) / 4 = 7862.3386
             43.75,      // (12.2500 + 18.490 + 42.2500 + 102.010) / 4 = 43.75
             339.7731    // (13.7617 + 54.4209 + 286.390 + 1004.52) / 4 = 339.7731
         }}

    };

    for (auto&& [i, tc] : std::views::enumerate(std::as_const(test_cases))) {
        INFO("\tTest Case: ", i);

        REQUIRE_EQ(tc.type.size(), tc.value.size());
        REQUIRE_EQ(tc.Y.size(), tc.num_datapoints);
        REQUIRE_EQ(tc.expected.size(), tc.num_solutions);

        std::vector<float> result =
            test_single_block(tc.X, tc.Y, tc.type, tc.value, tc.num_solutions, tc.num_datapoints);

        CHECK_EQ(result.size(), tc.expected.size());

        for (size_t j = 0; j < result.size(); j++) {
            INFO("Datapoint: ", j, "\tResult: ", result[j], "\tExpected: ", tc.expected[j]);
            CHECK_EQ(result[j], doctest::Approx(tc.expected[j]));
        }
    }
}

TEST_CASE("goblin::gp::gpu-evaluation::evaluate::dynamic_block") {
    using namespace test;

    struct TestCase {
        std::vector<float> X;
        std::vector<float> Y;
        std::vector<u8> type;
        std::vector<float> value;
        size_t num_solutions;
        size_t num_datapoints;
        std::vector<float> expected;
    };

    std::vector<TestCase> test_cases = {
        //////////////////////////////////////////
        /// SINGLE SOLUTION | SINGLE DATAPOINT ///
        //////////////////////////////////////////

        // (10 / x0) * (x1 - x0) = 5
        {{2, 3}, {5}, {C, I, O, I, I, O, O}, {Val(10), Idx(0), Div, Idx(1), Idx(0), Sub, Mul}, 1, 1, {0}},  // mse =
                                                                                                            // (5-5)^2
        {{2, 3},
         {10},
         {C, I, O, I, I, O, O},
         {Val(10), Idx(0), Div, Idx(1), Idx(0), Sub, Mul},
         1,
         1,
         {25}},  // mse = (5-10)^2
        {{2, 3}, {1}, {C, I, O, I, I, O, O}, {Val(10), Idx(0), Div, Idx(1), Idx(0), Sub, Mul}, 1, 1, {16}},  // mse =
                                                                                                             // (5-1)^2

        ////////////////////////////////////////////
        /// SINGLE SOLUTION | MULTIPLE DATAPOINT ///
        ////////////////////////////////////////////

        // x0, mse = (1 + 0 + 1 + 4) / 4 = 1.5
        {{3, 2, 1, 0}, {2, 2, 2, 2}, {I}, {Idx(0)}, 1, 4, {1.5}},

        // x0 + x1, mse = (4 + 0) / 2 = 2
        {{0, 1, 2, 3}, {4, 4}, {I, I, O}, {Idx(0), Idx(1), Add}, 1, 2, {2}},

        // (x1 * x2) + (10 - x0), tree outputs = {54, 68, 84, 102}, mse = (16+4+36+64)/4 = 30
        {{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
         {50, 70, 90, 110},
         {I, I, O, C, I, O, O},
         {Idx(1), Idx(2), Mul, Val(10), Idx(0), Sub, Add},
         1,
         4,
         {30}},

        ////////////////////////////////////////////
        /// MULTIPLE SOLUTION | SINGLE DATAPOINT ///
        ////////////////////////////////////////////

        // c0 = 2, 4, 6
        {{0}, {3}, {C, C, C}, {Val(2), Val(4), Val(6)}, 3, 1, {1, 1, 9}},

        // x0 = 1, 2, 3
        {{1, 2, 3}, {3}, {I, I, I}, {Idx(0), Idx(1), Idx(2)}, 3, 1, {4, 1, 0}},

        // x0 op x1, op = +, -, *, /, tree outputs = {6, 2, 8, 2}, mse = {0, 16, 4, 16}
        {{4, 2},
         {6},
         {I, I, O, I, I, O, I, I, O, I, I, O},
         {Idx(0), Idx(1), Add, Idx(0), Idx(1), Sub, Idx(0), Idx(1), Mul, Idx(0), Idx(1), Div},
         4,
         1,
         {0, 16, 4, 16}},

        //////////////////////////////////////////////
        /// MULTIPLE SOLUTION | MULTIPLE DATAPOINT ///
        //////////////////////////////////////////////

        // s0 = x0, s1 = x1, mse = {2.5, 8.5}
        {{1, 2, 3, 4}, {2, 0}, {I, I}, {Idx(0), Idx(1)}, 2, 2, {2.5, 8.5}},

        // s0 = c0 + x0, s1 = x0 - c1, mse = {13, 13}
        {{2, 7}, {5, 6}, {C, I, O, I, C, O}, {Val(4), Idx(0), Add, Idx(0), Val(2), Sub}, 2, 2, {13, 13}},

        // s0 = c0 + (x1 * (x2 / c1)), s1 = ((x0 * x2) - c1) / c0, s2 = x0 / ((x2 + c0) * c1)
        {{1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12},
         {5, 10, 20, 35},
         {
             C, I, I, C, O, O, O, I, I, O, C, O, C, O, I, I, C, O, C, O, O,
         },
         {
             Val(1.2), Idx(1), Idx(2), Val(-3.14), Div,    Mul,      Add, Idx(0),     Idx(2), Mul, Val(-4.5),
             Sub,      Val(5), Div,    Idx(0),     Idx(2), Val(0.1), Add, Val(0.25f), Mul,    Div,
         },
         3,
         4,
         {1965.5847f, 43.75f, 339.7731f}},
    };

    // Test with several blocks_per_individual values to exercise the multi-block
    // splitting logic. Results must be invariant to this parameter.
    std::vector<size_t> blocks_per_individual_values = {2, 4, 8};

    for (auto&& [i, tc] : std::views::enumerate(std::as_const(test_cases))) {
        for (auto bpi : blocks_per_individual_values) {
            INFO("Test Case: ", i, "\tBlocks per individual: ", bpi);

            REQUIRE_EQ(tc.type.size(), tc.value.size());
            REQUIRE_EQ(tc.Y.size(), tc.num_datapoints);
            REQUIRE_EQ(tc.expected.size(), tc.num_solutions);

            std::vector<float> result =
                test_dynamic_block(tc.X, tc.Y, tc.type, tc.value, tc.num_solutions, tc.num_datapoints, bpi);

            CHECK_EQ(result.size(), tc.expected.size());

            for (size_t j = 0; j < result.size(); j++) {
                INFO("Solution: ", j, "\tResult: ", result[j], "\tExpected: ", tc.expected[j]);
                CHECK_EQ(result[j], doctest::Approx(tc.expected[j]));
            }
        }
    }
}