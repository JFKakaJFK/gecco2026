#include "doctest/doctest.h"

#include <ranges>
#include <vector>

#include "goblin/ga-gp/evaluate.h"
#include "goblin/ga-gp/types.h"

using namespace goblin;

TEST_CASE("goblin::ga-gp::eval_kernel::compute_tree_output") {
    struct TestCase {
        std::vector<float> X;
        std::vector<NodeType> type;
        std::vector<float> value;
        int num_datapoints;
        int datapoint_index;
        float expected;
    };

    // Instantiation of test cases
    std::vector<TestCase> test_cases = {
        //////////////////////////////////////////
        /// SINGLE SOLUTION | SINGLE DATAPOINT ///
        ////////////////////////////////////////// 

        // constant at root
        { {0}, {C}, {Val(10)}, 1, 0, 10 },

        // input at root
        { {1, 2, 3, 4}, {I}, {Idx(0)}, 1, 0, 1 }, // x0
        { {1, 2, 3, 4}, {I}, {Idx(1)}, 1, 0, 2 }, // x1
        { {1, 2, 3, 4}, {I}, {Idx(2)}, 1, 0, 3 }, // x2
        { {1, 2, 3, 4}, {I}, {Idx(3)}, 1, 0, 4 }, // x3

        // operator at root
        { {2}, {C, I, O}, {Val(1), Idx(0), Add}, 1, 0, 3 }, // x0 + 1
        { {9}, {C, I, O}, {Val(2), Idx(0), Sub}, 1, 0, 7 }, // x0 - 2
        { {2}, {C, I, O}, {Val(4), Idx(0), Mul}, 1, 0, 8 }, // x0 * 4
        { {8}, {C, I, O}, {Val(4), Idx(0), Div}, 1, 0, 2 }, // x0 / 4

        // nested operators
        { {1}, {I, C, O, C, O}, {Idx(0), Val(5), Sub, Val(3), Add}, 1, 0, 7 }, // 3 + (5 - x0)
        { {2, 3}, {I, I, O, I, C, O, O}, {Idx(0), Idx(1), Sub, Idx(0), Val(10), Div, Mul}, 1, 0, 5 }, // (10 / x0) * (x1 - x0)
        
        ////////////////////////////////////////////
        /// SINGLE SOLUTION | MULTIPLE DATAPOINT ///
        ////////////////////////////////////////////
        
        // input at root
        { {3, 2, 1, 0}, {I}, {Idx(0)}, 4, 0, 3 },
        { {3, 2, 1, 0}, {I}, {Idx(0)}, 4, 1, 2 },
        { {3, 2, 1, 0}, {I}, {Idx(0)}, 4, 2, 1 },
        { {3, 2, 1, 0}, {I}, {Idx(0)}, 4, 3, 0 },
        
        // operator at root
        { {0, 1, 2, 3}, {I, I, O}, {Idx(1), Idx(0), Add}, 2, 0, {2} }, // x0 + x1
        { {0, 1, 2, 3}, {I, I, O}, {Idx(1), Idx(0), Add}, 2, 1, {4} }, // x0 + x1
        { {0, 1, 2, 3}, {I, I, O}, {Idx(1), Idx(0), Sub}, 2, 0, {-2} }, // x0 - x1
        { {0, 1, 2, 3}, {I, I, O}, {Idx(1), Idx(0), Sub}, 2, 1, {-2} }, // x0 - x1
        { {0, 1, 2, 3}, {I, I, O}, {Idx(1), Idx(0), Mul}, 2, 0, {0} }, // x0 * x1
        { {0, 1, 2, 3}, {I, I, O}, {Idx(1), Idx(0), Mul}, 2, 1, {3} }, // x0 * x1
        { {0, 1, 2, 3}, {I, I, O}, {Idx(1), Idx(0), Div}, 2, 0, {0} }, // x0 / x1
        { {0, 1, 2, 3}, {I, I, O}, {Idx(1), Idx(0), Div}, 2, 1, {1.0/3.0} }, // x0 / x1
        
        // nested operators
        { {1, 2, 3, 6, 12, 18}, {C, I, O, I, O}, {Val(4), Idx(0), Mul, Idx(1), Sub}, 3, 0, {2} }, // x1 - (x0 * 4)
        { {1, 2, 3, 6, 12, 18}, {C, I, O, I, O}, {Val(4), Idx(0), Mul, Idx(1), Sub}, 3, 1, {4} }, // x1 - (x0 * 4)
        { {1, 2, 3, 6, 12, 18}, {C, I, O, I, O}, {Val(4), Idx(0), Mul, Idx(1), Sub}, 3, 2, {6} }, // x1 - (x0 * 4)

        { {2, 3, 5, 3, 6, 9}, {C, I, O, I, O}, {Val(3), Idx(1), Div, Idx(0), Mul}, 3, 0, {2} }, // x0 * (x1 / c1)
        { {2, 3, 5, 3, 6, 9}, {C, I, O, I, O}, {Val(3), Idx(1), Div, Idx(0), Mul}, 3, 1, {6} }, // x0 * (x1 / c1)
        { {2, 3, 5, 3, 6, 9}, {C, I, O, I, O}, {Val(3), Idx(1), Div, Idx(0), Mul}, 3, 2, {15} }, // x0 * (x1 / c1)

        { {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {I, C, O, I, I, O, O}, {Idx(0), Val(10), Sub, Idx(2), Idx(1), Mul, Add}, 4, 0, {54} }, // (x1 * x2) + (10 - x0)
        { {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {I, C, O, I, I, O, O}, {Idx(0), Val(10), Sub, Idx(2), Idx(1), Mul, Add}, 4, 1, {68} }, // (x1 * x2) + (10 - x0)
        { {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {I, C, O, I, I, O, O}, {Idx(0), Val(10), Sub, Idx(2), Idx(1), Mul, Add}, 4, 2, {84} }, // (x1 * x2) + (10 - x0)
        { {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {I, C, O, I, I, O, O}, {Idx(0), Val(10), Sub, Idx(2), Idx(1), Mul, Add}, 4, 3, {102} }, // (x1 * x2) + (10 - x0)
    };

    for (auto&& [i, tc] : std::views::enumerate(std::as_const(test_cases))) {
        INFO("Test case index: ", i);

        REQUIRE_EQ(tc.type.size(), tc.value.size());

        float result = test_compute_output_kernel(tc.X, tc.type, tc.value, tc.num_datapoints, tc.datapoint_index);
        
        CHECK_EQ(result, doctest::Approx(tc.expected));
    }
}

TEST_CASE("goblin::ga-gp::eval_kernel::evaluate") {
    struct TestCase {
        std::vector<float> X;
        std::vector<float> Y;
        std::vector<NodeType> type;
        std::vector<float> value;
        int num_solutions;
        int num_datapoints;
        std::vector<float> expected;
    };

    // Instantiation of test cases
    std::vector<TestCase> test_cases = {
        //////////////////////////////////////////
        /// SINGLE SOLUTION | SINGLE DATAPOINT ///
        ////////////////////////////////////////// 

        // (10 / x0) * (x1 - x0) = 5
       { {2, 3}, {5}, {I, I, O, I, C, O, O}, {Idx(0), Idx(1), Sub, Idx(0), Val(10), Div, Mul}, 1, 1, {0} }, // se = (5 - 5)**2
       { {2, 3}, {10}, {I, I, O, I, C, O, O}, {Idx(0), Idx(1), Sub, Idx(0), Val(10), Div, Mul}, 1, 1, {25} }, // se = (5 - 10)**2
       { {2, 3}, {1}, {I, I, O, I, C, O, O}, {Idx(0), Idx(1), Sub, Idx(0), Val(10), Div, Mul}, 1, 1, {16} }, // se = (5 - 1)**2
    
        ////////////////////////////////////////////
        /// SINGLE SOLUTION | MULTIPLE DATAPOINT ///
        ////////////////////////////////////////////
        
        // x0
        { {3, 2, 1, 0}, {2, 2, 2, 2}, {I}, {Idx(0)}, 1, 4, {1, 0, 1, 4} },

        // x0 + x1
        { {0, 1, 2, 3}, {4, 4}, {I, I, O}, {Idx(1), Idx(0), Add}, 1, 2, {4, 0} }, // x0 + x1

        // (x1 * x2) + (10 - x0), tree outputs = {54, 68, 84, 102}
        { {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {50, 70, 90, 110}, {I, C, O, I, I, O, O}, {Idx(0), Val(10), Sub, Idx(2), Idx(1), Mul, Add}, 1, 4, {16, 4, 36, 64} }, 

        ////////////////////////////////////////////
        /// MULTIPLE SOLUTION | SINGLE DATAPOINT ///
        ////////////////////////////////////////////

        // c0 = 2, 4, 6
        { {0}, {3}, {C, C, C}, {Val(2), Val(4), Val(6)}, 3, 1, {1, 1, 9} },

        // x0 = 1, 2, 3
        { {1, 2, 3}, {3}, {I, I, I}, {Idx(0), Idx(1), Idx(2)}, 3, 1, {4, 1, 0} },

        // x0 op x1, op = +, -, *, /
        // tree outputs = {6, 2, 8, 2}
        { {4, 2}, {6}, {I, I, O, I, I, O, I, I, O, I, I, O}, {Idx(1), Idx(0), Add, Idx(1), Idx(0), Sub, Idx(1), Idx(0), Mul, Idx(1), Idx(0), Div}, 4, 1, {0, 16, 4, 16} }, // x0 op x1, op = +, -, *, /

        //////////////////////////////////////////////
        /// MULTIPLE SOLUTION | MULTIPLE DATAPOINT ///
        //////////////////////////////////////////////

        // s0 = x0, s1 = x1
        { {1, 2, 3, 4}, {2, 0},  {I, I}, {Idx(0), Idx(1)}, 2, 2, {1, 4, 1, 16} }, 

        // s0 = c0 + x0, s1 = x0 - c1
        { {2, 7}, {5, 6}, {I, C, O, C, I, O}, {Idx(0), Val(4), Add, Val(2), Idx(0), Sub}, 2, 2, {1, 25, 25, 1} }, 

        // s0 = c0 + (x1 * (x2 / c1)), s1 = ((x0 * x2) - c1) / c0, s2 = x0 / ((x2 + c0) * c1)
        { 
            {1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12}, // 3 inputs, 4 datapoints 
            {5, 10, 20, 35},
            {
                C, I, O, I, O, C, O, // Solution 1
                C, C, I, I, O, O, O, // Solution 2
                C, C, I, O, O, I, O, // Solution 3
            },
            {
                Val(-3.14f), Idx(2), Div, Idx(1), Mul, Val(1.2f), Add, // Solution 1
                Val(5), Val(-4.5f), Idx(2), Idx(0), Mul, Sub, Div, // Solution 2
                Val(0.25f), Val(0.1f), Idx(2), Add, Mul, Idx(0), Div, // Solution 3
            },
            3, 4, 
            {
                32.6136, 336.875, 1741.39, 5751.46, 
                12.2500, 18.4900, 42.2500, 102.010,
                13.7617, 54.4209, 286.390, 1004.52
            }
        }

    };

    for (auto&& [i, tc] : std::views::enumerate(std::as_const(test_cases))) {
        INFO("Test Case: ", i);

        REQUIRE_EQ(tc.type.size(), tc.value.size());
        REQUIRE_EQ(tc.Y.size(), tc.num_datapoints);
        REQUIRE_EQ(tc.expected.size(), tc.num_solutions * tc.num_datapoints);

        std::vector<float> result = test_evaluate_kernel(tc.X, tc.Y, tc.type, tc.value, tc.num_solutions, tc.num_datapoints);

        CHECK_EQ(result.size(), tc.expected.size());

        for (size_t j = 0; j < result.size(); j++) {
            INFO("Datapoint: ", j, "\tResult: ", result[j], "\tExpected: ", tc.expected[j]);
            CHECK_EQ(result[j], doctest::Approx(tc.expected[j]));
        }
    }
}

TEST_CASE("goblin::ga-gp::eval_kernel::compute_mse") {
    struct TestCase {
        std::vector<float> se;
        int num_solutions;
        int num_datapoints;
        std::vector<float> expected;
    };

    std::vector<TestCase> test_cases = {
        //////////////////////////////////////////
        /// SINGLE SOLUTION | SINGLE DATAPOINT ///
        ////////////////////////////////////////// 

        { {5.189f}, 1, 1, {5.189f} },

        ////////////////////////////////////////////
        /// SINGLE SOLUTION | MULTIPLE DATAPOINT ///
        ////////////////////////////////////////////

        { {1, 1, 3, 3}, 1, 4, {2} },
        { {5.21f, 13.62f, 128.175f}, 1, 3, {49.0017} },

        ////////////////////////////////////////////
        /// MULTIPLE SOLUTION | SINGLE DATAPOINT ///
        ////////////////////////////////////////////

        { {1, 2, 3}, 3, 1, {1, 2, 3} },
        { {5.21f, 13.62f, 128.175f}, 3, 1, {5.21f, 13.62f, 128.175f} },

        //////////////////////////////////////////////
        /// MULTIPLE SOLUTION | MULTIPLE DATAPOINT ///
        //////////////////////////////////////////////

        { {1, 2, 5, 10}, 2, 2, {1.5f, 7.5f} },
        { {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, 3, 4, {2.5, 6.5, 10.5} },
    };

    for (auto&& [i, tc] : std::views::enumerate(std::as_const(test_cases))) {
        INFO("Test Case: ", i);

        REQUIRE_EQ(tc.expected.size(), tc.num_solutions);

        std::vector<float> result = test_compute_mse_kernel(tc.se, tc.num_solutions, tc.num_datapoints);

        CHECK_EQ(result.size(), tc.expected.size());

        for (size_t j = 0; j < result.size(); j++) {
            INFO("Solution: ", j, "\tResult: ", result[j], "\tExpected: ", tc.expected[j]);
            CHECK_EQ(result[j], doctest::Approx(tc.expected[j]));
        }
    }

}