#include "doctest/doctest.h"

#include <vector>

#include "goblin/ga-gp/eval_kernel.h"

using namespace goblin;

struct TestCase {
    std::vector<float> X;
    std::vector<float> Y;
    std::vector<int> type;
    std::vector<float> value;
    int num_solutions;
    int num_datapoints;
    std::vector<float> expected;
};

TEST_CASE("goblin::ga-gp::eval_kernel") {
    // Instantiation of test cases
    std::vector<TestCase> test_cases = {
        //////////////////////////////////////////
        /// SINGLE SOLUTION | SINGLE DATAPOINT ///
        ////////////////////////////////////////// 

        // constant at root
        { {}, {}, {1}, {10}, 1, 1, {10} },

        // input at root
        { {1, 2, 3, 4}, {}, {0}, {0}, 1, 1, {1} }, // index = 0
        { {1, 2, 3, 4}, {}, {0}, {1}, 1, 1, {2} }, // index = 1
        { {1, 2, 3, 4}, {}, {0}, {2}, 1, 1, {3} }, // index = 2
        { {1, 2, 3, 4}, {}, {0}, {3}, 1, 1, {4} }, // index = 3

        // operator at root
        { {2}, {}, {2, 0, 1}, {0, 0, 1}, 1, 1, {3} }, // addition
        { {9}, {}, {2, 0, 1}, {1, 0, 2}, 1, 1, {7} }, // subtraction
        { {2}, {}, {2, 0, 1}, {2, 0, 4}, 1, 1, {8} }, // multiplication
        { {8}, {}, {2, 0, 1}, {3, 0, 4}, 1, 1, {2} }, // division

        // nested operators
        { {1}, {}, {2, 1, 2, 1, 0}, {0, 3, 1, 5, 0}, 1, 1, {7} }, // 3 + (5 - x0)
        { {2, 3}, {}, {2, 2, 1, 0, 2, 1, 1}, {2, 3, 10, 0, 1, 1, 0}, 1, 1, {5} }, // (10 / x0) * (x1 - x0), x0 = 2, x1 = 3

        ////////////////////////////////////////////
        /// SINGLE SOLUTION | MULTIPLE DATAPOINT ///
        ////////////////////////////////////////////
        
        // input at root
        { {0, 1, 2, 3}, {}, {0}, {0}, 1, 4, {0, 1, 2, 3} },
        
        // operator at root
        { {0, 1, 2, 3}, {}, {2, 1, 0}, {0, 2, 0}, 1, 4, {2, 3, 4, 5} }, //  2 + x0
        { {0, 1, 2, 3}, {}, {2, 0, 0}, {0, 0, 1}, 1, 2, {2, 4} }, // x0 + x1
        { {0, 1, 2, 3}, {}, {2, 1, 0}, {1, 2, 0}, 1, 4, {2, 1, 0, -1} }, //  2 - x0
        { {0, 1, 2, 3}, {}, {2, 0, 0}, {1, 0, 1}, 1, 2, {-2, -2} }, // x0 - x1
        { {0, 1, 2, 3}, {}, {2, 1, 0}, {2, 2, 0}, 1, 4, {0, 2, 4, 6} }, //  2 * x0
        { {0, 1, 2, 3}, {}, {2, 0, 0}, {2, 0, 1}, 1, 2, {0, 3} }, // x0 * x1
        { {0, 1, 2, 3}, {}, {2, 1, 0}, {3, 2, 0}, 1, 4, {0, 2, 1, 2.0/3.0} }, //  2 / x0
        { {0, 1, 2, 3}, {}, {2, 0, 0}, {3, 0, 1}, 1, 2, {0, 1.0/3.0} }, // x0 / x1
        
        // nested operators
        { {1, 2, 3, 6, 12, 18}, {}, {2, 0, 2, 0, 1}, {1, 1, 2, 0, 4}, 1, 3, {2, 4, 6} }, // x1 - (x0 * 4)
        { {2, 3, 5, 3, 6, 9}, {}, {2, 0, 2, 0, 1}, {2, 0, 3, 1, 3}, 1, 3, {2, 6, 15} }, // x0 * (x1 / c1)
        { {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {}, {2, 2, 0, 0, 2, 1, 0}, {0, 2, 1, 2, 1, 10, 0}, 1, 4, {54, 68, 84, 102} }, // (x1 * x2) + (10 - x0)

        ////////////////////////////////////////////
        /// MULTIPLE SOLUTION | SINGLE DATAPOINT ///
        ////////////////////////////////////////////

        // constant at root
        { {}, {}, {1, 1, 1}, {2, 4, 6}, 3, 1, {2, 4, 6} },

        // input at root
        { {1, 2, 3}, {}, {0, 0, 0}, {0, 1, 2}, 3, 1, {1, 2, 3} },

        // operator at root
        { {4, 2}, {}, {2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0}, {0, 0, 1, 1, 0, 1, 2, 0, 1, 3, 0, 1}, 4, 1, {6, 2, 8, 2} }, // x0 op x1, op = +, -, *, /

        //////////////////////////////////////////////
        /// MULTIPLE SOLUTION | MULTIPLE DATAPOINT ///
        //////////////////////////////////////////////

        // input at root
        { {1, 2, 3, 4}, {}, {0, 0}, {0, 1}, 2, 2, {1, 2, 3, 4} },

        // operator at root
        { {2, 7}, {}, {2, 1, 0, 2, 0, 1}, {0, 4, 0, 1, 0, 2}, 2, 2, {6, 11, 0, 5} } // c0 + x0, x0 - c1
    };

    for (auto& tc : test_cases) {
        REQUIRE_EQ(tc.type.size(), tc.value.size());
        REQUIRE_EQ(tc.expected.size(), tc.num_solutions * tc.num_datapoints);

        std::vector<float> result(tc.num_solutions * tc.num_datapoints);
        test_eval_kernel(tc.X, tc.Y, tc.type, tc.value, tc.num_solutions, tc.num_datapoints, result);
        
        for (size_t i = 0; i < result.size(); i++) {
            CHECK_EQ(result[i], doctest::Approx(tc.expected[i]));
        }
        
    }
}