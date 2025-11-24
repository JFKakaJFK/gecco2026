#include "doctest/doctest.h"

#include <ranges>
#include <vector>

#include "goblin/ga-gp/eval_kernel.h"
#include "goblin/ga-gp/types.h"

using namespace goblin;

constexpr NodeType C = NodeType::Constant;
constexpr NodeType I = NodeType::Input;
constexpr NodeType O = NodeType::Operator;

constexpr float Val(int x) { return static_cast<float>(x); }
constexpr float Idx(int idx) { return static_cast<float>(idx); }
constexpr float Op(Operator op) { return static_cast<float>(op); }

constexpr float Add = Op(Operator::Add);
constexpr float Sub = Op(Operator::Sub);
constexpr float Mul = Op(Operator::Mul);
constexpr float Div = Op(Operator::Div);

struct TestCase {
    std::vector<float> X;
    std::vector<NodeType> type;
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
        { {0}, {C}, {Val(10)}, 1, 1, {10} },

        // input at root
        { {1, 2, 3, 4}, {I}, {Idx(0)}, 1, 1, {1} }, // x0
        { {1, 2, 3, 4}, {I}, {Idx(1)}, 1, 1, {2} }, // x1
        { {1, 2, 3, 4}, {I}, {Idx(2)}, 1, 1, {3} }, // x2
        { {1, 2, 3, 4}, {I}, {Idx(3)}, 1, 1, {4} }, // x3

        // operator at root
        { {2}, {C, I, O}, {Val(1), Idx(0), Add}, 1, 1, {3} }, // x0 + 1
        { {9}, {C, I, O}, {Val(2), Idx(0), Sub}, 1, 1, {7} }, // x0 - 2
        { {2}, {C, I, O}, {Val(4), Idx(0), Mul}, 1, 1, {8} }, // x0 * 4
        { {8}, {C, I, O}, {Val(4), Idx(0), Div}, 1, 1, {2} }, // x0 / 4

        // nested operators
        { {1}, {I, C, O, C, O}, {Idx(0), Val(5), Sub, Val(3), Add}, 1, 1, {7} }, // 3 + (5 - x0)
        { {2, 3}, {I, I, O, I, C, O, O}, {Idx(0), Idx(1), Sub, Idx(0), Val(10), Div, Mul}, 1, 1, {5} }, // (10 / x0) * (x1 - x0), x0 = 2, x1 = 3
        ////////////////////////////////////////////
        /// SINGLE SOLUTION | MULTIPLE DATAPOINT ///
        ////////////////////////////////////////////
        
        // input at root
        { {0, 1, 2, 3}, {I}, {Idx(0)}, 1, 4, {0, 1, 2, 3} },
        
        // operator at root
        { {0, 1, 2, 3}, {I, C, O}, {Idx(0), Val(2), Add}, 1, 4, {2, 3, 4, 5} }, //  2 + x0
        { {0, 1, 2, 3}, {I, I, O}, {Idx(1), Idx(0), Add}, 1, 2, {2, 4} }, // x0 + x1
        { {0, 1, 2, 3}, {I, C, O}, {Idx(0), Val(2), Sub}, 1, 4, {2, 1, 0, -1} }, //  2 - x0
        { {0, 1, 2, 3}, {I, I, O}, {Idx(1), Idx(0), Sub}, 1, 2, {-2, -2} }, // x0 - x1
        { {0, 1, 2, 3}, {I, C, O}, {Idx(0), Val(2), Mul}, 1, 4, {0, 2, 4, 6} }, //  2 * x0
        { {0, 1, 2, 3}, {I, I, O}, {Idx(1), Idx(0), Mul}, 1, 2, {0, 3} }, // x0 * x1
        { {0, 1, 2, 3}, {I, C, O}, {Idx(0), Val(2), Div}, 1, 4, {0, 2, 1, 2.0/3.0} }, //  2 / x0
        { {0, 1, 2, 3}, {I, I, O}, {Idx(1), Idx(0), Div}, 1, 2, {0, 1.0/3.0} }, // x0 / x1
        
        // nested operators
        { {1, 2, 3, 6, 12, 18}, {C, I, O, I, O}, {Val(4), Idx(0), Mul, Idx(1), Sub}, 1, 3, {2, 4, 6} }, // x1 - (x0 * 4)
        { {2, 3, 5, 3, 6, 9}, {C, I, O, I, O}, {Val(3), Idx(1), Div, Idx(0), Mul}, 1, 3, {2, 6, 15} }, // x0 * (x1 / c1)
        { {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {I, C, O, I, I, O, O}, {Idx(0), Val(10), Sub, Idx(2), Idx(1), Mul, Add}, 1, 4, {54, 68, 84, 102} }, // (x1 * x2) + (10 - x0)

        ////////////////////////////////////////////
        /// MULTIPLE SOLUTION | SINGLE DATAPOINT ///
        ////////////////////////////////////////////

        // constant at root
        { {0}, {C, C, C}, {Val(2), Val(4), Val(6)}, 3, 1, {2, 4, 6} },

        // input at root
        { {1, 2, 3}, {I, I, I}, {Idx(0), Idx(1), Idx(2)}, 3, 1, {1, 2, 3} },

        // operator at root
        { {4, 2}, {I, I, O, I, I, O, I, I, O, I, I, O}, {Idx(1), Idx(0), Add, Idx(1), Idx(0), Sub, Idx(1), Idx(0), Mul, Idx(1), Idx(0), Div}, 4, 1, {6, 2, 8, 2} }, // x0 op x1, op = +, -, *, /

        //////////////////////////////////////////////
        /// MULTIPLE SOLUTION | MULTIPLE DATAPOINT ///
        //////////////////////////////////////////////

        // input at root
        { {1, 2, 3, 4}, {I, I}, {Idx(0), Idx(1)}, 2, 2, {1, 2, 3, 4} }, // s0 = x0, s1 = x1

        // operator at root
        { {2, 7}, {I, C, O, C, I, O}, {Idx(0), Val(4), Add, Val(2), Idx(0), Sub}, 2, 2, {6, 11, 0, 5} } // c0 + x0, x0 - c1
    };

    for (auto&& [i, tc] : std::views::enumerate(std::as_const(test_cases))) {
        INFO("Test case index: " << i);

        REQUIRE_EQ(tc.type.size(), tc.value.size());
        REQUIRE_EQ(tc.expected.size(), tc.num_solutions * tc.num_datapoints);

        std::vector<float> result;
        result.assign(tc.num_solutions * tc.num_datapoints, 0.0f);
        test_eval_kernel(tc.X, {}, tc.type, tc.value, tc.num_solutions, tc.num_datapoints, result);
        
        for (size_t j = 0; j < result.size(); j++) {
            INFO("Datapoint index: " << j);
            CHECK_EQ(result[j], doctest::Approx(tc.expected[j]));
        }
    }
}