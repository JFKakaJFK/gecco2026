#include "doctest/doctest.h"

#include <ranges>
#include <vector>

#include "goblin/ga-gp/helper.h"

using namespace goblin;

struct TestCase {
    int num_datapoints;
    int expected_block_size;
};

TEST_CASE("goblin::ga-gp::helper") {
    std::vector<TestCase> test_cases = {
        { 32, 32 }, 
        { 33, 64 },
        { 31, 32 },
        { 63, 64 },
        { 512, 512 },
        { 513, 544 },
        { 511, 512 },
        { 1024, 1024 },
        { 1025, 352 },
        { 1023, 1024 }
    };

    for (auto&& [i, tc] : std::views::enumerate(std::as_const(test_cases))) {
        INFO("Test case index: ", i);

        int result = compute_block_size(tc.num_datapoints);

        CHECK_EQ(result, tc.expected_block_size);
    }
}