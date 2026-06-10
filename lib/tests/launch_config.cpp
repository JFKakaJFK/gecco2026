#include "doctest/doctest.h"

#include <optional>
#include <ranges>
#include <vector>
#include <utility>

#include "goblin/gp/gpu_evaluation/launch_config.h"
#include "goblin/gp/gpu_evaluation/types.h"

using namespace goblin;



TEST_CASE("goblin::gp::gpu_evaluation::misc::kernel_dim::determine") {
    struct TestCase {
        size_t count;
        KernelDim expected;
    };

    std::vector<TestCase> test_cases = {
        { 32, KernelDim(32) }, 
        { 31, KernelDim(32) },
        { 63, KernelDim(64) },
        { 33, KernelDim(64) },
        { 512, KernelDim(512) },
        { 513, KernelDim(544) },
        { 511, KernelDim(512) },
        { 1024, KernelDim(1024) },
        { 1025, KernelDim(352) },
        { 1023, KernelDim(1024) },
        { 2048, KernelDim(1024) },
        { 2049, KernelDim(416) }, 
        { 2047, KernelDim(1024) },
        { 100000, KernelDim(800) },
        {73500, KernelDim(32)},
    };


    for (auto&& [i, tc] : std::views::enumerate(std::as_const(test_cases))) {
        INFO("Test case index: ", i);

        KernelDim result = KernelDim::determine(tc.count);

        CHECK_EQ(result.x, tc.expected.x);
        CHECK_EQ(result.y, tc.expected.y);
        CHECK_EQ(result.z, tc.expected.z);
    }
}

TEST_CASE("goblin::gp::gpu_evaluation::misc::kernel_config::single_block") {
    struct TestCase {
        size_t num_solutions;
        size_t num_datapoints;
        KernelConfig expected;
    };

    std::vector<TestCase> test_cases = {
        { 1, 1, KernelConfig(KernelDim(1), KernelDim(32)) },
        { 32, 1, KernelConfig(KernelDim(32), KernelDim(32)) },
        { 1, 32, KernelConfig(KernelDim(1), KernelDim(32)) },
        { 32, 32, KernelConfig(KernelDim(32), KernelDim(32)) },
        { 1, 1025, KernelConfig(KernelDim(1), KernelDim(1024)) },
        { 32, 1025, KernelConfig(KernelDim(32), KernelDim(1024)) },
        { 2048, 100000, KernelConfig(KernelDim(2048), KernelDim(1024)) },
    };


    for (auto&& [i, tc] : std::views::enumerate(std::as_const(test_cases))) {
        INFO("Test case index: ", i);

        KernelConfig result = KernelConfig::single_block(tc.num_solutions, tc.num_datapoints);

        INFO("Grid X: ", result.grid.x, " | ", tc.expected.grid.x);
        INFO("Grid Y: ", result.grid.y, " | ", tc.expected.grid.y);
        INFO("Grid Z: ", result.grid.z, " | ", tc.expected.grid.z);
        
        INFO("Block X: ", result.block.x, " | ", tc.expected.block.x);
        INFO("Block Y: ", result.block.y, " | ", tc.expected.block.y);
        INFO("Block Z: ", result.block.z, " | ", tc.expected.block.z);

        CHECK_EQ(result, tc.expected);
    }
}

TEST_CASE("goblin::gp::gpu_evaluation::misc::kernel_config::dynamic_block_reduction") {
    struct TestCase {
        size_t num_solutions;
        size_t num_partial;
        KernelConfig expected;
    };

    std::vector<TestCase> test_cases = {
        { 1, 1, KernelConfig(KernelDim(1), KernelDim(32)) },
        { 32, 1, KernelConfig(KernelDim(32), KernelDim(32)) },
        { 1, 32, KernelConfig(KernelDim(1), KernelDim(32)) },
        { 32, 32, KernelConfig(KernelDim(32), KernelDim(32)) },
        { 1, 1024, KernelConfig(KernelDim(1), KernelDim(1024)) },
        { 32, 1024, KernelConfig(KernelDim(32), KernelDim(1024)) },
        { 2048, 1024, KernelConfig(KernelDim(2048), KernelDim(1024)) },
    };


    for (auto&& [i, tc] : std::views::enumerate(std::as_const(test_cases))) {
        INFO("Test case index: ", i);

        KernelConfig result = KernelConfig::dynamic_block_reduction(tc.num_solutions, tc.num_partial);

        INFO("Grid X: ", result.grid.x, " | ", tc.expected.grid.x);
        INFO("Grid Y: ", result.grid.y, " | ", tc.expected.grid.y);
        INFO("Grid Z: ", result.grid.z, " | ", tc.expected.grid.z);

        INFO("Block X: ", result.block.x, " | ", tc.expected.block.x);
        INFO("Block Y: ", result.block.y, " | ", tc.expected.block.y);
        INFO("Block Z: ", result.block.z, " | ", tc.expected.block.z);

        CHECK_EQ(result, tc.expected);
    }
}

TEST_CASE("goblin::gp::gpu_evaluation::misc::launch_config::determine") {
    struct TestCase {
        size_t num_solutions;
        size_t num_datapoints;
        KernelVersion kernel_version;
        std::optional<size_t> num_sms;
        LaunchConfig expected;
    };

    std::vector<TestCase> test_cases = {
        { 1, 1, KernelVersion::SingleBlock, std::nullopt, LaunchConfig(
            KernelConfig(KernelDim(1), KernelDim(32)),
            KernelConfig(), KernelVersion::SingleBlock
        ) },
        { 32, 1, KernelVersion::SingleBlock, std::nullopt, LaunchConfig(
            KernelConfig(KernelDim(32), KernelDim(32)),
            KernelConfig(), KernelVersion::SingleBlock
        ) },
        { 1, 32, KernelVersion::SingleBlock, std::nullopt, LaunchConfig(
            KernelConfig(KernelDim(1), KernelDim(32)),
            KernelConfig(), KernelVersion::SingleBlock
        ) },
        { 32, 32, KernelVersion::SingleBlock, std::nullopt, LaunchConfig(
            KernelConfig(KernelDim(32), KernelDim(32)),
            KernelConfig(), KernelVersion::SingleBlock
        ) },
        { 1, 1025, KernelVersion::SingleBlock, std::nullopt, LaunchConfig(
            KernelConfig(KernelDim(1), KernelDim(1024)),
            KernelConfig(), KernelVersion::SingleBlock
        ) },
        { 32, 1025, KernelVersion::SingleBlock, std::nullopt, LaunchConfig(
            KernelConfig(KernelDim(32), KernelDim(1024)),
            KernelConfig(), KernelVersion::SingleBlock
        ) },
        { 1, 4096, KernelVersion::SingleBlock, std::nullopt, LaunchConfig(
            KernelConfig(KernelDim(1), KernelDim(1024)),
            KernelConfig(), KernelVersion::SingleBlock
        ) },
        { 2048, 100000, KernelVersion::SingleBlock, std::nullopt, LaunchConfig(
            KernelConfig(KernelDim(2048), KernelDim(1024)),
            KernelConfig(), KernelVersion::SingleBlock
        ) },
        { 1, 1, KernelVersion::DynamicBlock, 1, LaunchConfig(
            KernelConfig(KernelDim(1), KernelDim(32)),
            KernelConfig(), KernelVersion::SingleBlock
        ) },
        { 32, 1, KernelVersion::DynamicBlock, 32, LaunchConfig(
            KernelConfig(KernelDim(32), KernelDim(32)),
            KernelConfig(), KernelVersion::SingleBlock
        ) },
        { 32, 1, KernelVersion::DynamicBlock, 64, LaunchConfig(
            KernelConfig(KernelDim(32, 2), KernelDim(32)),
            KernelConfig(KernelDim(32), KernelDim(32)), KernelVersion::DynamicBlock
        ) },
        { 1, 32, KernelVersion::DynamicBlock, 1, LaunchConfig(
            KernelConfig(KernelDim(1), KernelDim(32)),
            KernelConfig(), KernelVersion::SingleBlock
        ) },
        { 32, 32, KernelVersion::DynamicBlock, 64, LaunchConfig(
            KernelConfig(KernelDim(32, 2), KernelDim(32)),
            KernelConfig(KernelDim(32), KernelDim(32)), KernelVersion::DynamicBlock
        ) },
        { 1, 1025, KernelVersion::DynamicBlock, 1, LaunchConfig(
            KernelConfig(KernelDim(1), KernelDim(1024)),
            KernelConfig(), KernelVersion::SingleBlock
        ) },
        { 32, 1025, KernelVersion::DynamicBlock, 128, LaunchConfig(
            KernelConfig(KernelDim(32, 4), KernelDim(288)),
            KernelConfig(KernelDim(32), KernelDim(32)), KernelVersion::DynamicBlock
        ) },
        { 1, 4096, KernelVersion::DynamicBlock, 120, LaunchConfig(
            KernelConfig(KernelDim(1, 120), KernelDim(64)),
            KernelConfig(KernelDim(1), KernelDim(128)), KernelVersion::DynamicBlock
        ) },
        { 2048, 100000, KernelVersion::DynamicBlock, 100, LaunchConfig(
            KernelConfig(KernelDim(2048), KernelDim(1024)),
            KernelConfig(), KernelVersion::SingleBlock
        ) },
    };


    for (auto&& [i, tc] : std::views::enumerate(std::as_const(test_cases))) {
        INFO("Test case index: ", i);

        LaunchConfig result = LaunchConfig::determine(tc.kernel_version, tc.num_solutions, tc.num_datapoints, 0, tc.num_sms);
        
        INFO("EVAL");

        INFO("Grid X: ", result.eval.grid.x, " | ", tc.expected.eval.grid.x);
        INFO("Grid Y: ", result.eval.grid.y, " | ", tc.expected.eval.grid.y);
        INFO("Grid Z: ", result.eval.grid.z, " | ", tc.expected.eval.grid.z);

        INFO("Block X: ", result.eval.block.x, " | ", tc.expected.eval.block.x);
        INFO("Block Y: ", result.eval.block.y, " | ", tc.expected.eval.block.y);
        INFO("Block Z: ", result.eval.block.z, " | ", tc.expected.eval.block.z);

        INFO("MSE");

        INFO("Grid X: ", result.mse.grid.x, " | ", tc.expected.mse.grid.x);
        INFO("Grid Y: ", result.mse.grid.y, " | ", tc.expected.mse.grid.y);
        INFO("Grid Z: ", result.mse.grid.z, " | ", tc.expected.mse.grid.z);

        INFO("Block X: ", result.mse.block.x, " | ", tc.expected.mse.block.x);
        INFO("Block Y: ", result.mse.block.y, " | ", tc.expected.mse.block.y);
        INFO("Block Z: ", result.mse.block.z, " | ", tc.expected.mse.block.z);

        INFO("Version: ", result.kernel_version, " | ", tc.expected.kernel_version);

        CHECK_EQ(result, tc.expected);
    }
}