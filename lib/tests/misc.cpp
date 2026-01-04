#include "doctest/doctest.h"

#include <ranges>
#include <vector>

#include "goblin/ga-gp/misc.h"
#include "goblin/ga-gp/types.h"

using namespace goblin;



TEST_CASE("goblin::ga-gp::misc::kernel_dim_determine") {
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
    };


    for (auto&& [i, tc] : std::views::enumerate(std::as_const(test_cases))) {
        INFO("Test case index: ", i);

        KernelDim result = KernelDim::determine(tc.count);

        CHECK_EQ(result, tc.expected);
    }
}

TEST_CASE("goblin::ga-gp::misc::kernel_config_for_eval") {
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
        { 1, 1025, KernelConfig(KernelDim(1, 3), KernelDim(352)) },
        { 32, 1025, KernelConfig(KernelDim(32, 3), KernelDim(352)) },
        { 2048, 100000, KernelConfig(KernelDim(2048, 125), KernelDim(800)) },
        { 4096, 1000000, KernelConfig(KernelDim(4096, 1250), KernelDim(800)) }
    };


    for (auto&& [i, tc] : std::views::enumerate(std::as_const(test_cases))) {
        INFO("Test case index: ", i);

        KernelConfig result = KernelConfig::for_eval(tc.num_solutions, tc.num_datapoints);

        INFO("Grid X: ", result.grid.x, " | ", tc.expected.grid.x);
        INFO("Grid Y: ", result.grid.y, " | ", tc.expected.grid.y);
        INFO("Grid Z: ", result.grid.z, " | ", tc.expected.grid.z);

        INFO("Block X: ", result.block.x, " | ", tc.expected.block.x);
        INFO("Block Y: ", result.block.y, " | ", tc.expected.block.y);
        INFO("Block Z: ", result.block.z, " | ", tc.expected.block.z);

        CHECK_EQ(result, tc.expected);
    }
}

TEST_CASE("goblin::ga-gp::misc::kernel_config_for_mse") {
    struct TestCase {
        size_t num_solutions;
        size_t num_partial;
        KernelVersion kernel_version;
        KernelConfig expected;
    };

    std::vector<TestCase> test_cases = {
        { 1, 1, KernelVersion::Baseline, KernelConfig(KernelDim(1), KernelDim(32)) },
        { 32, 1, KernelVersion::Baseline, KernelConfig(KernelDim(1), KernelDim(32)) },
        { 1, 32, KernelVersion::Baseline, KernelConfig(KernelDim(1), KernelDim(32)) },
        { 32, 32, KernelVersion::Baseline, KernelConfig(KernelDim(1), KernelDim(32)) },
        { 1, 1025, KernelVersion::Baseline, KernelConfig(KernelDim(1), KernelDim(32)) },
        { 32, 1025, KernelVersion::Baseline, KernelConfig(KernelDim(1), KernelDim(32)) },
        { 2048, 100000, KernelVersion::Baseline, KernelConfig(KernelDim(2), KernelDim(1024)) },
        { 1, 1, KernelVersion::BlockReduce, KernelConfig(KernelDim(1), KernelDim(32)) },
        { 32, 1, KernelVersion::BlockReduce, KernelConfig(KernelDim(32), KernelDim(32)) },
        { 1, 32, KernelVersion::BlockReduce, KernelConfig(KernelDim(1), KernelDim(32)) },
        { 32, 32, KernelVersion::BlockReduce, KernelConfig(KernelDim(32), KernelDim(32)) },
        { 1, 1024, KernelVersion::BlockReduce, KernelConfig(KernelDim(1), KernelDim(1024)) },
        { 32, 1024, KernelVersion::BlockReduce, KernelConfig(KernelDim(32), KernelDim(1024)) },
        { 2048, 1024, KernelVersion::BlockReduce, KernelConfig(KernelDim(2048), KernelDim(1024)) },
    };


    for (auto&& [i, tc] : std::views::enumerate(std::as_const(test_cases))) {
        INFO("Test case index: ", i);

        KernelConfig result = KernelConfig::for_mse(tc.num_solutions, tc.num_partial, tc.kernel_version);
        
        INFO("Grid X: ", result.grid.x, " | ", tc.expected.grid.x);
        INFO("Grid Y: ", result.grid.y, " | ", tc.expected.grid.y);
        INFO("Grid Z: ", result.grid.z, " | ", tc.expected.grid.z);

        INFO("Block X: ", result.block.x, " | ", tc.expected.block.x);
        INFO("Block Y: ", result.block.y, " | ", tc.expected.block.y);
        INFO("Block Z: ", result.block.z, " | ", tc.expected.block.z);

        CHECK_EQ(result, tc.expected);
    }
}

TEST_CASE("goblin::ga-gp::misc::kernel_config_for_single") {
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

        KernelConfig result = KernelConfig::for_single(tc.num_solutions, tc.num_datapoints);

        INFO("Grid X: ", result.grid.x, " | ", tc.expected.grid.x);
        INFO("Grid Y: ", result.grid.y, " | ", tc.expected.grid.y);
        INFO("Grid Z: ", result.grid.z, " | ", tc.expected.grid.z);
        
        INFO("Block X: ", result.block.x, " | ", tc.expected.block.x);
        INFO("Block Y: ", result.block.y, " | ", tc.expected.block.y);
        INFO("Block Z: ", result.block.z, " | ", tc.expected.block.z);

        CHECK_EQ(result, tc.expected);
    }
}

TEST_CASE("goblin::ga-gp::misc::launch_config_determine") {
    struct TestCase {
        size_t num_solutions;
        size_t num_datapoints;
        KernelVersion kernel_version;
        LaunchConfig expected;
    };

    std::vector<TestCase> test_cases = {
        { 1, 1, KernelVersion::Baseline, LaunchConfig(
            KernelConfig(KernelDim(1), KernelDim(32)),
            KernelConfig(KernelDim(1), KernelDim(32)),
            KernelVersion::Baseline
        ) },
        { 32, 1, KernelVersion::Baseline, LaunchConfig(
            KernelConfig(KernelDim(32), KernelDim(32)),
            KernelConfig(KernelDim(1), KernelDim(32)),
            KernelVersion::Baseline
        ) },
        { 1, 32, KernelVersion::Baseline, LaunchConfig(
            KernelConfig(KernelDim(1), KernelDim(32)),
            KernelConfig(KernelDim(1), KernelDim(32)),
            KernelVersion::Baseline
        ) },
        { 32, 32, KernelVersion::Baseline, LaunchConfig(
            KernelConfig(KernelDim(32), KernelDim(32)),
            KernelConfig(KernelDim(1), KernelDim(32)),
            KernelVersion::Baseline
        ) },
        { 1, 1025, KernelVersion::Baseline, LaunchConfig(
            KernelConfig(KernelDim(1, 3), KernelDim(352)),
            KernelConfig(KernelDim(1), KernelDim(32)),
            KernelVersion::Baseline
        ) },
        { 32, 1025, KernelVersion::Baseline, LaunchConfig(
            KernelConfig(KernelDim(32, 3), KernelDim(352)),
            KernelConfig(KernelDim(1), KernelDim(32)),
            KernelVersion::Baseline
        ) },
        { 2048, 100000, KernelVersion::Baseline, LaunchConfig(
            KernelConfig(KernelDim(2048, 125), KernelDim(800)),
            KernelConfig(KernelDim(2), KernelDim(1024)),
            KernelVersion::Baseline
        ) },
        { 1, 1, KernelVersion::BlockReduce, LaunchConfig(
            KernelConfig(KernelDim(1), KernelDim(32)),
            KernelConfig(KernelDim(1), KernelDim(32)),
            KernelVersion::BlockReduce
        ) },
        { 32, 1, KernelVersion::BlockReduce, LaunchConfig(
            KernelConfig(KernelDim(32), KernelDim(32)),
            KernelConfig(KernelDim(32), KernelDim(32)),
            KernelVersion::BlockReduce
        ) },
        { 1, 32, KernelVersion::BlockReduce, LaunchConfig(
            KernelConfig(KernelDim(1), KernelDim(32)),
            KernelConfig(KernelDim(1), KernelDim(32)),
            KernelVersion::BlockReduce
        ) },
        { 32, 32, KernelVersion::BlockReduce, LaunchConfig(
            KernelConfig(KernelDim(32), KernelDim(32)),
            KernelConfig(KernelDim(32), KernelDim(32)),
            KernelVersion::BlockReduce
        ) },
        { 1, 1025, KernelVersion::BlockReduce, LaunchConfig(
            KernelConfig(KernelDim(1, 3), KernelDim(352)),
            KernelConfig(KernelDim(1), KernelDim(32)),
            KernelVersion::BlockReduce
        ) },
        { 32, 1025, KernelVersion::BlockReduce, LaunchConfig(
            KernelConfig(KernelDim(32, 3), KernelDim(352)),
            KernelConfig(KernelDim(32), KernelDim(32)),
            KernelVersion::BlockReduce
        ) },
        { 1, 4096, KernelVersion::BlockReduce, LaunchConfig(
            KernelConfig(KernelDim(1, 4), KernelDim(1024)),
            KernelConfig(KernelDim(1), KernelDim(32)),
            KernelVersion::BlockReduce
        ) },
        { 2048, 100000, KernelVersion::BlockReduce, LaunchConfig(
            KernelConfig(KernelDim(2048, 125), KernelDim(800)),
            KernelConfig(KernelDim(2048), KernelDim(128)),
            KernelVersion::BlockReduce
        ) },
        { 1, 1, KernelVersion::SingleKernel, LaunchConfig(
            KernelConfig(KernelDim(1), KernelDim(32)),
            KernelConfig(), KernelVersion::SingleKernel
        ) },
        { 32, 1, KernelVersion::SingleKernel, LaunchConfig(
            KernelConfig(KernelDim(32), KernelDim(32)),
            KernelConfig(), KernelVersion::SingleKernel
        ) },
        { 1, 32, KernelVersion::SingleKernel, LaunchConfig(
            KernelConfig(KernelDim(1), KernelDim(32)),
            KernelConfig(), KernelVersion::SingleKernel
        ) },
        { 32, 32, KernelVersion::SingleKernel, LaunchConfig(
            KernelConfig(KernelDim(32), KernelDim(32)),
            KernelConfig(), KernelVersion::SingleKernel
        ) },
        { 1, 1025, KernelVersion::SingleKernel, LaunchConfig(
            KernelConfig(KernelDim(1), KernelDim(1024)),
            KernelConfig(), KernelVersion::SingleKernel
        ) },
        { 32, 1025, KernelVersion::SingleKernel, LaunchConfig(
            KernelConfig(KernelDim(32), KernelDim(1024)),
            KernelConfig(), KernelVersion::SingleKernel
        ) },
        { 1, 4096, KernelVersion::SingleKernel, LaunchConfig(
            KernelConfig(KernelDim(1), KernelDim(1024)),
            KernelConfig(), KernelVersion::SingleKernel
        ) },
        { 2048, 100000, KernelVersion::SingleKernel, LaunchConfig(
            KernelConfig(KernelDim(2048), KernelDim(1024)),
            KernelConfig(), KernelVersion::SingleKernel
        ) },
    };


    for (auto&& [i, tc] : std::views::enumerate(std::as_const(test_cases))) {
        INFO("Test case index: ", i);

        LaunchConfig result = LaunchConfig::determine(tc.kernel_version, tc.num_solutions, tc.num_datapoints, 0);
        
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