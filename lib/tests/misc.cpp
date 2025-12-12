#include "doctest/doctest.h"

#include <ranges>
#include <vector>

#include "goblin/ga-gp/misc.h"
#include "goblin/ga-gp/types.h"

using namespace goblin;



TEST_CASE("goblin::ga-gp::misc::kernel_dim_determine") {
    struct TestCase {
        int count;
        KernelDim expected;
    };

    std::vector<TestCase> test_cases = {
        { 32, KernelDim{32} }, 
        { 31, KernelDim{32} },
        { 63, KernelDim{64} },
        { 33, KernelDim{64} },
        { 512, KernelDim{512} },
        { 513, KernelDim{544} },
        { 511, KernelDim{512} },
        { 1024, KernelDim{1024} },
        { 1025, KernelDim{352} },
        { 1023, KernelDim{1024} },
        { 2048, KernelDim{1024} },
        { 2049, KernelDim{416} }, 
        { 2047, KernelDim{1024} },
        { 100000, KernelDim{800} },
    };

    for (auto&& [i, tc] : std::views::enumerate(std::as_const(test_cases))) {
        INFO("Test case index: ", i);

        KernelDim result = KernelDim::determine(tc.count);

        CHECK_EQ(result, tc.expected);
    }
}

TEST_CASE("goblin::ga-gp::misc::kernel_config_for_eval") {
    struct TestCase {
        int num_solutions;
        int num_datapoints;
        KernelConfig expected;
    };

    std::vector<TestCase> test_cases = {
        { 1, 1, KernelConfig{ KernelDim{32}, KernelDim{1} } },
        { 32, 1, KernelConfig{ KernelDim{32}, KernelDim{32} } },
        { 1, 32, KernelConfig{ KernelDim{32}, KernelDim{1} } },
        { 32, 32, KernelConfig{ KernelDim{32}, KernelDim{32} } },
        { 1, 1025, KernelConfig{ KernelDim{352}, KernelDim{1, 3} } },
        { 32, 1025, KernelConfig{ KernelDim{352}, KernelDim{32, 3} } },
        { 2048, 100000, KernelConfig{ KernelDim{800}, KernelDim{2048, 125} } },
    };

    for (auto&& [i, tc] : std::views::enumerate(std::as_const(test_cases))) {
        INFO("Test case index: ", i);

        KernelConfig result = KernelConfig::for_eval(tc.num_solutions, tc.num_datapoints);

        INFO("Block X: ", result.block.x, " | ", tc.expected.block.x);
        INFO("Block Y: ", result.block.y, " | ", tc.expected.block.y);
        INFO("Block Z: ", result.block.z, " | ", tc.expected.block.z);

        INFO("Grid X: ", result.grid.x, " | ", tc.expected.grid.x);
        INFO("Grid Y: ", result.grid.y, " | ", tc.expected.grid.y);
        INFO("Grid Z: ", result.grid.z, " | ", tc.expected.grid.z);

        CHECK_EQ(result, tc.expected);
    }
}

TEST_CASE("goblin::ga-gp::misc::kernel_config_for_mse") {
    struct TestCase {
        int num_solutions;
        int num_partial;
        KernelVersion kernel_version;
        KernelConfig expected;
    };

    std::vector<TestCase> test_cases = {
        { 1, 1, KernelVersion::Baseline, KernelConfig{ KernelDim{32}, KernelDim{1} } },
        { 32, 1, KernelVersion::Baseline, KernelConfig{ KernelDim{32}, KernelDim{1} } },
        { 1, 32, KernelVersion::Baseline, KernelConfig{ KernelDim{32}, KernelDim{1} } },
        { 32, 32, KernelVersion::Baseline, KernelConfig{ KernelDim{32}, KernelDim{1} } },
        { 1, 1025, KernelVersion::Baseline, KernelConfig{ KernelDim{32}, KernelDim{1} } },
        { 32, 1025, KernelVersion::Baseline, KernelConfig{ KernelDim{32}, KernelDim{1} } },
        { 2048, 100000, KernelVersion::Baseline, KernelConfig{ KernelDim{1024}, KernelDim{2} } },
        { 1, 1, KernelVersion::BlockReduce, KernelConfig{ KernelDim{32}, KernelDim{1} } },
        { 32, 1, KernelVersion::BlockReduce, KernelConfig{ KernelDim{32}, KernelDim{32} } },
        { 1, 32, KernelVersion::BlockReduce, KernelConfig{ KernelDim{32}, KernelDim{1} } },
        { 32, 32, KernelVersion::BlockReduce, KernelConfig{ KernelDim{32}, KernelDim{32} } },
        { 1, 1024, KernelVersion::BlockReduce, KernelConfig{ KernelDim{1024}, KernelDim{1} } },
        { 32, 1024, KernelVersion::BlockReduce, KernelConfig{ KernelDim{1024}, KernelDim{32} } },
        { 2048, 1024, KernelVersion::BlockReduce, KernelConfig{ KernelDim{1024}, KernelDim{2048} } },
    };

    for (auto&& [i, tc] : std::views::enumerate(std::as_const(test_cases))) {
        INFO("Test case index: ", i);

        KernelConfig result = KernelConfig::for_mse(tc.num_solutions, tc.num_partial, tc.kernel_version);
        
        INFO("Block X: ", result.block.x, " | ", tc.expected.block.x);
        INFO("Block Y: ", result.block.y, " | ", tc.expected.block.y);
        INFO("Block Z: ", result.block.z, " | ", tc.expected.block.z);

        INFO("Grid X: ", result.grid.x, " | ", tc.expected.grid.x);
        INFO("Grid Y: ", result.grid.y, " | ", tc.expected.grid.y);
        INFO("Grid Z: ", result.grid.z, " | ", tc.expected.grid.z);

        CHECK_EQ(result, tc.expected);
    }
}

TEST_CASE("goblin::ga-gp::misc::launch_config_determine") {
    struct TestCase {
        int num_solutions;
        int num_datapoints;
        KernelVersion kernel_version;
        LaunchConfig expected;
    };

    std::vector<TestCase> test_cases = {
        { 1, 1, KernelVersion::Baseline, LaunchConfig{ 
            KernelConfig{KernelDim{32}, KernelDim{1}}, 
            KernelConfig{KernelDim{32}, KernelDim{1}},
            KernelVersion::Baseline
        } },
        { 32, 1, KernelVersion::Baseline, LaunchConfig{ 
            KernelConfig{KernelDim{32}, KernelDim{32}}, 
            KernelConfig{KernelDim{32}, KernelDim{1}},
            KernelVersion::Baseline
        } },
        { 1, 32, KernelVersion::Baseline, LaunchConfig{ 
            KernelConfig{KernelDim{32}, KernelDim{1}}, 
            KernelConfig{KernelDim{32}, KernelDim{1}},
            KernelVersion::Baseline
        } },
        { 32, 32, KernelVersion::Baseline, LaunchConfig{ 
            KernelConfig{KernelDim{32}, KernelDim{32}}, 
            KernelConfig{KernelDim{32}, KernelDim{1}},
            KernelVersion::Baseline
        } },
        { 1, 1025, KernelVersion::Baseline, LaunchConfig{ 
            KernelConfig{KernelDim{352}, KernelDim{1, 3}}, 
            KernelConfig{KernelDim{32}, KernelDim{1}},
            KernelVersion::Baseline
        } },
        { 32, 1025, KernelVersion::Baseline, LaunchConfig{ 
            KernelConfig{KernelDim{352}, KernelDim{32, 3}}, 
            KernelConfig{KernelDim{32}, KernelDim{1}},
            KernelVersion::Baseline
        } },
        { 2048, 100000, KernelVersion::Baseline, LaunchConfig{ 
            KernelConfig{KernelDim{800}, KernelDim{2048, 125}}, 
            KernelConfig{KernelDim{1024}, KernelDim{2}},
            KernelVersion::Baseline
        } },
        { 1, 1, KernelVersion::BlockReduce, LaunchConfig{ 
            KernelConfig{KernelDim{32}, KernelDim{1}}, 
            KernelConfig{KernelDim{32}, KernelDim{1}},
            KernelVersion::BlockReduce
        } },
        { 32, 1, KernelVersion::BlockReduce, LaunchConfig{ 
            KernelConfig{KernelDim{32}, KernelDim{32}}, 
            KernelConfig{KernelDim{32}, KernelDim{32}},
            KernelVersion::BlockReduce
        } },
        { 1, 32, KernelVersion::BlockReduce, LaunchConfig{ 
            KernelConfig{KernelDim{32}, KernelDim{1}}, 
            KernelConfig{KernelDim{32}, KernelDim{1}},
            KernelVersion::BlockReduce
        } },
        { 32, 32, KernelVersion::BlockReduce, LaunchConfig{ 
            KernelConfig{KernelDim{32}, KernelDim{32}}, 
            KernelConfig{KernelDim{32}, KernelDim{32}},
            KernelVersion::BlockReduce
        } },
        { 1, 1025, KernelVersion::BlockReduce, LaunchConfig{ 
            KernelConfig{KernelDim{352}, KernelDim{1, 3}}, 
            KernelConfig{KernelDim{32}, KernelDim{1}},
            KernelVersion::BlockReduce
        } },
        { 32, 1025, KernelVersion::BlockReduce, LaunchConfig{ 
            KernelConfig{KernelDim{352}, KernelDim{32, 3}}, 
            KernelConfig{KernelDim{32}, KernelDim{32}},
            KernelVersion::BlockReduce
        } },
        { 1, 4096, KernelVersion::BlockReduce, LaunchConfig{
            KernelConfig{KernelDim{1024}, KernelDim{1, 4}}, 
            KernelConfig{KernelDim{32}, KernelDim{1}},
            KernelVersion::BlockReduce
        } },
        { 2048, 100000, KernelVersion::BlockReduce, LaunchConfig{ 
            KernelConfig{KernelDim{800}, KernelDim{2048, 125}}, 
            KernelConfig{KernelDim{128}, KernelDim{2048}},
            KernelVersion::BlockReduce
        } },
    };

    for (auto&& [i, tc] : std::views::enumerate(std::as_const(test_cases))) {
        INFO("Test case index: ", i);

        LaunchConfig result = LaunchConfig::determine(tc.num_solutions, tc.num_datapoints, tc.kernel_version);
        
        INFO("EVAL");
        INFO("Block X: ", result.eval.block.x, " | ", tc.expected.eval.block.x);
        INFO("Block Y: ", result.eval.block.y, " | ", tc.expected.eval.block.y);
        INFO("Block Z: ", result.eval.block.z, " | ", tc.expected.eval.block.z);

        INFO("Grid X: ", result.eval.grid.x, " | ", tc.expected.eval.grid.x);
        INFO("Grid Y: ", result.eval.grid.y, " | ", tc.expected.eval.grid.y);
        INFO("Grid Z: ", result.eval.grid.z, " | ", tc.expected.eval.grid.z);

        INFO("MSE");
        INFO("Block X: ", result.mse.block.x, " | ", tc.expected.mse.block.x);
        INFO("Block Y: ", result.mse.block.y, " | ", tc.expected.mse.block.y);
        INFO("Block Z: ", result.mse.block.z, " | ", tc.expected.mse.block.z);

        INFO("Grid X: ", result.mse.grid.x, " | ", tc.expected.mse.grid.x);
        INFO("Grid Y: ", result.mse.grid.y, " | ", tc.expected.mse.grid.y);
        INFO("Grid Z: ", result.mse.grid.z, " | ", tc.expected.mse.grid.z);

        INFO("Version: ", result.kernel_version, " | ", tc.expected.kernel_version);

        CHECK_EQ(result, tc.expected);
    }
}