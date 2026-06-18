#pragma once
#ifndef _GOBLIN_GA_GP_LAUNCH_CONFIG_H
#define _GOBLIN_GA_GP_LAUNCH_CONFIG_H

#include <algorithm>
#include <cassert>
#include <optional>

#include "goblin/gp/gpu_evaluation/types.h"
#include "goblin/lib/assert.h"

namespace goblin {

static constexpr size_t MAX_THREADS_PER_BLOCK = 1024;
static constexpr size_t WARP_SIZE = 32;

constexpr size_t round_up(size_t value, size_t multiple) {
    return ((value + multiple - 1) / multiple) * multiple;
}

constexpr size_t ceil_div(size_t a, size_t b) {
    return (a + b - 1) / b;
}

struct KernelDim {
    size_t x = 1;
    size_t y = 1;
    size_t z = 1;

    KernelDim() = default;
    KernelDim(size_t x, size_t y = 1, size_t z = 1) : x(x), y(y), z(z) {}

    // Finds the thread count in [WARP_SIZE, MAX_THREADS_PER_BLOCK] (step WARP_SIZE)
    // that minimizes idle threads when covering `count` items.
    static KernelDim determine(size_t count) {
        KernelDim best{WARP_SIZE};
        size_t min_redundant = MAX_THREADS_PER_BLOCK;

        for (size_t threads = MAX_THREADS_PER_BLOCK; threads > 0; threads -= WARP_SIZE) {
            size_t blocks_needed = ceil_div(count, threads);
            size_t redundant = (blocks_needed * threads) - count;

            if (redundant < min_redundant) {
                min_redundant = redundant;
                best.x = threads;
            }

            // Early exit if perfect fit is found
            if (redundant == 0) {
                break;
            }
        }

        return best;
    }

    void check() const { assert(x * y * z <= MAX_THREADS_PER_BLOCK); }

    constexpr bool operator==(const KernelDim& other) const { return x == other.x && y == other.y && z == other.z; }
};

struct KernelConfig {
    KernelDim grid;
    KernelDim block;

    KernelConfig() = default;
    KernelConfig(KernelDim grid, KernelDim block) : grid(grid), block(block) {}

    // One block per solution; threads cover all datapoints in a single pass.
    static KernelConfig single_block(size_t num_solutions, size_t num_datapoints) {
        KernelConfig config;
        config.grid.x = num_solutions;
        config.block.x = std::min(MAX_THREADS_PER_BLOCK, round_up(num_datapoints, WARP_SIZE));
        return config;
    }

    // Multiple blocks per solution; blocks partition the datapoints.
    static KernelConfig dynamic_block_evaluation(size_t num_solutions,
                                                 size_t num_datapoints,
                                                 size_t blocks_per_individual) {
        const size_t datapoints_per_block = ceil_div(num_datapoints, blocks_per_individual);
        KernelConfig config;
        config.grid.x = num_solutions;
        config.grid.y = blocks_per_individual;
        config.block.x = std::min(MAX_THREADS_PER_BLOCK, round_up(datapoints_per_block, WARP_SIZE));
        return config;
    }

    // One block per solution for the MSE reduction.
    static KernelConfig dynamic_block_reduction(size_t num_solutions, size_t num_partial) {
        KernelConfig config;
        config.grid.x = num_solutions;
        config.block.x = std::min(MAX_THREADS_PER_BLOCK, round_up(num_partial, WARP_SIZE));
        return config;
    }

    void check() const { block.check(); }

    constexpr bool operator==(const KernelConfig& other) const { return grid == other.grid && block == other.block; }
};

struct LaunchConfig {
    KernelConfig eval;
    KernelConfig mse;  // DynamicBlock only: reduction pass
    KernelVersion kernel_version = KernelVersion::SingleBlock;
    size_t num_solutions = 0;
    size_t num_datapoints = 0;
    size_t solution_length = 0;
    size_t blocks_per_individual = 1;  // DynamicBlock only: blocks assigned per solution
    size_t datapoints_per_block = 0;   // DynamicBlock only: datapoints handled per block
    size_t datapoints_per_thread = 0;  // datapoints each thread evaluates in its loop

    LaunchConfig() = default;

    LaunchConfig(KernelConfig eval, KernelConfig mse, KernelVersion version = KernelVersion::SingleBlock)
        : eval(eval), mse(mse), kernel_version(version) {}

    static LaunchConfig determine(KernelVersion kernel_version,
                                  size_t num_solutions,
                                  size_t num_datapoints,
                                  size_t solution_length,
                                  std::optional<size_t> num_sms) {
        LaunchConfig config;
        config.kernel_version = kernel_version;
        config.num_solutions = num_solutions;
        config.num_datapoints = num_datapoints;
        config.solution_length = solution_length;

        switch (kernel_version) {
            case KernelVersion::SingleBlock:
                config.eval = KernelConfig::single_block(num_solutions, num_datapoints);
                config.datapoints_per_thread = ceil_div(num_datapoints, config.eval.block.x);
                break;

            case KernelVersion::DynamicBlock:
                __goblin_runtime_assert(num_sms.has_value());

                if (num_solutions * 2 <= num_sms.value()) {
                    config.blocks_per_individual = num_sms.value() / num_solutions;
                    config.eval = KernelConfig::dynamic_block_evaluation(num_solutions, num_datapoints,
                                                                         config.blocks_per_individual);
                    config.datapoints_per_block = ceil_div(num_datapoints, config.blocks_per_individual);
                    config.datapoints_per_thread = ceil_div(config.datapoints_per_block, config.eval.block.x);
                    config.mse = KernelConfig::dynamic_block_reduction(num_solutions, config.blocks_per_individual);
                } else {
                    // Not enough SMs to benefit from multi-block; fall back to single-block.
                    return LaunchConfig::determine(KernelVersion::SingleBlock, num_solutions, num_datapoints,
                                                   solution_length, num_sms);
                }
                break;

            default:
                break;
        }

        return config;
    }

    void check() const {
        eval.check();
        mse.check();
    }

    constexpr bool operator==(const LaunchConfig& other) const {
        return eval == other.eval && mse == other.mse && kernel_version == other.kernel_version;
    }
};

}  // namespace goblin

#endif /* _GOBLIN_GA_GP_LAUNCH_CONFIG_H */
