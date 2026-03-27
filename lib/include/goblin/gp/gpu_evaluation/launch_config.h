#pragma once
#ifndef _GOBLIN_GA_GP_LAUNCH_CONFIG_H
#define _GOBLIN_GA_GP_LAUNCH_CONFIG_H

#include <algorithm>
#include <cassert>

#include "goblin/gp/gpu_evaluation/misc.h"
#include "goblin/gp/gpu_evaluation/types.h"
#include "goblin/lib/assert.h"

#define MAX_THREADS_PER_BLOCK 1024
#define WARP_SIZE 32

namespace goblin {

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
    KernelDim(size_t _x, size_t _y = 1, size_t _z = 1) : x(_x), y(_y), z(_z) {}

    // Finds the thread count in [WARP_SIZE, MAX_THREADS_PER_BLOCK] (step WARP_SIZE)
    // that minimises idle threads when covering `count` items.
    static KernelDim determine(size_t count) {
        KernelDim dim{WARP_SIZE};
        size_t min_redundant = MAX_THREADS_PER_BLOCK;

        for (size_t threads = MAX_THREADS_PER_BLOCK; threads > 0; threads -= WARP_SIZE) {
            // Round up division to determine number of blocks needed
            size_t blocks_needed = ceil_div(count, threads);
            size_t redundant = (blocks_needed * threads) - count;

            if (redundant < min_redundant) {
                min_redundant = redundant;
                dim.x = threads;
            }

            // Early exit if perfect fit is found
            if (redundant == 0) {
                break;
            }
        }

        return dim;
    }

    void check() const { assert(x * y * z <= MAX_THREADS_PER_BLOCK); }

    constexpr bool operator==(const KernelDim& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

struct KernelConfig {
    KernelDim grid;
    KernelDim block;

    KernelConfig() = default;
    KernelConfig(KernelDim _grid, KernelDim _block) : grid(_grid), block(_block) {}

    // One block per solution; threads cover datapoints. Used by Baseline/Restrict/SharedMemory/BlockReduce.
    static KernelConfig for_eval(size_t num_solutions, size_t num_datapoints) {
        KernelConfig config;
        config.block   = KernelDim::determine(num_datapoints);
        config.grid.x  = num_solutions;
        config.grid.y  = ceil_div(num_datapoints, config.block.x);
        return config;
    }

    // One block per solution; threads cover all datapoints in a single pass. Used by SingleKernel variants.
    static KernelConfig for_eval_single(size_t num_solutions, size_t num_datapoints) {
        KernelConfig config;
        config.grid.x  = num_solutions;
        config.block.x = std::min((size_t)MAX_THREADS_PER_BLOCK, round_up(num_datapoints, WARP_SIZE));
        return config;
    }

    // Multiple blocks per solution; blocks split the datapoints. Used by Hybrid.
    static KernelConfig for_eval_hybrid(size_t num_solutions, size_t num_datapoints, size_t blocks_per_individual) {
        const size_t datapoints_per_block = ceil_div(num_datapoints, blocks_per_individual);
        KernelConfig config;
        config.grid.x  = num_solutions;
        config.grid.y  = blocks_per_individual;
        config.block.x = std::min((size_t)MAX_THREADS_PER_BLOCK, round_up(datapoints_per_block, WARP_SIZE));
        return config;
    }

    // One thread per solution for the MSE reduction. Used by Baseline/Restrict/SharedMemory.
    static KernelConfig for_mse_simple(size_t num_solutions) {
        KernelConfig config;
        config.block  = KernelDim::determine(num_solutions);
        config.grid.x = ceil_div(num_solutions, config.block.x);
        return config;
    }

    // One block per solution for the MSE reduction. Used by BlockReduce and Hybrid.
    static KernelConfig for_mse_block(size_t num_solutions, size_t num_partial) {
        KernelConfig config;
        config.grid.x  = num_solutions;
        config.block.x = std::min((size_t)MAX_THREADS_PER_BLOCK, round_up(num_partial, WARP_SIZE));
        return config;
    }

    void check() const { block.check(); }

    constexpr bool operator==(const KernelConfig& other) const {
        return grid == other.grid && block == other.block;
    }
};

struct LaunchConfig {
    KernelConfig eval;
    KernelConfig mse;
    KernelVersion kernel_version    = KernelVersion::Baseline;
    size_t num_solutions            = 0;
    size_t num_datapoints           = 0;
    size_t solution_length          = 0;
    size_t blocks_per_individual    = 1;
    size_t datapoints_per_block     = 0;
    size_t datapoints_per_thread    = 0;

    LaunchConfig() = default;

    LaunchConfig(
        KernelConfig eval,
        KernelConfig mse,
        KernelVersion version = KernelVersion::Baseline
    ) : eval(eval),
        mse(mse),
        kernel_version(version) {}

    static LaunchConfig determine(
        KernelVersion kernel_version,
        size_t num_solutions,
        size_t num_datapoints,
        size_t solution_length,
        std::optional<size_t> num_sms
    ) {
        LaunchConfig config;
        config.kernel_version  = kernel_version;
        config.num_solutions   = num_solutions;
        config.num_datapoints  = num_datapoints;
        config.solution_length = solution_length;

        switch (kernel_version) {
            case KernelVersion::Baseline:
            case KernelVersion::Restrict:
            case KernelVersion::SharedMemory:
                config.eval = KernelConfig::for_eval(num_solutions, num_datapoints);
                config.mse  = KernelConfig::for_mse_simple(num_solutions);
                break;

            case KernelVersion::BlockReduce:
                config.eval = KernelConfig::for_eval(num_solutions, num_datapoints);
                config.mse  = KernelConfig::for_mse_block(num_solutions, config.eval.grid.y);
                break;

            case KernelVersion::SingleKernel:
            case KernelVersion::SingleKernelFMAF:
            case KernelVersion::SingleKernelInplace:
                config.eval = KernelConfig::for_eval_single(num_solutions, num_datapoints);
                config.datapoints_per_thread = ceil_div(num_datapoints, config.eval.block.x);
                break;

            case KernelVersion::Hybrid:
                __goblin_runtime_assert(num_sms.has_value());

                if (num_solutions * 2 <= num_sms.value()) {
                    config.blocks_per_individual = num_sms.value() / num_solutions;
                    config.eval = KernelConfig::for_eval_hybrid(
                        num_solutions, num_datapoints, config.blocks_per_individual);
                    config.datapoints_per_block  = ceil_div(num_datapoints, config.blocks_per_individual);
                    config.datapoints_per_thread = ceil_div(config.datapoints_per_block, config.eval.block.x);
                    config.mse = KernelConfig::for_mse_block(num_solutions, config.blocks_per_individual);
                } else {
                    // Not enough SMs to benefit from multi-block; fall back to single-block.
                    return LaunchConfig::determine(
                        KernelVersion::SingleKernelInplace, num_solutions, num_datapoints, solution_length, num_sms);
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
