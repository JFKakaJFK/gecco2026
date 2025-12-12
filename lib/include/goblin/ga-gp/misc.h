#pragma once
#ifndef _GOBLIN_GA_GP_MISC_H
#define _GOBLIN_GA_GP_MISC_H

#include "assert.h"

#include "goblin/ga-gp/types.h"

#define MAX_THREADS_PER_BLOCK 1024
#define WARP_SIZE 32

namespace goblin {

constexpr int round_up(int value, int multiple) { return ((value + multiple - 1) / multiple) * multiple; }
constexpr int ceil_div(int a, int b) { return (a + b - 1) / b; }

struct KernelDim {
    unsigned int x = 1;
    unsigned int y = 1;
    unsigned int z = 1;

    constexpr KernelDim() = default;
    constexpr KernelDim(unsigned int _x, unsigned int _y = 1, unsigned int _z = 1) 
        : x(_x), y(_y), z(_z) {}

    static inline KernelDim determine(int count, int max_threads = MAX_THREADS_PER_BLOCK) {
        KernelDim dim{WARP_SIZE};
        int min_redundant = max_threads;

        for (int threads = MAX_THREADS_PER_BLOCK; threads > 0; threads -= 32) {
            // Round up division to determine number of blocks needed
            int blocks_needed = ceil_div(count, threads);
            int redundant = blocks_needed * threads - count;

            if (redundant < min_redundant) {
                min_redundant = redundant;
                dim.x = threads;
            }

            // Early exit if perfect fit is found
            if (redundant == 0) break;
        }

        return dim;
    }

    void check() const {
        assert(x * y * z <= MAX_THREADS_PER_BLOCK);
    }

    constexpr bool operator==(const KernelDim& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

struct KernelConfig {
    KernelDim block;
    KernelDim grid;

    constexpr KernelConfig() = default;
    constexpr KernelConfig(KernelDim _block, KernelDim _grid) 
        : block(_block), grid(_grid) {}

    static inline KernelConfig for_eval(int num_solutions, int num_datapoints) {
        KernelConfig config;

        config.block = KernelDim::determine(num_datapoints);
        config.grid.x = num_solutions;
        config.grid.y = ceil_div(num_datapoints, config.block.x);

        return config;
    };

    static inline KernelConfig for_mse(int num_solutions, int num_partial, KernelVersion kernel_version) {
        KernelConfig config;

        if (kernel_version == KernelVersion::BlockReduce) {
            // Number of partial results determines the number of threads in a block
            // Rounded to a multiple of 32
            config.block.x = round_up(num_partial, WARP_SIZE); 
            // Each block determine the mse for a single solution
            config.grid.x = num_solutions;
        } else {
            config.block = KernelDim::determine(num_solutions);
            config.grid.x = ceil_div(num_solutions, config.block.x);
        }

        return config;
    }

    void check() const {
        block.check();
        // grid.check(); TODO
    }

    constexpr bool operator==(const KernelConfig& other) const {
        return block == other.block && grid == other.grid;
    }
};

struct LaunchConfig {
    KernelConfig eval;
    KernelConfig mse;
    KernelVersion kernel_version = KernelVersion::Baseline;

    constexpr LaunchConfig() = default;
    constexpr LaunchConfig(KernelConfig _eval, KernelConfig _mse, KernelVersion version = KernelVersion::Baseline)
        : eval(_eval), mse(_mse), kernel_version(version) {}

    static inline LaunchConfig determine(
        int num_solutions, 
        int num_datapoints, 
        KernelVersion kernel_version
    ) {
        KernelConfig eval_config = KernelConfig::for_eval(num_solutions, num_datapoints);
        KernelConfig mse_config = KernelConfig::for_mse(num_solutions, eval_config.grid.y, kernel_version);

        return LaunchConfig(eval_config, mse_config, kernel_version);
    }

    void check() const {
        eval.check();
        mse.check();
    }

    constexpr bool operator==(const LaunchConfig& other) const {
        return eval == other.eval && mse == other.mse && kernel_version == other.kernel_version;
    }
};

}

#endif /* _GOBLIN_GA_GP_MISC_H */