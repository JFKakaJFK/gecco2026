#pragma once
#ifndef _GOBLIN_GA_GP_MISC_H
#define _GOBLIN_GA_GP_MISC_H

#include <algorithm>
#include <cassert>

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

    KernelDim() = default;
    KernelDim(unsigned int _x, unsigned int _y = 1, unsigned int _z = 1) 
        : x(_x), y(_y), z(_z) {}

    static KernelDim determine(int count, int max_threads = MAX_THREADS_PER_BLOCK) {
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
    KernelDim grid;
    KernelDim block;
    
    KernelConfig() = default;
    KernelConfig(KernelDim _grid, KernelDim _block) 
        : grid(_grid), block(_block) {}

    static inline KernelConfig for_eval(int num_solutions, int num_datapoints) {
        KernelConfig config;

        config.block = KernelDim::determine(num_datapoints);
        config.grid.x = num_solutions;
        config.grid.y = ceil_div(num_datapoints, config.block.x);

        return config;
    };

    static KernelConfig for_mse(int num_solutions, int num_partial, KernelVersion kernel_version) {
        KernelConfig config;

        if (kernel_version == KernelVersion::BlockReduce) {
            // Each block determine the mse for a single solution
            config.grid.x = num_solutions;
            // Number of partial results determines the number of threads in a block
            // Rounded to a multiple of 32
            config.block.x = round_up(num_partial, WARP_SIZE); 
        } else {
            config.block = KernelDim::determine(num_solutions);
            config.grid.x = ceil_div(num_solutions, config.block.x);
        }

        return config;
    }

    static KernelConfig for_single(int num_solutions, int num_datapoints) {
        KernelConfig config;

        config.grid.x = num_solutions;
        config.block.x = std::min(MAX_THREADS_PER_BLOCK, round_up(num_datapoints, WARP_SIZE));

        return config;
    }

    void check() const {
        block.check();
        // grid.check(); TODO
    }

    constexpr bool operator==(const KernelConfig& other) const {
        return grid == other.grid && block == other.block;
    }
};

struct LaunchConfig {
    KernelConfig eval;
    KernelConfig mse;
    KernelVersion kernel_version = KernelVersion::Baseline;
    int num_solutions;
    int num_datapoints;
    int solution_length;
    int items_per_thread;

    LaunchConfig() = default;
    LaunchConfig(
        KernelConfig eval, 
        KernelConfig mse, 
        KernelVersion version = KernelVersion::Baseline,
        int num_solutions = 1,
        int num_datapoints = 1,
        int solution_length = 1,
        int items_per_thread = 1
    ) : eval(eval), 
        mse(mse), 
        kernel_version(version),
        num_solutions(num_solutions),
        num_datapoints(num_datapoints),
        solution_length(solution_length),
        items_per_thread(items_per_thread) {}

    static LaunchConfig determine(
        KernelVersion kernel_version,
        int num_solutions, 
        int num_datapoints, 
        int solution_length
    ) {
        LaunchConfig config;

        // Determine grid and block dimensions for kernels
        switch (kernel_version) {
            case (KernelVersion::Baseline):
            case (KernelVersion::Restrict):
            case (KernelVersion::SharedMemory):
            case (KernelVersion::BlockReduce):
                config.eval = KernelConfig::for_eval(num_solutions, num_datapoints);
                config.mse = KernelConfig::for_mse(num_solutions, config.eval.grid.y, kernel_version);
                break;
            case (KernelVersion::SingleKernel):
            case (KernelVersion::SingleKernelFMAF):
                config.eval = KernelConfig::for_single(num_solutions, num_datapoints);
                config.mse = KernelConfig();
                config.items_per_thread = (num_datapoints + config.eval.block.x - 1) / config.eval.block.x; 
                break;
            default:
                break;
        }

        config.kernel_version = kernel_version;
        config.num_solutions = num_solutions;
        config.num_datapoints = num_datapoints;
        config.solution_length = solution_length;
        
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

}

#endif /* _GOBLIN_GA_GP_MISC_H */