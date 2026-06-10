#pragma once
#ifndef _GOBLIN_GA_GP_EVAL_KERNEL_H
#define _GOBLIN_GA_GP_EVAL_KERNEL_H

#include <vector>

#include "goblin/gp/gpu_evaluation/launch_config.h"
#include "goblin/gp/gpu_evaluation/types.h"

namespace goblin {

// Hard limits enforced by the evaluation kernel's shared memory and per-thread stack.
static constexpr size_t EVAL_MAX_NODES      = 512;
static constexpr size_t EVAL_MAX_STACK_DEPTH = 16;

#ifdef __CUDACC__
__device__
float compute_tree_output(
    const float* __restrict__ X,
    const u8* __restrict__ type,
    const float* __restrict__ value,
    size_t solution_length,
    size_t num_datapoints,
    size_t datapoint_index
);

__global__
void single_block_kernel(
    const float* __restrict__ X,
    const float* __restrict__ Y,
    const u8* __restrict__ v_type,
    const float* __restrict__ v_value,
    float* __restrict__ result,
    size_t solution_length,
    size_t num_datapoints,
    size_t datapoints_per_thread
);

__global__
void dynamic_block_evaluate_kernel(
    const float* __restrict__ X,
    const float* __restrict__ Y,
    const u8* __restrict__ v_type,
    const float* __restrict__ v_value,
    float* __restrict__ partial,
    size_t solution_length,
    size_t num_datapoints,
    size_t datapoints_per_block,
    size_t datapoints_per_thread
);

__global__
void dynamic_block_reduction_kernel(
    const float* __restrict__ partial,
    float* __restrict__ result,
    size_t num_partial,
    size_t num_datapoints
);

__global__
void compute_tree_output_wrapper(
    float* X,
    const u8* type,
    float* value,
    float* result,
    size_t solution_length,
    size_t num_datapoints,
    size_t datapoint_index
);
#endif

// Wrappers

void single_block_wrapper(
    float* X,
    float* Y,
    u8* type,
    float* value,
    float* result,
    LaunchConfig config
);

void dynamic_block_evaluate_wrapper(
    float* X,
    float* Y,
    u8* type,
    float* value,
    float* partial,
    LaunchConfig config
);

void dynamic_block_reduction_wrapper(
    float* partial,
    float* result,
    LaunchConfig config
);

void kernel_wrapper(
    float* X,
    float* Y,
    u8* type,
    float* value,
    float* partial,
    float* result,
    LaunchConfig config
);

// Test Wrappers

float test_compute_tree_output(
    std::vector<float> h_X,
    std::vector<u8> h_type,
    std::vector<float> h_value,
    size_t num_datapoints,
    size_t datapoint_index,
    KernelVersion version
);

std::vector<float> test_single_block(
    std::vector<float> h_X,
    std::vector<float> h_Y,
    std::vector<u8> h_type,
    std::vector<float> h_value,
    size_t num_solutions,
    size_t num_datapoints
);

std::vector<float> test_dynamic_block(
    std::vector<float> h_X,
    std::vector<float> h_Y,
    std::vector<u8> h_type,
    std::vector<float> h_value,
    size_t num_solutions,
    size_t num_datapoints,
    size_t blocks_per_individual
);

}  // namespace goblin

#endif /* _GOBLIN_GA_GP_EVAL_KERNEL_H */
