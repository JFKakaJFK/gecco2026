#pragma once
#ifndef _GOBLIN_GA_GP_EVAL_KERNEL_H
#define _GOBLIN_GA_GP_EVAL_KERNEL_H

#include <vector>

#include "goblin/ga-gp/misc.h"

namespace goblin {

#ifdef __CUDACC__
__global__
void evaluate_kernel_baseline(
    float* X, 
    float* Y, 
    float* v_type, 
    float* v_value, 
    float* result,
    int solution_length, 
    int num_datapoints   
);

__global__
void evaluate_kernel_restrict(
    const float* __restrict__ X, 
    const float* __restrict__ Y, 
    const float* __restrict__ v_type, 
    const float* __restrict__ v_value, 
    float* __restrict__ result,
    int solution_length, 
    int num_datapoints
);

__global__
void evaluate_kernel_shared_memory(
    const float* __restrict__ X, 
    const float* __restrict__ Y, 
    const float* __restrict__ v_type, 
    const float* __restrict__ v_value, 
    float* __restrict__ result,
    int solution_length, 
    int num_datapoints
);

__device__
float compute_tree_output_baseline(
    float* X, 
    float* type,
    float* value,
    int solution_length,
    int num_datapoints,
    int datapoint_index
);

__device__
float compute_tree_output_restrict(
    const float* __restrict__ X, 
    const float* __restrict__ type,
    const float* __restrict__ value,
    int solution_length,
    int num_datapoints,
    int datapoint_index
);

__global__
void compute_mse_kernel_baseline(
    const float* __restrict__ partial, 
    float* __restrict__ result, 
    int num_solutions, 
    int num_datapoints
);

__global__
void mse_kernel_restrict(
    const float* __restrict__ partial, 
    float* __restrict__ result, 
    int num_solutions, 
    int num_datapoints
);

__global__
void compute_tree_output_wrapper(
    float* X, 
    float* type,
    float* value,
    float* result,
    int solution_length,
    int num_datapoints,
    int datapoint_index
);
#endif

void evaluate_kernel_wrapper(
    float* X, 
    float* Y, 
    float* type, 
    float* value, 
    float* partial,
    int solution_length, 
    int num_solutions,
    int num_datapoints,
    const LaunchConfig config
);

void mse_kernel_wrapper(
    float* partial, 
    float* result, 
    int num_solutions, 
    int num_datapoints,
    const LaunchConfig config
);

float test_compute_output_kernel(
    std::vector<float> h_X,
    std::vector<float> h_type,
    std::vector<float> h_value,
    int num_datapoints,
    int datapoint_index
);

std::vector<float> test_evaluate_kernel(
    std::vector<float> h_X, 
    std::vector<float> h_Y, 
    std::vector<float> h_type, 
    std::vector<float> h_value, 
    int num_solutions,
    int num_datapoints,
    KernelVersion version
);

std::vector<float> test_compute_mse_kernel(
    std::vector<float> se, 
    int num_solutions, 
    int num_datapoints,
    KernelVersion version
);

}

#endif /* _GOBLIN_GA_GP_EVAL_KERNEL_H */