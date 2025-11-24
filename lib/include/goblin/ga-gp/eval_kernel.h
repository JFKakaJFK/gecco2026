#pragma once
#ifndef _GOBLIN_GA_GP_EVAL_KERNEL_H
#define _GOBLIN_GA_GP_EVAL_KERNEL_H

#include "goblin/ga-gp/types.h"

namespace goblin {

void eval(
    const float* X, 
    const float* Y, 
    const NodeType* v_type, 
    const float* v_value, 
    int solution_length, 
    int num_datapoints,
    float* out
);

float test_eval_kernel(
    std::vector<float> h_X,
    std::vector<float> h_Y,
    std::vector<NodeType> h_type,
    std::vector<float> h_value,
    int num_solutions,
    int num_datapoints,
    std::vector<float>& result
);

}

#endif /* _GOBLIN_GA_GP_EVAL_KERNEL_H */