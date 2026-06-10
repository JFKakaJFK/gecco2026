#pragma once
#ifndef _GOBLIN_GA_GP_MISC_H
#define _GOBLIN_GA_GP_MISC_H

namespace goblin {

struct GpuInfo {
    int device_id;
    int num_sms;
};

GpuInfo get_gpu_info();

#ifdef __CUDACC__
void check(cudaError_t err, char const* func, char const* file, int line);
#define __CHECK_CUDA_ERR__(err) check((err), #err, __FILE__, __LINE__)
#endif

}

#endif /* _GOBLIN_GA_GP_MISC_H */
