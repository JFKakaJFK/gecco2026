#ifndef _GOBLIN_GA_GP_MISC_H
#define _GOBLIN_GA_GP_MISC_H

namespace goblin {

struct GpuInfo {
    int device_id;
    int num_sms;
};

GpuInfo get_gpu_info();

}

#endif /* _GOBLIN_GA_GP_MISC_H */
