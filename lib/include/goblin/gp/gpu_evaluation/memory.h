#pragma once
#ifndef _GOBLIN_GA_GP_MEMORY_H
#define _GOBLIN_GA_GP_MEMORY_H

namespace goblin {

template <typename T>
T* allocate_on_gpu(size_t count);

template <typename T>
void copy_to_gpu(T* d_ptr, const T* host_data, size_t count);

template <typename T>
T* allocate_and_copy(const T* host_data, size_t count);

template <typename T>
void copy_from_device(T* host_data, T* d_ptr, size_t count);

template <typename T>
void free_on_gpu(T* d_ptr);

template <typename T>
void zero_mem_on_gpu(T* d_ptr, size_t count);

}  // namespace goblin

#endif /* _GOBLIN_GA_GP_MEMORY_H */
