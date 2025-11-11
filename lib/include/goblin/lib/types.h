#ifndef _GOBLIN_LIB_TYPES_H
#define _GOBLIN_LIB_TYPES_H

#pragma once

#include <Eigen/Dense>
#include <cstdint>
#include <span>
#include <vector>

namespace goblin {

using u8 = std::uint8_t;
using u16 = std::uint16_t;
using u64 = std::uint64_t;
using usize = std::size_t;    // STL index type
using isize = std::ptrdiff_t; // ~= Eigen::Index

using f32 = float;
using f64 = double;

// currently the python conversion between nanobind/numpy is broken - discrete
// values + active don't translate?
using BType = u8;  // avoid implicit bitset types
using DType = u16;
using CType = f64;

template <typename T>
using Vec = Eigen::VectorX<T>;
template <typename T>
using Mat = Eigen::MatrixX<T>;
template <typename T>
using Array = Eigen::ArrayX<T>;
template <typename T>
using Arr2D = Eigen::ArrayXX<T>;
template <typename T>
using Ref = Eigen::Ref<T>;
template <typename T>
using CRef = const Eigen::Ref<const T>;
// allow non-contiguous strides at the expense of vectorization...
template <typename T>
using RefS = Eigen::Ref<T, 0, Eigen::InnerStride<>>;
template <typename T>
using CRefS = const Eigen::Ref<const T, 0, Eigen::InnerStride<>>;

using Active = Array<BType>;

template <typename T>
constexpr bool isna(const T& v) {
  return std::isnan(v) || std::isinf(v);
};
};  // namespace goblin

#endif /* _GOBLIN_LIB_TYPES_H */
