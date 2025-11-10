#ifndef _GOBLIN_LIB_ORDERING_H
#define _GOBLIN_LIB_ORDERING_H

#pragma once

#include <cassert>
#include <concepts>
#include <cstdint>
#include <string_view>
#include <type_traits>
#include <utility>

#include <Eigen/Dense>

#include "goblin/lib/types.h"

namespace goblin {

enum struct Ordering : u8 {
  Better = 0b10,
  Equal = 0b00,
  Worse = 0b01,
  NonDominated = 0b11,
};

inline constexpr Ordering operator|(Ordering lhs, Ordering rhs) noexcept {
  return static_cast<Ordering>(static_cast<u8>(lhs) | static_cast<u8>(rhs));
};

inline constexpr std::string_view format_as(const Ordering& o) noexcept {
  switch (o) {
    case Ordering::Equal:
      return std::string_view{"Ordering::Equal"};
    case Ordering::Better:
      return std::string_view{"Ordering::Better"};
    case Ordering::Worse:
      return std::string_view{"Ordering::Worse"};
    case Ordering::NonDominated:
      return std::string_view{"Ordering::NonDominated"};
  };
  std::unreachable();
};

inline std::ostream& operator<<(std::ostream& os, Ordering o) {
  return os << format_as(o);
};
};  // namespace goblin

// template <typename E>
//   requires std::is_enum_v<E> && requires { format_as(std::declval<E>()); }
// struct std::formatter<E>
//     : std::formatter<decltype(format_as(std::declval<E>()))> {
//   auto format(E e, auto &ctx) const {
//     return std::formatter<decltype(format_as(e))>::format(format_as(e), ctx);
//   }
// };

#endif /* _GOBLIN_LIB_ORDERING_H */
