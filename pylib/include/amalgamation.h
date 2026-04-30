// THIS FILE WAS GENERATED AUTOMATICALLY. DO NOT EDIT.

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                       goblin.h                                                                               //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef _GOBLIN_H
#define _GOBLIN_H


// clang-format off


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                       goblin/lib/assert.h included by goblin.h                                               //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include <format>
#include <stdexcept>

// TODO add a macro for error handling and input handling to that
// a simple runtime assert to check cheap invariants even with optimizations
// - use often, profile and only remove if runtime cost is considerable
#ifndef __goblin_runtime_assert
#define __goblin_runtime_assert(x)                                                                             \
  if (!(x)) {                                                                                                  \
    throw std::runtime_error(std::format("{}:{}:\n\t Runtime assertion `{}` failed", __FILE__, __LINE__, #x)); \
  } else  // eats up semicolon
#endif

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                       goblin/lib/types.h included by goblin.h                                                //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef _GOBLIN_LIB_TYPES_H
#define _GOBLIN_LIB_TYPES_H


// #define EIGEN_RUNTIME_NO_MALLOC  // enable runtime allocation testing

#include <Eigen/Dense>
#include <cstdint>
#include <span>
#include <vector>
#include <string>

namespace goblin {

using u8 = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using usize = std::size_t;     // STL index type
using isize = std::ptrdiff_t;  // ~= Eigen::Index

using f32 = float;
using f64 = double;

// currently the python conversion between nanobind/numpy is broken - discrete
// values + active don't translate?
using BType = u8;  // not using bool avoids implicit bitset types
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

// Is this a good idea?
// template <typename T>
// using Box = std::unique_ptr<T>;
// template <typename T>
// using Rc = std::shared_ptr<T>;

template <typename T>
constexpr bool isna(const T& v) {
  return std::isnan(v) || std::isinf(v);
};
};  // namespace goblin

#endif /* _GOBLIN_LIB_TYPES_H */

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                       goblin/lib/rng.h included by goblin.h                                                  //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef _GOBLIN_LIB_RNG_H
#define _GOBLIN_LIB_RNG_H

#include <algorithm>
#include <concepts>
#include <csetjmp>
#include <numeric>
#include <random>
#include <type_traits>
#include <optional>

#include <openrand/philox.h>
#include <openrand/squares.h>


namespace goblin {
using Rng = openrand::Philox;
// using Rng = std::mt19937;

// OpenRAND and other rngs don't have the same API other than std::uniform_random_bit_generator, so a wrapper for the
// RNG creation allows changing PRNG
template <typename R = Rng>
inline R seeded_rng(u64 state, u32 ctr = 0) {
  // OpenRAND-like
  if constexpr (std::is_constructible_v<R, u64, u32>) {
    return R(state, ctr);
  } else {
    // std::random-like
    std::seed_seq seed{static_cast<u32>(state), ctr};
    R rng;
    rng.seed(seed);
    return rng;
  }
};
inline Rng seeded_rng(std::optional<u64> seed = std::nullopt) {
  if (seed.has_value()) {
    return seeded_rng(seed.value());
  } else {
    std::random_device rd;
    std::uniform_int_distribution<u64> seed_dist(0, std::numeric_limits<u64>::max());

    return seeded_rng<Rng>(seed_dist(rd));
  }
};

// TODO possibly use the OpenRAND provided sampling methods (decreases rng portability, but it looks like any rng can be
// wrapped with openrand::BaseRNG<T> and the sampling methods both look convenient and decently fast)

// TODO possibly profile & look at (for faster rn generation)
// - https://www.pcg-random.org/posts/bounded-rands.html
// - https://github.com/swiftlang/swift/pull/39143#issue-comment-box

inline std::vector<usize> permute(Rng& rng, usize n) {
  std::vector<usize> p(n);
  std::iota(p.begin(), p.end(), 0);
  std::shuffle(p.begin(), p.end(), rng);
  return p;
};
};  // namespace goblin

static_assert(std::uniform_random_bit_generator<goblin::Rng>);

#endif /* _GOBLIN_LIB_RNG_H */

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                       goblin/lib/ordering.h included by goblin.h                                             //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef _GOBLIN_LIB_ORDERING_H
#define _GOBLIN_LIB_ORDERING_H


#include <cassert>
#include <string_view>
#include <utility>



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

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                       goblin/lib/fitness.h included by goblin.h                                              //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef _GOBLIN_LIB_FITNESS_H
#define _GOBLIN_LIB_FITNESS_H

#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>



namespace goblin {

/// Something that describes how good a solution is
class QualityBase {
 public:
  virtual std::unique_ptr<QualityBase> clone() const = 0;

  virtual ~QualityBase() = default;

  template <typename T>
  bool is() const {
    return dynamic_cast<const T*>(this) != nullptr;
  }
  template <typename T>
  const T& as() const {
#ifndef NDEBUG
    auto p = dynamic_cast<const T*>(this);
    assert(p != nullptr && "Quality type mismatch");
#endif
    return static_cast<const T&>(*this);
  }
  template <typename T>
  T& as() {
#ifndef NDEBUG
    auto p = dynamic_cast<T*>(this);
    assert(p != nullptr && "Quality type mismatch");
#endif
    return static_cast<T&>(*this);
  }
};

class FitnessBase {
 public:
  virtual usize num_objectives() const = 0;

  virtual Ordering cmp(const QualityBase& lhs, const QualityBase& rhs, std::optional<usize> objective) const = 0;

  virtual CType distance(const QualityBase& lhs, const QualityBase& rhs, std::optional<usize> objective) const = 0;

  virtual void log_header(std::ostream& os) const = 0;

  virtual void log(std::ostream& os, const QualityBase& quality) const = 0;

  virtual std::string format(const QualityBase& quality) const {
    std::stringstream ss;
    log(ss, quality);
    return ss.str();
  };

  virtual ~FitnessBase() = default;
};

class ArchiveFitnessBase : public FitnessBase {
 public:
  virtual std::unique_ptr<QualityBase> worst() const = 0;

  virtual ~ArchiveFitnessBase() = default;
};

/// Something that describes how good a solution is
class MOQuality : public QualityBase {
 public:
  std::unique_ptr<QualityBase> clone() const override { return std::make_unique<MOQuality>(*this); };

  Vec<CType> objectives;
  CType constraint_value;
};

class MOFitness : public ArchiveFitnessBase {
 public:
  MOFitness() = delete;
  MOFitness(usize num_objectives, bool minimize = true, CType epsilon = 0.0)
      : _num_objectives(num_objectives), _epsilon(epsilon), _minimize(minimize) {};

  void log_header(std::ostream& os) const override final { os << "objectives,constraint_value"; };

  void log(std::ostream& os, const QualityBase& quality) const override final {
    const auto& q = quality.as<MOQuality>();
    os << "\"[";
    for (usize i = 0; i < _num_objectives; i++) {
      if (i > 0) {
        os << ',';
      }
      os << q.objectives(i);
    }
    os << "]\"," << q.constraint_value;
  };

  std::string format(const QualityBase& quality) const override final {
    std::stringstream ss;
    log(ss, quality);
    return ss.str();
  };

  usize num_objectives() const override final { return _num_objectives; };

  Ordering cmp(const QualityBase& lhs,
               const QualityBase& rhs,
               std::optional<usize> objective = std::nullopt) const override final;

  CType distance(const QualityBase& lhs,
                 const QualityBase& rhs,
                 std::optional<usize> objective = std::nullopt) const override final;

  virtual std::unique_ptr<QualityBase> worst() const override {
    const CType inf = std::numeric_limits<CType>().infinity();
    auto q = std::make_unique<MOQuality>();
    q->objectives = Vec<CType>::Constant(_num_objectives, (_minimize ? inf : -inf));
    q->constraint_value = inf;
    return q;
  };

 private:
  inline Ordering cmp(const CType& lhs, const CType& rhs, double epsilon, bool minimize) const noexcept {
    bool lhs_finite = !isna(lhs);
    bool rhs_finite = !isna(rhs);
    if (!lhs_finite && !rhs_finite)
      return Ordering::NonDominated;
    if (!lhs_finite)
      return Ordering::Worse;
    if (!rhs_finite)
      return Ordering::Better;
    if (minimize) {
      if (lhs + epsilon < rhs)
        return Ordering::Better;
      if (lhs > rhs + epsilon)
        return Ordering::Worse;
    } else {
      if (lhs > rhs + epsilon)
        return Ordering::Better;
      if (lhs + epsilon < rhs)
        return Ordering::Worse;
    }
    return Ordering::Equal;
  };

  inline CType distance(const CType& lhs, const CType& rhs) const noexcept { return std::abs(lhs - rhs); };

  usize _num_objectives;
  CType _epsilon;
  bool _minimize;
};

};  // namespace goblin

#endif /* _GOBLIN_LIB_FITNESS_H */

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                       goblin/lib/subset.h included by goblin.h                                               //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef _GOBLIN_LIB_SUBSET_H
#define _GOBLIN_LIB_SUBSET_H

#include <cstdlib>
#include <functional>



namespace goblin {

struct Subset {
  std::vector<usize> discrete;
  std::vector<usize> continuous;

  inline void clear() {
    discrete.clear();
    continuous.clear();
  };

  inline usize size() const noexcept { return discrete.size() + continuous.size(); }

  inline bool empty() const noexcept { return discrete.empty() && continuous.empty(); }

  CType similarity(const Subset& other) const {
    usize matches = 0;

    for (usize i : discrete) {
      for (usize j : other.discrete) {
        if (i == j) {
          matches++;
        }
      }
    }
    for (usize i : continuous) {
      for (usize j : other.continuous) {
        if (i == j) {
          matches++;
        }
      }
    }

    /* // should be correct, but for now let's keep it simple
    usize i = 0, j = 0;
    // subset indices are sorted, so we only need to check until the next index
    // in the other set is higher
    while (i < discrete.size() && j < other.discrete.size()) {
      if (discrete[i] < other.discrete[j]) {
        i++;
      } else if (discrete[i] > other.discrete[j]) {
        j++;
      } else {
        i++;
        j++;
        matches++;
      }
    }
    i = 0;
    j = 0;
    while (i < continuous.size() && j < other.continuous.size()) {
      if (continuous[i] < other.continuous[j]) {
        i++;
      } else if (continuous[i] > other.continuous[j]) {
        j++;
      } else {
        i++;
        j++;
        matches++;
      }
    } */
    return static_cast<CType>(matches) / static_cast<CType>(std::max(size(), other.size()));
  };

  Subset merge(const Subset& other) const {
    assert(std::is_sorted(discrete.begin(), discrete.end()) &&
           std::is_sorted(other.discrete.begin(), other.discrete.end()) && "Discrete indices are not sorted!");
    assert(std::is_sorted(continuous.begin(), continuous.end()) &&
           std::is_sorted(other.continuous.begin(), other.continuous.end()) && "Continuous indices are not sorted!");

    Subset s;
    usize this_i = 0, other_i = 0, idx;
    while (this_i < discrete.size() || other_i < other.discrete.size()) {
      if (this_i >= discrete.size()) {
        idx = other.discrete[other_i++];
      } else if (other_i >= other.discrete.size()) {
        idx = discrete[this_i++];
      } else if (discrete[this_i] <= other.discrete[other_i]) {
        idx = discrete[this_i++];
      } else {
        idx = other.discrete[other_i++];
      }

      if (s.discrete.empty() || s.discrete.back() != idx) {
        s.discrete.push_back(idx);
      }
    }
    this_i = 0;
    other_i = 0;
    while (this_i < continuous.size() || other_i < other.continuous.size()) {
      if (this_i >= continuous.size()) {
        idx = other.continuous[other_i++];
      } else if (other_i >= other.continuous.size()) {
        idx = continuous[this_i++];
      } else if (continuous[this_i] <= other.continuous[other_i]) {
        idx = continuous[this_i++];
      } else {
        idx = other.continuous[other_i++];
      }

      if (s.continuous.empty() || s.continuous.back() != idx) {
        s.continuous.push_back(idx);
      }
    }
    return s;
  };
};

inline bool operator==(const Subset& lhs, const Subset& rhs) {
  return lhs.continuous == rhs.discrete && lhs.continuous == rhs.continuous;
}
inline bool operator!=(const Subset& lhs, const Subset& rhs) {
  return !(lhs == rhs);
}

using FOS = std::vector<Subset>;

}  // namespace goblin

#endif /* _GOBLIN_LIB_SUBSET_H */

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                       goblin/lib/solution.h included by goblin.h                                             //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef _GOBLIN_LIB_SOLUTION_H
#define _GOBLIN_LIB_SOLUTION_H




// Note the separate solution exists to hide the data ownership
// - without it, for operations using separate arenas like
// e.g. GOM with a donor from an archive the arena would have to be handled
// explicitly, but that is an implementation detail GOMEA shouldn't have to know
// about. And for other representations such as a vector of solution structs,
// this is also nicer.

namespace goblin {

struct SolutionExtensionKey {
  void* token = nullptr;
};

inline bool operator==(const SolutionExtensionKey& lhs, const SolutionExtensionKey& rhs) noexcept {
  return lhs.token == rhs.token;
}

inline bool operator!=(const SolutionExtensionKey& lhs, const SolutionExtensionKey& rhs) noexcept {
  return lhs.token != rhs.token;
}

struct SolutionExtensionKeyHash {
  usize operator()(const SolutionExtensionKey& key) const noexcept { return std::hash<void*>{}(key.token); };
};

class SolutionExtensionBase {
 public:
  virtual std::unique_ptr<SolutionExtensionBase> clone() const = 0;
  virtual SolutionExtensionKey key() const = 0;

  operator SolutionExtensionKey() const { return key(); }

  virtual ~SolutionExtensionBase() = default;
};

// Use CRTP to automatically add type (NOT instance) specific keys
template <typename Derived>
struct SolutionExtension : public SolutionExtensionBase {
  static SolutionExtensionKey type_key() {
    static u32 anchor{};
    return {&anchor};
  };

  SolutionExtensionKey key() const override { return type_key(); };
};

class SolutionBase {
 public:
  virtual bool has_extension(const SolutionExtensionKey& key) const = 0;
  virtual SolutionExtensionBase& get_or_insert_extension(const SolutionExtensionBase& extension) = 0;  // this is fine
  virtual std::optional<std::reference_wrapper<const SolutionExtensionBase>> get_extension(
      const SolutionExtensionKey& key) const = 0;
  virtual std::optional<std::reference_wrapper<SolutionExtensionBase>> get_extension(
      const SolutionExtensionKey& key) = 0;

  template <typename T>
  const T& extension() const {
    auto e = get_extension(T::type_key());
    if (!e.has_value()) {
      throw std::runtime_error("Invalid extension access!");
    }
#ifndef NDEBUG
    auto p = dynamic_cast<const T*>(&e.value().get());
    assert(p != nullptr && "Extension type mismatch");
#endif
    return static_cast<const T&>(e.value().get());
  };
  template <typename T>
  T& extension() {
    auto e = get_extension(T::type_key());
    if (!e.has_value()) {
      throw std::runtime_error("Invalid extension access!");
    }
#ifndef NDEBUG
    auto p = dynamic_cast<const T*>(&e.value().get());
    assert(p != nullptr && "Extension type mismatch");
#endif
    return static_cast<T&>(e.value().get());
  };
  virtual bool remove_extension(const SolutionExtensionKey& key) = 0;
  virtual void clear_extensions() = 0;

  // Instead of vector, shoould this be an ExtensionProxy that behaves like a vector/iterator but does not allocate full
  // copies?? (size, begin, end, proxy to underlying collection)
  virtual usize num_extensions() const = 0;
  virtual std::vector<std::reference_wrapper<const SolutionExtensionBase>> extensions() const = 0;
  virtual std::vector<std::reference_wrapper<SolutionExtensionBase>> extensions() = 0;

  virtual QualityBase& quality() = 0;
  virtual const QualityBase& quality() const = 0;
  virtual void assign_quality(const QualityBase& quality) = 0;

  template <typename T>
  const T& quality_as() const {
    const auto& q = quality();
#ifndef NDEBUG
    auto p = dynamic_cast<const T*>(&q);
    assert(p != nullptr && "Quality type mismatch");
#endif
    return static_cast<const T&>(q);
  }
  template <typename T>
  T& quality_as() {
    auto& q = quality();
#ifndef NDEBUG
    auto p = dynamic_cast<T*>(&q);
    assert(p != nullptr && "Quality type mismatch");
#endif
    return static_cast<T&>(q);
  }

  inline usize num_discrete() const { return discrete_values().size(); };

  virtual RefS<Vec<DType>> discrete_values() = 0;
  virtual CRefS<Vec<DType>> discrete_values() const = 0;
  virtual RefS<Array<BType>> discrete_active() = 0;
  virtual CRefS<Array<BType>> discrete_active() const = 0;

  inline usize num_continuous() const { return continuous_values().size(); };

  virtual RefS<Vec<CType>> continuous_values() = 0;
  virtual CRefS<Vec<CType>> continuous_values() const = 0;
  virtual RefS<Array<BType>> continuous_active() = 0;
  virtual CRefS<Array<BType>> continuous_active() const = 0;

  SolutionBase& operator=(const SolutionBase& other);

  virtual ~SolutionBase() {};
};

class Solution : public SolutionBase {
 public:
  Solution(std::unique_ptr<QualityBase> quality,
           std::optional<Vec<DType>> discrete_values = std::nullopt,
           std::optional<Vec<CType>> continuous_values = std::nullopt)
      : _quality(std::move(quality)) {
    if (discrete_values.has_value()) {
      _discrete_values = discrete_values.value();
      _discrete_active.resize(_discrete_values.size());
      std::fill(_discrete_active.begin(), _discrete_active.end(), true);
    }

    if (continuous_values.has_value()) {
      _continuous_values = continuous_values.value();
      _continuous_active.resize(_continuous_values.size());
      std::fill(_continuous_active.begin(), _continuous_active.end(), true);
    }
  }

  Solution(const Solution& other)
      : _discrete_values(other.discrete_values()),
        _discrete_active(other.discrete_active()),
        _continuous_values(other.continuous_values()),
        _continuous_active(other.continuous_active()),
        _quality(other.quality().clone()) {
    _extensions.clear();
    for (const auto& e : other.extensions()) {
      _extensions.push_back(e.get().clone());
    }
  };
  Solution(Solution&& other)
      : _discrete_values(std::move(other._discrete_values)),
        _discrete_active(std::move(other._discrete_active)),
        _continuous_values(std::move(other._continuous_values)),
        _continuous_active(std::move(other._continuous_active)),
        _extensions(std::move(other._extensions)),
        _quality(other._quality->clone()) {
    assert(_quality != nullptr);
  };

  Solution& operator=(const Solution& other) {
    if (&other != this) {
      __goblin_runtime_assert(other.num_discrete() == num_discrete());
      __goblin_runtime_assert(other.num_continuous() == num_continuous());

      discrete_values() = other.discrete_values();
      discrete_active() = other.discrete_active();

      continuous_values() = other.continuous_values();
      continuous_active() = other.continuous_active();

      clear_extensions();
      for (const auto& e : other.extensions()) {
        get_or_insert_extension(e);
      }

      assign_quality(other.quality());
    }
    return *this;
  }

  Solution& operator=(Solution&& other) {
    if (&other != this) {
      _discrete_values = std::move(other._discrete_values);
      _discrete_active = std::move(other._discrete_active);
      _continuous_values = std::move(other._continuous_values);
      _continuous_active = std::move(other._continuous_active);
      _extensions = std::move(other._extensions);
      _quality = other._quality->clone();
    }

    return *this;
  }

  // explicitly not explicit since implicit conversion is the intent
  Solution(const SolutionBase& s)
      : _discrete_values(s.discrete_values()),
        _discrete_active(s.discrete_active()),
        _continuous_values(s.continuous_values()),
        _continuous_active(s.continuous_active()),
        _quality(s.quality().clone()) {
    _extensions.clear();
    for (const auto& e : s.extensions()) {
      _extensions.push_back(e.get().clone());
    }
  };

  void assign_quality(const QualityBase& quality) override final { _quality = quality.clone(); };
  QualityBase& quality() override final {
    assert(_quality != nullptr);
    return *_quality;
  }
  const QualityBase& quality() const override final {
    assert(_quality != nullptr);
    return *_quality;
  }

  RefS<Vec<DType>> discrete_values() override final { return _discrete_values; }
  CRefS<Vec<DType>> discrete_values() const override final { return _discrete_values; }
  RefS<Array<BType>> discrete_active() override final { return _discrete_active; }
  CRefS<Array<BType>> discrete_active() const override final { return _discrete_active; }

  RefS<Vec<CType>> continuous_values() override final { return _continuous_values; }
  CRefS<Vec<CType>> continuous_values() const override final { return _continuous_values; }
  RefS<Array<BType>> continuous_active() override final { return _continuous_active; }
  CRefS<Array<BType>> continuous_active() const override final { return _continuous_active; }

  bool has_extension(const SolutionExtensionKey& key) const override final {
    for (auto& e : _extensions) {
      if ((*e).key() == key) {
        return true;
      }
    }
    return false;
  };
  SolutionExtensionBase& get_or_insert_extension(const SolutionExtensionBase& extension) override final {
    for (auto& e : _extensions) {
      if ((*e).key() == extension.key()) {
        return *e;
      }
    }
    _extensions.push_back(extension.clone());
    return *_extensions.back();
  };
  std::optional<std::reference_wrapper<const SolutionExtensionBase>> get_extension(
      const SolutionExtensionKey& key) const override final {
    for (const auto& e : _extensions) {
      if ((*e).key() == key) {
        return *e;
      }
    }
    return std::nullopt;
  };
  std::optional<std::reference_wrapper<SolutionExtensionBase>> get_extension(
      const SolutionExtensionKey& key) override final {
    for (auto& e : _extensions) {
      if ((*e).key() == key) {
        return *e;
      }
    }
    return std::nullopt;
  };
  bool remove_extension(const SolutionExtensionKey& key) override final {
    for (usize i = 0; i < _extensions.size(); i++) {
      if ((*_extensions[i]).key() == key) {
        std::swap(_extensions[i], _extensions.back());
        _extensions.pop_back();
        return true;
      }
    }
    return false;
  };
  void clear_extensions() override final { _extensions.clear(); };

  usize num_extensions() const override final { return _extensions.size(); };
  std::vector<std::reference_wrapper<const SolutionExtensionBase>> extensions() const override final {
    std::vector<std::reference_wrapper<const SolutionExtensionBase>> exts{};
    for (const auto& e : _extensions) {
      exts.push_back(*e);
    }
    return exts;
  };
  std::vector<std::reference_wrapper<SolutionExtensionBase>> extensions() override final {
    std::vector<std::reference_wrapper<SolutionExtensionBase>> exts{};
    for (auto& e : _extensions) {
      exts.push_back(*e);
    }
    return exts;
  };

 private:
  Vec<DType> _discrete_values;
  Array<BType> _discrete_active;
  Vec<CType> _continuous_values;
  Array<BType> _continuous_active;
  std::vector<std::unique_ptr<SolutionExtensionBase>> _extensions{};
  std::unique_ptr<QualityBase> _quality;
};

class SolutionSetBase {
 public:
  virtual SolutionBase& operator[](usize idx) = 0;
  virtual const SolutionBase& operator[](usize idx) const = 0;

  virtual usize size() const = 0;
  virtual bool empty() const = 0;

  virtual void add(const SolutionBase& s) = 0;

  virtual void remove_at(usize idx) = 0;
  virtual void remove_indices_sorted(const std::span<const usize>& indices) = 0;

  virtual void reserve(usize new_capacity) = 0;
  virtual void clear() = 0;

  virtual ~SolutionSetBase() {};
};

// TODO possibly convert the runtime to normal asserts if they measurably impact performance

struct AoSSet : public SolutionSetBase {
 public:
  SolutionBase& operator[](usize idx) override final {
    __goblin_runtime_assert(idx < solutions.size());
    return solutions[idx];
  }

  const SolutionBase& operator[](usize idx) const override final {
    __goblin_runtime_assert(idx < solutions.size());
    return solutions[idx];
  }

  usize size() const override final { return solutions.size(); }

  bool empty() const override final { return solutions.empty(); }

  void add(const SolutionBase& s) override final { solutions.emplace_back(s); }

  void remove_at(usize idx) override final {
    __goblin_runtime_assert(solutions.size() > 0);
    __goblin_runtime_assert(idx < solutions.size());

    usize last = solutions.size() - 1;
    if (idx != last) {
      solutions[idx] = solutions[last];
    }
    solutions.pop_back();
  }

  void remove_indices_sorted(const std::span<const usize>& indices) override final {
    // ! invalidates the solutions
    // ! indices has to be sorted in ascending order
    for (usize i = indices.size(); i > 0; --i) {
      remove_at(indices[i - 1]);
    }
  }

  void reserve(usize new_capacity) override final { solutions.reserve(new_capacity); }

  void clear() override final { solutions.clear(); }

 private:
  std::vector<Solution> solutions;
};

template <int StorageOrder>
class SoASet;

template <int StorageOrder>
class SolutionHandle : public SolutionBase {
 public:
  void assign_quality(const QualityBase& quality) override { arena->quality[idx] = quality.clone(); }
  QualityBase& quality() override final {
    assert(arena->quality[idx] != nullptr);
    return *arena->quality[idx];
  }
  const QualityBase& quality() const override final {
    assert(arena->quality[idx] != nullptr);
    return *arena->quality[idx];
  }

  RefS<Vec<DType>> discrete_values() override final { return arena->discrete.row(idx); }
  CRefS<Vec<DType>> discrete_values() const override final { return arena->discrete.row(idx); }
  RefS<Array<BType>> discrete_active() override final { return arena->discrete_active.row(idx); }
  CRefS<Array<BType>> discrete_active() const override final { return arena->discrete_active.row(idx); }

  RefS<Vec<CType>> continuous_values() override final { return arena->continuous.row(idx); }
  CRefS<Vec<CType>> continuous_values() const override final { return arena->continuous.row(idx); }
  RefS<Array<BType>> continuous_active() override final { return arena->continuous_active.row(idx); }
  CRefS<Array<BType>> continuous_active() const override final { return arena->continuous_active.row(idx); }

  bool has_extension(const SolutionExtensionKey& key) const override final {
    for (auto& e : arena->extensions[idx]) {
      if ((*e).key() == key) {
        return true;
      }
    }
    return false;
  };
  SolutionExtensionBase& get_or_insert_extension(const SolutionExtensionBase& extension) override final {
    for (auto& e : arena->extensions[idx]) {
      if ((*e).key() == extension.key()) {
        return *e;
      }
    }
    arena->extensions[idx].push_back(extension.clone());
    return *arena->extensions[idx].back();
  };
  std::optional<std::reference_wrapper<const SolutionExtensionBase>> get_extension(
      const SolutionExtensionKey& key) const override final {
    for (const auto& e : arena->extensions[idx]) {
      if ((*e).key() == key) {
        return *e;
      }
    }
    return std::nullopt;
  };
  std::optional<std::reference_wrapper<SolutionExtensionBase>> get_extension(
      const SolutionExtensionKey& key) override final {
    for (auto& e : arena->extensions[idx]) {
      if ((*e).key() == key) {
        return *e;
      }
    }
    return std::nullopt;
  };
  bool remove_extension(const SolutionExtensionKey& key) override final {
    for (usize i = 0; i < arena->extensions[idx].size(); i++) {
      if ((*arena->extensions[idx][i]).key() == key) {
        std::swap(arena->extensions[idx][i], arena->extensions[idx].back());
        arena->extensions[idx].pop_back();
        return true;
      }
    }
    return false;
  };
  void clear_extensions() override final { arena->extensions[idx].clear(); };

  usize num_extensions() const override final { return arena->extensions[idx].size(); };
  std::vector<std::reference_wrapper<const SolutionExtensionBase>> extensions() const override final {
    std::vector<std::reference_wrapper<const SolutionExtensionBase>> exts{};
    for (const auto& e : arena->extensions[idx]) {
      exts.push_back(*e);
    }
    return exts;
  };
  std::vector<std::reference_wrapper<SolutionExtensionBase>> extensions() override final {
    std::vector<std::reference_wrapper<SolutionExtensionBase>> exts{};
    for (auto& e : arena->extensions[idx]) {
      exts.push_back(*e);
    }
    return exts;
  };

  SolutionHandle(SoASet<StorageOrder>* arena, usize idx) : arena(arena), idx(idx) {};

 private:
  friend class SoASet<StorageOrder>;  // allow accessing arena + idx
  SoASet<StorageOrder>* arena;
  usize idx;
};

template <int StorageOrder>
class SoASet : public SolutionSetBase {
 public:
  SoASet() = default;

  // Needed since the handles need to be udpdated when moving...
  SoASet(SoASet&& other) noexcept
      : _size(other.size()),
        _capacity(other._capacity),
        _num_discrete(other._num_discrete),
        _num_continuous(other._num_continuous),
        handles(std::move(other.handles)),
        discrete(std::move(other.discrete)),
        continuous(std::move(other.continuous)),
        discrete_active(std::move(other.discrete_active)),
        continuous_active(std::move(other.continuous_active)),
        extensions(std::move(other.extensions)),
        quality(std::move(other.quality)) {
    for (auto& h : handles) {
      h.arena = this;
    }
  };

  // Needed since the handles need to be udpdated when moving...
  SoASet& operator=(SoASet&& other) noexcept {
    if (this != &other) {
      _size = other.size();
      _capacity = other._capacity;
      _num_discrete = other._num_discrete;
      _num_continuous = other._num_continuous;
      handles = std::move(other.handles);
      discrete = std::move(other.discrete);
      continuous = std::move(other.continuous);
      discrete_active = std::move(other.discrete_active);
      continuous_active = std::move(other.continuous_active);
      extensions = std::move(other.extensions);
      quality = std::move(other.quality);

      for (auto& h : handles) {
        h.arena = this;
      }
    }
    return *this;
  };

  /// @warning References returned by operator[] are invalidated by add()
  /// if reallocation occurs. Do not hold references across add() calls.
  SolutionBase& operator[](usize idx) override {
    __goblin_runtime_assert(idx < _size);
    return handles[idx];
  }

  /// @warning References returned by operator[] are invalidated by add()
  /// if reallocation occurs. Do not hold references across add() calls.
  const SolutionBase& operator[](usize idx) const override {
    __goblin_runtime_assert(idx < _size);
    return handles[idx];
  }

  usize size() const override final { return _size; }

  bool empty() const override final { return _size == 0; }

  void reserve(usize new_capacity) override final {
    if (new_capacity > _capacity) {
      discrete.conservativeResize(new_capacity, _num_discrete);
      continuous.conservativeResize(new_capacity, _num_continuous);

      discrete_active.conservativeResize(new_capacity, _num_discrete);
      continuous_active.conservativeResize(new_capacity, _num_continuous);

      quality.reserve(new_capacity);
      extensions.reserve(new_capacity);

      handles.reserve(new_capacity);
      for (usize i = _capacity; i < new_capacity; i++) {
        handles.emplace_back(this, i);
      };

      _capacity = new_capacity;
    }
  }

  // TODO, the following potentially causes a use-after-free:
  //
  // auto& s = set[0];
  // set.add(s); // can move handles!!!
  // s.quality(); // use-after-free
  // I'm not sure if that is still the case, but the same idea holds if the whole SoASet is moved (e.g. when you have a
  // vector of populations that gets resized and all populations move so that the arena pointer in the solution is out
  // of date)
  void add(const SolutionBase& s) override final {
    bool resize_needed = _size + 1 >= _capacity;
    if (_num_discrete == 0 && _num_continuous == 0 &&
        (s.num_discrete() != _num_discrete || s.num_continuous() != _num_continuous)) {
      _num_discrete = s.num_discrete();
      _num_continuous = s.num_continuous();
      resize_needed = true;
    }
    __goblin_runtime_assert(s.num_discrete() == _num_discrete);
    __goblin_runtime_assert(s.num_continuous() == _num_continuous);

    std::optional<usize> s_idx;
    if (resize_needed) {
      if (_size > 0) {
        // if s is from this solution set, resizing potentially
        // moves the handle s is pointing towards -> use-after-free
        // To prevent this, we check if this is the case and get the
        // arena offset before potentially deallocating the solution
        if (auto h = dynamic_cast<const SolutionHandle<StorageOrder>*>(&s); h != nullptr && h->arena == this) {
          s_idx = h->idx;
        }
      }

      reserve(_capacity == 0 ? 32 : static_cast<usize>(1.5 * static_cast<double>(_capacity)));
    }
    assert(_size + 1 < _capacity);

    const SolutionBase& _s = s_idx.has_value() ? operator[](s_idx.value()) : s;

    discrete.row(_size) = _s.discrete_values();
    continuous.row(_size) = _s.continuous_values();
    discrete_active.row(_size) = _s.discrete_active();
    continuous_active.row(_size) = _s.continuous_active();
    assert(quality.size() >= _size);
    if (quality.size() == _size) {
      quality.push_back(_s.quality().clone());
    } else {
      quality[_size] = _s.quality().clone();
    }

    if (extensions.size() == _size) {
      extensions.emplace_back();
    }
    extensions[_size].clear();
    for (const auto& e : _s.extensions()) {
      extensions[_size].push_back(e.get().clone());
    }

    _size++;
  }

  void remove_at(usize idx) override final {
    __goblin_runtime_assert(idx < _size);

    --_size;
    if (idx != _size) {
      discrete.row(idx) = discrete.row(_size);
      continuous.row(idx) = continuous.row(_size);
      discrete_active.row(idx) = discrete_active.row(_size);
      continuous_active.row(idx) = continuous_active.row(_size);
      std::swap(extensions[idx], extensions[_size]);
      extensions[_size].clear();
      quality[idx] = std::move(quality[_size]);

      // ! no need to adjust the handles, but any references to handles may be
      // out of date now...
    }
  }

  void remove_indices_sorted(const std::span<const usize>& indices) override final {
    // ! invalidates the solutions
    // ! indices has to be sorted in ascending order
    for (usize i = indices.size(); i > 0; --i) {
      remove_at(indices[i - 1]);
    }
  }

  void clear() override final { _size = 0; }

 private:
  friend class SolutionHandle<StorageOrder>;
  usize _size = 0;
  usize _capacity = 0;
  usize _num_discrete = 0;
  usize _num_continuous = 0;

  std::vector<SolutionHandle<StorageOrder>> handles;
  Eigen::Matrix<DType, Eigen::Dynamic, Eigen::Dynamic, StorageOrder> discrete;
  Eigen::Matrix<CType, Eigen::Dynamic, Eigen::Dynamic, StorageOrder> continuous;
  Eigen::Matrix<BType, Eigen::Dynamic, Eigen::Dynamic, StorageOrder> discrete_active;
  Eigen::Matrix<BType, Eigen::Dynamic, Eigen::Dynamic, StorageOrder> continuous_active;

  std::vector<std::vector<std::unique_ptr<SolutionExtensionBase>>> extensions;
  std::vector<std::unique_ptr<QualityBase>> quality;
};

using DefaultSolutionSet = AoSSet;

};  // namespace goblin

#endif /* _GOBLIN_LIB_SOLUTION_H */
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                       goblin/lib/archive.h included by goblin.h                                              //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef _GOBLIN_LIB_ARCHIVE_H
#define _GOBLIN_LIB_ARCHIVE_H

#include <print>
#include <tuple>



namespace goblin {

class ArchiveBase {
 public:
  virtual usize size() const = 0;
  bool empty() const { return size() == 0; };
  virtual const SolutionBase& operator[](usize idx) const = 0;

  // Unsafe because by being in the archive we assume the invariant that the
  // solution is non-dominated. Allowing potential changes to the solution
  // without re-checking non-dominance hence is not "safe" from the logic
  // perspective.
  virtual SolutionBase& unsafe_at(usize idx) = 0;

  usize change_count() const { return _change_count; };
  void reset_change_count() { _change_count = 0; };

  virtual const SolutionBase& so_solution(usize objective) const = 0;

  virtual void adapt() {};

  virtual void clear() = 0;

  virtual const ArchiveFitnessBase& fitness() const = 0;

  virtual std::unique_ptr<ArchiveBase> clone() const = 0;

  /// Updates the archive with the solution and returns whether the solution was
  /// accepted into the archive.
  bool update(const SolutionBase& solution,
              bool strict,               // If false, returns true for solutions that are
                                         // non-dominated by the archive but not accepted
              bool check_synched = true  // Only consider a solution accepted if it
                                         // also is accepted by all synched archives
  );

  void sync_with(ArchiveBase& other) { synched_archives.push_back(other); };
  void unsync_all() { synched_archives.clear(); };

  bool dominates(const SolutionBase& solution, bool strict) const;
  bool covers(const ArchiveBase& other) const;

  inline const SolutionBase& random_solution(Rng& rng) const {
    __goblin_runtime_assert(!empty());
    return operator[](std::uniform_int_distribution<usize>(0, size() - 1)(rng));
  };

  virtual ~ArchiveBase() {};

 protected:
  virtual std::tuple<bool, bool> update_archive(const SolutionBase& solution, bool strict) = 0;

  usize _change_count = 0;
  std::vector<std::reference_wrapper<ArchiveBase>> synched_archives;
};

class UnboundedArchive : public ArchiveBase {
 public:
  UnboundedArchive(const ArchiveFitnessBase& fitness) : _fitness(fitness) {};

  std::unique_ptr<ArchiveBase> clone() const override final { return std::make_unique<UnboundedArchive>(*this); };

  usize size() const override final { return _solutions.size(); }
  const SolutionBase& operator[](usize idx) const override final { return _solutions[idx]; }
  SolutionBase& unsafe_at(usize idx) override final { return _solutions[idx]; }

  const SolutionBase& so_solution(usize objective) const override final {
    __goblin_runtime_assert(objective < _so_solutions.size());
    return _so_solutions[objective];
  }

  void clear() override final {
    _so_solutions.clear();
    _solutions.clear();
  };

  std::tuple<bool, bool> update_archive(const SolutionBase& solution, bool strict) override final;
  const ArchiveFitnessBase& fitness() const override final { return _fitness; };

 private:
  // is_so_elite, is_dominated
  std::tuple<bool, bool> update_so_solutions(const SolutionBase& solution);

  DefaultSolutionSet _so_solutions;
  DefaultSolutionSet _solutions;
  const ArchiveFitnessBase& _fitness;
};

class AdaptiveGridArchive : public ArchiveBase {
 public:
  AdaptiveGridArchive(const ArchiveFitnessBase& fitness,
                      usize capacity,
                      float max_deviation = 0.25,
                      usize max_iterations = 25,
                      std::optional<CType> initial_discretization = std::nullopt)
      : _fitness(fitness),
        _capacity(capacity),
        _max_deviation(max_deviation),
        _c_min(static_cast<usize>(capacity * (1.0 - max_deviation))),
        _c_max(static_cast<usize>(capacity * (1.0 + max_deviation))),
        _max_iterations(max_iterations),
        // this is the maximum resolution of binary search with 25 steps...
        _max_resolution(std::pow(2.0, max_iterations)),
        _initial_discretization(initial_discretization) {
    __goblin_runtime_assert(capacity > 0);
    __goblin_runtime_assert(max_deviation > 0.0);
    __goblin_runtime_assert(!initial_discretization.has_value() || initial_discretization.value() > CType(0.0));
  };

  std::unique_ptr<ArchiveBase> clone() const override final { return std::make_unique<AdaptiveGridArchive>(*this); };

  usize size() const override final { return _solutions.size(); };
  const SolutionBase& operator[](usize idx) const override final { return _solutions[idx]; };
  SolutionBase& unsafe_at(usize idx) override final { return _solutions[idx]; };

  void clear() override final {
    _so_solutions.clear();
    _solutions.clear();
  };

  const SolutionBase& so_solution(usize objective) const override final { return _so_solutions[objective]; };

  std::tuple<bool, bool> update_archive(const SolutionBase& solution, bool strict) override final;

  void adapt() override final;

  const ArchiveFitnessBase& fitness() const override final { return _fitness; };

 private:
  bool same_box(const SolutionBase& lhs, const SolutionBase& rhs);
  bool update_so_solutions(const SolutionBase& solution);

  const ArchiveFitnessBase& _fitness;
  [[maybe_unused]] usize _capacity;
  [[maybe_unused]] float _max_deviation;
  usize _c_min;
  usize _c_max;
  usize _max_iterations;
  CType _max_resolution;

  std::optional<CType> _initial_discretization;
  std::vector<CType> _discretization;

  DefaultSolutionSet _so_solutions;
  DefaultSolutionSet _solutions;
};

};  // namespace goblin

#endif /* _GOBLIN_LIB_ARCHIVE_H */

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                       goblin/lib/budget.h included by goblin.h                                               //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef _GOBLIN_LIB_BUDGET_H
#define _GOBLIN_LIB_BUDGET_H

#include <chrono>



namespace goblin {
enum struct TerminationStatus : u8 {
  TimeLimitReached,
  GenerationLimitReached,
  EvaluationLimitReached,
  TargetReached,
  Converged,
  Aborted,
  Running
};

inline constexpr std::string_view format_as(const TerminationStatus& s) noexcept {
  switch (s) {
    case TerminationStatus::TimeLimitReached:
      return std::string_view{"TimeLimitReached"};
    case TerminationStatus::GenerationLimitReached:
      return std::string_view{"GenerationLimitReached"};
    case TerminationStatus::EvaluationLimitReached:
      return std::string_view{"EvaluationLimitReached"};
    case TerminationStatus::TargetReached:
      return std::string_view{"TargetReached"};
    case TerminationStatus::Converged:
      return std::string_view{"Converged"};
    case TerminationStatus::Aborted:
      return std::string_view{"Aborted"};
    default:
      return std::string_view{"Running"};
  };
}

inline std::ostream& operator<<(std::ostream& os, TerminationStatus s) {
  return os << format_as(s);
}

struct Budget {
  std::optional<u64> max_evaluations = std::nullopt;
  std::optional<u64> max_generations = std::nullopt;
  std::optional<std::chrono::nanoseconds> max_time = std::nullopt;
  std::optional<std::function<bool()>> termination_callback = std::nullopt;

  Budget(std::optional<u64> max_evaluations = std::nullopt,
         std::optional<u64> max_generations = std::nullopt,
         std::optional<u64> max_time_seconds = std::nullopt,
         std::optional<std::chrono::nanoseconds> max_duration = std::nullopt,
         std::optional<std::function<bool()>> termination_callback = std::nullopt)
      : max_evaluations(max_evaluations),
        max_generations(max_generations),
        max_time(max_time_seconds.has_value() ? std::make_optional<std::chrono::seconds>(max_time_seconds.value())
                                              : max_duration),
        termination_callback(termination_callback) {};

  std::optional<TerminationStatus> constexpr exhausted(usize generations,
                                                       usize evaluations,
                                                       std::chrono::nanoseconds elapsed) const {
    if (max_generations.has_value() && generations > max_generations.value()) {
      return std::make_optional(TerminationStatus::GenerationLimitReached);
    }
    if (max_evaluations.has_value() && evaluations >= max_evaluations.value()) {
      return std::make_optional(TerminationStatus::EvaluationLimitReached);
    }
    if (max_time.has_value() && elapsed >= max_time.value()) {
      return std::make_optional(TerminationStatus::TimeLimitReached);
    }

    return std::nullopt;
  }

  std::optional<TerminationStatus> constexpr exhausted_or_external_criterion_met(
      usize generations,
      usize evaluations,
      std::chrono::nanoseconds elapsed) const {
    if (termination_callback.has_value() && termination_callback.value()()) {
      return std::make_optional(TerminationStatus::Aborted);
    }

    return exhausted(generations, evaluations, elapsed);
  }
};
};  // namespace goblin

#endif /* _GOBLIN_LIB_BUDGET_H */

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                       goblin/lib/instance.h included by goblin.h                                             //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef _GOBLIN_LIB_INSTANCE_H
#define _GOBLIN_LIB_INSTANCE_H

#include <cstddef>


namespace goblin {

class CacheKey {
 public:
  explicit CacheKey(const std::string& key);
  CacheKey(std::vector<std::byte>&& key);

  bool operator==(const CacheKey& other) const { return hash_value_ == other.hash_value_ && key_ == other.key_; };

 private:
  friend struct std::hash<CacheKey>;

  std::vector<std::byte> key_;
  usize hash_value_;
};

class InstanceBase {
 public:
  usize num_objectives() const { return fitness().num_objectives(); };

  usize num_discrete() const { return discrete_domain_sizes().size(); };
  virtual CRef<Vec<DType>> discrete_domain_sizes() const = 0;

  usize num_continuous() const { return continuous_lower_bounds().size(); };
  virtual CRef<Vec<CType>> continuous_lower_bounds() const = 0;
  virtual CRef<Vec<CType>> continuous_upper_bounds() const = 0;

  virtual CRef<Vec<CType>> continuous_init_lower_bounds() const = 0;
  virtual CRef<Vec<CType>> continuous_init_upper_bounds() const = 0;

  // TODO support ordinal discrete spaces (e.g. bool per discrete to indicate categorical/ordinal)
  // TODO support permutation spaces (e.g. bool per continuous variable to indicate continuous/random keys)

  virtual void evaluate(Rng& rng, SolutionSetBase& solutions, const std::span<const usize>& indices) = 0;
  virtual void evaluate_partial(Rng& rng,
                                SolutionSetBase& solutions,
                                SolutionSetBase& parents,
                                const std::vector<const Subset*>& subsets,
                                const std::span<const usize>& indices) {
    evaluate(rng, solutions, indices);
  };

  void evaluate_solutions(SolutionSetBase& solutions, std::optional<u64> seed = std::nullopt) {
    Rng rng = seeded_rng(seed);
    std::vector<usize> indices(solutions.size());
    std::iota(indices.begin(), indices.end(), 0);
    evaluate(rng, solutions, indices);
  };

  /// Possibly adapts the problem in some way that may require re-evaluating any elites stored thus far (indicated by
  /// the return value)
  virtual bool adapt(Rng& rng) { return false; };

  /// Returns the gradient for each index of indices (row) and continuous variable (column) with respect to the
  /// optimization goal. The number of evaluations performed to calculate the gradients are added to `evaluations`;
  ///
  /// Note: The default implementation assumes that the passes subsets contain all active indices and the finite
  /// difference approach used requires up to `2 * indices.size() * num_continuous()` evaluations.
  virtual Mat<CType> gradients(Rng& rng,
                               SolutionSetBase& solutions,
                               SolutionSetBase& parents,
                               const std::vector<const Subset*>& subsets,
                               const std::span<const usize>& indices,
                               u64& evaluations);

  /// Performs gradient optimization steps of the continuous variables and returns the
  /// subset of `indices` that changed together with the number of evaluations performed.
  /// Guarantees that the quality of all solutions is up-to-date.
  virtual std::tuple<std::vector<usize>, u64> gradient_steps(Rng& rng,
                                                             SolutionSetBase& solutions,
                                                             SolutionSetBase& parents,
                                                             const std::span<const usize>& indices,
                                                             usize num_steps);

  virtual void add_random(Rng& rng, SolutionSetBase& solutions, usize count) const = 0;

  virtual const FitnessBase& fitness() const = 0;

  /// The fitness the archive should be use, in case the archive should record the front in more objectives than what
  /// should be optimized (as per https://arxiv.org/abs/2507.03777v1)
  ///
  /// Note: What `fitness()` optimizes must always be a compatible subset of what `archive_fitness()` optimizes.
  virtual const ArchiveFitnessBase& archive_fitness() const = 0;

  // // corresponds to e.g. ERCs / one constant per edge in GP
  // virtual bool always_inherit_continuous() const { return false; };

  /// The offspring inherits a subset of the decision variables from the donor, returning true if there was a change to
  /// the active variables and an evaluation is needed, possibly with problem specific modifications.
  ///
  /// Returns a `(any_active_changed, anything_changed)` tuple
  virtual std::tuple<bool, bool> inherit_discrete(SolutionBase& offspring,
                                                  const SolutionBase& donor,
                                                  const Subset& subset) const {
    bool any_active_changed = false, anything_changed = false;

    for (usize i : subset.discrete) {
      if (offspring.discrete_values()(i) != donor.discrete_values()(i)) {
        any_active_changed |= offspring.discrete_active()(i);
        anything_changed = true;
        offspring.discrete_values()(i) = donor.discrete_values()(i);
      }
    }
    return std::make_tuple(any_active_changed, anything_changed);
  };

  // useful for discrete linkage learning in GP
  // - needed to be able to perform constant binning
  // - needed to erase constant pool references
  virtual std::optional<CType> as_continuous(const SolutionBase& solution, usize discrete_index) const {
    return std::nullopt;
  };

  virtual bool target_reached(const ArchiveBase& archive) const { return false; };

  virtual void log_header(std::ostream& os) const {
    os << "values,";
    archive_fitness().log_header(os);
  };

  virtual void log_solution(std::ostream& os, const SolutionBase& solution) const {
    os << "([";
    for (usize i = 0; i < solution.num_discrete(); i++) {
      if (i > 0) {
        os << ",";
      }
      if (solution.discrete_active()(i)) {
// fmt to alwyas use the decimal instead of the ascii byte value for (unsigned)
// chars
#ifdef __cpp_lib_print
        std::print(os,
#else
        os << std::format(
#endif
                   "{:d}", solution.discrete_values()(i));
      } else {
        os << "█";
      }
    }
    os << "],[";
    for (usize i = 0; i < solution.num_continuous(); i++) {
      if (i > 0) {
        os << ",";
      }
      if (solution.continuous_active()(i)) {
        os << solution.continuous_values()(i);
      } else {
        os << "█";
      }
    }
    os << "])";
  };

  virtual void log(std::ostream& os, const SolutionBase& solution) const {
    os << '"';
    log_solution(os, solution);
    os << "\",";
    archive_fitness().log(os, solution.quality());
  };

  std::string format_solution(const SolutionBase& solution) const {
    std::stringstream ss;
    log_solution(ss, solution);
    return ss.str();
  };

  /// By returning std::nullopt, caching can be disabled on a per solution basis
  virtual std::optional<CacheKey> solution_cache_key(const SolutionBase& solution) const {
    return CacheKey(format_solution(solution));
  };

  u64 reevaluate_and_rebuild_archive(Rng& rng, ArchiveBase& archive) {
    // 1. put all solutions into a solutionset
    AoSSet solutions;
    std::vector<usize> indices;
    indices.reserve(archive.size());
    for (usize i = 0; i < archive.size(); i++) {
      solutions.add(archive[i]);
      indices.push_back(i);
    }
    // 2. evaluate them
    evaluate(rng, solutions, indices);
    // 3. re-build the archive
    archive.clear();
    for (usize i = 0; i < solutions.size(); i++) {
      archive.update(solutions[i], true);
    }

    return indices.size();
  };

  virtual const InstanceBase& unwrap() const { return *this; }

  virtual ~InstanceBase() = default;
};

/// Intermediate class for wrapping instances that by default forwards everything to the actual inner method. Still
/// allows accidentally not overwriting some methods, but at least defaults to the behaviour of the wrapped instance
/// instead of the default implementations of virtual methods.
class WrappedInstance : public InstanceBase {
 public:
  WrappedInstance(InstanceBase& instance) : inner(instance) {};

  CRef<Vec<DType>> discrete_domain_sizes() const override { return inner.discrete_domain_sizes(); }

  CRef<Vec<CType>> continuous_lower_bounds() const override { return inner.continuous_lower_bounds(); }
  CRef<Vec<CType>> continuous_upper_bounds() const override { return inner.continuous_upper_bounds(); }

  CRef<Vec<CType>> continuous_init_lower_bounds() const override { return inner.continuous_init_lower_bounds(); }
  CRef<Vec<CType>> continuous_init_upper_bounds() const override { return inner.continuous_init_upper_bounds(); }

  void evaluate(Rng& rng, SolutionSetBase& solutions, const std::span<const usize>& indices) override {
    return inner.evaluate(rng, solutions, indices);
  };
  void evaluate_partial(Rng& rng,
                        SolutionSetBase& solutions,
                        SolutionSetBase& parents,
                        const std::vector<const Subset*>& subsets,
                        const std::span<const usize>& indices) override {
    return inner.evaluate_partial(rng, solutions, parents, subsets, indices);
  };

  bool adapt(Rng& rng) override { return inner.adapt(rng); };

  Mat<CType> gradients(Rng& rng,
                       SolutionSetBase& solutions,
                       SolutionSetBase& parents,
                       const std::vector<const Subset*>& subsets,
                       const std::span<const usize>& indices,
                       u64& evaluations) override {
    return inner.gradients(rng, solutions, parents, subsets, indices, evaluations);
  }

  std::tuple<std::vector<usize>, u64> gradient_steps(Rng& rng,
                                                     SolutionSetBase& solutions,
                                                     SolutionSetBase& parents,
                                                     const std::span<const usize>& indices,
                                                     usize num_steps) override {
    return inner.gradient_steps(rng, solutions, parents, indices, num_steps);
  };

  void add_random(Rng& rng, SolutionSetBase& solutions, usize count) const override {
    return inner.add_random(rng, solutions, count);
  };

  const FitnessBase& fitness() const override { return inner.fitness(); };
  const ArchiveFitnessBase& archive_fitness() const override { return inner.archive_fitness(); };

  std::tuple<bool, bool> inherit_discrete(SolutionBase& offspring,
                                          const SolutionBase& donor,
                                          const Subset& subset) const override {
    return inner.inherit_discrete(offspring, donor, subset);
  };

  std::optional<CType> as_continuous(const SolutionBase& solution, usize discrete_index) const override {
    return inner.as_continuous(solution, discrete_index);
  };

  bool target_reached(const ArchiveBase& archive) const override { return inner.target_reached(archive); };

  void log_header(std::ostream& os) const override { return inner.log_header(os); };

  void log_solution(std::ostream& os, const SolutionBase& solution) const override {
    return inner.log_solution(os, solution);
  };

  void log(std::ostream& os, const SolutionBase& solution) const override { return inner.log(os, solution); };

  std::optional<CacheKey> solution_cache_key(const SolutionBase& solution) const override {
    return inner.solution_cache_key(solution);
  };

  const InstanceBase& unwrap() const override { return inner.unwrap(); }

  virtual ~WrappedInstance() = default;

 protected:
  InstanceBase& inner;
};

class CachedInstanceBase : public WrappedInstance {
 public:
  CachedInstanceBase(InstanceBase& instance) : WrappedInstance(instance) {};

  virtual usize hit_count() const = 0;
  virtual usize miss_count() const = 0;
  virtual usize access_count() const = 0;
  virtual usize entry_invalidation_count() const = 0;
  virtual usize cache_invalidation_count() const = 0;
  virtual usize evicted_count() const = 0;

  virtual CType hit_ratio() const = 0;
  virtual CType miss_ratio() const = 0;
  /// Proportion of cache entries used w.r.t. maximum size
  virtual CType utilization() const = 0;
  // TODO expose cache api & stats...
  virtual ~CachedInstanceBase() = default;
};

std::shared_ptr<CachedInstanceBase> Cached(std::shared_ptr<InstanceBase> problem,
                                           usize cache_size = 10000,
                                           std::string cache_policy = "lru");

};  // namespace goblin

#endif /* _GOBLIN_LIB_INSTANCE_H */

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                       goblin/lib/algorithms/upgma.h included by goblin.h                                     //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef _GOBLIN_LIB_UPGMA_H
#define _GOBLIN_LIB_UPGMA_H



namespace goblin {

/// Arthur's UPGMA implementation from
/// https://github.com/8uurg/Impact-of-Asynchrony-on-MBEAs/blob/07083732629661c7efccddb4c1b15e69cc46ff2e/EALib/src/gomea.cpp#L216
/// This implementation was chosen because it returns the whole dendrogram, it
/// is commented better and Arthur's code was easier to adapt to work with Eigen
/// matrices directly...
class UPGMA {
 public:
  struct Merge {
    usize left;
    usize right;
    CType distance;
    usize size;
  };

  /// UPGMA that clusters based on similarity (higher = closer), not distance
  /// (lower = closer)
  static std::vector<Merge> cluster(Rng& rng, Mat<CType>& similarity) {
    std::vector<Merge> merges;
    usize n = similarity.rows();
    // Every merge reduces the number of elements left by one.
    // As such there are n - 1 such merges to end up at the root.
    merges.reserve(n - 1);

    // The algorithm implemented here is named NN-chain, or nearest-neighbor
    // chain. And is a fast O(n^2) hierarchical clustering algorithm.

    // The first important implementation detail here is that we use
    // representatives, as we are merging nodes, the in the current state each
    // variable only appears once. This results in each variable uniquely
    // mapping to a subset at a point in time. As such we choose to represent
    // each subset by its smallest element contained within. Annoyingly enough,
    // this does mean that translating the output requires some care.

    // The chain is initially empty, first element will be picked randomly as
    // well.
    std::vector<usize> nn_chain;
    nn_chain.reserve(n);

    // We start off with the univariate marginal product
    // -- i.e. all variables are in subsets on their own.
    std::vector<usize> node_sizes(n);
    std::fill(node_sizes.begin(), node_sizes.end(), 1);
    // - keep track of unique indices as well so we can output a structure
    // similar to that provided by scipy.
    std::vector<usize> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    // and all of them are remaining.
    std::vector<usize> remaining(n);
    std::iota(remaining.begin(), remaining.end(), 0);

    // Another implementation detail, important for use in Evolutionary
    // Algorithms, is that ties should be broken randomly, as this results in
    // variation in the tree when the metric used is fixed and ties occur. i.e.
    // due to convergence or coincidence. Furthermore: we do not want to favor
    // one pair over another. dependent on the situation.
    std::shuffle(remaining.begin(), remaining.end(), rng);

    // A common operation in this algorithm is finding the nearest neighbor of
    // the current end of the chain to the remaining elements.
    //  (Sidenote: in our case this is the farthest element, as we are working
    //  with similarities, not distances)
    auto next = [&]() {
      usize idx = 0;
      CType s = -std::numeric_limits<CType>::infinity();
      usize leader = nn_chain.back();
      for (usize remaining_idx = 0; remaining_idx < remaining.size(); ++remaining_idx) {
        usize other = remaining[remaining_idx];
        if (similarity(other, leader) > s) {
          s = similarity(other, leader);
          idx = remaining_idx;
        }
      }
      return idx;
    };

    // another aspect than can be modified is the update of distances. in this
    // case we use UPGMA.
    auto merged_distance = [&](usize to_merge_i, usize to_merge_j, usize k) {
      usize size_i = node_sizes[to_merge_i];
      usize size_j = node_sizes[to_merge_j];
      usize size_k = node_sizes[k];
      return mergeUPGMA(similarity(to_merge_i, to_merge_j), similarity(to_merge_i, k), similarity(to_merge_j, k),
                        size_i, size_j, size_k);
    };

    // What is the next index to use after merging?
    usize next_merge_index = n;

    // While there is more than one subset remaining.
    while (remaining.size() > 1 || nn_chain.size() > 0) {
      if (nn_chain.size() == 0) {
        // To get started, pick the last index of the remaining list.
        // This is random, 'remaining' should be shuffled!
        usize leader = remaining.back();
        remaining.pop_back();
        nn_chain.push_back(leader);
      }
      if (nn_chain.size() == 1) {
        // With only one element, the next one in the chain is trivial
        usize remaining_idx = next();
        usize leader = remaining[remaining_idx];
        // Quickly remove the new item from remaining.
        std::swap(remaining[remaining_idx], remaining.back());
        remaining.pop_back();
        nn_chain.push_back(leader);
      }

      // In all other cases we need to check if we are closer to the previous
      // element than the next remaining element. If we are: we merge, otherwise
      // the next item is added to the chain.

      usize leader = nn_chain.back();
      usize previous = nn_chain[nn_chain.size() - 2];
      CType distance_previous = similarity(leader, previous);

      // If there are no elements remaining: just merge!
      // this can occur if the tree is more like a list and we started exactly
      // in the wrong node. or if we are merging the last two nodes.
      if (remaining.size() != 0) {
        usize remaining_idx = next();
        usize next_remaining = remaining[remaining_idx];
        CType distance_next_remaining = similarity(leader, next_remaining);

        // comparison is flipped here as we are working with similarities
        // instead of disimilarities. normally it would be
        // `distance_next_remaining < distance_previous`
        if (distance_next_remaining > distance_previous) {
          // next element is closer, add it to the chain.
          // Quickly remove the new item from remaining.
          std::swap(remaining[remaining_idx], remaining.back());
          remaining.pop_back();
          nn_chain.push_back(next_remaining);
          // and start back from the top -- technically going back to line 193
          // would work better as the if statements are always false from this
          // point onwards, unless a merge was performed.
          continue;
        }
      }

      // we have found a pair of mutual nearest neighbors: leader and previous.
      // now we need to merge them!
      // Determine the representative.
      usize representative = std::min(leader, previous);
      usize not_representative = std::max(leader, previous);

      // Update distances within the chain.
      for (usize i = 0; i < nn_chain.size() - 2; ++i) {
        usize other = nn_chain[i];
        similarity(representative, other) = merged_distance(leader, previous, other);
      }
      // Update distances for those remaining.
      for (usize i = 0; i < remaining.size(); ++i) {
        usize other = remaining[i];
        similarity(representative, other) = merged_distance(leader, previous, other);
      }

      // Update the size of the resultant node.
      node_sizes[representative] += node_sizes[not_representative];

      // Keep track of the merges
      merges.push_back(Merge{
          .left = indices[representative],
          .right = indices[not_representative],
          .distance = distance_previous,
          .size = node_sizes[representative],
      });

      // get a merge index for this element.
      indices[representative] = next_merge_index++;

      // Remove the last two items
      nn_chain.pop_back();
      nn_chain.pop_back();

      // add the current element back to remaining.
      remaining.push_back(representative);
      // and shuffle this element as well
      std::uniform_int_distribution<usize> idx(0, remaining.size() - 1);
      std::swap(remaining[idx(rng)], remaining.back());
    }

    return merges;
  };

 private:
  inline static CType mergeUPGMA(CType /* distance_ij */,
                                 CType distance_ik,
                                 CType distance_jk,
                                 usize size_i,
                                 usize size_j,
                                 usize /* size_k */) {
    CType weighted = static_cast<CType>(size_i) * distance_ik + static_cast<CType>(size_j) * distance_jk;
    return weighted / static_cast<CType>(size_i + size_j);
  };
};

};  // namespace goblin

#endif /* _GOBLIN_LIB_UPGMA_H */

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                       goblin/lib/algorithms/subset_selection.h included by goblin.h                          //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef _GOBLIN_LIB_SUBSET_SELECTION_H
#define _GOBLIN_LIB_SUBSET_SELECTION_H



namespace goblin {

template <typename DST>
inline std::tuple<std::vector<usize>, std::vector<usize>>
greedy_scattered_subset_selection(DST&& distance, usize pool_size, usize target_size, std::vector<usize>&& selection) {
  __goblin_runtime_assert(target_size <= pool_size);

  std::vector<usize> remaining(pool_size);
  std::iota(remaining.begin(), remaining.end(), 0);

  std::sort(selection.begin(), selection.end(), std::greater{});
  for (auto i : selection) {
    __goblin_runtime_assert(i < pool_size);
    std::swap(remaining[i], remaining.back());
    remaining.pop_back();
  }

  using dist_t = std::remove_cvref_t<decltype(std::declval<DST>()(0, 0))>;
  std::vector<dist_t> dist_to_selection(remaining.size(), std::numeric_limits<dist_t>::max());
  for (usize s = 0; s < selection.size() - 1; s++) {
    for (usize i = 0; i < remaining.size(); i++) {
      dist_to_selection[i] = std::min(dist_to_selection[i], distance(selection[s], remaining[i]));
    }
  }

  while (selection.size() < target_size) {
    // update distance to selection and get furthest
    usize furthest = 0;
    for (usize i = 0; i < remaining.size(); i++) {
      dist_to_selection[i] = std::min(dist_to_selection[i], distance(selection.back(), remaining[i]));
      if (dist_to_selection[i] > dist_to_selection[furthest]) {
        furthest = i;
      }
    }

    // add furthest element to the selection
    std::swap(dist_to_selection[furthest], dist_to_selection.back());
    std::swap(remaining[furthest], remaining.back());
    selection.push_back(remaining.back());
    remaining.pop_back();
  }

  return std::make_tuple(selection, remaining);
};

/// Given a list of remaining options and an initial solution,
/// the furthest remaining solution is repeatedly selected until the target size
/// is reached
template <typename DST>
inline std::tuple<std::vector<usize>, std::vector<usize>> greedy_scattered_subset_selection(DST&& distance,
                                                                                            usize pool_size,
                                                                                            usize target_size,
                                                                                            usize initial) {
  return greedy_scattered_subset_selection(distance, pool_size, target_size, std::move(std::vector<usize>{initial}));
};

};  // namespace goblin

#endif /* _GOBLIN_LIB_SUBSET_SELECTION_H */

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                       goblin/lib/algorithms/mo.h included by goblin.h                                        //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef _GOBLIN_LIB_ALGORITHMS_MO_H
#define _GOBLIN_LIB_ALGORITHMS_MO_H

#include <set>
#include <variant>


namespace goblin {

template <typename D>
inline std::tuple<std::vector<usize>, std::vector<std::set<usize>>> non_dominated_sorting(D&& cmp, usize pool_size) {
  std::vector<std::set<usize>> dominations(pool_size);
  std::vector<usize> domination_count(pool_size, 0);
  std::vector<usize> rank(pool_size);
  std::vector<std::set<usize>> fronts(1);

  for (usize i = 0; i < pool_size; i++) {
    for (usize j = 0; j < i; j++) {
      Ordering o = cmp(i, j);
      if (o == Ordering::Better) {
        dominations[i].insert(j);
        domination_count[j]++;
      } else if (o == Ordering::Worse) {
        dominations[j].insert(i);
        domination_count[i]++;
      }
    }
    if (domination_count[i] == 0) {
      rank[i] = 0;
      fronts.back().insert(i);
    }
  }

  while (!fronts.back().empty()) {
    std::set<usize> front;
    for (usize p : fronts.back()) {
      for (usize q : dominations[p]) {
        domination_count[q]--;
        if (domination_count[q] == 0) {
          rank[q] = fronts.size();
          front.insert(q);
        }
      }
    }
    fronts.push_back(front);
  }

  return std::make_tuple(rank, fronts);
};

/// 2D Hypervolume
///
/// Uses an O(n * log(n)) sweeping line approach
/// 1. Points are sorted in one dimension
/// 2. The areas of the resulting rectangles are added together
template <typename SD, typename D2R>
inline CType hypervolume2D_impl(SD&& signed_distance, D2R&& distance_to_reference, usize num_points) {
  if (num_points == 0) {
    return 0.0;
  } else if (num_points == 1) {
    return std::abs(distance_to_reference(0, 0) * distance_to_reference(0, 1));
  }

  std::vector<usize> points(num_points);
  std::iota(points.begin(), points.end(), 0);

  std::sort(points.begin(), points.end(),
            [&](const usize lhs, const usize rhs) { return signed_distance(lhs, rhs, 1) <= 0.0; });

  CType hv = 0.0;
  CType w = distance_to_reference(points[0], 0);
  for (usize i = 1; i < points.size(); i++) {
    hv += signed_distance(points[i], points[i - 1], 1) * w;
    w = std::max(w, distance_to_reference(points[i], 0));
  }
  return hv + distance_to_reference(points.back(), 1) * w;
};

template <typename SolutionSetLike>
inline CType hypervolume2D_dispatch(const SolutionSetLike& solutions,
                                    const FitnessBase& fitness,
                                    const QualityBase& reference_point) {
  return hypervolume2D_impl(
      [&](const usize lhs, const usize rhs, const usize objective) {
        auto dist = fitness.distance(solutions[lhs].quality(), solutions[rhs].quality(), objective);
        auto lhs_better =
            fitness.cmp(solutions[lhs].quality(), solutions[rhs].quality(), objective) == Ordering::Better;
        return lhs_better ? -dist : dist;
      },
      [&](const usize idx, const usize objective) {
        return fitness.distance(solutions[idx].quality(), reference_point, objective);
      },
      solutions.size());
};

inline CType hypervolume2D(const ArchiveBase& solutions,
                           const FitnessBase& fitness,
                           const QualityBase& reference_point) {
  return hypervolume2D_dispatch(solutions, fitness, reference_point);
};
inline CType hypervolume2D(const SolutionSetBase& solutions,
                           const FitnessBase& fitness,
                           const QualityBase& reference_point) {
  return hypervolume2D_dispatch(solutions, fitness, reference_point);
};

inline CType hypervolume2D(const Arr2D<CType> points, const Array<CType> reference_point) {
  return hypervolume2D_impl([&](const usize lhs, const usize rhs,
                                const usize objective) { return points(lhs, objective) - points(rhs, objective); },
                            [&](const usize idx, const usize objective) {
                              return std::abs(points(idx, objective) - reference_point(objective));
                            },
                            points.rows());
};

};  // namespace goblin

#endif /* _GOBLIN_LIB_ALGORITHMS_MO_H */

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                       goblin/lib/linkage_model.h included by goblin.h                                        //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef _GOBLIN_LIB_LINKAGE_MODEL_H
#define _GOBLIN_LIB_LINKAGE_MODEL_H




namespace goblin {
enum class VariableSet : u8 { Discrete = 0b01, Continuous = 0b10, Mixed = 0b11 };

inline constexpr bool operator&(VariableSet lhs, VariableSet rhs) noexcept {
  return static_cast<bool>(static_cast<u8>(lhs) & static_cast<u8>(rhs));
};

Mat<CType> estimate_entropy(const InstanceBase& problem,
                            const SolutionSetBase& solutions,
                            const std::span<const usize> indices,
                            const std::span<const usize> subset,
                            const std::string& intron_strategy,
                            bool merge_continuous,
                            std::optional<usize> num_continuous_bins);

class LinkageModelBase {
 public:
  // LinkageModelBase() = default;
  // LinkageModelBase(const LinkageModelBase&) = delete;
  // LinkageModelBase(LinkageModelBase&&) = delete;

  // LinkageModelBase& operator=(const LinkageModelBase&) = delete;
  // LinkageModelBase& operator=(LinkageModelBase&&) = delete;

  virtual void init(Rng& rng, InstanceBase& problem, SolutionSetBase& solutions, VariableSet variables) = 0;
  virtual FOS subsets(Rng& rng,
                      InstanceBase& problem,
                      SolutionSetBase& solutions,
                      const std::span<const usize> indices,
                      std::optional<std::reference_wrapper<const Mat<CType>>> covariance) const = 0;

  virtual bool is_static() const { return false; };

  virtual std::unique_ptr<LinkageModelBase> clone() const = 0;

  virtual ~LinkageModelBase() = default;  //{};
};

class UnivariateFOS final : public LinkageModelBase {
 public:
  UnivariateFOS(std::optional<Subset> subset = std::nullopt) : subset(subset.value_or(Subset{})) {}

  std::unique_ptr<LinkageModelBase> clone() const override final { return std::make_unique<UnivariateFOS>(*this); }

  void init(Rng& rng, InstanceBase& problem, SolutionSetBase& solutions, VariableSet variables) override final {
    if (subset.empty()) {
      if (variables & VariableSet::Discrete) {
        subset.discrete.reserve(problem.num_discrete());
        for (usize i = 0; i < problem.num_discrete(); i++) {
          subset.discrete.push_back(i);
        }
      }
      if (variables & VariableSet::Continuous) {
        subset.continuous.reserve(problem.num_continuous());
        for (usize i = 0; i < problem.num_continuous(); i++) {
          subset.continuous.push_back(i);
        }
      }
    }
  }

  FOS subsets(Rng& rng,
              InstanceBase& problem,
              SolutionSetBase& solutions,
              const std::span<const usize> indices,
              std::optional<std::reference_wrapper<const Mat<CType>>> covariance = std::nullopt) const override final {
    FOS fos;
    fos.resize(subset.size());
    {
      usize i = 0;
      for (usize j : subset.discrete) {
        fos[i++].discrete.push_back(j);
      }
      for (usize j : subset.continuous) {
        fos[i++].continuous.push_back(j);
      }
    }
    return fos;
  };

  bool is_static() const override final { return true; };

 private:
  Subset subset;
};

class FullFOS final : public LinkageModelBase {
 public:
  FullFOS(std::optional<Subset> subset = std::nullopt) : subset(subset.value_or(Subset{})) {}

  std::unique_ptr<LinkageModelBase> clone() const override final { return std::make_unique<FullFOS>(*this); }

  void init(Rng& rng, InstanceBase& problem, SolutionSetBase& solutions, VariableSet variables) override final {
    if (subset.empty()) {
      if (variables & VariableSet::Discrete) {
        subset.discrete.reserve(problem.num_discrete());
        for (usize i = 0; i < problem.num_discrete(); i++) {
          subset.discrete.push_back(i);
        }
      }
      if (variables & VariableSet::Continuous) {
        subset.continuous.reserve(problem.num_continuous());
        for (usize i = 0; i < problem.num_continuous(); i++) {
          subset.continuous.push_back(i);
        }
      }
    }
  }

  FOS subsets(Rng& rng,
              InstanceBase& problem,
              SolutionSetBase& solutions,
              const std::span<const usize> indices,
              std::optional<std::reference_wrapper<const Mat<CType>>> covariance = std::nullopt) const override final {
    __goblin_runtime_assert(subset.continuous.size() > 0 ||
                            subset.discrete.size() <
                                problem.num_discrete());  // Full discrete subset probably is a mistake, that just
                                                          // copies solutions instead of performing search...
    return {subset};
  };

  bool is_static() const override final { return true; };

 private:
  Subset subset;
};

class LinkageTreeFOS final : public LinkageModelBase {
 public:
  LinkageTreeFOS(std::string metric = "mi",  // nmi, pearson_r2, random

                 /// The intron strategy determines how knowledge about inactive variables
                 /// is used to modify the estimation of linkage when learning the linkage
                 /// tree
                 /// - none: Ignore introns
                 /// - any_active: introns are marked as such to remove random noise and
                 /// only variable pairs with at least one active variable are considered
                 /// - all_active: Only variable pairs where both variables are active are
                 /// considered
                 /// - mark_only: introns are marked as such to reduce noise, but pairs
                 /// consisting of only introns are still considered
                 std::string intron_strategy = "none",
                 bool merge_continuous = true,
                 std::optional<usize> num_continuous_bins = std::nullopt,
                 std::optional<CType> filter_parent_threshold = std::nullopt,    // 1e-6
                 std::optional<CType> filter_children_threshold = std::nullopt,  // 1 - 1e-6
                 std::optional<bool> filter_root = std::nullopt,  // default is true if discrete, false if
                                                                  // continuous/mixed
                 std::optional<usize> max_subset_size = std::nullopt,
                 bool normalize_initial_linkage_bias = false,
                 std::optional<Subset> subset = std::nullopt,
                 std::optional<Mat<CType>> custom_similarity = std::nullopt,
                 std::optional<CType> eta_custom_similarity = std::nullopt,
                 std::optional<std::string> custom_similarity_agg = std::nullopt,
                 std::optional<std::function<void(CRef<Mat<CType>>)>> similarity_callback = std::nullopt,
                 bool freeze = false)
      : subset(subset.value_or(Subset{})),
        custom_similarity(custom_similarity),
        similarity_callback(similarity_callback),
        metric(metric),
        intron_strategy(intron_strategy),
        num_continuous_bins(num_continuous_bins),
        filter_parent_threshold(filter_parent_threshold),
        filter_children_threshold(filter_children_threshold),
        filter_root(filter_root),
        max_subset_size(max_subset_size),
        eta_custom_similarity(eta_custom_similarity),
        custom_similarity_agg(custom_similarity_agg),
        merge_continuous(merge_continuous),
        normalize_initial_linkage_bias(normalize_initial_linkage_bias),
        freeze(freeze) {
    if (this->eta_custom_similarity.has_value()) {
      if (!(0.0 < this->eta_custom_similarity.value() && this->eta_custom_similarity.value() < 1.0)) {
        throw std::runtime_error("Invalid eta_custom_similarity, must be in (0, 1) if provided.");
      }
    }
  };

  std::unique_ptr<LinkageModelBase> clone() const override final { return std::make_unique<LinkageTreeFOS>(*this); }

  bool is_static() const override final { return freeze; };

  void register_similarity_callback(std::function<void(CRef<Mat<CType>>)>&& similarity_callback) {
    this->similarity_callback = similarity_callback;
  };

  void unregister_similarity_callback() { this->similarity_callback = std::nullopt; };

  void init(Rng& rng, InstanceBase& problem, SolutionSetBase& solutions, VariableSet variables) override final {
    if (subset.empty()) {
      if (variables & VariableSet::Discrete) {
        subset.discrete.reserve(problem.num_discrete());
        for (usize i = 0; i < problem.num_discrete(); i++) {
          subset.discrete.push_back(i);
        }
      }
      if (variables & VariableSet::Continuous) {
        subset.continuous.reserve(problem.num_continuous());
        for (usize i = 0; i < problem.num_continuous(); i++) {
          subset.continuous.push_back(i);
        }
      }
    }

    if (custom_similarity.has_value()) {
      if (static_cast<usize>(custom_similarity.value().rows()) != subset.size() ||
          static_cast<usize>(custom_similarity.value().cols()) != subset.size()) {
        throw std::runtime_error(
            "Custom similarity matrix has invalid dimensions, expected a square matrix corresponding to the subset "
            "size.");
      }
    }

    if (normalize_initial_linkage_bias && variables == VariableSet::Discrete) {
      // __goblin_runtime_assert(intron_strategy == "none");
      // intron awareness and Marco's linkage normalization are incompatible
      // - conditioning the unbiasing matrix to only consider active variables
      // would lead to potential division by 0 issues (you can't normalize no
      // active values) and learning linkage in an intron aware manner likely
      // reduces the bias already
      // TODO - instead of asserting non-zero entropy, I now silently set the
      // multiplier to 1 for such variables/variable pairs -> could work again

      // Assumption is that the initial values are non-uniformally distributed,
      // suggesting linkage where there likely is none. To fix this, the linkage
      // can be normalized to the identity matrix as per
      // https://arxiv.org/pdf/1904.02050v3#section.5
      std::vector<usize> indices(solutions.size());
      std::iota(indices.begin(), indices.end(), 0);
      auto H = estimate_entropy(problem, solutions, indices, subset.discrete, intron_strategy, merge_continuous,
                                num_continuous_bins);

      usize l = subset.discrete.size();
      initial_bias_adjustments.resize(l, l);
      for (usize i = 0; i < l; i++) {
        initial_bias_adjustments(i, i) = H(i, i) > CType(0.0) ? CType(1.0) / H(i, i) : CType(1.0);

        for (usize j = 0; j < i; j++) {
          initial_bias_adjustments(i, j) = H(i, j) > CType(0.0) ? CType(2.0) / H(i, j) : CType(1.0);
          initial_bias_adjustments(j, i) = initial_bias_adjustments(i, j);
        }
      }
    }
  };

  Mat<CType> compute_similarity(
      Rng& rng,
      InstanceBase& problem,
      SolutionSetBase& solutions,
      const std::span<const usize> indices,
      std::optional<std::reference_wrapper<const Mat<CType>>> covariance = std::nullopt) const {
    const isize l = subset.size();
    const bool is_discrete = subset.discrete.size() > 0;
    const bool is_continuous = subset.continuous.size() > 0;

    Mat<CType> similarity;
    if (custom_similarity.has_value() && !eta_custom_similarity.has_value()) {
      similarity = custom_similarity.value();
    } else if (metric == "random") {
      similarity.resize(l, l);
      std::uniform_real_distribution<CType> U(0.0, 1.0);
      for (isize i = 0; i < l; i++) {
        for (isize j = 0; j <= i; j++) {
          similarity(i, j) = U(rng);
          similarity(j, i) = similarity(i, j);
        }
      }
    } else if (is_discrete && is_continuous) {
      similarity.resize(l, l);
      // No mixed subset learning (yet?)
      // https://homepages.cwi.nl/~bosman/publications/2016_learningandexploiting.pdf
      throw std::runtime_error("Mixed linkage learning isn't supported.");
    } else if (is_discrete) {
      similarity = estimate_entropy(problem, solutions, indices, subset.discrete, intron_strategy, merge_continuous,
                                    num_continuous_bins);
      if (normalize_initial_linkage_bias && initial_bias_adjustments.size() > 0) {
        similarity.array() *= initial_bias_adjustments.array();
      }
      entropy2similarity(similarity);
    } else if (is_continuous) {
      similarity.resize(l, l);
      if (!covariance.has_value()) {
        throw std::runtime_error("Continuous linkage learning requires a covariance matrix.");
      } else if (covariance.value().get().rows() != l || covariance.value().get().cols() != l) {
        throw std::runtime_error(
            "Covariance matrix has invalid dimensions, expected a square matrix corresponding to the subset size.");
      }

      Mat<CType> cov = covariance.value();
      for (isize i = 0; i < l; i++) {
        for (isize j = 0; j < i; j++) {
          // https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
          double r = cov(i, j) / (std::sqrt(cov(i, i)) * std::sqrt(cov(j, j)));
          similarity(i, j) = std::log(std::sqrt(1.0 / (1.0 - r * r)));
          similarity(j, i) = similarity(i, j);
        }
      }
    } else {
      throw std::runtime_error("Unknown subset of variables to learn a LT for.");
    }

    if (custom_similarity.has_value() && custom_similarity_agg.has_value()) {
      const auto& agg = custom_similarity_agg.value();
      if (agg == "add") {
        similarity = similarity + custom_similarity.value();
      } else if (agg == "mul") {
        similarity = similarity.array() * custom_similarity.value().array();
      } else if (agg == "max") {
        similarity = similarity.array().max(custom_similarity.value().array());
      } else {
        throw std::runtime_error("Unknown or unsupported similarity aggregation method.");
      }
    } else if (custom_similarity.has_value() && eta_custom_similarity.has_value()) {
      similarity = (1.0 - eta_custom_similarity.value()) * similarity +
                   eta_custom_similarity.value() * custom_similarity.value();
    }

    if (similarity_callback.has_value()) {
      similarity_callback.value()(similarity);
    }
    return similarity;
  };

  FOS learn_lt(Rng& rng, Mat<CType> similarity, bool filter_root_default) const {
    auto merges = UPGMA::cluster(rng, similarity);

    // 3. merge subsets...
    isize l = similarity.rows();
    FOS fos(2 * l - 1);
    {  // init with the univariate FOS
      usize i = 0;
      for (usize j : subset.discrete) {
        fos[i++].discrete.push_back(j);
      }
      for (usize j : subset.continuous) {
        fos[i++].continuous.push_back(j);
      }
    }

    // and add all of the merges performed during clustering as subsets
    usize num_merges = merges.size();
    if (num_merges > 0 && filter_root.value_or(filter_root_default)) {
      num_merges--;
    }
    for (usize i = 0; i < num_merges; i++) {
      if (max_subset_size.has_value() && merges[i].size > max_subset_size.value()) {
        continue;
      }

      // Clustering happens based on similarity, not distance!
      auto merge_similarity = merges[i].distance;

      // https://homepages.cwi.nl/~bosman/publications/2013_moreconciseand.pdf
      // The linkage is effectively 0, so we can assume that the subsets are
      // independent and thus the merged subset is probably not beneficial
      if (filter_parent_threshold.has_value() &&
          merge_similarity <= filter_parent_threshold.value()) {  // no added benefit is assumed
        continue;
      }

      fos[i + l] = fos[merges[i].left].merge(fos[merges[i].right]);

      // https://homepages.cwi.nl/~bosman/publications/2013_moreconciseand.pdf
      // https://arxiv.org/pdf/2109.05259#subsubsection.3.1.3
      // The larger building block is linked stronger (or at least as strong),
      // so we don't break up building blocks by excluding the smaller subsets
      if (filter_children_threshold.has_value() && merge_similarity >= filter_children_threshold.value()) {
        fos[merges[i].left].clear();
        fos[merges[i].right].clear();
      }
    }

    // remove the empty subsets
    std::erase_if(fos, [](const auto& s) { return s.size() == 0; });
    return fos;
  };

  FOS learn_lt_original(Rng& rng, Mat<CType> similarity, bool filter_root_default) const {
    const isize numberOfVariables = subset.size();
    const isize maximumSetSize = max_subset_size.value_or(numberOfVariables);

    Mat<CType> S = similarity;

    auto determineNearestNeighbour = [&](usize index, const std::vector<std::vector<isize>>& mpm) {
      usize result = 0;

      if (result == index)
        result++;

      for (usize i = 1; i < mpm.size(); i++) {
        if (mpm[i].size() > static_cast<usize>(numberOfVariables)) {
          assert(false);
        }
        if (i != index) {
          if (mpm[index].size() + mpm[result].size() > static_cast<usize>(maximumSetSize)) {
            if (mpm[i].size() < mpm[result].size()) {
              result = i;
            }
          } else if (mpm[index].size() + mpm[i].size() <= static_cast<usize>(maximumSetSize)) {
            if ((S(index, i) > S(index, result)) ||
                ((S(index, i) == S(index, result)) && (mpm[i].size() < mpm[result].size()))) {
              result = i;
            }
          }
        }
      }
      return result;
    };

    std::vector<std::vector<isize>> fos_;
    fos_.reserve(2 * numberOfVariables - 1);

    std::vector<isize> mpmFOSMap;
    std::vector<isize> mpmFOSMapNew;

    /* Initialize MPM to the univariate factorization */
    std::vector<isize> order(numberOfVariables);
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), rng);

    std::vector<std::vector<isize>> mpm(numberOfVariables);
    std::vector<std::vector<isize>> mpmNew(numberOfVariables);

    for (isize i = 0; i < numberOfVariables; i++) {
      mpm[i].push_back(order[i]);
    }

    /* Initialize LT to the initial MPM */
    fos_.resize(numberOfVariables);

    isize FOSsIndex = 0;
    for (isize i = 0; i < numberOfVariables; i++) {
      fos_[i] = mpm[i];
      mpmFOSMap.push_back(i);
      FOSsIndex++;
    }

    for (isize i = 0; i < numberOfVariables; ++i) {
      for (isize j = 0; j < numberOfVariables; j++)
        S(i, j) = similarity(mpm[i][0], mpm[j][0]);

      S(i, i) = 0;
    }
    // printf("Initialized similarity matrix. (%.3fs)\n",getTime(t)/1000.0);

    std::vector<usize> NN_chain;
    NN_chain.resize(numberOfVariables + 2);
    isize NN_chain_length = 0;
    bool done = false;
    while (!done) {
      if (NN_chain_length == 0) {
        NN_chain[NN_chain_length] =
            std::uniform_int_distribution<usize>(0, mpm.size() - 1)(rng);  //  utils::rng() % mpm.size();

        NN_chain_length++;
      }

      while (NN_chain_length < 3) {
        NN_chain[NN_chain_length] = determineNearestNeighbour(NN_chain[NN_chain_length - 1], mpm);
        NN_chain_length++;
      }

      while (NN_chain[NN_chain_length - 3] != NN_chain[NN_chain_length - 1]) {
        NN_chain[NN_chain_length] = determineNearestNeighbour(NN_chain[NN_chain_length - 1], mpm);
        if (((S(NN_chain[NN_chain_length - 1], NN_chain[NN_chain_length]) ==
              S(NN_chain[NN_chain_length - 1], NN_chain[NN_chain_length - 2]))) &&
            (NN_chain[NN_chain_length] != NN_chain[NN_chain_length - 2]))
          NN_chain[NN_chain_length] = NN_chain[NN_chain_length - 2];

        NN_chain_length++;
        if (NN_chain_length > numberOfVariables)
          break;
      }

      usize r0 = NN_chain[NN_chain_length - 2];
      usize r1 = NN_chain[NN_chain_length - 1];
      bool skipFOSElement = false;
      // TODO
      // if( filtered )
      // {
      // 	if ( (similarityMeasure == linkage::MI || similarityMeasure == linkage::NMI ) )
      // 	{
      // 		if( S(r1, r0) >= 1-(1e-6))
      // 			skipFOSElement = true;
      // 	}
      // 	else
      // 	{
      // 		if( S(r1, r0) == 0 )
      // 			skipFOSElement = true;
      // 	}
      // }

      if (r1 >= mpm.size() || r0 >= mpm.size() ||
          mpm[r0].size() + mpm[r1].size() > static_cast<usize>(maximumSetSize)) {
        NN_chain_length = 1;
        NN_chain[0] = 0;
        if (maximumSetSize < numberOfVariables) {
          done = true;
          for (usize i = 1; i < mpm.size(); i++) {
            if (mpm[i].size() + mpm[NN_chain[0]].size() <= static_cast<usize>(maximumSetSize)) {
              done = false;
              // printf("%d %d [%d + %d]\n",i,NN_chain[0],mpm[i].size(),mpm[NN_chain[0]].size());
            }
            if (mpm[i].size() < mpm[NN_chain[0]].size())
              NN_chain[0] = i;
          }
          if (done)
            break;
        }
        continue;
      }

      if (r0 > r1) {
        isize rswap = r0;
        r0 = r1;
        r1 = rswap;
      }
      NN_chain_length -= 3;

      if (r1 < mpm.size()) {
        if (mpm[r0].size() + mpm[r1].size() > static_cast<usize>(maximumSetSize)) {
          done = true;
          break;
        }
        std::vector<isize> indices(mpm[r0].size() + mpm[r1].size());

        usize i = 0;
        for (usize j = 0; j < mpm[r0].size(); j++) {
          indices[i] = mpm[r0][j];
          i++;
        }

        for (usize j = 0; j < mpm[r1].size(); j++) {
          indices[i] = mpm[r1][j];
          i++;
        }

        if (!skipFOSElement) {
          fos_.push_back(indices);
          FOSsIndex++;
          assert(fos_.size() == static_cast<usize>(FOSsIndex));
        }

        double mul0 = static_cast<double>(mpm[r0].size()) / static_cast<double>(mpm[r0].size() + mpm[r1].size());
        double mul1 = static_cast<double>(mpm[r1].size()) / static_cast<double>(mpm[r0].size() + mpm[r1].size());
        for (usize i = 0; i < mpm.size(); i++) {
          if ((i != r0) && (i != r1)) {
            S(i, r0) = mul0 * S(i, r0) + mul1 * S(i, r1);
            S(r0, i) = S(i, r0);
          }
        }

        mpmNew.resize(mpm.size() - 1);
        mpmFOSMapNew.resize(mpmFOSMap.size() - 1);
        for (usize i = 0; i < mpmNew.size(); i++) {
          mpmNew[i] = mpm[i];
          mpmFOSMapNew[i] = mpmFOSMap[i];
        }

        mpmNew[r0] = indices;
        mpmFOSMapNew[r0] = FOSsIndex - 1;

        if (r1 < mpm.size() - 1) {
          mpmNew[r1] = mpm[mpm.size() - 1];
          mpmFOSMapNew[r1] = mpmFOSMap[mpm.size() - 1];

          for (usize i = 0; i < r1; i++) {
            S(i, r1) = S(i, mpm.size() - 1);
            S(r1, i) = S(i, r1);
          }

          for (usize j = r1 + 1; j < mpmNew.size(); j++) {
            S(r1, j) = S(j, mpm.size() - 1);
            S(j, r1) = S(r1, j);
          }
        }

        for (i = 0; i < static_cast<usize>(NN_chain_length); i++) {
          if (NN_chain[i] == mpm.size() - 1) {
            NN_chain[i] = r1;
            break;
          }
        }

        mpm = mpmNew;
        mpmFOSMap = mpmFOSMapNew;

        if (!filter_root.value_or(filter_root_default)) {
          if (mpm.size() == 1)
            done = true;
        } else {
          if (mpm.size() == 2)
            done = true;
        }
      }
    }

    // Map subset indices to structs
    FOS fos;
    fos.resize(fos_.size());
    for (usize i = 0; i < fos_.size(); i++) {
      for (usize j : fos_[i]) {
        if (j < subset.discrete.size()) {
          fos[i].discrete.push_back(subset.discrete[j]);
        } else {
          fos[i].continuous.push_back(subset.continuous[j - subset.discrete.size()]);
        }
      }
    }

    return fos;
  };

  FOS subsets(Rng& rng,
              InstanceBase& problem,
              SolutionSetBase& solutions,
              const std::span<const usize> indices,
              std::optional<std::reference_wrapper<const Mat<CType>>> covariance = std::nullopt) const override final {
    // 1. get similarity matrix based on the measure...
    Mat<CType> similarity = compute_similarity(rng, problem, solutions, indices, covariance);
    assert(static_cast<usize>(similarity.rows()) == subset.size() &&
           static_cast<usize>(similarity.cols()) == subset.size());

    // 2. turn that into a FOS by clustering + merging subsets
    bool is_discrete = subset.discrete.size() > 0;
    bool is_continuous = subset.continuous.size() > 0;
    bool filter_root_default = is_discrete && !is_continuous;
    FOS fos = learn_lt(rng, similarity, filter_root_default);
    // FOS fos = learn_lt_original(rng, similarity, filter_root_default);

    // std::println("Population:");
    // for (auto i : indices) {
    //   std::println("  {}", problem.format_solution(solutions[i]));
    // }

    // std::println("Similarity:");
    // for (usize i = 0; i < similarity.rows(); i++) {
    //   for (usize j = 0; j < similarity.cols(); j++) {
    //     std::print("{:>4.2f},", similarity(i, j));
    //   }
    //   std::println("");
    // }

    // std::println("FOS:");
    // usize i = 0;
    // for (auto &s : fos) {
    //   std::println("{:>3d}: {{d: {}, c: {}}}", i++, s.discrete,
    //   s.continuous);
    // }

    return fos;
  };

 private:
  void entropy2similarity(Mat<CType>& H) const {
    // entropy -> MI/NMI
    CType tmp;
    if (metric == "mi") {
      auto& MI = H;
      for (isize i = 0; i < H.rows(); i++) {
        for (isize j = 0; j < i; j++) {
          tmp = H(i, i) + H(j, j) - H(i, j);
          MI(i, j) = tmp;
          MI(j, i) = tmp;
        }
      }
    } else if (metric == "nmi") {
      auto& NMI = H;
      for (isize i = 0; i < H.rows(); i++) {
        for (isize j = 0; j < i; j++) {
          tmp = H(i, j) > 0 ? (((H(i, i) + H(j, j)) / H(i, j)) - CType(1.0)) : CType(0.0);
          NMI(i, j) = tmp;
          NMI(j, i) = tmp;
        }
      }
    } else {
      // TODO raise unknown or unsupported for discrete domain
      __goblin_runtime_assert(metric != metric);
    }
  };

  Subset subset;
  std::optional<Mat<CType>> custom_similarity;
  std::optional<std::function<void(CRef<Mat<CType>>)>> similarity_callback;
  std::string metric;
  std::string intron_strategy;
  std::optional<usize> num_continuous_bins;
  std::optional<CType> filter_parent_threshold;
  std::optional<CType> filter_children_threshold;
  std::optional<bool> filter_root;
  std::optional<usize> max_subset_size;
  std::optional<CType> eta_custom_similarity;
  std::optional<std::string> custom_similarity_agg;
  bool merge_continuous;
  bool normalize_initial_linkage_bias;
  bool freeze;

  Mat<CType> initial_bias_adjustments;
};

class CombinedFOS final : public LinkageModelBase {
 public:
  CombinedFOS(const std::vector<std::unique_ptr<LinkageModelBase>>& linkage_models) {
    models.reserve(linkage_models.size());
    for (usize i = 0; i < linkage_models.size(); i++) {
      models.push_back(linkage_models[i]->clone());
    }
  }

  void add_model(const LinkageModelBase& model) { models.push_back(model.clone()); }

  // Explicitly disallow copies to tell the Python binding generation that
  // a vector of unique pointers cannot be copied
  // (the other option would be to explicitly define a version that clones the data)
  CombinedFOS(const CombinedFOS&) = delete;
  CombinedFOS& operator=(const CombinedFOS&) = delete;

  // But moving is allowed
  CombinedFOS(CombinedFOS&&) = default;
  CombinedFOS& operator=(CombinedFOS&&) = default;

  std::unique_ptr<LinkageModelBase> clone() const override final { return std::make_unique<CombinedFOS>(models); }

  void init(Rng& rng, InstanceBase& problem, SolutionSetBase& solutions, VariableSet variables) override final {
    for (usize i = 0; i < models.size(); i++) {
      models[i]->init(rng, problem, solutions, variables);
    }
  }

  FOS subsets(Rng& rng,
              InstanceBase& problem,
              SolutionSetBase& solutions,
              const std::span<const usize> indices,
              std::optional<std::reference_wrapper<const Mat<CType>>> covariance = std::nullopt) const override final {
    FOS combined;
    for (usize i = 0; i < models.size(); i++) {
      FOS fos = models[i]->subsets(rng, problem, solutions, indices, covariance);

      // remove any duplicates
      usize j = 0;
      while (j < fos.size()) {
        bool is_new = true;
        for (auto& s : combined) {
          if (fos[j] == s) {
            is_new = false;
            std::swap(fos[j], fos.back());
            fos.pop_back();
            break;
          }
        }

        if (is_new) {
          j++;
        }
      }

      combined.insert(combined.end(), fos.begin(), fos.end());
    }
    return combined;
  };

  bool is_static() const override final {
    for (usize i = 0; i < models.size(); i++) {
      if (!models[i]->is_static()) {
        return false;
      }
    }
    return true;
  };

 private:
  std::vector<std::unique_ptr<LinkageModelBase>> models;
};
};  // namespace goblin

#endif /* _GOBLIN_LIB_LINKAGE_MODEL_H */
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                       goblin/lib/method.h included by goblin.h                                               //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef _GOBLIN_LIB_METHOD_H
#define _GOBLIN_LIB_METHOD_H



namespace goblin {
class MethodBase {
 public:
  virtual std::tuple<std::shared_ptr<ArchiveBase>, TerminationStatus> run(InstanceBase& problem,
                                                                          const Budget& budget,
                                                                          std::optional<u64> seed,
                                                                          std::optional<usize> population_size) = 0;

  // TODO scipy.optimize compatible .minimize function

  /// Evaluations used and time elapsed can be collected without knowing about
  /// method details, but not the generation/iteraction. This method provides a
  /// hook for tracking the progress over generations
  virtual std::optional<u64> current_generation() const { return std::nullopt; };

  /// Size and generations of the currently active population if available for multi-start schemes
  virtual std::optional<std::tuple<usize, u64>> current_population() const { return std::nullopt; };

  virtual ~MethodBase() {};
};
};  // namespace goblin

#endif /* _GOBLIN_LIB_METHOD_H */

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                       goblin/lib/init.h included by goblin.h                                                 //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef _GOBLIN_LIB_INIT_H
#define _GOBLIN_LIB_INIT_H



namespace goblin {

class InitBase {
 public:
  virtual std::tuple<Mat<DType>, Mat<CType>> sample(Rng& rng, const InstanceBase& problem, usize count) const = 0;

  void add_random(Rng& rng, const InstanceBase& problem, SolutionSetBase& solutions, usize count) const {
    auto [dvals, cvals] = sample(rng, problem, count);

    assert(static_cast<usize>(dvals.rows()) == count);
    assert(static_cast<usize>(dvals.cols()) == problem.num_discrete());
    assert(static_cast<usize>(cvals.rows()) == count);
    assert(static_cast<usize>(cvals.cols()) == problem.num_continuous());

    for (usize i = 0; i < count; i++) {
      solutions.add(
          Solution(problem.archive_fitness().worst(),
                   problem.num_discrete() > 0 ? std::make_optional<Vec<DType>>(dvals.row(i)) : std::nullopt,
                   problem.num_continuous() > 0 ? std::make_optional<Vec<CType>>(cvals.row(i)) : std::nullopt));
    }
  };

  virtual ~InitBase() = default;
};

class DiscreteInitBase {
 public:
  virtual Mat<DType> sample(Rng& rng, const InstanceBase& problem, usize count) const = 0;

  virtual ~DiscreteInitBase() = default;
};

class ContinuousInitBase {
 public:
  virtual Mat<CType> sample(Rng& rng, const InstanceBase& problem, usize count) const = 0;

  virtual ~ContinuousInitBase() = default;
};

class RandomDInit final : public DiscreteInitBase {
 public:
  Mat<DType> sample(Rng& rng, const InstanceBase& problem, usize count) const override final {
    Mat<DType> dvals(count, problem.num_discrete());
    for (isize c = 0; c < dvals.cols(); c++) {
      std::uniform_int_distribution<DType> d(0, problem.discrete_domain_sizes()(c) - DType(1));
      for (usize i = 0; i < count; i++) {
        dvals(i, c) = d(rng);
      }
    }
    return dvals;
  };
};

class RandomCInit final : public ContinuousInitBase {
 public:
  Mat<CType> sample(Rng& rng, const InstanceBase& problem, usize count) const override final {
    Mat<CType> cvals(count, problem.num_continuous());
    for (isize c = 0; c < cvals.cols(); c++) {
      std::uniform_real_distribution<CType> d(problem.continuous_init_lower_bounds()(c),
                                              problem.continuous_init_upper_bounds()(c));
      for (usize i = 0; i < count; i++) {
        cvals(i, c) = d(rng);
      }
    }
    return cvals;
  };
};

class RandomInit final : public InitBase {
 public:
  std::tuple<Mat<DType>, Mat<CType>> sample(Rng& rng, const InstanceBase& problem, usize count) const override final {
    return std::make_tuple<Mat<DType>, Mat<CType>>(RandomDInit().sample(rng, problem, count),
                                                   RandomCInit().sample(rng, problem, count));
  };
};

class CombinedInit final : public InitBase {
 public:
  CombinedInit(std::optional<std::shared_ptr<DiscreteInitBase>> discrete = std::nullopt,
               std::optional<std::shared_ptr<ContinuousInitBase>> continuous = std::nullopt)
      : _discrete(discrete.value_or(std::make_shared<RandomDInit>())),
        _continuous(continuous.value_or(std::make_shared<RandomCInit>())) {};

  std::tuple<Mat<DType>, Mat<CType>> sample(Rng& rng, const InstanceBase& problem, usize count) const override final {
    return std::make_tuple<Mat<DType>, Mat<CType>>(_discrete->sample(rng, problem, count),
                                                   _continuous->sample(rng, problem, count));
  };

 private:
  std::shared_ptr<DiscreteInitBase> _discrete;
  std::shared_ptr<ContinuousInitBase> _continuous;
};

using AnyInit = std::variant<std::shared_ptr<InitBase>,
                             std::shared_ptr<DiscreteInitBase>,
                             std::shared_ptr<ContinuousInitBase>,
                             std::tuple<std::shared_ptr<DiscreteInitBase>, std::shared_ptr<ContinuousInitBase>>>;

inline std::shared_ptr<InitBase> from_any_init(AnyInit any_init) {
  if (std::holds_alternative<std::shared_ptr<InitBase>>(any_init)) {
    return std::get<std::shared_ptr<InitBase>>(any_init);
  } else if (std::holds_alternative<std::shared_ptr<DiscreteInitBase>>(any_init)) {
    return std::make_shared<CombinedInit>(std::get<std::shared_ptr<DiscreteInitBase>>(any_init), std::nullopt);
  } else if (std::holds_alternative<std::shared_ptr<ContinuousInitBase>>(any_init)) {
    return std::make_shared<CombinedInit>(std::nullopt, std::get<std::shared_ptr<ContinuousInitBase>>(any_init));
  } else if (std::holds_alternative<std::tuple<std::shared_ptr<DiscreteInitBase>, std::shared_ptr<ContinuousInitBase>>>(
                 any_init)) {
    auto [discrete, continuous] =
        std::get<std::tuple<std::shared_ptr<DiscreteInitBase>, std::shared_ptr<ContinuousInitBase>>>(any_init);
    return std::make_shared<CombinedInit>(discrete, continuous);
  } else {
    std::unreachable();
  }
};

class CompleteInit final : public DiscreteInitBase {
 public:
  Mat<DType> sample(Rng& rng, const InstanceBase& problem, usize count) const override final {
    Mat<DType> dvals(count, problem.num_discrete());
    for (isize c = 0; c < dvals.cols(); c++) {
      std::vector<DType> perm(problem.discrete_domain_sizes()(c));
      std::iota(perm.begin(), perm.end(), 0);

      usize i = perm.size();
      for (usize r = 0; r < count; r++) {
        if (i >= perm.size()) {
          i = 0;
          std::shuffle(perm.begin(), perm.end(), rng);
        }

        dvals(r, c) = perm[i++];
      }
    }
    return dvals;
  };
};

};  // namespace goblin

#endif /* _GOBLIN_LIB_INIT_H */


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                       goblin/gp/context.h included by goblin.h                                               //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef _GOBLIN_GP_CONTEXT_H
#define _GOBLIN_GP_CONTEXT_H

#include <queue>


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                       goblin/gp/operator.h included by goblin/gp/context.h                                   //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef _GOBLIN_GP_OPERATOR_H
#define _GOBLIN_GP_OPERATOR_H



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                       goblin/gp/template.h included by goblin/gp/operator.h                                  //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef _GOBLIN_GP_TEMPLATE_H
#define _GOBLIN_GP_TEMPLATE_H



namespace goblin {
struct TemplateNode {
  std::vector<TemplateNode> children;
  size_t max_num_nodes;

  static TemplateNode full_nary(usize branching_factor, usize depth) {
    TemplateNode root;
    if (depth > 0) {
      root.children.reserve(branching_factor);
      for (usize i = 0; i < branching_factor; i++) {
        root.children.push_back(full_nary(branching_factor, depth - 1));
      }
    }
    return root;
  };

  usize size() const {
    __goblin_runtime_assert(is_tree());
    usize s = 0;
    visit([&s](const auto& _) { s++; });
    return s;
  };

  usize max_num_children() const {
    __goblin_runtime_assert(is_tree());
    usize mnc = 0;
    visit([&mnc](const auto& n) { mnc = std::max(mnc, n.children.size()); });
    return mnc;
  };

  template <typename F>
  void visit(F fn) const {
    fn(*this);
    for (auto& c : children) {
      c.visit(fn);
    }
  };

  bool is_tree() const {
    std::set<const TemplateNode*> visited;
    std::vector<const TemplateNode*> queue{this};
    while (!queue.empty()) {
      auto current = queue.back();
      queue.pop_back();
      if (visited.contains(current)) {
        return false;
      }
      visited.insert(current);
      for (auto& c : current->children) {
        queue.push_back(&c);
      }
    }
    return true;
  };

  bool is_cycle_free() const {
    std::set<const TemplateNode*> visited;
    std::set<const TemplateNode*> path;
    return is_cycle_free_helper(this, path, visited);
  };

 private:
  bool is_cycle_free_helper(const TemplateNode* current,
                            std::set<const TemplateNode*>& path,
                            std::set<const TemplateNode*>& visited) const {
    path.insert(current);
    visited.insert(current);
    for (const auto& c : current->children) {
      // predecessor in the current path is also a child -> cycle found
      // (same condition also holds for any not yet visited child)
      if (path.contains(&c) || (!visited.contains(&c) && !is_cycle_free_helper(&c, path, visited))) {
        return false;
      }
    }
    path.erase(current);
    return true;
  };
};

struct Template {
  std::vector<TemplateNode> outputs;
  std::vector<TemplateNode> subexpressions;

  Template() = default;
  Template(std::vector<TemplateNode> outputs, std::vector<TemplateNode> subexpressions)
      : outputs(outputs), subexpressions(subexpressions) {
    if (!is_valid()) {
      throw std::runtime_error("Template is not valid!");
    }
  }

  usize size() const {
    usize s = 0;
    for (auto& o : outputs) {
      s += o.size();
    }
    for (auto& o : subexpressions) {
      s += o.size();
    }
    return s;
  };

  usize max_num_children() const {
    usize mnc = 0;
    for (auto& o : outputs) {
      mnc = std::max(mnc, o.max_num_children());
    }
    for (auto& o : subexpressions) {
      mnc = std::max(mnc, o.max_num_children());
    }
    return mnc;
  };

  void add_output(TemplateNode output) {
    outputs.emplace_back(output);
    if (!is_valid()) {
      throw std::runtime_error("Template is not valid!");
    }
  };

  void add_subtree(TemplateNode subexpression) {
    subexpressions.emplace_back(subexpression);
    if (!is_valid()) {
      throw std::runtime_error("Template is not valid!");
    }
  };

  bool is_valid() const {
    for (auto& s : subexpressions) {
      if (!s.is_tree()) {
        return false;
      }
    }
    for (auto& o : outputs) {
      if (!o.is_tree()) {
        return false;
      }
    }
    return true;
  }
};

};  // namespace goblin

#endif /* _GOBLIN_GP_TEMPLATE_H */

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                       goblin/gp/gpu_evaluation/types.h included by goblin/gp/operator.h                      //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef _GOBLIN_GA_GP_TYPES_H
#define _GOBLIN_GA_GP_TYPES_H


namespace goblin {

using u8 = std::uint8_t;

enum class KernelVersion : u8 {
    Baseline,
    Restrict,
    SharedMemory,
    BlockReduce,
    SingleKernel,
    SingleKernelFMAF,
    SingleKernelInplace,
    Hybrid
};

constexpr std::string_view to_string(KernelVersion v) {
    switch (v) {
        case KernelVersion::Baseline:            return "Baseline";
        case KernelVersion::Restrict:            return "Restrict";
        case KernelVersion::SharedMemory:        return "SharedMemory";
        case KernelVersion::BlockReduce:         return "BlockReduce";
        case KernelVersion::SingleKernel:        return "SingleKernel";
        case KernelVersion::SingleKernelFMAF:    return "SingleKernelFMAF";
        case KernelVersion::SingleKernelInplace: return "SingleKernelInPlace";
        case KernelVersion::Hybrid:              return "Hybrid";
    }

    return "Unknown KernelVersion";
}

enum class NodeType : u8 {
    Input,
    Constant,
    Operator,
    Parameter
};

enum class Operator : u8 {
    Add,
    Sub,
    Mul,
    Div,
    Sin,
    Cos,
    Exp,
    Log,
    Square,
    Sqrt,
    Pow,
    Abs,
    Min,
    Max
};

// The following declarations are used to create more readable test cases
namespace test {
    constexpr u8 C = static_cast<u8>(NodeType::Constant);
    constexpr u8 I = static_cast<u8>(NodeType::Input);
    constexpr u8 O = static_cast<u8>(NodeType::Operator);

    constexpr float Val(float x) { return x; }
    constexpr float Val(int x) { return static_cast<float>(x); }
    constexpr float Val(double x) { return static_cast<float>(x); }
    constexpr float Idx(int idx) { return static_cast<float>(idx); }
    constexpr float Op(Operator op) { return static_cast<float>(op); }

    constexpr float Add = Op(Operator::Add);
    constexpr float Sub = Op(Operator::Sub);
    constexpr float Mul = Op(Operator::Mul);
    constexpr float Div = Op(Operator::Div);
    constexpr float Sin = Op(Operator::Sin);
    constexpr float Cos = Op(Operator::Cos);
    constexpr float Exp = Op(Operator::Exp);
    constexpr float Log = Op(Operator::Log);
    constexpr float Square = Op(Operator::Square);
    constexpr float Sqrt = Op(Operator::Sqrt);
    constexpr float Pow = Op(Operator::Pow);
    constexpr float Abs = Op(Operator::Abs);
    constexpr float Min = Op(Operator::Min);
    constexpr float Max = Op(Operator::Max);
}

}

#endif /* _GOBLIN_GA_GP_TYPES_H */

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                       goblin/gp/operator.h continued                                                         //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace goblin {
// TODO multi-arity (not here), but allow nodes to not use all children but a
// chosen subset (=partial permutation)
// TODO types - every column (input and output Arr2Drices) now has a type etc...
// enum struct Type: unsigned char {
//     Bool = 0,
//     Float = 1
// };
// struct Signature {
//     std::vector<Type> args;
//     Type rty;
// };
// using CColRef = std::variant<
//     CRefS<Array<BType>>,
//     CRefS<Array<CType>>
// >;
// using ColRef = std::variant<
//     RefS<Array<BType>>,
//     RefS<Array<CType>>
// >;
// template<typename C>
// inline constexpr Type typeof(const C& col){
//     if (std::same_as<typename C::Scalar, CType>){
//         return Type::Float;
//     } else if constexpr (std::same_as<typename C::Scalar, BType>){
//         return Type::Bool;
//     } else {
//         std::unreachable();
//     }
// };

// The operators and their derivatives were more or less copied over from
// https://github.com/matigekunstintelligentie/MultiGPG/blob/main/src/operator.hpp

class OperatorBase {
 public:
  virtual usize min_arity() const = 0;
  virtual usize max_arity() const = 0;

  virtual bool is_commutative() const = 0;

  virtual void apply(Ref<Array<CType>> out, CRef<Arr2D<CType>> args) const = 0;
  virtual void apply_buf(Arr2D<CType>& buf, usize out, const std::span<const usize>& args) const {
    throw std::runtime_error("Not implemented.");
  };

  virtual bool has_gradient() const { return false; };
  virtual void apply_grad(Ref<Array<CType>> out,
                          Ref<Array<CType>> d_out,
                          CRef<Arr2D<CType>> args,
                          CRef<Arr2D<CType>> d_args) const {
    throw std::runtime_error("Gradients not supported.");
  };

  Array<CType> operator()(CRef<Arr2D<CType>> args) const {
    Array<CType> out(args.rows());
    apply(out, args);
    return out;
  };

  virtual std::optional<uint8_t> gpu_operator_id() const { return std::nullopt; }

  virtual std::string format(const std::span<const std::string>& args) const = 0;

  virtual ~OperatorBase() = default;
};

class OpIdentity : public OperatorBase {
 public:
  usize min_arity() const override final { return 1; };
  usize max_arity() const override final { return 1; };

  bool is_commutative() const override final { return false; };

  void apply(Ref<Array<CType>> out, CRef<Arr2D<CType>> args) const override final { out = args.col(0); };

  void apply_buf(Arr2D<CType>& buf, usize out, const std::span<const usize>& args) const override final {
    buf.col(out) = buf.col(args[0]);
  };

  bool has_gradient() const override final { return true; };
  void apply_grad(Ref<Array<CType>> out,
                  Ref<Array<CType>> d_out,
                  CRef<Arr2D<CType>> args,
                  CRef<Arr2D<CType>> d_args) const override final {
    apply(out, args);

    d_out = d_args.col(0);
  };

  std::string format(const std::span<const std::string>& args) const override final { return args[0]; };
};

class OpAdd : public OperatorBase {
 public:
  usize min_arity() const override final { return 2; };
  usize max_arity() const override final { return std::numeric_limits<usize>::max(); };

  bool is_commutative() const override final { return true; };

  void apply(Ref<Array<CType>> out, CRef<Arr2D<CType>> args) const override final { out = args.rowwise().sum(); };

  void apply_buf(Arr2D<CType>& buf, usize out, const std::span<const usize>& args) const override final {
    buf.col(out) = buf.col(args[0]);
    for (usize i = 1; i < args.size(); i++) {
      buf.col(out) += buf.col(args[i]);
    }
  };

  bool has_gradient() const override final { return true; };
  void apply_grad(Ref<Array<CType>> out,
                  Ref<Array<CType>> d_out,
                  CRef<Arr2D<CType>> args,
                  CRef<Arr2D<CType>> d_args) const override final {
    apply(out, args);
    d_out = d_args.rowwise().sum();
  };

  std::optional<uint8_t> gpu_operator_id() const override final {
    return static_cast<uint8_t>(Operator::Add);
  }

  std::string format(const std::span<const std::string>& args) const override final {
    std::ostringstream ss;
    ss << '(';
    for (usize i = 0; i < args.size(); i++) {
      if (i > 0) {
        ss << " + ";
      }
      ss << args[i];
    }
    ss << ')';
    return ss.str();
  };
};

class OpSub : public OperatorBase {
 public:
  usize min_arity() const override final { return 1; };
  usize max_arity() const override final { return std::numeric_limits<usize>::max(); };

  bool is_commutative() const override final {
    // well actually: all arguments after the first one are interchangeable
    return false;
  };

  void apply(Ref<Array<CType>> out, CRef<Arr2D<CType>> args) const override final {
    if (args.cols() > 1) {
      out = args.col(0) - args(Eigen::placeholders::all, Eigen::seqN(1, args.cols() - 1)).rowwise().sum();
    } else {
      out = -args.col(0);
    }
  };

  void apply_buf(Arr2D<CType>& buf, usize out, const std::span<const usize>& args) const override final {
    if (args.size() > 1) {
      buf.col(out) = buf.col(args[0]);
      for (usize i = 1; i < args.size(); i++) {
        buf.col(out) -= buf.col(args[i]);
      }
    } else {
      buf.col(out) -= buf.col(args[0]);
    }
  };

  bool has_gradient() const override final { return true; };
  void apply_grad(Ref<Array<CType>> out,
                  Ref<Array<CType>> d_out,
                  CRef<Arr2D<CType>> args,
                  CRef<Arr2D<CType>> d_args) const override final {
    apply(out, args);
    if (args.cols() > 1) {
      d_out = d_args.col(0) - d_args(Eigen::placeholders::all, Eigen::seqN(1, d_args.cols() - 1)).rowwise().sum();
    } else {
      d_out = -d_args.col(0);
    }
  };

  std::string format(const std::span<const std::string>& args) const override final {
    std::ostringstream ss;
    if (args.size() == 1) {
      ss << "(-" << args[0] << ')';
    } else {
      ss << '(';
      for (usize i = 0; i < args.size(); i++) {
        if (i > 0) {
          ss << " - ";
        }
        ss << args[i];
      }
      ss << ')';
    }
    return ss.str();
  };
};

class OpSubGPU : public OperatorBase {
 public:
  usize min_arity() const override final { return 2; };
  usize max_arity() const override final { return std::numeric_limits<usize>::max(); };

  bool is_commutative() const override final {
    // well actually: all arguments after the first one are interchangeable
    return false;
  };

  void apply(Ref<Array<CType>> out, CRef<Arr2D<CType>> args) const override final {
    if (args.cols() > 1) {
      out = args.col(0) - args(Eigen::placeholders::all, Eigen::seqN(1, args.cols() - 1)).rowwise().sum();
    } else {
      out = -args.col(0);
    }
  };

  void apply_buf(Arr2D<CType>& buf, usize out, const std::span<const usize>& args) const override final {
    if (args.size() > 1) {
      buf.col(out) = buf.col(args[0]);
      for (usize i = 1; i < args.size(); i++) {
        buf.col(out) -= buf.col(args[i]);
      }
    } else {
      buf.col(out) -= buf.col(args[0]);
    }
  };

  bool has_gradient() const override final { return true; };
  void apply_grad(Ref<Array<CType>> out,
                  Ref<Array<CType>> d_out,
                  CRef<Arr2D<CType>> args,
                  CRef<Arr2D<CType>> d_args) const override final {
    apply(out, args);
    if (args.cols() > 1) {
      d_out = d_args.col(0) - d_args(Eigen::placeholders::all, Eigen::seqN(1, d_args.cols() - 1)).rowwise().sum();
    } else {
      d_out = -d_args.col(0);
    }
  };

  std::optional<uint8_t> gpu_operator_id() const override final {
    return static_cast<uint8_t>(Operator::Sub);
  }

  std::string format(const std::span<const std::string>& args) const override final {
    std::ostringstream ss;
    if (args.size() == 1) {
      ss << "(-" << args[0] << ')';
    } else {
      ss << '(';
      for (usize i = 0; i < args.size(); i++) {
        if (i > 0) {
          ss << " - ";
        }
        ss << args[i];
      }
      ss << ')';
    }
    return ss.str();
  };
};

class OpMul : public OperatorBase {
 public:
  usize min_arity() const override final { return 2; };
  usize max_arity() const override final { return std::numeric_limits<usize>::max(); };

  bool is_commutative() const override final { return true; };

  void apply(Ref<Array<CType>> out, CRef<Arr2D<CType>> args) const override final { out = args.rowwise().prod(); };

  void apply_buf(Arr2D<CType>& buf, usize out, const std::span<const usize>& args) const override final {
    buf.col(out) = buf.col(args[0]);
    for (usize i = 1; i < args.size(); i++) {
      buf.col(out) *= buf.col(args[i]);
    }
  };

  bool has_gradient() const override final { return true; };
  void apply_grad(Ref<Array<CType>> out,
                  Ref<Array<CType>> d_out,
                  CRef<Arr2D<CType>> args,
                  CRef<Arr2D<CType>> d_args) const override final {
    assert(d_args.cols() >= 2);
    apply(out, args);

    // sum(df_i * prod(f_{j!=i}))
    d_out = d_args.col(0) * args(Eigen::placeholders::all, Eigen::seq(1, args.cols() - 1)).rowwise().prod();

    for (isize i = 1; i < d_args.cols() - 1; i++) {
      d_out += args(Eigen::placeholders::all, Eigen::seq(0, i - 1)).rowwise().prod() * d_args.col(i) *
               args(Eigen::placeholders::all, Eigen::seq(i + 1, args.cols() - 1)).rowwise().prod();
    }
    d_out +=
        args(Eigen::placeholders::all, Eigen::seq(0, args.cols() - 2)).rowwise().prod() * d_args.col(d_args.cols() - 1);
  };

  std::optional<uint8_t> gpu_operator_id() const override final {
    return static_cast<uint8_t>(Operator::Mul);
  }

  std::string format(const std::span<const std::string>& args) const override final {
    std::ostringstream ss;
    ss << '(';
    for (usize i = 0; i < args.size(); i++) {
      if (i > 0) {
        ss << " * ";
      }
      ss << args[i];
    }
    ss << ')';
    return ss.str();
  };
};

class OpDiv : public OperatorBase {
 public:
  usize min_arity() const override final { return 2; };
  usize max_arity() const override final { return std::numeric_limits<usize>::max(); };

  bool is_commutative() const override final {
    // first argument is not commutative
    return false;
  };

  void apply(Ref<Array<CType>> out, CRef<Arr2D<CType>> args) const override final {
    out = args.col(0) / args(Eigen::placeholders::all, Eigen::seq(1, args.cols() - 1)).rowwise().prod();
  };

  void apply_buf(Arr2D<CType>& buf, usize out, const std::span<const usize>& args) const override final {
    buf.col(out) = buf.col(args[1]);
    for (usize i = 2; i < args.size(); i++) {
      buf.col(out) *= buf.col(args[i]);
    }
    buf.col(out) = buf.col(args[0]) / buf.col(out);
  };

  bool has_gradient() const override final { return true; };
  void apply_grad(Ref<Array<CType>> out,
                  Ref<Array<CType>> d_out,
                  CRef<Arr2D<CType>> args,
                  CRef<Arr2D<CType>> d_args) const override final {
    assert(d_args.cols() >= 2);

    auto denom = args(Eigen::placeholders::all, Eigen::seq(1, args.cols() - 1)).rowwise().prod();
    out = args.col(0) / denom;

    d_out = d_args.col(0) / denom;
    for (isize i = 1; i < args.cols(); i++) {
      d_out -= d_args.col(i) * out / args.col(i);
    }
  };

  std::optional<uint8_t> gpu_operator_id() const override final {
    return static_cast<uint8_t>(Operator::Div);
  }

  std::string format(const std::span<const std::string>& args) const override final {
    std::ostringstream ss;
    ss << '(' << args[0] << " / ";
    if (args.size() > 2) {
      ss << '(';
      for (usize i = 1; i < args.size(); i++) {
        if (i > 1) {
          ss << " * ";
        }
        ss << args[i];
      }
      ss << ')';
    } else {
      ss << args[1];
    }
    ss << ')';
    return ss.str();
  };
};

class OpSin : public OperatorBase {
 public:
  usize min_arity() const override final { return 1; };
  usize max_arity() const override final { return 1; };

  bool is_commutative() const override final { return false; };

  void apply(Ref<Array<CType>> out, CRef<Arr2D<CType>> args) const override final { out = args.col(0).sin(); };

  void apply_buf(Arr2D<CType>& buf, usize out, const std::span<const usize>& args) const override final {
    buf.col(out) = buf.col(args[0]).sin();
  };

  bool has_gradient() const override final { return true; };
  void apply_grad(Ref<Array<CType>> out,
                  Ref<Array<CType>> d_out,
                  CRef<Arr2D<CType>> args,
                  CRef<Arr2D<CType>> d_args) const override final {
    apply(out, args);

    d_out = args.col(0).cos() * d_args.col(0);
  };

  std::optional<uint8_t> gpu_operator_id() const override final {
    return static_cast<uint8_t>(Operator::Sin);
  }

  std::string format(const std::span<const std::string>& args) const override final {
    return std::format("sin({})", args[0]);
  };
};

class OpCos : public OperatorBase {
 public:
  usize min_arity() const override final { return 1; };
  usize max_arity() const override final { return 1; };

  bool is_commutative() const override final { return false; };

  void apply(Ref<Array<CType>> out, CRef<Arr2D<CType>> args) const override final { out = args.col(0).cos(); };

  void apply_buf(Arr2D<CType>& buf, usize out, const std::span<const usize>& args) const override final {
    buf.col(out) = buf.col(args[0]).cos();
  };

  bool has_gradient() const override final { return true; };
  void apply_grad(Ref<Array<CType>> out,
                  Ref<Array<CType>> d_out,
                  CRef<Arr2D<CType>> args,
                  CRef<Arr2D<CType>> d_args) const override final {
    apply(out, args);

    d_out = -args.col(0).sin() * d_args.col(0);
  };

  std::optional<uint8_t> gpu_operator_id() const override final {
    return static_cast<uint8_t>(Operator::Cos);
  }

  std::string format(const std::span<const std::string>& args) const override final {
    return std::format("cos({})", args[0]);
  };
};

class OpExp : public OperatorBase {
 public:
  usize min_arity() const override final { return 1; };
  usize max_arity() const override final { return 1; };

  bool is_commutative() const override final { return false; };

  void apply(Ref<Array<CType>> out, CRef<Arr2D<CType>> args) const override final { out = args.col(0).exp(); };

  void apply_buf(Arr2D<CType>& buf, usize out, const std::span<const usize>& args) const override final {
    buf.col(out) = buf.col(args[0]).exp();
  };

  bool has_gradient() const override final { return true; };
  void apply_grad(Ref<Array<CType>> out,
                  Ref<Array<CType>> d_out,
                  CRef<Arr2D<CType>> args,
                  CRef<Arr2D<CType>> d_args) const override final {
    apply(out, args);

    d_out = out * d_args.col(0);
  };

  std::optional<uint8_t> gpu_operator_id() const override final {
    return static_cast<uint8_t>(Operator::Exp);
  }

  std::string format(const std::span<const std::string>& args) const override final {
    return std::format("exp({})", args[0]);
  };
};

class OpLog : public OperatorBase {
 public:
  usize min_arity() const override final { return 1; };
  usize max_arity() const override final { return 1; };

  bool is_commutative() const override final { return false; };

  void apply(Ref<Array<CType>> out, CRef<Arr2D<CType>> args) const override final { out = args.col(0).log(); };

  void apply_buf(Arr2D<CType>& buf, usize out, const std::span<const usize>& args) const override final {
    buf.col(out) = buf.col(args[0]).log();
  };

  bool has_gradient() const override final { return true; };
  void apply_grad(Ref<Array<CType>> out,
                  Ref<Array<CType>> d_out,
                  CRef<Arr2D<CType>> args,
                  CRef<Arr2D<CType>> d_args) const override final {
    apply(out, args);

    d_out = d_args.col(0) / args.col(0);
  };

  std::optional<uint8_t> gpu_operator_id() const override final {
    return static_cast<uint8_t>(Operator::Log);
  }

  std::string format(const std::span<const std::string>& args) const override final {
    return std::format("log({})", args[0]);
  };
};

class OpSquare : public OperatorBase {
 public:
  usize min_arity() const override final { return 1; };
  usize max_arity() const override final { return 1; };

  bool is_commutative() const override final { return false; };

  void apply(Ref<Array<CType>> out, CRef<Arr2D<CType>> args) const override final { out = args.col(0).square(); };

  void apply_buf(Arr2D<CType>& buf, usize out, const std::span<const usize>& args) const override final {
    buf.col(out) = buf.col(args[0]).square();
  };

  bool has_gradient() const override final { return true; };
  void apply_grad(Ref<Array<CType>> out,
                  Ref<Array<CType>> d_out,
                  CRef<Arr2D<CType>> args,
                  CRef<Arr2D<CType>> d_args) const override final {
    apply(out, args);

    d_out = CType(2.0) * args.col(0) * d_args.col(0);
  };

  std::optional<uint8_t> gpu_operator_id() const override final {
    return static_cast<uint8_t>(Operator::Square);
  }

  std::string format(const std::span<const std::string>& args) const override final {
    return std::format("pow({}, 2)", args[0]);
  };
};


class OpSqrt : public OperatorBase {
 public:
  usize min_arity() const override final { return 1; };
  usize max_arity() const override final { return 1; };

  bool is_commutative() const override final { return false; };

  void apply(Ref<Array<CType>> out, CRef<Arr2D<CType>> args) const override final { out = args.col(0).sqrt(); };

  void apply_buf(Arr2D<CType>& buf, usize out, const std::span<const usize>& args) const override final {
    buf.col(out) = buf.col(args[0]).sqrt();
  };

  bool has_gradient() const override final { return true; };
  void apply_grad(Ref<Array<CType>> out,
                  Ref<Array<CType>> d_out,
                  CRef<Arr2D<CType>> args,
                  CRef<Arr2D<CType>> d_args) const override final {
    apply(out, args);

    d_out = d_args.col(0) / (out + out);
  };

  std::optional<uint8_t> gpu_operator_id() const override final {
    return static_cast<uint8_t>(Operator::Sqrt);
  }

  std::string format(const std::span<const std::string>& args) const override final {
    return std::format("sqrt({})", args[0]);
  };
};

class OpPow : public OperatorBase {
 public:
  usize min_arity() const override final { return 2; };
  usize max_arity() const override final { return 2; };

  bool is_commutative() const override final { return false; };

  void apply(Ref<Array<CType>> out, CRef<Arr2D<CType>> args) const override final {
    out = args.col(0).pow(args.col(1));
  };

  void apply_buf(Arr2D<CType>& buf, usize out, const std::span<const usize>& args) const override final {
    buf.col(out) = buf.col(args[0]).pow(buf.col(args[1]));
  };

  bool has_gradient() const override final { return true; };
  void apply_grad(Ref<Array<CType>> out,
                  Ref<Array<CType>> d_out,
                  CRef<Arr2D<CType>> args,
                  CRef<Arr2D<CType>> d_args) const override final {
    apply(out, args);

    d_out = args.col(0).pow(args.col(1) - CType(1.0)) *
            (args.col(0) * d_args.col(1) * args.col(0).log() + args.col(1) * d_args.col(0));
  };

  std::optional<uint8_t> gpu_operator_id() const override final {
    return static_cast<uint8_t>(Operator::Pow);
  }

  std::string format(const std::span<const std::string>& args) const override final {
    return std::format("pow({}, {})", args[0], args[1]);
  };
};

class OpAbs : public OperatorBase {
 public:
  usize min_arity() const override final { return 1; };
  usize max_arity() const override final { return 1; };

  bool is_commutative() const override final { return false; };

  void apply(Ref<Array<CType>> out, CRef<Arr2D<CType>> args) const override final { out = args.col(0).abs(); };

  void apply_buf(Arr2D<CType>& buf, usize out, const std::span<const usize>& args) const override final {
    buf.col(out) = buf.col(args[0]).abs();
  };

  bool has_gradient() const override final { return true; };
  void apply_grad(Ref<Array<CType>> out,
                  Ref<Array<CType>> d_out,
                  CRef<Arr2D<CType>> args,
                  CRef<Arr2D<CType>> d_args) const override final {
    apply(out, args);

    d_out = args.col(0) * d_args.col(0) / out;
  };

  std::optional<uint8_t> gpu_operator_id() const override final {
    return static_cast<uint8_t>(Operator::Abs);
  }

  std::string format(const std::span<const std::string>& args) const override final {
    return std::format("abs({})", args[0]);
  };
};

class OpMin : public OperatorBase {
 public:
  usize min_arity() const override final { return 2; };
  usize max_arity() const override final { return std::numeric_limits<usize>::max(); };

  bool is_commutative() const override final { return true; };

  void apply(Ref<Array<CType>> out, CRef<Arr2D<CType>> args) const override final { out = args.rowwise().minCoeff(); };

  void apply_buf(Arr2D<CType>& buf, usize out, const std::span<const usize>& args) const override final {
    buf.col(out) = buf.col(args[0]).minCoeff();
  };

  bool has_gradient() const override final { return true; };
  void apply_grad(Ref<Array<CType>> out,
                  Ref<Array<CType>> d_out,
                  CRef<Arr2D<CType>> args,
                  CRef<Arr2D<CType>> d_args) const override final {
    for (isize i = 0; i < args.rows(); i++) {
      usize arg_min;
      out(i) = args.row(i).minCoeff(&arg_min);
      d_out(i) = d_args(i, arg_min);
    }
  };

  std::optional<uint8_t> gpu_operator_id() const override final {
    return static_cast<uint8_t>(Operator::Min);
  }

  std::string format(const std::span<const std::string>& args) const override final {
    std::ostringstream ss;
    ss << "min(";
    for (usize i = 0; i < args.size(); i++) {
      if (i > 0) {
        ss << ", ";
      }
      ss << args[i];
    }
    ss << ')';
    return ss.str();
  };
};

class OpMax : public OperatorBase {
 public:
  usize min_arity() const override final { return 2; };
  usize max_arity() const override final { return std::numeric_limits<usize>::max(); };

  bool is_commutative() const override final { return true; };

  void apply(Ref<Array<CType>> out, CRef<Arr2D<CType>> args) const override final { out = args.rowwise().maxCoeff(); };

  void apply_buf(Arr2D<CType>& buf, usize out, const std::span<const usize>& args) const override final {
    buf.col(out) = buf.col(args[0]).maxCoeff();
  };

  bool has_gradient() const override final { return true; };
  void apply_grad(Ref<Array<CType>> out,
                  Ref<Array<CType>> d_out,
                  CRef<Arr2D<CType>> args,
                  CRef<Arr2D<CType>> d_args) const override final {
    for (isize i = 0; i < args.rows(); i++) {
      usize arg_max;
      out(i) = args.row(i).maxCoeff(&arg_max);
      d_out(i) = d_args(i, arg_max);
    }
  };

  std::optional<uint8_t> gpu_operator_id() const override final {
    return static_cast<uint8_t>(Operator::Max);
  }

  std::string format(const std::span<const std::string>& args) const override final {
    std::ostringstream ss;
    ss << "max(";
    for (usize i = 0; i < args.size(); i++) {
      if (i > 0) {
        ss << ", ";
      }
      ss << args[i];
    }
    ss << ')';
    return ss.str();
  };
};
};  // namespace goblin

#endif /* _GOBLIN_GP_OPERATOR_H */


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                       goblin/gp/context.h continued                                                          //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace goblin {

enum class ConstantRepr : u8 { ERCs, Edges, Pool, None };

enum class ValueKind : u8 {
  Input,     // input feature idx
  Constant,  // constant marker/pool idx
  Operator,  // operator idx
  Arg,       // subfunction argument idx
  Subtree,   // subtree idx
  Parameter  // function class parameter idx
};

/// The lookup tables needed to map the linear representation to the encoded
/// semantics and the methods for computing the output, active nodes and sympy
/// conversion.
///
/// A two-step approach is used, where each semantic symbol (e.g. operators or
/// input features) are mapped to a value. However, the domain for e.g. leaf
/// nodes does not contain all values, so a second mapping is used to map
/// between node domain and value.
class GPContext {
 public:
  GPContext(usize num_inputs,
            Template expression_template,
            std::vector<std::shared_ptr<OperatorBase>> operators,
            usize num_parameters = 0,
            std::string_view constant_representation = "ercs",  // ercs, edges, pool or none for no constants
            usize constant_pool_size = 10,
            bool enable_subfunctions = false,  // ADF vs ADT
            std::optional<usize> max_expression_size = std::nullopt,
            bool use_apply_buf = true)
      : const_repr(constant_representation == "pool"
                       ? ConstantRepr::Pool
                       : (constant_representation == "ercs"
                              ? ConstantRepr::ERCs
                              : (constant_representation == "none" ? ConstantRepr::None : ConstantRepr::Edges))),
        num_inputs(num_inputs),
        num_outputs(expression_template.outputs.size()),
        num_subexpressions(expression_template.subexpressions.size()),
        num_discrete(expression_template.size()),
        num_continuous(const_repr == ConstantRepr::Pool ? constant_pool_size
                                                        : (const_repr == ConstantRepr::None ? 0 : num_discrete)),
        max_expression_size(max_expression_size.value_or(num_discrete)),
        num_parameters(num_parameters),
        max_num_children(expression_template.max_num_children()),
        enable_subfunctions(enable_subfunctions),
        use_apply_buf(use_apply_buf),
        operators(std::move(operators)) {
    __goblin_runtime_assert(expression_template.is_valid());
    usize num_constant_values = const_repr == ConstantRepr::ERCs   ? 1
                                : const_repr == ConstantRepr::Pool ? constant_pool_size
                                                                   : 0;
    usize num_subtree_args = enable_subfunctions ? max_num_children : 0;

    usize num_values = num_inputs + num_constant_values + num_subtree_args + num_subexpressions +
                       this->operators.size() + num_parameters;
    __goblin_runtime_assert(num_values <= static_cast<usize>(std::numeric_limits<DType>::max()));

    // value lookup tables

    value_kind.reserve(num_values);
    value_min_arity.reserve(num_values);
    value_max_arity.reserve(num_values);
    value_idx.reserve(num_values);
    for (usize i = 0; i < num_inputs; i++) {
      value_kind.push_back(ValueKind::Input);
      value_min_arity.push_back(0);
      value_max_arity.push_back(0);
      value_idx.push_back(i);
    }
    for (usize i = 0; i < num_constant_values; i++) {
      value_kind.push_back(ValueKind::Constant);
      value_min_arity.push_back(0);
      value_max_arity.push_back(0);
      value_idx.push_back(i);
    }
    for (usize i = 0; i < num_subtree_args; i++) {
      value_kind.push_back(ValueKind::Arg);
      value_min_arity.push_back(0);
      value_max_arity.push_back(0);
      value_idx.push_back(i);
    }
    for (usize i = 0; i < num_subexpressions; i++) {
      value_kind.push_back(ValueKind::Subtree);
      value_min_arity.push_back(enable_subfunctions ? 1 : 0);
      value_max_arity.push_back(enable_subfunctions ? num_subtree_args : 0);
      value_idx.push_back(i);
    }
    for (usize i = 0; i < this->operators.size(); i++) {
      op_idx2value.push_back(value_kind.size());
      value_kind.push_back(ValueKind::Operator);
      value_min_arity.push_back(this->operators[i]->min_arity());
      value_max_arity.push_back(std::min(max_num_children, this->operators[i]->max_arity()));
      value_idx.push_back(i);
    }
    for (usize i = 0; i < num_parameters; i++) {
      value_kind.push_back(ValueKind::Parameter);
      value_min_arity.push_back(0);
      value_max_arity.push_back(0);
      value_idx.push_back(i);
    }
    assert(value_kind.size() == num_values && "Not all values have been assigned.");

    // template structure lookup tables
    subtree_roots.resize(expression_template.subexpressions.size());
    output_roots.resize(expression_template.outputs.size());

    domain_sizes.resize(num_discrete);
    domain2value.resize(num_discrete, num_values);

    root.resize(num_discrete);
    sizes.resize(num_discrete);

    depth.resize(num_discrete);
    height.resize(num_discrete);
    children.resize(num_discrete);
    nodes.resize(num_discrete);

    _parent.resize(num_discrete, num_discrete);
    _value2domain = Arr2D<DType>::Constant(num_discrete, num_values, num_values);

    usize index = 0;
    usize nsubtrees = expression_template.subexpressions.size();
    usize ntrees = nsubtrees + expression_template.outputs.size();
    for (usize i = 0; i < ntrees; i++) {
      usize tree_root = index++;
      bool is_subtree = i < nsubtrees;

      if (is_subtree) {
        subtree_roots[i] = tree_root;
      } else {
        output_roots[i - nsubtrees] = tree_root;
      }

      std::queue<std::tuple<const TemplateNode*, usize>> node_queue;
      node_queue.emplace(
          (is_subtree ? &expression_template.subexpressions[i] : &expression_template.outputs[i - nsubtrees]),
          tree_root);
      while (!node_queue.empty()) {
        auto [nptr, idx] = node_queue.front();
        node_queue.pop();

        // structure lookup tables

        root[idx] = tree_root;
        sizes[idx] = nptr->size();

        height[idx] = 1;
        nodes[idx] = {idx};

        if (idx == tree_root) {
          depth[idx] = 0;
        } else {
          auto p_idx = parent(idx);
          assert(p_idx.has_value() && "Nodes that are not tree roots must have a parent.");

          depth[idx] = depth[p_idx.value()] + 1;

          while (p_idx.has_value()) {
            nodes[p_idx.value()].push_back(idx);
            p_idx = parent(p_idx.value());
          }
        }

        for (const auto& c : nptr->children) {
          usize c_idx = index++;
          _parent[c_idx] = idx;
          children[idx].push_back(c_idx);
          node_queue.emplace(&c, c_idx);
        }

        // domain <-> value mapping

        // The domain for a variable are all values, except the invalid ones
        // (e.g. arg placeholders in outputs, functions in leafs or subtree
        // references that could lead to cycles)
        domain_sizes[idx] = 0;
        for (usize value = 0, domain_value; value < num_values; value++) {
          // output trees cannot have arguments
          bool is_invalid_subtree_arg = !is_subtree && value_kind[value] == ValueKind::Arg;
          // subtrees can only call previous subtrees to prevent cycles
          bool is_invalid_subfunction_index =
              is_subtree && value_kind[value] == ValueKind::Subtree && value_idx[value] >= i;
          bool is_arity_mismatch = value_min_arity[value] > children[idx].size();
          if (!(is_invalid_subtree_arg || is_invalid_subfunction_index || is_arity_mismatch)) {
            domain_value = domain_sizes[idx]++;

            domain2value(idx, domain_value) = value;
            _value2domain(idx, value) = domain_value;
          }
        }
      }
    }

    assert(index == num_discrete && "The domain has not been defined for all discrete variables.");
  };

  inline std::optional<DType> value2domain(usize index, DType value) const {
    auto dval = _value2domain(index, value);
    if (dval < domain_sizes[index]) {
      return dval;
    }
    return std::nullopt;
  };

  inline std::optional<usize> parent(usize index) const {
    auto p_idx = _parent[index];
    if (p_idx < num_discrete) {
      return p_idx;
    }
    return std::nullopt;
  };

  // A helper that prints the expression in a human readable format
  void debug_log_expressions(std::ostream& os,
                             const SolutionBase& solution,
                             std::optional<usize> node = std::nullopt,
                             std::string indent = "") const {
    // abuse default parameters to not have to declare a helper...
    if (node.has_value()) {  // a node
      usize idx = node.value();

      for (usize i = 0; i < indent.size(); i++) {
        indent[i] = ' ';
      }
      indent += "└─ ";

      // lookup the value of the current node
      DType value = domain2value(idx, solution.discrete_values()(idx));
      usize v_idx = value_idx[value];

      os << indent << idx << " (V = " << static_cast<usize>(value) << ", "
         << (solution.discrete_active()(idx) ? "active" : "inactive") << "): ";

      if (value_kind[value] == ValueKind::Input) {
        os << "Input[" << v_idx << "]\n";
      } else if (value_kind[value] == ValueKind::Constant) {
        usize ci = const_repr == ConstantRepr::Pool ? v_idx : idx;
        os << "Constant[" << v_idx << "] -> " << solution.continuous_values()(ci) << "\n";
      } else if (value_kind[value] == ValueKind::Parameter) {
        os << "Parameter[" << v_idx << "]\n";
      } else if (value_kind[value] == ValueKind::Arg) {
        os << "Arg[" << v_idx << "]\n";
      } else if (value_kind[value] == ValueKind::Subtree) {
        os << "Fn[" << v_idx << "] -> node " << subtree_roots[v_idx] << '\n';
      } else if (value_kind[value] == ValueKind::Operator) {
        os << "Op[" << v_idx << "] (arity = " << std::min(children[idx].size(), value_max_arity[value]) << ")\n";
      } else {
        std::unreachable();
      }

      for (usize c : children[idx]) {
        debug_log_expressions(os, solution, c, std::string{indent});
      }
    } else {  // the base case - just print all subtrees and outputs
      for (usize n : subtree_roots) {
        os << "Subtree\n";
        debug_log_expressions(os, solution, n, /* indent = */ "");
      }
      for (usize n : output_roots) {
        os << "Output\n";
        debug_log_expressions(os, solution, n, /* indent = */ "");
      }
      os << std::flush;
    }
  }

  // Returns a sympy-like string for each subexpression tree in isolation.
  // Subtree references within a subtree are inlined (subtrees can only call
  // earlier ones). Arg nodes (enable_subfunctions=true) appear as arg0, arg1, ...
  std::vector<std::string> subtrees_to_sympy(const SolutionBase& solution) const {
    std::vector<std::string> subtree_exprs(num_subexpressions, "SIZE OVERFLOW");
    std::vector<std::string> arg_stack;
    arg_stack.reserve(max_expression_size);

    std::vector<std::pair<usize, bool>> stack;
    std::vector<usize> postorder;

    for (usize si = 0; si < num_subexpressions; si++) {
      postorder.clear();
      stack.clear();
      stack.emplace_back(subtree_roots[si], false);

      while (!stack.empty()) {
        auto [idx, post] = stack.back();
        stack.pop_back();

        DType value = domain2value(idx, solution.discrete_values()(idx));

        if (!post && value_kind[value] == ValueKind::Operator) {
          stack.emplace_back(idx, true);
          usize arity = std::min(children[idx].size(), value_max_arity[value]);
          for (usize i = arity; i > 0;) {
            stack.emplace_back(children[idx][--i], false);
          }
        } else {
          postorder.push_back(idx);
        }
      }

      arg_stack.clear();
      for (usize idx : postorder) {
        DType value = domain2value(idx, solution.discrete_values()(idx));
        usize v_idx = value_idx[value];

        if (value_kind[value] == ValueKind::Input) {
          arg_stack.push_back(std::format("x{:d}", v_idx));
        } else if (value_kind[value] == ValueKind::Constant) {
          usize ci = const_repr == ConstantRepr::Pool ? v_idx : idx;
          arg_stack.push_back(std::format("{}", solution.continuous_values()(ci)));
        } else if (value_kind[value] == ValueKind::Parameter) {
          arg_stack.push_back(std::format("c{:d}", v_idx));
        } else if (value_kind[value] == ValueKind::Arg) {
          arg_stack.push_back(std::format("arg{:d}", v_idx));
        } else if (value_kind[value] == ValueKind::Subtree) {
          arg_stack.push_back(subtree_exprs[v_idx]);
        } else if (value_kind[value] == ValueKind::Operator) {
          usize arity = std::min(children[idx].size(), value_max_arity[value]);
          usize base = arg_stack.size() - arity;
          std::span<const std::string> args{arg_stack.end() - arity, arg_stack.end()};
          arg_stack[base] = operators[v_idx]->format(args);
          arg_stack.resize(base + 1);
        }

        if (const_repr == ConstantRepr::Edges) {
          arg_stack.back() = std::format("({} * ({}))", solution.continuous_values()(idx), arg_stack.back());
        }
      }

      subtree_exprs[si] = arg_stack.empty() ? "" : arg_stack.back();
    }

    return subtree_exprs;
  };

  // Returns all trees in postfix/reverse polish notation (https://en.wikipedia.org/wiki/Reverse_Polish_notation) and
  // without references if the total number of nodes exceeds the `max_expression_size` or `std::nullopt` otherwise.
  //
  // The variables are marked active/inactive only if the solution is not const.
  // Note that the reason why it's `nodes_post_order` and not `values_post_order` is that for constants/operator calls
  // the corresponding index/arity isn't fully determined by the value alone.
  template <typename S>
  std::optional<std::vector<std::vector<usize>>> nodes_post_order(S& solution, bool discount_size, usize& size) const {
    // initially we haven't visited anything, so we set everything to be inactive
    if constexpr (!std::is_const<S>()) {
      solution.discrete_active().array() = false;
      solution.continuous_active().array() = false;
    }

    std::vector<std::vector<usize>> nodes;
    nodes.reserve(num_outputs);

    // in the modular GP-GOMEA paper (https://arxiv.org/pdf/2505.01262v1) there is this concept of "discounted" size to
    // not punish re-using subfunctions by only counting the subfunction nodes once.
    Array<u32> visited = Array<u32>::Zero(num_discrete);

    // to resolve subfunction arguments, we need to know the calling node
    // (and if that is another argument, we need the calling node of that tree and so on...)
    std::vector<usize> call_stack;
    call_stack.reserve(max_expression_size);

    // for each we need to visit, we need the node index, the call stack idx and whether the node already was visited
    // (for functions the first time is in-order, and the second time is post-order)
    std::vector<std::tuple<usize, isize, bool>> node_stack;
    node_stack.reserve(max_expression_size);

    // for each output, walk the tree in post-order
    size = 0;  // initially the size is 0 (size in GP is somewhat arbitary - even without subftrees/args which are not
               // counted, simplification typically also has an effect and it's not necessarily a good proxy for
               // "interpretability" in the first place)
    for (usize n : output_roots) {
      // add and allocate for the output
      nodes.emplace_back();
      auto& tree = nodes.back();
      tree.reserve(max_expression_size - size);

      // housekeeping: reset the call stack and node_stack per output
      call_stack.clear();
      node_stack.clear();
      node_stack.emplace_back(n, 0, false);

      // nodes are visited up to twice - once in-order, and once post-order after the children have been taken care of
      while (!node_stack.empty()) {
        // we hit the max size, but have a next node since this is inside the loop
        if (size + tree.size() == max_expression_size) {
          return std::nullopt;
        }

        // get the node and mark it as active
        auto [idx, call_stack_idx, is_post_order] = node_stack.back();
        usize node_stack_idx = node_stack.size() - 1;

        if constexpr (!std::is_const<S>()) {
          solution.discrete_active()(idx) = true;
        }

        // lookup the value of the current node
        DType value = domain2value(idx, solution.discrete_values()(idx));
        usize v_idx = value_idx[value];

        // since this only a traversal without any evaluation, we only have to
        // check if this is an actual value or if we need to resolve arguments or other indirections
        bool update_tree = false;

        // we only need to look at the node if this is the first time we see it - in the post-order visit all we have to
        // do is add it to the tree
        if (!is_post_order) {
          // In the reference modular GP-GOMEA implementation, the "discounted size" is defined
          // here:
          // https://github.com/matigekunstintelligentie/MultiGPG/blob/21094c016f93457df173935a1ec702568c6c2b24/src/individual.hpp#L98
          // and here:
          // https://github.com/matigekunstintelligentie/MultiGPG/blob/21094c016f93457df173935a1ec702568c6c2b24/src/node.hpp#L90
          // - in no case are references counted, i.e. the visit count for arg/fn nodes needs to be reset to 0 later on
          // track the size (repeat visits don't count in the "discounted" setting)
          if (discount_size) {
            visited(idx) = 1;
          }

          if (value_kind[value] == ValueKind::Arg) {
            visited(idx) = 0;

            // we need to replace the argument with the corresponding child of the caller
            // and then replace the stack entry with with the actual argument
            //
            // but the caller might need some resolving if it is not the root of a tree
            // (every function call adds to the call_stack, so it is not necessarily true that the previous call stack
            // entry corresponds to the caller of the current subtree - it might just be an ancestor in the current tree
            // that also calls another subfunction...)

            // get the previous call stack entry
            usize calling_node = call_stack[call_stack_idx];

            // since we move up the call chain, we need to go at least one call/frame backward, but maybe more
            isize num_frames = 1;

            // check if the calling node is an ancestor of the current node - if so, we need to move up the hierarchy to
            // find the ancestor clostest to the root that is on the call_stack... (the first call into this subtree
            // must have the actual caller)
            auto pidx = parent(idx);
            while (pidx.has_value()) {
              if (pidx.value() == calling_node) {
                // note: this works since we don't allow cycles by restricting the domain, i.e. there can only ever be
                // one "active" call to this subfunction, guaranteeing that the calling node of the highest ancestor is
                // the actual calling node.
                calling_node = call_stack[call_stack_idx - num_frames++];
              }
              pidx = parent(pidx.value());
            }

            // now that we have the caller, we can finally replace the stack entry with with the actual argument
            auto& cnodes = children[calling_node];
            assert(idx != cnodes[v_idx % cnodes.size()] && "Self reference found.");

            node_stack.pop_back();
            node_stack.emplace_back(cnodes[v_idx % cnodes.size()],
                                    call_stack_idx - num_frames,  // use the stack index of the (resolved) caller
                                    false);
          } else if (value_kind[value] == ValueKind::Subtree) {
            visited(idx) = 0;
            assert(root[idx] != subtree_roots[v_idx] && "Cyclic subtree call detected.");
            // we need to replace the actual subtree with the called subtree

            // first update the call stack
            call_stack.push_back(idx);

            // then replace the stack entry with the called subtree
            std::get<2>(node_stack[node_stack_idx]) =
                true;  // this is a reference type, but we need the post-order visit to keep the call stack in sync
            node_stack.emplace_back(
                subtree_roots[v_idx],
                call_stack.size() - 1,  // a call always needs to use the top of the stack, no
                                        // matter where the current call_stack_idx is (!there might be a chain of calls
                                        // between the root containing the actual caller of this node!)
                false);
          } else if (value_kind[value] == ValueKind::Operator) {
            // the operator stays on the stack, but the next visit is post-order
            std::get<2>(node_stack[node_stack_idx]) = true;

            // all the arguments need to be added (the stack reverses the order, so this is in reverse)
            usize arity = std::min(children[idx].size(), value_max_arity[value]);
            for (usize i = arity; i > 0;) {
              // the child's calling node/call_stack_idx is the same as the parents
              node_stack.emplace_back(children[idx][--i], call_stack_idx, false);
            }
          } else if (value_min_arity[value] > 0) {
            // for anything that is not a leaf, we need to resolve the arguments, so another branch is needed if
            // non-terminal kinds are added
            throw std::runtime_error("Encountered unhandled non-leaf node.");
          } else {
            if constexpr (!std::is_const<S>()) {
              if (value_kind[value] == ValueKind::Constant) {
                solution.continuous_active()(const_repr == ConstantRepr::Pool ? v_idx : idx) = true;
              }
            }

            // this is a leaf, so the in-order is the post-order visit
            update_tree = true;
            node_stack.pop_back();  // this is the post-order visit, so no need to visit again
          }
        } else {
          if (value_kind[value] == ValueKind::Subtree) {
            // in the previous in-order visit, this node was pushed on the call stack so it has to be removed now
            call_stack.pop_back();
          } else {
            // this is a non-reference post-order visit, so we need to update the tree
            update_tree = true;
          }

          // remove post order nodes from the stack
          node_stack.pop_back();
        }

        // finally, if this is a leaf or if this is a non-reference post-order visit, then we add it to the tree
        if (update_tree) {
          // Indirections like Subtree/Arg calls are not kept, so only the constants for actual "values", not
          // "references" are used
          if (const_repr == ConstantRepr::Edges) {
            if constexpr (!std::is_const<S>()) {
              solution.continuous_active()(idx) = true;
            }
          }

          tree.push_back(idx);
        }
      }

      size += tree.size();
    }

    if (discount_size) {
      size = visited.sum();
    }

    return nodes;
  }

  std::vector<std::string> to_sympy(const SolutionBase& solution) const {
    // extract the expression in post-order
    usize _size;
    auto nodes = nodes_post_order(solution, false, _size);

    std::vector<std::string> exprs(num_outputs, "SIZE OVERFLOW");

    // check if we are done already
    if (!nodes.has_value()) {
      return exprs;
    }

    // keep a stack for the arguments
    std::vector<std::string> arg_stack;
    arg_stack.reserve(max_expression_size);

    const auto trees = nodes.value();
    for (usize i = 0; i < trees.size(); i++) {
      const auto& tree = trees[i];

      // housekeeping: initially, there are no arguments
      arg_stack.clear();

      // the nodes are in postfix notation, so we evaluate from left to right
      for (usize j = 0; j < tree.size(); j++) {
        usize idx = tree[j];

        // lookup the value of the current node
        usize value = domain2value(idx, solution.discrete_values()(idx));
        usize v_idx = value_idx[value];

        // at this point, all references have been resolved
        assert(value_kind[value] != ValueKind::Arg && "Unresolved argument.");
        assert(value_kind[value] != ValueKind::Subtree && "Unresolved subtree call.");

        // resolve value lookups / function calls
        if (value_kind[value] == ValueKind::Input) {
          arg_stack.push_back(std::format("x{:d}", v_idx));
        } else if (value_kind[value] == ValueKind::Parameter) {
          arg_stack.push_back(std::format("c{:d}", v_idx));
        } else if (value_kind[value] == ValueKind::Constant) {
          usize ci = const_repr == ConstantRepr::Pool ? v_idx : idx;
          arg_stack.push_back(std::format("{}", solution.continuous_values()(ci)));
        } else if (value_kind[value] == ValueKind::Operator) {
          usize arity = std::min(children[idx].size(), value_max_arity[value]);
          usize arg_stack_idx = arg_stack.size() - arity;

          // the arguments are in the correct order on the stack, so we just need to get the last arity indices on the
          // arg_stack
          std::span<const std::string> args{arg_stack.end() - arity, arg_stack.end()};
          arg_stack[arg_stack_idx] = operators[v_idx]->format(args);

          // pop the now used arguments from the stack, but keep the op result
          arg_stack.resize(arg_stack_idx + 1);
        } else {
          std::unreachable();  // if this triggers, either not all reference types have been removed or a non-reference
                               // value kind has been added...
        }

        if (const_repr == ConstantRepr::Edges) {
          arg_stack.back() = std::format("({} * ({}))", solution.continuous_values()(idx), arg_stack.back());
        }
      }

      // at the end the stack only contains the tree output
      exprs[i] = arg_stack.back();
    }

    return exprs;
  };

  template <typename Scalar>
  std::optional<Arr2D<Scalar>> compute_outputs(SolutionBase& solution,
                                               const Arr2D<Scalar>& X,
                                               const Array<Scalar>& params) const {
    Arr2D<Scalar> eval_buffer;
    usize size;
    return compute_outputs(eval_buffer, solution, X, params, size);
  }

  template <typename Scalar>
  std::optional<Arr2D<Scalar>> compute_outputs(Arr2D<Scalar>& eval_buffer,
                                               SolutionBase& solution,
                                               const Arr2D<Scalar>& X,
                                               const Array<Scalar>& params,
                                               usize& size) const {
    // the expression is evaluated in two steps:
    // 1. the actual expression is extracted from the template in-order
    // 2. the operations are interpreted in reverse

    auto nodes = nodes_post_order(solution, /* TODO discount_size = */ true, size);

    // check if we even have to evaluate
    if (!nodes.has_value()) {
      return std::nullopt;
    }

    // ensure the buffer is allocated
    eval_buffer.resize(X.rows(), max_expression_size);

    // evaluation of postfix expressions assumes a stack model, i.e. results are pushed onto as stack, arguments
    // retrieved from the stack and at the end, the single stack entry is the result. Since arguments might be consist
    // of nested operations, the buffer indices corresponding to the actual results are needed somewhere.
    std::vector<usize> arg_stack;
    arg_stack.reserve(max_expression_size);

    // for each output, evaluate the tree
    Arr2D<Scalar> outputs(X.rows(), num_outputs);

    // Eigen::internal::set_is_malloc_allowed(false);

    const auto trees = nodes.value();
    for (usize i = 0; i < trees.size(); i++) {
      const auto& tree = trees[i];

      // housekeeping: initially, there are no arguments
      arg_stack.clear();

      // the nodes are in postfix notation, so we evaluate from left to right
      for (usize j = 0; j < tree.size(); j++) {
        usize idx = tree[j];

        // lookup the value of the current node
        usize value = domain2value(idx, solution.discrete_values()(idx));
        usize v_idx = value_idx[value];

        // at this point, all references have been resolved
        assert(value_kind[value] != ValueKind::Arg && "Unresolved argument.");
        assert(value_kind[value] != ValueKind::Subtree && "Unresolved subtree call.");

        // resolve value lookups / function calls
        if (value_kind[value] == ValueKind::Input) {
          eval_buffer.col(j) = X.col(v_idx);
        } else if (value_kind[value] == ValueKind::Parameter) {
          eval_buffer.col(j) = params(v_idx);
        } else if (value_kind[value] == ValueKind::Constant) {
          usize ci = const_repr == ConstantRepr::Pool ? v_idx : idx;
          eval_buffer.col(j) = solution.continuous_values()(ci);
        } else if (value_kind[value] == ValueKind::Operator) {
          usize arity = std::min(children[idx].size(), value_max_arity[value]);

          // the arguments are in the correct order on the stack, so we just need to get the last arity indices on the
          // arg_stack
          std::span<const usize> child_indices{arg_stack.end() - arity, arg_stack.end()};

          if (use_apply_buf) {
            operators[v_idx]->apply_buf(eval_buffer, j, child_indices);
          } else {
            operators[v_idx]->apply(eval_buffer.col(j), eval_buffer(Eigen::placeholders::all, child_indices));
          }

          // pop the now used arguments from the stack
          arg_stack.resize(arg_stack.size() - arity);
        } else {
          std::unreachable();  // if this triggers, either not all reference types have been removed or a non-reference
                               // value kind has been added...
        }

        if (const_repr == ConstantRepr::Edges) {
          eval_buffer.col(j) *= solution.continuous_values()(idx);
        }

        // since there are no more references, each node output is
        arg_stack.push_back(j);
      }

      // at the end the stack only contains the tree output, and that is at the buffer position for the last tree node
      assert(arg_stack.size() == 1);
      assert(arg_stack.back() == tree.size() - 1);
      outputs.col(i) = eval_buffer.col(tree.size() - 1);
    }

    // Eigen::internal::set_is_malloc_allowed(true);

    return outputs;
  }

  // Matrix of size `num_discrete x num_discrete`, where the entry i,j
  // corresponds to the average proximity to the subtree root of nodes i and j (1.0 is close, 0.0 is distant)
  // if both are from the same tree, otherwise 0
  Mat<CType> normalized_root_proximity() const {
    Mat<CType> proximity(num_discrete, num_discrete);
    CType norm = 0.0;
    for (usize i = 0; i < num_discrete; i++) {
      CType di = static_cast<CType>(depth[i]);
      if (di > norm) {
        norm = di;
      }
    }
    norm += 1.0;
    for (usize i = 0; i < num_discrete; i++) {
      for (usize j = 0; j <= i; j++) {
        proximity(i, j) =
            root[i] == root[j] ? (static_cast<CType>(depth[i]) + static_cast<CType>(depth[j])) * 0.5 : norm;
        proximity(j, i) = proximity(i, j);
      }
    }

    return norm > 0.0 ? 1.0 - proximity.array() / norm : proximity;
  };

  // Normalized node proximity [1.0: same node, 0.0: no connection]
  Mat<CType> normalized_node_proximity() const {
    Mat<CType> proximity(num_discrete, num_discrete);
    CType norm = 0.0;  // = max distance + 1
    for (usize i = 0; i < num_discrete; i++) {
      for (usize j = 0; j <= i; j++) {
        if (root[i] == root[j]) {
          proximity(i, j) = 0.0;
          usize ni = i, nj = j;
          // while the lowest common ancestor was not found, replace the deeper node with its parent until the paths
          // meet at the closest common ancestor
          while (ni != nj) {
            if (depth[ni] > depth[nj]) {
              assert(parent(ni).has_value() && "Since depth > 0, either the depth or parent lookup tables are wrong.");
              ni = parent(ni).value();
            } else {
              assert(depth[nj] > 0 &&
                     "Both are not the same, so at least one must have a non-zero depth since i and j are in the same "
                     "tree");
              assert(parent(nj).has_value() && "Since depth > 0, either the depth or parent lookup tables are wrong.");
              nj = parent(nj).value();
            }
            proximity(i, j) += 1.0;
          }

          if (norm < proximity(i, j)) {
            norm = proximity(i, j);
          }
        } else {
          // use -1 as sentinel for disconnected values
          proximity(i, j) = -1.0;
        }
      }
    }
    norm += 1.0;  // norm = max_distance + 1

    for (usize i = 0; i < num_discrete; i++) {
      for (usize j = 0; j <= i; j++) {
        proximity(i, j) = proximity(i, j) < 0.0 ? 0.0 : 1.0 - proximity(i, j) / norm;

        proximity(j, i) = proximity(i, j);
      }
    }

    return proximity;
  };

  // Normalized node proximity [1.0: same node, 0.0: no connection]
  Mat<CType> normalized_wVIG() const {
    Mat<CType> proximity(num_discrete, num_discrete);
    for (usize i = 0; i < num_discrete; i++) {
      for (usize j = 0; j <= i; j++) {
        if (root[i] == root[j]) {
          proximity(i, j) = 0.0;
          usize ni = i, nj = j;
          // while the lowest common ancestor was not found, replace the deeper node with its parent until the paths
          // meet at the closest common ancestor
          while (ni != nj) {
            if (depth[ni] > depth[nj]) {
              assert(parent(ni).has_value() && "Since depth > 0, either the depth or parent lookup tables are wrong.");
              ni = parent(ni).value();
            } else {
              assert(depth[nj] > 0 &&
                     "Both are not the same, so at least one must have a non-zero depth since i and j are in the same "
                     "tree");
              assert(parent(nj).has_value() && "Since depth > 0, either the depth or parent lookup tables are wrong.");
              nj = parent(nj).value();
            }
            proximity(i, j) += 1.0;
          }
        } else {
          // use -1 as sentinel for disconnected values
          proximity(i, j) = -1.0;
        }
      }
    }

    for (usize i = 0; i < num_discrete; i++) {
      for (usize j = 0; j <= i; j++) {
        proximity(i, j) = proximity(i, j) <= 0.0 ? 0.0 : 1.0 / proximity(i, j);

        proximity(j, i) = proximity(i, j);
      }
    }

    return proximity;
  };

  Mat<CType> subtree_co_occurrences() const {
    Mat<CType> proximity = Mat<CType>::Zero(num_discrete, num_discrete);

    for (usize n = 0; n < num_discrete; n++) {
      const auto& ns = nodes[n];
      for (usize i = 0; i < ns.size(); i++) {
        for (usize j = 0; j <= i; j++) {
          proximity(ns[i], ns[j]) += 1;
          proximity(ns[j], ns[i]) += 1;
        }
      }
    }

    return proximity;
  };

  template <typename F>
  void visit_tree(const SolutionBase& solution, usize root, F&& visit) const {
    std::queue<usize> q;
    q.emplace(root);
    while (!q.empty()) {
      usize current = q.front();
      q.pop();

      visit(current);

      // lookup the value of the current node
      DType value = domain2value(current, solution.discrete_values()(current));
      usize v_idx = value_idx[value];

      usize nc = std::min(value_max_arity[v_idx], children[current].size());
      for (usize i = 0; i < nc; i++) {
        q.emplace(children[current][i]);
      }
    }
  };

  bool copy_tree(const SolutionBase& source, usize source_node, SolutionBase& target, usize target_node) const {
    std::vector<std::tuple<usize, DType>> backup;
    std::queue<std::tuple<usize, usize>> q;
    q.emplace(source_node, target_node);

    bool successful = true;
    while (!q.empty()) {
      auto [from, to] = q.front();
      q.pop();

      // lookup the value of the current node
      DType value = domain2value(from, source.discrete_values()(from));
      usize v_idx = value_idx[value];

      // same node value must be permissible at target location
      auto v = value2domain(to, value);
      if (!v.has_value()) {
        successful = false;
        break;
      }
      backup.emplace_back(to, target.discrete_values()(to));
      target.discrete_values()(to) = v.value();

      usize nc = std::min(value_max_arity[v_idx], children[from].size());
      // target must at least have the same number of children
      if (children[to].size() < nc) {
        successful = false;
        break;
      }
      for (usize i = 0; i < nc; i++) {
        q.emplace(children[from][i], children[to][i]);
      }
    }
    // revert changes
    if (!successful) {
      for (auto [n, v] : backup) {
        target.discrete_values()(n) = v;
      }
    }
    return successful;
  };

  std::vector<usize> active_nodes(const SolutionBase& solution) const {
    std::vector<usize> active;
    for (usize root : subtree_roots) {
      if (solution.discrete_active()(root)) {
        visit_tree(solution, root, [&active](auto n) { active.push_back(n); });
      }
    }
    for (usize root : output_roots) {
      visit_tree(solution, root, [&active](auto n) { active.push_back(n); });
    }
    return active;
  };

  std::vector<usize> active_constant_indices(const SolutionBase& solution) const {
    std::vector<usize> active;
    for (usize n : active_nodes(solution)) {
      // lookup the value of the current node
      DType value = domain2value(n, solution.discrete_values()(n));
      if (value_kind[value] == ValueKind::Constant) {
        usize ci = const_repr == ConstantRepr::Pool ? value_idx[value] : n;
        active.push_back(ci);
      }
    }

    return active;
  };
  // Transforms trees in postfix/reverse polish notation (https://en.wikipedia.org/wiki/Reverse_Polish_notation)
  //
  // Subfunctions are resolved within this function, so the transformed trees will not contain
  // any subfunction/argument nodes
  // Multi-output trees are stored one after another at constant intervals determined by 'max_expression_size'.
  // This value is probably much too large for most trees, however it is very difficult to determine the
  // maximum tree size when subfunctions are involved. Therefore a save maximum value is chosen.
  template <typename S>
  bool to_gpu_representation(
    S& solution,
    std::vector<uint8_t>& node_type,
    std::vector<float>& node_value,
    usize& size,
    bool discount_size = false
  ) const {
    bool overflowed = false;
    // initially we haven't visited anything, so we set everything to be inactive
    if constexpr (!std::is_const<S>()) {
      solution.discrete_active().array() = false;
      solution.continuous_active().array() = false;
    }

    Array<u32> visited = Array<u32>::Zero(num_discrete);

    std::vector<usize> call_stack;
    call_stack.reserve(max_expression_size);

    // (node, call_stack_idx, is_post_order)
    std::vector<std::tuple<usize, isize, bool>> node_stack;
    node_stack.reserve(max_expression_size);

    size = 0;

    // Vectors to hold temporary type and value data
    std::vector<uint8_t> temp_type;
    std::vector<float> temp_value;
    temp_type.reserve(max_expression_size);
    temp_value.reserve(max_expression_size);

    for (usize n : output_roots) {
      // Ensure vectors are clear for next output tree
      temp_type.clear();
      temp_value.clear();

      // housekeeping: reset the call stack and node_stack per output
      call_stack.clear();
      node_stack.clear();
      node_stack.emplace_back(n, 0, false);

      // While there are still nodes to visit
      while(!node_stack.empty()) {
        // Hit the max size, but still have more nodes to process - clear the stack to break
        // out of the loop and fall through to the padding code below, keeping the GPU buffer
        // layout intact for all subsequent solutions in the batch.
        if (size + temp_type.size() == max_expression_size) {
          overflowed = true;
          node_stack.clear();
          break;
        }

        // Pop the top node from the stack
        auto [idx, call_stack_idx, is_post_order] = node_stack.back();
        usize node_stack_idx = node_stack.size() - 1;

        // Mark current node as active
        if constexpr (!std::is_const<S>()) {
          solution.discrete_active()(idx) = true;
        }

        // Get the type and value for the current node
        DType value = domain2value(idx, solution.discrete_values()(idx));
        usize v_idx = value_idx[value];
        enum ValueKind type = value_kind[value];

        bool update_tree = false;

        // only look at the node if this is the first time we see it - in the post-order visit all
        // we have to do is add it to the temp_type and temp_value
        if (!is_post_order) {
          if (discount_size) {
            visited(idx) = 1;
          }

          if (type == ValueKind::Arg) {
            visited(idx) = 0;

            usize calling_node = call_stack[call_stack_idx];
            isize num_frames = 1;

            auto pidx = parent(idx);
            while (pidx.has_value()) {
              if (pidx.value() == calling_node) {
                calling_node = call_stack[call_stack_idx - num_frames++];
              }
              pidx = parent(pidx.value());
            }

            auto& cnodes = children[calling_node];
            assert(idx != cnodes[v_idx % cnodes.size()] && "Self reference found.");

            node_stack.pop_back();
            node_stack.emplace_back(cnodes[v_idx % cnodes.size()],
                                    call_stack_idx - num_frames,
                                    false);
          } else if (type == ValueKind::Subtree) {
            visited(idx) = 0;
            assert(root[idx] != subtree_roots[v_idx] && "Cyclic subtree call detected.");

            call_stack.push_back(idx);

            std::get<2>(node_stack[node_stack_idx]) = true;
            node_stack.emplace_back(subtree_roots[v_idx],
                                    call_stack.size() - 1,
                                    false);
          } else if (type == ValueKind::Operator) {
            std::get<2>(node_stack[node_stack_idx]) = true;

            usize arity = std::min(children[idx].size(), value_max_arity[value]);

            for (usize i = arity; i > 0;) {
              node_stack.emplace_back(children[idx][--i], call_stack_idx, false);
            }
          } else if (value_min_arity[value] > 0) {
            throw std::runtime_error("Encountered unhandled non-leaf node.");
          } else {
            if constexpr (!std::is_const<S>()) {
              if (value_kind[value] == ValueKind::Constant) {
                solution.continuous_active()(const_repr == ConstantRepr::Pool ? v_idx : idx) = true;
              }
            }

            update_tree = true;
            node_stack.pop_back();
          }
        } else {
          if (type == ValueKind::Subtree) {
            call_stack.pop_back();
          } else {
            update_tree = true;
          }

          node_stack.pop_back();
        }

        if (update_tree) {
          if (const_repr == ConstantRepr::Edges) {
            if constexpr (!std::is_const<S>()) {
              solution.continuous_active()(idx) = true;
            }
          }

          temp_type.push_back(static_cast<uint8_t>(type));

          if (type == ValueKind::Input) {
            // Push the index of the input feature, will be used to access the input matrix on GPU
            temp_value.push_back(v_idx);
          } else if (type == ValueKind::Parameter) {
            // Push the index of the parameter, will be used to access the parameter array on GPU
            temp_value.push_back(v_idx);
          } else if (type == ValueKind::Constant) {
            usize ci = const_repr == ConstantRepr::Pool ? v_idx : idx;
            // Push the constant value, will be used directly in the evaluation on GPU
            temp_value.push_back(static_cast<float>(solution.continuous_values()(ci)));
          } else if (type == ValueKind::Operator) {
            // Push the operator index, will be used to apply the operator on GPU
            auto gpu_id = this->operators[v_idx]->gpu_operator_id();
            if (!gpu_id.has_value()) {
              throw std::runtime_error("Operator not GPU-compatible");
            }
            temp_value.push_back(static_cast<float>(gpu_id.value()));
          } else {
            std::unreachable();  // if this triggers, a new ValueKind was added without handling it here
          }

          assert(temp_type.size() == temp_value.size() && "temp_type and temp_value are out of sync.");
        }
      }

      size += temp_type.size();

      // Pad vectors with placeholder values such that the data is at constant intervals in memory
      temp_type.resize(max_expression_size, std::numeric_limits<uint8_t>::max());
      temp_value.resize(max_expression_size, std::numeric_limits<float>::max());

      // Append temporary vectors to final vectors
      node_type.insert(node_type.end(), temp_type.begin(), temp_type.end());
      node_value.insert(node_value.end(), temp_value.begin(), temp_value.end());
    }

    if (discount_size) {
      size = visited.sum();
    }

    return overflowed;
  }

  // // TODO allow gradients w.r.t. specific continuous indices OR parameter
  // // indices
  // template <typename Scalar>
  // Arr2D<Scalar> compute_outputs_grad(SolutionBase& solution, Arr2D<Scalar>& X, Array<Scalar>& params) const {
  //   std::unreachable();
  // };

  // // std::string to_dot(const SolutionBase &solution) const {
  // //   std::unreachable();
  // // };

  ConstantRepr const_repr;

  usize num_inputs;
  usize num_outputs;
  usize num_subexpressions;
  usize num_discrete;
  usize num_continuous;
  usize max_expression_size;
  usize num_parameters;
  usize max_num_children;
  bool enable_subfunctions;
  bool use_apply_buf;

  std::vector<std::shared_ptr<OperatorBase>> operators;
  std::vector<usize> op_idx2value;

  std::vector<ValueKind> value_kind;
  std::vector<usize> value_min_arity;
  std::vector<usize> value_max_arity;
  std::vector<usize> value_idx;

  std::vector<usize> subtree_roots;  // indices of all subtree root nodes
  std::vector<usize> output_roots;   // indices of all output root nodes

  Vec<DType> domain_sizes;    // node -> domain size
  Arr2D<DType> domain2value;  // node domain -> value

  std::vector<usize> root;                   // node -> current tree root
  std::vector<usize> sizes;                  // node -> size of subtree starting at node,
                                             // ignoring inactive values
  std::vector<usize> depth;                  // node -> node depth
  std::vector<usize> height;                 // node -> node height
  std::vector<std::vector<usize>> children;  // node -> child node indices

  std::vector<std::vector<usize>> nodes;  // node -> indices corresponding to the subtree starting at this
                                          // node (without subtrees)

 private:
  Arr2D<DType> _value2domain;
  std::vector<usize> _parent;  // node -> parent or invalid index for root nodes

  // scratch buffers reused across evaluations to avoid per-call heap allocations
  mutable Array<u32> _scratch_visited;
  mutable std::vector<usize> _scratch_call_stack;
  mutable std::vector<std::tuple<usize, isize, bool>> _scratch_node_stack;
  mutable std::vector<usize> _scratch_arg_stack;
};
};  // namespace goblin

#endif /* _GOBLIN_GP_CONTEXT_H */

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                       goblin/gp/init.h included by goblin.h                                                  //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef _GOBLIN_GP_INIT_H
#define _GOBLIN_GP_INIT_H



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                       goblin/gp/instance.h included by goblin/gp/init.h                                      //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef _GOBLIN_GP_INSTANCE_H
#define _GOBLIN_GP_INSTANCE_H



namespace goblin {

class GPInstanceBase : public InstanceBase {
 public:
  virtual const GPContext& context() const = 0;

  virtual ~GPInstanceBase() = default;
};

class PyGPInstance : public GPInstanceBase {
  GPContext ctx;
  std::function<std::tuple<Vec<CType>, CType>(std::function<std::optional<Arr2D<CType>>(Arr2D<CType>)>,
                                              std::vector<std::string>,
                                              usize)>
      evalfn;
  MOFitness _fitness;
  std::shared_ptr<InitBase> init;

  Vec<CType> init_lb;
  Vec<CType> init_ub;

  Vec<CType> lb;
  Vec<CType> ub;

  Arr2D<CType> eval_buffer;

 public:
  PyGPInstance(GPContext ctx,
               std::function<std::tuple<Vec<CType>, CType>(std::function<std::optional<Arr2D<CType>>(Arr2D<CType>)>,
                                                           std::vector<std::string>,
                                                           usize)> evaluate,
               usize num_objectives,
               AnyInit init,
               bool minimize = true,
               CType erc_init_lb = -10.0,
               CType erc_init_ub = 10.0)
      : ctx(ctx),
        evalfn(evaluate),
        _fitness(MOFitness(num_objectives, minimize)),
        init(from_any_init(init)),
        init_lb(Vec<CType>::Constant(ctx.num_continuous, erc_init_lb)),
        init_ub(Vec<CType>::Constant(ctx.num_continuous, erc_init_ub)) {
    lb = Vec<CType>::Constant(ctx.num_continuous, -std::numeric_limits<CType>::infinity());
    ub = Vec<CType>::Constant(ctx.num_continuous, std::numeric_limits<CType>::infinity());
  };

  CRef<Vec<DType>> discrete_domain_sizes() const override final { return ctx.domain_sizes; };

  CRef<Vec<CType>> continuous_lower_bounds() const override final { return lb; }
  CRef<Vec<CType>> continuous_upper_bounds() const override final { return ub; }

  CRef<Vec<CType>> continuous_init_lower_bounds() const override final { return init_lb; }
  CRef<Vec<CType>> continuous_init_upper_bounds() const override final { return init_ub; }

  void evaluate(Rng& rng, SolutionSetBase& solutions, const std::span<const usize>& indices) override final {
    Array<CType> params;
    for (auto i : indices) {
      std::function<std::optional<Arr2D<CType>>(Arr2D<CType>)> predict = [&](Arr2D<CType> inputs) {
        usize size;
        return ctx.compute_outputs(eval_buffer, solutions[i], inputs, params, size);
      };

      solutions[i].discrete_active().fill(false);
      solutions[i].continuous_active().fill(false);

      std::vector<std::string> exprs = ctx.to_sympy(solutions[i]);
      usize size;
      ctx.nodes_post_order(solutions[i], false, size);
      auto [objectives, cv] = evalfn(predict, exprs, size);

      MOQuality q;
      q.objectives = objectives;
      q.constraint_value = cv;
      solutions[i].assign_quality(q);
    }
  };

  void add_random(Rng& rng, SolutionSetBase& solutions, usize count) const override final {
    return init->add_random(rng, *this, solutions, count);
  };

  const FitnessBase& fitness() const override final { return _fitness; };

  const ArchiveFitnessBase& archive_fitness() const override final { return _fitness; };

  virtual std::tuple<bool, bool> inherit_discrete(SolutionBase& offspring,
                                                  const SolutionBase& donor,
                                                  const Subset& subset) const override {
    const bool inherit_continuous = ctx.const_repr == ConstantRepr::ERCs || ctx.const_repr == ConstantRepr::Edges;

    // the pool size is not tied to the number of discrete variables, so the full pool instead of the paired values is
    // inherited...
    const bool inherit_by_index = ctx.const_repr != ConstantRepr::Pool;

    bool any_active_changed = false, anything_changed = false;
    for (usize i : subset.discrete) {
      if (offspring.discrete_values()(i) != donor.discrete_values()(i)) {
        any_active_changed |= offspring.discrete_active()(i);
        anything_changed = true;
        offspring.discrete_values()(i) = donor.discrete_values()(i);
      }

      // TODO for GCS: inherit child arities + permutations

      if (inherit_continuous && inherit_by_index) {
        // TODO sufficiently relatively + absolutely different or no check, but floating point equality is not really
        // useful...
        //
        // yes, the indices here should be from the discrete subset!
        if (offspring.continuous_values()(i) != donor.continuous_values()(i)) {
          any_active_changed |= offspring.continuous_active()(i);
          anything_changed = true;
          offspring.continuous_values()(i) = donor.continuous_values()(i);
        }
      }
    }

    if (inherit_continuous && !inherit_by_index) {
      // note: arguably just inheriting all continuous variables even if the inherited discrete values might not even be
      // constants is not the best idea - but earlier experiments on another codebase suggested that more
      // appropriate/interpolating continuous mixing doesn't really work and here it also is more for completeness and
      // not used by default...
      for (usize i = 0; i < num_continuous(); i++) {
        // TODO sufficiently relatively + absolutely different or no check, but floating point equality is not really
        // useful...
        if (offspring.continuous_values()(i) != donor.continuous_values()(i)) {
          any_active_changed |= offspring.continuous_active()(i);
          anything_changed = true;
          offspring.continuous_values()(i) = donor.continuous_values()(i);
        }
      }
    }

    return std::make_tuple(any_active_changed, anything_changed);
  }

  std::optional<CType> as_continuous(const SolutionBase& solution, usize discrete_index) const override final {
    auto value = ctx.domain2value(discrete_index, solution.discrete_values()(discrete_index));
    if (ctx.value_kind[value] == ValueKind::Constant) {
      return solution.continuous_values()(ctx.const_repr == ConstantRepr::Pool ? ctx.value_idx[value] : discrete_index);
    }
    return std::nullopt;
  };

  void log_solution(std::ostream& os, const SolutionBase& solution) const override final {
    auto exprs = ctx.to_sympy(solution);
    for (usize i = 0; i < exprs.size(); i++) {
      if (i > 0) {
        os << " , ";
      }
      os << exprs[i];
    }
  };

  const GPContext& context() const override final { return ctx; };
};

};  // namespace goblin

#endif /* _GOBLIN_GP_INSTANCE_H */


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                       goblin/gp/init.h continued                                                             //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace goblin {

class GrowInit final : public DiscreteInitBase {
 public:
  GrowInit(std::optional<double> p_terminal = std::nullopt, std::optional<double> p_constant = std::nullopt)
      : p_terminal(p_terminal), p_constant(p_constant) {};

  Mat<DType> sample(Rng& rng, const InstanceBase& problem, usize count) const override final {
    try {
      const auto& ctx = dynamic_cast<const GPInstanceBase&>(problem).context();

      Mat<DType> dvals(count, problem.num_discrete());
      for (isize c = 0; c < dvals.cols(); c++) {
        std::vector<usize> const_terminals;      // numeric constant symbols
        std::vector<usize> non_const_terminals;  // features/args/params
        std::vector<usize> non_terminals;        // operators/subfunctions

        usize domain_size = problem.discrete_domain_sizes()(c);
        if (domain_size == 0) {
          throw std::runtime_error("Variable domain must be non-empty.");
        }
        for (usize domain_value = 0; domain_value < domain_size; domain_value++) {
          DType value = ctx.domain2value(c, domain_value);
          if (ctx.value_min_arity[value] > 0) {
            non_terminals.push_back(domain_value);
          } else if (ctx.value_kind[value] == ValueKind::Constant) {
            const_terminals.push_back(domain_value);
          } else {
            non_const_terminals.push_back(domain_value);
          }
        }

        if (const_terminals.empty() && non_const_terminals.empty()) {
          throw std::runtime_error("Variable domain must include terminal symbols.");
        }

        double actual_p_terminal =
            p_terminal.value_or(1.0 - static_cast<double>(non_terminals.size()) / static_cast<double>(domain_size));
        double actual_p_constant = p_constant.value_or(
            static_cast<double>(const_terminals.size()) /
            (static_cast<double>(const_terminals.size()) + static_cast<double>(non_const_terminals.size())));
        if (const_terminals.empty()) {
          actual_p_constant = 0.0;
        }
        if (non_terminals.empty()) {
          actual_p_terminal = 1.0;
        }

        bool is_leaf = ctx.children[c].empty();
        std::uniform_real_distribution<double> U(0.0, 1.0);
        std::uniform_int_distribution<DType> ct_dist(0, const_terminals.size() - 1);
        std::uniform_int_distribution<DType> nct_dist(0, non_const_terminals.size() - 1);
        std::uniform_int_distribution<DType> nt_dist(0, non_terminals.size() - 1);

        for (usize i = 0; i < count; i++) {
          if (is_leaf || U(rng) < actual_p_terminal) {
            if (U(rng) < actual_p_constant) {
              dvals(i, c) = const_terminals[ct_dist(rng)];
            } else {
              dvals(i, c) = non_const_terminals[nct_dist(rng)];
            }
          } else {
            dvals(i, c) = non_terminals[nt_dist(rng)];
          }
        }
      }
      return dvals;

    } catch (const std::bad_cast& e) {
      __goblin_runtime_assert(false);
    }
  };

 private:
  std::optional<double> p_terminal{};
  std::optional<double> p_constant{};
};

class FullInit final : public DiscreteInitBase {
 public:
  FullInit(std::optional<double> p_constant = std::nullopt) : p_constant(p_constant) {};

  Mat<DType> sample(Rng& rng, const InstanceBase& problem, usize count) const override final {
    try {
      const auto& ctx = dynamic_cast<const GPInstanceBase&>(problem).context();

      Mat<DType> dvals(count, problem.num_discrete());
      for (isize c = 0; c < dvals.cols(); c++) {
        std::vector<usize> const_terminals;      // numeric constant symbols
        std::vector<usize> non_const_terminals;  // features/args/params
        std::vector<usize> non_terminals;        // operators/subfunctions

        usize domain_size = problem.discrete_domain_sizes()(c);
        if (domain_size == 0) {
          throw std::runtime_error("Variable domain must be non-empty.");
        }
        for (usize domain_value = 0; domain_value < domain_size; domain_value++) {
          DType value = ctx.domain2value(c, domain_value);
          if (ctx.value_min_arity[value] > 0) {
            non_terminals.push_back(domain_value);
          } else if (ctx.value_kind[value] == ValueKind::Constant) {
            const_terminals.push_back(domain_value);
          } else {
            non_const_terminals.push_back(domain_value);
          }
        }

        bool is_leaf = ctx.children[c].empty();
        if (is_leaf) {
          if (const_terminals.empty() && non_const_terminals.empty()) {
            throw std::runtime_error("Variable domain must include terminal symbols.");
          }

          double actual_p_constant = p_constant.value_or(
              static_cast<double>(const_terminals.size()) /
              (static_cast<double>(const_terminals.size()) + static_cast<double>(non_const_terminals.size())));
          if (const_terminals.empty()) {
            actual_p_constant = 0.0;
          }

          std::uniform_real_distribution<double> U(0.0, 1.0);
          std::uniform_int_distribution<DType> ct_dist(0, const_terminals.size() - 1);
          std::uniform_int_distribution<DType> nct_dist(0, non_const_terminals.size() - 1);

          for (usize i = 0; i < count; i++) {
            if (U(rng) < actual_p_constant) {
              dvals(i, c) = const_terminals[ct_dist(rng)];
            } else {
              dvals(i, c) = non_const_terminals[nct_dist(rng)];
            }
          }
        } else {
          if (non_terminals.empty()) {
            throw std::runtime_error("Variable domain must include non-terminal symbols for non-leaf nodes.");
          }

          std::uniform_int_distribution<DType> nt_dist(0, non_terminals.size() - 1);

          for (usize i = 0; i < count; i++) {
            dvals(i, c) = non_terminals[nt_dist(rng)];
          }
        }
      }
      return dvals;

    } catch (const std::bad_cast& e) {
      __goblin_runtime_assert(false);
    }
  };

 private:
  std::optional<double> p_constant{};
};

class HalfHalfInit final : public DiscreteInitBase {
 public:
  HalfHalfInit(std::optional<double> p_terminal = std::nullopt, std::optional<double> p_constant = std::nullopt)
      : grow(GrowInit(p_terminal, p_constant)), full(FullInit(p_constant)) {};

  Mat<DType> sample(Rng& rng, const InstanceBase& problem, usize count) const override final {
    Mat<DType> dvals(count, problem.num_discrete());

    // Grow is "for each node sample from the full domain"
    // = Random uniform initialization per node
    usize grow_count = count / 2;
    dvals(Eigen::seq(0, grow_count - 1), Eigen::placeholders::all) = grow.sample(rng, problem, grow_count);

    usize full_count = count - grow_count;
    dvals(Eigen::seq(grow_count, count - 1), Eigen::placeholders::all) = full.sample(rng, problem, full_count);

    return dvals;
  };

 private:
  GrowInit grow{};
  FullInit full{};
};

/// PTC2 as per https://cs.gmu.edu/~sean/papers/treecreation.pdf
class PTC2Init final : public DiscreteInitBase {
 public:
  PTC2Init(std::optional<double> p_constant = std::nullopt) : p_constant(p_constant) {};

  Mat<DType> sample(Rng& rng, const InstanceBase& problem, usize count) const override final {
    try {
      const auto& ctx = dynamic_cast<const GPInstanceBase&>(problem).context();

      Mat<DType> dvals(count, problem.num_discrete());

      std::vector<std::vector<usize>> non_terminals(dvals.cols());        // functions/subfunctions
      std::vector<std::vector<usize>> const_terminals(dvals.cols());      // numeric constant symbols
      std::vector<std::vector<usize>> non_const_terminals(dvals.cols());  // features/args/params

      for (usize i = 0; i < problem.num_discrete(); i++) {
        usize domain_size = problem.discrete_domain_sizes()(i);
        for (usize domain_value = 0; domain_value < domain_size; domain_value++) {
          DType value = ctx.domain2value(i, domain_value);
          if (ctx.value_min_arity[value] > 0) {
            non_terminals[i].push_back(domain_value);
          } else if (ctx.value_kind[value] == ValueKind::Constant) {
            const_terminals[i].push_back(domain_value);
          } else {
            non_const_terminals[i].push_back(domain_value);
          }
        }
        if (const_terminals[i].empty() && non_const_terminals[i].empty()) {
          throw std::runtime_error("Variable domain must include terminal symbols.");
        }
      }

      usize num_roots = ctx.output_roots.size() + ctx.subtree_roots.size();
      for (isize i = 0; i < dvals.rows(); i++) {
        for (usize root_idx = 0; root_idx < num_roots; root_idx++) {
          usize root = root_idx < ctx.output_roots.size() ? ctx.output_roots[root_idx]
                                                          : ctx.subtree_roots[root_idx - ctx.output_roots.size()];
          sample_tree(rng, problem, ctx, non_terminals, const_terminals, non_const_terminals, dvals.row(i), root);
        }
      }
      return dvals;

    } catch (const std::bad_cast& e) {
      __goblin_runtime_assert(false);
    }
  };

 private:
  void sample_terminal(Rng& rng,
                       const InstanceBase& problem,
                       const GPContext& ctx,
                       const std::vector<std::vector<usize>>& non_terminals,
                       const std::vector<std::vector<usize>>& const_terminals,
                       const std::vector<std::vector<usize>>& non_const_terminals,
                       RefS<Vec<DType>> values,
                       usize idx) const {
    // 1. determine if it is a constant or other terminal
    double actual_p_constant = p_constant.value_or(
        static_cast<double>(const_terminals[idx].size()) /
        (static_cast<double>(const_terminals[idx].size()) + static_cast<double>(non_const_terminals[idx].size())));
    if (const_terminals[idx].empty()) {
      actual_p_constant = 0.0;
    }

    // 2. sample and place the terminal
    std::uniform_real_distribution<double> U(0.0, 1.0);
    if (U(rng) < actual_p_constant) {
      std::uniform_int_distribution<usize> ct_dist(0, const_terminals[idx].size() - 1);
      values(idx) = const_terminals[idx][ct_dist(rng)];
    } else {
      std::uniform_int_distribution<usize> nct_dist(0, non_const_terminals[idx].size() - 1);
      values(idx) = non_const_terminals[idx][nct_dist(rng)];
    }

    // 3. enqueue any children to ensure all nodes get initialized
    for (auto c : ctx.children[idx]) {
      sample_tree(rng, problem, ctx, non_terminals, const_terminals, non_const_terminals, values, c);
    }
  }

  void sample_tree(Rng& rng,
                   const InstanceBase& problem,
                   const GPContext& ctx,
                   const std::vector<std::vector<usize>>& non_terminals,
                   const std::vector<std::vector<usize>>& const_terminals,
                   const std::vector<std::vector<usize>>& non_const_terminals,
                   RefS<Vec<DType>> values,
                   usize idx) const {
    std::uniform_int_distribution<usize> size_dist(1, ctx.nodes[idx].size());
    usize target_size = size_dist(rng);

    if (target_size < 2) {
      sample_terminal(rng, problem, ctx, non_terminals, const_terminals, non_const_terminals, values, idx);
    } else {
      std::vector<usize> holes{idx};
      usize current_size = 0;

      // more functions are needed to reach the target size
      while (!holes.empty() && holes.size() + current_size < target_size) {
        // 1. get a random hole
        std::swap(holes[std::uniform_int_distribution<usize>(0, holes.size() - 1)(rng)], holes.back());
        usize i = holes.back();
        holes.pop_back();

        // 2. fill the hole
        if (ctx.children[i].empty()) {
          sample_terminal(rng, problem, ctx, non_terminals, const_terminals, non_const_terminals, values, i);
        } else {
          if (non_terminals[i].empty()) {
            throw std::runtime_error("Variable domain must include non-terminal symbols.");
          }
          std::uniform_int_distribution<usize> nt_dist(0, non_terminals[i].size() - 1);
          values(i) = non_terminals[i][nt_dist(rng)];
        }
        current_size++;

        // 3. update the list of holes and fill intron subtrees recursively
        usize value = ctx.domain2value(i, values(i));
        usize arity = ctx.value_max_arity[value];
        for (usize j = 0; j < ctx.children[i].size(); j++) {
          if (j < arity) {
            holes.push_back(ctx.children[i][j]);
          } else {
            sample_tree(rng, problem, ctx, non_terminals, const_terminals, non_const_terminals, values,
                        ctx.children[i][j]);
          }
        }
      }

      // once the size has been reached, fill the remaining nodes with terminals
      for (usize i : holes) {
        sample_terminal(rng, problem, ctx, non_terminals, const_terminals, non_const_terminals, values, i);
      }
    }
  };

  std::optional<double> p_constant{};
};

/// Complete init does not really work as intended for GP, since a lot of the diversity is distributed to inactive
/// individuals. This version aims to have the active and inactive values per variable be "probabilistically complete"
/// in isolation.
class RecursiveCompleteInit final : public DiscreteInitBase {
 public:
  Mat<DType> sample(Rng& rng, const InstanceBase& problem, usize count) const override final {
    try {
      const auto& ctx = dynamic_cast<const GPInstanceBase&>(problem).context();

      Mat<DType> dvals(count, problem.num_discrete());
      Arr2D<BType> dactive(count, problem.num_discrete());

      std::vector<usize> active_indices;
      active_indices.reserve(count);
      std::vector<usize> inactive_indices;
      inactive_indices.reserve(count);

      usize num_roots = ctx.output_roots.size() + ctx.subtree_roots.size();
      for (usize root_idx = 0; root_idx < num_roots; root_idx++) {
        usize root = root_idx < ctx.output_roots.size() ? ctx.output_roots[root_idx]
                                                        : ctx.subtree_roots[root_idx - ctx.output_roots.size()];
        std::vector<usize> queue{root};
        dactive.col(root) = true;

        while (!queue.empty()) {
          usize current = queue.back();
          queue.pop_back();

          active_indices.clear();
          inactive_indices.clear();
          for (usize i = 0; i < count; i++) {
            if (dactive(i, current)) {
              active_indices.push_back(i);
            } else {
              inactive_indices.push_back(i);
            }
          }

          // TODO is it better to maximize the number of active variables by sampling terminals only once?
          std::vector<DType> perm(problem.discrete_domain_sizes()(current));
          std::iota(perm.begin(), perm.end(), 0);

          usize i = perm.size();
          for (usize r : active_indices) {
            if (i >= perm.size()) {
              i = 0;
              std::shuffle(perm.begin(), perm.end(), rng);
            }

            dvals(r, current) = perm[i++];
          }
          for (usize r : inactive_indices) {
            if (i >= perm.size()) {
              i = 0;
              std::shuffle(perm.begin(), perm.end(), rng);
            }

            dvals(r, current) = perm[i++];
          }

          for (usize ci = 0; ci < ctx.children[current].size(); ci++) {
            usize c = ctx.children[current][ci];

            queue.push_back(c);
            for (usize i = 0; i < count; i++) {
              DType value = ctx.domain2value(current, dvals(i, current));
              // child is active if the parent (=current) is active
              // and the parent arity contains the child...
              dactive(i, c) = dactive(i, current) && ctx.value_max_arity[value] > ci;
            }
          }
        }
      }

      // for (isize c = 0; c < dvals.cols(); c++) {
      //   std::vector<usize> idx2domain;

      //   bool is_leaf = ctx.children[c].empty();

      //   for (usize domain_value = 0; domain_value < problem.discrete_domain_sizes()(c); domain_value++) {
      //     DType value = ctx.domain2value(c, domain_value);
      //     if (is_leaf ? ctx.value_min_arity[value] == 0  // terminals only
      //                 : ctx.value_min_arity[value] > 0   // non-terminals only
      //     ) {
      //       idx2domain.push_back(domain_value);
      //     }
      //   }

      //   __goblin_runtime_assert(idx2domain.size() >= 1);

      //   std::uniform_int_distribution<DType> d(0, idx2domain.size() - 1);
      //   for (usize i = 0; i < count; i++) {
      //     dvals(i, c) = idx2domain[d(rng)];
      //   }
      // }
      return dvals;

    } catch (const std::bad_cast& e) {
      return CompleteInit().sample(rng, problem, count);
    }
  };
};

/// Complete init does not really work as intended for GP, since a lot of the diversity is distributed to inactive
/// individuals. This version aims to have the active and inactive values per variable be "probabilistically complete"
/// in isolation.
class RecursiveCompleteInit2 final : public DiscreteInitBase {
 public:
  RecursiveCompleteInit2(std::optional<double> p_terminal = std::nullopt,
                         std::optional<double> p_constant = std::nullopt)
      : p_terminal(p_terminal), p_constant(p_constant) {};

  Mat<DType> sample(Rng& rng, const InstanceBase& problem, usize count) const override final {
    try {
      const auto& ctx = dynamic_cast<const GPInstanceBase&>(problem).context();

      Mat<DType> dvals(count, problem.num_discrete());
      Arr2D<BType> dactive(count, problem.num_discrete());

      std::vector<std::vector<DType>> non_terminals(dvals.cols());        // functions/subfunctions
      std::vector<std::vector<DType>> const_terminals(dvals.cols());      // numeric constant symbols
      std::vector<std::vector<DType>> non_const_terminals(dvals.cols());  // features/args/params

      for (usize i = 0; i < problem.num_discrete(); i++) {
        usize domain_size = problem.discrete_domain_sizes()(i);
        for (usize domain_value = 0; domain_value < domain_size; domain_value++) {
          DType value = ctx.domain2value(i, domain_value);
          if (ctx.value_min_arity[value] > 0) {
            non_terminals[i].push_back(domain_value);
          } else if (ctx.value_kind[value] == ValueKind::Constant) {
            const_terminals[i].push_back(domain_value);
          } else {
            non_const_terminals[i].push_back(domain_value);
          }
        }
        if (const_terminals[i].empty() && non_const_terminals[i].empty()) {
          throw std::runtime_error("Variable domain must include terminal symbols.");
        }
      }

      std::vector<usize> active_indices;
      active_indices.reserve(count);
      std::vector<usize> inactive_indices;
      inactive_indices.reserve(count);

      usize num_roots = ctx.output_roots.size() + ctx.subtree_roots.size();
      for (usize root_idx = 0; root_idx < num_roots; root_idx++) {
        usize root = root_idx < ctx.output_roots.size() ? ctx.output_roots[root_idx]
                                                        : ctx.subtree_roots[root_idx - ctx.output_roots.size()];
        std::vector<usize> queue{root};
        dactive.col(root) = true;

        while (!queue.empty()) {
          usize current = queue.back();
          queue.pop_back();

          active_indices.clear();
          inactive_indices.clear();
          for (usize i = 0; i < count; i++) {
            if (dactive(i, current)) {
              active_indices.push_back(i);
            } else {
              inactive_indices.push_back(i);
            }
          }

          dvals(active_indices, current) = sample_nodes(rng, problem, non_terminals, const_terminals,
                                                        non_const_terminals, active_indices.size(), current);
          dvals(inactive_indices, current) = sample_nodes(rng, problem, non_terminals, const_terminals,
                                                          non_const_terminals, inactive_indices.size(), current);

          for (usize ci = 0; ci < ctx.children[current].size(); ci++) {
            usize c = ctx.children[current][ci];

            queue.push_back(c);
            for (usize i = 0; i < count; i++) {
              DType value = ctx.domain2value(current, dvals(i, current));
              // child is active if the parent (=current) is active
              // and the parent arity contains the child...
              dactive(i, c) = dactive(i, current) && ctx.value_max_arity[value] > ci;
            }
          }
        }
      }

      return dvals;
    } catch (const std::bad_cast& e) {
      return CompleteInit().sample(rng, problem, count);
    }
  };

 private:
  Vec<DType> sample_nodes(Rng& rng,
                          const InstanceBase& problem,
                          const std::vector<std::vector<DType>>& non_terminals,
                          const std::vector<std::vector<DType>>& const_terminals,
                          const std::vector<std::vector<DType>>& non_const_terminals,
                          usize total,
                          usize idx) const {
    usize domain_size = problem.discrete_domain_sizes()(idx);

    double actual_p_terminal =
        p_terminal.value_or(1.0 - static_cast<double>(non_terminals[idx].size()) / static_cast<double>(domain_size));
    double actual_p_constant = p_constant.value_or(
        static_cast<double>(const_terminals[idx].size()) /
        (static_cast<double>(const_terminals[idx].size()) + static_cast<double>(non_const_terminals[idx].size())));
    if (const_terminals[idx].empty()) {
      actual_p_constant = 0.0;
    }

    usize num_non_terminals =
        non_terminals[idx].empty() ? 0 : static_cast<usize>((1.0 - actual_p_terminal) * static_cast<double>(total));
    usize num_const_terminals = actual_p_constant * static_cast<double>(total - num_non_terminals);
    usize num_non_const_terminals = total - num_non_terminals - num_const_terminals;

    Vec<DType> values(total);

    std::vector<DType> perm;
    sample_complete(rng, non_terminals[idx], values(Eigen::seqN(0, num_non_terminals)));
    sample_complete(rng, const_terminals[idx], values(Eigen::seqN(num_non_terminals, num_const_terminals)));
    sample_complete(rng, non_const_terminals[idx],
                    values(Eigen::seqN(num_non_terminals + num_const_terminals, num_non_const_terminals)));

    std::shuffle(values.begin(), values.end(), rng);

    return values;
  };

  void sample_complete(Rng& rng, const std::vector<DType>& pool, Ref<Vec<DType>> values) const {
    std::vector<DType> perm(pool.size());
    std::iota(perm.begin(), perm.end(), 0);

    usize i = perm.size();
    for (isize vi = 0; vi < values.size(); vi++) {
      if (i >= perm.size()) {
        i = 0;
        std::shuffle(perm.begin(), perm.end(), rng);
      }

      values(vi) = pool[perm[i++]];
    }
  };

  std::optional<double> p_terminal{};
  std::optional<double> p_constant{};
};

};  // namespace goblin

#endif /* _GOBLIN_GP_INIT_H */

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                       goblin/gp/sr.h included by goblin.h                                                    //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef _GOBLIN_GP_SR_H
#define _GOBLIN_GP_SR_H


#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                       goblin/gp/gpu_evaluation/launch_config.h included by goblin/gp/sr.h                    //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef _GOBLIN_GA_GP_LAUNCH_CONFIG_H
#define _GOBLIN_GA_GP_LAUNCH_CONFIG_H



#define MAX_THREADS_PER_BLOCK 1024
#define WARP_SIZE 32

namespace goblin {

constexpr size_t round_up(size_t value, size_t multiple) {
    return ((value + multiple - 1) / multiple) * multiple;
}

constexpr size_t ceil_div(size_t a, size_t b) {
    return (a + b - 1) / b;
}

struct KernelDim {
    size_t x = 1;
    size_t y = 1;
    size_t z = 1;

    KernelDim() = default;
    KernelDim(size_t _x, size_t _y = 1, size_t _z = 1) : x(_x), y(_y), z(_z) {}

    // Finds the thread count in [WARP_SIZE, MAX_THREADS_PER_BLOCK] (step WARP_SIZE)
    // that minimises idle threads when covering `count` items.
    static KernelDim determine(size_t count) {
        KernelDim dim{WARP_SIZE};
        size_t min_redundant = MAX_THREADS_PER_BLOCK;

        for (size_t threads = MAX_THREADS_PER_BLOCK; threads > 0; threads -= WARP_SIZE) {
            // Round up division to determine number of blocks needed
            size_t blocks_needed = ceil_div(count, threads);
            size_t redundant = (blocks_needed * threads) - count;

            if (redundant < min_redundant) {
                min_redundant = redundant;
                dim.x = threads;
            }

            // Early exit if perfect fit is found
            if (redundant == 0) {
                break;
            }
        }

        return dim;
    }

    void check() const { assert(x * y * z <= MAX_THREADS_PER_BLOCK); }

    constexpr bool operator==(const KernelDim& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

struct KernelConfig {
    KernelDim grid;
    KernelDim block;

    KernelConfig() = default;
    KernelConfig(KernelDim _grid, KernelDim _block) : grid(_grid), block(_block) {}

    // One block per solution; threads cover datapoints. Used by Baseline/Restrict/SharedMemory/BlockReduce.
    static KernelConfig for_eval(size_t num_solutions, size_t num_datapoints) {
        KernelConfig config;
        config.block   = KernelDim::determine(num_datapoints);
        config.grid.x  = num_solutions;
        config.grid.y  = ceil_div(num_datapoints, config.block.x);
        return config;
    }

    // One block per solution; threads cover all datapoints in a single pass. Used by SingleKernel variants.
    static KernelConfig for_eval_single(size_t num_solutions, size_t num_datapoints) {
        KernelConfig config;
        config.grid.x  = num_solutions;
        config.block.x = std::min((size_t)MAX_THREADS_PER_BLOCK, round_up(num_datapoints, WARP_SIZE));
        return config;
    }

    // Multiple blocks per solution; blocks split the datapoints. Used by Hybrid.
    static KernelConfig for_eval_hybrid(size_t num_solutions, size_t num_datapoints, size_t blocks_per_individual) {
        const size_t datapoints_per_block = ceil_div(num_datapoints, blocks_per_individual);
        KernelConfig config;
        config.grid.x  = num_solutions;
        config.grid.y  = blocks_per_individual;
        config.block.x = std::min((size_t)MAX_THREADS_PER_BLOCK, round_up(datapoints_per_block, WARP_SIZE));
        return config;
    }

    // One thread per solution for the MSE reduction. Used by Baseline/Restrict/SharedMemory.
    static KernelConfig for_mse_simple(size_t num_solutions) {
        KernelConfig config;
        config.block  = KernelDim::determine(num_solutions);
        config.grid.x = ceil_div(num_solutions, config.block.x);
        return config;
    }

    // One block per solution for the MSE reduction. Used by BlockReduce and Hybrid.
    static KernelConfig for_mse_block(size_t num_solutions, size_t num_partial) {
        KernelConfig config;
        config.grid.x  = num_solutions;
        config.block.x = std::min((size_t)MAX_THREADS_PER_BLOCK, round_up(num_partial, WARP_SIZE));
        return config;
    }

    void check() const { block.check(); }

    constexpr bool operator==(const KernelConfig& other) const {
        return grid == other.grid && block == other.block;
    }
};

struct LaunchConfig {
    KernelConfig eval;
    KernelConfig mse;
    KernelVersion kernel_version    = KernelVersion::Baseline;
    size_t num_solutions            = 0;
    size_t num_datapoints           = 0;
    size_t solution_length          = 0;
    size_t blocks_per_individual    = 1;
    size_t datapoints_per_block     = 0;
    size_t datapoints_per_thread    = 0;

    LaunchConfig() = default;

    LaunchConfig(
        KernelConfig eval,
        KernelConfig mse,
        KernelVersion version = KernelVersion::Baseline
    ) : eval(eval),
        mse(mse),
        kernel_version(version) {}

    static LaunchConfig determine(
        KernelVersion kernel_version,
        size_t num_solutions,
        size_t num_datapoints,
        size_t solution_length,
        std::optional<size_t> num_sms
    ) {
        LaunchConfig config;
        config.kernel_version  = kernel_version;
        config.num_solutions   = num_solutions;
        config.num_datapoints  = num_datapoints;
        config.solution_length = solution_length;

        switch (kernel_version) {
            case KernelVersion::Baseline:
            case KernelVersion::Restrict:
            case KernelVersion::SharedMemory:
                config.eval = KernelConfig::for_eval(num_solutions, num_datapoints);
                config.mse  = KernelConfig::for_mse_simple(num_solutions);
                break;

            case KernelVersion::BlockReduce:
                config.eval = KernelConfig::for_eval(num_solutions, num_datapoints);
                config.mse  = KernelConfig::for_mse_block(num_solutions, config.eval.grid.y);
                break;

            case KernelVersion::SingleKernel:
            case KernelVersion::SingleKernelFMAF:
            case KernelVersion::SingleKernelInplace:
                config.eval = KernelConfig::for_eval_single(num_solutions, num_datapoints);
                config.datapoints_per_thread = ceil_div(num_datapoints, config.eval.block.x);
                break;

            case KernelVersion::Hybrid:
                __goblin_runtime_assert(num_sms.has_value());

                if (num_solutions * 2 <= num_sms.value()) {
                    config.blocks_per_individual = num_sms.value() / num_solutions;
                    config.eval = KernelConfig::for_eval_hybrid(
                        num_solutions, num_datapoints, config.blocks_per_individual);
                    config.datapoints_per_block  = ceil_div(num_datapoints, config.blocks_per_individual);
                    config.datapoints_per_thread = ceil_div(config.datapoints_per_block, config.eval.block.x);
                    config.mse = KernelConfig::for_mse_block(num_solutions, config.blocks_per_individual);
                } else {
                    // Not enough SMs to benefit from multi-block; fall back to single-block.
                    return LaunchConfig::determine(
                        KernelVersion::SingleKernelInplace, num_solutions, num_datapoints, solution_length, num_sms);
                }
                break;

            default:
                break;
        }

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

}  // namespace goblin

#endif /* _GOBLIN_GA_GP_LAUNCH_CONFIG_H */

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                       goblin/gp/sr.h continued                                                               //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef GOBLIN_HAS_CUDA

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                       goblin/gp/gpu_evaluation/evaluate.h included by goblin/gp/sr.h                         //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef _GOBLIN_GA_GP_EVAL_KERNEL_H
#define _GOBLIN_GA_GP_EVAL_KERNEL_H



namespace goblin {

using u8 = std::uint8_t;

#ifdef __CUDACC__
__global__
void evaluate_kernel_baseline(
    float* X,
    float* Y,
    const u8* v_type,
    float* v_value,
    float* partial,
    size_t solution_length,
    size_t num_datapoints
);

__global__
void evaluate_kernel_restrict(
    const float* __restrict__ X,
    const float* __restrict__ Y,
    const u8* __restrict__ v_type,
    const float* __restrict__ v_value,
    float* __restrict__ v,
    size_t solution_length,
    size_t num_datapoints
);

__global__
void evaluate_kernel_shared_memory(
    const float* __restrict__ X,
    const float* __restrict__ Y,
    const u8* __restrict__ v_type,
    const float* __restrict__ v_value,
    float* __restrict__ partial,
    size_t solution_length,
    size_t num_datapoints
);

__device__
float compute_tree_output_baseline(
    float* X,
    const u8* type,
    const float* value,
    size_t solution_length,
    size_t num_datapoints,
    size_t datapoint_index
);

__device__
float compute_tree_output_restrict(
    const float* __restrict__ X,
    const u8* __restrict__ type,
    const float* __restrict__ value,
    size_t solution_length,
    size_t num_datapoints,
    size_t datapoint_index
);

__device__
float compute_tree_output_inplace(
    const float* __restrict__ X,
    const u8* __restrict__ type,
    const float* __restrict__ value,
    size_t solution_length,
    size_t num_datapoints,
    size_t datapoint_index
);

__global__
void compute_mse_kernel_baseline(
    const float* __restrict__ partial,
    float* __restrict__ result,
    size_t num_solutions,
    size_t num_datapoints
);

__global__
void mse_kernel_restrict(
    const float* __restrict__ partial,
    float* __restrict__ result,
    size_t num_solutions,
    size_t num_datapoints
);

__global__
void evaluate_kernel_hybrid(
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
void compute_tree_output_wrapper(
    float* X,
    const u8* type,
    float* value,
    float* result,
    size_t solution_length,
    size_t num_datapoints,
    size_t datapoint_index,
    KernelVersion version
);
#endif

void evaluate_kernel_wrapper(
    float* X,
    float* Y,
    u8* type,
    float* value,
    float* partial,
    LaunchConfig config
);

void mse_kernel_wrapper(
    float* partial,
    float* result,
    LaunchConfig config
);

void evaluate_mse_kernel_wrapper(
    float* X,
    float* Y,
    u8* type,
    float* value,
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

float test_compute_output_kernel(
    std::vector<float> h_X,
    std::vector<u8> h_type,
    std::vector<float> h_value,
    size_t num_datapoints,
    size_t datapoint_index,
    KernelVersion version
);

std::vector<float> test_evaluate_kernel(
    std::vector<float> h_X,
    std::vector<float> h_Y,
    std::vector<u8> h_type,
    std::vector<float> h_value,
    size_t num_solutions,
    size_t num_datapoints,
    KernelVersion version
);

std::vector<float> test_compute_mse_kernel(
    std::vector<float> partial,
    size_t num_solutions,
    size_t num_datapoints,
    KernelVersion version
);

std::vector<float> test_evaluate_mse_kernel(
    std::vector<float> h_X,
    std::vector<float> h_Y,
    std::vector<u8> h_type,
    std::vector<float> h_value,
    size_t num_solutions,
    size_t num_datapoints,
    KernelVersion version
);

std::vector<float> test_kernel_hybrid(
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
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                       goblin/gp/gpu_evaluation/memory.h included by goblin/gp/sr.h                           //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef _GOBLIN_GA_GP_HELPER_H
#define _GOBLIN_GA_GP_HELPER_H

namespace goblin {

#ifdef __CUDACC__
void check(cudaError_t err, char const* func, char const* file, int line);
#endif

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

};  // namespace goblin

#endif /* _GOBLIN_GA_GP_HELPER_H */
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                       goblin/gp/gpu_evaluation/misc.h included by goblin/gp/sr.h                             //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
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

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                       goblin/gp/sr.h continued                                                               //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif // GOBLIN_HAS_CUDA



namespace goblin {

class SRQuality : public MOQuality {
 public:
  std::unique_ptr<QualityBase> clone() const override { return std::make_unique<SRQuality>(*this); };

  /// Linear scaling parameters
  Arr2D<CType> ls_params{};

  /*
  The test accuracy uses interior mutability (i.e. it ignores const) since it is not
  part of what defines a solution or its accuracy - as indicated by the name, it is never
  used to make any decisions and only tracked for analysis purposes. By making it mutable
  it an be lazily computed only when requested.
   */
  /// Optional test set accuracy
  mutable std::optional<MOQuality> test_quality = std::nullopt;
};

class SRFitness : public MOFitness {
 public:
  SRFitness(usize num_objectives, bool minimize = true, CType epsilon = 0.0)
      : MOFitness(num_objectives, minimize, epsilon) {}

  std::unique_ptr<QualityBase> worst() const override final {
    const CType inf = std::numeric_limits<CType>().infinity();
    auto q = std::make_unique<SRQuality>();
    q->objectives = Vec<CType>::Constant(num_objectives(), inf);
    q->constraint_value = inf;
    return q;
  };
};

class SRProblem : public GPInstanceBase {
  using ScalarType = CType;  // TODO template the implementation and add a wrapper class - by doing so the wrapper can
                             // at compile time delegate to different ScalarTypes (float, double, mpfr, autodiff
                             // versions) while still having a nice Python API...

 public:
  SRProblem(GPContext ctx,
            Arr2D<CType> X_train,
            Arr2D<CType> Y_train,
            std::optional<Arr2D<CType>> X_test = std::nullopt,
            std::optional<Arr2D<CType>> Y_test = std::nullopt,
            std::variant<std::string, std::vector<std::string>> objectives =
                "mse",  /// The objectives that should be recorded in the archive, and by default the objectives that
                        /// are optimized.

            // TODO I really don't like this API - it is the way it is because this way is convenient for
            // the Python bindings, but makes adding custom objectives hard. Dependency injection would be
            // better, but isn't perfect either - a fully decoupled design ("given the solution, give me the
            // objective value") would potentially mean recomputing the output multiple times, and passing a
            // few fixed values (e.g. the solution, the output, the size) isn't enough for some objectives
            // that could be interesting (e.g. diversity, effective information criterion,...). Until I have
            // a better API design, I will leave this hardcoded, and maybe look at how other GP/SR libraries
            // do it to see if there are better solutions.

            std::optional<usize> objectives_to_optimize =
                std::nullopt,  /// The number of objectives to optimize in case those differ from the `objectives`
                               /// parameter, corresponds to the first `objectives_to_optimize` entries in `objectives`.
            bool linear_scaling = true,
            std::optional<AnyInit> init = std::nullopt,
            CType constant_init_lower_bound = -1.0,
            CType constant_init_upper_bound = 1.0,
            std::optional<std::vector<CType>> target_objectives = std::nullopt,
            std::string gradient_mode = "forward",
            CType gradient_epsilon = 1e-5,
            CType archive_epsilon = 0.0,
            std::optional<bool> always_inherit_continuous = std::nullopt,
            std::optional<usize> batch_size = std::nullopt,
            std::optional<KernelVersion> kernel_version =
                std::nullopt  /// When set, evaluation runs on the GPU using the specified kernel. Requires GOBLIN_HAS_CUDA.
            )
      : ctx(ctx),
        linear_scaling(linear_scaling),
        objectives(std::holds_alternative<std::string>(objectives)
                       ? std::vector<std::string>{std::get<std::string>(objectives)}
                       : std::get<std::vector<std::string>>(objectives)),
        X_train(X_train.cast<ScalarType>()),
        Y_train(Y_train.cast<ScalarType>()),
        _archive_fitness(SRFitness(this->objectives.size(), /* minimize = */ true, archive_epsilon)),
        _fitness(SRFitness(objectives_to_optimize.value_or(this->objectives.size()))),
        _init(from_any_init(init.value_or(std::make_shared<HalfHalfInit>()))),
        _target(_archive_fitness),
        _gradient_mode(gradient_mode),
        _gradient_epsilon(gradient_epsilon),
        _always_inherit_continuous(always_inherit_continuous),
        _batch_size(batch_size) {
#ifdef GOBLIN_HAS_CUDA
    _kernel_version = kernel_version;
#else
    __goblin_runtime_assert(!kernel_version.has_value() && "GPU evaluation requires GOBLIN_HAS_CUDA");
    (void)kernel_version;
#endif
    __goblin_runtime_assert(this->objectives.size() > 0);
    __goblin_runtime_assert(
        !objectives_to_optimize.has_value() ||
        (objectives_to_optimize.value() > 0 && objectives_to_optimize.value() <= this->objectives.size()));

    _num_continuous = this->ctx.num_continuous;
    _continuous_upper_bounds = Vec<CType>::Constant(_num_continuous, std::numeric_limits<CType>::max());
    _continuous_lower_bounds = -_continuous_upper_bounds;

    __goblin_runtime_assert(!isna(constant_init_lower_bound));
    __goblin_runtime_assert(!isna(constant_init_upper_bound));
    __goblin_runtime_assert(constant_init_lower_bound < constant_init_upper_bound);
    _continuous_init_lower_bounds = Vec<CType>::Constant(_num_continuous, constant_init_lower_bound);
    _continuous_init_upper_bounds = Vec<CType>::Constant(_num_continuous, constant_init_upper_bound);

    __goblin_runtime_assert(this->X_train.rows() == this->Y_train.rows());
    __goblin_runtime_assert(static_cast<usize>(this->X_train.cols()) == ctx.num_inputs);
    __goblin_runtime_assert(static_cast<usize>(this->Y_train.cols()) == ctx.num_outputs);

    var_Y_train = (this->Y_train.rowwise() - this->Y_train.colwise().mean()).square().colwise().mean();
    // ~0 => 1 (R2 is not defined, so we fall back to the MSE by not
    // normalizing...)
    for (isize i = 0; i < var_Y_train.size(); i++) {
      if (std::abs(var_Y_train(i)) < CType(1e-12)) {
        var_Y_train(i) = 1.0;
      }
    }

    if (X_test.has_value()) {
      __goblin_runtime_assert(Y_test.has_value());
      this->X_test = X_test.value().cast<ScalarType>();
      this->Y_test = Y_test.value().cast<ScalarType>();

      __goblin_runtime_assert(this->X_train.cols() == this->X_test.cols());
      __goblin_runtime_assert(this->Y_train.cols() == this->Y_test.cols());
      __goblin_runtime_assert(this->X_test.rows() == this->Y_test.rows());

      var_Y_test = (this->Y_test.rowwise() - this->Y_test.colwise().mean()).square().colwise().mean();
      // ~0 => 1 (R2 is not defined, so we fall back to the MSE by not
      // normalizing...)
      for (isize i = 0; i < var_Y_test.size(); i++) {
        if (std::abs(var_Y_test(i)) < CType(1e-12)) {
          var_Y_test(i) = 1.0;
        }
      }
    }

    if (target_objectives.has_value()) {
      register_target(target_objectives.value());
    }

#ifdef GOBLIN_HAS_CUDA
    if (_kernel_version.has_value()) {
      _init_gpu();
    }
#endif
  };

#ifdef GOBLIN_HAS_CUDA
  ~SRProblem() { free_gpu(); }

  /// Switch between CPU (nullopt) and GPU (KernelVersion) evaluation at runtime.
  /// Copies training data to the GPU the first time GPU mode is enabled.
  void set_kernel_version(std::optional<KernelVersion> kernel_version) {
    if (kernel_version.has_value() && d_X == nullptr) {
      _copy_data_to_gpu();
    }
    _kernel_version = kernel_version;
  }

  /// Free all GPU memory. Does not change the kernel version.
  void free_gpu() {
    _free_data_on_gpu();
    _free_solution_on_gpu();
    _free_results_on_gpu();
  }
#endif

  bool adapt(Rng& rng) override final {
    if (_batch_size.has_value() && _batch_size.value() < static_cast<usize>(X_train.rows())) {
      // TODO refactor out into something like PyTorch's DataLoader/Sampler and allow more sophisticated sampling
      // strategies Surprisingly (?) PyTorch does not have any fancy strategies that take the data distribution into
      // account (https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.Sampler) I'd expect something like a
      // "parallel greedy scattered subset selection" to perform well since each batch tries to represent the whole
      // training data distribution (i.e. in random order, assign the furthest row to the current batch until all rows
      // are assigned to a batch, where the number of batches is ceil(dataset_size / batch_size) - so basically
      // stratified sampling)
      auto perm = permute(rng, X_train.rows());
      perm.resize(_batch_size.value());

      X_batch = X_train(perm, Eigen::placeholders::all);
      Y_batch = Y_train(perm, Eigen::placeholders::all);

      var_Y_batch = (Y_batch.rowwise() - Y_batch.colwise().mean()).square().colwise().mean();
      return true;
    } else {
      return false;
    }
  };

  CRef<Vec<DType>> discrete_domain_sizes() const override final { return ctx.domain_sizes; };

  CRef<Vec<CType>> continuous_lower_bounds() const override final { return _continuous_lower_bounds; };
  CRef<Vec<CType>> continuous_upper_bounds() const override final { return _continuous_upper_bounds; };

  CRef<Vec<CType>> continuous_init_lower_bounds() const override final { return _continuous_init_lower_bounds; };
  CRef<Vec<CType>> continuous_init_upper_bounds() const override final { return _continuous_init_upper_bounds; };

  void evaluate(Rng& rng, SolutionSetBase& solutions, const std::span<const usize>& indices) override final {
#ifdef GOBLIN_HAS_CUDA
    if (_kernel_version.has_value()) {
      _evaluate_gpu(solutions, indices);
      return;
    }
#endif
    _evaluate_cpu(rng, solutions, indices);
  };

  void add_random(Rng& rng, SolutionSetBase& solutions, usize count) const override final {
    _init->add_random(rng, *this, solutions, count);
#ifndef NDEBUG
    auto p = dynamic_cast<SRQuality*>(&solutions[solutions.size() - 1].quality());
    assert(p != nullptr && "Quality mismatch");
#endif
  };

  const FitnessBase& fitness() const override final { return _fitness; };

  const ArchiveFitnessBase& archive_fitness() const override final { return _archive_fitness; };

  virtual std::tuple<bool, bool> inherit_discrete(SolutionBase& offspring,
                                                  const SolutionBase& donor,
                                                  const Subset& subset) const override {
    const bool inherit_continuous = _always_inherit_continuous.value_or(ctx.const_repr == ConstantRepr::ERCs ||
                                                                        ctx.const_repr == ConstantRepr::Edges) &&
                                    ctx.const_repr != ConstantRepr::None;

    // the pool size is not tied to the number of discrete variables, so the full pool instead of the paired values is
    // inherited...
    const bool inherit_by_index = ctx.const_repr != ConstantRepr::Pool;

    bool any_active_changed = false, anything_changed = false;
    for (usize i : subset.discrete) {
      if (offspring.discrete_values()(i) != donor.discrete_values()(i)) {
        any_active_changed |= offspring.discrete_active()(i);
        anything_changed = true;
        offspring.discrete_values()(i) = donor.discrete_values()(i);
      }

      // TODO for GCS: inherit child arities + permutations

      if (inherit_continuous && inherit_by_index) {
        // TODO sufficiently relatively + absolutely different or no check, but floating point equality is not really
        // useful...
        //
        // yes, the indices here should be from the discrete subset!
        if (offspring.continuous_values()(i) != donor.continuous_values()(i)) {
          any_active_changed |= offspring.continuous_active()(i);
          anything_changed = true;
          offspring.continuous_values()(i) = donor.continuous_values()(i);
        }
      }
    }

    if (inherit_continuous && !inherit_by_index) {
      // note: arguably just inheriting all continuous variables even if the inherited discrete values might not even be
      // constants is not the best idea - but earlier experiments on another codebase suggested that more
      // appropriate/interpolating continuous mixing doesn't really work and here it also is more for completeness and
      // not used by default...
      for (usize i = 0; i < num_continuous(); i++) {
        // TODO sufficiently relatively + absolutely different or no check, but floating point equality is not really
        // useful...
        if (offspring.continuous_values()(i) != donor.continuous_values()(i)) {
          any_active_changed |= offspring.continuous_active()(i);
          anything_changed = true;
          offspring.continuous_values()(i) = donor.continuous_values()(i);
        }
      }
    }

    return std::make_tuple(any_active_changed, anything_changed);
  }

  std::optional<CType> as_continuous(const SolutionBase& solution, usize discrete_index) const override final {
    auto value = ctx.domain2value(discrete_index, solution.discrete_values()(discrete_index));
    if (ctx.value_kind[value] == ValueKind::Constant) {
      return solution.continuous_values()(ctx.const_repr == ConstantRepr::Pool ? ctx.value_idx[value] : discrete_index);
    }
    return std::nullopt;
  };

  void register_target(CRefS<Vec<CType>> target_objectives) {
    _target.clear();
    Solution s(
        archive_fitness().worst(),
        num_discrete() > 0 ? std::make_optional<Vec<DType>>(Vec<DType>::Zero(num_discrete())) : std::nullopt,
        num_continuous() > 0 ? std::make_optional<Vec<CType>>(Vec<CType>::Zero(num_continuous())) : std::nullopt);
    s.quality_as<SRQuality>().objectives = target_objectives;
    __goblin_runtime_assert(static_cast<usize>(s.quality_as<SRQuality>().objectives.size()) >=
                            fitness().num_objectives());
    s.quality_as<SRQuality>().constraint_value = 0.0;
    _target.update(s, false);
  };

  void register_target(std::vector<CType> target_objectives) {
    register_target(Eigen::Map<Vec<CType>>(target_objectives.data(), target_objectives.size()));
  };

  bool target_reached(const ArchiveBase& archive) const override final {
    if (!_target.empty()) {
      return archive.covers(_target);
    } else {
      return false;
    }
  };

  void log_header(std::ostream& os) const override final {
    for (usize i = 0; i < ctx.num_subexpressions; i++) {
      os << "subtree_" << i << ",";
    }
    os << "expressions,";
    for (auto& o : objectives) {
      os << o << "_train,";
    }
    if (Y_test.size() > 0) {
      for (auto& o : objectives) {
        os << o << "_test,";
      }
    }

    archive_fitness().log_header(os);
  };

  void log(std::ostream& os, const SolutionBase& solution) const override final {
    for (const auto& expr : ctx.subtrees_to_sympy(solution)) {
      os << '"' << expr << "\",";
    }
    os << '"';
    log_solution(os, solution);
    os << "\",";

    const auto& q = solution.quality_as<SRQuality>();
    for (usize i = 0; i < objectives.size(); i++) {
      os << q.objectives(i) << ',';
    }
    if (Y_test.size() > 0) {
      if (!q.test_quality.has_value()) {
        evaluate_test(solution);
      }
      for (usize i = 0; i < objectives.size(); i++) {
        os << q.test_quality.value().objectives(i) << ',';
      }
    }
    archive_fitness().log(os, solution.quality());
  };

  void log_solution(std::ostream& os, const SolutionBase& solution) const override final {
    auto exprs = ctx.to_sympy(solution);
    for (usize i = 0; i < exprs.size(); i++) {
      if (i > 0) {
        os << " , ";
      }
      if (linear_scaling) {
        const auto& q = solution.quality_as<SRQuality>();
        if (static_cast<usize>(q.ls_params.cols()) != ctx.num_outputs) {
          os << exprs[i];  // for the edge case where unevaluated solutions are logged...
        } else {
          os << q.ls_params(0, i) << " + (" << q.ls_params(1, i) << " * (" << exprs[i] << "))";
        }
      } else {
        os << exprs[i];
      }
    }
  };

  void evaluate_test(const SolutionBase& solution) const {
    const auto& q = solution.quality_as<SRQuality>();
    if (Y_test.size() > 0 && !q.test_quality.has_value()) {
      Solution copy = solution;  // copy is needed since active variables are not mutable...

      Array<ScalarType> params;  // TODO fit FC params...
      auto& cq = copy.quality_as<SRQuality>();
      eval_one(copy, X_test, Y_test, var_Y_test, params, false, cq, cq.ls_params);

      q.test_quality = MOQuality();
      q.test_quality.value().objectives = copy.quality_as<SRQuality>().objectives;
      q.test_quality.value().constraint_value = copy.quality_as<SRQuality>().constraint_value;
    }
  };

  const GPContext& context() const override final { return ctx; }

  std::tuple<std::vector<usize>, u64> gradient_steps(Rng& rng,
                                                     SolutionSetBase& solutions,
                                                     SolutionSetBase& parents,
                                                     const std::span<const usize>& indices,
                                                     usize num_steps) override final;

  GPContext ctx;
  bool linear_scaling;
  std::vector<std::string> objectives;
  Arr2D<ScalarType> X_train;
  Arr2D<ScalarType> Y_train;
  Array<ScalarType> var_Y_train;
  Arr2D<ScalarType> X_batch{};
  Arr2D<ScalarType> Y_batch{};
  Array<ScalarType> var_Y_batch{};
  Arr2D<ScalarType> X_test;
  Arr2D<ScalarType> Y_test;
  Array<ScalarType> var_Y_test;

 private:
  // --- CPU evaluation ---

  void _evaluate_cpu(Rng& rng, SolutionSetBase& solutions, const std::span<const usize>& indices) {
    // initialize the first batch if needed
    if (_batch_size.has_value() && X_batch.size() == 0) {
      adapt(rng);
    }

    Array<ScalarType> params;
    for (auto i : indices) {
      auto& q = solutions[i].quality_as<SRQuality>();
      q.test_quality = std::nullopt;  // Non test evaluations indicate that the test quality is likely out of date...
      if (_batch_size.has_value()) {
        eval_one(solutions[i], X_batch, Y_batch, var_Y_batch, params, true, q, q.ls_params);
      } else {
        eval_one(solutions[i], X_train, Y_train, var_Y_train, params, true, q, q.ls_params);
      }
    }
  }

  // solution, X, Y, train/test, quality to write to
  void eval_one(SolutionBase& solution,
                const Arr2D<ScalarType>& X,
                const Arr2D<ScalarType>& Y,
                const Array<ScalarType>& var_Y,
                const Array<ScalarType>& params,
                bool is_train,
                MOQuality& quality,
                Arr2D<CType>& ls_params) const {
    usize expression_size;
    auto out = ctx.compute_outputs(_eval_buffer, solution, X, params, expression_size);

    if (!out.has_value()) {
      quality.objectives.array() = std::numeric_limits<CType>::infinity();
      quality.constraint_value = 1.0;
      return;
    }

    Arr2D<ScalarType> Y_pred = out.value();
    if (linear_scaling && is_train) {
      Mat<ScalarType> A_ls = Mat<ScalarType>::Ones(Y_train.rows(), 2);
      ls_params.resize(2, ctx.num_outputs);
      for (usize o = 0; o < ctx.num_outputs; o++) {
        A_ls.col(1) = Y_pred.col(o);
        Vec<ScalarType> b = A_ls.colPivHouseholderQr().solve(Y_train.matrix().col(o));
        ls_params.col(o) = A_ls.colPivHouseholderQr().solve(Y_train.matrix().col(o));
      }
    }

    quality.constraint_value = 0.0;
    for (usize j = 0; j < objectives.size(); j++) {
      if (objectives[j] == "mse") {
        if (linear_scaling) {
          quality.objectives(j) = 0.0;
          for (usize o = 0; o < ctx.num_outputs; o++) {
            CType intercept = ls_params(0, o);  // solution.continuous_values()(ctx.num_continuous + 2 * o);
            CType slope = ls_params(1, o);      // solution.continuous_values()(ctx.num_continuous + 2 * o + 1);
            quality.objectives(j) += ((intercept + slope * Y_pred.col(o)) - Y.col(o)).square().mean();
          }
        } else {
          quality.objectives(j) = (Y_pred - Y).square().mean();
        }
      } else if (objectives[j] == "nmse") {
        if (linear_scaling) {
          quality.objectives(j) = 0.0;
          for (usize o = 0; o < ctx.num_outputs; o++) {
            CType intercept = ls_params(0, o);  // solution.continuous_values()(ctx.num_continuous + 2 * o);
            CType slope = ls_params(1, o);      // solution.continuous_values()(ctx.num_continuous + 2 * o + 1);
            quality.objectives(j) += ((intercept + slope * Y_pred.col(o)) - Y.col(o)).square().mean() / var_Y(o);
          }
        } else {
          quality.objectives(j) = 0.0;
          for (usize o = 0; o < ctx.num_outputs; o++) {
            quality.objectives(j) += (Y_pred.col(o) - Y.col(o)).square().mean() / var_Y(o);
          }
        }
      } else if (objectives[j] == "size") {
        quality.objectives(j) = static_cast<CType>(expression_size);
      }
    }
  };

  // --- GPU evaluation ---
#ifdef GOBLIN_HAS_CUDA
  void _evaluate_gpu(SolutionSetBase& solutions, const std::span<const usize>& indices) {
    usize expression_size;
    size_t num_solutions = indices.size();

    if (num_solutions == 0) {
      return;
    }

    // Transform solutions to GPU compatible representation
    std::vector<u8> node_type;
    std::vector<float> node_value;
    std::vector<bool> overflowed(num_solutions, false);

    size_t k = 0;
    for (auto i : indices) {
      overflowed[k++] = ctx.to_gpu_representation(solutions[i], node_type, node_value, expression_size);
    }

    __goblin_runtime_assert(node_type.size() == node_value.size());

    const size_t solution_length = ctx.max_expression_size;
    const LaunchConfig config = LaunchConfig::determine(_kernel_version.value(), num_solutions, _num_datapoints, solution_length, _num_sms);

    // const LaunchConfig config = (_kernel_version.value() == KernelVersion::SingleKernelInplace)
        // ? LaunchConfig::determine_auto(num_solutions, X_train.rows(), solution_length, _num_sms)
        // : LaunchConfig::determine(_kernel_version.value(), num_solutions, X_train.rows(), solution_length);

    config.check();

    _copy_solutions_to_gpu(node_type, node_value);
    _allocate_results_on_gpu(config);

    kernel_wrapper(d_X, d_Y, d_type, d_value, d_partial, d_result, config);

    std::vector<float> result(num_solutions);
    copy_from_device(result.data(), d_result, num_solutions);

    k = 0;
    for (auto i : indices) {
      auto& quality = solutions[i].quality_as<SRQuality>();
      if (overflowed[k]) {
        quality.objectives.array() = std::numeric_limits<CType>::infinity();
        quality.constraint_value = 1.0;
      } else {
        quality.constraint_value = 0.0;
        quality.objectives(0) = result[k];
      }
      k++;
    }
  }

  void _init_gpu() {
    _num_sms = get_gpu_info().num_sms;
    _num_datapoints = X_train.rows();

    _copy_data_to_gpu();
  }

  void _copy_data_to_gpu() {
    Arr2D<float> X32 = X_train.cast<float>();
    Arr2D<float> Y32 = Y_train.cast<float>();
    d_X = allocate_and_copy(X32.data(), X32.size());
    d_Y = allocate_and_copy(Y32.data(), Y32.size());
  }

  void _copy_solutions_to_gpu(std::vector<u8> node_type, std::vector<float> node_value) {
    const size_t num_elements = node_type.size();

    if (_num_solutions_allocated < num_elements) {
      _free_solution_on_gpu();
      d_type = allocate_and_copy(node_type.data(), node_type.size());
      d_value = allocate_and_copy(node_value.data(), node_value.size());
      _num_solutions_allocated = num_elements;
    } else {
      copy_to_gpu(d_type, node_type.data(), node_type.size());
      copy_to_gpu(d_value, node_value.data(), node_value.size());
    }
  }

  void _allocate_results_on_gpu(const LaunchConfig& config) {
    if (config.kernel_version != KernelVersion::SingleKernel &&
        config.kernel_version != KernelVersion::SingleKernelFMAF &&
        config.kernel_version != KernelVersion::SingleKernelInplace) {
      const size_t num_partials =
          (config.kernel_version == KernelVersion::BlockReduce ||
           config.kernel_version == KernelVersion::Hybrid)
              ? config.num_solutions * config.eval.grid.y
              : config.num_solutions * config.num_datapoints;

      if (_num_partials_allocated < num_partials) {
        free_on_gpu(d_partial);
        d_partial = allocate_on_gpu<float>(num_partials);
        _num_partials_allocated = num_partials;
      }
    }

    if (_num_results_allocated < config.num_solutions) {
      free_on_gpu(d_result);
      d_result = allocate_on_gpu<float>(config.num_solutions);
      _num_results_allocated = config.num_solutions;
    }
  }

  void _free_data_on_gpu() {
    free_on_gpu(d_X);
    free_on_gpu(d_Y);
    d_X = nullptr;
    d_Y = nullptr;
  }

  void _free_solution_on_gpu() {
    free_on_gpu(d_type);
    free_on_gpu(d_value);
    d_type = nullptr;
    d_value = nullptr;
    _num_solutions_allocated = 0;
  }

  void _free_results_on_gpu() {
    free_on_gpu(d_partial);
    free_on_gpu(d_result);
    d_partial = nullptr;
    d_result = nullptr;
    _num_partials_allocated = 0;
    _num_results_allocated = 0;
  }
#endif // GOBLIN_HAS_CUDA

  SRFitness _archive_fitness;
  SRFitness _fitness;
  std::shared_ptr<InitBase> _init;
  UnboundedArchive _target;
  usize _num_continuous{};
  Vec<CType> _continuous_lower_bounds{};
  Vec<CType> _continuous_upper_bounds{};
  Vec<CType> _continuous_init_lower_bounds{};
  Vec<CType> _continuous_init_upper_bounds{};

  // CPU fields
  std::string _gradient_mode{};
  CType _gradient_epsilon{};
  std::optional<bool> _always_inherit_continuous{};
  std::optional<usize> _batch_size{};
  mutable Arr2D<ScalarType> _eval_buffer{};

#ifdef GOBLIN_HAS_CUDA
  // GPU fields
  std::optional<KernelVersion> _kernel_version{};
  int _num_sms = 0;
  size_t _num_datapoints = 0;
  size_t _num_solutions_allocated = 0;
  size_t _num_partials_allocated = 0;
  size_t _num_results_allocated = 0;
  float* d_X = nullptr;
  float* d_Y = nullptr;
  uint8_t* d_type = nullptr;
  float* d_value = nullptr;
  float* d_partial = nullptr;
  float* d_result = nullptr;
#endif  // GOBLIN_HAS_CUDA
};

};  // namespace goblin

#endif /* _GOBLIN_GP_SR_H */



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                       goblin.h continued                                                                     //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef GOBLIN_HAS_CUDA
#endif

inline bool has_gpu_support() {
#ifdef GOBLIN_HAS_CUDA
  return true;
#else
  return false;
#endif
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                       goblin/bench/functions.h included by goblin.h                                          //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef _GOBLIN_BENCH_FUNCTIONS_H
#define _GOBLIN_BENCH_FUNCTIONS_H


namespace goblin {

class ObjectiveBase {
 public:
  virtual usize num_discrete() const = 0;
  virtual usize num_continuous() const = 0;

  virtual std::tuple<CType, CType> evaluate(RefS<Vec<DType>> discrete_values,
                                            RefS<Vec<CType>> continuous_values,
                                            RefS<Array<BType>> discrete_active,
                                            RefS<Array<BType>> continuous_active) = 0;

  virtual std::tuple<CType, CType> evaluate_partial(RefS<Vec<DType>> discrete_values,
                                                    RefS<Vec<CType>> continuous_values,
                                                    RefS<Array<BType>> discrete_active,
                                                    RefS<Array<BType>> continuous_active,
                                                    CRefS<Vec<DType>> parent_discrete_values,
                                                    CRefS<Vec<CType>> parent_continuous_values,
                                                    CRefS<Array<BType>> parent_discrete_active,
                                                    CRefS<Array<BType>> parent_continuous_active,
                                                    const CType parent_objective_value,
                                                    const CType parent_constraint_value,
                                                    const std::span<const usize>& discrete_indices,
                                                    const std::span<const usize>& continuous_indices) {
    return evaluate(discrete_values, continuous_values, discrete_active, continuous_active);
  };

  virtual ~ObjectiveBase() {};
};

};  // namespace goblin

#endif /* _GOBLIN_BENCH_FUNCTIONS_H */

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                       goblin/bench/functions/combinators.h included by goblin.h                              //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef _GOBLIN_BENCH_FUNCTIONS_COMBINATORS_H
#define _GOBLIN_BENCH_FUNCTIONS_COMBINATORS_H


#include <numbers>


// TODO
// - [ ] Permute (scramble arguments to function, either fixed or random perm)
// - [ ] Translate (translate the continuous function inputs, fixed
// transformation or random)
// - [x] Rotate (rotate continuous function inputs, fixed angle or random per
// variable pair)
// - [x] Sum (add multiple functions)
// - [x] Concat (add smaller functions to make a larger function)
//   - [ ] Allow overlap
// - [x] Repeat (repeat the same function multiple times)
//   - [ ] Allow overlap
// - [x] Min (return min of several functions)
// - [x] Max (return max of several functions)
// - [x] BBO (no partial evals)
// - [x] Masked (no active information)
// - [?] Conditional - for doing something with introns?
//       For this to be useful, ideally Objectivebase
//       overloads a lot of operators to allow constructing
//       bools from `ObjectiveBase`s (e.g. Rastrigin(3) < Sphere(3))
// - [ ] Testing

namespace goblin {

class BBO final : public ObjectiveBase {
 public:
  BBO(std::shared_ptr<ObjectiveBase> objective) : fn(objective) {};

  usize num_discrete() const override final { return fn->num_discrete(); };
  usize num_continuous() const override final { return fn->num_continuous(); };

  std::tuple<CType, CType> evaluate(RefS<Vec<DType>> discrete_values,
                                    RefS<Vec<CType>> continuous_values,
                                    RefS<Array<BType>> discrete_active,
                                    RefS<Array<BType>> continuous_active) override final {
    return fn->evaluate(discrete_values, continuous_values, discrete_active, continuous_active);
  };

 private:
  std::shared_ptr<ObjectiveBase> fn;
};

class Masked final : public ObjectiveBase {
 public:
  Masked(std::shared_ptr<ObjectiveBase> objective) : fn(objective) {};

  usize num_discrete() const override final { return fn->num_discrete(); };
  usize num_continuous() const override final { return fn->num_continuous(); };

  std::tuple<CType, CType> evaluate(RefS<Vec<DType>> discrete_values,
                                    RefS<Vec<CType>> continuous_values,
                                    RefS<Array<BType>> discrete_active,
                                    RefS<Array<BType>> continuous_active) override final {
    discrete_active.fill(true);
    continuous_active.fill(true);
    return fn->evaluate(discrete_values, continuous_values, discrete_active, continuous_active);
  };

  std::tuple<CType, CType> evaluate_partial(RefS<Vec<DType>> discrete_values,
                                            RefS<Vec<CType>> continuous_values,
                                            RefS<Array<BType>> discrete_active,
                                            RefS<Array<BType>> continuous_active,
                                            CRefS<Vec<DType>> parent_discrete_values,
                                            CRefS<Vec<CType>> parent_continuous_values,
                                            CRefS<Array<BType>> parent_discrete_active,
                                            CRefS<Array<BType>> parent_continuous_active,
                                            const CType parent_objective_value,
                                            const CType parent_constraint_value,
                                            const std::span<const usize>& discrete_indices,
                                            const std::span<const usize>& continuous_indices) override final {
    discrete_active.fill(true);
    continuous_active.fill(true);
    return fn->evaluate_partial(discrete_values, continuous_values, discrete_active, continuous_active,
                                parent_discrete_values, parent_continuous_values, parent_discrete_active,
                                parent_continuous_active, parent_objective_value, parent_constraint_value,
                                discrete_indices, continuous_indices);
  };

 private:
  std::shared_ptr<ObjectiveBase> fn;
};

class Inverted final : public ObjectiveBase {
 public:
  Inverted(std::shared_ptr<ObjectiveBase> objective) : fn(objective) {};

  usize num_discrete() const override final { return fn->num_discrete(); };
  usize num_continuous() const override final { return fn->num_continuous(); };

  std::tuple<CType, CType> evaluate(RefS<Vec<DType>> discrete_values,
                                    RefS<Vec<CType>> continuous_values,
                                    RefS<Array<BType>> discrete_active,
                                    RefS<Array<BType>> continuous_active) override final {
    // TODO this does not forward modifications to the actual solutions
    // (non-issue for my use cases so far, but still a violation of the api...)
    Vec<DType> d_inverted = DType(1) - discrete_values.array();
    Vec<CType> c_inverted = CType(0.0) - continuous_values.array();
    return fn->evaluate(d_inverted, c_inverted, discrete_active, continuous_active);
  };

  std::tuple<CType, CType> evaluate_partial(RefS<Vec<DType>> discrete_values,
                                            RefS<Vec<CType>> continuous_values,
                                            RefS<Array<BType>> discrete_active,
                                            RefS<Array<BType>> continuous_active,
                                            CRefS<Vec<DType>> parent_discrete_values,
                                            CRefS<Vec<CType>> parent_continuous_values,
                                            CRefS<Array<BType>> parent_discrete_active,
                                            CRefS<Array<BType>> parent_continuous_active,
                                            const CType parent_objective_value,
                                            const CType parent_constraint_value,
                                            const std::span<const usize>& discrete_indices,
                                            const std::span<const usize>& continuous_indices) override final {
    // TODO this does not forward modifications to the actual solutions
    // (non-issue for my use cases so far, but still a violation of the api...)
    Vec<DType> d_inverted = DType(1) - discrete_values.array();
    Vec<CType> c_inverted = CType(0.0) - continuous_values.array();
    Vec<DType> dp_inverted = DType(1) - parent_discrete_values.array();
    Vec<CType> cp_inverted = CType(0.0) - parent_continuous_values.array();
    return fn->evaluate_partial(d_inverted, c_inverted, discrete_active, continuous_active, dp_inverted, cp_inverted,
                                parent_discrete_active, parent_continuous_active, parent_objective_value,
                                parent_constraint_value, discrete_indices, continuous_indices);
  };

 private:
  std::shared_ptr<ObjectiveBase> fn;
};

class Rotated final : public ObjectiveBase {
 public:
  Rotated(std::shared_ptr<ObjectiveBase> objective,
          CType rotation_angle = -45.0,
          std::optional<usize> rotation_block_size = std::nullopt)
      : fn(objective) {
    __goblin_runtime_assert(rotation_angle != 0.0);

    usize block_size = rotation_block_size.value_or(fn->num_continuous());

    Mat<CType> tmp = Mat<CType>::Identity(block_size, block_size);

    block_rotation_matrix = Mat<CType>::Identity(block_size, block_size);

    CType theta = (rotation_angle / CType(180.0)) * std::numbers::pi_v<CType>;
    CType cos_theta = std::cos(theta);
    CType sin_theta = std::sin(theta);
    for (usize i = 0; i < block_size - 1; i++) {
      for (usize j = i + 1; j < block_size; j++) {
        tmp(i, i) = cos_theta;
        tmp(i, j) = -sin_theta;
        tmp(j, i) = sin_theta;
        tmp(j, j) = cos_theta;

        block_rotation_matrix *= tmp;

        // tmp.setIdentity();
        tmp(i, i) = CType(1.0);
        tmp(i, j) = CType(0.0);
        tmp(j, i) = CType(0.0);
        tmp(j, j) = CType(1.0);
      }
    }
  };

  Rotated(std::shared_ptr<ObjectiveBase> objective, usize seed, std::optional<usize> rotation_block_size = std::nullopt)
      : fn(objective) {
    Rng rng = seeded_rng(seed);

    usize block_size = rotation_block_size.value_or(fn->num_continuous());

    Mat<CType> tmp = Mat<CType>::Identity(block_size, block_size);
    block_rotation_matrix = Mat<CType>::Identity(block_size, block_size);

    std::uniform_real_distribution<CType> angle(0.0, 360.0);

    for (usize i = 0; i < block_size - 1; i++) {
      for (usize j = i + 1; j < block_size; j++) {
        CType theta = (angle(rng) / CType(180.0)) * std::numbers::pi_v<CType>;
        CType cos_theta = std::cos(theta);
        CType sin_theta = std::sin(theta);
        tmp(i, i) = cos_theta;
        tmp(i, j) = -sin_theta;
        tmp(j, i) = sin_theta;
        tmp(j, j) = cos_theta;

        block_rotation_matrix *= tmp;

        // tmp.setIdentity();
        tmp(i, i) = CType(1.0);
        tmp(i, j) = CType(0.0);
        tmp(j, i) = CType(0.0);
        tmp(j, j) = CType(1.0);
      }
    }
  };

  usize num_discrete() const override final { return fn->num_discrete(); };
  usize num_continuous() const override final { return fn->num_continuous(); };

  std::tuple<CType, CType> evaluate(RefS<Vec<DType>> discrete_values,
                                    RefS<Vec<CType>> continuous_values,
                                    RefS<Array<BType>> discrete_active,
                                    RefS<Array<BType>> continuous_active) override final {
    discrete_active.fill(true);
    continuous_active.fill(true);
    auto r_values = rotated(continuous_values);
    return fn->evaluate(discrete_values, r_values, discrete_active, continuous_active);
  };

  std::tuple<CType, CType> evaluate_partial(RefS<Vec<DType>> discrete_values,
                                            RefS<Vec<CType>> continuous_values,
                                            RefS<Array<BType>> discrete_active,
                                            RefS<Array<BType>> continuous_active,
                                            CRefS<Vec<DType>> parent_discrete_values,
                                            CRefS<Vec<CType>> parent_continuous_values,
                                            CRefS<Array<BType>> parent_discrete_active,
                                            CRefS<Array<BType>> parent_continuous_active,
                                            const CType parent_objective_value,
                                            const CType parent_constraint_value,
                                            const std::span<const usize>& discrete_indices,
                                            const std::span<const usize>& continuous_indices) override final {
    discrete_active.fill(true);
    continuous_active.fill(true);
    auto r_values = rotated(continuous_values);
    auto r_parent_values = rotated(parent_continuous_values);
    return fn->evaluate_partial(discrete_values, r_values, discrete_active, continuous_active, parent_discrete_values,
                                r_parent_values, parent_discrete_active, parent_continuous_active,
                                parent_objective_value, parent_constraint_value, discrete_indices, continuous_indices);
  };

 private:
  template <typename V>
  Vec<CType> rotated(V v) {
    const isize block_size = block_rotation_matrix.rows();
    Vec<CType> r(v.size());
    Vec<CType> tmp = Vec<CType>::Zero(block_size);
    for (isize i = 0; i < v.size(); i += block_size) {
      for (isize j = 0; j < block_size; j++) {
        tmp(j) = i + j < v.size() ? v(i + j) : 0.0;
      }
      r(Eigen::seqN(i, block_size)) = block_rotation_matrix * tmp;
    }
    return r;
  };

  std::shared_ptr<ObjectiveBase> fn{};
  Mat<CType> block_rotation_matrix{};
};

class Sum final : public ObjectiveBase {
 public:
  Sum(std::vector<std::shared_ptr<ObjectiveBase>> objectives) : fns(std::move(objectives)) {
    _num_discrete = 0;
    _num_continuous = 0;
    for (auto& o : fns) {
      _num_discrete = std::max(_num_discrete, o->num_discrete());
      _num_continuous = std::max(_num_continuous, o->num_continuous());
    }
  };

  usize num_discrete() const override final { return _num_discrete; };
  usize num_continuous() const override final { return _num_continuous; };

  std::tuple<CType, CType> evaluate(RefS<Vec<DType>> discrete_values,
                                    RefS<Vec<CType>> continuous_values,
                                    RefS<Array<BType>> discrete_active,
                                    RefS<Array<BType>> continuous_active) override final {
    CType ov = CType(0.0), cv = CType(0.0);
    for (auto& o : fns) {
      auto [fov, fcv] = o->evaluate(discrete_values, continuous_values, discrete_active, continuous_active);
      ov += fov;
      cv += std::max(CType(0.0), cv);
    }
    return std::make_tuple(ov, cv);
  };

  std::tuple<CType, CType> evaluate_partial(RefS<Vec<DType>> discrete_values,
                                            RefS<Vec<CType>> continuous_values,
                                            RefS<Array<BType>> discrete_active,
                                            RefS<Array<BType>> continuous_active,
                                            CRefS<Vec<DType>> parent_discrete_values,
                                            CRefS<Vec<CType>> parent_continuous_values,
                                            CRefS<Array<BType>> parent_discrete_active,
                                            CRefS<Array<BType>> parent_continuous_active,
                                            const CType parent_objective_value,
                                            const CType parent_constraint_value,
                                            const std::span<const usize>& discrete_indices,
                                            const std::span<const usize>& continuous_indices) override final {
    CType ov = CType(0.0), cv = CType(0.0);
    for (auto& o : fns) {
      auto [fov, fcv] = o->evaluate_partial(discrete_values, continuous_values, discrete_active, continuous_active,
                                            parent_discrete_values, parent_continuous_values, parent_discrete_active,
                                            parent_continuous_active, parent_objective_value, parent_constraint_value,
                                            discrete_indices, continuous_indices);
      ov += fov;
      cv += std::max(CType(0.0), cv);
    }
    return std::make_tuple(ov, cv);
  };

 private:
  std::vector<std::shared_ptr<ObjectiveBase>> fns;
  usize _num_discrete = 0;
  usize _num_continuous = 0;
};

class Max final : public ObjectiveBase {
 public:
  Max(std::vector<std::shared_ptr<ObjectiveBase>> objectives) : fns(std::move(objectives)) {
    _num_discrete = 0;
    _num_continuous = 0;
    for (auto& o : fns) {
      _num_discrete = std::max(_num_discrete, o->num_discrete());
      _num_continuous = std::max(_num_continuous, o->num_continuous());
    }
  };

  usize num_discrete() const override final { return _num_discrete; };
  usize num_continuous() const override final { return _num_continuous; };

  std::tuple<CType, CType> evaluate(RefS<Vec<DType>> discrete_values,
                                    RefS<Vec<CType>> continuous_values,
                                    RefS<Array<BType>> discrete_active,
                                    RefS<Array<BType>> continuous_active) override final {
    CType ov = -std::numeric_limits<CType>().infinity(), cv = CType(0.0);
    for (auto& o : fns) {
      auto [fov, fcv] = o->evaluate(discrete_values, continuous_values, discrete_active, continuous_active);
      ov = std::max(ov, fov);
      cv += std::max(CType(0.0), cv);
    }
    return std::make_tuple(ov, cv);
  };

  std::tuple<CType, CType> evaluate_partial(RefS<Vec<DType>> discrete_values,
                                            RefS<Vec<CType>> continuous_values,
                                            RefS<Array<BType>> discrete_active,
                                            RefS<Array<BType>> continuous_active,
                                            CRefS<Vec<DType>> parent_discrete_values,
                                            CRefS<Vec<CType>> parent_continuous_values,
                                            CRefS<Array<BType>> parent_discrete_active,
                                            CRefS<Array<BType>> parent_continuous_active,
                                            const CType parent_objective_value,
                                            const CType parent_constraint_value,
                                            const std::span<const usize>& discrete_indices,
                                            const std::span<const usize>& continuous_indices) override final {
    CType ov = -std::numeric_limits<CType>().infinity(), cv = CType(0.0);
    for (auto& o : fns) {
      auto [fov, fcv] = o->evaluate_partial(discrete_values, continuous_values, discrete_active, continuous_active,
                                            parent_discrete_values, parent_continuous_values, parent_discrete_active,
                                            parent_continuous_active, parent_objective_value, parent_constraint_value,
                                            discrete_indices, continuous_indices);
      ov = std::max(ov, fov);
      cv += std::max(CType(0.0), cv);
    }
    return std::make_tuple(ov, cv);
  };

 private:
  std::vector<std::shared_ptr<ObjectiveBase>> fns;
  usize _num_discrete = 0;
  usize _num_continuous = 0;
};

class Min final : public ObjectiveBase {
 public:
  Min(std::vector<std::shared_ptr<ObjectiveBase>> objectives) : fns(std::move(objectives)) {
    _num_discrete = 0;
    _num_continuous = 0;
    for (auto& o : fns) {
      _num_discrete = std::max(_num_discrete, o->num_discrete());
      _num_continuous = std::max(_num_continuous, o->num_continuous());
    }
  };

  usize num_discrete() const override final { return _num_discrete; };
  usize num_continuous() const override final { return _num_continuous; };

  std::tuple<CType, CType> evaluate(RefS<Vec<DType>> discrete_values,
                                    RefS<Vec<CType>> continuous_values,
                                    RefS<Array<BType>> discrete_active,
                                    RefS<Array<BType>> continuous_active) override final {
    CType ov = std::numeric_limits<CType>().infinity(), cv = CType(0.0);
    for (auto& o : fns) {
      auto [fov, fcv] = o->evaluate(discrete_values, continuous_values, discrete_active, continuous_active);
      ov = std::min(ov, fov);
      cv += std::max(CType(0.0), cv);
    }
    return std::make_tuple(ov, cv);
  };

  std::tuple<CType, CType> evaluate_partial(RefS<Vec<DType>> discrete_values,
                                            RefS<Vec<CType>> continuous_values,
                                            RefS<Array<BType>> discrete_active,
                                            RefS<Array<BType>> continuous_active,
                                            CRefS<Vec<DType>> parent_discrete_values,
                                            CRefS<Vec<CType>> parent_continuous_values,
                                            CRefS<Array<BType>> parent_discrete_active,
                                            CRefS<Array<BType>> parent_continuous_active,
                                            const CType parent_objective_value,
                                            const CType parent_constraint_value,
                                            const std::span<const usize>& discrete_indices,
                                            const std::span<const usize>& continuous_indices) override final {
    CType ov = std::numeric_limits<CType>().infinity(), cv = CType(0.0);
    for (auto& o : fns) {
      auto [fov, fcv] = o->evaluate_partial(discrete_values, continuous_values, discrete_active, continuous_active,
                                            parent_discrete_values, parent_continuous_values, parent_discrete_active,
                                            parent_continuous_active, parent_objective_value, parent_constraint_value,
                                            discrete_indices, continuous_indices);
      ov = std::min(ov, fov);
      cv += std::max(CType(0.0), cv);
    }
    return std::make_tuple(ov, cv);
  };

 private:
  std::vector<std::shared_ptr<ObjectiveBase>> fns;
  usize _num_discrete = 0;
  usize _num_continuous = 0;
};

class Concat final : public ObjectiveBase {
 public:
  Concat(std::vector<std::shared_ptr<ObjectiveBase>> objectives) : fns(std::move(objectives)) {
    _num_discrete = 0;
    _num_continuous = 0;
    for (auto& o : fns) {
      _num_discrete += o->num_discrete();
      _num_continuous += o->num_continuous();
    }
  };

  usize num_discrete() const override final { return _num_discrete; };
  usize num_continuous() const override final { return _num_continuous; };

  std::tuple<CType, CType> evaluate(RefS<Vec<DType>> discrete_values,
                                    RefS<Vec<CType>> continuous_values,
                                    RefS<Array<BType>> discrete_active,
                                    RefS<Array<BType>> continuous_active) override final {
    CType ov = CType(0.0), cv = CType(0.0);
    usize d_offset = 0, c_offset = 0, d_len, c_len;
    for (auto& o : fns) {
      d_len = o->num_discrete();
      c_len = o->num_continuous();
      auto d_seq = Eigen::seqN(d_offset, d_len);
      auto c_seq = Eigen::seqN(c_offset, c_len);
      auto [fov, fcv] = o->evaluate(discrete_values(d_seq), continuous_values(c_seq), discrete_active(d_seq),
                                    continuous_active(c_seq));
      ov += fov;
      cv += std::max(CType(0.0), cv);
      d_offset += d_len;
      c_offset += c_len;
    }
    return std::make_tuple(ov, cv);
  };

  std::tuple<CType, CType> evaluate_partial(RefS<Vec<DType>> discrete_values,
                                            RefS<Vec<CType>> continuous_values,
                                            RefS<Array<BType>> discrete_active,
                                            RefS<Array<BType>> continuous_active,
                                            CRefS<Vec<DType>> parent_discrete_values,
                                            CRefS<Vec<CType>> parent_continuous_values,
                                            CRefS<Array<BType>> parent_discrete_active,
                                            CRefS<Array<BType>> parent_continuous_active,
                                            const CType parent_objective_value,
                                            const CType parent_constraint_value,
                                            const std::span<const usize>& discrete_indices,
                                            const std::span<const usize>& continuous_indices) override final {
    std::vector<usize> d_indices, c_indices;
    CType ov = CType(0.0), cv = CType(0.0);
    usize d_offset = 0, c_offset = 0, d_len, c_len;
    for (auto& o : fns) {
      d_len = o->num_discrete();
      c_len = o->num_continuous();
      auto d_seq = Eigen::seqN(d_offset, d_len);
      auto c_seq = Eigen::seqN(c_offset, c_len);

      // discrete/continuous indices are the indices falling into d_seq/c_seq without the offset
      d_indices.clear();
      d_indices.reserve(d_len);
      for (usize i : discrete_indices) {
        if (d_offset <= i && i < d_offset + d_len) {
          d_indices.push_back(i - d_offset);
        }
      }

      c_indices.clear();
      c_indices.reserve(c_len);
      for (usize i : continuous_indices) {
        if (c_offset <= i && i < c_offset + c_len) {
          c_indices.push_back(i - c_offset);
        }
      }

      auto [fov, fcv] = o->evaluate_partial(
          discrete_values(d_seq), continuous_values(c_seq), discrete_active(d_seq), continuous_active(c_seq),
          parent_discrete_values(d_seq), parent_continuous_values(c_seq), parent_discrete_active(d_seq),
          parent_continuous_active(c_seq), parent_objective_value, parent_constraint_value, d_indices, c_indices);
      ov += fov;
      cv += std::max(CType(0.0), cv);
      d_offset += d_len;
      c_offset += c_len;
    }
    return std::make_tuple(ov, cv);
  };

 private:
  std::vector<std::shared_ptr<ObjectiveBase>> fns;
  usize _num_discrete = 0;
  usize _num_continuous = 0;
};

class Repeat final : public ObjectiveBase {
 public:
  Repeat(std::shared_ptr<ObjectiveBase> objective, usize repeats) : fn(objective), _repeats(repeats) {
    __goblin_runtime_assert(repeats > 0);
    _num_discrete = fn->num_discrete() * _repeats;
    _num_continuous = fn->num_continuous() * _repeats;
  };

  usize num_discrete() const override final { return _num_discrete; };
  usize num_continuous() const override final { return _num_continuous; };

  std::tuple<CType, CType> evaluate(RefS<Vec<DType>> discrete_values,
                                    RefS<Vec<CType>> continuous_values,
                                    RefS<Array<BType>> discrete_active,
                                    RefS<Array<BType>> continuous_active) override final {
    CType ov = CType(0.0), cv = CType(0.0);
    usize d_offset = 0, c_offset = 0, d_len, c_len;
    for (usize i = 0; i < _repeats; i++) {
      d_len = fn->num_discrete();
      c_len = fn->num_continuous();
      auto d_seq = Eigen::seqN(d_offset, d_len);
      auto c_seq = Eigen::seqN(c_offset, c_len);
      auto [fov, fcv] = fn->evaluate(discrete_values(d_seq), continuous_values(c_seq), discrete_active(d_seq),
                                     continuous_active(c_seq));
      ov += fov;
      cv += std::max(CType(0.0), cv);
      d_offset += d_len;
      c_offset += c_len;
    }
    return std::make_tuple(ov, cv);
  };

  std::tuple<CType, CType> evaluate_partial(RefS<Vec<DType>> discrete_values,
                                            RefS<Vec<CType>> continuous_values,
                                            RefS<Array<BType>> discrete_active,
                                            RefS<Array<BType>> continuous_active,
                                            CRefS<Vec<DType>> parent_discrete_values,
                                            CRefS<Vec<CType>> parent_continuous_values,
                                            CRefS<Array<BType>> parent_discrete_active,
                                            CRefS<Array<BType>> parent_continuous_active,
                                            const CType parent_objective_value,
                                            const CType parent_constraint_value,
                                            const std::span<const usize>& discrete_indices,
                                            const std::span<const usize>& continuous_indices) override final {
    std::vector<usize> d_indices, c_indices;
    CType ov = CType(0.0), cv = CType(0.0);
    usize d_offset = 0, c_offset = 0, d_len, c_len;
    for (usize i = 0; i < _repeats; i++) {
      d_len = fn->num_discrete();
      c_len = fn->num_continuous();
      auto d_seq = Eigen::seqN(d_offset, d_len);
      auto c_seq = Eigen::seqN(c_offset, c_len);

      // discrete/continuous indices are the indices falling into d_seq/c_seq without the offset
      d_indices.clear();
      d_indices.reserve(d_len);
      for (usize i : discrete_indices) {
        if (d_offset <= i && i < d_offset + d_len) {
          d_indices.push_back(i - d_offset);
        }
      }

      c_indices.clear();
      c_indices.reserve(c_len);
      for (usize i : continuous_indices) {
        if (c_offset <= i && i < c_offset + c_len) {
          c_indices.push_back(i - c_offset);
        }
      }

      auto [fov, fcv] = fn->evaluate_partial(
          discrete_values(d_seq), continuous_values(c_seq), discrete_active(d_seq), continuous_active(c_seq),
          parent_discrete_values(d_seq), parent_continuous_values(c_seq), parent_discrete_active(d_seq),
          parent_continuous_active(c_seq), parent_objective_value, parent_constraint_value, d_indices, c_indices);
      ov += fov;
      cv += std::max(CType(0.0), cv);
      d_offset += d_len;
      c_offset += c_len;
    }
    return std::make_tuple(ov, cv);
  };

 private:
  std::shared_ptr<ObjectiveBase> fn;
  usize _repeats;
  usize _num_discrete = 0;
  usize _num_continuous = 0;
};

};  // namespace goblin

#endif /* _GOBLIN_BENCH_FUNCTIONS_COMBINATORS_H */

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                       goblin/bench/functions/discrete.h included by goblin.h                                 //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef _GOBLIN_BENCH_FUNCTIONS_DISCRETE_H
#define _GOBLIN_BENCH_FUNCTIONS_DISCRETE_H


namespace goblin {

class OneMax final : public ObjectiveBase {
 public:
  OneMax(usize ndims) : dims(ndims) {};

  usize num_discrete() const override final { return dims; };
  usize num_continuous() const override final { return 0; };

  std::tuple<CType, CType> evaluate(RefS<Vec<DType>> discrete_values,
                                    RefS<Vec<CType>> continuous_values,
                                    RefS<Array<BType>> discrete_active,
                                    RefS<Array<BType>> continuous_active) override final {
    discrete_active.fill(true);
    return std::make_tuple(discrete_values.array().cast<double>().sum(), 0.0);
  };

  std::tuple<CType, CType> evaluate_partial(RefS<Vec<DType>> discrete_values,
                                            RefS<Vec<CType>> continuous_values,
                                            RefS<Array<BType>> discrete_active,
                                            RefS<Array<BType>> continuous_active,
                                            CRefS<Vec<DType>> parent_discrete_values,
                                            CRefS<Vec<CType>> parent_continuous_values,
                                            CRefS<Array<BType>> parent_discrete_active,
                                            CRefS<Array<BType>> parent_continuous_active,
                                            const CType parent_objective_value,
                                            const CType parent_constraint_value,
                                            const std::span<const usize>& discrete_indices,
                                            const std::span<const usize>& continuous_indices) override final {
    discrete_active.fill(true);
    return std::make_tuple(parent_objective_value + discrete_values(discrete_indices).array().cast<double>().sum() -
                               parent_discrete_values(discrete_indices).array().cast<double>().sum(),
                           0.0);
  };

 private:
  usize dims;
};

class ZeroMax final : public ObjectiveBase {
 public:
  ZeroMax(usize ndims) : dims(ndims) {};

  usize num_discrete() const override final { return dims; };
  usize num_continuous() const override final { return 0; };

  std::tuple<CType, CType> evaluate(RefS<Vec<DType>> discrete_values,
                                    RefS<Vec<CType>> continuous_values,
                                    RefS<Array<BType>> discrete_active,
                                    RefS<Array<BType>> continuous_active) override final {
    discrete_active.fill(true);
    return std::make_tuple(discrete_values.size() - discrete_values.array().cast<double>().sum(), 0.0);
  };

  std::tuple<CType, CType> evaluate_partial(RefS<Vec<DType>> discrete_values,
                                            RefS<Vec<CType>> continuous_values,
                                            RefS<Array<BType>> discrete_active,
                                            RefS<Array<BType>> continuous_active,
                                            CRefS<Vec<DType>> parent_discrete_values,
                                            CRefS<Vec<CType>> parent_continuous_values,
                                            CRefS<Array<BType>> parent_discrete_active,
                                            CRefS<Array<BType>> parent_continuous_active,
                                            const CType parent_objective_value,
                                            const CType parent_constraint_value,
                                            const std::span<const usize>& discrete_indices,
                                            const std::span<const usize>& continuous_indices) override final {
    discrete_active.fill(true);
    return std::make_tuple(parent_objective_value - discrete_values(discrete_indices).array().cast<double>().sum() +
                               parent_discrete_values(discrete_indices).array().cast<double>().sum(),
                           0.0);
  };

 private:
  usize dims;
};

class LeadingOnes final : public ObjectiveBase {
 public:
  LeadingOnes(usize ndims) : dims(ndims) {};

  usize num_discrete() const override final { return dims; };
  usize num_continuous() const override final { return 0; };

  std::tuple<CType, CType> evaluate(RefS<Vec<DType>> discrete_values,
                                    RefS<Vec<CType>> continuous_values,
                                    RefS<Array<BType>> discrete_active,
                                    RefS<Array<BType>> continuous_active) override final {
    CType ov = CType(0.0);
    for (usize i = 0; i < dims; i++) {
      discrete_active(i) = true;
      if (discrete_values(i) != DType(1)) {
        break;
      }
      ov += CType(1.0);
    }
    return std::make_tuple(ov, 0.0);
  };

 private:
  usize dims;
};

class TrailingZeros final : public ObjectiveBase {
 public:
  TrailingZeros(usize ndims) : dims(ndims) {};

  usize num_discrete() const override final { return dims; };
  usize num_continuous() const override final { return 0; };

  std::tuple<CType, CType> evaluate(RefS<Vec<DType>> discrete_values,
                                    RefS<Vec<CType>> continuous_values,
                                    RefS<Array<BType>> discrete_active,
                                    RefS<Array<BType>> continuous_active) override final {
    CType ov = CType(0.0);
    for (usize i = dims; i > 0;) {
      i--;
      discrete_active(i) = true;
      if (discrete_values(i) != DType(0)) {
        break;
      }
      ov += CType(1.0);
    }
    return std::make_tuple(ov, 0.0);
  };

 private:
  usize dims;
};

class HLeadingOnes final : public ObjectiveBase {
 public:
  HLeadingOnes(usize ndims, usize branching_factor = 2) : dims(ndims), branching_factor(branching_factor) {
    if (dims == 0) {
      throw std::runtime_error("At least one variable is required.");
    }
  };

  usize num_discrete() const override final { return dims; };
  usize num_continuous() const override final { return 0; };

  std::tuple<CType, CType> evaluate(RefS<Vec<DType>> discrete_values,
                                    RefS<Vec<CType>> continuous_values,
                                    RefS<Array<BType>> discrete_active,
                                    RefS<Array<BType>> continuous_active) override final {
    CType ov = CType(0.0);
    eval_helper(discrete_values, discrete_active, 0, ov);
    return std::make_tuple(ov, 0.0);
  };

 private:
  void eval_helper(RefS<Vec<DType>> discrete_values, RefS<Array<BType>> discrete_active, usize i, double& ov) {
    discrete_active(i) = true;
    if (discrete_values(i) > 0) {
      ov += 1.0;

      usize num_active_children = std::min(static_cast<usize>(discrete_values(i)), branching_factor);
      for (usize j = 0, c; j < num_active_children; j++) {
        c = branching_factor * i + j + 1;  // index of j-th child of i
        if (c < dims) {
          eval_helper(discrete_values, discrete_active, c, ov);
        }
      }
    }
  };

  usize dims;
  usize branching_factor;
};

class DeceptiveTrap final : public ObjectiveBase {
 public:
  DeceptiveTrap(usize ndims) : DeceptiveTrap(ndims, ndims) {};
  DeceptiveTrap(usize ndims, usize block_size) : dims(ndims), block_size(block_size) {};

  usize num_discrete() const override final { return dims; };
  usize num_continuous() const override final { return 0; };

  std::tuple<CType, CType> evaluate(RefS<Vec<DType>> discrete_values,
                                    RefS<Vec<CType>> continuous_values,
                                    RefS<Array<BType>> discrete_active,
                                    RefS<Array<BType>> continuous_active) override final {
    discrete_active.fill(true);

    CType ov = CType(0.0);
    for (usize i = 0; i < dims; i += block_size) {
      int unitation = discrete_values(Eigen::seqN(i, std::min(block_size, dims - i))).cast<int>().sum();
      ov += unitation == static_cast<isize>(block_size) ? block_size : static_cast<isize>(block_size) - unitation - 1;
    }
    return std::make_tuple(ov, 0.0);
  };

 private:
  usize dims;
  usize block_size;
};

class BimodalTrap final : public ObjectiveBase {
 public:
  BimodalTrap(usize ndims) : BimodalTrap(ndims, ndims) {};
  BimodalTrap(usize ndims, usize block_size) : dims(ndims), block_size(block_size) {};

  usize num_discrete() const override final { return dims; };
  usize num_continuous() const override final { return 0; };

  std::tuple<CType, CType> evaluate(RefS<Vec<DType>> discrete_values,
                                    RefS<Vec<CType>> continuous_values,
                                    RefS<Array<BType>> discrete_active,
                                    RefS<Array<BType>> continuous_active) override final {
    discrete_active.fill(true);

    CType ov = CType(0.0);
    for (usize i = 0; i < dims; i += block_size) {
      isize unitation = discrete_values(Eigen::seqN(i, std::min(block_size, dims - i))).cast<isize>().sum();
      ov += unitation == 0 || unitation == static_cast<isize>(block_size)
                ? static_cast<isize>(block_size)
                : std::abs(static_cast<isize>(2 * unitation - block_size - 2));
    }
    return std::make_tuple(ov, 0.0);
  };

 private:
  usize dims;
  usize block_size;
};

};  // namespace goblin

#endif /* _GOBLIN_BENCH_FUNCTIONS_DISCRETE_H */

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                       goblin/bench/functions/continuous.h included by goblin.h                               //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef _GOBLIN_BENCH_FUNCTIONS_CONTINUOUS_H
#define _GOBLIN_BENCH_FUNCTIONS_CONTINUOUS_H



namespace goblin {

class Sphere final : public ObjectiveBase {
 public:
  Sphere(usize ndims) : dims(ndims) {};

  usize num_discrete() const override final { return 0; };
  usize num_continuous() const override final { return dims; };

  std::tuple<CType, CType> evaluate(RefS<Vec<DType>> discrete_values,
                                    RefS<Vec<CType>> continuous_values,
                                    RefS<Array<BType>> discrete_active,
                                    RefS<Array<BType>> continuous_active) override final {
    continuous_active.fill(true);
    return std::make_tuple(continuous_values.array().pow(2).sum(), 0.0);
  };

  std::tuple<CType, CType> evaluate_partial(RefS<Vec<DType>> discrete_values,
                                            RefS<Vec<CType>> continuous_values,
                                            RefS<Array<BType>> discrete_active,
                                            RefS<Array<BType>> continuous_active,
                                            CRefS<Vec<DType>> parent_discrete_values,
                                            CRefS<Vec<CType>> parent_continuous_values,
                                            CRefS<Array<BType>> parent_discrete_active,
                                            CRefS<Array<BType>> parent_continuous_active,
                                            const CType parent_objective_value,
                                            const CType parent_constraint_value,
                                            const std::span<const usize>& discrete_indices,
                                            const std::span<const usize>& continuous_indices) override final {
    continuous_active.fill(true);
    return std::make_tuple(parent_objective_value + continuous_values(continuous_indices).array().pow(2).sum() -
                               parent_continuous_values(continuous_indices).array().pow(2).sum(),
                           0.0);
  };

 private:
  usize dims;
};

class Rosenbrock final : public ObjectiveBase {
 public:
  Rosenbrock(usize ndims) : dims(ndims) {};

  usize num_discrete() const override final { return 0; };
  usize num_continuous() const override final { return dims; };

  std::tuple<CType, CType> evaluate(RefS<Vec<DType>> discrete_values,
                                    RefS<Vec<CType>> continuous_values,
                                    RefS<Array<BType>> discrete_active,
                                    RefS<Array<BType>> continuous_active) override final {
    continuous_active.fill(true);
    CType ov = 0.0;
    for (usize i = 0; i < dims - 1; i++) {
      auto x = continuous_values(i);
      auto y = continuous_values(i + 1);
      ov += 100 * (y - x * x) * (y - x * x) + (1.0 - x) * (1.0 - x);
    }

    return std::make_tuple(ov, 0.0);
  };

 private:
  usize dims;
};

class Rastrigin final : public ObjectiveBase {
 public:
  Rastrigin(usize ndims) : dims(ndims) {};

  usize num_discrete() const override final { return 0; };
  usize num_continuous() const override final { return dims; };

  std::tuple<CType, CType> evaluate(RefS<Vec<DType>> discrete_values,
                                    RefS<Vec<CType>> continuous_values,
                                    RefS<Array<BType>> discrete_active,
                                    RefS<Array<BType>> continuous_active) override final {
    continuous_active.fill(true);
    CType ov = CType(10.0) * static_cast<CType>(dims)

               + continuous_values.norm() -
               CType(10.0) * (continuous_values.array() * CType(2.0 * std::numbers::pi)).cos().sum();

    return std::make_tuple(ov, 0.0);
  };

 private:
  usize dims;
};

class Griewank final : public ObjectiveBase {
 public:
  Griewank(usize ndims) : dims(ndims) {};

  usize num_discrete() const override final { return 0; };
  usize num_continuous() const override final { return dims; };

  std::tuple<CType, CType> evaluate(RefS<Vec<DType>> discrete_values,
                                    RefS<Vec<CType>> continuous_values,
                                    RefS<Array<BType>> discrete_active,
                                    RefS<Array<BType>> continuous_active) override final {
    continuous_active.fill(true);

    CType prod = 1.0;
    for (usize i = 0; i < dims; i++) {
      prod *= std::cos(continuous_values(i) / std::sqrt(static_cast<CType>(i + 1)));
    }
    CType ov = continuous_values.norm() / CType(4000.0) - prod + CType(1.0);
    return std::make_tuple(ov, 0.0);
  };

 private:
  usize dims;
};

class Ellipsoid final : public ObjectiveBase {
 public:
  Ellipsoid(usize ndims) : dims(ndims) {};

  usize num_discrete() const override final { return 0; };
  usize num_continuous() const override final { return dims; };

  std::tuple<CType, CType> evaluate(RefS<Vec<DType>> discrete_values,
                                    RefS<Vec<CType>> continuous_values,
                                    RefS<Array<BType>> discrete_active,
                                    RefS<Array<BType>> continuous_active) override final {
    continuous_active.fill(true);

    CType ov = 0.0;
    for (usize i = 0; i < dims; i++) {
      ov += std::pow(10.0, 6.0 * static_cast<CType>(i) / static_cast<CType>(dims - 1)) * continuous_values(i) *
            continuous_values(i);
    }
    return std::make_tuple(ov, 0.0);
  };

 private:
  usize dims;
};

class CirclesInASquare final : public ObjectiveBase {
 public:
  CirclesInASquare(usize ndims) : dims(ndims) { __goblin_runtime_assert(ndims > 2 && ndims % 2 == 0); };

  usize num_discrete() const override final { return 0; };
  usize num_continuous() const override final { return dims; };

  std::tuple<CType, CType> evaluate(RefS<Vec<DType>> discrete_values,
                                    RefS<Vec<CType>> continuous_values,
                                    RefS<Array<BType>> discrete_active,
                                    RefS<Array<BType>> continuous_active) override final {
    continuous_active.fill(true);
    CType min_dist = 1e308, cv = 0;
    for (usize i = 0; i < dims; i += 2) {
      auto circle_i = continuous_values(Eigen::seqN(i, 2));
      cv += (circle_i.array().max(1.0) - 1).sum() - circle_i.array().min(0.0).sum();
      for (usize j = i + 2; j < dims; j += 2) {
        auto circle_j = continuous_values(Eigen::seqN(j, 2));
        CType dist = (circle_i - circle_j).norm();
        min_dist = std::min(min_dist, dist);
      }
    }

    return std::make_tuple(-min_dist, cv);
  };

 private:
  usize dims;
};

};  // namespace goblin

#endif /* _GOBLIN_BENCH_FUNCTIONS_CONTINUOUS_H */

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                       goblin/bench/functions/mixed.h included by goblin.h                                    //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef _GOBLIN_BENCH_FUNCTIONS_MIXED_H
#define _GOBLIN_BENCH_FUNCTIONS_MIXED_H



namespace goblin {

class LeadingSpheres final : public ObjectiveBase {
 public:
  LeadingSpheres(usize ndims) : dims(ndims) {};

  usize num_discrete() const override final { return dims; };
  usize num_continuous() const override final { return dims; };

  std::tuple<CType, CType> evaluate(RefS<Vec<DType>> discrete_values,
                                    RefS<Vec<CType>> continuous_values,
                                    RefS<Array<BType>> discrete_active,
                                    RefS<Array<BType>> continuous_active) override final {
    CType ov = CType(0.0);
    CType cv = CType(0.0);
    bool active = true;
    for (usize i = 0; i < dims; i++) {
      if (active) {
        discrete_active(i) = true;
        if (discrete_values(i) != DType(1)) {
          active = false;
        }

        continuous_active(i) = true;

        // use a scaled sigmoid to map from [0, oo) to [0, 1)
        auto scaled = 2.0 / (1.0 + std::exp(-continuous_values(i))) - 1.0;
        if (isna(scaled)) {
          cv = 1.0;
        }
        ov += scaled * scaled;
      } else {
        ov += 1.0;
      }
    }
    return std::make_tuple(ov, cv);
  };

 private:
  usize dims;
};

};  // namespace goblin

#endif /* _GOBLIN_BENCH_FUNCTIONS_MIXED_H */

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                       goblin/bench/problem.h included by goblin.h                                            //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef _GOBLIN_BENCH_PROBLEM_H
#define _GOBLIN_BENCH_PROBLEM_H



namespace goblin {

class MOFunctionBase {
 public:
  virtual usize num_objectives() const = 0;
  virtual usize num_discrete() const = 0;
  virtual usize num_continuous() const = 0;

  virtual void evaluate(SolutionBase& solution) = 0;

  virtual void evaluate_partial(SolutionBase& solution, const SolutionBase& parent, const Subset& subset) {
    evaluate(solution);
  };

  virtual ~MOFunctionBase() {};
};

class PyFunctionBase : MOFunctionBase {
 public:
  virtual usize num_objectives() const override = 0;
  virtual usize num_discrete() const override = 0;
  virtual usize num_continuous() const override = 0;

  virtual std::tuple<Vec<CType>, CType> eval(SolutionBase& solution) = 0;

  void evaluate(SolutionBase& solution) override {
    solution.discrete_active().fill(true);
    solution.continuous_active().fill(true);
    auto [objectives, cv] = eval(solution);
    solution.quality_as<MOQuality>().objectives = objectives;
    solution.quality_as<MOQuality>().constraint_value = cv;
  };
  void evaluate_partial(SolutionBase& solution, const SolutionBase& parent, const Subset& subset) override {
    evaluate(solution);
  };

  virtual ~PyFunctionBase() {};
};

class Objectives final : public MOFunctionBase {
 public:
  Objectives(std::vector<std::shared_ptr<ObjectiveBase>> objectives) : objectives(std::move(objectives)) {
    _num_discrete = 0;
    _num_continuous = 0;
    for (auto& o : this->objectives) {
      _num_discrete = std::max(_num_discrete, o->num_discrete());
      _num_continuous = std::max(_num_continuous, o->num_continuous());
    }
  };

  usize num_objectives() const override final { return objectives.size(); }
  usize num_discrete() const override final { return _num_discrete; }
  usize num_continuous() const override final { return _num_continuous; }

  void evaluate(SolutionBase& solution) override final {
    solution.discrete_active().fill(false);
    solution.continuous_active().fill(false);
    solution.quality_as<MOQuality>().constraint_value = 0.0;
    for (usize i = 0; i < num_objectives(); i++) {
      auto [ov, cv] = objectives[i]->evaluate(solution.discrete_values(), solution.continuous_values(),
                                              solution.discrete_active(), solution.continuous_active());
      solution.quality_as<MOQuality>().objectives(i) = ov;
      solution.quality_as<MOQuality>().constraint_value += std::max(CType(0.0), cv);
    }
  };

  void evaluate_partial(SolutionBase& solution, const SolutionBase& parent, const Subset& subset) override final {
    solution.discrete_active().fill(false);
    solution.continuous_active().fill(false);
    const auto& pq = parent.quality_as<MOQuality>();
    solution.quality_as<MOQuality>().constraint_value = 0.0;
    for (usize i = 0; i < num_objectives(); i++) {
      auto [ov, cv] = objectives[i]->evaluate_partial(
          solution.discrete_values(), solution.continuous_values(), solution.discrete_active(),
          solution.continuous_active(), parent.discrete_values(), parent.continuous_values(), parent.discrete_active(),
          parent.continuous_active(), pq.objectives(i), pq.constraint_value, subset.discrete, subset.continuous);
      solution.quality_as<MOQuality>().objectives(i) = ov;
      solution.quality_as<MOQuality>().constraint_value += std::max(CType(0.0), cv);
    }
  };

 private:
  usize _num_discrete = 0;
  usize _num_continuous = 0;
  std::vector<std::shared_ptr<ObjectiveBase>> objectives;
};

class BenchmarkInstance final : public InstanceBase {
 public:
  BenchmarkInstance(std::variant<std::vector<std::shared_ptr<ObjectiveBase>>,
                                 std::shared_ptr<MOFunctionBase>,
                                 std::shared_ptr<ObjectiveBase>> objectives,
                    std::variant<DType, Vec<DType>> discrete_domain = DType(2),
                    std::variant<CType, Vec<CType>> continuous_lower_bound = -std::numeric_limits<CType>::infinity(),
                    std::variant<CType, Vec<CType>> continuous_upper_bound = std::numeric_limits<CType>::infinity(),
                    std::variant<CType, Vec<CType>> continuous_init_lower_bound = CType(0.0),
                    std::variant<CType, Vec<CType>> continuous_init_upper_bound = CType(1.0),
                    std::optional<AnyInit> init = std::nullopt,
                    std::optional<std::variant<Vec<CType>, std::tuple<Mat<DType>, Mat<CType>>, std::vector<CType>>>
                        target = std::nullopt,
                    std::optional<usize> target_archive_size = std::nullopt)
      : _objectives(std::holds_alternative<std::shared_ptr<MOFunctionBase>>(objectives)
                        ? std::get<std::shared_ptr<MOFunctionBase>>(objectives)
                        : (std::holds_alternative<std::shared_ptr<ObjectiveBase>>(objectives)
                               ? std::static_pointer_cast<MOFunctionBase>(
                                     std::make_shared<Objectives>(std::vector<std::shared_ptr<ObjectiveBase>>{
                                         std::get<std::shared_ptr<ObjectiveBase>>(objectives)}))
                               : std::static_pointer_cast<MOFunctionBase>(std::make_shared<Objectives>(
                                     std::move(std::get<std::vector<std::shared_ptr<ObjectiveBase>>>(objectives)))))),
        _fitness(MOFitness(_objectives->num_objectives(),
                           /* minimize */ _objectives->num_continuous() > 0)),
        _init(from_any_init(init.value_or(std::make_shared<CompleteInit>()))),
        _target(_fitness),
        _target_archive_size(target_archive_size) {
    if (std::holds_alternative<DType>(discrete_domain)) {
      _discrete_domain_sizes.resize(_objectives->num_discrete());
      _discrete_domain_sizes.fill(std::get<DType>(discrete_domain));
    } else {
      _discrete_domain_sizes = std::get<Vec<DType>>(discrete_domain);
    }
    __goblin_runtime_assert(static_cast<usize>(_discrete_domain_sizes.size()) == _objectives->num_discrete());

    if (std::holds_alternative<CType>(continuous_lower_bound)) {
      _continuous_lower_bounds.resize(_objectives->num_continuous());
      _continuous_lower_bounds.fill(std::get<CType>(continuous_lower_bound));
    } else {
      _continuous_lower_bounds = std::get<Vec<CType>>(continuous_lower_bound);
    }
    __goblin_runtime_assert(static_cast<usize>(_continuous_lower_bounds.size()) == _objectives->num_continuous());

    if (std::holds_alternative<CType>(continuous_upper_bound)) {
      _continuous_upper_bounds.resize(_objectives->num_continuous());
      _continuous_upper_bounds.fill(std::get<CType>(continuous_upper_bound));
    } else {
      _continuous_upper_bounds = std::get<Vec<CType>>(continuous_upper_bound);
    }
    __goblin_runtime_assert(static_cast<usize>(_continuous_upper_bounds.size()) == _objectives->num_continuous());

    set_initial_bounds(continuous_init_lower_bound, continuous_init_upper_bound);

    for (usize i = 0; i < _objectives->num_continuous(); i++) {
      __goblin_runtime_assert(!isna(_continuous_init_lower_bounds(i)));
      __goblin_runtime_assert(!isna(_continuous_init_upper_bounds(i)));
      __goblin_runtime_assert(_continuous_init_lower_bounds(i) < _continuous_init_upper_bounds(i));
    }

    if (target.has_value()) {
      if (std::holds_alternative<Vec<CType>>(target.value())) {
        register_target(std::get<Vec<CType>>(target.value()));
      } else if (std::holds_alternative<std::vector<CType>>(target.value())) {
        register_target(std::get<std::vector<CType>>(target.value()));
      } else {
        // this assumes that the evaluation function is reasonably cheap
        // (lazy evaluation would be an option, but more complex and not really
        // that beneficial)
        auto [dvals, cvals] = std::get<1>(target.value());
        register_target_front(dvals, cvals);
      }
    }
  };

  void set_init(AnyInit init) { _init = from_any_init(init); }

  void set_initial_bounds(std::variant<CType, Vec<CType>> continuous_init_lower_bound = CType(0.0),
                          std::variant<CType, Vec<CType>> continuous_init_upper_bound = CType(1.0)) {
    if (std::holds_alternative<CType>(continuous_init_lower_bound)) {
      _continuous_init_lower_bounds.resize(_objectives->num_continuous());
      _continuous_init_lower_bounds.fill(std::get<CType>(continuous_init_lower_bound));
    } else {
      _continuous_init_lower_bounds = std::get<Vec<CType>>(continuous_init_lower_bound);
    }
    __goblin_runtime_assert(static_cast<usize>(_continuous_init_lower_bounds.size()) == _objectives->num_continuous());

    if (std::holds_alternative<CType>(continuous_init_upper_bound)) {
      _continuous_init_upper_bounds.resize(_objectives->num_continuous());
      _continuous_init_upper_bounds.fill(std::get<CType>(continuous_init_upper_bound));
    } else {
      _continuous_init_upper_bounds = std::get<Vec<CType>>(continuous_init_upper_bound);
    }
    __goblin_runtime_assert(static_cast<usize>(_continuous_init_upper_bounds.size()) == _objectives->num_continuous());
  };

  void register_target(CRefS<Vec<CType>> target_objectives) {
    _target.clear();
    Solution s(
        archive_fitness().worst(),
        num_discrete() > 0 ? std::make_optional<Vec<DType>>(Vec<DType>::Zero(num_discrete())) : std::nullopt,
        num_continuous() > 0 ? std::make_optional<Vec<CType>>(Vec<CType>::Zero(num_continuous())) : std::nullopt);
    auto& q = s.quality_as<MOQuality>();
    q.objectives = target_objectives;
    __goblin_runtime_assert(static_cast<usize>(q.objectives.size()) >= fitness().num_objectives());
    q.constraint_value = 0.0;
    _target.update(s, false);
  };

  void register_target(std::vector<CType> target_objectives) {
    register_target(Eigen::Map<Vec<CType>>(target_objectives.data(), target_objectives.size()));
  };

  void register_target_front(const ArchiveBase& other) {
    _target.clear();
    for (usize i = 0; i < other.size(); i++) {
      _target.update(other[i], false);
    }
  };

  void register_target_front(Mat<DType> discrete, Mat<CType> continuous) {
    _target.clear();
    __goblin_runtime_assert(discrete.rows() == continuous.rows());
    __goblin_runtime_assert(static_cast<usize>(discrete.cols()) == num_discrete());
    __goblin_runtime_assert(static_cast<usize>(continuous.cols()) == num_continuous());
    for (isize i = 0; i < discrete.rows(); i++) {
      Solution s(archive_fitness().worst(),
                 num_discrete() > 0 ? std::make_optional<Vec<DType>>(discrete.row(i)) : std::nullopt,
                 num_continuous() > 0 ? std::make_optional<Vec<CType>>(continuous.row(i)) : std::nullopt);
      _objectives->evaluate(s);
      _target.update(s, false);
    }
  };

  void register_target_archive_size(usize target_archive_size) {
    __goblin_runtime_assert(target_archive_size > 1);
    _target_archive_size = target_archive_size;
  };

  CRef<Vec<DType>> discrete_domain_sizes() const override final { return _discrete_domain_sizes; };

  CRef<Vec<CType>> continuous_lower_bounds() const override final { return _continuous_lower_bounds; };
  CRef<Vec<CType>> continuous_upper_bounds() const override final { return _continuous_upper_bounds; };

  void add_random(Rng& rng, SolutionSetBase& solutions, usize count) const override final {
    _init->add_random(rng, *this, solutions, count);
  };

  CRef<Vec<CType>> continuous_init_lower_bounds() const override final { return _continuous_init_lower_bounds; }
  CRef<Vec<CType>> continuous_init_upper_bounds() const override final { return _continuous_init_upper_bounds; }

  void evaluate(Rng& rng, SolutionSetBase& solutions, const std::span<const usize>& indices) override final {
    for (auto i : indices) {
      _objectives->evaluate(solutions[i]);
    }
  };

  void evaluate_partial(Rng& rng,
                        SolutionSetBase& solutions,
                        SolutionSetBase& parents,
                        const std::vector<const Subset*>& subsets,
                        const std::span<const usize>& indices) override final {
    for (auto i : indices) {
      _objectives->evaluate_partial(solutions[i], parents[i], *(subsets[i]));
    }
  };

  const FitnessBase& fitness() const override final { return _fitness; }

  const ArchiveFitnessBase& archive_fitness() const override final { return _fitness; }

  bool target_reached(const ArchiveBase& archive) const override final {
    if (!_target.empty()) {
      return archive.covers(_target);
    } else if (_target_archive_size.has_value()) {
      return archive.size() >= _target_archive_size.value();
    } else {
      return false;
    }
  };

 private:
  std::shared_ptr<MOFunctionBase> _objectives;
  MOFitness _fitness;
  std::shared_ptr<InitBase> _init;
  UnboundedArchive _target;
  std::optional<usize> _target_archive_size;
  Vec<DType> _discrete_domain_sizes;
  Vec<CType> _continuous_lower_bounds;
  Vec<CType> _continuous_upper_bounds;

  Vec<CType> _continuous_init_lower_bounds;
  Vec<CType> _continuous_init_upper_bounds;
};

};  // namespace goblin

#endif /* _GOBLIN_BENCH_PROBLEM_H */

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                       goblin/bench/timer.h included by goblin.h                                              //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef _GOBLIN_BENCH_TIMER_H
#define _GOBLIN_BENCH_TIMER_H



namespace goblin {
class Timer {
 public:
  void start() { start_time_ = std::make_optional(std::chrono::high_resolution_clock::now()); }

  void stop() {
    auto now = std::chrono::high_resolution_clock::now();
    if (start_time_.has_value()) {
      total_time_ += std::chrono::duration_cast<std::chrono::nanoseconds>(now - start_time_.value());
      start_time_ = std::nullopt;
    }
  }

  std::chrono::nanoseconds elapsed() const { return total_time_; }

 private:
  std::chrono::nanoseconds total_time_{0};
  std::optional<std::chrono::high_resolution_clock::time_point> start_time_ = std::nullopt;
};
};  // namespace goblin

#endif /* _GOBLIN_BENCH_TIMER_H */

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                       goblin/bench/tracked.h included by goblin.h                                            //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef _GOBLIN_BENCH_TRACKED_H
#define _GOBLIN_BENCH_TRACKED_H

#include <exception>
#include <filesystem>
#include <fstream>
#include <ostream>


namespace goblin {

template <typename T>
inline void log_helper(std::ostream& os, const std::vector<T>& span, bool escape = true, bool _indent = false) {
  if (escape) {
    os << '"';
  }
  os << '[';
  usize i = 0;
  for (const auto& e : span) {
    if (i++ > 0) {
      os << ',';
    }
    if constexpr (std::same_as<T, char> || std::same_as<T, u8>) {
#ifdef __cpp_lib_print
      std::print(os,
#else
      os << std::format(
#endif
                 "{:d}", e);
    } else {
      os << e;
    }
  }
  os << ']';
  if (escape) {
    os << '"';
  }
};

template <typename EigenLike>
inline void log_helper(std::ostream& os, const EigenLike& m, bool escape = true, bool indent = false) {
  if (escape) {
    os << '"';
  }
  os << '[';
  if (m.rows() > 1 && m.cols() > 1) {
    for (isize r = 0; r < m.rows(); r++) {
      if (r > 0) {
        os << ',';
      }
      if (indent) {
        os << "\n  ";
      }
      os << '[';
      for (isize c = 0; c < m.cols(); c++) {
        if (c > 0) {
          os << ',';
        }

        // fmt to alwyas use the decimal instead of the ascii byte value for
        // (unsigned) chars
        if constexpr (std::same_as<typename EigenLike::Scalar, char> || std::same_as<typename EigenLike::Scalar, u8>) {
#ifdef __cpp_lib_print
          std::print(os,
#else
          os << std::format(
#endif
                     "{:d}", m(r, c));
        } else {
          os << m(r, c);
        }
      }
      os << ']';
    }
    if (indent) {
      os << '\n';
    }
  } else {
    for (isize i = 0; i < m.size(); i++) {
      if (i > 0) {
        os << ',';
      }
      // fmt to alwyas use the decimal instead of the ascii byte value for
      // (unsigned) chars
      if constexpr (std::same_as<typename EigenLike::Scalar, char> || std::same_as<typename EigenLike::Scalar, u8>) {
#ifdef __cpp_lib_print
        std::print(os,
#else
        os << std::format(
#endif
                   "{:d}", m(i));
      } else {
        os << m(i);
      }
    }
  }
  os << ']';
  if (escape) {
    os << '"';
  }
};

template <typename T>
inline std::string log_helper(const T& t, bool escape = true, bool indent = false) {
  std::ostringstream os;
  log_helper(os, t, escape, indent);
  return os.str();
};

class TrackingOptions {
 public:
  TrackingOptions() = delete;
  // TODO at some point think about enabling dynamically setting the logging
  // precision for floating points
  // TODO at some point allow these params on Tracked::run to reduce the amount
  // of config object nesting?
  TrackingOptions(std::filesystem::path logpath,
                  std::optional<std::vector<std::tuple<std::string, std::string>>> log_info = std::nullopt,
                  usize archive_capacity = 100,
                  u64 max_evaluations_until_archive_adaption = 100000,
                  bool consider_evaluation_time = true,
                  bool report_intermediate_results = true,
                  /// Report every time the elitist archive gets updated (for when algorithm behaviour is more
                  /// interesting than the results)
                  bool report_on_archive_change = false,
                  u64 initial_evaluations_until_next_report = 10,
                  u64 eval_factor = 2,
                  u64 max_evaluations_until_next_report = 1000000,
                  u64 initial_generations_until_next_report = 1,
                  u64 generation_factor = 2,
                  u64 max_generations_until_next_report = 100,
                  std::chrono::nanoseconds initial_time_until_next_report = std::chrono::seconds(1),
                  u64 time_factor = 2,
                  std::chrono::nanoseconds max_time_until_next_report = std::chrono::minutes(10))
      : archive_capacity(archive_capacity),
        max_evaluations_until_archive_adaption(max_evaluations_until_archive_adaption),
        consider_evaluation_time(consider_evaluation_time),
        report_intermediate_results(report_intermediate_results),
        report_on_archive_change(report_on_archive_change),
        initial_evaluations_until_next_report(initial_evaluations_until_next_report),
        eval_factor(eval_factor),
        max_evaluations_until_next_report(max_evaluations_until_next_report),
        initial_generations_until_next_report(initial_generations_until_next_report),
        generation_factor(generation_factor),
        max_generations_until_next_report(max_generations_until_next_report),
        initial_time_until_next_report(initial_time_until_next_report),
        time_factor(time_factor),
        max_time_until_next_report(max_time_until_next_report),
        logpath(logpath) {
    if (log_info.has_value()) {
      for (auto& kv : log_info.value()) {
        // TODO escape any '"' here?
        log_info_headers += std::get<0>(kv) + ',';
        log_info_values += std::get<1>(kv) + ',';
      }
    }
  };

  usize archive_capacity;
  u64 max_evaluations_until_archive_adaption;
  bool consider_evaluation_time;
  bool report_intermediate_results;
  bool report_on_archive_change;

  u64 initial_evaluations_until_next_report;
  u64 eval_factor;  // 1 is linear, >= 2 is exponential spacing
  u64 max_evaluations_until_next_report;

  u64 initial_generations_until_next_report;
  u64 generation_factor;  // 1 is linear, >= 2 is exponential spacing
  u64 max_generations_until_next_report;

  std::chrono::nanoseconds initial_time_until_next_report;
  u64 time_factor;  // 1 is linear, >= 2 is exponential spacing
  std::chrono::nanoseconds max_time_until_next_report;

 private:
  std::filesystem::path logpath;

  // key-value pairs to log, e.g.
  // [(method_name,AMaLGaM),(problem_name,Sphere),(dims,10),(run,99)]
  std::string log_info_headers;
  std::string log_info_values;

  friend class Tracked;
};

/// An instance that intercepts evaluations
class Tracked final : public WrappedInstance {
 public:
  Tracked() = delete;

  void evaluate(Rng& rng, SolutionSetBase& solutions, const std::span<const usize>& indices) override final {
    wrap_eval([&](const std::span<const usize>& _indices) { inner.evaluate(rng, solutions, _indices); }, solutions,
              indices);
  };
  void evaluate_partial(Rng& rng,
                        SolutionSetBase& solutions,
                        SolutionSetBase& parents,
                        const std::vector<const Subset*>& subsets,
                        const std::span<const usize>& indices) override final {
    wrap_eval(
        [&](const std::span<const usize>& _indices) {
          inner.evaluate_partial(rng, solutions, parents, subsets, _indices);
        },
        solutions, indices);
  };

  bool adapt(Rng& rng) override final {
    if (inner.adapt(rng)) {
      alg_timer.stop();
      // TODO Should these evaluations count (to both time/evals)? They are tracking only so for now they are not
      // counted, but with the consequence that if you track evaluations more evaluations than allowed by the budget
      // will be performed to arguably more correctly track the algorithm performance...
      inner.reevaluate_and_rebuild_archive(rng, archive);
      alg_timer.start();
      return true;
    } else {
      return false;
    }
  };

  Mat<CType> gradients(Rng& rng,
                       SolutionSetBase& solutions,
                       SolutionSetBase& parents,
                       const std::vector<const Subset*>& subsets,
                       const std::span<const usize>& indices,
                       u64& evaluations) override final {
    u64 evals_before = this->evaluations, _evals = evaluations;
    Mat<CType> res = inner.gradients(rng, solutions, parents, subsets, indices, evaluations);
    this->evaluations = std::max(this->evaluations, evals_before + evaluations - _evals);
    return res;
  }

  std::tuple<std::vector<usize>, u64> gradient_steps(Rng& rng,
                                                     SolutionSetBase& solutions,
                                                     SolutionSetBase& parents,
                                                     const std::span<const usize>& indices,
                                                     usize num_steps) override final {
    u64 evals_before = evaluations;
    auto res = inner.gradient_steps(rng, solutions, parents, indices, num_steps);

    alg_timer.stop();
    evaluations = std::max(evaluations, evals_before + /* evaluations */ std::get<1>(res));

    bool archive_changed = false;
    for (usize i : /* changed_indices */ std::get<0>(res)) {
      archive_changed |= archive.update(solutions[i], true);
    }
    // there is not enough information about the behaviour of gradient_steps for more granular reports
    if (archive_changed && config.report_on_archive_change) {
      report(archive);
    }

    if (inner.target_reached(archive)) {
      status = TerminationStatus::TargetReached;
      throw TrackingException("");
    }

    alg_timer.start();
    return res;
  }

  /// This can be used by the algorithm to log when debugging to log an
  /// `ArchiveBase`/`SolutionSetBase`-like type. Both the passed headers and
  /// values need to be empty, or valid csv columns with a ',' separator at the
  /// end (possibly escaping other ',' occurrences with '"').
  template <typename P>
  void request_debug_report(std::filesystem::path debug_logpath,
                            const P& solutions,
                            std::string_view debug_headers,
                            std::string_view debug_values) {
    // close the actual logfile to open up the debug logpath next
    if (logfile.is_open()) {
      logfile.close();
    }

    // temporarily change the logpath
    auto tracked_generation = generation;
    generation = method.current_generation();
    std::swap(debug_logpath, config.logpath);
    report(solutions, debug_headers, debug_values);
    std::swap(debug_logpath, config.logpath);
    generation = tracked_generation;

    // close the debug logfile to open up the actual logpath again next
    if (logfile.is_open()) {
      logfile.close();
    }
  };

  void request_debug_report(std::filesystem::path debug_logpath,
                            std::string_view debug_headers,
                            std::string_view debug_values) {
    request_debug_report(debug_logpath, archive, debug_headers, debug_values);
  };

  static std::tuple<std::shared_ptr<ArchiveBase>, TerminationStatus> run(
      InstanceBase& instance,
      MethodBase& method,
      Budget& budget,
      TrackingOptions config,
      std::optional<usize> seed = std::nullopt,
      std::optional<usize> population_size = std::nullopt) {
    std::random_device rd;
    usize _seed = seed.has_value() ? seed.value()
                                   : std::uniform_int_distribution<usize>(1, std::numeric_limits<usize>::max())(rd);
    Tracked ti(instance, method, budget, config, _seed);
    try {
      ti.alg_timer.start();
      auto [_, alg_status] = method.run(ti, budget, _seed, population_size);
      ti.alg_timer.stop();
      ti.status = alg_status;

      // TODO why is this necessary? (alg should not be able to hit the target without the exception being thrown...)
      if (instance.target_reached(ti.archive)) {
        ti.status = TerminationStatus::TargetReached;
      }

      // // effectively guess the reason for stopping - assume convergence
      // // unless the budget is exhausted or the target was reached
      // auto elapsed = ti.alg_timer.elapsed() + ti.eval_timer.elapsed();
      // auto ts =
      //     budget.exhausted(ti.generation.value_or(0), ti.evaluations,
      //     elapsed);
      // if (ts.has_value()) {
      //   ti.status = ts.value();
      // } else if (instance.target_reached(ti.archive)) {
      //   ti.status = TerminationStatus::TargetReached;
      // } else {
      //   ti.status = TerminationStatus::Converged;
      // }
    } catch (const TrackingException& e) {
    }

    ti.report(ti.archive);

    return std::make_tuple(std::make_shared<AdaptiveGridArchive>(std::move(ti.archive)), ti.status);
  };

 private:
  struct TrackingException : std::runtime_error {
    using std::runtime_error::runtime_error;
  };

  Tracked(InstanceBase& instance, MethodBase& method, Budget& budget, TrackingOptions config, usize seed)
      : WrappedInstance(instance),
        // instance(instance),
        method(method),
        budget(budget),
        config(config),
        seed(seed),
        status(TerminationStatus::Running),
        archive(inner.archive_fitness(), config.archive_capacity),
        generation(std::nullopt),
        last_generation(0),
        generations_at_next_report(config.initial_generations_until_next_report),
        evaluations(0),
        evaluations_at_next_report(config.initial_evaluations_until_next_report),
        time_elapsed_at_next_report(config.initial_time_until_next_report) {};

  template <typename E>
  void wrap_eval(E eval, SolutionSetBase& solutions, const std::span<const usize>& indices) {
    alg_timer.stop();

    // TODO if evaluations == 0 optionally add all solutions to the archive, not just indices => correct tracking for
    // warm start scenarios where some solutions already have a fitness

    // check if the budget was exhausted while the algorithm was running
    auto elapsed = config.consider_evaluation_time ? alg_timer.elapsed() + eval_timer.elapsed() : alg_timer.elapsed();
    generation = method.current_generation();
    auto ts = budget.exhausted(generation.value_or(0), evaluations, elapsed);
    if (ts.has_value()) {
      status = ts.value();
      throw TrackingException("");
    }

    // actually evaluate, but ensure we don't go beyond the evaluation limit
    u64 evaluations_performed;
    if (budget.max_evaluations.has_value() && evaluations + indices.size() > budget.max_evaluations.value()) {
      auto evals_left = budget.max_evaluations.value() - evaluations;
      eval_timer.start();
      eval(indices.first(evals_left));
      eval_timer.stop();
      evaluations_performed = evals_left;

      assert(budget.exhausted(0, evaluations + evaluations_performed, elapsed).has_value() &&
             "Evaluation limit was not reached!");
    } else {
      eval_timer.start();
      eval(indices);
      eval_timer.stop();
      evaluations_performed = indices.size();
    }

    // update the internal archive, and possibly stop if the target was reached
    for (usize i = 0; i < evaluations_performed; i++) {
      bool archive_changed = archive.update(solutions[indices[i]], true);
      evaluations++;  // update the evaluations one at time to be "truthful" in case of an early return before all
                      // evaluations performed were considered...

      if (archive_changed && config.report_on_archive_change) {
        report(archive);
      }

      // the vtr is checked for each solution to level the playing field between batched algorithms and algorithms
      // evaluating one by one
      if (inner.target_reached(archive)) {
        status = TerminationStatus::TargetReached;
        throw TrackingException("");
      }
    }

    evaluations_since_last_archive_adaption += evaluations_performed;

    // check if we need to stop because the evaluation time/evaluations
    // exhausted the budget
    elapsed = config.consider_evaluation_time ? alg_timer.elapsed() + eval_timer.elapsed() : alg_timer.elapsed();
    ts = budget.exhausted(0, evaluations, elapsed);
    if (ts.has_value()) {
      status = ts.value();
      throw TrackingException("");
    }

    // if we made it here, possibly report (if not, a final report
    // will be generated - no need for doing the final report twice)
    if (should_report()) {
      report(archive);
    }
    if (evaluations_since_last_archive_adaption >= config.max_evaluations_until_archive_adaption) {
      evaluations_since_last_archive_adaption = 0;
      archive.adapt();
    }

    alg_timer.start();
  };

  bool should_report() {
    if (!config.report_intermediate_results)
      return false;

    bool report_needed = false;
    if (evaluations >= evaluations_at_next_report) {
      report_needed = true;
      evaluations_at_next_report =
          std::min(config.eval_factor > 1 ? evaluations * config.eval_factor
                                          : evaluations + config.initial_evaluations_until_next_report,
                   evaluations + config.max_evaluations_until_next_report);
    }

    u64 g = generation.value_or(0);
    if (g >= generations_at_next_report && g != last_generation) {
      last_generation = g;
      report_needed = true;
      generations_at_next_report =
          std::min(config.generation_factor > 1 ? g * config.generation_factor
                                                : g + config.initial_generations_until_next_report,
                   g + config.max_generations_until_next_report);
    }

    auto elapsed = config.consider_evaluation_time ? alg_timer.elapsed() + eval_timer.elapsed() : alg_timer.elapsed();
    if (elapsed > time_elapsed_at_next_report) {
      report_needed = true;
      time_elapsed_at_next_report = std::min(
          config.time_factor > 1
              ? std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed * config.time_factor)
              : std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed + config.initial_time_until_next_report),
          elapsed + config.max_time_until_next_report);
    }

    return report_needed;
  };

  template <typename A>
  void report(const A& solutions, std::string_view debug_headers = "", std::string_view debug_values = "") {
    namespace fs = std::filesystem;
    typedef std::chrono::duration<double> Seconds;

    if (config.logpath == "/dev/null") {
      return;
    }

    if (!logfile.is_open()) {
      if (!config.logpath.parent_path().empty()) {
        fs::create_directories(config.logpath.parent_path());
      }

      // clear the file if it was not cleared before
      if (truncated_files.contains(config.logpath)) {
        logfile.open(config.logpath, std::ios::out | std::ios::app | std::ios::ate);
      } else {
        truncated_files.insert(config.logpath);
        logfile.open(config.logpath, std::ios::out | std::ios::trunc);
      }

      if (fs::is_empty(config.logpath)) {
        // clang-format off
        logfile <<
            "status,"
            "evaluations,"
            "generation,"
            "total_time_seconds,"
            "alg_time_seconds,"
            "eval_time_seconds,"
            "current_population_size,"
            "current_population_generation,"
            << config.log_info_headers
            << debug_headers <<
            "seed,"
            "discrete,"
            "discrete_active,"
            "continuous,"
            "continuous_active,"
        ;
        // clang-format on
        inner.log_header(logfile);
        logfile << std::endl;  // here we want to flush
      }
    }

    std::string gen = generation.has_value() ? std::to_string(generation.value()) : "", pop_size = "", pop_gen = "";
    auto pop_info = method.current_population();
    if (pop_info.has_value()) {
      auto [p_size, p_gen] = pop_info.value();
      pop_size = std::to_string(p_size);
      pop_gen = std::to_string(p_gen);
    }
    Seconds alg_time = alg_timer.elapsed();
    Seconds eval_time = eval_timer.elapsed();
    Seconds total_time = alg_time + eval_time;

    auto common =
        std::format("{},{},{},{},{},{},{},{},{}{}{},", format_as(status), evaluations, gen, total_time.count(),
                    alg_time.count(), eval_time.count(), pop_size, pop_gen, config.log_info_values, debug_values, seed);

    for (usize i = 0; i < solutions.size(); i++) {
      const auto& s = solutions[i];
      // clang-format off
        logfile << common;
        log_helper(logfile,   s.discrete_values(), true); logfile << ',';
        log_helper(logfile,   s.discrete_active(), true); logfile << ',';
        log_helper(logfile, s.continuous_values(), true); logfile << ',';
        log_helper(logfile, s.continuous_active(), true); logfile << ',';
      // clang-format on
      inner.log(logfile, s);
      logfile << "\n";
    }
    logfile << std::flush;
  };

  // InstanceBase& instance;
  MethodBase& method;
  Budget& budget;
  TrackingOptions config;
  usize seed;

  TerminationStatus status;
  AdaptiveGridArchive archive;

  Timer alg_timer;
  Timer eval_timer;

  std::optional<u64> generation;
  u64 last_generation;
  u64 generations_at_next_report;

  u64 evaluations;
  u64 evaluations_at_next_report;
  std::chrono::nanoseconds time_elapsed_at_next_report;

  usize evaluations_since_last_archive_adaption;

  std::ofstream logfile;
  std::set<std::filesystem::path> truncated_files;
};

/// Tracked running was intended to unify reporting across algorithms
/// - this method abuses that functionality to re-use that logging for
/// other purposes controlled by the algorithm, not the tracking
inline void debug_log(InstanceBase& problem,
                      std::string_view path,
                      std::string_view headers = "",
                      std::string_view values = "",
                      std::optional<std::reference_wrapper<const SolutionSetBase>> population = std::nullopt) {
  if (auto ti = dynamic_cast<Tracked*>(&problem); ti != nullptr) {
    if (population.has_value()) {
      ti->request_debug_report(path, population.value().get(), headers, values);
    } else {
      ti->request_debug_report(path, headers, values);
    }
  } else {
    throw std::runtime_error(
        "Debug log called on an incompatible problem instance. Try using "
        "`Tracked::run` to enable logging.");
  }
};

template <typename T>
inline std::string iterator2str(T&& it) {
  std::ostringstream os;
  os << '[';
  usize i = 0;
  for (const auto& e : it) {
    if (i++ > 0) {
      os << ',';
    }
    os << e;
  }
  os << ']';
  return os.str();
};

};  // namespace goblin

#endif /* _GOBLIN_BENCH_TRACKED_H */


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                       goblin/methods/ims.h included by goblin.h                                              //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef _GOBLIN_LIB_IMS_H
#define _GOBLIN_LIB_IMS_H



namespace goblin {

struct IMSOptions {
  usize initial_population_size = 2;
  usize max_num_populations = 25;
  usize subgeneration_factor = 4;
  bool restart_stale_populations = false;
  bool stop_covered_populations = false;
  usize initial_num_clusters = 1;
  std::optional<usize> archive_capacity = 100;
  bool so_parameter_space_clustering = false;  // TODO remove or implement parameter space clustering
  usize additional_clusters_per_start = 1;
  std::optional<usize> generations_without_improvement_until_restart = std::nullopt;

  bool reevaluate_solutions_after_adaption = true;

  std::optional<std::string> population_logfile = std::nullopt;
  std::string population_log_resolution = "archive";
};

template <typename P>
class IMS final : public MethodBase {
  using C = std::function<
      P(InstanceBase& /* problem */, ArchiveBase& /* global_archive */, usize /* size */, usize /* num_clusters
                                                                                                 */
        )>;

 public:
  IMS(C create_population, IMSOptions options = IMSOptions())
      : create_population(create_population), options(options), total_generations(0) {};

  std::tuple<std::shared_ptr<ArchiveBase>, TerminationStatus> run(
      InstanceBase& problem,
      const Budget& budget,
      std::optional<u64> seed = std::nullopt,
      std::optional<usize> population_size = std::nullopt) override final {
    bool is_multi_objective = problem.num_objectives() > 1;
    IMSOptions opts = options;
    if (opts.initial_num_clusters == 0) {
      opts.initial_num_clusters = problem.num_objectives() + (is_multi_objective ? 1 : 0);
    }

    if (opts.initial_population_size <= opts.initial_num_clusters) {
      opts.initial_population_size *= opts.initial_num_clusters;
    }

    if (population_size.has_value()) {
      opts.initial_population_size = population_size.value();
      opts.max_num_populations = 1;
    }

    if (!opts.so_parameter_space_clustering && !is_multi_objective) {
      opts.additional_clusters_per_start = 0;
    }

    Rng rng = seeded_rng(seed);
    auto archive =
        opts.archive_capacity.has_value() && opts.archive_capacity.value() > 0
            ? std::static_pointer_cast<ArchiveBase>(std::make_shared<UnboundedArchive>(problem.archive_fitness()))
            : std::static_pointer_cast<ArchiveBase>(
                  std::make_shared<AdaptiveGridArchive>(problem.archive_fitness(), opts.archive_capacity.value()));

    std::vector<P> populations;
    populations.reserve(opts.max_num_populations);
    sizes.clear();
    sizes.reserve(opts.max_num_populations);
    generations.clear();
    generations.reserve(opts.max_num_populations);
    std::vector<bool> running;
    running.reserve(opts.max_num_populations);
    std::vector<usize> generations_since_last_improvement;
    generations_since_last_improvement.reserve(opts.max_num_populations);

    total_generations = 0;
    u64 evaluations = 0;
    std::chrono::high_resolution_clock::time_point t_start = std::chrono::high_resolution_clock::now();

    auto should_terminate = [&](usize additional_evaluations = 0,
                                bool check_external_criterion = false) -> std::optional<TerminationStatus> {
      std::chrono::high_resolution_clock::time_point now = std::chrono::high_resolution_clock::now();
      std::chrono::nanoseconds elapsed = now - t_start;

      auto status = check_external_criterion
                        ? budget.exhausted_or_external_criterion_met(total_generations,
                                                                     evaluations + additional_evaluations, elapsed)
                        : budget.exhausted(total_generations, evaluations + additional_evaluations, elapsed);
      if (status.has_value()) {
        // std::println("{}", format_as(status.value()));
        return status;
      }

      if (problem.target_reached(*archive)) {
        // std::println("{}", format_as(TerminationStatus::TargetReached));
        return std::make_optional(TerminationStatus::TargetReached);
      }

      return std::nullopt;
    };

    auto any_running = [&]() {
      for (auto r : running) {
        if (r) {
          return true;
        }
      }
      return false;
    };

    Vec<CType> avg_dist_to_so_elite(opts.max_num_populations);
    p_idx = 0;
    while (!should_terminate(/* additional_evaluations = */ 0, /* check_external_criterion = */ true) &&
           (populations.size() < opts.max_num_populations || opts.restart_stale_populations || any_running())) {
      // init/restart population if necessary
      if (p_idx >= populations.size()) {
        usize size = opts.initial_population_size * std::pow(2, p_idx);
        sizes.push_back(size);
        populations.push_back(create_population(
            problem, *archive, size, opts.initial_num_clusters + p_idx * opts.additional_clusters_per_start));
        generations.push_back(0);
        generations_since_last_improvement.push_back(0);
        running.push_back(true);
      } else if (!running[p_idx] && opts.restart_stale_populations &&
                 (p_idx == opts.max_num_populations - 1 || (!opts.stop_covered_populations && is_multi_objective))) {
        // std::println("[IMS]: Restarting population {}", p_idx);
        populations[p_idx].restart();
        generations[p_idx] = 0;
        generations_since_last_improvement[p_idx] = 0;
        running[p_idx] = true;
      }

      // do a step, optionally terminate this or smaller ones
      generations[p_idx]++;  // this needs to always be increased, no matter if we do
                             // a step or not
      if (running[p_idx]) {
        // TODO not the right place, but at some point add the machinery to store good but dominated alternative
        // solutions?
        // TODO this is batching as Marco does it, but Evi (https://arxiv.org/pdf/2402.12510v1#subsection.4.2) keeps the
        // elite archive updated with full evaluations, and local archives are fully evaluated at the end of each
        // generation -> this corresponds to adding an option to adapt to set it to the full problem (if available) and
        // to add yet another archive that only stores fully evaluated solutions here
        bool reevaluate_solutions = false;
        if (problem.adapt(rng)) {
          // re-evaluate & re-build the archives if the problem instance changed (e.g. different mini batch)
          evaluations += problem.reevaluate_and_rebuild_archive(rng, *archive);
          evaluations += problem.reevaluate_and_rebuild_archive(rng, populations[p_idx].archive());

          // potentially expensive, but might be needed to avoid "faulty" acceptance decisions (if new fitness is
          // considerably worse, chances are that small improvements are not accepted and the old, out-of-date fitness
          // is kept...)
          reevaluate_solutions = opts.reevaluate_solutions_after_adaption;
        }

        archive->reset_change_count();
        evaluations += populations[p_idx].perform_generation(rng, should_terminate, reevaluate_solutions);
        total_generations++;

        if (opts.population_logfile.has_value()) {
          AoSSet p;  // copy is needed because p needs to be non-const, and that is the case because logging for the sr
                     // problem at this point in time might do a test set evaluation...
          if (opts.population_log_resolution == "archive") {
            const auto& a = populations[p_idx].archive();
            for (usize i = 0; i < a.size(); i++) {
              p.add(a[i]);
            }
          } else if (opts.population_log_resolution == "population") {
            const auto& s = populations[p_idx].get_solutions();
            for (usize i = 0; i < s.size(); i++) {
              p.add(s[i]);
            }
          } else {
            throw std::runtime_error("Unknown population log resolution.");
          }

          debug_log(problem, opts.population_logfile.value(), "", "", p);
        }

        if (archive->change_count() > 0) {
          generations_since_last_improvement[p_idx] = 0;
        } else {
          generations_since_last_improvement[p_idx]++;
        }

        archive->adapt();

        if (!is_multi_objective || opts.stop_covered_populations) {
          for (usize j = 0; j < p_idx; j++) {
            if (populations[p_idx].archive().covers(populations[j].archive())) {
              running[j] = false;
            }
          }
        }
        if (!is_multi_objective) {  //  && problem.num_discrete() == 0) {  // continuous only
          // since we only have relative comparisons, this roughly is equal to the usual avg fitness of larger
          // population is better condition
          avg_dist_to_so_elite(p_idx) = populations[p_idx].avg_dist_to_global_so_elite();
          for (usize j = 0; j < p_idx; j++) {
            if (avg_dist_to_so_elite(j) > avg_dist_to_so_elite(p_idx)) {
              running[j] = false;
            }
          }
        }
        if (populations[p_idx].converged() ||
            (opts.restart_stale_populations &&
             generations_since_last_improvement[p_idx] >
                 opts.generations_without_improvement_until_restart.value_or(total_generations))) {
          running[p_idx] = false;
        }
      }

      // go to the next population to do a step with
      p_idx = generations[p_idx] % opts.subgeneration_factor == 0 ? (p_idx + 1) % opts.max_num_populations : 0;
    }

    // std::println(
    //     "{} && ({} || {} || {}) - G: {} / #P: {} of {}", !should_terminate(),
    //     populations.size() < opts.max_num_populations,
    //     opts.restart_stale_populations, any_running(), total_generations,
    //     populations.size(), opts.max_num_populations);

    // for (usize p_idx = 0; p_idx < archive->size(); p_idx++) {
    //   std::println("{}", problem.format_solution((*archive)[p_idx]));
    // }

    return std::make_tuple(archive, should_terminate().value_or(TerminationStatus::Converged));
  };

  std::optional<u64> current_generation() const override final { return total_generations; };

  std::optional<std::tuple<usize, u64>> current_population() const override {
    return std::make_tuple(sizes[p_idx], generations[p_idx]);
  };

 private:
  C create_population;
  IMSOptions options;
  // The whole reason run is not a static method -
  // generation reporting needs to be accessible (via an IMS instance)
  u64 total_generations;

  usize p_idx;
  std::vector<usize> sizes;
  std::vector<u64> generations;
};

};  // namespace goblin

#endif /* _GOBLIN_LIB_IMS_H */

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                       goblin/methods/amalgam.h included by goblin.h                                          //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef _GOBLIN_AMALGAM_H
#define _GOBLIN_AMALGAM_H




namespace goblin {

class AMaLGaM final : public MethodBase {
 public:
  AMaLGaM(double init_lower_bound = -1.0,
          double init_upper_bound = 1.0,
          double fitness_variance_threshold = 0.0,
          usize max_num_populations = 25,
          usize max_evaluations = 1000000,
          std::optional<double> target_value = std::nullopt,
          std::optional<usize> base_population_size = std::nullopt,
          std::optional<usize> seed = std::nullopt)
      : init_lower_bound(init_lower_bound),
        init_upper_bound(init_upper_bound),
        fitness_variance_threshold(fitness_variance_threshold),
        max_num_populations(max_num_populations),
        max_evaluations(max_evaluations),
        target_value(target_value),
        base_population_size(base_population_size),
        seed(seed) {
    __goblin_runtime_assert(!isna(init_lower_bound));
    __goblin_runtime_assert(!isna(init_upper_bound));
    __goblin_runtime_assert(init_lower_bound < init_upper_bound);
  }

  void set_init_bounds(double lower, double upper) {
    __goblin_runtime_assert(!isna(lower));
    __goblin_runtime_assert(!isna(upper));
    __goblin_runtime_assert(lower < upper);
    init_lower_bound = lower;
    init_upper_bound = upper;
  };

  std::tuple<Vec<double>, double, double> run(
      std::function<std::tuple<double, double>(const Ref<Vec<double>>&)>& function,
      std::vector<std::tuple<double, double>> bounds);

  std::optional<u64> current_generation() const override final;

  std::tuple<std::shared_ptr<ArchiveBase>, TerminationStatus> run(
      InstanceBase& problem,
      const Budget& budget,
      std::optional<u64> seed = std::nullopt,
      std::optional<usize> population_size = std::nullopt) override final {
    if (problem.num_discrete() > 0 || problem.num_continuous() < 1 || problem.num_objectives() > 1) {
      __goblin_runtime_assert(false);  // Problem not supported
    }

    auto archive = std::make_shared<UnboundedArchive>(problem.archive_fitness());
    Rng rng = seeded_rng(seed);

    // make a copy to persist other options between calls on the same instance
    AMaLGaM alg = *this;

    alg.seed = seed;

    if (population_size.has_value()) {
      alg.max_num_populations = 1;
      alg.base_population_size = population_size.value();
    }

    alg.init_lower_bound = problem.continuous_init_lower_bounds().maxCoeff();
    alg.init_upper_bound = problem.continuous_init_upper_bounds().minCoeff();

    if (budget.max_evaluations.has_value()) {
      alg.max_evaluations = budget.max_evaluations.value();
    }

    std::vector<std::tuple<double, double>> bounds;
    for (usize i = 0; i < problem.num_continuous(); i++) {
      bounds.emplace_back(problem.continuous_lower_bounds()(i), problem.continuous_upper_bounds()(i));
    }

    AoSSet s;
    s.add(Solution(problem.archive_fitness().worst(), std::nullopt, Vec<double>::Zero(problem.num_continuous())));
    std::vector<usize> idxs{0};
    std::function<std::tuple<double, double>(const Eigen::Ref<Eigen::VectorXd>& parameters)> fn =
        [&](const Eigen::Ref<Eigen::VectorXd>& parameters) {
          s[0].continuous_values() = parameters;
          problem.evaluate(rng, s, idxs);

          archive->update(s[0], true);
          if (problem.target_reached(*archive)) {
            throw std::runtime_error("");
          }

          auto& q = s[0].quality_as<MOQuality>();
          return std::make_tuple(q.objectives(0), q.constraint_value);
        };
    try {
      alg.run(fn, bounds);
    } catch (std::runtime_error& e) {
    }

    // TODO better terminationstatus guess
    // - target reached if target was reached
    // - evaluations
    return std::make_tuple(archive, TerminationStatus::Converged);
  };

 private:
  double init_lower_bound = -1.0;
  double init_upper_bound = 1.0;
  double fitness_variance_threshold = 0.0;
  usize max_num_populations = 25;
  usize max_evaluations = 1000000;
  std::optional<double> target_value = std::nullopt;
  std::optional<usize> base_population_size = std::nullopt;
  std::optional<usize> seed = std::nullopt;
};

};  // namespace goblin

#endif /* _GOBLIN_AMALGAM_H */

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                       goblin/methods/library.h included by goblin.h                                          //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef _GOBLIN_GOMEA_LIBRARY_H
#define _GOBLIN_GOMEA_LIBRARY_H


#include <gomea/src/common/linkage_config.hpp>
#include <gomea/src/discrete/Config.hpp>
#include <gomea/src/discrete/gomeaIMS.hpp>
#include <gomea/src/real_valued/Config.hpp>
#include <gomea/src/real_valued/rv-gomea.hpp>


// Doesn't work yet since we store the full class, not a pointer...
// // forward declaration to avoid pulling in the library headers in the header
// namespace gomea {
// struct linkage_config_t;

// namespace discrete {
// struct Config;
// struct gomeaIMS;
// };  // namespace discrete

// namespace realvalued {
// struct Config;
// struct rvg_t;
// };  // namespace realvalued
// };  // namespace gomea

namespace goblin {
class DiscreteGOMEA final : public MethodBase {
 public:
  DiscreteGOMEA(std::string_view linkage_model = "LinkageTree",
                std::string similarity_metric = "MI",
                bool gene_invariant = false,  // enables GI-GOMEA
                bool forced_improvements = true,
                bool filter_linkage = false,
                std::optional<usize> max_subset_size = std::nullopt,  // only used for the linkage tree FOS
                usize base_population_size = 2,         // The size of the first population in the multi-start scheme.
                usize max_number_of_populations = 100,  // The maximum number of populations in the multi-start scheme.
                usize subgeneration_factor = 4,         // The subgeneration factor in the multi-start scheme.
                usize max_archive_size = 0,
                std::string fos_order = "default"  // parallel, fixed
  );

  std::tuple<std::shared_ptr<ArchiveBase>, TerminationStatus> run(
      InstanceBase& problem,
      const Budget& budget,
      std::optional<u64> seed = std::nullopt,
      std::optional<usize> population_size = std::nullopt) override final;

  std::optional<u64> current_generation() const override final;

 private:
  gomea::linkage_config_t linkage_config;
  gomea::discrete::Config config;
  std::unique_ptr<gomea::discrete::gomeaIMS> instance;
};

class RvGOMEA final : public MethodBase {
 public:
  RvGOMEA(std::string_view linkage_model = "Full",
          usize base_population_size = 10,       // The size of the first population in the multi-start scheme.
          usize max_number_of_populations = 25,  // The maximum number of populations in the multi-start scheme.
          usize subgeneration_factor = 8,        // The subgeneration factor in the multi-start scheme.
          usize max_nis = 100,                   // The maximum number of subsequent generations without an
                                                 // improvement while the distribution multiplier is <= 1.0.
          double selection_percentile = 0.35,    // The selection truncation percentile
                                                 // (in [1/population_size,1]).
          double distribution_multiplier_decrease = 0.9,  // The multiplicative distribution multiplier decrease.
          double standard_deviation_threshold = 1.0,  // The maximum ratio of the distance of the average improvement
                                                      // to the mean compared to the distance of one standard deviation
                                                      // before triggering AVS (SDR mechanism).
          double fitness_variance_tolerance = 0.0,    // The minimum fitness variance level that is allowed.
          bool selection_during_gom = true,           // Update the current distribution
                                                      // estimate for each GOM step/FOS subset
          bool update_elitist_during_gom = true       // Update the current elite for each GOM step/FOS subset
  );

  std::tuple<std::shared_ptr<ArchiveBase>, TerminationStatus> run(
      InstanceBase& problem,
      const Budget& budget,
      std::optional<u64> seed = std::nullopt,
      std::optional<usize> population_size = std::nullopt) override final;

  std::optional<u64> current_generation() const override final;

 private:
  gomea::realvalued::Config config;
  std::unique_ptr<gomea::realvalued::rvg_t> instance;
};
};  // namespace goblin

#endif /* _GOBLIN_GOMEA_LIBRARY_H */

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                       goblin/methods/mo_binary.h included by goblin.h                                        //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef _GOBLIN_MO_BINARY_GOMEA_H
#define _GOBLIN_MO_BINARY_GOMEA_H




namespace goblin {

class MOBinaryGOMEA final : public MethodBase {
 public:
  MOBinaryGOMEA(usize initial_population_size = 8,
                usize max_num_populations = 25,
                usize subgeneration_factor = 2,
                usize target_archive_size = 100,
                std::string mutation = "none",
                bool stop_covered_populations = false)
      : initial_population_size(initial_population_size),
        max_num_populations(max_num_populations),
        subgeneration_factor(subgeneration_factor),
        target_archive_size(target_archive_size),
        mutation(mutation),
        stop_covered_populations(stop_covered_populations) {};

  std::optional<u64> current_generation() const override final;

  std::tuple<std::shared_ptr<ArchiveBase>, TerminationStatus> run(
      InstanceBase& problem,
      const Budget& budget,
      std::optional<u64> seed = std::nullopt,
      std::optional<usize> population_size = std::nullopt) override final {
    if (problem.num_discrete() < 1 || problem.num_continuous() > 0 ||
        (problem.discrete_domain_sizes().array() > DType(2)).any()) {
      __goblin_runtime_assert(false);  // problem not supported
    }

    auto archive = std::make_shared<UnboundedArchive>(problem.archive_fitness());
    TerminationStatus status = TerminationStatus::Converged;
    Rng rng = seeded_rng(seed);

    AoSSet s;
    s.add(Solution(problem.archive_fitness().worst(), Vec<DType>::Zero(problem.num_discrete()), std::nullopt));
    std::vector<usize> idxs{0};
    std::function<void(char*, double*, double*)> fn = [&](char* solution, double* obj, double* con) {
      for (usize i = 0; i < problem.num_discrete(); i++) {
        s[0].discrete_values()(i) = static_cast<DType>(solution[i]);
      }
      problem.evaluate(rng, s, idxs);

      archive->update(s[0], true);
      if (problem.target_reached(*archive)) {
        status = TerminationStatus::TargetReached;
        throw std::runtime_error("");
      }

      auto& q = s[0].quality_as<MOQuality>();
      for (usize i = 0; i < problem.num_objectives(); i++) {
        obj[i] = q.objectives(i);
      }
      *con = q.constraint_value;
    };

    try {
      run_helper(fn, problem.num_objectives(), problem.num_discrete(),
                 population_size.value_or(initial_population_size),
                 population_size.has_value() ? 1 : max_num_populations, subgeneration_factor, target_archive_size,
                 mutation, stop_covered_populations, seed, budget.max_evaluations);
    } catch (std::runtime_error& e) {
    }
    return std::make_tuple(archive, status);
  };

 private:
  void run_helper(std::function<void(char*, double*, double*)> evaluate,
                  usize num_objectives,
                  usize num_parameters,
                  usize initial_population_size,
                  usize max_num_populations,
                  usize subgeneration_factor,
                  usize target_archive_size,
                  std::string mutation,
                  bool stop_covered_populations,
                  std::optional<u64> seed,
                  std::optional<usize> max_evaluations);

  usize initial_population_size;
  usize max_num_populations;
  usize subgeneration_factor;
  usize target_archive_size;
  std::string mutation;
  bool stop_covered_populations;
};

};  // namespace goblin

#endif /* _GOBLIN_MO_BINARY_GOMEA_H */

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                       goblin/methods/mixed.h included by goblin.h                                            //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef _GOBLIN_MIXED_GOMEA_H
#define _GOBLIN_MIXED_GOMEA_H



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                       goblin/methods/continuous.h included by goblin/methods/mixed.h                         //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef _GOBLIN_METHODS_CONTINUOUS_H
#define _GOBLIN_METHODS_CONTINUOUS_H


#include <Eigen/Cholesky>
#include <Eigen/QR>


namespace goblin {

static Vec<CType> estimate_mean(const SolutionSetBase& solutions,
                                const std::span<const usize> active_indices,
                                const std::span<const usize> active_counts,
                                double selection_percentile,
                                const std::span<const usize> indices,
                                bool intron_aware) {
  Vec<CType> mean = Vec<CType>::Zero(indices.size());
  usize selection_size = 0;
  if (intron_aware) {
    for (usize i : indices) {
      selection_size = std::max(selection_size, static_cast<usize>(selection_percentile * active_counts[i]));
    }

    if (selection_size > 0) {
      bool any_left = true;
      std::vector<usize> num_left(indices.size(), selection_size);
      for (usize i : active_indices) {
        any_left = false;
        for (usize j = 0; j < indices.size(); j++) {
          if (num_left[j] > 0 &&
              (solutions[i].continuous_active()(
                   indices[j])                    // index is active -> we use the value under selection pressure
               || active_counts[indices[j]] == 0  // index is not active -> we still want a non-zero mean to hopefully
                                                  // make the cholesky decomposition work (= we want to keep diversity)
               )) {
            mean(j) += solutions[i].continuous_values()(indices[j]);
            num_left[j]--;
            any_left |= num_left[j] > 0;
          }
        }
        if (!any_left) {
          break;
        }
      }
      for (usize j = 0; j < indices.size(); j++) {
        mean(j) /= static_cast<CType>(selection_size - num_left[j]);
      }
    }
  }

  if (selection_size == 0) {
    selection_size = selection_percentile * active_indices.size();
    for (usize i = 0; i < selection_size; i++) {
      mean += solutions[active_indices[i]].continuous_values()(indices);
    }
    mean /= static_cast<CType>(selection_size);
  }
  return mean;
};

static Mat<CType> estimate_cov(const SolutionSetBase& solutions,
                               const std::span<const usize> active_indices,
                               const std::span<const usize> active_counts,
                               double selection_percentile,
                               const Vec<CType>& mean,
                               const std::span<const usize> indices,
                               bool intron_aware) {
  assert(static_cast<usize>(mean.size()) == indices.size());
  Mat<CType> cov = Mat<CType>::Zero(indices.size(), indices.size());
  if (intron_aware) {
    for (usize i = 0; i < indices.size(); i++) {
      if (active_counts[indices[i]] == 0) {
        // estimate the full univariate covariance if there are no active variables to keep diversity (no selection)
        for (usize l : active_indices) {
          cov(i, i) += std::pow(solutions[l].continuous_values()(indices[i]) - mean(i), 2);
        }
        cov(i, i) /= static_cast<CType>(active_indices.size());
      } else {
        // otherwise estimate the covariance between active variables
        for (usize j = i; j < indices.size(); j++) {
          if (active_counts[indices[j]] > 0) {
            usize selection_size =
                selection_percentile * std::max(active_counts[indices[i]], active_counts[indices[j]]);
            usize actual_size = 0;

            for (usize l : active_indices) {
              if (solutions[l].continuous_active()(indices[i]) && solutions[l].continuous_active()(indices[j])) {
                cov(i, j) += (solutions[l].continuous_values()(indices[i]) - mean(i)) *
                             (solutions[l].continuous_values()(indices[j]) - mean(j));
                actual_size++;
                if (actual_size >= selection_size) {
                  break;
                }
              }
            }
            if (actual_size > 0) {
              // this is fine - that just means that i != j and i and j never appear together so the covariance should
              // be 0
              cov(i, j) /= static_cast<CType>(actual_size);
              cov(j, i) = cov(i, j);
            }
          }
        }
      }
    }
  } else {
    usize selection_size = selection_percentile * active_indices.size();
    for (usize i = 0; i < selection_size; i++) {
      Vec<CType> v = solutions[active_indices[i]].continuous_values()(indices) - mean;
      cov.noalias() += v * v.transpose();
    }
    cov /= static_cast<CType>(selection_size);
  }
  return cov;
};

// Performs an inplace cholesky decomposition. If the decomposition fails, jitter is added to the diagonal to increase
// the rank until finally the univariate diagonal is used.
template <typename Derived>
inline void cholesky_inplace(Eigen::MatrixBase<Derived>& out, const usize num_tries = 1) {
  using S = typename Derived::Scalar;

#ifdef DEBUG
  Mat<S> cov = out;
#endif

  Eigen::LLT<Mat<S>, Eigen::Lower> cholesky_decomposition(out);
  usize tries = 1;
  S jitter_added = 0.0;
  S epsilon = 1e-10;
  while (cholesky_decomposition.info() != Eigen::Success && tries++ < num_tries) {
    // add epsilon to the diagonal to increase the rank
    out.diagonal().array() += epsilon;
    jitter_added += epsilon;
    epsilon *= 10.0;
    cholesky_decomposition = Eigen::LLT<Mat<S>, Eigen::Lower>(out);
  }

  if (cholesky_decomposition.info() == Eigen::Success) {
    out = cholesky_decomposition.matrixL();

#ifdef DEBUG
    Mat<S> res = out * out.transpose();
    for (isize i = 0; i < cov.rows(); i++) {
      for (isize j = 0; j < cov.cols(); j++) {
        assert(std::abs(res(i, j) - cov(i, j)) < 1e-8);
      }
    }
#endif
  } else {
    // std::println("!!! CHOLESKY FAILED !!!\n{}", log_helper(out, /* escape = */ false, /* indent = */ true));
    // std::println("!!! CHOLESKY FAILED !!!");

    // covariance diagonal without jitter, made positive and sqrt to match the expecation that out * out.T = input
    Vec<S> univariate = (out.diagonal().array() - jitter_added);
    for (isize i = 0; i < univariate.size(); i++) {
      univariate(i) = univariate(i) > 0.0 ? std::sqrt(univariate(i)) : 0.0;
    }
    out.setZero();
    out.diagonal() = univariate;

#ifdef DEBUG
    Mat<S> res = out * out.transpose();
    for (isize i = 0; i < cov.rows(); i++) {
      for (isize j = 0; j < cov.cols(); j++) {
        assert((i == j ? std::abs(res(i, j) - cov(i, j)) : res(i, j)) < 1e-8);
      }
    }
#endif
  }
};

class RvSubsetStateBase {
 public:
  CType distribution_multiplier = 1.0;

  virtual ~RvSubsetStateBase() = default;
};

/// Separate sampling models roughly as per https://ir.cwi.nl/pub/30344/30344.pdf
/// TODO ask anton about API - feels still a bit crude...
class RvSamplingModelBase {
 public:
  virtual std::unique_ptr<RvSubsetStateBase> init(const Subset& subset) const = 0;
  virtual std::unique_ptr<RvSubsetStateBase> inherit(const Subset& subset,
                                                     const Subset& previous_subset,
                                                     const RvSubsetStateBase& previous_state) const = 0;
  virtual void update(const SolutionSetBase& solutions,
                      const std::span<const usize> by_fitness_decreasing,
                      const std::span<const usize> active_counts,
                      bool intron_aware,
                      const Subset& subset,
                      double selection_percentile,
                      RvSubsetStateBase& state) const = 0;
  virtual Vec<CType> sample(Rng& rng, const RvSubsetStateBase& state) const = 0;
  virtual void adapt(const SolutionSetBase& solutions,
                     const std::span<const usize> improved_indices,
                     double oob_ratio,
                     bool max_nis_reached,
                     bool intron_aware,
                     const Subset& subset,
                     RvSubsetStateBase& state) const {};

  virtual bool converged(const RvSubsetStateBase& state) const = 0;

  virtual ~RvSamplingModelBase() = default;
};

class AMaLGaMSubsetState final : public RvSubsetStateBase {
 public:
  bool enable_smoothing = false;
  Vec<CType> mean;
  Mat<CType>
      cov;  // for storing the unscaled covariance, since in the incremental version the scaling happens after the
            // smoothing:
            // https://github.com/renzoscholman/irv-gomea/blob/89c62fcf007f858bd8ab9c2ba7955c7db80df121/src/distribution.cpp#L510
  Mat<CType> L;
};

class AMaLGaMSamplingModel final : public RvSamplingModelBase {
 public:
  AMaLGaMSamplingModel(bool use_mahalanobis_distance_for_sdr =
                           false,  // uses mahalanobis distance instead of max(abs(z)) for computing the SDR
                       CType eta_cov = 1.0,
                       CType std_deviation_ratio_threshold = 1.0,
                       CType distribution_multiplier_decrease = 0.9,
                       CType distribution_multiplier_increase = 1.0 / 0.9,
                       CType min_distribution_multiplier = 1e-10,
                       usize num_cholesky_tries = 1)
      : use_mahalanobis_distance_for_sdr(use_mahalanobis_distance_for_sdr),
        eta_cov(eta_cov),
        std_deviation_ratio_threshold(std_deviation_ratio_threshold),
        distribution_multiplier_decrease(distribution_multiplier_decrease),
        distribution_multiplier_increase(distribution_multiplier_increase),
        min_distribution_multiplier(min_distribution_multiplier),
        num_cholesky_tries(num_cholesky_tries) {
    __goblin_runtime_assert(0.0 < eta_cov && eta_cov <= 1.0);
    __goblin_runtime_assert(std_deviation_ratio_threshold >= 0.0);
    __goblin_runtime_assert(0.0 < distribution_multiplier_decrease && distribution_multiplier_decrease <= 1.0);
    __goblin_runtime_assert(1.0 <= distribution_multiplier_increase);
    __goblin_runtime_assert(min_distribution_multiplier >= 0.0);
    __goblin_runtime_assert(num_cholesky_tries >= 1);
  };

  std::unique_ptr<RvSubsetStateBase> init(const Subset& subset) const override final {
    usize size = subset.continuous.size();
    auto s = std::make_unique<AMaLGaMSubsetState>();
    s->enable_smoothing = false;
    s->distribution_multiplier = 1.0;
    s->mean = Vec<CType>::Zero(size);
    s->cov = Mat<CType>::Zero(size, size);
    return s;
  };

  std::unique_ptr<RvSubsetStateBase> inherit(const Subset& subset,
                                             const Subset& previous_subset,
                                             const RvSubsetStateBase& previous_state) const override final {
    const auto& prev = static_cast<const AMaLGaMSubsetState&>(previous_state);
    const auto& cs = subset.continuous;
    const auto& ps = previous_subset.continuous;

    auto s = std::make_unique<AMaLGaMSubsetState>();
    s->distribution_multiplier = prev.distribution_multiplier;
    s->mean = Vec<CType>::Zero(cs.size());
    s->cov = Mat<CType>::Zero(cs.size(), cs.size());

    s->enable_smoothing = true;
    for (usize i = 0; i < cs.size(); i++) {
      bool matched = false;
      for (usize pi = 0; pi < ps.size(); pi++) {
        if (cs[i] == ps[pi]) {
          matched = true;
          s->mean(i) = prev.mean(pi);

          for (usize j = 0; j <= i; j++) {
            for (usize pj = 0; pj <= pi; pj++) {
              if (cs[j] == ps[pj]) {
                s->cov(i, j) = prev.cov(pi, pj);
                s->cov(j, i) = s->cov(i, j);
                break;
              }
            }
          }
          break;
        }
      }
      if (!matched) {
        // If not all variables are present in the previous state, a non-zero smoothing factor (eta_cov < 1.0) would
        // shrink the distribution
        s->enable_smoothing = false;
      }
    }

    return s;
  };

  void update(const SolutionSetBase& solutions,
              const std::span<const usize> by_fitness_decreasing,
              const std::span<const usize> active_counts,
              bool intron_aware,
              const Subset& subset,
              double selection_percentile,
              RvSubsetStateBase& state) const override final {
    const auto& cs = subset.continuous;
    auto& s = static_cast<AMaLGaMSubsetState&>(state);

    // usize selection_size = selection_percentile * by_fitness_decreasing.size();

    assert(static_cast<usize>(s.mean.size()) == cs.size());
    s.mean = estimate_mean(solutions, by_fitness_decreasing, active_counts, selection_percentile, cs, intron_aware);

    // Change the focus of the search to the best solution
    if (s.distribution_multiplier < 1.0) {
      for (usize i = 0; i < cs.size(); i++) {
        if (solutions[by_fitness_decreasing[0]].continuous_active()(cs[i])) {
          s.mean(i) = solutions[by_fitness_decreasing[0]].continuous_values()(cs[i]);
        }
      }
    }

    Mat<CType> new_cov =
        estimate_cov(solutions, by_fitness_decreasing, active_counts, selection_percentile, s.mean, cs, intron_aware);

    if (!s.enable_smoothing) {
      s.cov = new_cov;
      s.enable_smoothing = true;
    } else {
      s.cov = eta_cov * new_cov + (1.0 - eta_cov) * s.cov;
    }

    s.L = s.cov * s.distribution_multiplier;

    // // if we don't have enough solutions we fall back to the univariate diagonal
    // if (selection_size < static_cast<usize>(cs.size()) + 1) {
    //   s.L.setZero();
    //   s.L.diagonal() = (s.cov * s.distribution_multiplier).diagonal();
    //   for (isize i = 0; i < s.L.diagonal().size(); i++) {
    //     s.L.diagonal()(i) = s.L.diagonal()(i) > 0.0 ? std::sqrt(s.L.diagonal()(i)) : 0.0;
    //   }
    // } else {
    //   cholesky_inplace(s.L);
    // }

    cholesky_inplace(s.L, num_cholesky_tries);
  };

  Vec<CType> sample(Rng& rng, const RvSubsetStateBase& state) const override final {
    thread_local static std::normal_distribution<CType> N(0.0, 1.0);

    const auto& s = static_cast<const AMaLGaMSubsetState&>(state);

    Vec<CType> z(s.mean.size());
    // sample from N(0, I)
    for (isize i = 0; i < z.size(); i++) {
      z(i) = N(rng);
    }
    // scale to N(s.mean, s.cov * s.distribution_multiplier)
    return (s.L.triangularView<Eigen::Lower>() * z) + s.mean;
  };

  void adapt(const SolutionSetBase& solutions,
             const std::span<const usize> improved_indices,
             double oob_ratio,
             bool max_nis_reached,
             bool intron_aware,
             const Subset& subset,
             RvSubsetStateBase& state) const override final {
    auto& s = static_cast<AMaLGaMSubsetState&>(state);

    if (oob_ratio > 0.9) {
      s.distribution_multiplier *= 0.5;
    }

    if (!improved_indices.empty()) {
      if (s.distribution_multiplier < 1.0) {
        s.distribution_multiplier = 1.0;
      }

      CType std_deviation_ratio = compute_SDR(solutions, improved_indices, subset, s.mean, s.L, intron_aware);
      if (std_deviation_ratio > std_deviation_ratio_threshold) {
        s.distribution_multiplier *= distribution_multiplier_increase;
      }
    } else if (max_nis_reached) {
      s.distribution_multiplier *= distribution_multiplier_decrease;
    } else {
      if (s.distribution_multiplier > 1.0) {
        s.distribution_multiplier *= distribution_multiplier_decrease;
      }

      if (s.distribution_multiplier < 1.0) {
        s.distribution_multiplier = 1.0;
      }
    }
  };

  bool converged(const RvSubsetStateBase& state) const override final {
    return state.distribution_multiplier <= min_distribution_multiplier;
  };

 private:
  CType compute_SDR(const SolutionSetBase& solutions,
                    const std::span<const usize> improved_indices,
                    const Subset& subset,
                    const Vec<CType>& mean,
                    const Mat<CType>& L,
                    bool intron_aware) const {
    const auto& s = subset.continuous;

    Vec<CType> avg_params = Vec<CType>::Zero(s.size());
    if (intron_aware) {
      Array<CType> num_active = Array<CType>::Zero(s.size());
      for (usize i : improved_indices) {
        for (usize j = 0; j < s.size(); j++) {
          if (solutions[i].continuous_active()(s[j])) {
            avg_params(j) += solutions[i].continuous_values()(s[j]);
            num_active(j) += 1.0;
          }
        }
      }
      for (usize j = 0; j < s.size(); j++) {
        if (num_active(j) > 0.0) {
          avg_params(j) /= num_active(j);
        }
      }
    } else {
      for (usize i : improved_indices) {
        avg_params += solutions[i].continuous_values()(s);
      }
      avg_params /= static_cast<CType>(improved_indices.size());
    }

    // ? L.triangularView<Eigen::Lower>().solve(avg_params - mean);
    Mat<CType> L_inv = L.completeOrthogonalDecomposition().pseudoInverse();

    avg_params = ((L_inv * avg_params) - mean);

    if (use_mahalanobis_distance_for_sdr) {
      return std::sqrt(avg_params.transpose() * avg_params);  // average std_dev from the mean
    } else {
      return avg_params.array().abs().maxCoeff();  // max std_dev from the mean
    }
  }

  bool use_mahalanobis_distance_for_sdr;
  CType eta_cov;
  CType std_deviation_ratio_threshold;
  CType distribution_multiplier_decrease;
  CType distribution_multiplier_increase;
  CType min_distribution_multiplier;
  usize num_cholesky_tries;
};

struct RvOptions {
  bool enabled = true;
  bool intron_aware = false;
  bool intron_aware_intermediate_updates = false;
  bool intron_aware_mean_estimation = false;
  bool intron_aware_cov_estimation = false;
  bool intron_aware_ams = false;

  double selection_percentile = 0.35;
  double p_accept = 0.05;

  bool init_ams_from_population_mean = true;
  // If randomized, the AMS indices are randomly picked from all active solutions.
  // Otherwise the first `floor(selection_percentile * 0.5)` active solutions are used.
  bool randomize_ams_indices = false;
  bool enable_partial_ams = true;
  bool enable_full_ams = true;
  CType delta_ams = 2.0;
  CType eta_ams = 1.0;

  usize max_nis = 100;
  usize num_forced_improvement_tries = 8;  // 8 is the RV GOMEA default if I did not miscalculate (1.0 / 2^8 < 0.01)

  // In the GBO setting with partial evaluations, numerical errors of partial fitness updates
  // can accumulate and it might be needed to perform full evaluations once in a while
  //
  // In that case, the default number of generations until re-evaluation is `50`
  //
  // Note that in this setting all archive solutions should be re-evaluated in case the target seems to have been
  // reached to ensure that this is not due to numeric errors. This does not happen in this version.
  std::optional<u64> generations_until_full_evaluation = std::nullopt;

  std::optional<std::string> population_logfile = std::nullopt;
  std::optional<std::string> selection_logfile = std::nullopt;
  std::optional<std::string> subset_logfile = std::nullopt;
  std::optional<std::string> sample_logfile = std::nullopt;

  bool check_synched_archives = false;

  void validate() {
    __goblin_runtime_assert(0.0 <= p_accept && p_accept < 1.0);
    __goblin_runtime_assert(0.0 <= selection_percentile && selection_percentile < 1.0);
    __goblin_runtime_assert(0.0 < eta_ams && eta_ams <= 1.0);
    __goblin_runtime_assert(0.0 <= delta_ams);
    __goblin_runtime_assert(max_nis > 0);
  };
};

inline std::vector<usize> sort_by_quality_decreasing(const FitnessBase& fitness,
                                                     const SolutionSetBase& solutions,
                                                     const std::vector<usize>& indices,
                                                     std::optional<usize> objective) {
  std::vector<usize> by_fitness;
  if (objective.has_value()) {
    by_fitness = indices;
    std::sort(by_fitness.begin(), by_fitness.end(), [&solutions, &fitness, &objective](usize lhs, usize rhs) {
      return fitness.cmp(solutions[lhs].quality(), solutions[rhs].quality(), objective) == Ordering::Better;
    });
  } else {
    auto [ranks, fronts] = non_dominated_sorting(
        [&solutions, &fitness, &indices](usize lhs, usize rhs) {
          return fitness.cmp(solutions[indices[lhs]].quality(), solutions[indices[rhs]].quality(), std::nullopt);
        },
        indices.size());

    by_fitness.reserve(indices.size());
    for (auto& front : fronts) {
      // TODO re-order fronts to maximize scattering in parameter space?
      //
      // TODO bug: we don't reduce the size to the selection size...
      for (usize i : front) {
        by_fitness.push_back(indices[i]);
      }
    }
  }

  return by_fitness;
};

class RvState {
 public:
  RvState(RvOptions options, const LinkageModelBase& linkage_model, const RvSamplingModelBase& sampling_model)
      : options(options), linkage_model(linkage_model.clone()), sampling_model(sampling_model) {
    this->options.validate();
  };

  usize perform_generation(Rng& rng,
                           ArchiveBase& archive,
                           InstanceBase& problem,
                           SolutionSetBase& solutions,
                           SolutionSetBase& parents,
                           const std::vector<usize>& solution_clusters,
                           const std::vector<std::vector<usize>>& cluster_solutions) {
    // initialization
    if (generation == 0 || num_clusters != cluster_solutions.size() ||
        static_cast<usize>(num_continuous) != problem.num_continuous() || solution_nis.size() != solutions.size()) {
      linkage_model->init(rng, problem, solutions, VariableSet::Continuous);

      num_clusters = cluster_solutions.size();
      num_continuous = problem.num_continuous();

      solutions_to_evaluate.reserve(solutions.size());
      eval_subsets.resize(solutions.size());

      full.continuous.resize(num_continuous);
      std::iota(full.continuous.begin(), full.continuous.end(), 0);

      improved_indices.resize(num_clusters);

      cluster_active = Array<BType>::Zero(num_clusters);
      mean.resize(num_clusters);
      if (options.init_ams_from_population_mean) {
        for (usize k = 0; k < num_clusters; k++) {
          mean[k] = Vec<CType>::Zero(num_continuous);
          for (usize i : cluster_solutions[k]) {
            mean[k] += solutions[i].continuous_values();
          }
          mean[k] /= static_cast<CType>(cluster_solutions[k].size());
        }
      }
      mean_shift.resize(num_clusters);

      num_oob.resize(num_clusters, 0);
      num_samples.resize(num_clusters, 0);

      no_improvement_stretch.resize(num_clusters, 0);

      solution_improved.resize(solutions.size());
      solution_nis.resize(solutions.size(), 0);

      subsets.resize(num_clusters);
      subset_states.resize(num_clusters);

      // // enable partial ams only if the linkage model is not the full fos - in that case, we stay as close to what
      // // AMaLGaM does as possible...
      // auto ptr = reinterpret_cast<const FullFOS*>(&linkage_model);
      // enable_partial_ams = options.enable_partial_ams_for_full_fos || ptr == nullptr;
    }

    // Re-evaluate the whole population if enabled
    // (for the GBO setting where the partial fitness updates cause numerical drift)
    if (options.generations_until_full_evaluation.has_value() &&
        (generation + 1) % options.generations_until_full_evaluation.value() == 0) {
      solutions_to_evaluate.resize(solutions.size());
      std::iota(solutions_to_evaluate.begin(), solutions_to_evaluate.end(), 0);
      problem.evaluate(rng, solutions, solutions_to_evaluate);
      for (usize i = 0; i < solutions.size(); i++) {
        parents[i] = solutions[i];
      }
    }

    if (options.population_logfile.has_value()) {
      AoSSet p;
      p.add(solutions[0]);
      for (usize i = 0; i < solutions.size(); i++) {
        p[0] = solutions[i];
        debug_log(problem, options.population_logfile.value(), "cluster,", std::format("{},", solution_clusters[i]), p);
      }
    }

    // linkage learning, distribution estimation, ams index assignments per cluster
    auto ams_indices = select_and_learn_linkage(rng, archive, problem, solutions, cluster_solutions);

    if (!(cluster_active.any())) {
      return 0;
    }

    std::fill(solution_improved.begin(), solution_improved.end(), false);

    // shuffle per cluster subset order
    usize max_num_subsets = 0;
    for (usize k = 0; k < num_clusters; k++) {
      max_num_subsets = std::max(max_num_subsets, subsets[k].size());
    }
    Mat<usize> subset_orders(num_clusters, max_num_subsets);
    subset_orders.rowwise() = Vec<usize>::LinSpaced(max_num_subsets, 0, max_num_subsets - 1).transpose();
    for (auto row : subset_orders.rowwise()) {
      std::shuffle(row.begin(), row.end(), rng);
    }

    // variation per solution
    usize evaluations = 0;
    usize subset_idx = 0;
    while (subset_idx < max_num_subsets) {
      evaluations += gom_step(rng, archive, problem, solutions, parents, solution_clusters, cluster_solutions,
                              subset_orders, subset_idx++, ams_indices);
    }

    if (options.enable_full_ams) {
      evaluations += full_ams(rng, archive, problem, solutions, parents, solution_clusters, ams_indices);
    }

    // solution NIS update
    std::vector<bool> any_improved(num_clusters, false);
    for (usize i = 0; i < solutions.size(); i++) {
      if (solution_improved[i]) {
        solution_nis[i] = 0;
        any_improved[solution_clusters[i]] = true;
      } else if (solutions[i].continuous_active().any()) {
        // the nis is only updated for active solutions
        solution_nis[i]++;
      }
    }

    if (options.num_forced_improvement_tries > 0) {
      evaluations += forced_improvements(rng, archive, problem, solutions, parents, solution_clusters, subset_orders,
                                         max_num_subsets);
    }

    // cluster NIS update
    for (usize k = 0; k < num_clusters; k++) {
      if (any_improved[k]) {
        no_improvement_stretch[k] = 0;
      } else {
        bool all_leq_one = true;
        for (const auto& s : subset_states[k]) {
          if (s->distribution_multiplier > 1.0) {
            all_leq_one = false;
            break;
          }
        }
        if (all_leq_one) {
          no_improvement_stretch[k]++;
        }
      }
    }

    generation++;
    return evaluations;
  };

  bool converged() const {
    if (num_clusters > 0) {
      for (usize k = 0; k < num_clusters; k++) {
        for (usize i = 0; i < subsets[k].size(); i++) {
          if (!sampling_model.converged(*subset_states[k][i])) {
            return false;
          }
        }
      }
      return true;
    } else {
      return !options.enabled;
    }
  };

  // private:

  std::vector<std::set<usize>> select_and_learn_linkage(Rng& rng,
                                                        ArchiveBase& archive,
                                                        InstanceBase& problem,
                                                        SolutionSetBase& solutions,
                                                        const std::vector<std::vector<usize>>& cluster_solutions) {
    std::vector<std::vector<usize>> active_indices(num_clusters);
    std::vector<std::set<usize>> ams_indices(num_clusters);  // TODO use an unordered_set?

    // TODO parallel
    for (usize k = 0; k < num_clusters; k++) {
      // sort cluster by fitness
      auto by_fitness = sort_by_quality_decreasing(problem.fitness(), solutions, cluster_solutions[k],
                                                   k < problem.num_objectives() ? std::make_optional(k) : std::nullopt);

      // derive the actual selection sizes from the number of active solutions
      active_indices[k].reserve(cluster_solutions[k].size());
      std::vector<usize> active_counts(num_continuous, 0);
      for (auto i : by_fitness) {
        if (options.intron_aware || options.intron_aware_mean_estimation || options.intron_aware_cov_estimation) {
          bool any_active = false;
          for (isize j = 0; j < num_continuous; j++) {
            if (solutions[i].continuous_active()(j)) {
              active_counts[j]++;
              any_active = true;
            }
          }
          if (any_active) {
            active_indices[k].push_back(i);
          }
        } else {
          active_indices[k].push_back(i);
        }
      }

      bool enough_solutions = static_cast<usize>(options.selection_percentile * active_indices[k].size()) >= 2;
      cluster_active(k) = enough_solutions;

      if (!cluster_active(k)) {
        std::println("CLUSTER {} INACTIVE", k);
        continue;
      }

      if (!options.intron_aware && options.selection_logfile.has_value()) {
        usize selection_size = options.selection_percentile * active_indices[k].size();
        AoSSet p;
        p.add(solutions[0]);
        for (usize i = 0; i < selection_size; i++) {
          p[0] = solutions[active_indices[k][i]];
          debug_log(problem, options.selection_logfile.value(), "cluster,", std::format("{},", k), p);
        }
      }

      Vec<CType> new_mean =
          estimate_mean(solutions, active_indices[k], active_counts, options.selection_percentile, full.continuous,
                        options.intron_aware || options.intron_aware_mean_estimation);

      // initialize the (previous) cluster mean
      if (mean[k].size() != num_continuous) {
        mean[k] = new_mean;
      }

      // updates
      if (mean_shift[k].size() != num_continuous || options.eta_ams >= 1.0) {
        mean_shift[k] = new_mean - mean[k];
      } else {
        mean_shift[k] = options.eta_ams * (new_mean - mean[k]) + (1.0 - options.eta_ams) * mean_shift[k];
      }

      mean[k] = new_mean;

      // update subsets
      if (subsets[k].empty()) {  // init if empty
        Mat<CType> cov =
            estimate_cov(solutions, active_indices[k], active_counts, options.selection_percentile, mean[k],
                         full.continuous, options.intron_aware || options.intron_aware_cov_estimation);
        subsets[k] = linkage_model->subsets(rng, problem, solutions, cluster_solutions[k], cov);

        for (usize i = 0; i < subsets[k].size(); i++) {
          subset_states[k].push_back(sampling_model.init(subsets[k][i]));
        }
      } else if (!linkage_model->is_static()) {  // update only if the fos is not static
        Mat<CType> cov =
            estimate_cov(solutions, active_indices[k], active_counts, options.selection_percentile, mean[k],
                         full.continuous, options.intron_aware || options.intron_aware_cov_estimation);
        FOS new_fos = linkage_model->subsets(rng, problem, solutions, cluster_solutions[k], cov);

        std::vector<std::unique_ptr<RvSubsetStateBase>> new_subset_states;
        new_subset_states.reserve(new_fos.size());

        // the assumption is that the number of subsets can change, so a 1:1 mapping might not be possible in any case
        // - to avoid any issues, we simply inherit the distribution multiplier from the "closest" previous subset in
        // O(n2) instead of the hungarian algorithm in O(n3) // TODO test if that is not a performance regression over
        // doing the hungarian algorithm - but that does not handle changing numbers of subsets...
        for (usize i = 0; i < new_fos.size(); i++) {
          usize best = 0;
          double similarity = -1.0;
          for (usize j = 0; j < subsets[k].size(); j++) {
            double sim = new_fos[i].similarity(subsets[k][j]);
            if (sim > similarity) {
              similarity = sim;
              best = j;
            }
          }

          new_subset_states.push_back(sampling_model.inherit(new_fos[i], subsets[k][best], *subset_states[k][best]));
        }

        subsets[k] = std::move(new_fos);
        subset_states[k] = std::move(new_subset_states);
      }

      for (usize i = 0; i < subsets[k].size(); i++) {
        sampling_model.update(solutions, active_indices[k], active_counts, options.intron_aware, subsets[k][i],
                              options.selection_percentile, *subset_states[k][i]);

        if (options.subset_logfile.has_value()) {
          const auto& s = subsets[k][i].continuous;
          if (auto p = dynamic_cast<const AMaLGaMSubsetState*>(subset_states[k][i].get()); p != nullptr) {
            debug_log(problem, options.subset_logfile.value(), "cluster,subset,mean,mean_shift,L,dmul,kind,",
                      std::format("{},{},{},{},{},{},amalgam,", k, log_helper(s), log_helper(mean[k](s)),
                                  log_helper(mean_shift[k](s)), log_helper(p->L), p->distribution_multiplier));
          }
        }
      }

      {  // assign ams indices
        usize ams_pool_size =
            options.intron_aware || options.intron_aware_ams ? active_indices[k].size() : cluster_solutions[k].size();
        usize num_ams_solutions = 0.5 * options.selection_percentile * ams_pool_size;
        std::vector<usize> perm;
        if (options.randomize_ams_indices) {
          perm = permute(rng, ams_pool_size);
          perm.resize(num_ams_solutions);
        } else {
          perm.resize(num_ams_solutions);
          std::iota(perm.begin(), perm.end(), 0);
        }

        for (auto i : perm) {
          ams_indices[k].insert(active_indices[k][i]);
        }
      }
    }

    return ams_indices;
  };

  std::tuple<bool, bool, bool> should_accept(Rng& rng,
                                             const FitnessBase& fitness,
                                             const ArchiveBase& archive,
                                             const SolutionBase& solution,
                                             const SolutionBase& parent,
                                             std::optional<usize> objective,
                                             bool strict) {
    thread_local static std::uniform_real_distribution<double> p(0.0, 1.0);
    if (objective.has_value()) {
      // improvement in the extreme direction or sideways improvement in another
      // objective
      Ordering o = fitness.cmp(solution.quality(), parent.quality(), objective);
      if (o == Ordering::Better || (fitness.num_objectives() > 1 && fitness.cmp(solution.quality(), parent.quality(),
                                                                                std::nullopt) == Ordering::Better)) {
        return std::make_tuple(true, false,
                               fitness.cmp(solution.quality(), archive.so_solution(objective.value()).quality(),
                                           objective) == Ordering::Better);
      }

      if (!strict && options.p_accept > 0.0 && p(rng) < options.p_accept) {
        return std::make_tuple(true, true, false);
      }
    } else {
      bool is_non_dominated = !archive.dominates(solution, true);
      Ordering o = fitness.cmp(solution.quality(), parent.quality(), std::nullopt);
      if (o == Ordering::Better || is_non_dominated) {
        return std::make_tuple(true, false, is_non_dominated);
      }
    }

    return std::make_tuple(false, false, false);
  };

  template <typename Derived>
  void sample(Rng& rng, const InstanceBase& problem, usize k, usize fos_idx, Eigen::MatrixBase<Derived>&& out) {
    thread_local static std::uniform_real_distribution<CType> U(0.0, 1.0);
    const usize TRIES = 100;

    const auto& s = subsets[k][fos_idx].continuous;

    for (usize i = 0; i < TRIES; i++) {
      num_samples[k]++;

      out = sampling_model.sample(rng, *subset_states[k][fos_idx]);

      // bounds check
      if ((problem.continuous_lower_bounds()(s).array() <= out.array()).all() &&
          (out.array() <= problem.continuous_upper_bounds()(s).array()).all()) {
        return;
      }
      num_oob[k]++;
    }

    // otherwise we sample uniformally in the init bounds
    for (usize j = 0; j < s.size(); j++) {
      out(j) *= U(rng) * (problem.continuous_init_upper_bounds()(s[j]) - problem.continuous_init_lower_bounds()(s[j])) +
                problem.continuous_init_lower_bounds()(s[j]);
    }
  };

  usize gom_step(Rng& rng,
                 ArchiveBase& archive,
                 InstanceBase& problem,
                 SolutionSetBase& solutions,
                 SolutionSetBase& parents,
                 const std::vector<usize>& solution_clusters,
                 const std::vector<std::vector<usize>>& cluster_solutions,
                 const Arr2D<usize>& subset_orders,
                 usize subset_idx,
                 const std::vector<std::set<usize>>& ams_indices) {
    solutions_to_evaluate.clear();

    for (usize k = 0; k < num_clusters; k++) {
      num_oob[k] = 0;
      num_samples[k] = 0;

      improved_indices[k].clear();
    }

    bool do_partial_ams =
        // we do want to do partial ams
        options.enable_partial_ams &&
        // and we have a mean shift
        (generation > 0 || options.init_ams_from_population_mean);

    // TODO parallel?
    for (usize i = 0; i < solutions.size(); i++) {
      auto k = solution_clusters[i];
      if (!cluster_active(k)) {
        continue;
      }

      auto fos_idx = subset_orders(k, subset_idx);

      // some clusters might have more subsets...
      if (fos_idx < subsets[k].size() && subsets[k][fos_idx].continuous.size() > 0) {
        eval_subsets[i] = &subsets[k][fos_idx];
        const auto& s = subsets[k][fos_idx].continuous;

        sample(rng, problem, k, fos_idx, solutions[i].continuous_values()(s));
        if (do_partial_ams && ams_indices[k].contains(i)) {
          Vec<CType> shifted_values;
          CType shift_magnitude = 1.0;
          while (shift_magnitude > 1e-10) {  // 34 times (log2(1/1e-10))
            shifted_values = solutions[i].continuous_values()(s) +
                             shift_magnitude * subset_states[k][fos_idx]->distribution_multiplier * options.delta_ams *
                                 mean_shift[k](s);

            if ((problem.continuous_lower_bounds()(s).array() <= shifted_values.array()).all() &&
                (shifted_values.array() <= problem.continuous_upper_bounds()(s).array()).all()) {
              solutions[i].continuous_values()(s) = shifted_values;
              break;
            }

            shift_magnitude *= 0.5;
          }
        }

        bool evaluation_needed = solutions[i].continuous_active()(s).array().any();
        if (evaluation_needed) {
          solutions_to_evaluate.push_back(i);
        } else {  // the parent needs to be updated even if we don't evaluate so that when a change is rejected in the
                  // future, the parent is up to date
          parents[i] = solutions[i];
        }
      }
    }

    if (solutions_to_evaluate.empty())
      return 0;

    problem.evaluate_partial(rng, solutions, parents, eval_subsets, solutions_to_evaluate);

    if (options.sample_logfile.has_value()) {
      AoSSet gom_solutions;
      for (usize i : solutions_to_evaluate) {
        gom_solutions.add(solutions[i]);
      }
      debug_log(problem, options.sample_logfile.value(), "step,", "gom,", gom_solutions);
    }

    for (usize i : solutions_to_evaluate) {
      auto k = solution_clusters[i];
      std::optional<usize> objective = k < problem.fitness().num_objectives() ? std::make_optional(k) : std::nullopt;

      // acceptance - first see what improved, do the intermediate update
      // and then really update the archive
      auto [accept, accept_randomly, improved] =
          should_accept(rng, problem.fitness(), archive, solutions[i], parents[i], objective, false);
      if (accept || accept_randomly) {
        parents[i] = solutions[i];

        if (!accept_randomly) {
          solution_improved[i] = true;
        }

        if (improved) {
          improved_indices[k].push_back(i);
        }
      } else {
        solutions[i] = parents[i];
        // solutions[i].reject(parents[i], problem.always_inherit_continuous(), *eval_subsets[i]);
      }
    }

    for (usize k = 0; k < num_clusters; k++) {
      // intermediate update
      if (cluster_active(k)) {
        auto fos_idx = subset_orders(k, subset_idx);
        if (fos_idx < subsets[k].size()) {
          sampling_model.adapt(solutions, improved_indices[k],
                               static_cast<double>(num_oob[k]) / static_cast<double>(num_samples[k]),
                               no_improvement_stretch[k] >= options.max_nis,
                               options.intron_aware || options.intron_aware_intermediate_updates, subsets[k][fos_idx],
                               *subset_states[k][fos_idx]);
        }
      }

      // archive update
      for (usize i : improved_indices[k]) {
        archive.update(solutions[i], /* strict = */ true, options.check_synched_archives);
      }
    }

    return solutions_to_evaluate.size();
  };

  usize full_ams(Rng& rng,
                 ArchiveBase& archive,
                 InstanceBase& problem,
                 SolutionSetBase& solutions,
                 SolutionSetBase& parents,
                 const std::vector<usize>& solution_clusters,
                 const std::vector<std::set<usize>>& ams_indices) {
    if (!options.enable_full_ams || (generation == 0 && !options.init_ams_from_population_mean)) {
      return 0;
    }

    solutions_to_evaluate.clear();

    for (usize k = 0; k < num_clusters; k++) {
      if (cluster_active(k)) {
        for (usize i : ams_indices[k]) {
          eval_subsets[i] = &full;

          // Eigen reference, writes back to the solution...
          auto values = solutions[i].continuous_values();

          CType shift_magnitude = 1.0;
          bool in_bounds = false;
          while (shift_magnitude > 1e-10) {
            values = parents[i].continuous_values() + shift_magnitude * options.delta_ams * mean_shift[k];

            if ((problem.continuous_lower_bounds().array() <= values.array()).all() &&
                (values.array() <= problem.continuous_upper_bounds().array()).all()) {
              // always true since the ams indices come from a pre-selection of active only indices...
              // bool evaluation_needed = !options.intron_aware || solutions[i].continuous_active().array().any();
              // if(evaluation_needed){
              //     solutions_to_evaluate.push_back(i);
              // }

              // solutions[i].continuous_values() = values; // not needed since values already is a reference, this
              // just is a self assignment...
              solutions_to_evaluate.push_back(i);
              in_bounds = true;
              break;
            }

            shift_magnitude *= 0.5;
          }
          if (!in_bounds) {
            // std::println("OOB {}, nothing to do.", i);
            values = parents[i].continuous_values();
          }
        }
      }
    }

    if (solutions_to_evaluate.empty()) {
      return 0;
    }

    problem.evaluate_partial(rng, solutions, parents, eval_subsets, solutions_to_evaluate);

    if (options.sample_logfile.has_value()) {
      AoSSet ams_solutions;
      for (usize i : solutions_to_evaluate) {
        ams_solutions.add(solutions[i]);
      }
      debug_log(problem, options.sample_logfile.value(), "step,", "ams,", ams_solutions);
    }

    for (usize i : solutions_to_evaluate) {
      auto k = solution_clusters[i];
      std::optional<usize> objective = k < problem.fitness().num_objectives() ? std::make_optional(k) : std::nullopt;

      auto [accept, accept_randomly, improved] =
          should_accept(rng, problem.fitness(), archive, solutions[i], parents[i], objective, false);

      if (accept) {
        parents[i] = solutions[i];

        if (!accept_randomly) {
          solution_improved[i] = true;
        }

        if (improved) {
          archive.update(solutions[i], /* strict = */ false, options.check_synched_archives);
        }
      } else {
        solutions[i] = parents[i];
        // solutions[i].reject(parents[i], problem.always_inherit_continuous(), *eval_subsets[i]);
      }
    }

    return solutions_to_evaluate.size();
  };

  usize forced_improvements(Rng& rng,
                            ArchiveBase& archive,
                            InstanceBase& problem,
                            SolutionSetBase& solutions,
                            SolutionSetBase& parents,
                            const std::vector<usize>& solution_clusters,
                            Mat<usize>& subset_orders,
                            usize max_num_subsets) {
    std::vector<const SolutionBase*> closest_elites;
    closest_elites.reserve(solution_clusters.size());

    std::vector<usize> solutions_to_improve;
    solutions_to_improve.reserve(solutions.size());
    for (usize i = 0; i < solution_clusters.size(); i++) {
      usize k = solution_clusters[i];
      if (solution_nis[i] > options.max_nis && solutions[i].continuous_active().array().any()) {
        // rv fi only works if there the elite solution actually has active rv values...
        if (k < problem.num_objectives()) {
          const auto* e = &archive.so_solution(k);
          if (problem.fitness().cmp(solutions[i].quality(), e->quality(), k) == Ordering::Worse &&
              e->continuous_active().any()) {
            solutions_to_improve.push_back(i);
            closest_elites.push_back(e);
          }
        } else {
          CType closest_dist = std::numeric_limits<CType>::infinity();
          const SolutionBase* closest = nullptr;
          for (usize j = 0; j < archive.size(); j++) {
            const SolutionBase* e = &archive[j];
            if (e->continuous_active().any()) {
              // TODO norm this distance? If so, how?
              CType dist = problem.fitness().distance(solutions[i].quality(), e->quality(), std::nullopt);
              if (dist < closest_dist) {
                closest_dist = dist;
                closest = e;
              }
            }
          }
          if (!isna(closest_dist) && closest) {
            solutions_to_improve.push_back(i);
            closest_elites.push_back(closest);
          }
        }
      }
    }

    for (auto row : subset_orders.rowwise()) {
      std::shuffle(row.begin(), row.end(), rng);
    }

    std::vector<usize> eval2improve_idx;
    eval2improve_idx.reserve(solutions_to_improve.size());

    usize subset_idx = 0;
    usize evaluations = 0;
    usize tries_left = options.num_forced_improvement_tries;
    CType alpha = 0.5;
    while (tries_left > 0 && !solutions_to_improve.empty()) {
      tries_left--;

      while (subset_idx < max_num_subsets && !solutions_to_improve.empty()) {
        eval2improve_idx.clear();
        solutions_to_evaluate.clear();

        for (usize j = 0; j < solutions_to_improve.size(); j++) {
          auto i = solutions_to_improve[j];
          usize k = solution_clusters[i];

          auto fos_idx = subset_orders(k, subset_idx);

          // due to filtering/max_subset_size, some clusters might have more
          // subsets...
          if (fos_idx < subsets[k].size()) {
            const auto& s = subsets[k][fos_idx].continuous;
            bool active_overlap = false;
            for (usize l : s) {
              if (solutions[i].continuous_active()(l)) {
                active_overlap = true;
                break;
              }
            }
            if (active_overlap) {
              eval_subsets[i] = &subsets[k][fos_idx];

              solutions[i].continuous_values()(s) = alpha * parents[i].continuous_values()(s) +
                                                    (CType(1.0) - alpha) * closest_elites[j]->continuous_values()(s);
              solutions_to_evaluate.push_back(i);
              eval2improve_idx.push_back(j);
            }
          }
        }

        problem.evaluate_partial(rng, solutions, parents, eval_subsets, solutions_to_evaluate);
        evaluations += solutions_to_evaluate.size();

        // accept in a random order and remove the improved solutions from the
        // worklist
        {
          std::vector<usize> perm(solutions_to_evaluate.size());
          std::iota(perm.begin(), perm.end(), 0);
          std::shuffle(perm.begin(), perm.end(), rng);

          solutions_to_evaluate.clear();
          auto& indices_to_remove = solutions_to_evaluate;

          for (usize j : perm) {
            auto i = solutions_to_improve[eval2improve_idx[j]];

            auto k = solution_clusters[i];
            std::optional<usize> objective =
                k < problem.fitness().num_objectives() ? std::make_optional(k) : std::nullopt;

            auto [changed, _, improved] =
                should_accept(rng, problem.fitness(), archive, solutions[i], parents[i], objective, true);

            if (changed) {
              parents[i] = solutions[i];
              indices_to_remove.push_back(eval2improve_idx[j]);

              if (improved) {
                archive.update(solutions[i], /* strict = */ false, options.check_synched_archives);
              }
            } else {
              solutions[i] = parents[i];
              // solutions[i].reject(parents[i], problem.always_inherit_continuous(), *eval_subsets[i]);
            }
          }
          // sort indices in reverse - otherwise removing smaller indices might
          // cause invalid memory accesses when trying to remove a larger index
          // later on...
          std::sort(indices_to_remove.begin(), indices_to_remove.end(), std::greater{});
          for (usize i : indices_to_remove) {
            std::swap(solutions_to_improve[i], solutions_to_improve.back());
            solutions_to_improve.pop_back();

            std::swap(closest_elites[i], closest_elites.back());
            closest_elites.pop_back();
          }
        }

        subset_idx++;
      }

      alpha *= 0.5;
    }
    return evaluations;
  };

  // std::tuple<std::shared_ptr<ArchiveBase>, TerminationStatus>
  // run(InstanceBase &problem, const Budget &budget, std::optional<usize> seed,
  //     std::optional<usize> population_size){
  //         // ims go brr
  //     }

  std::optional<usize> current_generation() const  // override final
  {
    return generation;
  };

  RvOptions options;

 private:
  std::unique_ptr<LinkageModelBase> linkage_model;
  const RvSamplingModelBase& sampling_model;

  usize num_clusters = 0;
  isize num_continuous = 0;
  std::vector<usize> solutions_to_evaluate;
  std::vector<std::vector<usize>> improved_indices;
  std::vector<const Subset*>
      eval_subsets;  // pointers because 1. we want to avoid copies and 2. the view should be nullable
  Subset full;

  Array<BType> cluster_active;
  std::vector<Vec<CType>> mean;        // per cluster mean
  std::vector<Vec<CType>> mean_shift;  // per cluster mean shift

  std::vector<u64> num_oob;
  std::vector<u64> num_samples;

  std::vector<u64> no_improvement_stretch;  // per cluster
  std::vector<bool> solution_improved;
  std::vector<u64> solution_nis;

  std::vector<std::vector<std::unique_ptr<RvSubsetStateBase>>> subset_states;
  std::vector<FOS> subsets;

  u64 generation = 0;
};

};  // namespace goblin

#endif /* _GOBLIN_METHODS_CONTINUOUS_H */


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                       goblin/methods/mixed.h continued                                                       //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef NDEBUG
#define __assert_fitness_invariant(s) check_fitness_invariant(rng, s, std::format("{}:{}", __FILE__, __LINE__))
#define __assert_gom_backup_invariant() check_gom_backups(std::format("{}:{}", __FILE__, __LINE__))
#define __assert_invariants()            \
  __assert_fitness_invariant(solutions); \
  __assert_fitness_invariant(parents);   \
  __assert_fitness_invariant(donors);    \
  __assert_gom_backup_invariant()
#else
#define __assert_fitness_invariant(s)
#define __assert_gom_backup_invariant()
#define __assert_invariants()
#endif

namespace goblin {

inline std::tuple<std::vector<usize>, std::vector<std::vector<usize>>, std::vector<std::vector<usize>>>
create_and_register_clusters(Rng& rng,
                             const ArchiveBase& archive,
                             const FitnessBase& fitness,
                             const SolutionSetBase& solutions,
                             usize num_clusters,
                             usize donor_pool_size,
                             const SolutionSetBase& previous_solutions,
                             std::vector<usize>& previous_clusters) {
  assert(donor_pool_size > 0);
  std::vector<usize> solution_clusters(solutions.size(), 0);
  std::vector<std::vector<usize>> cluster_solutions(num_clusters);
  std::vector<std::vector<usize>> cluster_donors;

  // return single cluster with all solutions
  if (num_clusters < 2 || solutions.size() < num_clusters) {
    cluster_solutions[0].resize(solutions.size());
    std::iota(cluster_solutions[0].begin(), cluster_solutions[0].end(), 0);

    cluster_donors.resize(1);
    cluster_donors[0].resize(solutions.size());
    std::iota(cluster_donors[0].begin(), cluster_donors[0].end(), 0);

    return std::make_tuple(solution_clusters, cluster_solutions, cluster_donors);
  }

  usize num_objectives = fitness.num_objectives();

  // clustering here should also use relative comparisons, not absolute ones

  // 1. compute normalized pair-wise objective space distance within solutions
  //    Since only relative distances are used, the normalization uses the
  //    maximum difference for each objective
  Vec<CType> max_dist = Vec<CType>::Zero(num_objectives);
  Vec<CType> min_dist = Vec<CType>::Constant(num_objectives, std::numeric_limits<CType>::infinity());
  Array<usize> best_idx(num_objectives);
  for (usize o = 0; o < num_objectives; o++) {
    const auto& s = archive.so_solution(o).quality();

    for (usize i = 0; i < solutions.size(); i++) {
      auto dist = fitness.distance(solutions[i].quality(), s, o);
      if (!isna(dist)) {
        if (dist > max_dist(o)) {
          max_dist(o) = dist;
        }
        if (dist < min_dist(o)) {
          min_dist(o) = dist;
          best_idx(o) = i;
        }
      }
    }
  }
  assert(min_dist.array().isFinite().all());
  assert(max_dist.array().isFinite().all());

  Vec<CType> dist_norm = (max_dist - min_dist);
  Mat<CType> distance(solutions.size(), solutions.size());
  for (usize i = 0; i < solutions.size(); i++) {
    for (usize j = 0; j < i; j++) {
      CType d_ij(0.0);
      for (usize o = 0; o < num_objectives; o++) {
        auto dist = fitness.distance(solutions[i].quality(), solutions[j].quality(), o);
        d_ij += std::pow(dist / dist_norm(o), 2);
      }
      // ensure non-finite, i.e. replace infinite/na distances with 0.0
      // -> will never be selected during leader selection and
      //    will be assigned to efffectively random clusters
      distance(i, j) = isna(d_ij) ? CType(0.0) : std::sqrt(d_ij);
      distance(j, i) = distance(i, j);
    }
  }

  // 2. greedy scattered subset selection for leaders
  std::uniform_int_distribution<usize> random_objective(0, num_objectives - 1);
  auto [leaders, remaining] =
      greedy_scattered_subset_selection(distance, solutions.size(), num_clusters, best_idx(random_objective(rng)));

  // Depending on which paper you look at, K-means is used to improve the
  // cluster leader assignments here - but https://ir.cwi.nl/pub/23049/23049D.pdf (Chapter 4.6)
  // argues that that the k-means step does not necessarily help

  // 3. round robin cluster assignments (for clusters in random order, add
  // closest solution)

  for (usize k = 0; k < num_clusters; k++) {
    solution_clusters[leaders[k]] = k;
  }
  std::vector<usize> cluster_perm(num_clusters);
  std::iota(cluster_perm.begin(), cluster_perm.end(), 0);
  while (!remaining.empty()) {
    std::shuffle(cluster_perm.begin(), cluster_perm.end(), rng);

    for (auto k : cluster_perm) {
      usize closest_idx;
      distance(leaders[k], remaining).minCoeff(&closest_idx);

      solution_clusters[remaining[closest_idx]] = k;
      std::swap(remaining[closest_idx], remaining.back());
      remaining.pop_back();

      if (remaining.empty()) {
        break;
      }
    }
  }

  // 4. determine SO clusters and re-order clusters
  Mat<CType> min_dist_to_elite =
      Mat<CType>::Constant(num_clusters, num_objectives, std::numeric_limits<CType>::infinity());
  for (usize i = 0; i < solution_clusters.size(); i++) {
    for (usize o = 0; o < num_objectives; o++) {
      min_dist_to_elite(solution_clusters[i], o) =
          std::min(min_dist_to_elite(solution_clusters[i], o),
                   fitness.distance(archive.so_solution(o).quality(), solutions[i].quality(), o));
    }
  }
  std::vector<usize> remaining_clusters(num_clusters);
  std::iota(remaining_clusters.begin(), remaining_clusters.end(), 0);
  std::iota(cluster_perm.begin(), cluster_perm.end(), 0);  // [k_old] -> k_new
  for (usize o : permute(rng, num_objectives)) {
    // find cluster with best solution w.r.t. the current objective
    // and assign that cluster to the objective
    usize best = 0;
    for (usize i = 0; i < remaining_clusters.size(); i++) {
      if (min_dist_to_elite(remaining_clusters[i], o) < min_dist_to_elite(remaining_clusters[best], o)) {
        best = i;
      }
    }

    cluster_perm[remaining_clusters[best]] = o;
    std::swap(remaining_clusters[best], remaining_clusters.back());
    remaining_clusters.pop_back();
  }

  // 5. (if previous objectives + solution_cluster assignments are passed) -
  // perform cluster registration by minimizing maximum matched cluster distance
  // (= average distance between cluster solutions)
  if (!previous_clusters.empty() && remaining_clusters.size() > 1) {
    // assign remaining clusters to clusters [num_objectives,num_clusters) from
    // previous...

    // TODO this does extra work (we don't care about the SO clusters here
    // anymore)
    Mat<CType> generational_distance = Mat<CType>::Zero(num_clusters, num_clusters);
    for (usize i = 0; i < solutions.size(); i++) {
      for (usize j = 0; j < previous_solutions.size(); j++) {
        CType d_ij(0.0);
        for (usize o = 0; o < num_objectives; o++) {
          auto dist = fitness.distance(solutions[i].quality(), previous_solutions[j].quality(), o);
          d_ij += std::pow(dist / dist_norm(o), 2);
        }

        generational_distance(solution_clusters[i], previous_clusters[j]) += isna(d_ij) ? dist_norm.norm() : d_ij;
      }
    }

    // matching - consider all permutations and pick the one with smallest
    // distance sum
    // find best distances that when applied to new clusters matches old
    // clusters
    std::vector<usize> perm(remaining_clusters.size());
    std::iota(perm.begin(), perm.end(), 0);

    std::vector<usize> best_permutation = perm;
    CType best_cumulative_distance = 0.0;
    for (usize i = 0; i < remaining_clusters.size(); i++) {
      best_cumulative_distance += generational_distance(num_objectives + perm[i], num_objectives + i);
    }
    while (std::next_permutation(perm.begin(), perm.end())) {
      CType dist = 0.0;
      for (usize i = 0; i < remaining.size(); i++) {
        dist += generational_distance(num_objectives + perm[i], num_objectives + i);
      }
      if (dist < best_cumulative_distance) {
        best_cumulative_distance = dist;
        best_permutation = perm;
      }
    };

    for (usize i = 0; i < remaining_clusters.size(); i++) {
      cluster_perm[remaining_clusters[i]] = num_objectives + best_permutation[i];
    }
  } else {
    // if we don't do cluster registration, then we still need to make sure that each cluster is in cluster_perm
    for (usize i : remaining_clusters) {
      cluster_perm[i] = i;
    }
  }

  // 6. apply cluster permutation, i.e. apply permutation to solution_clusters
  for (usize i = 0; i < solution_clusters.size(); i++) {
    solution_clusters[i] = cluster_perm[solution_clusters[i]];
  }

  // fill reverse mapping
  for (usize k = 0; k < num_clusters; k++) {
    cluster_solutions[k].reserve(solutions.size() / num_clusters + 1);
  }
  for (usize i = 0; i < solution_clusters.size(); i++) {
    cluster_solutions[solution_clusters[i]].push_back(i);
  }

  // 7. assign donor indices -> closest donor pool size solutins to cluster
  std::vector<usize> indices(solutions.size());
  std::iota(indices.begin(), indices.end(), 0);

  cluster_donors.resize(num_clusters);
  for (usize i = 0; i < num_clusters; i++) {
    usize k = cluster_perm[i];
    std::sort(indices.begin(), indices.end(),
              [&](const auto& lhs, const auto& rhs) { return distance(leaders[i], lhs) < distance(leaders[i], rhs); });

    cluster_donors[k].insert(cluster_donors[k].end(), indices.begin(), indices.begin() + donor_pool_size);
  }

#ifndef NDEBUG
  for (usize i = 0; i < num_clusters; i++) {
    assert(cluster_donors[i].size() == std::min(donor_pool_size, solutions.size()) &&
           "All clusters are supposed to have the same, non-zero donor pool size.");
  }
#endif

  return std::make_tuple(solution_clusters, cluster_solutions, cluster_donors);
};

struct FosStats {
  std::vector<CType> solution_activation_rate;  // whats the proportion of solutions where initially at least one of the
                                                // variables in the subset is active?
  std::vector<CType>
      variables_activation_rate;  // conditioned on solutions with at least one active variables in the subset, whats
                                  // the proportion of variables in the subset that are active on average?
  std::vector<u64> usage_count;   // how often was the FOS used? (without FI, - should be the population size)
  std::vector<u64> evaluation_count;  // how often was an evaluation needed? (i.e. active parts were modified)
  std::vector<u64> acceptance_count;  // how often was the change accepted? (after evaluation)
  std::vector<CType> cumulative_fitness_difference;  // how big were the accepted improvements?
  std::vector<u64> finite_acceptance_count;  // how many improvements had a finite fitness difference to their parent?
                                             // (inf/nan mess up the average...)
  Mat<CType> similarity;
};

struct PopulationOptions {
  double donor_pool_size_multiplier = 2.0;
  std::optional<usize> max_nis = std::nullopt;
  bool forced_improvements = true;
  bool enable_mixed_forced_improvements = true;
  double target_continuous_to_discrete_balance = 1.0;
  bool sequential_gom = false;  // performs GOM sequentially per solution, incompatible with other mechanisms
  bool strict_elite_acceptance =
      false;  // should the single objective elite solutions accept only strict improvements or also neutral changes?

  double donor_search_proportion = 0.0;  // the fraction of solutions to consider before skipping an evaluation in case
                                         // of all subset variables being identical between the solution and donor
  std::optional<std::string> subset_logfile = std::nullopt;
  u64 generation = 0;
  u64 initial_generations_until_next_fos_log = 5;  // > 0, subset stats are logged every
  u64 fos_log_factor = 2;                          // 1 is linear, 2 is exponential log spacing

  // Coefficient mutation as per https://doi.org/10.1145/3520304.3534036
  double continuous_mutation_probability = 0.0;
  CType continuous_mutation_temperature = 0.1;
  CType continuous_mutation_decay_factor = 0.9;
  std::optional<usize> continuous_mutation_decay_patience = 5;

  bool mutate_before_gradient_step = true;
  usize gradient_step_frequency = 0;
  usize gradient_step_count = 10;
};

template <typename SolutionSet>
class Population {
 public:
  Population(InstanceBase& problem,
             ArchiveBase& global_archive,
             const LinkageModelBase& discrete_model,
             const LinkageModelBase& continuous_model,
             const RvSamplingModelBase& sampling_model,
             usize size,
             usize num_clusters,
             const PopulationOptions& options,
             const RvOptions& rv_options)
      : problem(problem),
        global_archive(global_archive),
        discrete_model(discrete_model.clone()),
        rv_state(rv_options, continuous_model, sampling_model),
        options(options),
        local_archive(global_archive.clone()),
        size(size),
        num_clusters(num_clusters),
        donor_pool_size(std::min(size,
                                 static_cast<usize>(options.donor_pool_size_multiplier * static_cast<double>(size) /
                                                    static_cast<double>(num_clusters)))),
        max_nis(options.max_nis.value_or(1 + static_cast<usize>(std::log10(size)))) {
    __goblin_runtime_assert(!isna(options.target_continuous_to_discrete_balance) &&
                            options.target_continuous_to_discrete_balance >= 0.0);

    __goblin_runtime_assert(!options.continuous_mutation_decay_patience.has_value() ||
                            options.continuous_mutation_decay_patience.value() > 0);
    __goblin_runtime_assert(0.0 <= options.continuous_mutation_probability &&
                            options.continuous_mutation_probability <= 1.0);
  };

  void restart() { solutions.clear(); };

  template <typename T>
  u64 perform_generation(Rng& rng, T should_terminate, bool reevaluate_solutions) {
    u64 evaluations = 0;
    bool is_discrete = problem.num_discrete() > 0;
    bool is_continuous = problem.num_continuous() > 0;
    __goblin_runtime_assert(is_discrete || is_continuous);

    // ======= initialization (if necessary) =======
    if (solutions.empty()) {
      evaluations += initialize(rng);
      reevaluate_solutions = false;  // solutions are freshly evaluated, so no need to do it twice

      if (should_terminate(evaluations)) {
        return evaluations;
      }
    }

    {  // ======= clustering =======
      bool perform_cluster_registration = is_continuous && rv_state.options.enabled;
      // the local archive is used since the elites from that should be in this
      // population
      if (!perform_cluster_registration) {
        solution_clusters.clear();
      }
      std::tie(solution_clusters, cluster_solutions, cluster_donors) = create_and_register_clusters(
          rng, *local_archive, problem.fitness(), solutions, num_clusters, donor_pool_size, donors, solution_clusters);
    }

    if (reevaluate_solutions) {
      solutions_to_evaluate.clear();
      solutions_to_evaluate.resize(solutions.size());
      std::iota(solutions_to_evaluate.begin(), solutions_to_evaluate.end(), 0);

      problem.evaluate(rng, solutions, solutions_to_evaluate);
      // ensure the parents also have the updated fitness...
      for (usize i = 0; i < solutions.size(); i++) {
        parents[i].assign_quality(solutions[i].quality());
      }

      evaluations += solutions_to_evaluate.size();
    }

    // after this, donors == parents == solutions holds
    for (usize i = 0; i < size; i++) {
      donors[i] = solutions[i];

      // if acceptance for elites is strict, find the current elite solution for each single-objective cluster
      if (options.strict_elite_acceptance) {
        usize k = solution_clusters[i];

        if (k < problem.num_objectives() &&
            problem.fitness().cmp(solutions[i].quality(), solutions[so_elite_idx[k]].quality(), std::nullopt) ==
                Ordering::Better) {
          so_elite_idx[k] = i;
        }
      }
    }
    __assert_gom_backup_invariant();

    // ======= state/model updates =======

    usize max_discrete_subset_count = 0;
    if (is_discrete) {
      max_discrete_subset_count = learn_discrete_linkage(rng);

      solution_changed.clear();
      solution_changed.resize(size, false);

      // and fix the subset order for each solution
      // (colwise, each column is contiguous and one step)
      if (static_cast<usize>(subset_orders.cols()) < max_discrete_subset_count || !discrete_model->is_static()) {
        subset_orders.resize(size, max_discrete_subset_count);

        // for(auto row: subset_orders.rowwise()){
        //     std::iota(row.begin(), row.begin() + max_discrete_subset_count, 0);
        // }
        for (usize i = 0; i < max_discrete_subset_count; i++) {
          subset_orders.col(i).array() = i;
        }
      }

      for (auto row : subset_orders.rowwise()) {
        std::shuffle(row.begin(), row.begin() + max_discrete_subset_count, rng);
      }
    }

    // ======= housekeeping before variation =======
    // update local archive, prepare changed flags

    local_archive->adapt();
    local_archive->reset_change_count();

    // ======= variation/evaluation/selection =======
    __assert_invariants();

    if (options.sequential_gom) {
      solutions_to_evaluate.resize(1);
      for (usize i = 0; i < size; i++) {
        usize k = solution_clusters[i];
        auto objective = k < problem.num_objectives() ? std::make_optional(k) : std::nullopt;

        perm.resize(cluster_donors[k].size());
        std::iota(perm.begin(), perm.end(), 0);

        usize max_donor_search_iterations = std::min(options.donor_search_proportion, 1.0) * perm.size();

        for (usize subset_idx = 0; subset_idx < max_discrete_subset_count; subset_idx++) {
          if (subset_idx < cluster_FOS[k].size()) {
            subsets[i] = &cluster_FOS[k][subset_orders(i, subset_idx)];

            usize perm_idx = 0;
            bool evaluation_needed, anything_changed;
            do {
              std::swap(perm[perm_idx], perm[std::uniform_int_distribution<usize>(perm_idx, perm.size() - 1)(rng)]);
              usize donor_idx = cluster_donors[k][perm[perm_idx++]];
              if (i == donor_idx) {
                continue;
              }

              std::tie(evaluation_needed, anything_changed) =
                  problem.inherit_discrete(solutions[i], donors[donor_idx], *subsets[i]);

              if (evaluation_needed) {
                solutions_to_evaluate[0] = i;

                problem.evaluate_partial(rng, solutions, parents, subsets, solutions_to_evaluate);

                if (accept_and_update_archive(i, objective,
                                              /* strict */ false)) {
                  parents[i] = solutions[i];
                  solution_changed[i] = true;
                } else {
                  solutions[i] = parents[i];
                }
              } else if (anything_changed) {
                parents[i] = solutions[i];
              }

            } while (!evaluation_needed && perm_idx < max_donor_search_iterations);
          }
        }
      }
    } else {
      usize subset_idx = 0;
      std::uniform_real_distribution<double> U(0.0, 1.0);
      bool can_do_discrete_step = is_discrete && subset_idx < max_discrete_subset_count;
      do {
        // std::println("LOOP");
        bool do_discrete_step;
        if (!is_continuous || !rv_state.options.enabled) {
          do_discrete_step = can_do_discrete_step;
        } else if (!can_do_discrete_step) {
          do_discrete_step = false;
        } else {
          double actual_evaluation_balance =
              discrete_evaluations > 0.0
                  ? continuous_evaluations / discrete_evaluations
                  // since no discrete evaluations were performed we either flip a coin or force a first discrete step
                  // if a first continuous step was already done
                  : (continuous_evaluations > 0.0 ? 2.0 : 1.0) * options.target_continuous_to_discrete_balance;
          // maps the fraction to a percentage such that 0.5 is the target balance,
          // less means too many continuous evaluations, more means too many discrete evaluations
          double p_discrete = 0.5 * actual_evaluation_balance / options.target_continuous_to_discrete_balance;

          do_discrete_step = U(rng) < p_discrete;

          // std::println("p(RV) = {} ({}/{})", 1.0 - p_discrete, continuous_evaluations, discrete_evaluations);
        }

        u64 evals = 0;
        // we first do the continuous step - it might not do anything (not enough active variables or already
        // converged), so we still want to be able to do a discrete step instead
        if (is_continuous && rv_state.options.enabled && !do_discrete_step && !rv_state.converged()) {
          // std::println("RV STEP");
          // RV-GOMEA uses the elite in the population (~= local archive) for forced improvements + adaptive variance
          // scalling (AVS) evals = rv_state.perform_generation(rng, global_archive, problem, solutions, parents,
          // solution_clusters, cluster_solutions);
          evals = rv_state.perform_generation(rng, *local_archive, problem, solutions, parents, solution_clusters,
                                              cluster_solutions);
          __assert_invariants();
          if (evals < 1) {
            // std::println("RV STEP SKIPPED !!!!");
          }
          evaluations += evals;
          continuous_evaluations += evals;
        }

        if (do_discrete_step || (can_do_discrete_step && evals == 0)) {
          // std::println("GP STEP");
          evals = discrete_gom_step(rng, subset_idx++);
          __assert_invariants();
          evaluations += evals;
          discrete_evaluations += evals;

          if (evals < 1) {
            // std::println("GP STEP SKIPPED !!!!");
          }
        }

        if (is_continuous && options.continuous_mutation_probability > 0.0) {
          // std::println("MUT STEP");
          evaluations += continuous_mutation_step(rng);
          __assert_invariants();
        }

        if (should_terminate(evaluations).has_value()) {
          return evaluations;
        }

        can_do_discrete_step = is_discrete && subset_idx < max_discrete_subset_count;
      } while (can_do_discrete_step);  // (subset_idx < max_discrete_subset_count);
    }

    // std::println(">>> GOM END - {} evals", evaluations);

    if (!fos_stats.empty()) {
      log_subset_statistics();
    }

    if (is_continuous && options.gradient_step_frequency > 0 &&
        iterations_since_last_gradient_step++ % options.gradient_step_frequency == 0) {
      // std::println("GRADIENT STEP");
      evaluations += gradient_step(rng);
      __assert_invariants();
    }

    if (options.forced_improvements && is_discrete) {
      // std::println("FI STEP");
      evaluations += forced_improvements(rng, should_terminate, max_discrete_subset_count);
      __assert_invariants();

      // update no improvement stretches
      for (usize i = 0; i < size; i++) {
        if (problem.fitness().cmp(solutions[i].quality(), donors[i].quality(), std::nullopt) == Ordering::Better) {
          solution_nis[i] = 0;
        } else {
          solution_nis[i]++;
        }
      }
    }

    // ======= housekeeping after variation =======
    if (local_archive->change_count() > 0) {
      no_improvement_stretch = 0;
    } else {
      no_improvement_stretch++;
    }

    if (options.continuous_mutation_decay_patience.has_value() &&
        options.continuous_mutation_decay_patience.value() > 0 &&
        no_improvement_stretch % options.continuous_mutation_decay_patience.value() == 0) {
      options.continuous_mutation_temperature *= options.continuous_mutation_decay_factor;
    }
    no_evaluations_performed = evaluations == 0;

    generation++;
    // std::println(">>> GEN END - {} evals", evaluations);

    return evaluations;
  };

  bool converged() {
    if (problem.num_objectives() == 1) {
      if (problem.num_continuous() == 0 && all_solutions_identical()) {
        return true;
      }
      // since we only have relative comparisons, this roughly is equal to the usual fitness variance == 0.0 condition
      if (problem.num_discrete() == 0 && (avg_dist_to_local_so_elite() == 0.0 || rv_state.converged())) {
        return true;
      }
    }

    // this condition does not make sense for continuous only problems as it is way more strict than the NIS/FI
    // mechanisms which force convergence if no progress is made
    return problem.num_discrete() > 0 && no_improvement_stretch >= std::log10(solutions.size());  // && no_evaluations_performed;
  };

  bool all_solutions_identical() const {
    CType epsilon = 1e-10;
    const auto& s0 = solutions[0];
    for (usize i = 1; i < solutions.size(); i++) {
      const auto& si = solutions[i];

      // what is active differs or any active variable is different
      for (usize j = 0; j < problem.num_discrete(); j++) {
        if (s0.discrete_active()(j) != si.discrete_active()(j) ||
            (s0.discrete_active()(j) && (s0.discrete_values()(j) != si.discrete_values()(j)))) {
          return false;
        }
      }
      for (usize j = 0; j < problem.num_continuous(); j++) {
        if (s0.continuous_active()(j) != si.continuous_active()(j) ||
            (s0.continuous_active()(j) && std::abs(s0.continuous_values()(j) - si.continuous_values()(j)) > epsilon)) {
          return false;
        }
      }
    }
    return true;
  };

  CType avg_dist_to_global_so_elite() const {
    const auto& e = global_archive.so_solution(0);
    CType avg_dist = 0.0;
    for (usize i = 0; i < solutions.size(); i++) {
      avg_dist += problem.fitness().distance(e.quality(), solutions[i].quality(), 0);
    }
    return avg_dist / static_cast<CType>(solutions.size());
  };

  CType avg_dist_to_local_so_elite() const {
    const auto& e = local_archive->so_solution(0);
    CType avg_dist = 0.0;
    for (usize i = 0; i < solutions.size(); i++) {
      avg_dist += problem.fitness().distance(e.quality(), solutions[i].quality(), 0);
    }
    return avg_dist / static_cast<CType>(solutions.size());
  };

  /// For single objective optimization, this just is a roundabout way
  /// to return the elite to check if an IMS population should stop
  ArchiveBase& archive() const { return *local_archive; };

  const SolutionSetBase& get_solutions() const { return solutions; };

 private:
  void log_subset_statistics() {
    generations_until_next_fos_log =
        options.initial_generations_until_next_fos_log > 0 ? options.initial_generations_until_next_fos_log : 1;
    options.initial_generations_until_next_fos_log *= options.fos_log_factor;

    AoSSet s;
    s.add(local_archive->so_solution(0));  // solution does not matter, but there should only be a single one...
    assert(fos_stats.size() == num_clusters);
    for (usize k = 0; k < num_clusters; k++) {
      const auto& stats = fos_stats[k];
      std::vector<CType> evaluation_rates(cluster_FOS[k].size());
      std::vector<CType> acceptance_rates(cluster_FOS[k].size());
      std::vector<CType> avg_improvements(cluster_FOS[k].size());
      std::stringstream os;
      os << "\"[";
      for (usize i = 0; i < cluster_FOS[k].size(); i++) {
        evaluation_rates[i] = stats.usage_count[i] > 0 ? static_cast<CType>(stats.evaluation_count[i]) /
                                                             static_cast<CType>(stats.usage_count[i])
                                                       : 0.0;
        acceptance_rates[i] = stats.evaluation_count[i] > 0 ? static_cast<CType>(stats.acceptance_count[i]) /
                                                                  static_cast<CType>(stats.evaluation_count[i])
                                                            : 0.0;
        avg_improvements[i] =
            stats.finite_acceptance_count[i] > 0
                ? stats.cumulative_fitness_difference[i] / static_cast<CType>(stats.finite_acceptance_count[i])
                : 0.0;
        if (i > 0) {
          os << ',';
        }
        log_helper(os, cluster_FOS[k][i].discrete, /* escape = */ false);
      }
      os << "]\"";
      debug_log(problem, options.subset_logfile.value(),
                "population_size,cluster,similarity,subsets,usage_count,evaluation_rate,acceptance_rate,avg_"
                "improvement,solution_activation_rate,variables_activation_rate,",
                std::format("{},{},{},{},{},{},{},{},{},{},", size, k, log_helper(stats.similarity), os.str(),
                            log_helper(stats.usage_count), log_helper(evaluation_rates), log_helper(acceptance_rates),
                            log_helper(avg_improvements), log_helper(stats.solution_activation_rate),
                            log_helper(stats.variables_activation_rate)),
                s);
      fos_stats.clear();
    }
  };
  usize learn_discrete_linkage(Rng& rng) {
    // learn per cluster linkage models
    if (cluster_FOS.empty() || !discrete_model->is_static()) {
      cluster_FOS.clear();

      if (options.subset_logfile.has_value() && generation == generations_until_next_fos_log) {
        Mat<CType> sim(0, 0);
        if (auto p = dynamic_cast<LinkageTreeFOS*>(discrete_model.get()); p != nullptr) {
          p->register_similarity_callback([&sim](const auto& s) { sim = s; });
        }

        fos_stats.clear();
        fos_stats.resize(num_clusters);
        for (usize k = 0; k < num_clusters; k++) {
          cluster_FOS.push_back(discrete_model->subsets(rng, problem, solutions, cluster_solutions[k], std::nullopt));

          fos_stats[k].similarity = sim;
          fos_stats[k].usage_count.resize(cluster_FOS[k].size(), 0);
          fos_stats[k].evaluation_count.resize(cluster_FOS[k].size(), 0);
          fos_stats[k].acceptance_count.resize(cluster_FOS[k].size(), 0);
          fos_stats[k].cumulative_fitness_difference.resize(cluster_FOS[k].size(), 0.0);
          fos_stats[k].finite_acceptance_count.resize(cluster_FOS[k].size(), 0);
          fos_stats[k].solution_activation_rate.resize(cluster_FOS[k].size(), 0.0);
          fos_stats[k].variables_activation_rate.resize(cluster_FOS[k].size(), 0.0);
          for (usize i = 0; i < cluster_solutions[k].size(); i++) {
            const RefS<Array<BType>> s_a = solutions[cluster_solutions[k][i]].discrete_active();
            for (usize fos_idx = 0; fos_idx < cluster_FOS[k].size(); fos_idx++) {
              Array<BType> subset_active = s_a(cluster_FOS[k][fos_idx].discrete);
              if (subset_active.any()) {
                fos_stats[k].variables_activation_rate[fos_idx] +=
                    subset_active.template cast<CType>().sum() /
                    static_cast<CType>(cluster_FOS[k][fos_idx].discrete.size());
                fos_stats[k].solution_activation_rate[fos_idx] += 1.0;
              }
            }
          }
          for (usize fos_idx = 0; fos_idx < cluster_FOS[k].size(); fos_idx++) {
            fos_stats[k].variables_activation_rate[fos_idx] /= fos_stats[k].solution_activation_rate[fos_idx];
            fos_stats[k].solution_activation_rate[fos_idx] /= static_cast<CType>(cluster_solutions[k].size());
          }
        }

        if (auto p = dynamic_cast<LinkageTreeFOS*>(discrete_model.get()); p != nullptr) {
          p->unregister_similarity_callback();
        }
      } else {
        for (usize k = 0; k < num_clusters; k++) {
          cluster_FOS.push_back(discrete_model->subsets(rng, problem, solutions, cluster_solutions[k], std::nullopt));
        }
      }
    }

    usize max_discrete_subset_count = 0;
    for (usize k = 0; k < num_clusters; k++) {
      max_discrete_subset_count = std::max(max_discrete_subset_count, cluster_FOS[k].size());
    }
    return max_discrete_subset_count;
  };

  void check_gom_backups(std::string_view info) {
    assert(solutions.size() == parents.size());
    for (usize i = 0; i < solutions.size(); i++) {
      std::string s_discrete = log_helper(solutions[i].discrete_values(), /* escape = */ false, /* indent = */ false),
                  s_dactive = log_helper(solutions[i].discrete_active(), /* escape = */ false, /* indent = */ false),
                  s_continuous =
                      log_helper(solutions[i].continuous_values(), /* escape = */ false, /* indent = */ false),
                  s_cactive = log_helper(solutions[i].continuous_active(), /* escape = */ false, /* indent = */ false),
                  p_discrete = log_helper(parents[i].discrete_values(), /* escape = */ false, /* indent = */ false),
                  p_dactive = log_helper(parents[i].discrete_active(), /* escape = */ false, /* indent = */ false),
                  p_continuous = log_helper(parents[i].continuous_values(), /* escape = */ false, /* indent = */ false),
                  p_cactive = log_helper(parents[i].continuous_active(), /* escape = */ false, /* indent = */ false),
                  s_quality = problem.fitness().format(solutions[i].quality()),
                  p_quality = problem.fitness().format(parents[i].quality());

      // we expect an exact match - after all, the values should have been copied over without a re-evaluation...
      bool discrete_ok = s_discrete == p_discrete;
      bool dactive_ok = s_dactive == p_dactive;
      bool continuous_ok = s_continuous == p_continuous;
      bool cactive_ok = s_cactive == p_cactive;
      bool quality_ok = s_quality == p_quality;
      if (!(discrete_ok && dactive_ok && continuous_ok && cactive_ok && quality_ok)) {
        std::println("{}", info);
        std::println("GOM backup/parent does not match solution {}: ", i);
        std::println("Discrete: ({})", discrete_ok);
        std::println("  Solution: {}", s_discrete);
        std::println("  Parent:   {}", p_discrete);
        std::println("Discrete Active: ({})", dactive_ok);
        std::println("  Solution: {}", s_dactive);
        std::println("  Parent:   {}", p_dactive);
        std::println("Continuous: ({})", continuous_ok);
        std::println("  Solution: {}", s_continuous);
        std::println("  Parent:   {}", p_continuous);
        std::println("Continuous Active: ({})", cactive_ok);
        std::println("  Solution: {}", s_cactive);
        std::println("  Parent:   {}", p_cactive);
        std::println("  Quality: ({})", quality_ok);
        std::println("  Solution: {}", s_quality);
        std::println("  Parent:   {}", p_quality);
        std::abort();
      }
    }
  };

  // checks that the fitness matches the solution
  void check_fitness_invariant(Rng& rng, SolutionSet& set, std::string_view info) {
    SolutionSet copy;
    for (usize i = 0; i < set.size(); i++) {
      copy.add(set[i]);
    }
    std::vector<usize> indices(copy.size());
    std::iota(indices.begin(), indices.end(), 0);
    problem.evaluate(rng, copy, indices);

    for (auto i : indices) {
      auto expected = copy[i].template quality_as<MOQuality>(), actual = set[i].template quality_as<MOQuality>();
      bool definitely_different =
          (expected.objectives.array().isFinite() != actual.objectives.array().isFinite()).any();
      if (expected.objectives.array().isFinite().all()) {
        definitely_different |= ((expected.objectives - actual.objectives).array().abs() >= 1e-6).any();
      }
      definitely_different |=
          isna(expected.constraint_value) != isna(actual.constraint_value) ||
          (!isna(expected.constraint_value) && std::abs(expected.constraint_value - actual.constraint_value) >= 1e-6);
      if (definitely_different) {
        std::println("{}", info);
        std::println("Fitness invariant violated at index {}: ", i);
        std::println("Expected: {} / '{}'", problem.format_solution(copy[i]), problem.fitness().format(expected));
        std::println("Actual:   {} / '{}'", problem.format_solution(set[i]), problem.fitness().format(actual));
        std::abort();
      }
    }
  };

  // Returns whether a solution should be accepted or not. The parameter `strict` determines if random walks in neutral
  // fitness landscape are allowed or not.
  bool accept_and_update_archive(usize idx, std::optional<usize> objective, bool strict) {
    Ordering o = problem.fitness().cmp(solutions[idx].quality(), parents[idx].quality(), objective);

    if (o == Ordering::Worse) {
      return false;
    }

    bool non_dominated = local_archive->update(solutions[idx], strict);

    if (options.strict_elite_acceptance && objective.has_value() && idx == so_elite_idx[objective.value()]) {
      return o == Ordering::Better;
    }

    // if strict: we want clear improvements,
    // i.e. better in SO or at least non-dominated in MO
    return !strict || o == Ordering::Better || (!objective.has_value() && non_dominated);
  };

  u64 initialize(Rng& rng) {
    // reset the local archive
    local_archive->clear();
    local_archive->unsync_all();
    local_archive->sync_with(global_archive);

    // populate solution, donor and backup buffers
    solutions_to_evaluate.resize(size);
    std::iota(solutions_to_evaluate.begin(), solutions_to_evaluate.end(), 0);

    solutions.reserve(size);
    problem.add_random(rng, solutions, size);
    assert(solutions.size() == size);
    problem.evaluate(rng, solutions, solutions_to_evaluate);

    donors.clear();
    donors.reserve(size);
    parents.clear();
    parents.reserve(size);
    assert(donors.size() == 0);
    assert(parents.size() == 0);
    for (usize i = 0; i < size; i++) {
      donors.add(solutions[i]);
      parents.add(solutions[i]);
      local_archive->update(solutions[i], true);
    }

    // other state
    solution_nis.clear();
    solution_nis.resize(size, 0);

    so_elite_idx.resize(num_clusters);

    no_improvement_stretch = 0;
    no_evaluations_performed = false;
    iterations_since_last_gradient_step = 0;

    solutions_to_improve.reserve(size);
    subsets.resize(size);
    perm.reserve(size);

    // This callback is needed to support learning the linkage
    // normalization matrix from https://arxiv.org/pdf/1904.02050
    // and to tell the linkage model about how many variables there are in case that was not set beforehand
    discrete_model->init(rng, problem, solutions, VariableSet::Discrete);

    discrete_evaluations = 0.0;
    continuous_evaluations = 0.0;

    generation = 0;
    generations_until_next_fos_log = 0;

    return solutions_to_evaluate.size();
  };

  u64 discrete_gom_step(Rng& rng, usize subset_idx) {
    std::vector<usize> donor_pool;
    {
      usize max_donor_pool_size = cluster_donors[0].size();
      assert(max_donor_pool_size > 0);
      for (usize k = 1; k < num_clusters; k++) {
        assert(cluster_donors[k].size() > 0);
        max_donor_pool_size = std::max(max_donor_pool_size, cluster_donors[k].size());
      }
      donor_pool.resize(max_donor_pool_size);
      std::iota(donor_pool.begin(), donor_pool.end(), 0);
    }

    bool record_fos_stats = !fos_stats.empty();

    solutions_to_evaluate.clear();

    // TODO parallel?
    for (usize i = 0; i < solutions.size(); i++) {
      auto k = solution_clusters[i];
      auto fos_idx = subset_orders(i, subset_idx);

      // due to filtering/max_subset_size, some clusters might have more
      // subsets...
      if (fos_idx < cluster_FOS[k].size()) {
        if (record_fos_stats) {
          assert(fos_idx < fos_stats[k].usage_count.size());
          fos_stats[k].usage_count[fos_idx]++;
        }
        subsets[i] = &cluster_FOS[k][fos_idx];
        assert(subsets[i]->discrete.size() > 0);

        // the library does donor search, so this also is added behind a flag to allow fair comparisons to the
        // reference version...
        usize max_donor_search_iterations = std::min(options.donor_search_proportion, 1.0) * cluster_donors[k].size();
        usize donor_idx, donor_pool_idx = 0;

        bool evaluation_needed, anything_changed;
        do {
          // do a partial Fisher-Yates shuffle
          std::swap(donor_pool[donor_pool_idx],
                    donor_pool[std::uniform_int_distribution<usize>(donor_pool_idx, donor_pool.size() - 1)(rng)]);

          donor_idx = donor_pool[donor_pool_idx++];
          if (donor_idx >= cluster_donors[k].size() || i == cluster_donors[k][donor_idx]) {
            continue;
          }

          std::tie(evaluation_needed, anything_changed) =
              problem.inherit_discrete(solutions[i], donors[cluster_donors[k][donor_idx]], *subsets[i]);

          if (evaluation_needed) {  // parent will be updated during acceptance
            solutions_to_evaluate.push_back(i);

            if (record_fos_stats) {
              fos_stats[k].evaluation_count[fos_idx]++;
            }
          } else if (anything_changed) {  // no acceptance, parent has to be updated now
            parents[i] = solutions[i];
          }
        } while (!evaluation_needed && donor_pool_idx < max_donor_search_iterations);
      }
    }

    if (solutions_to_evaluate.empty())
      return 0;

    problem.evaluate_partial(rng, solutions, parents, subsets, solutions_to_evaluate);

    // acceptance happens after one step for all solutions rather than the
    // default of all steps for one solution after the other
    std::shuffle(solutions_to_evaluate.begin(), solutions_to_evaluate.end(), rng);
    for (usize i : solutions_to_evaluate) {
      auto k = solution_clusters[i];
      std::optional<usize> objective = k < problem.fitness().num_objectives() ? std::make_optional(k) : std::nullopt;

      if (!accept_and_update_archive(i, objective,
                                     /* strict */ false)) {
        solutions[i] = parents[i];

      } else {
        if (record_fos_stats) {
          auto fos_idx = subset_orders(i, subset_idx);

          assert(fos_idx < cluster_FOS[k].size());
          fos_stats[k].acceptance_count[fos_idx]++;
          CType dist = problem.fitness().distance(solutions[i].quality(), parents[i].quality(), objective);
          if (!isna(dist)) {
            fos_stats[k].finite_acceptance_count[fos_idx]++;
            fos_stats[k].cumulative_fitness_difference[fos_idx] += dist;
          }
        }
        solution_changed[i] = true;
        parents[i] = solutions[i];
      }
    }

    return solutions_to_evaluate.size();
  }

  // TODO so far this only does discrete FI, but it should be mixed as in the other version...
  template <typename T>
  u64 forced_improvements(Rng& rng, T& should_terminate, usize max_discrete_subset_count) {
    u64 evaluations = 0;

    solutions_to_improve.clear();
    for (usize i = 0; i < size; i++) {
      if (!solution_changed[i] || solution_nis[i] > max_nis) {
        solutions_to_improve.push_back(i);

        // ensure we use a different order
        std::shuffle(subset_orders.row(i).begin(), subset_orders.row(i).end(), rng);
      }
    }

    std::vector<usize> eval2improve_idx;
    eval2improve_idx.reserve(solutions_to_improve.size());

    std::uniform_real_distribution<double> U(0.0, 1.0);
    // RV must be enabled and there need to be continuous values that aren't already inherited...
    bool enable_rv_steps =
        options.enable_mixed_forced_improvements && rv_state.options.enabled && problem.num_continuous() > 0;
    CType alpha = 0.5;
    Subset rv_full;
    if (enable_rv_steps) {
      for (usize i = 0; i < problem.num_continuous(); i++) {
        rv_full.continuous.push_back(i);
      }
    }

    usize subset_idx = 0;
    usize rv_fi_tries = 0;
    while (
        // not all solutions improved
        !solutions_to_improve.empty() &&
        (
            // discrete FI not done
            subset_idx < max_discrete_subset_count ||
            // continuous FI enabled and not done
            (enable_rv_steps && rv_fi_tries < rv_state.options.num_forced_improvement_tries))) {
      eval2improve_idx.clear();
      solutions_to_evaluate.clear();

      double p_rv = enable_rv_steps
                        ? (static_cast<double>(rv_state.options.num_forced_improvement_tries - rv_fi_tries) /
                           static_cast<double>(max_discrete_subset_count - subset_idx +
                                               rv_state.options.num_forced_improvement_tries - rv_fi_tries))
                        : 0.0;

      bool is_rv_step = U(rng) < p_rv;

      // TODO parallel?
      for (usize j = 0; j < solutions_to_improve.size(); j++) {
        auto i = solutions_to_improve[j];
        auto k = solution_clusters[i];

        // clang-format off
        const auto& donor = k < problem.fitness().num_objectives()
            ? global_archive.so_solution(k)
            : global_archive.random_solution(rng);
        // clang-format on

        if (is_rv_step) {
          // there must be overlap in the active constants...
          if ((solutions[i].continuous_active() && donor.continuous_active()).any()) {
            subsets[i] = &rv_full;

            for (usize l = 0; l < problem.num_continuous(); l++) {
              if (solutions[i].continuous_active()(l) && donor.continuous_active()(l)) {
                solutions[i].continuous_values()(l) =
                    alpha * parents[i].continuous_values()(l) + (CType(1.0) - alpha) * donor.continuous_values()(l);
              }
            }
            solutions_to_evaluate.push_back(i);
            eval2improve_idx.push_back(j);
          }
        } else {
          auto fos_idx = subset_orders(i, subset_idx);

          // due to filtering/max_subset_size, some clusters might have more
          // subsets...
          if (fos_idx < cluster_FOS[k].size()) {
            subsets[i] = &cluster_FOS[k][fos_idx];

            auto [evaluation_needed, anything_changed] = problem.inherit_discrete(solutions[i], donor, *subsets[i]);
            if (evaluation_needed) {  // parent will be updated during acceptance
              eval2improve_idx.push_back(j);
              solutions_to_evaluate.push_back(i);
            } else if (anything_changed) {  // no acceptance, so we need to update the parent
              parents[i] = solutions[i];
            }
          }
        }
      }

      problem.evaluate_partial(rng, solutions, parents, subsets, solutions_to_evaluate);
      evaluations += solutions_to_evaluate.size();

      // accept in a random order and remove the improved solutions from the
      // worklist
      {
        perm.resize(solutions_to_evaluate.size());
        std::iota(perm.begin(), perm.end(), 0);
        std::shuffle(perm.begin(), perm.end(), rng);

        solutions_to_evaluate.clear();
        auto& indices_to_remove = solutions_to_evaluate;
        for (usize j : perm) {
          auto i = solutions_to_improve[eval2improve_idx[j]];

          auto k = solution_clusters[i];
          std::optional<usize> objective =
              k < problem.fitness().num_objectives() ? std::make_optional(k) : std::nullopt;

          if (!accept_and_update_archive(i, objective,
                                         /* strict */ true)) {
            solutions[i] = parents[i];
          } else {
            solution_changed[i] = true;
            parents[i] = solutions[i];

            // this solution can stop forced improvements
            indices_to_remove.push_back(eval2improve_idx[j]);
          }
        }
        // sort indices in reverse - otherwise removing smaller indices might
        // cause invalid memory accesses when trying to remove a larger index
        // later on...
        std::sort(indices_to_remove.begin(), indices_to_remove.end(), std::greater{});
        for (usize i : indices_to_remove) {
          std::swap(solutions_to_improve[i], solutions_to_improve.back());
          solutions_to_improve.pop_back();
        }
      }

      if (is_rv_step) {
        alpha *= 0.5;
        rv_fi_tries++;
      } else {
        subset_idx++;
      }

      if (should_terminate(evaluations).has_value()) {
        return evaluations;
      }
    }

    // replace any remaining solutions with non-dominated solutions from the
    // archive

    // TODO parallel
    for (usize i : solutions_to_improve) {
      auto k = solution_clusters[i];
      solutions[i] =
          k < problem.fitness().num_objectives() ? global_archive.so_solution(k) : global_archive.random_solution(rng);
      parents[i] = solutions[i];
    }

    return evaluations;
  };

  template <typename S>
  void mutate_continuous(Rng& rng, S& solution, bool& any_active_changed, Subset& changed) const {
    thread_local static std::uniform_real_distribution<double> U(0.0, 1.0);
    thread_local static std::normal_distribution<double> N(0.0, 1.0);

    any_active_changed = false;
    if (options.continuous_mutation_probability > 0.0) {
      if (changed.continuous.empty()) {
        changed.continuous.reserve(solution.num_continuous());

        for (usize i = 0; i < solution.num_continuous(); i++) {
          if (U(rng) < options.continuous_mutation_probability) {
            solution.continuous_values()(i) +=
                N(rng) * options.continuous_mutation_temperature * solution.continuous_values()(i);
            changed.continuous.push_back(i);
            any_active_changed |= solution.continuous_active()(i);
          }
        }
      } else {
        for (usize i : changed.continuous) {
          if (U(rng) < options.continuous_mutation_probability) {
            solution.continuous_values()(i) +=
                N(rng) * options.continuous_mutation_temperature * solution.continuous_values()(i);
            any_active_changed |= solution.continuous_active()(i);
          }
        }
      }
    }
  }

  u64 continuous_mutation_step(Rng& rng) {
    solutions_to_evaluate.clear();
    assert(options.continuous_mutation_probability > 0.0);

    std::vector<Subset> _subsets(solutions.size());

    // TODO parallel?
    for (usize i = 0; i < solutions.size(); i++) {
      bool evaluation_needed;
      mutate_continuous(rng, solutions[i], evaluation_needed, _subsets[i]);
      subsets[i] = &_subsets[i];

      if (evaluation_needed) {
        solutions_to_evaluate.push_back(i);
      }
    }

    if (solutions_to_evaluate.empty())
      return 0;

    problem.evaluate_partial(rng, solutions, parents, subsets, solutions_to_evaluate);

    std::shuffle(solutions_to_evaluate.begin(), solutions_to_evaluate.end(), rng);
    for (usize i : solutions_to_evaluate) {
      auto k = solution_clusters[i];
      std::optional<usize> objective = k < problem.fitness().num_objectives() ? std::make_optional(k) : std::nullopt;

      if (!accept_and_update_archive(i, objective,
                                     /* strict */ false)) {
        solutions[i] = parents[i];
      } else {
        parents[i] = solutions[i];
      }
    }

    return solutions_to_evaluate.size();
  }

  u64 gradient_step(Rng& rng) {
    solutions_to_evaluate.clear();

    // the mutation before doing a gradient step is decoupled from the other mutation operator
    // - i.e. it should be possible to enable randomization before the gradient optimization without having to also
    // enable the other mutation operator. To make this work, the continuous_mutation_probability is temporarily
    // overwritten
    auto backup_mutation_probability = options.continuous_mutation_probability;
    options.continuous_mutation_probability = 1.0;

    // TODO parallel?
    for (usize i = 0; i < solutions.size(); i++) {
      bool evaluation_needed;
      if (options.mutate_before_gradient_step) {
        Subset _;
        mutate_continuous(rng, solutions[i], evaluation_needed, _);
      } else {
        evaluation_needed = solutions[i].continuous_active().any();
      }

      if (evaluation_needed) {
        solutions_to_evaluate.push_back(i);
      }
    }

    options.continuous_mutation_probability = backup_mutation_probability;

    // If no solution has active continuous values, there is nothing more to do
    if (solutions_to_evaluate.empty())
      return 0;

    auto [changed_indices, evaluations] =
        problem.gradient_steps(rng, solutions, parents, solutions_to_evaluate, options.gradient_step_count);

    // acceptance is still needed since the gradient step isn't guaranteed to be an improvement - e.g. too large
    // steps can be regressions
    for (usize i : changed_indices) {
      auto k = solution_clusters[i];
      std::optional<usize> objective = k < problem.fitness().num_objectives() ? std::make_optional(k) : std::nullopt;

      if (!accept_and_update_archive(i, objective,
                                     /* strict */ false)) {
        solutions[i] = parents[i];
      } else {
        // solution_changed[i] = true;
        parents[i] = solutions[i];
      }
    }

    return evaluations;
  }

  // state requiring initialization

  InstanceBase& problem;
  ArchiveBase& global_archive;
  std::unique_ptr<LinkageModelBase> discrete_model;
  RvState rv_state;
  PopulationOptions options;
  std::unique_ptr<ArchiveBase> local_archive;

  // options

  usize size;
  usize num_clusters;
  usize donor_pool_size;
  usize max_nis;

  // state that is required across generations (absolutely needs to be stored)

  double discrete_evaluations = 0.0;
  double continuous_evaluations = 0.0;
  usize no_improvement_stretch;
  bool no_evaluations_performed;
  usize iterations_since_last_gradient_step;

  std::vector<usize> so_elite_idx;

  SolutionSet donors;  // previous population
  SolutionSet solutions;

  std::vector<usize> solution_nis;
  std::vector<usize> solution_clusters;               // solution -> cluster, previous values
                                                      // required for cluster registration
  std::vector<std::vector<usize>> cluster_solutions;  // cluster -> solutions

  // buffers that are re-used where possible

  SolutionSet parents;                             // gom backups
  std::vector<FOS> cluster_FOS;                    // cluster -> FOS
  std::vector<std::vector<usize>> cluster_donors;  // cluster -> donors
  std::vector<bool> solution_changed;
  std::vector<usize> solutions_to_evaluate;
  std::vector<usize> solutions_to_improve;
  std::vector<usize> perm;
  Mat<usize> subset_orders;            // per solution subset permutations
  std::vector<const Subset*> subsets;  // pointers because 1. we want to avoid copies and 2. the view
                                       // should be nullable
  std::vector<FosStats> fos_stats;

  u64 generation;
  u64 generations_until_next_fos_log;
};

class MixedGOMEA : public MethodBase {
 public:
  MixedGOMEA(PopulationOptions population_options = PopulationOptions(),
             RvOptions rv_options = RvOptions(),
             IMSOptions ims_options = IMSOptions(),
             std::shared_ptr<LinkageModelBase> discrete_model = std::make_shared<LinkageTreeFOS>(),
             std::shared_ptr<LinkageModelBase> continuous_model = std::make_shared<FullFOS>(),
             std::shared_ptr<RvSamplingModelBase> sampling_model = std::make_shared<AMaLGaMSamplingModel>(),
             std::string repr = "aos")
      : population_options(population_options),
        rv_options(rv_options),
        ims_options(ims_options),
        ims_runner(std::nullopt),
        discrete_model(discrete_model),
        continuous_model(continuous_model),
        sampling_model(sampling_model),
        repr(repr) {};

  std::tuple<std::shared_ptr<ArchiveBase>, TerminationStatus> run(
      InstanceBase& problem,
      const Budget& budget,
      std::optional<u64> seed = std::nullopt,

      std::optional<usize> population_size = std::nullopt) override {
    // MAYBE add a check to ensure the desired population can hold the solution
    // type, but how? One option that DOES NOT WORK is to 1. get a solution of
    // the actual type with problem.add_random, then convert to the desired set
    // and back - if that works, everything is good. The problem is that all
    // solutionsets convert a `const SolutionSetBase&` to the internal storage
    // type, so if there is more information it is already lost earlier... => Is
    // there even a way to check before evaluation? If not, just ignore this and
    // let the first evaluation fail due to a type mismatch...
    if (repr == "soa_cols") {
      ims_runner = IMS<Population<SoASet<Eigen::ColMajor>>>(
          std::bind_front(&MixedGOMEA::create_population<SoASet<Eigen::ColMajor>>, this), ims_options);
      return std::get<IMS<Population<SoASet<Eigen::ColMajor>>>>(*ims_runner)
          .run(problem, budget, seed, population_size);
    } else if (repr == "soa_rows") {
      ims_runner = IMS<Population<SoASet<Eigen::RowMajor>>>(
          std::bind_front(&MixedGOMEA::create_population<SoASet<Eigen::RowMajor>>, this), ims_options);
      return std::get<IMS<Population<SoASet<Eigen::RowMajor>>>>(*ims_runner)
          .run(problem, budget, seed, population_size);
    } else {
      ims_runner = IMS<Population<AoSSet>>(std::bind_front(&MixedGOMEA::create_population<AoSSet>, this), ims_options);
      return std::get<IMS<Population<AoSSet>>>(*ims_runner).run(problem, budget, seed, population_size);
    }
  };

  std::optional<u64> current_generation() const override {
    if (!ims_runner.has_value()) {
      return std::nullopt;
    }
    return std::visit([](const auto& r) { return r.current_generation(); }, ims_runner.value());
  };

  std::optional<std::tuple<usize, u64>> current_population() const override {
    if (!ims_runner.has_value()) {
      return std::nullopt;
    }
    return std::visit([](const auto& r) { return r.current_population(); }, ims_runner.value());
  };

 private:
  template <typename SolutionSet>
  Population<SolutionSet> create_population(InstanceBase& problem,
                                            ArchiveBase& global_archive,
                                            usize size,
                                            usize num_clusters) {
    return Population<SolutionSet>(problem, global_archive, *discrete_model, *continuous_model, *sampling_model, size,
                                   num_clusters, population_options, rv_options);
  };

  PopulationOptions population_options;
  RvOptions rv_options;
  IMSOptions ims_options;
  std::optional<std::variant<IMS<Population<AoSSet>>,
                             IMS<Population<SoASet<Eigen::ColMajor>>>,
                             IMS<Population<SoASet<Eigen::RowMajor>>>>>
      ims_runner;
  std::shared_ptr<LinkageModelBase> discrete_model;
  std::shared_ptr<LinkageModelBase> continuous_model;
  std::shared_ptr<RvSamplingModelBase> sampling_model;
  std::string repr;
};
};  // namespace goblin

#endif /* _GOBLIN_MIXED_GOMEA_H */
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                       goblin/methods/classic/common.h included by goblin.h                                   //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef _GOBLIN_CLASSIC_COMMON_H
#define _GOBLIN_CLASSIC_COMMON_H

#include <iterator>


namespace goblin {
namespace classic {
class SelectionStrategyBase {
 public:
  virtual std::vector<usize> select(Rng& rng,
                                    const FitnessBase& fitness,
                                    const SolutionSetBase& solutions,
                                    usize target_size) const = 0;

  virtual ~SelectionStrategyBase() = default;
};

class TournamentSelection : public SelectionStrategyBase {
  usize tournament_size;
  bool with_replacement;

 public:
  TournamentSelection(usize tournament_size = 2, bool with_replacement = false)
      : tournament_size(tournament_size), with_replacement(with_replacement) {
    if (tournament_size < 2) {
      throw std::runtime_error("Tournament size must be greater than 1.");
    }
  };

  std::vector<usize> select(Rng& rng,
                            const FitnessBase& fitness,
                            const SolutionSetBase& solutions,
                            usize target_size) const override final {
    std::vector<usize> selection;
    selection.reserve(target_size);

    std::vector<usize> candidates(tournament_size);

    std::uniform_int_distribution<usize> U(0, solutions.size() - 1);

    std::vector<usize> perm;
    if (!with_replacement) {
      perm = permute(rng, solutions.size());
    }

    usize perm_idx = 0;
    while (selection.size() < target_size) {
      // fill candidate pool with/without replacement
      candidates.clear();
      while (candidates.size() < tournament_size) {
        if (with_replacement) {
          candidates.push_back(U(rng));
        } else {
          if (perm_idx >= perm.size()) {
            std::shuffle(perm.begin(), perm.end(), rng);
            perm_idx = 0;
          }
          candidates.push_back(perm[perm_idx++]);
        }
      }

      // add winner to selection
      std::sort(candidates.begin(), candidates.end(), [&solutions, &fitness](usize lhs, usize rhs) {
        return fitness.cmp(solutions[lhs].quality(), solutions[rhs].quality(), std::nullopt) == Ordering::Better;
      });
      selection.push_back(candidates[0]);
    }

    return selection;
  };
};

class TruncationSelection : public SelectionStrategyBase {
 public:
  std::vector<usize> select(Rng& rng,
                            const FitnessBase& fitness,
                            const SolutionSetBase& solutions,
                            usize target_size) const override final {
    std::vector<usize> selection;

    if (solutions.size() <= target_size) {
      selection.resize(solutions.size());
      std::iota(selection.begin(), selection.end(), 0);
      return selection;
    }

    if (fitness.num_objectives() < 2) {
      // single-objective: sort, then truncate
      selection.resize(solutions.size());
      std::iota(selection.begin(), selection.end(), 0);
      std::sort(selection.begin(), selection.end(), [&solutions, &fitness](usize lhs, usize rhs) {
        return fitness.cmp(solutions[lhs].quality(), solutions[rhs].quality(), std::nullopt) == Ordering::Better;
      });
      selection.resize(target_size);
    } else {
      // multi-objective: non-dominated sorting, then add/truncate fronts until the target size is reached
      auto [ranks, fronts] = non_dominated_sorting(
          [&](usize lhs, usize rhs) {
            return fitness.cmp(solutions[lhs].quality(), solutions[rhs].quality(), std::nullopt);
          },
          solutions.size());

      selection.reserve(target_size);
      for (auto& front : fronts) {
        if (selection.size() >= target_size) {
          break;
        }
        if (selection.size() + front.size() <= target_size) {
          for (usize i : front) {
            selection.push_back(i);
          }
        } else {
          // for the last front, select based on objective space diversity
          std::vector<usize> f2s(front.begin(), front.end());
          auto [selected, _] = greedy_scattered_subset_selection(
              [&](const usize lhs, const usize rhs) {
                return fitness.distance(solutions[f2s[lhs]].quality(), solutions[f2s[lhs]].quality(), std::nullopt);
              },
              /* pool_size = */ f2s.size(),
              /* target_size = */ target_size - selection.size(),
              /* initial = */ std::uniform_int_distribution<usize>(0, f2s.size() - 1)(rng));

          for (usize i : selected) {
            selection.push_back(f2s[i]);
          }
        }
      }
    }

    return selection;
  };
};

class EABase : public MethodBase {
  AoSSet population{};
  u64 generation{};

 protected:
  usize population_size{};

 public:
  EABase() = delete;
  EABase(usize population_size) : population_size(population_size) {};

  virtual u64 step(Rng& rng, InstanceBase& problem, SolutionSetBase& population, ArchiveBase& archive) const = 0;

  void set_population(const SolutionSetBase& population) {
    this->population_size = population.size();
    this->population.clear();
    for (usize i = 0; i < population.size(); i++) {
      this->population.add(population[i]);
    }
  }

  AoSSet get_population() { return population; }

  std::tuple<std::shared_ptr<ArchiveBase>, TerminationStatus> run(
      InstanceBase& problem,
      const Budget& budget,
      std::optional<u64> seed = std::nullopt,
      std::optional<usize> population_size = std::nullopt) override {
    usize n = population_size.value_or(this->population_size);

    generation = 0;
    u64 evaluations = n;
    std::chrono::high_resolution_clock::time_point t_start = std::chrono::high_resolution_clock::now();

    Rng rng = seeded_rng(seed);

    // create & evaluate initial population
    if (population.size() < n) {
      problem.add_random(rng, population, n - population.size());
    }

    std::vector<usize> solutions_to_evaluate;
    solutions_to_evaluate.reserve(n);
    auto worst = problem.archive_fitness().worst();
    for (usize i = 0; i < n; i++) {
      if (problem.fitness().cmp(population[i].quality(), *worst, std::nullopt) != Ordering::Better) {
        solutions_to_evaluate.push_back(i);
      }
    }
    problem.evaluate(rng, population, solutions_to_evaluate);

    auto archive = std::make_shared<UnboundedArchive>(problem.archive_fitness());
    for (usize i = 0; i < n; i++) {
      archive->update(population[i], false);
    }

    auto status = TerminationStatus::Running;
    while (true) {
      // check termination criterion
      auto s = budget.exhausted(generation, evaluations, std::chrono::high_resolution_clock::now() - t_start);
      if (s.has_value()) {
        status = s.value();
        break;
      }
      if (problem.target_reached(*archive)) {
        status = TerminationStatus::TargetReached;
        break;
      }

      evaluations += step(rng, problem, population, *archive);

      generation++;
    }

    return std::make_tuple(archive, status);
  };

  std::optional<u64> current_generation() const override { return generation; };
  std::optional<std::tuple<usize, u64>> current_population() const override {
    return std::make_tuple(population_size, generation);
  };
};

};  // namespace classic
};  // namespace goblin

#endif /* _GOBLIN_CLASSIC_COMMON_H */

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                       goblin/methods/classic/de.h included by goblin.h                                       //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef _GOBLIN_CLASSIC_DE_H
#define _GOBLIN_CLASSIC_DE_H


namespace goblin {
namespace classic {

class DEStrategyBase {
 public:
  virtual std::tuple<Solution, bool> trial_vector(Rng& rng,
                                                  InstanceBase& problem,
                                                  const SolutionSetBase& population,
                                                  const ArchiveBase& archive,
                                                  usize idx,
                                                  Subset& subset) const = 0;
  virtual ~DEStrategyBase() = default;
};

class Rand1Bin : public DEStrategyBase {
  // strategy for selectig first vector
  enum R0Strategy : u8 {
    Random,        // uniformally randomly
    Best,          // best so far
    CurrentToBest  // best - current
  };

  enum ScaleStrategy : u8 {
    Constant,  // F is constant
    Dither,    // F_i ~ N(F, 1) (F shared amongst variables)
    Jitter     // F_i ~ N(F, 1) (F re-sampled per variable)
  };

  double F{};                  // scale factor / differential weight
  double Cr{};                 // crossover probability
  R0Strategy r0_strategy{};    // what is the base vector?
  ScaleStrategy F_strategy{};  // how is F selected?

 public:
  Rand1Bin(double F = 0.8, double Cr = 0.9, std::string base = "best", std::string scale = "dither") : F(F), Cr(Cr) {
    if (F <= 0.0 || F > 2.0) {
      throw std::runtime_error("F must be in (0, 2]");
    }
    if (Cr < 0.0 || Cr > 1.0) {
      throw std::runtime_error("Cr must be in [0, 1]");
    }

    if (base == "best") {
      r0_strategy = R0Strategy::Best;
    } else if (base == "current-to-best") {
      r0_strategy = R0Strategy::CurrentToBest;
    } else if (base == "random") {
      r0_strategy = R0Strategy::Random;
    } else {
      throw std::runtime_error("Unknown strategy for selecting base vector: '" + base + "'");
    }

    if (scale == "constant") {
      F_strategy = ScaleStrategy::Constant;
    } else if (scale == "dither") {
      F_strategy = ScaleStrategy::Dither;
    } else if (scale == "jitter") {
      F_strategy = ScaleStrategy::Jitter;
    } else {
      throw std::runtime_error("Unknown scaling strategy '" + scale + "'");
    }
  };

  std::tuple<Solution, bool> trial_vector(Rng& rng,
                                          InstanceBase& problem,
                                          const SolutionSetBase& population,
                                          const ArchiveBase& archive,
                                          usize idx,
                                          Subset& subset) const override final {
    const usize n = population.size();
    if (n < 4) {
      throw std::runtime_error("DE requires a population size >= 4!");
    }
    const usize D = problem.num_continuous();

    std::uniform_int_distribution<usize> P(0, n - 1);
    std::uniform_real_distribution<double> U(0.0, 1.0);

    usize r0, r1, r2;
    do {
      r1 = P(rng);
    } while (r1 == idx);
    do {
      r2 = P(rng);
    } while (r2 == idx || r1 == r2);
    if (r0_strategy == R0Strategy::Random) {
      do {
        r0 = P(rng);
      } while (r0 == idx || r0 == r1 || r0 == r2);
    }

    Solution o = population[idx];

    usize jrand = std::uniform_int_distribution<usize>(0, D - 1)(rng);

    auto trial = o.continuous_values();
    const auto x_r0 = (r0_strategy == R0Strategy::Random ? population[r0] : archive.so_solution(0)).continuous_values();
    const auto x_r1 = population[r1].continuous_values();
    const auto x_r2 = population[r2].continuous_values();

    const auto lb = problem.continuous_lower_bounds();
    const auto ub = problem.continuous_upper_bounds();

    bool any_active_changed = false;

    std::normal_distribution<CType> dF(F, 1.0);
    CType F_actual = F_strategy == ScaleStrategy::Dither ? dF(rng) : F;
    for (usize j = 0; j < D; j++) {
      if (j == jrand || U(rng) < Cr) {
        if (F_strategy == ScaleStrategy::Jitter) {
          F_actual = dF(rng);
        }
        if (r0_strategy == R0Strategy::CurrentToBest) {
          trial(j) = trial(j) + F_actual * ((x_r0(j) - trial(j)) + (x_r1(j) - x_r2(j)));
        } else {
          trial(j) = x_r0(j) + F_actual * (x_r1(j) - x_r2(j));
        }

        trial(j) = std::clamp(trial(j), lb(j), ub(j));

        subset.continuous.push_back(j);
        any_active_changed |= o.continuous_active()(j);
      }
    }

    return std::make_tuple(o, any_active_changed);
  };
};

class DE : public EABase {
 private:
  // options
  std::shared_ptr<DEStrategyBase> strategy;

  // temporary buffers
  mutable AoSSet offspring;
  mutable std::vector<usize> solutions_to_evaluate;
  mutable std::vector<Subset> subsets;
  mutable std::vector<const Subset*> subset_refs;

 public:
  DE(usize population_size = 100, std::shared_ptr<DEStrategyBase> strategy = std::make_shared<Rand1Bin>())
      : EABase(population_size), strategy(strategy) {
    if (population_size < 4) {
      throw std::runtime_error("DE requires a population size >= 4!");
    }
  };

  u64 step(Rng& rng, InstanceBase& problem, SolutionSetBase& population, ArchiveBase& archive) const override final {
    if (problem.num_continuous() < 1) {
      return 0;
    }
    std::uniform_real_distribution<double> U(0.0, 1.0);
    usize n = population.size();
    if (n < 4) {
      throw std::runtime_error("DE requires a population size >= 4!");
    }

    // housekeeping
    offspring.clear();
    solutions_to_evaluate.clear();
    solutions_to_evaluate.reserve(n);

    // The problem interface supports partial evaluations and provides information
    // on which decision variables are active (or not) in case the problem can have
    // conditionally inactive variables.
    // To take full advantage of such problem settings, partial evaluations are performed
    // and only on offspring solutions where the active variables changed.
    // The crossover masks/subset of changed variables between offspring and parent need
    // to be passed to the evaluation call to support this.
    subsets.resize(n);
    subset_refs.resize(n);
    for (usize i = 0; i < n; i++) {
      subsets[i].continuous.reserve(problem.num_continuous());
      subset_refs[i] = &subsets[i];
    }

    // create trial and mutant vectors
    for (usize i = 0; i < n; i++) {
      subsets[i].continuous.clear();
      auto [trial, any_active_changed] = strategy->trial_vector(rng, problem, population, archive, i, subsets[i]);
      offspring.add(trial);

      if (any_active_changed) {
        solutions_to_evaluate.push_back(i);
      }
    }

    // evaluation
    problem.evaluate_partial(rng, offspring, population, subset_refs, solutions_to_evaluate);

    // acceptance
    for (usize i : solutions_to_evaluate) {
      if (problem.fitness().cmp(offspring[i].quality(), population[i].quality(), std::nullopt) != Ordering::Worse) {
        population[i] = offspring[i];

        archive.update(offspring[i], false);
      }
    }

    return solutions_to_evaluate.size();
  };
};

};  // namespace classic
};  // namespace goblin

#endif /* _GOBLIN_CLASSIC_DE_H */

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                       goblin/methods/classic/es.h included by goblin.h                                       //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef _GOBLIN_ES_H
#define _GOBLIN_ES_H


namespace goblin {
namespace classic {

enum class ESStrategy : u8 {
  /// Single variance for all variables
  SingleVariance,
  /// Separate variance for all variables
  MultipleVariance,
  /// Full covariance matrix
  FullVariance,
  /// Directional variance for one arbitrary direction, single variance in all other directions
  DirectedVariance
};

class ESStrategyParameters : public SolutionExtension<ESStrategyParameters> {
 public:
  ESStrategy strategy{};
  Vec<CType> parameters{};

  std::unique_ptr<SolutionExtensionBase> clone() const override final {
    return std::make_unique<ESStrategyParameters>(*this);
  };
};

class ES : public EABase {
 private:
  // options
  usize num_parents{};
  usize num_offspring{};
  double epsilon{};
  std::optional<double> tau{};
  std::optional<double> tau_i{};
  double beta{};
  bool steady_state{};
  ESStrategy strategy{};

  // temporary buffers
  mutable AoSSet offspring;
  mutable std::vector<usize> solutions_to_evaluate;
  mutable std::vector<usize> parent_pool;

  void init_strategy_params(Rng& rng, SolutionBase& solution) const {
    if (solution.has_extension(ESStrategyParameters::type_key())) {
      return;
    }

    ESStrategyParameters ext;
    ext.strategy = strategy;
    usize num_params;
    switch (strategy) {
      case ESStrategy::SingleVariance:
        num_params = 1;
        break;
      case ESStrategy::MultipleVariance:
        num_params = solution.num_continuous();
        break;
      case ESStrategy::FullVariance:
        num_params = ((solution.num_continuous() + 1) * solution.num_continuous()) / 2;
        break;
      case ESStrategy::DirectedVariance:
        num_params = solution.num_continuous() + 1;
        break;
      default:
        throw std::runtime_error("Strategy initialization not implemented!");
    }
    ext.parameters.resize(num_params);

    std::normal_distribution<CType> N(0.0, 1.0);
    for (usize i = 0; i < num_params; i++) {
      ext.parameters(i) = N(rng);
    }

    solution.get_or_insert_extension(ext);
  };

  Solution recombine_parents(Rng& rng, const SolutionSetBase& parents) const {
    parent_pool.resize(parents.size());
    const usize np = std::min(parent_pool.size(), num_parents);
    std::iota(parent_pool.begin(), parent_pool.end(), 0);
    std::shuffle(parent_pool.begin(), parent_pool.end(), rng);

    Solution combined = parents[parent_pool[0]];
    auto& ext = combined.extension<ESStrategyParameters>();

    for (usize i = 1; i < np; i++) {
      combined.continuous_values() += parents[parent_pool[i]].continuous_values();
      const auto& p_ext = parents[parent_pool[i]].extension<ESStrategyParameters>();
      if (p_ext.strategy != ext.strategy) {
        throw std::runtime_error("Cannot recombine between different strategy types!");
      }
      ext.parameters += p_ext.parameters;
    }
    combined.continuous_values() /= static_cast<CType>(np);
    ext.parameters /= static_cast<CType>(np);

    return combined;
  }

  Solution mutate_and_sample(Rng& rng, const SolutionBase& parent) const {
    const usize l = parent.num_continuous();
    std::normal_distribution<CType> N(0.0, 1.0);

    Solution o = parent;
    auto& ext = o.extension<ESStrategyParameters>();

    Vec<CType> z = Vec<CType>::Zero(parent.num_continuous());

    if (ext.strategy == ESStrategy::SingleVariance) {
      // parameter update
      CType t = tau.value_or(std::sqrt(1.0 / l));
      ext.parameters(0) = std::max(ext.parameters(0) * std::exp(N(rng) * t), epsilon);

      // sample
      for (usize i = 0; i < l; i++) {
        z(i) = N(rng) * ext.parameters(0);
      }
    } else if (ext.strategy == ESStrategy::MultipleVariance) {
      // parameter update
      CType t_shared = N(rng) * tau.value_or(std::sqrt(1.0 / (2.0 * l)));
      CType t_i = tau_i.value_or(std::sqrt(1.0 / (2.0 * std::sqrt(l))));
      for (usize i = 0; i < l; i++) {
        ext.parameters(i) = std::max(ext.parameters(i) * std::exp(t_shared + N(rng) * t_i), epsilon);
      }

      // sample
      for (usize i = 0; i < l; i++) {
        z(i) = N(rng) * ext.parameters(i);
      }
    } else if (ext.strategy == ESStrategy::FullVariance) {
      // parameter update
      CType t_shared = N(rng) * tau.value_or(std::sqrt(1.0 / (2.0 * l)));
      CType t_i = tau_i.value_or(std::sqrt(1.0 / (2.0 * std::sqrt(l))));

      // std dev update
      for (usize i = 0; i < l; i++) {
        ext.parameters(i) = std::max(ext.parameters(i) * std::exp(t_shared + N(rng) * t_i), epsilon);
      }

      // rotation update
      const auto pi = std::numbers::pi_v<CType>;
      for (usize i = l; i < ext.parameters.size(); i++) {
        ext.parameters(i) += N(rng) * beta;

        // keeps the angles meaningful (not really needed since sin/cos are periodic)
        if (ext.parameters(i) > pi) {
          ext.parameters(i) -= 2.0 * pi;
        } else if (ext.parameters(i) < -pi) {
          ext.parameters(i) += 2.0 * pi;
        }
      }

      // sample
      for (usize i = 0; i < l; i++) {
        z(i) = N(rng) * ext.parameters(i);
      }

      // apply rotations
      for (usize p = 1; p <= l - 1; p++) {
        for (usize q = p + 1; q <= l; q++) {
          // formula is 1-indexed in paper, but matrix indices are not
          usize j = ((2 * l - p) * (p + 1)) / 2 + q - 2 * l - 1;

          CType angle_j = ext.parameters(l + j);
          CType s = std::sin(angle_j);
          CType c = std::cos(angle_j);

          CType zp = z(p - 1), zq = z(q - 1);
          z(p - 1) = c * zp - s * zq;
          z(q - 1) = s * zp + c * zq;
        }
      }
    } else if (ext.strategy == ESStrategy::DirectedVariance) {
      // parameter update
      CType t_shared = N(rng) * tau.value_or(std::sqrt(1.0 / (2.0 * l)));
      CType t_i = tau_i.value_or(std::sqrt(1.0 / (2.0 * std::sqrt(l))));

      // std dev update
      for (usize i = 0; i < 2; i++) {
        ext.parameters(i) = std::max(ext.parameters(i) * std::exp(t_shared + N(rng) * t_i), epsilon);
      }

      // rotation update
      const auto pi = std::numbers::pi_v<CType>;
      for (usize i = 2; i < ext.parameters.size(); i++) {
        ext.parameters(i) += N(rng) * beta;

        // keeps the angles meaningful (not really needed since sin/cos are periodic)
        if (ext.parameters(i) > pi) {
          ext.parameters(i) -= 2.0 * pi;
        } else if (ext.parameters(i) < -pi) {
          ext.parameters(i) += 2.0 * pi;
        }
      }

      // sample
      z(0) = N(rng) * ext.parameters(1);
      for (usize i = 1; i < l; i++) {
        z(i) = N(rng) * ext.parameters(0);
      }

      // apply rotations
      for (usize i = 0; i < l - 1; i++) {
        CType angle_j = ext.parameters(2 + i);
        CType s = std::sin(angle_j);
        CType c = std::cos(angle_j);

        CType zp = z(i), zq = z(i + 1);
        z(i) = c * zp - s * zq;
        z(i + 1) = s * zp + c * zq;
      }
    } else {
      throw std::runtime_error("Strategy sampling not implemented!");
    }

    o.continuous_values() += z;

    return o;
  };

 public:
  ES(usize population_size = 8,  // mu
     usize num_parents = 1,      // rho
     usize num_offspring = 50,   // lambda
     // steady state (mu + lambda) vs generational (mu, lambda)
     bool steady_state = true,
     std::string strategy = "single",
     double epsilon = 1e-6,
     std::optional<double> tau = std::nullopt,
     std::optional<double> tau_i = std::nullopt,
     double beta = 0.0873  // ~ 5degrees
     )
      : EABase(population_size),
        num_parents(num_parents),
        num_offspring(num_offspring),
        epsilon(epsilon),
        tau(tau),
        tau_i(tau_i),
        beta(beta),
        steady_state(steady_state) {
    if (population_size < 1) {
      throw std::runtime_error("ES requires a population size >= 1!");
    }
    if (num_parents > population_size) {
      this->num_parents = population_size;
    }
    if (num_offspring < 1) {
      throw std::runtime_error("ES requires at least one offspring per solution!");
    }

    if (strategy == "single") {
      this->strategy = ESStrategy::SingleVariance;
    } else if (strategy == "multiple") {
      this->strategy = ESStrategy::MultipleVariance;
    } else if (strategy == "full") {
      this->strategy = ESStrategy::FullVariance;
    } else if (strategy == "directed") {
      this->strategy = ESStrategy::DirectedVariance;
    } else {
      throw std::runtime_error("Unknown ES strategy: '" + strategy + "'");
    }

    if (epsilon <= 0.0) {
      throw std::runtime_error("Epsilon must be > 0.0!");
    }
  };

  u64 step(Rng& rng, InstanceBase& problem, SolutionSetBase& population, ArchiveBase& archive) const override final {
    if (problem.num_continuous() < 1) {
      return 0;
    }
    std::uniform_real_distribution<double> U(0.0, 1.0);
    usize n = population.size();
    if (n < 1) {
      throw std::runtime_error("ES requires a population size >= 1!");
    }

    // housekeeping
    offspring.clear();
    solutions_to_evaluate.clear();
    solutions_to_evaluate.reserve(num_offspring);

    // initialize the strategy parameters if necessary
    for (usize i = 0; i < n; i++) {
      init_strategy_params(rng, population[i]);
    }

    // variation
    const auto lb = problem.continuous_lower_bounds();
    const auto ub = problem.continuous_upper_bounds();

    for (usize i = 0; i < num_offspring; i++) {
      if (num_parents > 1) {
        Solution parent = recombine_parents(rng, population);
        offspring.add(mutate_and_sample(rng, parent));
      } else {
        offspring.add(mutate_and_sample(rng, population[i % n]));
      }

      // boundary handling by clamping
      auto x = offspring[i].continuous_values();
      x = x.cwiseMax(lb).cwiseMin(ub);

      solutions_to_evaluate.push_back(i);
    }

    // evaluation & archive update
    problem.evaluate(rng, offspring, solutions_to_evaluate);
    for (usize i : solutions_to_evaluate) {
      archive.update(offspring[i], false);
    }

    // (optionally) O = O + P
    if (steady_state) {
      for (usize i = 0; i < population.size(); i++) {
        offspring.add(population[i]);
      }
    }

    // selection
    auto selection = TruncationSelection().select(rng, problem.fitness(), offspring, n);
    population.clear();
    for (usize i : selection) {
      population.add(offspring[i]);
    }

    return solutions_to_evaluate.size();
  };
};

};  // namespace classic
}  // namespace goblin

#endif /* _GOBLIN_ES_H */

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                       goblin/methods/classic/pso.h included by goblin.h                                      //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef _GOBLIN_CLASSIC_PSO_H
#define _GOBLIN_CLASSIC_PSO_H


namespace goblin {
namespace classic {

class PSOTopologyBase {
 public:
  virtual std::vector<usize> neighbours(Rng& rng, const SolutionSetBase& population, usize idx) const = 0;
  virtual ~PSOTopologyBase() = default;
};

class RingTopology : public PSOTopologyBase {
  usize num_neighbours{};

 public:
  /// Considers a population of size N as ring where the ends wrap (nodes 0 and N-1 are neighbours). The number of
  /// neighbours determines the (symmetric) reach of each node around the ring, where `num_neighbours >= floor(N-1) / 2`
  /// corresponds to a fully connected star topology.
  RingTopology(usize num_neighbours = 2) : num_neighbours(num_neighbours) {
    if (num_neighbours < 1) {
      throw std::runtime_error("The neighbourhood must extend to at least 1 neighbour in each direction.");
    }
  };

  std::vector<usize> neighbours(Rng& rng, const SolutionSetBase& population, usize idx) const {
    const usize N = population.size();
    const usize n = std::min(2 * num_neighbours + 1, N);
    std::vector<usize> nbs;
    if (n >= N) {
      nbs.resize(N);
      std::iota(nbs.begin(), nbs.end(), 0);
      return nbs;
    }

    nbs.reserve(n);
    isize i = static_cast<isize>(idx) - num_neighbours;
    while (nbs.size() < n) {
      if (i < 0) {
        i += N;
      } else if (i >= N) {
        i -= N;
      }
      nbs.push_back(i);
      i++;
    }

    return nbs;
  };
};

class PSOState : public SolutionExtension<PSOState> {
  std::unique_ptr<QualityBase> _previous_best_quality{};

 public:
  Vec<CType> velocity{};
  Vec<CType> previous_best{};

  const QualityBase& previous_best_quality() const { return *_previous_best_quality; }
  QualityBase& previous_best_quality() { return *_previous_best_quality; }

  void assign_previous_best_quality(const QualityBase& quality) { _previous_best_quality = quality.clone(); }

  PSOState() = default;
  ~PSOState() = default;
  PSOState(const PSOState& other)
      : _previous_best_quality(other._previous_best_quality->clone()),
        velocity(other.velocity),
        previous_best(other.previous_best) {};
  PSOState(PSOState&& other)
      : _previous_best_quality(std::move(other._previous_best_quality)),
        velocity(std::move(other.velocity)),
        previous_best(std::move(other.previous_best)) {};
  PSOState& operator=(const PSOState& other) {
    if (&other != this) {
      velocity = other.velocity;
      previous_best = other.previous_best;
      _previous_best_quality = other._previous_best_quality->clone();
    }
    return *this;
  }
  PSOState& operator=(PSOState&& other) {
    if (&other != this) {
      velocity = std::move(other.velocity);
      previous_best = std::move(other.previous_best);
      _previous_best_quality = std::move(other._previous_best_quality);
    }
    return *this;
  }

  std::unique_ptr<SolutionExtensionBase> clone() const override final { return std::make_unique<PSOState>(*this); };
};

class PSO : public EABase {
 private:
  // options
  std::shared_ptr<PSOTopologyBase> topology;
  double inertia{};
  double cognitive{};
  double social{};

  // temporary buffers
  mutable std::vector<usize>
      solutions_to_evaluate;  // mutable because the buffer does not contain persistent state, and there is little point
                              // in re-allocating the memory each iteration

 public:
  PSO(usize population_size = 25,
      double inertia = 0.729,
      double cognitive = 1.494,
      double social = 1.494,
      std::shared_ptr<PSOTopologyBase> topology = std::make_shared<RingTopology>())
      : EABase(population_size), topology(topology), inertia(inertia), cognitive(cognitive), social(social) {
    if (population_size < 2) {
      throw std::runtime_error("PSO requires a population size >= 2!");
    }
  };

  u64 step(Rng& rng, InstanceBase& problem, SolutionSetBase& population, ArchiveBase& archive) const override final {
    if (problem.num_continuous() < 1) {
      return 0;
    }
    std::uniform_real_distribution<double> U(0.0, 1.0);
    const usize n = population.size();
    const usize D = problem.num_continuous();
    if (n < 2) {
      throw std::runtime_error("PSO requires a population size >= 2!");
    }

    // housekeeping
    solutions_to_evaluate.clear();
    solutions_to_evaluate.reserve(n);

    // update the previous best value
    for (usize i = 0; i < n; i++) {
      // initialize if not set
      if (!population[i].has_extension(PSOState::type_key())) {
        PSOState ext;
        ext.velocity.resize(D);
        for (usize j = 0; j < D; j++) {
          ext.velocity(j) = 0.0;
        };
        ext.previous_best = population[i].continuous_values();
        ext.assign_previous_best_quality(population[i].quality());
        population[i].get_or_insert_extension(ext);
      } else {
        auto& ext = population[i].extension<PSOState>();
        if (problem.fitness().cmp(population[i].quality(), ext.previous_best_quality(), std::nullopt) !=
            Ordering::Worse) {
          ext.assign_previous_best_quality(population[i].quality());
          ext.previous_best = population[i].continuous_values();
        }
      }
    }

    // update the positions of each particle
    Vec<CType> r0 = Vec<CType>::Zero(D);
    Vec<CType> r1 = Vec<CType>::Zero(D);
    for (usize i = 0; i < n; i++) {
      auto& ext = population[i].extension<PSOState>();

      // social update
      auto neighbours = topology->neighbours(rng, population, i);
      if (neighbours.size() < 1) {
        throw std::runtime_error("Invalid, empty neighbourhood");
      }
      usize best_nb_idx = 0;
      for (usize j = 1; j < neighbours.size(); j++) {
        if (problem.fitness().cmp(population[neighbours[j]].extension<PSOState>().previous_best_quality(),
                                  population[neighbours[best_nb_idx]].extension<PSOState>().previous_best_quality(),
                                  std::nullopt) != Ordering::Worse) {
          best_nb_idx = j;
        }
      }

      // velocity update
      auto x = population[i].continuous_values();
      const auto pb = ext.previous_best;
      const auto gb = population[neighbours[best_nb_idx]].extension<PSOState>().previous_best;

      for (usize j = 0; j < D; j++) {
        r0(j) = U(rng);
        r1(j) = U(rng);
      }

      ext.velocity = inertia * ext.velocity + cognitive * r0.cwiseProduct(pb - x) + social * r1.cwiseProduct(gb - x);

      // position update
      x += ext.velocity;

      // clamp to boundaries
      const auto lb = problem.continuous_lower_bounds();
      const auto ub = problem.continuous_upper_bounds();
      x = x.cwiseMax(lb).cwiseMin(ub);

      solutions_to_evaluate.push_back(i);
    }

    // evaluation (no partial evaluations since the velocity updates all variables at once)
    problem.evaluate(rng, population, solutions_to_evaluate);

    for (usize i : solutions_to_evaluate) {
      archive.update(population[i], false);
    }

    return solutions_to_evaluate.size();
  };
};

};  // namespace classic
};  // namespace goblin

#endif /* _GOBLIN_CLASSIC_PSO_H */

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                       goblin/methods/classic/simple_ga.h included by goblin.h                                //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef _GOBLIN_SIMPLE_GA_H
#define _GOBLIN_SIMPLE_GA_H



namespace goblin {
namespace classic {
/// Strategy used to generate the crossover masks to exchange information between two parents
class DiscreteCrossoverBase {
 public:
  virtual Subset crossover_mask(Rng& rng,
                                InstanceBase& problem,
                                const SolutionBase& donor,
                                SolutionBase& offspring) const {
    throw std::runtime_error("Either implement this method or override the full crossover behaviour!");
  };

  virtual bool crossover(Rng& rng, InstanceBase& problem, const SolutionBase& donor, SolutionBase& offspring) const {
    auto mask = crossover_mask(rng, problem, donor, offspring);

    auto [evaluation_needed, _] = problem.inherit_discrete(offspring, donor, mask);

    return evaluation_needed;
  };

  virtual ~DiscreteCrossoverBase() = default;
};

class CombinedCrossover : public DiscreteCrossoverBase {
  std::vector<std::tuple<std::shared_ptr<DiscreteCrossoverBase>, double>> operators;

 public:
  CombinedCrossover() = delete;
  CombinedCrossover(std::vector<std::tuple<std::shared_ptr<DiscreteCrossoverBase>, double>>&& operators,
                    bool normalize = true)
      : operators(std::move(operators)) {
    if (this->operators.empty()) {
      throw std::runtime_error("At least one operator is required!");
    }

    if (normalize) {
      double norm = 0.0;
      for (auto& [_, probability] : operators) {
        norm += probability;
      }
      for (auto& [_, probability] : operators) {
        probability /= norm;
      }
    }
  };

  bool crossover(Rng& rng,
                 InstanceBase& problem,
                 const SolutionBase& donor,
                 SolutionBase& offspring) const override final {
    bool evaluation_needed = false;
    std::uniform_real_distribution<double> U(0.0, 1.0);
    double r = U(rng);
    for (auto& [op, probability] : operators) {
      if (r < probability) {
        evaluation_needed |= op->crossover(rng, problem, donor, offspring);
        break;
      } else {
        r -= probability;
      }
    }
    return evaluation_needed;
  };
};

class UniformCrossover : public DiscreteCrossoverBase {
  double p_crossover{};

 public:
  UniformCrossover(double p_crossover = 0.5) : p_crossover(p_crossover) {
    if (p_crossover <= 0.0 || 1.0 <= p_crossover) {
      throw std::runtime_error("Crossover probability must be in (0,1) to perform variation, not copying!");
    }
  };

  Subset crossover_mask(Rng& rng,
                        InstanceBase& problem,
                        const SolutionBase& donor,
                        SolutionBase& offspring) const override final {
    Subset mask;
    std::uniform_real_distribution<double> U(0.0, 1.0);
    for (usize i = 0; i < problem.num_discrete(); i++) {
      if (U(rng) < p_crossover) {
        mask.discrete.push_back(i);
      }
    }
    return mask;
  };
};

class NPointCrossover : public DiscreteCrossoverBase {
  usize num_points{};

 public:
  NPointCrossover(usize num_points = 1) : num_points(num_points) {
    if (num_points < 1) {
      throw std::runtime_error("At least one crossover point is needed to perform variation, not copying!");
    }
  };

  Subset crossover_mask(Rng& rng,
                        InstanceBase& problem,
                        const SolutionBase& donor,
                        SolutionBase& offspring) const override final {
    const usize l = problem.num_discrete();
    Subset mask;
    // here the two endpoints are excluded to ensure not all values come from the same parent
    auto points = permute(rng, l - 1);
    if (num_points < l) {
      points.resize(num_points);
    }
    std::sort(points.begin(), points.end());

    // swap the indices between every other set of points
    for (usize i = 0; i < points.size(); i += 2) {
      // + 1 since the first real crossover point is between index 0 and 1, not before index 0
      usize start = points[i] + 1;
      usize end = i + 1 < points.size() ? points[i + 1] + 1 : l;
      for (usize j = start; j < end; j++) {
        mask.discrete.push_back(j);
      }
    }
    return mask;
  };
};

class DiscreteMutationBase {
 public:
  virtual void mutate(Rng& rng, InstanceBase& problem, SolutionBase& offspring) const = 0;
  virtual ~DiscreteMutationBase() = default;
};

class CombinedMutation : public DiscreteMutationBase {
  std::vector<std::tuple<std::shared_ptr<DiscreteMutationBase>, double>> operators;

 public:
  CombinedMutation() = delete;
  CombinedMutation(std::vector<std::tuple<std::shared_ptr<DiscreteMutationBase>, double>>&& operators,
                   bool normalize = true)
      : operators(std::move(operators)) {
    if (this->operators.empty()) {
      throw std::runtime_error("At least one operator is required!");
    }

    if (normalize) {
      double norm = 0.0;
      for (auto& [_, probability] : operators) {
        norm += probability;
      }
      for (auto& [_, probability] : operators) {
        probability /= norm;
      }
    }
  };

  void mutate(Rng& rng, InstanceBase& problem, SolutionBase& offspring) const override final {
    std::uniform_real_distribution<double> U(0.0, 1.0);
    double r = U(rng);
    for (auto& [op, probability] : operators) {
      if (r < probability) {
        op->mutate(rng, problem, offspring);
        break;
      } else {
        r -= probability;
      }
    }
  };
};

class RandomMutation : public DiscreteMutationBase {
  std::optional<double> p_mutation;

 public:
  RandomMutation(std::optional<double> p_mutation = std::nullopt) : p_mutation(p_mutation) {
    if (p_mutation.value_or(0.0) >= 1.0) {
      throw std::runtime_error("A mutation rate of 100% performs random search!");
    }
  };

  void mutate(Rng& rng, InstanceBase& problem, SolutionBase& offspring) const override final {
    std::uniform_real_distribution<double> U(0.0, 1.0);

    const usize l = problem.num_discrete();
    const double p_mut = p_mutation.value_or(1.0 / static_cast<double>(l));

    if (p_mut > 0.0) {
      for (usize i = 0; i < l; i++) {
        const usize d_i = problem.discrete_domain_sizes()(i);
        std::uniform_int_distribution<usize> D(0, d_i - 1);
        if (U(rng) < p_mut && d_i > 1) {
          usize v = D(rng);
          if (offspring.discrete_values()(i) == v) {
            v = (v + 1) % d_i;
          }
          offspring.discrete_values()(i) = v;
        }
      }
    }
  }
};

/// A mutation operator that assumes an ordinal relationship for discrete variables
class LocalizedMutation : public DiscreteMutationBase {
  std::optional<double> p_mutation;
  double strength;
  bool wrap;

 public:
  LocalizedMutation(std::optional<double> p_mutation = std::nullopt, double strength = 0.05, bool wrap = false)
      : p_mutation(p_mutation), strength(strength), wrap(wrap) {
    if (p_mutation.value_or(0.0) >= 1.0) {
      throw std::runtime_error("A mutation rate of 100% performs random search!");
    }

    if (strength <= 0.0) {
      throw std::runtime_error("The mutation strength must be positive!");
    }
  };

  void mutate(Rng& rng, InstanceBase& problem, SolutionBase& offspring) const override final {
    std::uniform_real_distribution<double> U(0.0, 1.0);
    std::normal_distribution<double> N(0.0, 1.0);

    const usize l = problem.num_discrete();
    const double p_mut = p_mutation.value_or(1.0 / static_cast<double>(l));

    if (p_mut > 0.0) {
      for (usize i = 0; i < l; i++) {
        const usize d_i = problem.discrete_domain_sizes()(i);
        if (U(rng) < p_mut && d_i > 1) {
          double v = static_cast<double>(offspring.discrete_values()(i));
          v += N(rng) * strength * static_cast<double>(d_i);
          if (wrap) {
            if (v < 0.0) {  // wrap around by adding d_i * ceil(|v| / d_i)
              v += static_cast<double>(d_i) * std::ceil(-v / static_cast<double>(d_i));
              v = std::fmod(v, d_i);
            }
          } else {
            v = std::clamp(v, 0.0, static_cast<double>(d_i - 1));
          }

          offspring.discrete_values()(i) = static_cast<DType>(v);
        }
      }
    }
  }
};

class SimpleGA : public EABase {
 private:
  // options
  std::shared_ptr<SelectionStrategyBase> selection_strategy;
  std::shared_ptr<DiscreteCrossoverBase> crossover_strategy;
  std::shared_ptr<DiscreteMutationBase> mutation_strategy;
  bool steady_state{};

  // temporary buffers
  mutable AoSSet offspring;
  mutable std::vector<usize> solutions_to_evaluate;
  mutable std::vector<Subset> subsets;
  mutable std::vector<const Subset*> subset_refs;

 public:
  SimpleGA(usize population_size = 100,
           std::shared_ptr<DiscreteCrossoverBase> crossover = std::make_shared<UniformCrossover>(),
           std::shared_ptr<DiscreteMutationBase> mutation = std::make_shared<RandomMutation>(),
           bool steady_state =
               true,  // steady_state vs generational: select from P + O or just from O after generating more offspring?
           std::shared_ptr<SelectionStrategyBase> selection = std::make_shared<TournamentSelection>(4))
      : EABase(population_size),
        selection_strategy(selection),
        crossover_strategy(crossover),
        mutation_strategy(mutation),
        steady_state(steady_state) {
    if (!selection) {
      throw std::runtime_error("No selection provided!");
    }
    if (!crossover) {
      throw std::runtime_error("No crossover provided!");
    }
    if (!mutation) {
      throw std::runtime_error("No mutation provided!");
    }
    if (population_size % 2 != 0) {
      throw std::runtime_error("Population size must be even!");
    }
    if (auto p = dynamic_cast<TruncationSelection*>(&*selection_strategy); p != nullptr && !steady_state) {
      // generational: need to select population_size parents -> no selection pressure with truncation selection
      throw std::runtime_error("Truncation selection is not compatible with a generational replacement scheme!");
    }
  };

  void check_changes(const SolutionBase& parent,
                     const SolutionBase& offspring,
                     std::vector<usize>& changed_indices,
                     bool& evaluation_needed) const {
    changed_indices.clear();
    for (usize i = 0; i < parent.num_discrete(); i++) {
      // something changed if the values are different, but we only need to evaluate if at least one variable active in
      // the parent changed or the fitness should still be the same
      if (parent.discrete_values()(i) != offspring.discrete_values()(i)) {
        changed_indices.push_back(i);
        if (parent.discrete_active()(i)) {
          evaluation_needed = true;
        }
      }
    }
  };

  u64 step(Rng& rng, InstanceBase& problem, SolutionSetBase& population, ArchiveBase& archive) const override final {
    if (problem.num_discrete() < 1) {
      return 0;
    }
    std::uniform_real_distribution<double> U(0.0, 1.0);
    const usize n = population.size();
    if (n % 2 != 0) {
      throw std::runtime_error("Population size must be even!");
    }

    // The problem interface supports partial evaluations and provides information
    // on which decision variables are active (or not) in case the problem can have
    // conditionally inactive variables.
    // To take full advantage of such problem settings, partial evaluations are performed
    // and only on offspring solutions where the active variables changed.
    // To support this, the subset of changed variables between offspring and parent need
    // to be passed to the evaluation call.

    // variation
    subsets.resize(n);
    subset_refs.resize(n);
    offspring.clear();
    solutions_to_evaluate.clear();
    solutions_to_evaluate.reserve(n);
    std::vector<usize> parent_indices = permute(rng, n);
    for (usize i = 0; i < n; i++) {
      // copy to offspring
      offspring.add(population[i]);
      bool evaluation_needed = false;

      const auto& donor = population[parent_indices[i]];

      // perform crossover
      if (crossover_strategy) {
        evaluation_needed |= crossover_strategy->crossover(rng, problem, donor, offspring[i]);
      }

      // apply mutation
      if (mutation_strategy) {
        mutation_strategy->mutate(rng, problem, offspring[i]);
      }

      // check what changed to allow for partial evaluations & exploiting introns
      check_changes(population[i], offspring[i], subsets[i].discrete, evaluation_needed);
      if (evaluation_needed) {
        solutions_to_evaluate.push_back(i);
        subset_refs[i] = &subsets[i];
      }
    }

    // evaluation & archive update
    problem.evaluate_partial(rng, offspring, population, subset_refs, solutions_to_evaluate);
    for (usize i : solutions_to_evaluate) {
      archive.update(offspring[i], false);
    }

    // add offspring to selection pool (if steady-state)
    if (steady_state) {
      for (usize i = 0; i < n; i++) {
        offspring.add(population[i]);
      }
    }

    // selection
    auto selection = selection_strategy->select(rng, problem.fitness(), offspring, n);
    population.clear();
    for (auto i : selection) {
      population.add(offspring[i]);
    }

    return solutions_to_evaluate.size();
  };
};

};  // namespace classic
}  // namespace goblin

#endif /* _GOBLIN_SIMPLE_GA_H */

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                       goblin/methods/classic/standard_gp.h included by goblin.h                              //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef _GOBLIN_STANDARD_GP_H
#define _GOBLIN_STANDARD_GP_H


namespace goblin {
namespace classic {

class GPVariationOperatorBase {
 public:
  virtual void apply(Rng& rng,
                     const InstanceBase& problem,
                     const GPContext& ctx,
                     const SolutionSetBase& population,
                     SolutionBase& offspring) const = 0;
  virtual ~GPVariationOperatorBase() = default;
};

class Chained : public GPVariationOperatorBase {
  std::vector<std::tuple<std::shared_ptr<GPVariationOperatorBase>, double>> operators;

 public:
  Chained() = delete;
  Chained(std::vector<std::tuple<std::shared_ptr<GPVariationOperatorBase>, double>>&& operators)
      : operators(std::move(operators)) {
    if (this->operators.empty()) {
      throw std::runtime_error("At least one operator is required!");
    }
  };

  void apply(Rng& rng,
             const InstanceBase& problem,
             const GPContext& ctx,
             const SolutionSetBase& population,
             SolutionBase& offspring) const override final {
    std::uniform_real_distribution<double> U(0.0, 1.0);
    for (auto& [op, probability] : operators) {
      if (U(rng) < probability) {
        op->apply(rng, problem, ctx, population, offspring);
      }
    }
  };
};

class SubtreeCrossover : public GPVariationOperatorBase {
 public:
  void apply(Rng& rng,
             const InstanceBase& problem,
             const GPContext& ctx,
             const SolutionSetBase& population,
             SolutionBase& offspring) const override final {
    // get donor
    const auto& donor = population[std::uniform_int_distribution<usize>(0, population.size() - 1)(rng)];
    auto donor_nodes = ctx.active_nodes(donor);

    // get random node
    auto nodes = ctx.active_nodes(offspring);
    usize node_idx = nodes[std::uniform_int_distribution<usize>(0, nodes.size() - 1)(rng)];

    // replace with random donor subtree
    std::shuffle(donor_nodes.begin(), donor_nodes.end(), rng);
    for (usize i : donor_nodes) {
      // this can fail due to tree shape constraints (max_depth/size)
      if (ctx.copy_tree(donor, i, offspring, node_idx)) {
        break;
      }
    }
  }
};

/// Replaces random subtree with a random subtree
class SubtreeMutation : public GPVariationOperatorBase {
 public:
  void apply(Rng& rng,
             const InstanceBase& problem,
             const GPContext& ctx,
             const SolutionSetBase& population,
             SolutionBase& offspring) const override final {
    // get random node index
    auto nodes = ctx.active_nodes(offspring);
    std::shuffle(nodes.begin(), nodes.end(), rng);

    // get random tree
    AoSSet s;
    problem.add_random(rng, s, 1);

    ctx.copy_tree(s[0], nodes[0], offspring, nodes[0]);
  };
};

/// Randomly mutates constants
class ConstantMutation : public GPVariationOperatorBase {
  double probability{};
  CType temperature{};

 public:
  ConstantMutation(double probability = 0.25, CType temperature = 0.25)
      : probability(probability), temperature(temperature) {};

  void apply(Rng& rng,
             const InstanceBase& problem,
             const GPContext& ctx,
             const SolutionSetBase& population,
             SolutionBase& offspring) const override final {
    std::uniform_real_distribution<double> U(0.0, 1.0);
    std::normal_distribution<CType> N(0.0, 1.0);

    // get random node index
    auto indices = ctx.active_constant_indices(offspring);

    for (usize ci : indices) {
      if (U(rng) < probability) {
        offspring.continuous_values()(ci) += N(rng) * offspring.continuous_values()(ci) * temperature;
      }
    }
  };
};

class StandardGP : public EABase {
 private:
  // options
  std::shared_ptr<SelectionStrategyBase> selection_strategy;
  std::shared_ptr<GPVariationOperatorBase> variation_operator;
  bool steady_state{};

  // temporary buffers
  mutable AoSSet offspring;
  mutable std::vector<usize> solutions_to_evaluate;

 public:
  StandardGP(usize population_size = 32,
             bool steady_state = true,  // steady_state vs generational: select from P + O or just from O after
                                        // generating more offspring?
             std::shared_ptr<GPVariationOperatorBase> variation_operator = std::shared_ptr<GPVariationOperatorBase>(),
             std::shared_ptr<SelectionStrategyBase> selection_strategy = std::make_shared<TournamentSelection>(2))
      : EABase(population_size),
        selection_strategy(selection_strategy),
        variation_operator(
            variation_operator != nullptr
                ? variation_operator
                : std::make_shared<Chained>(std::vector<std::tuple<std::shared_ptr<GPVariationOperatorBase>, double>>{
                      std::make_tuple(std::make_shared<SubtreeCrossover>(), 1.0),
                      std::make_tuple(std::make_shared<SubtreeMutation>(), 0.25),
                      std::make_tuple(std::make_shared<ConstantMutation>(), 0.25)})),
        steady_state(steady_state) {
    if (auto p = dynamic_cast<TruncationSelection*>(&*selection_strategy); p != nullptr && !steady_state) {
      // generational: need to select population_size solutions from population_size offspring -> no selection pressure
      // with truncation selection
      throw std::runtime_error("Truncation selection is not compatible with a generational replacement scheme!");
    }
  };

  u64 step(Rng& rng, InstanceBase& problem, SolutionSetBase& population, ArchiveBase& archive) const override final {
    if (auto p = dynamic_cast<const GPInstanceBase*>(&problem.unwrap()); p != nullptr) {
      const auto& ctx = p->context();
      const usize n = population.size();

      // variation
      offspring.clear();
      solutions_to_evaluate.clear();
      solutions_to_evaluate.reserve(n);
      for (usize i = 0; i < n; i++) {
        offspring.add(population[i]);
        variation_operator->apply(rng, problem, ctx, population, offspring[i]);
        solutions_to_evaluate.push_back(i);
      }

      // evaluation & archive update
      problem.evaluate(rng, offspring, solutions_to_evaluate);
      for (usize i : solutions_to_evaluate) {
        archive.update(offspring[i], false);
      }

      if (steady_state) {
        // steady state selection from P + O
        for (usize i = 0; i < n; i++) {
          offspring.add(population[i]);
        }
      }

      auto selection = selection_strategy->select(rng, problem.fitness(), offspring, n);
      population.clear();
      for (auto i : selection) {
        population.add(offspring[i]);
      }

      return solutions_to_evaluate.size();
    } else {
      throw std::runtime_error("Not a GP problem!");
    }
  };
};

};  // namespace classic
}  // namespace goblin

#endif /* _GOBLIN_STANDARD_GP_H */


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                       goblin/examples/voronoi.h included by goblin.h                                         //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef _GOBLIN_EXAMPLES_VORONOI_H
#define _GOBLIN_EXAMPLES_VORONOI_H



template <typename C>
class KDTree {
 private:
  using Scalar = typename C::Scalar;
  using usize = std::size_t;
  using isize = std::ptrdiff_t;

  struct Node {
    usize idx;
    Node* left;
    Node* right;
  };

  Node* root;
  std::vector<Node> nodes;

  usize best_idx;
  Scalar best_dist;

  Node* build_helper(usize begin, usize end, isize dim, const C& coords) {
    if (end <= begin) {
      return nullptr;
    }

    // find midpoint in current dimension
    usize mid = begin + (end - begin) / 2;
    auto i = nodes.begin();
    std::nth_element(i + begin, i + mid, i + end,
                     [&](const Node& lhs, const Node& rhs) { return coords(dim, lhs.idx) < coords(dim, rhs.idx); });

    // recurse with next dimension
    dim = (dim + 1) % coords.rows();
    nodes[mid].left = build_helper(begin, mid, dim, coords);
    nodes[mid].right = build_helper(mid + 1, end, dim, coords);
    return &nodes[mid];
  };

  template <typename P>
  void closest_helper(Node* node, isize dim, const C& coords, const P& point) {
    if (node == nullptr) {
      return;
    }

    const isize n_dims = coords.rows();

    // Note that (coords(node->idx) - point).square().sum()
    // silently produces incorrect results due to broadcasting fun
    Scalar dist = 0.0;
    for (isize i = 0; i < n_dims; i++) {
      dist += std::pow(coords(i, node->idx) - point(i), 2);
    }

    if (dist < best_dist) {
      best_dist = dist;
      best_idx = node->idx;
    }

    if (best_dist == 0.0) {
      return;
    }

    Scalar dist_x = coords(dim, node->idx) - point(dim);
    dim = (dim + 1) % n_dims;
    closest_helper(dist_x > 0.0 ? node->left : node->right, dim, coords, point);
    if (dist_x * dist_x < best_dist) {
      closest_helper(dist_x > 0.0 ? node->right : node->left, dim, coords, point);
    }
  }

 public:
  // No default copying since the nodes are pointer-based
  KDTree(const KDTree&) = delete;
  KDTree& operator=(const KDTree&) = delete;

  KDTree() = default;
  void build(const C& coords, usize size) {
    if (coords.cols() < size) {
      throw std::runtime_error("Not enough coordinates passed!");
    }
    nodes.resize(size);
    for (usize i = 0; i < nodes.size(); i++) {
      nodes[i].idx = i;
    }
    root = build_helper(0, nodes.size(), /* dim = */ 0, coords);
  };

  template <typename P>
  usize closest(const C& coords, const P& point) {
    if (root == nullptr) {
      throw std::runtime_error("Empty tree!");
    }

    best_idx = 0;
    best_dist = std::numeric_limits<Scalar>::infinity();
    closest_helper(root, /* dim = */ 0, coords, point);
    return best_idx;
  }
};

namespace goblin {
namespace voronoi {
class VoronoiImageReconstruction : public InstanceBase {
  const usize VARS_PER_CELL = 6;
  const usize ENABLED = 0;
  const usize X_COORD = 1;
  const usize Y_COORD = 2;
  const usize COLOR_R = 3;
  const usize COLOR_G = 4;
  const usize COLOR_B = 5;

  const usize NUM_COLOR_VALUES = 256;

 public:
  VoronoiImageReconstruction(
      const Arr2D<DType>& target_image,
      usize width,
      usize height,
      usize min_num_cells = 10,
      usize max_num_cells = 100,
      std::optional<AnyInit> init = std::nullopt,
      bool complexity_objective = false,
      bool track_complexity = false,
      /// Minimum number of cells after which a kd-tree should be used to determine the nearest cell center
      usize kdtree_threshold = 50)
      : _fitness(  // this preference is optimized
            /* num_objectives = */ complexity_objective ? 2 : 1,
            /* minimize = */ true),
        _archive_fitness(  // this one is used for the archive
            /* num_objectives = */ (complexity_objective || track_complexity) ? 2 : 1,
            /* minimize = */ true),
        target_image(target_image.cast<float>()),
        init(from_any_init(init.value_or(std::make_shared<CompleteInit>()))),
        width(width),
        height(height),
        min_num_cells(min_num_cells),
        max_num_cells(max_num_cells),
        kdtree_threshold(kdtree_threshold) {
    const usize num_pixels = target_image.rows();
    if (num_pixels != width * height) {
      throw std::runtime_error(std::format("Image data ({}pixels) does not match withd and height ({} * {} = {})",
                                           num_pixels, width, height, width * height));
    }

    if (min_num_cells > max_num_cells) {
      std::swap(min_num_cells, max_num_cells);
      std::swap(this->min_num_cells, this->max_num_cells);
    }

    if (min_num_cells < 1) {
      throw std::runtime_error("At least one cell is required!");
    }

    if (max_num_cells >= num_pixels) {
      throw std::runtime_error("More voronoi cells than pixels in the image!");
    }

    // set up domain for each variable as [0, num_values)
    _discrete_domain_sizes.resize(max_num_cells * VARS_PER_CELL);
    for (usize i = 0; i < max_num_cells; i++) {
      usize j = i * VARS_PER_CELL;

      // (enabled, X, Y, R, G, B)
      _discrete_domain_sizes[j + ENABLED] = i < min_num_cells ? 1 : 2;
      _discrete_domain_sizes[j + X_COORD] = width;
      _discrete_domain_sizes[j + Y_COORD] = height;
      _discrete_domain_sizes[j + COLOR_R] = NUM_COLOR_VALUES;
      _discrete_domain_sizes[j + COLOR_G] = NUM_COLOR_VALUES;
      _discrete_domain_sizes[j + COLOR_B] = NUM_COLOR_VALUES;
    }
  };

  CRef<Vec<DType>> discrete_domain_sizes() const override final { return _discrete_domain_sizes; };

  CRef<Vec<CType>> continuous_lower_bounds() const override final { return _continuous_lower_bounds; };
  CRef<Vec<CType>> continuous_upper_bounds() const override final { return _continuous_upper_bounds; };

  CRef<Vec<CType>> continuous_init_lower_bounds() const override final { return _continuous_init_lower_bounds; };
  CRef<Vec<CType>> continuous_init_upper_bounds() const override final { return _continuous_init_upper_bounds; };

  std::tuple<Arr2D<u8>, usize, usize> image_data(const SolutionBase& solution, float scale = 1.0) const {
    if (scale <= 0.0) {
      throw std::runtime_error("The image scale must be > 0!");
    }

    const usize w = scale * width;
    const usize h = scale * height;

    Arr2D<float> image(w * h, 3);

    // extract and scale centers
    usize num_cells = 0;
    Arr2D<float> centers(max_num_cells, VARS_PER_CELL);
    for (usize j = 0; j < max_num_cells; j++) {
      usize k = j * VARS_PER_CELL;
      if (j < min_num_cells || solution.discrete_values()(k + ENABLED)) {
        centers.row(num_cells) = solution.discrete_values()(Eigen::seqN(k, VARS_PER_CELL)).cast<float>();
        num_cells++;
      }
    }
    centers(Eigen::seqN(0, num_cells), X_COORD) *= scale;
    centers(Eigen::seqN(0, num_cells), Y_COORD) *= scale;

    auto closest = [&](float x, float y) {
      float dist = std::numeric_limits<float>::infinity();
      usize closest_idx = 0;
      for (usize k = 0; k < num_cells; k++) {
        float d = std::pow(x - centers(k, X_COORD), 2) + std::pow(y - centers(k, Y_COORD), 2);
        if (d < dist) {
          dist = d;
          closest_idx = k;
        }
      }
      return closest_idx;
    };

    for (usize x = 0; x < w; x++) {
      for (usize y = 0; y < h; y++) {
        usize i = y * w + x;

        usize cell_idx = closest(x, y);

        image.row(i) = centers(cell_idx, Eigen::seqN(COLOR_R, 3));
      }
    }

    return std::make_tuple(image.cast<u8>(), w, h);
  };

  void evaluate(Rng& rng, SolutionSetBase& solutions, const std::span<const usize>& indices) override final {
    const usize num_pixels = width * height;
    Arr2D<float> centers(2, max_num_cells);
    Arr2D<float> colors(max_num_cells, 3);
    Array<float> pixel(2);
    KDTree<decltype(centers)> kdt;

    for (usize i : indices) {
      auto& s = solutions[i];

      // extract centers, mark inactive cells as inactive
      usize num_cells = 0;
      for (usize j = 0; j < max_num_cells; j++) {
        usize k = j * VARS_PER_CELL;
        bool cell_is_active = j < min_num_cells || s.discrete_values()(k + ENABLED);
        s.discrete_active()(Eigen::seqN(k, VARS_PER_CELL)) = cell_is_active;
        if (cell_is_active) {
          if (j < min_num_cells) {
            s.discrete_active()(k + ENABLED) = false;
          }

          centers(0, num_cells) = s.discrete_values()(k + X_COORD);
          centers(1, num_cells) = s.discrete_values()(k + Y_COORD);

          colors(num_cells, 0) = s.discrete_values()(k + COLOR_R);
          colors(num_cells, 1) = s.discrete_values()(k + COLOR_G);
          colors(num_cells, 2) = s.discrete_values()(k + COLOR_B);

          num_cells++;
        }
      }

      bool use_kdtree = num_cells >= kdtree_threshold;
      if (use_kdtree) {
        kdt.build(centers, num_cells);
      }

      // compute per-pixel mismatch
      float reconstruction_error = 0.0;
      for (usize x = 0; x < width; x++) {
        for (usize y = 0; y < height; y++) {
          usize j = y * width + x;

          usize cell_idx;
          if (use_kdtree) {
            pixel(0) = x;
            pixel(1) = y;
            cell_idx = kdt.closest(centers, pixel);
          } else {
            float best_dist = std::numeric_limits<float>::infinity();
            cell_idx = 0;
            for (usize k = 0; k < num_cells; k++) {
              float dx = centers(0, k) - static_cast<float>(x);
              float dy = centers(1, k) - static_cast<float>(y);
              float dist = dx * dx + dy * dy;
              if (dist < best_dist) {
                best_dist = dist;
                cell_idx = k;
              }
            }
          }

          reconstruction_error += (target_image.row(j) - colors.row(cell_idx)).square().sum();
        }
      }
      reconstruction_error /= static_cast<float>(num_pixels);

      s.quality_as<MOQuality>().objectives(0) = reconstruction_error;
      s.quality_as<MOQuality>().objectives(1) = num_cells;
      s.quality_as<MOQuality>().constraint_value = 0.0;
    }
  }

  void add_random(Rng& rng, SolutionSetBase& solutions, usize count) const override final {
    init->add_random(rng, *this, solutions, count);
  }

  const FitnessBase& fitness() const override final { return _fitness; };
  const ArchiveFitnessBase& archive_fitness() const override final { return _archive_fitness; };

  void log_solution(std::ostream& os, const SolutionBase& solution) const override final {
    Array<float> c(3);
    os << '{';
    bool first = true;
    for (usize i = 0; i < max_num_cells; i++) {
      usize j = i * VARS_PER_CELL;
      if (j < min_num_cells || solution.discrete_values()(j + ENABLED)) {
        if (!first) {
          os << ", ";
        }

        c = solution.discrete_values()(Eigen::seqN(j + COLOR_R, 3)).cast<float>();

        // (x, y): (r, g, b)
        os << '(' << usize(solution.discrete_values()(j + X_COORD)) << ", "
           << usize(solution.discrete_values()(j + Y_COORD)) << "): (" << c(0) << ", " << c(1) << ", " << c(2) << ")";

        first = false;
      }
    }
    os << '}';
  }

 private:
  MOFitness _fitness;
  MOFitness _archive_fitness;
  Arr2D<float> target_image;
  std::shared_ptr<InitBase> init;
  usize width;
  usize height;
  usize min_num_cells;
  usize max_num_cells;
  usize kdtree_threshold;

  Vec<DType> _discrete_domain_sizes{};
  Vec<CType> _continuous_lower_bounds{};
  Vec<CType> _continuous_upper_bounds{};

  Vec<CType> _continuous_init_lower_bounds{};
  Vec<CType> _continuous_init_upper_bounds{};
};

// It is important to use the fully qualified name for the base class (i.e. goblin::classic::DiscreteCrossoverBase) so
// that the bindings are generated correctly
class YourCustomCrossover : public goblin::classic::DiscreteCrossoverBase {
  bool crossover(Rng& rng,
                 InstanceBase& problem,
                 const SolutionBase& donor,
                 SolutionBase& offspring) const override final {
    // do whatever you want here
    return goblin::classic::UniformCrossover().crossover(rng, problem, donor, offspring);
  }
};

class YourCustomMutation : public goblin::classic::DiscreteMutationBase {
  void mutate(Rng& rng, InstanceBase& problem, SolutionBase& offspring) const override final {
    // do whatever you want here, for example toggle cells randomly
    const usize VARS_PER_CELL = 6;
    const usize l = problem.num_discrete();
    const usize max_num_cells = l / VARS_PER_CELL;
    const auto domain = problem.discrete_domain_sizes();

    std::uniform_real_distribution<double> U(0.0, 1.0);

    usize min_num_cells = 0;
    while (domain(min_num_cells) < 2) {
      min_num_cells++;
    }

    if (min_num_cells < max_num_cells) {
      double p_mut = 1.0 / static_cast<double>(max_num_cells - min_num_cells);

      for (usize i = min_num_cells; i < max_num_cells; i++) {
        usize offset = i * VARS_PER_CELL;
        if (U(rng) < p_mut) {
          offspring.discrete_values()(offset) = !offspring.discrete_values()(offset);
        }
      }
    }
  }
};

};  // namespace voronoi
};  // namespace goblin

#endif /* _GOBLIN_EXAMPLES_VORONOI_H */


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                       goblin.h continued                                                                     //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// clang-format on

#endif /* _GOBLIN_H */
