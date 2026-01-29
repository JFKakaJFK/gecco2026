#pragma once
#ifndef _GOBLIN_LIB_SOLUTION_H
#define _GOBLIN_LIB_SOLUTION_H

#include <algorithm>
#include <cassert>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <memory>
#include <cstdlib>
#include <functional>
#include <optional>
#include <span>
#include <type_traits>

#include <Eigen/Dense>

#include "goblin/lib/assert.h"
#include "goblin/lib/fitness.h"
#include "goblin/lib/rng.h"
#include "goblin/lib/types.h"

// Note the separate solution exists to hide the data ownership
// - without it, for operations using separate arenas like
// e.g. GOM with a donor from an archive the arena would have to be handled
// explicitly, but that is an implementation detail GOMEA shouldn't have to know
// about. And for other representations such as a vector of solution structs,
// this is also nicer.

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

/* imported from another header
class QualityBase {
 public:

 virtual std::unique_ptr<QualityBase> clone() const = 0;

 virtual ~QualityBase() = default;
};
 */

struct SolutionExtensionKey {
  void* token = nullptr;
};

inline bool operator==(const SolutionExtensionKey& lhs, const SolutionExtensionKey& rhs) noexcept { return lhs.token == rhs.token; }

inline bool operator!=(const SolutionExtensionKey& lhs, const SolutionExtensionKey& rhs) noexcept { return lhs.token != rhs.token; }

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
  virtual SolutionExtensionBase& get_or_insert_extension(const SolutionExtensionBase& extension) = 0; // this is fine
  virtual std::optional<std::reference_wrapper<const SolutionExtensionBase>> get_extension(
      const SolutionExtensionKey& key) const = 0;
  virtual std::optional<std::reference_wrapper<SolutionExtensionBase>> get_extension(
      const SolutionExtensionKey& key) = 0;
  virtual bool remove_extension(const SolutionExtensionKey& key) = 0;
  virtual void clear_extensions() = 0;

  // Instead of vector, shoould this be an ExtensionProxy that behaves like a vector/iterator but does not allocate full copies?? (size, begin, end, proxy to underlying collection)
  virtual usize num_extensions() const = 0;
  virtual std::vector<std::reference_wrapper<const SolutionExtensionBase>> extensions() const = 0;
  virtual std::vector<std::reference_wrapper<SolutionExtensionBase>> extensions() = 0;

  virtual QualityBase& quality() = 0; // nb::rv_policy::reference_internal
  virtual const QualityBase& quality() const = 0; // nb::rv_policy::reference_internal
  virtual void assign_quality(const QualityBase& quality) = 0;

  inline usize num_discrete() const { return discrete_values().size(); };

  virtual RefS<Vec<DType>> discrete_values() = 0;
  virtual CRefS<Vec<DType>> discrete_values() const = 0;
  virtual RefS<Active> discrete_active() = 0;
  virtual CRefS<Active> discrete_active() const = 0;

  inline usize num_continuous() const { return continuous_values().size(); };

  virtual RefS<Vec<CType>> continuous_values() = 0;
  virtual CRefS<Vec<CType>> continuous_values() const = 0;
  virtual RefS<Active> continuous_active() = 0;
  virtual CRefS<Active> continuous_active() const = 0;

  SolutionBase& operator=(const SolutionBase& other) {
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
  };

  /// Inherits a subset of the decision variables from the donor, returning true
  /// if there was a change to the active variables and an evaluation is needed.
  ///
  /// The `always_inherit_continuous` determines if the corresponding continuous
  /// variables are also inherited for discrete only subsets.
  virtual std::tuple<bool, bool> inherit(const SolutionBase& donor,
                                         const Subset& subset,
                                         bool always_inherit_continuous) {
    bool any_active_changed = false, anything_changed = false;
    bool is_continuous = subset.continuous.size() > 0;
    bool is_discrete = subset.discrete.size() > 0;

    // assert((!always_inherit_continuous || num_continuous() >= num_discrete()) &&
    //        "All discrete indices must be valid continuous indices if the continuous "
    //        "values should be inherited with the discrete ones.");

    if (is_discrete) {
      bool inherit_continuous = !is_continuous && always_inherit_continuous;
      bool inherit_by_index = num_continuous() >= num_discrete();
      for (usize di, i = 0; i < subset.discrete.size(); i++) {
        di = subset.discrete[i];
        if (discrete_values()(di) != donor.discrete_values()(di)) {
          any_active_changed |= discrete_active()(di);
          anything_changed = true;
          discrete_values()(di) = donor.discrete_values()(di);
        }

        // yes, the indices here should be from the discrete subset!
        if (inherit_continuous && inherit_by_index) {
          if (continuous_values()(di) != donor.continuous_values()(di)) {
            any_active_changed |= continuous_active()(di);
            anything_changed = true;
            continuous_values()(di) = donor.continuous_values()(di);
          }
        }
      }

      if (inherit_continuous && !inherit_by_index) {
        for (usize i = 0; i < num_continuous(); i++) {
          // TODO sufficiently relatively + absolutely different or no check, but floating point equality is not really
          // useful...
          if (continuous_values()(i) != donor.continuous_values()(i)) {
            any_active_changed |= continuous_active()(i);
            anything_changed = true;
            continuous_values()(i) = donor.continuous_values()(i);
          }
        }
      }
    }
    if (is_continuous) {
      for (usize ci, i = 0; i < subset.continuous.size(); i++) {
        ci = subset.continuous[i];
        // TODO sufficiently relatively + absolutely different or no check, but floating point equality is not really
        // useful...
        if (continuous_values()(ci) != donor.continuous_values()(ci)) {
          any_active_changed |= continuous_active()(ci);
          anything_changed = true;
          continuous_values()(ci) = donor.continuous_values()(ci);
        }
      }
    }

    return std::make_tuple(any_active_changed, anything_changed);
  };

  // virtual void reject(const SolutionBase& backup,
  //                     bool always_inherit_continuous,
  //                     std::optional<std::reference_wrapper<const Subset>> subset = std::nullopt) {
  //   quality() = backup.quality();

  //   if (subset.has_value()) {
  //     auto& s = subset.value().get();
  //     if (!s.discrete.empty()) {
  //       discrete_values()(s.discrete) = backup.discrete_values()(s.discrete);
  //       if (always_inherit_continuous) {
  //         continuous_values()(s.discrete) = backup.continuous_values()(s.discrete);
  //       }
  //     }
  //     if (!s.continuous.empty()) {
  //       continuous_values()(s.continuous) = backup.continuous_values()(s.continuous);
  //     }
  //   } else {
  //     discrete_values() = backup.discrete_values();
  //     continuous_values() = backup.continuous_values();
  //   }

  //   // The active variables always have to be restored in full,
  //   // since variables outside the subset can become active...
  //   discrete_active() = backup.discrete_active();
  //   continuous_active() = backup.continuous_active();
  // };

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
        _quality(std::move(other._quality)) {};

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
    _discrete_values = std::move(other._discrete_values);
    _discrete_active = std::move(other._discrete_active);
    _continuous_values = std::move(other._continuous_values);
    _continuous_active = std::move(other._continuous_active);
    _extensions = std::move(other._extensions);
    _quality = std::move(other._quality);

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

  void assign_quality(const QualityBase& quality) override final {
      _quality = quality.clone();
  };
  QualityBase& quality() override final { return *_quality; } // nb::rv_policy::reference_internal
  const QualityBase& quality() const override final { return *_quality; } // nb::rv_policy::reference_internal

  RefS<Vec<DType>> discrete_values() override final { return _discrete_values; }
  CRefS<Vec<DType>> discrete_values() const override final { return _discrete_values; }
  RefS<Active> discrete_active() override final { return _discrete_active; }
  CRefS<Active> discrete_active() const override final { return _discrete_active; }

  RefS<Vec<CType>> continuous_values() override final { return _continuous_values; }
  CRefS<Vec<CType>> continuous_values() const override final { return _continuous_values; }
  RefS<Active> continuous_active() override final { return _continuous_active; }
  CRefS<Active> continuous_active() const override final { return _continuous_active; }

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
  Active _discrete_active;
  Vec<CType> _continuous_values;
  Active _continuous_active;
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
 void assign_quality(const QualityBase& quality) override {
     arena->quality[idx] = quality.clone();
 }
  QualityBase& quality() override final { return *arena->quality[idx]; }
  const QualityBase& quality() const override final { return *arena->quality[idx]; }

  RefS<Vec<DType>> discrete_values() override final { return arena->discrete.row(idx); }
  CRefS<Vec<DType>> discrete_values() const override final { return arena->discrete.row(idx); }
  RefS<Active> discrete_active() override final { return arena->discrete_active.row(idx); }
  CRefS<Active> discrete_active() const override final { return arena->discrete_active.row(idx); }

  RefS<Vec<CType>> continuous_values() override final { return arena->continuous.row(idx); }
  CRefS<Vec<CType>> continuous_values() const override final { return arena->continuous.row(idx); }
  RefS<Active> continuous_active() override final { return arena->continuous_active.row(idx); }
  CRefS<Active> continuous_active() const override final { return arena->continuous_active.row(idx); }

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
      std::swap(extensions[idx], extensions[_size]); extensions[_size].clear();
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
