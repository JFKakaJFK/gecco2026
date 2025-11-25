#ifndef _GOBLIN_LIB_SOLUTION_H
#define _GOBLIN_LIB_SOLUTION_H

#include "fitness.h"
#include "types.h"
#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <concepts>
#include <cstdint>
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

class SolutionBase {
 public:
  virtual Quality& quality() = 0;
  virtual const Quality& quality() const = 0;

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

      quality() = other.quality();
    }

    return *this;
  };

  /// Inherits a subset of the decision variables from the donor, returning true
  /// if there was a change to the active variables and an evaluation is needed.
  ///
  /// The `always_inherit_continuous` determines if the corresponding continuous
  /// variables are also inherited for discrete only subsets.
  virtual bool inherit(const SolutionBase& donor, const Subset& subset, bool always_inherit_continuous) {
    bool any_active_changed = false;
    bool is_continuous = subset.continuous.size() > 0;
    bool is_discrete = subset.discrete.size() > 0;

    // note: remove if costly
    if (!is_continuous && !is_discrete) {
      return false;
    }

    if (is_discrete) {
      for (usize di, i = 0; i < subset.discrete.size(); i++) {
        di = subset.discrete[i];
        if (discrete_values()(di) != donor.discrete_values()(di)) {
          any_active_changed |= discrete_active()(di);
          discrete_values()(di) = donor.discrete_values()(di);
        }

        // yes, the indices here should be from the discrete subset!
        if (!is_continuous && always_inherit_continuous) {
          if (continuous_values()(di) != donor.continuous_values()(di)) {
            any_active_changed |= continuous_active()(di);
            continuous_values()(di) = donor.continuous_values()(di);
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
          continuous_values()(ci) = donor.continuous_values()(ci);
        }
      }
    }
    // else if (always_inherit_continuous) {
    //   // yes, the indices here should be from the discrete subset!
    //   // TODO do I need to mark anything as active here? - i.e. if the constant is active and this leads to a change,
    //   then an eval is needed... continuous_values()(subset.discrete) = donor.continuous_values()(subset.discrete);
    // }

    return any_active_changed;
  };

  virtual void reject(const SolutionBase& backup,
                      bool always_inherit_continuous,
                      std::optional<std::reference_wrapper<const Subset>> subset = std::nullopt) {
    quality() = backup.quality();

    if (subset.has_value()) {
      auto& s = subset.value().get();
      if (!s.discrete.empty()) {
        discrete_values()(s.discrete) = backup.discrete_values()(s.discrete);
        if (always_inherit_continuous) {
          continuous_values()(s.discrete) = backup.continuous_values()(s.discrete);
        }
      }
      if (!s.continuous.empty()) {
        continuous_values()(s.continuous) = backup.continuous_values()(s.continuous);
      }
    } else {
      discrete_values() = backup.discrete_values();
      continuous_values() = backup.continuous_values();
    }

    // The active variables always have to be restored in full,
    // since variables outside the subset can become active...
    discrete_active() = backup.discrete_active();
    continuous_active() = backup.continuous_active();
  };

  virtual ~SolutionBase() {};
};

class Solution : public SolutionBase {
 public:
  Solution(Quality quality,
           std::optional<Vec<DType>> discrete_values = std::nullopt,
           std::optional<Vec<CType>> continuous_values = std::nullopt)
      : _quality(quality) {
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

  Solution(const SolutionBase& s)
      : _discrete_values(s.discrete_values()),
        _discrete_active(s.discrete_active()),
        _continuous_values(s.continuous_values()),
        _continuous_active(s.continuous_active()),
        _quality(s.quality()) {};

  Quality& quality() override final { return _quality; }
  const Quality& quality() const override final { return _quality; }

  RefS<Vec<DType>> discrete_values() override final { return _discrete_values; }
  CRefS<Vec<DType>> discrete_values() const override final { return _discrete_values; }
  RefS<Active> discrete_active() override final { return _discrete_active; }
  CRefS<Active> discrete_active() const override final { return _discrete_active; }

  RefS<Vec<CType>> continuous_values() override final { return _continuous_values; }
  CRefS<Vec<CType>> continuous_values() const override final { return _continuous_values; }
  RefS<Active> continuous_active() override final { return _continuous_active; }
  CRefS<Active> continuous_active() const override final { return _continuous_active; }

 private:
  Vec<DType> _discrete_values;
  Active _discrete_active;
  Vec<CType> _continuous_values;
  Active _continuous_active;
  Quality _quality;
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
  Quality& quality() override final { return arena->quality[idx]; }
  const Quality& quality() const override final { return arena->quality[idx]; }

  RefS<Vec<DType>> discrete_values() override final { return arena->discrete.row(idx); }
  CRefS<Vec<DType>> discrete_values() const override final { return arena->discrete.row(idx); }
  RefS<Active> discrete_active() override final { return arena->discrete_active.row(idx); }
  CRefS<Active> discrete_active() const override final { return arena->discrete_active.row(idx); }

  RefS<Vec<CType>> continuous_values() override final { return arena->continuous.row(idx); }
  CRefS<Vec<CType>> continuous_values() const override final { return arena->continuous.row(idx); }
  RefS<Active> continuous_active() override final { return arena->continuous_active.row(idx); }
  CRefS<Active> continuous_active() const override final { return arena->continuous_active.row(idx); }

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
      quality.push_back(_s.quality());
    } else {
      quality[_size] = _s.quality();
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
      quality[idx] = quality[_size];

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

  std::vector<Quality> quality;
};

using DefaultSolutionSet = AoSSet;

};  // namespace goblin

#endif /* _GOBLIN_LIB_SOLUTION_H */
