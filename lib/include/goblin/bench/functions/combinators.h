#ifndef _GOBLIN_BENCH_FUNCTIONS_COMBINATORS_H
#define _GOBLIN_BENCH_FUNCTIONS_COMBINATORS_H

#pragma once

#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <random>

#include "goblin/bench/functions.h"
#include "goblin/lib/assert.h"
#include "goblin/lib/rng.h"

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
                                    RefS<Active> discrete_active,
                                    RefS<Active> continuous_active) override final {
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
                                    RefS<Active> discrete_active,
                                    RefS<Active> continuous_active) override final {
    discrete_active.fill(true);
    continuous_active.fill(true);
    return fn->evaluate(discrete_values, continuous_values, discrete_active, continuous_active);
  };

  std::tuple<CType, CType> evaluate(RefS<Vec<DType>> discrete_values,
                                    RefS<Vec<CType>> continuous_values,
                                    RefS<Active> discrete_active,
                                    RefS<Active> continuous_active,
                                    CRefS<Vec<DType>> parent_discrete_values,
                                    CRefS<Vec<CType>> parent_continuous_values,
                                    CRefS<Active> parent_discrete_active,
                                    CRefS<Active> parent_continuous_active,
                                    const CType parent_objective_value,
                                    const CType parent_constraint_value,
                                    const std::span<const usize>& discrete_indices,
                                    const std::span<const usize>& continuous_indices) override final {
    discrete_active.fill(true);
    continuous_active.fill(true);
    return fn->evaluate(discrete_values, continuous_values, discrete_active, continuous_active, parent_discrete_values,
                        parent_continuous_values, parent_discrete_active, parent_continuous_active,
                        parent_objective_value, parent_constraint_value, discrete_indices, continuous_indices);
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
                                    RefS<Active> discrete_active,
                                    RefS<Active> continuous_active) override final {
    // TODO this does not forward modifications to the actual solutions
    // (non-issue for my use cases so far, but still a violation of the api...)
    Vec<DType> d_inverted = DType(1) - discrete_values.array();
    Vec<CType> c_inverted = CType(0.0) - continuous_values.array();
    return fn->evaluate(d_inverted, c_inverted, discrete_active, continuous_active);
  };

  std::tuple<CType, CType> evaluate(RefS<Vec<DType>> discrete_values,
                                    RefS<Vec<CType>> continuous_values,
                                    RefS<Active> discrete_active,
                                    RefS<Active> continuous_active,
                                    CRefS<Vec<DType>> parent_discrete_values,
                                    CRefS<Vec<CType>> parent_continuous_values,
                                    CRefS<Active> parent_discrete_active,
                                    CRefS<Active> parent_continuous_active,
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
    return fn->evaluate(d_inverted, c_inverted, discrete_active, continuous_active, dp_inverted, cp_inverted,
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

    Mat<CType> tmp(block_size, block_size);

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

        block_rotation_matrix.noalias() = block_rotation_matrix * tmp;

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
    Rng rng(seed, 0);

    usize block_size = rotation_block_size.value_or(fn->num_continuous());

    Mat<CType> tmp(block_size, block_size);

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

        block_rotation_matrix.noalias() = block_rotation_matrix * tmp;

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
                                    RefS<Active> discrete_active,
                                    RefS<Active> continuous_active) override final {
    discrete_active.fill(true);
    continuous_active.fill(true);
    auto r_values = rotated(continuous_values);
    return fn->evaluate(discrete_values, r_values, discrete_active, continuous_active);
  };

  std::tuple<CType, CType> evaluate(RefS<Vec<DType>> discrete_values,
                                    RefS<Vec<CType>> continuous_values,
                                    RefS<Active> discrete_active,
                                    RefS<Active> continuous_active,
                                    CRefS<Vec<DType>> parent_discrete_values,
                                    CRefS<Vec<CType>> parent_continuous_values,
                                    CRefS<Active> parent_discrete_active,
                                    CRefS<Active> parent_continuous_active,
                                    const CType parent_objective_value,
                                    const CType parent_constraint_value,
                                    const std::span<const usize>& discrete_indices,
                                    const std::span<const usize>& continuous_indices) override final {
    discrete_active.fill(true);
    continuous_active.fill(true);
    auto r_values = rotated(continuous_values);
    auto r_parent_values = rotated(parent_continuous_values);
    return fn->evaluate(discrete_values, r_values, discrete_active, continuous_active, parent_discrete_values,
                        r_parent_values, parent_discrete_active, parent_continuous_active, parent_objective_value,
                        parent_constraint_value, discrete_indices, continuous_indices);
  };

 private:
  template <typename V>
  Vec<CType> rotated(V v) {
    Vec<CType> r(v.size());
    Vec<CType> tmp = Vec<CType>::Zero(block_rotation_matrix.rows());
    for (usize i = 0; i < v.size(); i += block_rotation_matrix.rows()) {
      usize l = std::min(v.size(), block_rotation_matrix.rows());
      tmp.setZero();
      tmp(Eigen::seqN(0, l)) = v(Eigen::seqN(i, l));
      r(Eigen::seqN(i, l)) = (block_rotation_matrix * tmp)(Eigen::seqN(0, l));
    }
    return r;
  };

  std::shared_ptr<ObjectiveBase> fn;
  Mat<CType> block_rotation_matrix;
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
                                    RefS<Active> discrete_active,
                                    RefS<Active> continuous_active) override final {
    CType ov = CType(0.0), cv = CType(0.0);
    for (auto& o : fns) {
      auto [fov, fcv] = o->evaluate(discrete_values, continuous_values, discrete_active, continuous_active);
      ov += fov;
      cv += std::max(CType(0.0), cv);
    }
    return std::make_tuple(ov, cv);
  };

  std::tuple<CType, CType> evaluate(RefS<Vec<DType>> discrete_values,
                                    RefS<Vec<CType>> continuous_values,
                                    RefS<Active> discrete_active,
                                    RefS<Active> continuous_active,
                                    CRefS<Vec<DType>> parent_discrete_values,
                                    CRefS<Vec<CType>> parent_continuous_values,
                                    CRefS<Active> parent_discrete_active,
                                    CRefS<Active> parent_continuous_active,
                                    const CType parent_objective_value,
                                    const CType parent_constraint_value,
                                    const std::span<const usize>& discrete_indices,
                                    const std::span<const usize>& continuous_indices) override final {
    CType ov = CType(0.0), cv = CType(0.0);
    for (auto& o : fns) {
      auto [fov, fcv] =
          o->evaluate(discrete_values, continuous_values, discrete_active, continuous_active, parent_discrete_values,
                      parent_continuous_values, parent_discrete_active, parent_continuous_active,
                      parent_objective_value, parent_constraint_value, discrete_indices, continuous_indices);
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
                                    RefS<Active> discrete_active,
                                    RefS<Active> continuous_active) override final {
    CType ov = -std::numeric_limits<CType>().infinity(), cv = CType(0.0);
    for (auto& o : fns) {
      auto [fov, fcv] = o->evaluate(discrete_values, continuous_values, discrete_active, continuous_active);
      ov = std::max(ov, fov);
      cv += std::max(CType(0.0), cv);
    }
    return std::make_tuple(ov, cv);
  };

  std::tuple<CType, CType> evaluate(RefS<Vec<DType>> discrete_values,
                                    RefS<Vec<CType>> continuous_values,
                                    RefS<Active> discrete_active,
                                    RefS<Active> continuous_active,
                                    CRefS<Vec<DType>> parent_discrete_values,
                                    CRefS<Vec<CType>> parent_continuous_values,
                                    CRefS<Active> parent_discrete_active,
                                    CRefS<Active> parent_continuous_active,
                                    const CType parent_objective_value,
                                    const CType parent_constraint_value,
                                    const std::span<const usize>& discrete_indices,
                                    const std::span<const usize>& continuous_indices) override final {
    CType ov = -std::numeric_limits<CType>().infinity(), cv = CType(0.0);
    for (auto& o : fns) {
      auto [fov, fcv] =
          o->evaluate(discrete_values, continuous_values, discrete_active, continuous_active, parent_discrete_values,
                      parent_continuous_values, parent_discrete_active, parent_continuous_active,
                      parent_objective_value, parent_constraint_value, discrete_indices, continuous_indices);
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
                                    RefS<Active> discrete_active,
                                    RefS<Active> continuous_active) override final {
    CType ov = std::numeric_limits<CType>().infinity(), cv = CType(0.0);
    for (auto& o : fns) {
      auto [fov, fcv] = o->evaluate(discrete_values, continuous_values, discrete_active, continuous_active);
      ov = std::min(ov, fov);
      cv += std::max(CType(0.0), cv);
    }
    return std::make_tuple(ov, cv);
  };

  std::tuple<CType, CType> evaluate(RefS<Vec<DType>> discrete_values,
                                    RefS<Vec<CType>> continuous_values,
                                    RefS<Active> discrete_active,
                                    RefS<Active> continuous_active,
                                    CRefS<Vec<DType>> parent_discrete_values,
                                    CRefS<Vec<CType>> parent_continuous_values,
                                    CRefS<Active> parent_discrete_active,
                                    CRefS<Active> parent_continuous_active,
                                    const CType parent_objective_value,
                                    const CType parent_constraint_value,
                                    const std::span<const usize>& discrete_indices,
                                    const std::span<const usize>& continuous_indices) override final {
    CType ov = std::numeric_limits<CType>().infinity(), cv = CType(0.0);
    for (auto& o : fns) {
      auto [fov, fcv] =
          o->evaluate(discrete_values, continuous_values, discrete_active, continuous_active, parent_discrete_values,
                      parent_continuous_values, parent_discrete_active, parent_continuous_active,
                      parent_objective_value, parent_constraint_value, discrete_indices, continuous_indices);
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
                                    RefS<Active> discrete_active,
                                    RefS<Active> continuous_active) override final {
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

  std::tuple<CType, CType> evaluate(RefS<Vec<DType>> discrete_values,
                                    RefS<Vec<CType>> continuous_values,
                                    RefS<Active> discrete_active,
                                    RefS<Active> continuous_active,
                                    CRefS<Vec<DType>> parent_discrete_values,
                                    CRefS<Vec<CType>> parent_continuous_values,
                                    CRefS<Active> parent_discrete_active,
                                    CRefS<Active> parent_continuous_active,
                                    const CType parent_objective_value,
                                    const CType parent_constraint_value,
                                    const std::span<const usize>& discrete_indices,
                                    const std::span<const usize>& continuous_indices) override final {
    CType ov = CType(0.0), cv = CType(0.0);
    usize d_offset = 0, c_offset = 0, d_len, c_len;
    for (auto& o : fns) {
      d_len = o->num_discrete();
      c_len = o->num_continuous();
      auto d_seq = Eigen::seqN(d_offset, d_len);
      auto c_seq = Eigen::seqN(c_offset, c_len);
      auto [fov, fcv] = o->evaluate(
          discrete_values(d_seq), continuous_values(c_seq), discrete_active(d_seq), continuous_active(c_seq),
          parent_discrete_values(d_seq), parent_continuous_values(c_seq), parent_discrete_active(d_seq),
          parent_continuous_active(c_seq), parent_objective_value, parent_constraint_value,
          discrete_indices.subspan(d_offset, d_len), continuous_indices.subspan(c_offset, c_len));
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
                                    RefS<Active> discrete_active,
                                    RefS<Active> continuous_active) override final {
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

  std::tuple<CType, CType> evaluate(RefS<Vec<DType>> discrete_values,
                                    RefS<Vec<CType>> continuous_values,
                                    RefS<Active> discrete_active,
                                    RefS<Active> continuous_active,
                                    CRefS<Vec<DType>> parent_discrete_values,
                                    CRefS<Vec<CType>> parent_continuous_values,
                                    CRefS<Active> parent_discrete_active,
                                    CRefS<Active> parent_continuous_active,
                                    const CType parent_objective_value,
                                    const CType parent_constraint_value,
                                    const std::span<const usize>& discrete_indices,
                                    const std::span<const usize>& continuous_indices) override final {
    CType ov = CType(0.0), cv = CType(0.0);
    usize d_offset = 0, c_offset = 0, d_len, c_len;
    for (usize i = 0; i < _repeats; i++) {
      d_len = fn->num_discrete();
      c_len = fn->num_continuous();
      auto d_seq = Eigen::seqN(d_offset, d_len);
      auto c_seq = Eigen::seqN(c_offset, c_len);
      auto [fov, fcv] = fn->evaluate(
          discrete_values(d_seq), continuous_values(c_seq), discrete_active(d_seq), continuous_active(c_seq),
          parent_discrete_values(d_seq), parent_continuous_values(c_seq), parent_discrete_active(d_seq),
          parent_continuous_active(c_seq), parent_objective_value, parent_constraint_value,
          discrete_indices.subspan(d_offset, d_len), continuous_indices.subspan(c_offset, c_len));
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
