#include "goblin/lib/instance.h"

namespace goblin {
Mat<CType> InstanceBase::gradients(Rng& rng,
                                   SolutionSetBase& solutions,
                                   SolutionSetBase& parents,
                                   const std::vector<const Subset*>& subsets,
                                   const std::span<const usize>& indices,
                                   u64& evaluations) {
  const CType e = 1e-6;
  const CType ee = e + e;

  Mat<CType> grads = Mat<CType>::Zero(indices.size(), num_continuous());
  if (num_continuous() == 0) {
    return grads;
  }

  // back up qualities
  std::vector<std::unique_ptr<QualityBase>> actual;
  actual.reserve(indices.size());
  for (size_t i = 0; i < indices.size(); i++) {
    actual.push_back(solutions[indices[i]].quality().clone());
  }

  std::vector<std::unique_ptr<QualityBase>> q_e;
  q_e.reserve(indices.size());

  std::vector<usize> solutions_to_evaluate;
  solutions_to_evaluate.reserve(indices.size());
  for (size_t c = 0; c < num_continuous(); c++) {
    solutions_to_evaluate.clear();

    // -e
    for (size_t _i = 0, i; _i < indices.size(); _i++) {
      i = indices[_i];
      if (solutions[i].continuous_active()(c)) {
        solutions[i].continuous_values()(c) -= e;
        solutions_to_evaluate.push_back(i);
      }
    }

    evaluate_partial(rng, solutions, parents, subsets, solutions_to_evaluate);
    evaluations += solutions_to_evaluate.size();

    q_e.clear();
    for (size_t i = 0; i < indices.size(); i++) {
        q_e.push_back(solutions[indices[i]].quality().clone());
    }

    // +e
    for (size_t _i = 0, i; _i < indices.size(); _i++) {
      i = indices[_i];
      if (solutions[i].continuous_active()(c)) {
        solutions[i].continuous_values()(c) += ee;
      }
    }

    evaluate_partial(rng, solutions, parents, subsets, solutions_to_evaluate);
    evaluations += solutions_to_evaluate.size();

    for (size_t i = 0; i < indices.size(); i++) {
      CType dist = fitness().distance(*q_e[i], solutions[indices[i]].quality(), std::nullopt) / ee;
      if (!isna(dist) && dist > 0.0) {
        Ordering o = fitness().cmp(*q_e[i], solutions[indices[i]].quality(), std::nullopt);
        if (o == Ordering::Better) {
          grads(i, c) = dist;
        } else if (o == Ordering::Worse) {
          grads(i, c) = -dist;
        }
      }
    }

    // restore values
    for (size_t _i = 0, i; _i < indices.size(); _i++) {
      i = indices[_i];
      if (solutions[i].continuous_active()(c)) {
        solutions[i].continuous_values()(c) -= e;
      }
    }
  }

  // restore actual quality
  for (size_t i = 0; i < indices.size(); i++) {
    solutions[indices[i]].assign_quality(*actual[i]);
  }
  return grads;
}

std::tuple<std::vector<usize>, u64> InstanceBase::gradient_steps(Rng& rng,
                                                                 SolutionSetBase& solutions,
                                                                 SolutionSetBase& parents,
                                                                 const std::span<const usize>& indices,
                                                                 usize num_steps) {
  const CType e = 1e-6;
  const CType learning_rate = 0.1;

  std::vector<usize> changed_solutions;
  if (num_continuous() == 0) {
    return std::make_tuple(changed_solutions, 0);
  }

  bool any_solution_changed = false;
  std::vector<Subset> _subsets(solutions.size());
  std::vector<const Subset*> subsets(solutions.size());
  for (usize i = 0; i < indices.size(); i++) {
    for (usize j = 0; j < num_continuous(); j++) {
      if (solutions[indices[i]].continuous_active()(j)) {
        _subsets[indices[i]].continuous.push_back(j);
        subsets[indices[i]] = &_subsets[indices[i]];
        any_solution_changed = true;
      }
    }
  }
  if (!any_solution_changed) {
    return std::make_tuple(changed_solutions, 0);
  }

  u64 evaluations = 0;

  std::vector<bool> solution_changed(indices.size(), false);
  for (usize s = 0; s < num_steps && any_solution_changed; s++) {
    any_solution_changed = false;

    auto grads = gradients(rng, solutions, parents, subsets, indices, evaluations);

    for (usize i = 0; i < indices.size(); i++) {
      for (usize j = 0; j < num_continuous(); j++) {
        if (std::abs(grads(i, j)) > e) {
          solutions[indices[i]].continuous_values()(j) -= learning_rate * grads(i, j);
          solution_changed[i] = true;
          any_solution_changed = true;
        }
      }
    }
  }

  changed_solutions.reserve(indices.size());
  for (usize i = 0; i < indices.size(); i++) {
    if (solution_changed[i]) {
      changed_solutions.push_back(indices[i]);
    }
  }
  // update the quality now that the solution has been changed
  evaluate_partial(rng, solutions, parents, subsets, changed_solutions);
  evaluations += changed_solutions.size();
  return std::make_tuple(changed_solutions, evaluations);
}
};  // namespace goblin
