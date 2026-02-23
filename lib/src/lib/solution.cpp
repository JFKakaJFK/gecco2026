#include "goblin/lib/linkage_model.h"

namespace goblin {

SolutionBase& SolutionBase::operator=(const SolutionBase& other) {
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

};  // namespace goblin
