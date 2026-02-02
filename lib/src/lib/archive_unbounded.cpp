#include "goblin/lib/archive.h"

namespace goblin {

std::tuple<bool, bool> UnboundedArchive::update_archive(const SolutionBase& solution, bool strict) {
  std::vector<usize> sorted_dominations;
  auto [is_so_elite, is_dominated] = update_so_solutions(solution);
  if (is_dominated) {
    return std::make_tuple(false, false);
  }
  for (usize i = 0; i < _solutions.size(); i++) {
    auto o = fitness().cmp(solution.quality(), _solutions[i].quality(), std::nullopt);
    if (o == Ordering::Better) {
      sorted_dominations.push_back(i);
    } else if (o == Ordering::Worse) {
      return std::make_tuple(false, false);
    } else if (o == Ordering::Equal) {
      // TODO only accept if not in the set already
      return std::make_tuple(!strict, false);
    }
  }

  if (!sorted_dominations.empty()) {
    //   std::span<usize> si(sorted_dominations.data(),
    //   sorted_dominations.size());
    // _solutions.remove_indices_sorted(si); //sorted_dominations);
    for (usize i = sorted_dominations.size(); i > 0; i--) {
      _solutions.remove_at(sorted_dominations[i - 1]);
    }
  }

  _solutions.add(solution);

  return std::make_tuple(true, true);
};

std::tuple<bool, bool> UnboundedArchive::update_so_solutions(const SolutionBase& solution) {
  if (_so_solutions.empty()) {
    _so_solutions.reserve(fitness().num_objectives());
    for (usize obj = 0; obj < fitness().num_objectives(); obj++) {
      _so_solutions.add(solution);
    }
    return std::make_tuple(true, false);
  } else {
    for (usize obj = 0; obj < fitness().num_objectives(); obj++) {
      if (fitness().cmp(solution.quality(), _so_solutions[obj].quality(), obj) == Ordering::Better) {
        _so_solutions[obj] = solution;
        return std::make_tuple(true, false);
      }
      if (fitness().cmp(solution.quality(), _so_solutions[obj].quality(), std::nullopt) == Ordering::Worse) {
        return std::make_tuple(false, true);
      }
    }
    return std::make_tuple(false, false);
  }
};
};  // namespace goblin
