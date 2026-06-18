#include <cassert>
#include "goblin/lib/archive.h"

namespace goblin {

std::tuple<bool, bool> AdaptiveGridArchive::update_archive(const SolutionBase& solution, bool strict) {
    std::vector<usize> sorted_dominations;
    auto [is_so_elite, is_dominated] = update_so_solutions(solution);
    if (is_dominated) {
        return std::make_tuple(false, false);
    }

    for (usize i = 0; i < _solutions.size(); i++) {
        auto o = _fitness.cmp(solution.quality(), _solutions[i].quality(), std::nullopt);
        if (o == Ordering::Better) {
            sorted_dominations.push_back(i);
        } else if (o == Ordering::Worse) {
            assert(!is_so_elite && "If a new elite is dominated by another solution, the archive state is corrupt.");
            return std::make_tuple(false, false);
        } else if (o == Ordering::Equal || same_box(solution, _solutions[i])) {
            // box is already occupied and the solution is not better
            bool replace = is_so_elite;

            // if discrete: check if diversity in parameter space is added
            if (!replace && solution.num_discrete() > 0) {
                usize max_dist = std::numeric_limits<usize>::max();
                usize max_other_dist = std::numeric_limits<usize>::max();
                usize dist, other_dist;
                for (usize j = 0; j < _solutions.size(); j++) {
                    if (j == i)
                        continue;
                    dist = 0;
                    other_dist = 0;
                    for (usize k = 0; k < solution.num_discrete(); k++) {
                        if (solution.discrete_active()[k] &&
                            solution.discrete_values()[k] != _solutions[j].discrete_values()[k]) {
                            dist++;
                        }
                        if (_solutions[i].discrete_active()[k] &&
                            _solutions[i].discrete_values()[k] != _solutions[j].discrete_values()[k]) {
                            other_dist++;
                        }
                    }
                    if (dist < max_dist) {
                        max_dist = dist;
                    }
                    if (other_dist < max_other_dist) {
                        max_other_dist = dist;
                    }
                }
                if (max_dist > max_other_dist) {
                    replace = true;
                }
            }

            if (replace) {
                sorted_dominations.push_back(i);
            } else {
                return std::make_tuple(!strict, false);
            }
        }
    }

    if (!sorted_dominations.empty()) {
        // _solutions.remove_sorted_indices(sorted_dominations);
        for (usize i = sorted_dominations.size(); i > 0; i--) {
            _solutions.remove_at(sorted_dominations[i - 1]);
        }
    }

    _solutions.add(solution);

    return std::make_tuple(true, true);
};

void AdaptiveGridArchive::adapt() {
    if (size() > _c_max && _max_iterations > 0) {
        // get the maximum finite distance to the best so solution for each
        // objective
        std::vector<CType> max_objective_dist(_fitness.num_objectives(), 0.0);
        for (usize i = 0; i < _solutions.size(); i++) {
            for (usize obj = 0; obj < _fitness.num_objectives(); obj++) {
                auto d = _fitness.distance(_so_solutions[obj].quality(), _solutions[i].quality(), obj);
                if (!isna(d) && d > max_objective_dist[obj]) {
                    max_objective_dist[obj] = d;
                }
            }
        }

        DefaultSolutionSet backup = _solutions;
        usize cc = _change_count;

        CType low = 1.0;
        CType high = _max_resolution;
        CType prev = 0.0;

        if (_discretization.empty()) {
            _discretization.resize(_fitness.num_objectives(), 0.0);
        }

        for (usize i = 0; i < _max_iterations; i++) {
            CType mid = 0.5 * (low + high);

            if (prev > 0 && prev == mid)
                break;
            prev = mid;

            for (usize obj = 0; obj < _fitness.num_objectives(); obj++) {
                _discretization[obj] = max_objective_dist[obj] / mid;
                // shouldn't matter - we don't divide by it and semantically it just
                // means "not identical" is a different box assert(discretization[obj]
                // > 0.0);
            }

            // clear the archive and add everything back with the new discretization
            _solutions.clear();
            for (usize obj = 0; obj < _fitness.num_objectives(); obj++) {
                update(_so_solutions[obj], true);
            }
            for (usize j = 0; j < backup.size(); j++) {
                update(backup[j], true);
            }

            if (_solutions.size() < _c_min) {
                low = mid;
            } else if (_solutions.size() == _c_min) {
                break;
            } else {
                high = mid;
            }
        }

        // change count should not be affected by re-discretizing
        _change_count = cc;
    }
};

bool AdaptiveGridArchive::same_box(const SolutionBase& lhs, const SolutionBase& rhs) {
    if (_discretization.empty() && _initial_discretization.has_value()) {
        _discretization.resize(_fitness.num_objectives(), _initial_discretization.value());
    }
    if (!_discretization.empty()) {
        __goblin_runtime_assert(_discretization.size() == _fitness.num_objectives());
        for (usize obj = 0; obj < _fitness.num_objectives(); obj++) {
            // The original code uses floor(f1 / discretization) == floor(f2 /
            // discretization), but since I only have relative comparisons, this is
            // as close as it gets and semantically almost the same. Conceptually,
            // the original code uses an infinite grid and here the grid cells are
            // centered around each solution
            if (_fitness.distance(lhs.quality(), rhs.quality(), obj) > _discretization[obj]) {
                return false;
            }
        }
        return true;
    }
    return false;
};

std::tuple<bool, bool> AdaptiveGridArchive::update_so_solutions(const SolutionBase& solution) {
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
