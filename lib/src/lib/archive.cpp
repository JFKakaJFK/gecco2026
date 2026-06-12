#include "goblin/lib/archive.h"

namespace goblin {

bool ArchiveBase::update(const SolutionBase& solution,
                         bool strict,        // If false, returns true for solutions that are
                                             // non-dominated by the archive but not accepted
                         bool check_synched  // Only consider a solution accepted if it
                                             // also is accepted by all synched archives
) {
  auto [accepted, accepted_strict] = update_archive(solution, strict);
  // the assumption is that the synched archives always are at least as //
  // "good" as this archive (after all, we update them with each improvement
  // on this archive!), so if the solution is not improving // this archive
  // there is no reason to update the synched ones
  if (accepted_strict) {
    _change_count++;
    for (auto& a : synched_archives) {
      bool a_accepted = a.get().update(solution, strict);
      if (check_synched) {
        // The synching is only used to keep a local archive to detect
        // stagnation GOMEA typically uses the global archive for testing if
        // something is an improvement and for FI
        accepted &= a_accepted;
      }
    }
  }
  return accepted;
};

bool ArchiveBase::dominates(const SolutionBase& solution, bool strict) const {
  if (!empty()) {
    for (usize i = 0; i < fitness().num_objectives(); i++) {
      auto o = fitness().cmp(so_solution(i).quality(), solution.quality(), std::nullopt);
      if (o == Ordering::Better || (!strict && o == Ordering::Equal)) {
        return true;
      }
    }

    for (usize i = 0; i < size(); i++) {
      auto o = fitness().cmp(operator[](i).quality(), solution.quality(), std::nullopt);
      if (o == Ordering::Better || (!strict && o == Ordering::Equal)) {
        return true;
      }
    }
  }

  return false;
};

bool ArchiveBase::covers(const ArchiveBase& other) const {
  if (other.empty())
    return true;  // any archive covers the empty archive
  if (empty()) {
    return false;  // an empty archive covers nothing
  } else {
    // at this point we return false if other contains a solution not matched
    // by this, otherwise the archive other is indeed covered by this

    // neither archive is empty, so the so solutions must be set...
    for (usize obj = 0; obj < fitness().num_objectives(); obj++) {
      auto o = fitness().cmp(so_solution(obj).quality(), other.so_solution(obj).quality(), obj);
      if (o == Ordering::NonDominated || o == Ordering::Worse) {
        return false;
      }
    }

    for (usize other_idx = 0; other_idx < other.size(); other_idx++) {
      bool covered = false;
      for (usize idx = 0; idx < size(); idx++) {
        auto o = fitness().cmp(operator[](idx).quality(), other[other_idx].quality(), std::nullopt);
        if (o == Ordering::Better || o == Ordering::Equal) {
          covered = true;
          break;
        } else if (o == Ordering::Worse) {
          // Both archives only contain non-dominated solutions,
          // so if a solution s in this archive is dominated by other AND
          // s is non-dominated in this archive, at least one solution in
          // other cannot be matched by this archive
          return false;
        }
      }
      if (!covered)
        return false;
    }

    return true;
  }
};

bool ArchiveBase::approximately_covers(const ArchiveBase& other, CType epsilon) const {
  if (other.empty())
    return true;  // any archive covers the empty archive
  if (empty()) {
    return false;  // an empty archive covers nothing
  } else {
    CType min_dist;
    // for each solution in other
    for (usize i = 0; i < other.size(); i++) {
      // there must be at least one solution in this that is within epsilon in objective space
      for (usize j = 0; j < size(); j++) {
        CType d = fitness().distance(operator[](j).quality(), other[i].quality(), std::nullopt);
        if (j == 0 || d < min_dist) {
          min_dist = d;
        }
      }
      if (min_dist > epsilon) {
        return false;
      }
    }
    return true;
  }
};
};  // namespace goblin
