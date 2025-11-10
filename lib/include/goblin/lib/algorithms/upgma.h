#ifndef _GOBLIN_LIB_UPGMA_H
#define _GOBLIN_LIB_UPGMA_H

#pragma once

#include "goblin/lib/rng.h"
#include "goblin/lib/types.h"

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
