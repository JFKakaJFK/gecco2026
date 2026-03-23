#pragma once
#ifndef _GOBLIN_EXAMPLES_VORONOI_H
#define _GOBLIN_EXAMPLES_VORONOI_H

#include <stdexcept>
#include <cmath>
#include <limits>
#include <format>
#include <vector>
#include <algorithm>
#include <random>

#include "goblin/lib/instance.h"
#include "goblin/lib/init.h"
#include "goblin/methods/classic/simple_ga.h"

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
