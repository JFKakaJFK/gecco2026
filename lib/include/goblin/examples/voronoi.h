#pragma once
#ifndef _GOBLIN_EXAMPLES_VORONOI_H
#define _GOBLIN_EXAMPLES_VORONOI_H

#include <stdexcept>
#include <cmath>
#include <limits>
#include <format>
#include <algorithm>
#include <vector>

#include <print>

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

  Node* build(usize begin, usize end, isize dim, const C& coords) {
    if (end <= begin) {
      return nullptr;
    }

    // find midpoint in current dimension
    usize mid = begin + (end - begin) / 2;
    auto i = nodes.begin();
    std::nth_element(i + begin, i + mid, i + end,
                     [&](const Node& lhs, const Node& rhs) { return coords(lhs.idx, dim) < coords(rhs.idx, dim); });

    // recurse with next dimension
    dim = (dim + 1) % coords.cols();
    nodes[mid].left = build(begin, mid, dim, coords);
    nodes[mid].right = build(mid + 1, end, dim, coords);
    return &nodes[mid];
  };

  template <typename P>
  void closest_helper(Node* node, isize dim, const C& coords, const P& point) {
    if (node == nullptr) {
      return;
    }

    const isize n_dims = coords.cols();

    // Note that (coords(node->idx) - point).square().sum()
    // silently produces incorrect results due to broadcasting fun
    Scalar dist = 0.0;
    for (isize i = 0; i < n_dims; i++) {
      dist += std::pow(coords(node->idx, i) - point(i), 2);
    }

    if (dist < best_dist) {
      best_dist = dist;
      best_idx = node->idx;
    }

    if (best_dist == 0.0) {
      return;
    }

    Scalar dist_x = coords(node->idx, dim) - point(dim);
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
  void fill(const C& coords, usize size) {
    if (coords.rows() < size) {
      throw std::runtime_error("Not enough coordinates passed!");
    }
    nodes.resize(size);
    for (usize i = 0; i < nodes.size(); i++) {
      nodes[i].idx = i;
    }
    root = build(0, nodes.size(), /* dim = */ 0, coords);
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
  VoronoiImageReconstruction(const Arr2D<DType>& target_image,
                             usize width,
                             usize height,
                             usize min_num_cells = 10,
                             usize max_num_cells = 100,
                             std::optional<AnyInit> init = std::nullopt,
                             bool complexity_objective = false,
                             bool use_kdtree = false)
      : _fitness(
            /* num_objectives = */ complexity_objective ? 2 : 1,
            /* minimize = */ true),
        target_image(target_image.cast<float>()),
        init(from_any_init(init.value_or(std::make_shared<CompleteInit>()))),
        width(width),
        height(height),
        min_num_cells(min_num_cells),
        max_num_cells(max_num_cells),
        complexity_objective(complexity_objective),
        use_kdtree(use_kdtree) {
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

    image_coords.resize(2, num_pixels);
    for (usize x = 0; x < width; x++) {
      for (usize y = 0; y < height; y++) {
        usize i = y * width + x;
        image_coords(i, 0) = x;
        image_coords(i, 1) = y;
      }
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

    // TODO use KDTree
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
    Arr2D<float> centers(max_num_cells, 2);
    Arr2D<float> colors(max_num_cells, 3);
    Array<float> pixel(2);

    for (usize i : indices) {
      auto& s = solutions[i];

      // extract centers, mark inactive cells as inactive
      usize num_cells = 0;
      for (usize j = 0; j < max_num_cells; j++) {
        usize k = j * VARS_PER_CELL;
        if (j < min_num_cells || s.discrete_values()(k + ENABLED)) {
          s.discrete_active()(Eigen::seqN(k, VARS_PER_CELL)) = true;
          if (j < min_num_cells) {
            s.discrete_active()(k + ENABLED) = false;
          }

          centers(num_cells, 0) = s.discrete_values()(k + X_COORD);
          centers(num_cells, 1) = s.discrete_values()(k + Y_COORD);

          colors(num_cells, 0) = s.discrete_values()(k + COLOR_R);
          colors(num_cells, 1) = s.discrete_values()(k + COLOR_G);
          colors(num_cells, 2) = s.discrete_values()(k + COLOR_B);

          num_cells++;
        } else {
          s.discrete_active()(Eigen::seqN(k, VARS_PER_CELL)) = false;
        }
      }

      KDTree<decltype(centers)> kdt;
      if (use_kdtree) {
        kdt.fill(centers, num_cells);
      }

      auto closest = [&](float x, float y) {
        float dist = std::numeric_limits<float>::infinity();
        usize closest_idx = 0;
        for (usize k = 0; k < num_cells; k++) {
            float dx = x - centers(k, 0), dy = y - centers(k, 1);
          // float d = std::pow(x - centers(k, 0), 2) + std::pow(y - centers(k, 1), 2);
          float d = dx * dx + dy * dy;
          if (d < dist) {
            dist = d;
            closest_idx = k;
          }
        }
        return closest_idx;
      };

      // TODO create image of positions, rowwise argmin dist...

      // compute per-pixel mismatch
      float reconstruction_error = 0.0;
      for (usize x = 0; x < width; x++) {
        for (usize y = 0; y < height; y++) {
          usize j = y * width + x;

          usize cell_idx;
          if (use_kdtree) {
            pixel(0) = static_cast<float>(x);
            pixel(1) = static_cast<float>(y);
            cell_idx = kdt.closest(centers, pixel);
          } else {
            cell_idx = closest(x, y);
          }

          reconstruction_error += (target_image.row(j) - colors.row(cell_idx)).square().sum();
        }
      }
      reconstruction_error /= static_cast<float>(width * height);

      s.quality_as<MOQuality>().objectives(0) = reconstruction_error;
      if (complexity_objective) {
        s.quality_as<MOQuality>().objectives(1) = static_cast<float>(num_cells);
      }
      s.quality_as<MOQuality>().constraint_value = 0.0;
    }
  }

  void add_random(Rng& rng, SolutionSetBase& solutions, usize count) const override final {
    init->add_random(rng, *this, solutions, count);
  }

  const FitnessBase& fitness() const override final { return _fitness; };
  const ArchiveFitnessBase& archive_fitness() const override final { return _fitness; };

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
  Arr2D<float> target_image;
  Arr2D<float> image_coords;
  std::shared_ptr<InitBase> init;
  usize width;
  usize height;
  usize min_num_cells;
  usize max_num_cells;
  bool complexity_objective;
  bool use_kdtree;

  Vec<DType> _discrete_domain_sizes{};
  Vec<CType> _continuous_lower_bounds{};
  Vec<CType> _continuous_upper_bounds{};

  Vec<CType> _continuous_init_lower_bounds{};
  Vec<CType> _continuous_init_upper_bounds{};
};

// class XOver : public common::DiscreteCrossoverBase {

// }
};  // namespace goblin

#endif /* _GOBLIN_EXAMPLES_VORONOI_H */
