#pragma once
#ifndef _GOBLIN_GP_TEMPLATE_H
#define _GOBLIN_GP_TEMPLATE_H

#include <set>
#include <tuple>
#include <vector>
#include <stdexcept>

#include "goblin/lib/assert.h"
#include "goblin/lib/types.h"

namespace goblin {
struct TemplateNode {
  std::vector<TemplateNode> children;
  size_t max_num_nodes;

  static TemplateNode full_nary(usize branching_factor, usize depth) {
    TemplateNode root;
    if (depth > 0) {
      root.children.reserve(branching_factor);
      for (usize i = 0; i < branching_factor; i++) {
        root.children.push_back(full_nary(branching_factor, depth - 1));
      }
    }

    // Calculate maximum number of nodes in the template
    root.max_num_nodes = calc_max_num_nodes(depth, branching_factor);

    return root;
  };

  usize size() const {
    __goblin_runtime_assert(is_tree());
    usize s = 0;
    visit([&s](const auto& _) { s++; });
    return s;
  };

  usize max_num_children() const {
    __goblin_runtime_assert(is_tree());
    usize mnc = 0;
    visit([&mnc](const auto& n) { mnc = std::max(mnc, n.children.size()); });
    return mnc;
  };

  template <typename F>
  void visit(F fn) const {
    fn(*this);
    for (auto& c : children) {
      c.visit(fn);
    }
  };

  bool is_tree() const {
    std::set<const TemplateNode*> visited;
    std::vector<const TemplateNode*> queue{this};
    while (!queue.empty()) {
      auto current = queue.back();
      queue.pop_back();
      if (visited.contains(current)) {
        return false;
      }
      visited.insert(current);
      for (auto& c : current->children) {
        queue.push_back(&c);
      }
    }
    return true;
  };

  bool is_cycle_free() const {
    std::set<const TemplateNode*> visited;
    std::set<const TemplateNode*> path;
    return is_cycle_free_helper(this, path, visited);
  };

 private:
  bool is_cycle_free_helper(const TemplateNode* current,
                            std::set<const TemplateNode*>& path,
                            std::set<const TemplateNode*>& visited) const {
    path.insert(current);
    visited.insert(current);
    for (const auto& c : current->children) {
      // predecessor in the current path is also a child -> cycle found
      // (same condition also holds for any not yet visited child)
      if (path.contains(&c) || (!visited.contains(&c) && !is_cycle_free_helper(&c, path, visited))) {
        return false;
      }
    }
    path.erase(current);
    return true;
  };

  static size_t calc_max_num_nodes(size_t branching_factor, size_t depth) {
    if (branching_factor == 0) return depth == 0 ? 1 : 0;
    if (branching_factor == 1) return depth + 1;

    auto ipow = [](size_t base, size_t exp) {
      size_t result = 1;
      for (size_t i = 0; i < exp; ++i)
        result *= base;
      return result;
    };

    return (ipow(branching_factor, depth + 1) - 1) / (branching_factor - 1);
  }
};

struct Template {
  std::vector<TemplateNode> outputs;
  std::vector<TemplateNode> subexpressions;

  Template() = default;
  Template(std::vector<TemplateNode> outputs, std::vector<TemplateNode> subexpressions)
      : outputs(outputs), subexpressions(subexpressions) {
    if (!is_valid()) {
      throw std::runtime_error("Template is not valid!");
    }
  }

  usize size() const {
    usize s = 0;
    for (auto& o : outputs) {
      s += o.size();
    }
    for (auto& o : subexpressions) {
      s += o.size();
    }
    return s;
  };

  usize max_num_children() const {
    usize mnc = 0;
    for (auto& o : outputs) {
      mnc = std::max(mnc, o.max_num_children());
    }
    for (auto& o : subexpressions) {
      mnc = std::max(mnc, o.max_num_children());
    }
    return mnc;
  };

  void add_output(TemplateNode output) {
    outputs.emplace_back(output);
    if (!is_valid()) {
      throw std::runtime_error("Template is not valid!");
    }
  };

  void add_subtree(TemplateNode subexpression) {
    subexpressions.emplace_back(subexpression);
    if (!is_valid()) {
      throw std::runtime_error("Template is not valid!");
    }
  };

  bool is_valid() const {
    for (auto& s : subexpressions) {
      if (!s.is_tree()) {
        return false;
      }
    }
    for (auto& o : outputs) {
      if (!o.is_tree()) {
        return false;
      }
    }
    return true;
  }
};

};  // namespace goblin

#endif /* _GOBLIN_GP_TEMPLATE_H */
