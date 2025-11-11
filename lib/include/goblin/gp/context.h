#pragma once
#ifndef _GOBLIN_GP_CONTEXT_H
#define _GOBLIN_GP_CONTEXT_H

#include <format>
#include <functional>
#include <iostream>
#include <limits>
#include <optional>
#include <print>
#include <set>
#include <string>
#include <vector>

#include "goblin/gp/operator.h"
#include "goblin/gp/template.h"
#include "goblin/lib/assert.h"
#include "goblin/lib/solution.h"
#include "goblin/lib/types.h"

namespace goblin {

enum class ConstantRepr : u8 { ERCs, Edges, Pool, None };

enum class ValueKind : u8 {
  Input,     // input feature idx
  Constant,  // constant marker/pool idx
  Operator,  // operator idx
  Arg,       // subfunction argument idx
  Subtree,   // subtree idx
  Parameter  // function class parameter idx
};

/// The lookup tables needed to map the linear representation to the encoded
/// semantics and the methods for computing the output, active nodes and sympy
/// conversion.
///
/// A two-step approach is used, where each semantic symbol (e.g. operators or
/// input features) are mapped to a value. However, the domain for e.g. leaf
/// nodes does not contain all values, so a second mapping is used to map
/// between node domain and value.
class GPContext {
 public:
  GPContext(usize num_inputs,
            Template expression_template,
            std::vector<std::shared_ptr<OperatorBase>> operators,
            usize num_parameters = 0,
            std::string_view constant_representation = "ercs",  // ercs, edges, pool or none for no constants
            usize constant_pool_size = 10,
            bool enable_subfunctions = false,  // ADF vs ADT
            usize max_expression_size = 50)
      : const_repr(constant_representation == "pool"
                       ? ConstantRepr::Pool
                       : (constant_representation == "ercs"
                              ? ConstantRepr::ERCs
                              : (constant_representation == "none" ? ConstantRepr::None : ConstantRepr::Edges))),
        num_inputs(num_inputs),
        num_outputs(expression_template.outputs.size()),
        num_subexpressions(expression_template.subexpressions.size()),
        num_discrete(expression_template.size()),
        num_continuous(const_repr == ConstantRepr::Pool ? constant_pool_size
                                                        : (const_repr == ConstantRepr::None ? 0 : num_discrete)),
        max_expression_size(max_expression_size),
        num_parameters(num_parameters),
        operators(std::move(operators)) {
    __goblin_runtime_assert(expression_template.is_valid());

    usize max_num_children = expression_template.max_num_children();
    usize num_constant_values = const_repr == ConstantRepr::ERCs   ? 1
                                : const_repr == ConstantRepr::Pool ? constant_pool_size
                                                                   : 0;
    usize num_subtree_args = enable_subfunctions ? max_num_children : 0;

    usize num_values = num_inputs + num_constant_values + num_subtree_args + num_subexpressions +
                       this->operators.size() + num_parameters;
    __goblin_runtime_assert(num_values <= static_cast<usize>(std::numeric_limits<DType>::max()));

    // value lookup tables

    value_kind.reserve(num_values);
    value_min_arity.reserve(num_values);
    value_max_arity.reserve(num_values);
    value_idx.reserve(num_values);
    for (usize i = 0; i < num_inputs; i++) {
      value_kind.push_back(ValueKind::Input);
      value_min_arity.push_back(0);
      value_max_arity.push_back(0);
      value_idx.push_back(i);
    }
    for (usize i = 0; i < num_constant_values; i++) {
      value_kind.push_back(ValueKind::Constant);
      value_min_arity.push_back(0);
      value_max_arity.push_back(0);
      value_idx.push_back(i);
    }
    for (usize i = 0; i < num_subtree_args; i++) {
      value_kind.push_back(ValueKind::Arg);
      value_min_arity.push_back(0);
      value_max_arity.push_back(0);
      value_idx.push_back(i);
    }
    for (usize i = 0; i < num_subexpressions; i++) {
      value_kind.push_back(ValueKind::Subtree);
      value_min_arity.push_back(enable_subfunctions ? 1 : 0);
      value_max_arity.push_back(enable_subfunctions ? num_subtree_args : 0);
      value_idx.push_back(i);
    }
    for (usize i = 0; i < this->operators.size(); i++) {
      op_idx2value.push_back(value_kind.size());
      value_kind.push_back(ValueKind::Operator);
      value_min_arity.push_back(this->operators[i]->min_arity());
      value_max_arity.push_back(std::min(max_num_children, this->operators[i]->max_arity()));
      value_idx.push_back(i);
    }
    for (usize i = 0; i < num_parameters; i++) {
      value_kind.push_back(ValueKind::Parameter);
      value_min_arity.push_back(0);
      value_max_arity.push_back(0);
      value_idx.push_back(i);
    }
    assert(value_kind.size() == num_values);

    // template structure lookup tables
    subtree_roots.resize(expression_template.subexpressions.size());
    output_roots.resize(expression_template.outputs.size());

    domain_sizes.resize(num_discrete);
    domain2value.resize(num_discrete, num_values);

    root.resize(num_discrete);
    sizes.resize(num_discrete);

    depth.resize(num_discrete);
    height.resize(num_discrete);
    children.resize(num_discrete);
    nodes.resize(num_discrete);

    _parent.resize(num_discrete, num_discrete);
    _value2domain = Arr2D<DType>::Constant(num_discrete, num_values, num_values);

    usize index = 0;
    for (usize i = 0; i < expression_template.subexpressions.size() + expression_template.outputs.size(); i++) {
      usize tree_root = index;
      bool is_subtree = expression_template.subexpressions.size();

      if (is_subtree) {
        subtree_roots[i] = tree_root;
      } else {
        output_roots[i] = tree_root;
      }

      std::vector<std::tuple<const TemplateNode*, usize>> queue{std::make_tuple(
          is_subtree ? &expression_template.subexpressions[i] : &expression_template.outputs[i], index++)};
      while (!queue.empty()) {
        auto [nptr, idx] = queue.back();
        queue.pop_back();

        // structure lookup tables

        root[idx] = tree_root;
        sizes[idx] = nptr->size();

        height[idx] = 1;
        nodes[idx] = {idx};

        auto p_idx = parent(idx);
        if (p_idx.has_value()) {
          depth[idx] = depth[p_idx.value()] + 1;

          while (p_idx.has_value()) {
            nodes[p_idx.value()].push_back(idx);
            p_idx = parent(p_idx.value());
          }
        } else {
          depth[idx] = 0;
        }

        for (const auto& c : nptr->children) {
          usize c_idx = index++;
          _parent[c_idx] = idx;
          children[idx].push_back(c_idx);
          queue.emplace_back(&c, c_idx);
        }

        // domain <-> value mapping

        // The domain for a variable are all values, except the invalid ones
        // (e.g. arg placeholders in outputs, functions in leafs or subtree
        // references that could lead to cycles)
        domain_sizes[idx] = 0;
        for (usize value = 0, domain_value; value < num_values; value++) {
          bool is_invalid_subtree_arg = !is_subtree && value_kind[value] == ValueKind::Arg;
          bool is_invalid_subfunction_index = value_kind[value] == ValueKind::Subtree && value_idx[value] >= i;
          bool is_arity_mismatch = value_min_arity[value] > children[idx].size();
          if (!(is_invalid_subtree_arg || is_invalid_subfunction_index || is_arity_mismatch)) {
            domain_value = domain_sizes[idx]++;

            domain2value(idx, domain_value) = value;
            _value2domain(idx, value) = domain_value;
          }
        }
      }
    }

    assert(index == num_discrete);
  };

  inline std::optional<DType> value2domain(usize index, DType value) {
    auto dval = _value2domain(index, value);
    if (dval < domain_sizes[index]) {
      return dval;
    }
    return std::nullopt;
  };

  inline std::optional<usize> parent(usize index) {
    auto p_idx = _parent[index];
    if (p_idx < num_discrete) {
      return p_idx;
    }
    return std::nullopt;
  };

  std::vector<std::string> to_sympy(const SolutionBase& solution) const {
    // TODO remove recursion & call stack copying by doing it iteratively
    usize size = 0;

    std::function<std::string(usize, std::vector<usize>)> helper = [&](usize idx, std::vector<usize> call_stack) {
      std::string res;
      if (size >= max_expression_size) {
        res = "SIZE OVERFLOW";
        return res;
      }
      size++;

      assert(idx < num_discrete);
      usize domain_value = solution.discrete_values()(idx);
      assert(domain_value < static_cast<usize>(domain2value.cols()));
      assert(domain_value < domain_sizes[idx]);
      DType value = domain2value(idx, domain_value);
      assert(value < value_idx.size());
      usize v_idx = value_idx[value];

      if (value_kind[value] == ValueKind::Input) {
        res = std::format("x{:d}", v_idx);
      } else if (value_kind[value] == ValueKind::Parameter) {
        res = std::format("c{:d}", v_idx);
      } else if (value_kind[value] == ValueKind::Constant) {
        res = std::format("{}", solution.continuous_values()(const_repr == ConstantRepr::Pool ? v_idx : idx));
      } else if (value_kind[value] == ValueKind::Arg) {
        size--;  // arg node is transparent
        usize calling_node = call_stack.back();
        call_stack.pop_back();
        res = helper(children[calling_node][v_idx % children[calling_node].size()], call_stack);
      } else if (value_kind[value] == ValueKind::Subtree) {
        size--;  // subtree call is transparent
        res = helper(subtree_roots[v_idx], call_stack);
      } else if (value_kind[value] == ValueKind::Operator) {
        usize arity = std::min(children[idx].size(), value_max_arity[value]);
        std::vector<std::string> args;
        args.reserve(arity);
        for (usize i = 0; i < arity; i++) {
          std::string arg = helper(children[idx][i], call_stack);
          args.push_back(arg);
        }
        res = operators[v_idx]->format(args);
      } else {
        std::unreachable();
      }

      if (const_repr == ConstantRepr::Edges) {
        res = std::format("({} * {})", solution.continuous_values()(idx), res);
      }
      return res;
    };

    std::vector<std::string> outputs;
    outputs.reserve(num_outputs);
    for (usize i = 0; i < num_outputs; i++) {
      // Works
      auto repr = helper(output_roots[i], {});
      outputs.push_back(repr);

      // This causes a container-overflow - apparently
      // recursive lambdas that capture by reference have rough edges...
      // outputs.push_back(helper(output_roots[i], {}));
    }

    return outputs;
  };

  template <typename Scalar>
  Arr2D<Scalar> compute_outputs(SolutionBase& solution, const Arr2D<Scalar>& X, const Array<Scalar>& params) const {
    Arr2D<Scalar> eval_buffer;
    usize size;
    return compute_outputs(eval_buffer, solution, X, params, size);
  }

  template <typename Scalar>
  Arr2D<Scalar> compute_outputs(Arr2D<Scalar>& eval_buffer,
                                SolutionBase& solution,
                                const Arr2D<Scalar>& X,
                                const Array<Scalar>& params,
                                usize& size) const {
    // TODO remove recursion & call stack copying by doing it iteratively
    usize buffer_idx;
    eval_buffer.resize(X.rows(), max_expression_size);

    size = 0;
    std::function<void(usize, std::vector<usize>)> helper = [&](usize idx, std::vector<usize> call_stack) {
      if (buffer_idx >= max_expression_size) {
        eval_buffer.col(0).array() = std::numeric_limits<Scalar>::quiet_NaN();
        return;
      }

      DType value = domain2value(idx, solution.discrete_values()(idx));
      solution.discrete_active()(idx) = true;
      usize v_idx = value_idx[value];

      usize buf_idx = buffer_idx;

      size++;
      if (value_kind[value] == ValueKind::Input) {
        eval_buffer.col(buffer_idx++) = X.col(v_idx);
      } else if (value_kind[value] == ValueKind::Parameter) {
        eval_buffer.col(buffer_idx++) = params[v_idx];
      } else if (value_kind[value] == ValueKind::Constant) {
        usize ci = const_repr == ConstantRepr::Pool ? v_idx : idx;
        solution.continuous_active()(ci) = true;
        eval_buffer.col(buffer_idx++) = solution.continuous_values()(ci);
      } else if (value_kind[value] == ValueKind::Arg) {
        // arg node is transparent - no need to change the buffer idx
        size--;
        usize calling_node = call_stack.back();
        call_stack.pop_back();
        helper(children[calling_node][v_idx % children[calling_node].size()], call_stack);
      } else if (value_kind[value] == ValueKind::Subtree) {
        // subtree call is transparent - no need to change the buffer idx
        size--;
        helper(subtree_roots[v_idx], call_stack);
      } else if (value_kind[value] == ValueKind::Operator) {
        usize arity = std::min(children[idx].size(), operators[v_idx]->max_arity());
        std::vector<usize> child_indices(arity);
        for (usize i = 0; i < arity; i++) {
          child_indices[i] = ++buffer_idx;
          helper(children[idx][i], call_stack);
        }
        operators[v_idx]->apply(eval_buffer.col(buf_idx), eval_buffer(Eigen::placeholders::all, child_indices));
      } else {
        std::unreachable();
      }

      if (const_repr == ConstantRepr::Edges) {
        solution.continuous_active()(idx) = true;
        eval_buffer.col(buf_idx) *= solution.continuous_values()(idx);
      }
    };

    solution.discrete_active() = false;
    solution.continuous_active() = false;
    Arr2D<Scalar> outputs(X.rows(), num_outputs);
    for (usize i = 0; i < num_outputs; i++) {
      buffer_idx = 0;
      helper(output_roots[i], {});
      outputs.col(i) = eval_buffer.col(0);

      // std::cout << eval_buffer << std::endl;
    }

    return outputs;
  };

  // TODO allow gradients w.r.t. specific continuous indices OR parameter
  // indices
  template <typename Scalar>
  Arr2D<Scalar> compute_outputs_grad(SolutionBase& solution, Arr2D<Scalar>& X, Array<Scalar>& params) const {
    std::unreachable();
  };

  // std::string to_dot(const SolutionBase &solution) const {
  //   std::unreachable();
  // };

  ConstantRepr const_repr;

  usize num_inputs;
  usize num_outputs;
  usize num_subexpressions;
  usize num_discrete;
  usize num_continuous;
  usize max_expression_size;
  usize num_parameters;

  std::vector<std::shared_ptr<OperatorBase>> operators;
  std::vector<usize> op_idx2value;

  std::vector<ValueKind> value_kind;
  std::vector<usize> value_min_arity;
  std::vector<usize> value_max_arity;
  std::vector<usize> value_idx;

  std::vector<usize> subtree_roots;  // indices of all subtree root nodes
  std::vector<usize> output_roots;   // indices of all output root nodes

  Vec<DType> domain_sizes;    // node -> domain size
  Arr2D<DType> domain2value;  // node domain -> value

  std::vector<usize> root;                   // node -> current tree root
  std::vector<usize> sizes;                  // node -> size of subtree starting at node,
                                             // ignoring inactive values
  std::vector<usize> depth;                  // node -> node depth
  std::vector<usize> height;                 // node -> node height
  std::vector<std::vector<usize>> children;  // node -> child node indices

  std::vector<std::vector<usize>> nodes;  // node -> indices corresponding to the subtree starting at this
                                          // node (without subtrees)

 private:
  Arr2D<DType> _value2domain;
  std::vector<usize> _parent;  // node -> parent or invalid index for root nodes
};
};  // namespace goblin

#endif /* _GOBLIN_GP_CONTEXT_H */
