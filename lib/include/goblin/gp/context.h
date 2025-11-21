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
#include <type_traits>
#include <span>
#include <stdexcept>
#include <iterator>
#include <ranges>

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
        max_num_children(expression_template.max_num_children()),
        operators(std::move(operators)) {
    __goblin_runtime_assert(expression_template.is_valid());
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
    usize nsubtrees = expression_template.subexpressions.size();
    usize ntrees = nsubtrees + expression_template.outputs.size();
    for (usize i = 0; i < ntrees; i++) {
      usize tree_root = index++;
      bool is_subtree = i < nsubtrees;

      if (is_subtree) {
        subtree_roots[i] = tree_root;
      } else {
        output_roots[i - nsubtrees] = tree_root;
      }

      std::vector<std::tuple<const TemplateNode*, usize>> node_stack{std::make_tuple(
          (is_subtree ? &expression_template.subexpressions[i] : &expression_template.outputs[i - nsubtrees]),
          tree_root)};
      while (!node_stack.empty()) {
        auto [nptr, idx] = node_stack.back();
        node_stack.pop_back();

        // structure lookup tables

        root[idx] = tree_root;
        sizes[idx] = nptr->size();

        height[idx] = 1;
        nodes[idx] = {idx};

        if (idx == tree_root) {
          depth[idx] = 0;
        } else {
          auto p_idx = parent(idx);
          assert(p_idx.has_value());

          depth[idx] = depth[p_idx.value()] + 1;

          while (p_idx.has_value()) {
            nodes[p_idx.value()].push_back(idx);
            p_idx = parent(p_idx.value());
          }
        }

        for (const auto& c : nptr->children) {
          usize c_idx = index++;
          _parent[c_idx] = idx;
          children[idx].push_back(c_idx);
          node_stack.emplace_back(&c, c_idx);
        }

        // domain <-> value mapping

        // The domain for a variable are all values, except the invalid ones
        // (e.g. arg placeholders in outputs, functions in leafs or subtree
        // references that could lead to cycles)
        domain_sizes[idx] = 0;
        for (usize value = 0, domain_value; value < num_values; value++) {
          // output trees cannot have arguments
          bool is_invalid_subtree_arg = !is_subtree && value_kind[value] == ValueKind::Arg;
          // subtrees can only call previous subtrees to prevent cycles
          bool is_invalid_subfunction_index =
              is_subtree && value_kind[value] == ValueKind::Subtree && value_idx[value] >= i;
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

    std::function<std::string(usize, std::vector<usize>&)> helper = [&](usize idx, std::vector<usize>& call_stack) {
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
        assert(!call_stack.empty());
        size--;  // arg node is transparent
        usize calling_node = call_stack.back();
        call_stack.pop_back();
        res = helper(children[calling_node][v_idx % children[calling_node].size()], call_stack);
        call_stack.push_back(calling_node);
      } else if (value_kind[value] == ValueKind::Subtree) {
        size--;  // subtree call is transparent
        call_stack.push_back(idx);
        res = helper(subtree_roots[v_idx], call_stack);
        call_stack.pop_back();
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
      std::vector<usize> call_stack;
      auto repr = helper(output_roots[i], call_stack);
      outputs.push_back(repr);

      // This causes a container-overflow - apparently
      // recursive lambdas that capture by reference have rough edges...
      // outputs.push_back(helper(output_roots[i], {}));
    }

    return outputs;
  };

  template <typename Scalar>
  std::optional<Arr2D<Scalar>> compute_outputs(SolutionBase& solution, const Arr2D<Scalar>& X, const Array<Scalar>& params) const {
    Arr2D<Scalar> eval_buffer;
    usize size;
    return compute_outputs(eval_buffer, solution, X, params, size);
  }

  template <typename Scalar>
  Arr2D<Scalar> compute_outputs2(Arr2D<Scalar>& eval_buffer,
                                 SolutionBase& solution,
                                 const Arr2D<Scalar>& X,
                                 const Array<Scalar>& params,
                                 usize& size) const {
    // TODO remove recursion & call stack copying by doing it iteratively
    usize buffer_idx;
    eval_buffer.resize(X.rows(), max_expression_size);

    size = 0;
    std::function<void(usize, std::vector<usize>&)> helper = [&](usize idx, std::vector<usize>& call_stack) {
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
        call_stack.push_back(calling_node);
      } else if (value_kind[value] == ValueKind::Subtree) {
        // subtree call is transparent - no need to change the buffer idx
        size--;
        call_stack.push_back(idx);
        helper(subtree_roots[v_idx], call_stack);
        call_stack.pop_back();
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

    solution.discrete_active().array() = false;
    solution.continuous_active().array() = false;
    Arr2D<Scalar> outputs(X.rows(), num_outputs);
    for (usize i = 0; i < num_outputs; i++) {
      buffer_idx = 0;
      std::vector<usize> call_stack;
      helper(output_roots[i], call_stack);
      outputs.col(i) = eval_buffer.col(0);

      // std::cout << eval_buffer << std::endl;
    }

    return outputs;
  };

  // Returns all non-reference nodes almost in-order (argument order is reversed) and the output indices.
  //
  // The variables are marked active/inactive only if the solution is not const. This also returns the node indices, not
  // the values directly since some information like arity/constant lookups still need more information.
  template <typename S>
  std::tuple<std::vector<usize>, std::vector<usize>> nodes_in_order(S& solution,
                                                                    bool discount_size,
                                                                    bool& size_overflow,
                                                                    usize& size) const {
    // initially we haven't visited anything, so we set everything to be inactive
    if constexpr (!std::is_const<S>()) {
      solution.discrete_active().array() = false;
      solution.continuous_active().array() = false;
    }
    size_overflow = false;

    std::vector<usize> nodes;
    nodes.reserve(max_expression_size);
    std::vector<usize> output_indices;
    output_indices.reserve(output_roots.size());

    // in the modular GP-GOMEA paper there is this concept of "discounted" size to not punish re-using subfunctions by
    // only counting the subfunction nodes once.
    Array<usize> visited = Array<usize>::Zero(num_discrete);

    // to resolve subfunction arguments, we need to know the calling node
    // (this stack only increases, to avoid revisiting nodes and more importantly to not invalidate stack indices)
    std::vector<usize> call_stack;
    call_stack.reserve(max_expression_size);

    // for each we need to visit, we need the node index and the call stack idx
    std::vector<std::tuple<usize, usize>> node_stack;
    node_stack.reserve(max_expression_size);

    // for each output, walk the tree in order and add the nodes to `values`
    for (usize n : output_roots) {
      // housekeeping: reset the call stack and node_stack per output
      call_stack.clear();
      node_stack.clear();
      node_stack.emplace_back(n, 0);

      // Since all subtrees are added to nodes here, we need to have some way to know
      // when one tree ends and another starts (not sure if this is the best choice - doing all trees at once makes
      // tracking the size easier but the evaluation more complex; in the end it matters that there is a size bound for
      // pre-allocation, but not really if that bound is per tree or for all trees together...)
      output_indices.push_back(nodes.size());

      while (!node_stack.empty()) {
        // we hit the max size, but have a next node since this is inside the loop
        if (nodes.size() == max_expression_size) {
          size_overflow = true;
          size = max_expression_size + 1;
          return std::make_tuple(nodes, output_indices);
        }

        // get the node and mark it as active
        auto [idx, call_stack_idx] = node_stack.back();
        node_stack.pop_back();
        if constexpr (!std::is_const<S>()) {
          solution.discrete_active()(idx) = true;
        }

        // track the size (repeat visits don't count in the "discounted" setting)
        if (discount_size) {
          visited(idx) = 1;
        } else {
          visited(idx) += 1;
        }

        // lookup the value of the current node
        DType value = domain2value(idx, solution.discrete_values()(idx));
        usize v_idx = value_idx[value];

        if (value_kind[value] == ValueKind::Arg) {
          // we need to replace the argument with the corresponding child of the caller
          usize calling_node = call_stack[call_stack_idx];
          auto& cnodes = children[calling_node];
          node_stack.emplace_back(cnodes[v_idx % cnodes.size()],
                                  call_stack_idx - 1  // use the stack index of the caller
          );
        } else if (value_kind[value] == ValueKind::Subtree) {
          // we need to replace the actual subtree with the called subtree
          call_stack.push_back(idx);
          node_stack.emplace_back(subtree_roots[v_idx],
                                  call_stack.size() - 1  // a call always needs to use the top of the stack, no matter
                                                         // where the current call_stack_idx is (!)
          );
        } else if (value_kind[value] == ValueKind::Operator) {
          // add the operator
          nodes.push_back(idx);

          // then the arguments
          usize arity = std::min(children[idx].size(), value_max_arity[value]);
          for (usize i = 0; i < arity; i++) {
            // the child's calling node/call_stack_idx is the same as the parents
            node_stack.emplace_back(children[idx][i], call_stack_idx);
          }
        } else if (value_min_arity[value] > 0) {
          // for anything that is not a leaf, we need to resolve the arguments, so another branch is needed if
          // non-terminal kinds are added
          throw std::runtime_error("Encountered unhandled non-leaf node.");
        } else {
          if constexpr (!std::is_const<S>()) {
            if (value_kind[value] == ValueKind::Constant) {
              solution.continuous_active()(const_repr == ConstantRepr::Pool ? v_idx : idx) = true;
            }
          }
          // for leafs we only have to add the node
          nodes.push_back(idx);
        }

        if (const_repr == ConstantRepr::Edges) {
          if constexpr (!std::is_const<S>()) {
            solution.continuous_active()(idx) = true;
          }
        }
      }
    }

    // TODO check with Peter if Joe counts the fn/arg nodes in https://arxiv.org/pdf/2505.01262v1 (I don't count them
    // here) size = visited.sum();
    size = discount_size ? visited.sum() : nodes.size();

    return std::make_tuple(nodes, output_indices);
  }

  // Returns all trees in postfix/reverse polish notation (https://en.wikipedia.org/wiki/Reverse_Polish_notation) and
  // without references if the total number of nodes exceeds the `max_expression_size` or `std::nullopt` otherwise.
  //
  // The variables are marked active/inactive only if the solution is not const.
  // Note that the reason why it's `nodes_post_order` and not `values_post_order` is that for constants/operator calls the corresponding index/arity isn't fully determined by the value alone.
  template <typename S>
  std::optional<std::vector<std::vector<usize>>> nodes_post_order(S& solution, bool discount_size, usize& size) const {
    // initially we haven't visited anything, so we set everything to be inactive
    if constexpr (!std::is_const<S>()) {
      solution.discrete_active().array() = false;
      solution.continuous_active().array() = false;
    }

    std::vector<std::vector<usize>> nodes;
    nodes.reserve(num_outputs);

    // in the modular GP-GOMEA paper (https://arxiv.org/pdf/2505.01262v1) there is this concept of "discounted" size to
    // not punish re-using subfunctions by only counting the subfunction nodes once.
    Array<u32> visited = Array<u32>::Zero(num_discrete);

    // to resolve subfunction arguments, we need to know the calling node
    // (this stack only increases, to avoid revisiting nodes and more importantly to not invalidate stack indices)
    std::vector<usize> call_stack;
    call_stack.reserve(max_expression_size);

    // for each we need to visit, we need the node index, the call stack idx and whether the node already was visited
    // (for functions the first time is in-order, and the second time is post-order)
    std::vector<std::tuple<usize, usize, bool>> node_stack;
    node_stack.reserve(max_expression_size);

    // for each output, walk the tree in post-order
    size = 0;  // initially the size is 0 (size in GP is somewhat arbitary - even without subftrees/args which are not counted, simplification typically also has an effect and it's not necessarily a good proxy for "interpretability" in the first place)
    for (usize n : output_roots) {
      // add and allocate for the output
      nodes.emplace_back();
      auto& tree = nodes.back();
      tree.reserve(max_expression_size - size);

      // housekeeping: reset the call stack and node_stack per output
      call_stack.clear();
      node_stack.clear();
      node_stack.emplace_back(n, 0, false);

      // nodes are visited up to twice - once in-order, and once post-order after the children have been taken care of
      while (!node_stack.empty()) {
        // we hit the max size, but have a next node since this is inside the loop
        if (size + tree.size() == max_expression_size) {
          return std::nullopt;
        }

        // get the node and mark it as active
        auto [idx, call_stack_idx, is_post_order] = node_stack.back();
        usize node_stack_idx = node_stack.size() - 1;

        if constexpr (!std::is_const<S>()) {
          solution.discrete_active()(idx) = true;
        }

        // lookup the value of the current node
        DType value = domain2value(idx, solution.discrete_values()(idx));
        usize v_idx = value_idx[value];

        // since this only a traversal without any evaluation, we only have to
        // check if this is an actual value or if we need to resolve arguments or other indirections
        bool is_leaf = false;

        // we only need to look at the node if this is the first time we see it - in the post-order visit all we have to
        // do is add it to the tree
        if (!is_post_order) {
          // In the reference modular GP-GOMEA implementation, the "discounted size" is defined
          // here:
          // https://github.com/matigekunstintelligentie/MultiGPG/blob/21094c016f93457df173935a1ec702568c6c2b24/src/individual.hpp#L98
          // and here:
          // https://github.com/matigekunstintelligentie/MultiGPG/blob/21094c016f93457df173935a1ec702568c6c2b24/src/node.hpp#L90
          // - in no case are references counted, i.e. the visit count for arg/fn nodes needs to be reset to 0 later on
          // track the size (repeat visits don't count in the "discounted" setting)
          if (discount_size) {
            visited(idx) = 1;
          }

          if (value_kind[value] == ValueKind::Arg) {
            visited(idx) = 0;

            // we need to replace the argument with the corresponding child of the caller
            usize calling_node = call_stack[call_stack_idx];
            auto& cnodes = children[calling_node];

            // and replace the stack entry with with the actual argument
            node_stack.pop_back();
            node_stack.emplace_back(cnodes[v_idx % cnodes.size()],
                                    call_stack_idx - 1,  // use the stack index of the caller
                                    false);
          } else if (value_kind[value] == ValueKind::Subtree) {
            visited(idx) = 0;
            // we need to replace the actual subtree with the called subtree

            // first update the call stack
            call_stack.push_back(idx);

            // then replace the stack entry with the called subtree
            node_stack.pop_back();
            node_stack.emplace_back(subtree_roots[v_idx],
                                    call_stack.size() - 1,  // a call always needs to use the top of the stack, no
                                                            // matter where the current call_stack_idx is (!)
                                    false);
          } else if (value_kind[value] == ValueKind::Operator) {
            // the operator stays on the stack, but the next visit is post-order
            std::get<2>(node_stack[node_stack_idx]) = true;

            // all the arguments need to be added (the stack reverses the order, so this is in reverse)
            usize arity = std::min(children[idx].size(), value_max_arity[value]);
            for (usize i = arity; i > 0;) {
              // the child's calling node/call_stack_idx is the same as the parents
              node_stack.emplace_back(children[idx][--i], call_stack_idx, false);
            }
          } else if (value_min_arity[value] > 0) {
            // for anything that is not a leaf, we need to resolve the arguments, so another branch is needed if
            // non-terminal kinds are added
            throw std::runtime_error("Encountered unhandled non-leaf node.");
          } else {
            if constexpr (!std::is_const<S>()) {
              if (value_kind[value] == ValueKind::Constant) {
                solution.continuous_active()(const_repr == ConstantRepr::Pool ? v_idx : idx) = true;
              }
            }
            is_leaf = true;
          }
        }

        // if this is a leaf or if this is the post-order visit, then we remove it from the stack and add it to the tree
        if (is_post_order || is_leaf) {
          // Indirections like Subtree/Arg calls are not kept, so only the constants for actual "values", not
          // "references" are used
          if (const_repr == ConstantRepr::Edges) {
            if constexpr (!std::is_const<S>()) {
              solution.continuous_active()(idx) = true;
            }
          }

          node_stack.pop_back();
          tree.push_back(idx);
        }
      }

      size += tree.size();
    }

    if (discount_size) {
      size = visited.sum();
    }

    return nodes;
  }

  template <typename Scalar>
  std::optional<Arr2D<Scalar>> compute_outputs(Arr2D<Scalar>& eval_buffer,
                                SolutionBase& solution,
                                const Arr2D<Scalar>& X,
                                const Array<Scalar>& params,
                                usize& size) const {
    // the expression is evaluated in two steps:
    // 1. the actual expression is extracted from the template in-order
    // 2. the operations are interpreted in reverse

    auto nodes = nodes_post_order(solution, /* TODO discount_size = */ true, size);

    // check if we even have to evaluate
    if (!nodes.has_value()) {
      return std::nullopt;
    }

    // ensure the buffer is allocated
    eval_buffer.resize(X.rows(), max_expression_size);

    // evaluation of postfix expressions assumes a stack model, i.e. results are pushed onto as stack, arguments
    // retrieved from the stack and at the end, the single stack entry is the result. Since arguments might be consist
    // of nested operations, the buffer indices corresponding to the actual results are needed somewhere.
    std::vector<usize> arg_stack;
    arg_stack.reserve(max_expression_size);

    // for each output, evaluate the tree
    Arr2D<Scalar> outputs(X.rows(), num_outputs);
    const auto trees = nodes.value();
    for (usize i = 0; i < trees.size(); i++) {
      const auto& tree = trees[i];

      // housekeeping: initially, there are no arguments
      arg_stack.clear();

      // the nodes are in postfix notation, so we evaluate from left to right
      for (usize j = 0; j < tree.size(); j++) {
        usize idx = tree[j];

        // lookup the value of the current node
        usize value = domain2value(idx, solution.discrete_values()(idx));
        usize v_idx = value_idx[value];

        // at this point, all references have been resolved
        assert(value_kind[value] != ValueKind::Arg);
        assert(value_kind[value] != ValueKind::Subtree);

        // resolve value lookups / function calls
        if (value_kind[value] == ValueKind::Input) {
          eval_buffer.col(j) = X.col(v_idx);
        } else if (value_kind[value] == ValueKind::Parameter) {
          eval_buffer.col(j) = params(v_idx);
        } else if (value_kind[value] == ValueKind::Constant) {
          usize ci = const_repr == ConstantRepr::Pool ? v_idx : idx;
          eval_buffer.col(j) = solution.continuous_values()(ci);
        } else if (value_kind[value] == ValueKind::Operator) {
          usize arity = std::min(children[idx].size(), value_max_arity[value]);

          // the arguments are in the correct order on the stack, so we just need to get the last arity indices on the arg_stack
          std::span<const usize> child_indices{arg_stack.end() - arity, arg_stack.end()};

          operators[v_idx]->apply(eval_buffer.col(j), eval_buffer(Eigen::placeholders::all, child_indices));

          // pop the now used arguments from the stack
          arg_stack.resize(arg_stack.size() - arity);
        } else {
          std::unreachable(); // if this triggers, either not all reference types have been removed or a non-reference value kind has been added...
        }

        if (const_repr == ConstantRepr::Edges) {
          eval_buffer.col(j) *= solution.continuous_values()(idx);
        }

        // since there are no more references, each node output is
        arg_stack.push_back(j);
      }

      // at the end the stack only contains the tree output, and that is at the buffer position for the last tree node
      assert(arg_stack.size() == 1);
      assert(arg_stack.back() == tree.size() - 1);
      outputs.col(i) = eval_buffer.col(tree.size() - 1);
    }

    return outputs;
  }

  template <typename Scalar>
  Arr2D<Scalar> compute_outputs3(Arr2D<Scalar>& eval_buffer,
                                 SolutionBase& solution,
                                 const Arr2D<Scalar>& X,
                                 const Array<Scalar>& params,
                                 usize& size) const {
    // the expression is evaluated in two steps:
    // 1. the actual expression is extracted from the template in-order
    // 2. the operations are interpreted in reverse

    // Node indices are used here to not loose the last bit of information we still need - how should we interpret the
    // node?
    bool size_overflow;
    auto [nodes, output_indices] = nodes_in_order(solution, /* TODO discount_size = */ false, size_overflow, size);

    // check if we even have to evaluate (but right now we still need )
    Arr2D<Scalar> outputs(X.rows(), num_outputs);
    if (size_overflow) {
      outputs.array() = std::numeric_limits<Scalar>::quiet_NaN();
      return outputs;
    }

    // ensure the buffer is allocated
    eval_buffer.resize(X.rows(), max_expression_size);

    std::vector<usize> arg_stack;
    arg_stack.reserve(nodes.size());

    // actually evaluate each active node in reverse (from leaf to root)
    for (usize i = nodes.size(), o = output_indices.size() - 1; i > 0;) {
      usize idx = nodes[--i];

      // lookup the value of the current node
      usize value = domain2value(idx, solution.discrete_values()(idx));
      usize v_idx = value_idx[value];

      // at this point, all references have been resolved
      assert(value_kind[value] != ValueKind::Arg);
      assert(value_kind[value] != ValueKind::Subtree);

      // then resolve value lookups / function calls
      if (value_kind[value] == ValueKind::Input) {
        eval_buffer.col(i) = X.col(v_idx);
        arg_stack.push_back(i);
      } else if (value_kind[value] == ValueKind::Parameter) {
        eval_buffer.col(i) = params(v_idx);
        arg_stack.push_back(i);
      } else if (value_kind[value] == ValueKind::Constant) {
        usize ci = const_repr == ConstantRepr::Pool ? v_idx : idx;
        eval_buffer.col(i) = solution.continuous_values()(ci);
        arg_stack.push_back(i);
      } else if (value_kind[value] == ValueKind::Operator) {
        usize arity = std::min(children[idx].size(), value_max_arity[value]);
        assert(arg_stack.size() >= arity);

        std::span<const usize> child_indices{arg_stack.end() - arity, arg_stack.end()};
        operators[v_idx]->apply(eval_buffer.col(i), eval_buffer(Eigen::placeholders::all, child_indices));

        // after we compute the function, we need to remove the arguments and add the result to the argument stack
        for (usize j = 0; j < arity; j++) {
          arg_stack.pop_back();
        }
        arg_stack.push_back(i);
      } else {
        std::unreachable();
      }

      if (const_repr == ConstantRepr::Edges) {
        eval_buffer.col(i) *= solution.continuous_values()(idx);
      }

      // Since all subtrees are in nodes, we need the output indices to know when a subexpression is done
      // If that's the case, we need to reset the arg stack and might as well already copy the output from the buffer
      if (i == output_indices[o]) {
        outputs.col(o--) = eval_buffer.col(i);
        arg_stack.clear();
      }
    }

    return outputs;
  };

  // // TODO allow gradients w.r.t. specific continuous indices OR parameter
  // // indices
  // template <typename Scalar>
  // Arr2D<Scalar> compute_outputs_grad(SolutionBase& solution, Arr2D<Scalar>& X, Array<Scalar>& params) const {
  //   std::unreachable();
  // };

  // // std::string to_dot(const SolutionBase &solution) const {
  // //   std::unreachable();
  // // };

  ConstantRepr const_repr;

  usize num_inputs;
  usize num_outputs;
  usize num_subexpressions;
  usize num_discrete;
  usize num_continuous;
  usize max_expression_size;
  usize num_parameters;
  usize max_num_children;

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
