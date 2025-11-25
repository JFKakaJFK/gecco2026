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

  inline std::optional<DType> value2domain(usize index, DType value) const {
    auto dval = _value2domain(index, value);
    if (dval < domain_sizes[index]) {
      return dval;
    }
    return std::nullopt;
  };

  inline std::optional<usize> parent(usize index) const {
    auto p_idx = _parent[index];
    if (p_idx < num_discrete) {
      return p_idx;
    }
    return std::nullopt;
  };

  // A helper that prints the expression in a human readable format
  void debug_log_expressions(std::ostream& os,
                             const SolutionBase& solution,
                             std::optional<usize> node = std::nullopt,
                             std::string indent = "") const {
    // abuse default parameters to not have to declare a helper...
    if (node.has_value()) {  // a node
      usize idx = node.value();

      for (usize i = 0; i < indent.size(); i++) {
        indent[i] = ' ';
      }
      indent += "└─ ";

      // lookup the value of the current node
      DType value = domain2value(idx, solution.discrete_values()(idx));
      usize v_idx = value_idx[value];

      os << indent << idx << " (V = " << static_cast<usize>(value) << ", "
         << (solution.discrete_active()(idx) ? "active" : "inactive") << "): ";

      if (value_kind[value] == ValueKind::Input) {
        os << "Input[" << v_idx << "]\n";
      } else if (value_kind[value] == ValueKind::Constant) {
        usize ci = const_repr == ConstantRepr::Pool ? v_idx : idx;
        os << "Constant[" << v_idx << "] -> " << solution.continuous_values()(ci) << "\n";
      } else if (value_kind[value] == ValueKind::Parameter) {
        os << "Parameter[" << v_idx << "]\n";
      } else if (value_kind[value] == ValueKind::Arg) {
        os << "Arg[" << v_idx << "]\n";
      } else if (value_kind[value] == ValueKind::Subtree) {
        os << "Fn[" << v_idx << "] -> node " << subtree_roots[v_idx] << '\n';
      } else if (value_kind[value] == ValueKind::Operator) {
        os << "Op[" << v_idx << "] (arity = " << std::min(children[idx].size(), value_max_arity[value]) << ")\n";
      } else {
        std::unreachable();
      }

      for (usize c : children[idx]) {
        debug_log_expressions(os, solution, c, std::string{indent});
      }
    } else {  // the base case - just print all subtrees and outputs
      for (usize n : subtree_roots) {
        os << "Subtree\n";
        debug_log_expressions(os, solution, n, /* indent = */ "");
      }
      for (usize n : output_roots) {
        os << "Output\n";
        debug_log_expressions(os, solution, n, /* indent = */ "");
      }
      os << std::flush;
    }
  }

  // Returns all trees in postfix/reverse polish notation (https://en.wikipedia.org/wiki/Reverse_Polish_notation) and
  // without references if the total number of nodes exceeds the `max_expression_size` or `std::nullopt` otherwise.
  //
  // The variables are marked active/inactive only if the solution is not const.
  // Note that the reason why it's `nodes_post_order` and not `values_post_order` is that for constants/operator calls
  // the corresponding index/arity isn't fully determined by the value alone.
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
    // (and if that is another argument, we need the calling node of that tree and so on...)
    std::vector<usize> call_stack;
    call_stack.reserve(max_expression_size);

    // for each we need to visit, we need the node index, the call stack idx and whether the node already was visited
    // (for functions the first time is in-order, and the second time is post-order)
    std::vector<std::tuple<usize, isize, bool>> node_stack;
    node_stack.reserve(max_expression_size);

    // for each output, walk the tree in post-order
    size = 0;  // initially the size is 0 (size in GP is somewhat arbitary - even without subftrees/args which are not
               // counted, simplification typically also has an effect and it's not necessarily a good proxy for
               // "interpretability" in the first place)
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
        bool update_tree = false;

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
            // and then replace the stack entry with with the actual argument
            //
            // but the caller might need some resolving if it is not the root of a tree
            // (every function call adds to the call_stack, so it is not necessarily true that the previous call stack
            // entry corresponds to the caller of the current subtree - it might just be an ancestor in the current tree
            // that also calls another subfunction...)

            // get the previous call stack entry
            usize calling_node = call_stack[call_stack_idx];

            // since we move up the call chain, we need to go at least one call/frame backward, but maybe more
            isize num_frames = 1;

            // check if the calling node is an ancestor of the current node - if so, we need to move up the hierarchy to
            // find the ancestor clostest to the root that is on the call_stack... (the first call into this subtree
            // must have the actual caller)
            auto pidx = parent(idx);
            while (pidx.has_value()) {
              if (pidx.value() == calling_node) {
                // note: this works since we don't allow cycles by restricting the domain, i.e. there can only ever be
                // one "active" call to this subfunction, guaranteeing that the calling node of the highest ancestor is
                // the actual calling node.
                calling_node = call_stack[call_stack_idx - num_frames++];
              }
              pidx = parent(pidx.value());
            }

            // now that we have the caller, we can finally replace the stack entry with with the actual argument
            auto& cnodes = children[calling_node];
            assert(idx != cnodes[v_idx % cnodes.size()]);  // no self references

            node_stack.pop_back();
            node_stack.emplace_back(cnodes[v_idx % cnodes.size()],
                                    call_stack_idx - num_frames,  // use the stack index of the (resolved) caller
                                    false);
          } else if (value_kind[value] == ValueKind::Subtree) {
            visited(idx) = 0;
            assert(root[idx] != subtree_roots[v_idx]);  // no loops allowed
            // we need to replace the actual subtree with the called subtree

            // first update the call stack
            call_stack.push_back(idx);

            // then replace the stack entry with the called subtree
            std::get<2>(node_stack[node_stack_idx]) =
                true;  // this is a reference type, but we need the post-order visit to keep the call stack in sync
            node_stack.emplace_back(
                subtree_roots[v_idx],
                call_stack.size() - 1,  // a call always needs to use the top of the stack, no
                                        // matter where the current call_stack_idx is (!there might be a chain of calls
                                        // between the root containing the actual caller of this node!)
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

            // this is a leaf, so the in-order is the post-order visit
            update_tree = true;
            node_stack.pop_back();  // this is the post-order visit, so no need to visit again
          }
        } else {
            if (value_kind[value] == ValueKind::Subtree) {
          // in the previous in-order visit, this node was pushed on the call stack so it has to be removed now
          call_stack.pop_back();
        } else {
          // this is a non-reference post-order visit, so we need to update the tree
          update_tree = true;
        }

            // remove post order nodes from the stack
            node_stack.pop_back();
        }

        // finally, if this is a leaf or if this is a non-reference post-order visit, then we add it to the tree
        if (update_tree) {
          // Indirections like Subtree/Arg calls are not kept, so only the constants for actual "values", not
          // "references" are used
          if (const_repr == ConstantRepr::Edges) {
            if constexpr (!std::is_const<S>()) {
              solution.continuous_active()(idx) = true;
            }
          }

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

  std::vector<std::string> to_sympy(const SolutionBase& solution) const {
    // extract the expression in post-order
    usize _size;
    auto nodes = nodes_post_order(solution, false, _size);

    std::vector<std::string> exprs(num_outputs, "SIZE OVERFLOW");

    // check if we are done already
    if (!nodes.has_value()) {
      return exprs;
    }

    // keep a stack for the arguments
    std::vector<std::string> arg_stack;
    arg_stack.reserve(max_expression_size);

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
          arg_stack.push_back(std::format("x{:d}", v_idx));
        } else if (value_kind[value] == ValueKind::Parameter) {
          arg_stack.push_back(std::format("c{:d}", v_idx));
        } else if (value_kind[value] == ValueKind::Constant) {
          usize ci = const_repr == ConstantRepr::Pool ? v_idx : idx;
          arg_stack.push_back(std::format("{}", solution.continuous_values()(ci)));
        } else if (value_kind[value] == ValueKind::Operator) {
          usize arity = std::min(children[idx].size(), value_max_arity[value]);
          usize arg_stack_idx = arg_stack.size() - arity;

          // the arguments are in the correct order on the stack, so we just need to get the last arity indices on the
          // arg_stack
          std::span<const std::string> args{arg_stack.end() - arity, arg_stack.end()};
          arg_stack[arg_stack_idx] = operators[v_idx]->format(args);

          // pop the now used arguments from the stack, but keep the op result
          arg_stack.resize(arg_stack_idx + 1);
        } else {
          std::unreachable();  // if this triggers, either not all reference types have been removed or a non-reference
                               // value kind has been added...
        }

        if (const_repr == ConstantRepr::Edges) {
          arg_stack.back() = std::format("({} * ({}))", solution.continuous_values()(idx), arg_stack.back());
        }
      }

      // at the end the stack only contains the tree output
      exprs[i] = arg_stack.back();
    }

    return exprs;
  };

  template <typename Scalar>
  std::optional<Arr2D<Scalar>> compute_outputs(SolutionBase& solution,
                                               const Arr2D<Scalar>& X,
                                               const Array<Scalar>& params) const {
    Arr2D<Scalar> eval_buffer;
    usize size;
    return compute_outputs(eval_buffer, solution, X, params, size);
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

          // the arguments are in the correct order on the stack, so we just need to get the last arity indices on the
          // arg_stack
          std::span<const usize> child_indices{arg_stack.end() - arity, arg_stack.end()};

          operators[v_idx]->apply(eval_buffer.col(j), eval_buffer(Eigen::placeholders::all, child_indices));

          // pop the now used arguments from the stack
          arg_stack.resize(arg_stack.size() - arity);
        } else {
          std::unreachable();  // if this triggers, either not all reference types have been removed or a non-reference
                               // value kind has been added...
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

  void to_gpu_repr(SolutionBase& solution, std::vector<u8>& node_type, std::vector<float>& node_value) const {
    // TODO implement multi-output (multiple trees per solution) parsing

    std::vector<usize> stack;

    // Vectors to hold temporary type and value data
    std::vector<u8> temp_type;
    std::vector<float> temp_value;

    // Push root node on stack
    stack.push_back(output_roots[0]);

    while (!stack.empty()) {
      // Pop the top node from the stack
      usize node = stack.back();
      stack.pop_back();

      // Get the type and value for the current node
      DType domain_value = domain2value(node, solution.discrete_values()(node));
      usize v_idx = value_idx[domain_value];
      enum ValueKind type = value_kind[domain_value];

      temp_type.push_back(static_cast<u8>(type));  

     if (type == ValueKind::Input) {
        // Push the index of the input feature, will be used to access the input matrix on GPU
        temp_value.push_back(v_idx);
      } else if (type == ValueKind::Parameter) {
        // Push the index of the parameter, will be used to access the parameter array on GPU
        temp_value.push_back(v_idx);
      } else if (type == ValueKind::Constant) {
        usize ci = const_repr == ConstantRepr::Pool ? v_idx : node;
        // Push the constant value, will be used directly in the evaluation on GPU
        temp_value.push_back(static_cast<float>(solution.continuous_values()(ci)));
      } else if (type == ValueKind::Arg) {
        // TODO implement arg handling
        continue;
      } else if (type == ValueKind::Subtree) {
        // TODO implement subtree handling
        continue;
      } else if (type == ValueKind::Operator) {
        // Push the operator index, will be used to apply the operator on GPU
        temp_value.push_back(static_cast<float>(v_idx));

        // Push the children of the current node onto the stack
        usize arity = std::min(children[node].size(), operators[v_idx]->max_arity());
        for (usize j = arity; j > 0; j--) {
          stack.push_back(children[node][j - 1]);
        }
      }
    } 

    // Reverse the vectors to allow forward iteration on the GPU
    std::reverse(temp_type.begin(), temp_type.end());
    std::reverse(temp_value.begin(), temp_value.end());

    // Pad vectors with placeholder values such that the solutions are at constant intervals in memory
    // TODO investigate coalesced memory access
    temp_type.resize(num_discrete, std::numeric_limits<u8>::max());
    temp_value.resize(num_discrete, std::numeric_limits<float>::max());

    // Append temporary vectors to final vectors
    node_type.insert(node_type.end(), temp_type.begin(), temp_type.end());
    node_value.insert(node_value.end(), temp_value.begin(), temp_value.end());
  }

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
