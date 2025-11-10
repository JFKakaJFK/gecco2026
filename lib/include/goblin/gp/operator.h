#ifndef _GOBLIN_GP_OPERATOR_H
#define _GOBLIN_GP_OPERATOR_H

#pragma once

#include <cassert>
#include <format>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <variant>
#include <vector>

#include "goblin/gp/template.h"
#include "goblin/lib/assert.h"
#include "goblin/lib/types.h"

namespace goblin {
// TODO multi-arity (not here), but allow nodes to not use all children but a
// chosen subset (=partial permutation)
// TODO types - every column (input and output Arr2Drices) now has a type etc...
// enum struct Type: unsigned char {
//     Bool = 0,
//     Float = 1
// };
// struct Signature {
//     std::vector<Type> args;
//     Type rty;
// };
// using CColRef = std::variant<
//     CRefS<Array<BType>>,
//     CRefS<Array<CType>>
// >;
// using ColRef = std::variant<
//     RefS<Array<BType>>,
//     RefS<Array<CType>>
// >;
// template<typename C>
// inline constexpr Type typeof(const C& col){
//     if (std::same_as<typename C::Scalar, CType>){
//         return Type::Float;
//     } else if constexpr (std::same_as<typename C::Scalar, BType>){
//         return Type::Bool;
//     } else {
//         std::unreachable();
//     }
// };

// The operators and their derivatives were more or less copied over from
// https://github.com/matigekunstintelligentie/MultiGPG/blob/main/src/operator.hpp

class OperatorBase {
 public:
  virtual usize min_arity() const = 0;
  virtual usize max_arity() const = 0;

  virtual bool is_commutative() const = 0;

  virtual void apply(Ref<Array<CType>> out, CRef<Arr2D<CType>> args) const = 0;

  virtual bool has_gradient() const { return false; };
  virtual void apply_grad(Ref<Array<CType>> out,
                          Ref<Array<CType>> d_out,
                          CRef<Arr2D<CType>> args,
                          CRef<Arr2D<CType>> d_args) const {
    throw std::runtime_error("Gradients not supported.");
  };

  Array<CType> operator()(CRef<Arr2D<CType>> args) const {
    Array<CType> out(args.rows());
    apply(out, args);
    return out;
  };

  virtual std::string format(const std::vector<std::string>& args) const = 0;

  virtual ~OperatorBase() = default;
};

class OpAdd : public OperatorBase {
 public:
  usize min_arity() const override final { return 2; };
  usize max_arity() const override final { return std::numeric_limits<usize>::max(); };

  bool is_commutative() const override final { return true; };

  void apply(Ref<Array<CType>> out, CRef<Arr2D<CType>> args) const override final { out = args.rowwise().sum(); };

  bool has_gradient() const override final { return true; };
  void apply_grad(Ref<Array<CType>> out,
                  Ref<Array<CType>> d_out,
                  CRef<Arr2D<CType>> args,
                  CRef<Arr2D<CType>> d_args) const override final {
    apply(out, args);
    d_out = d_args.rowwise().sum();
  };

  std::string format(const std::vector<std::string>& args) const override final {
    std::ostringstream ss;
    ss << '(';
    for (usize i = 0; i < args.size(); i++) {
      if (i > 0) {
        ss << " + ";
      }
      ss << args[i];
    }
    ss << ')';
    return ss.str();
  };
};

class OpSub : public OperatorBase {
 public:
  usize min_arity() const override final { return 1; };
  usize max_arity() const override final { return std::numeric_limits<usize>::max(); };

  bool is_commutative() const override final {
    // well actually: all arguments after the first one are interchangeable
    return false;
  };

  void apply(Ref<Array<CType>> out, CRef<Arr2D<CType>> args) const override final {
    if (args.cols() > 1) {
      out = args.col(0) - args(Eigen::placeholders::all, Eigen::seqN(1, args.cols() - 1)).rowwise().sum();
    } else {
      out = -args.col(0);
    }
  };

  bool has_gradient() const override final { return true; };
  void apply_grad(Ref<Array<CType>> out,
                  Ref<Array<CType>> d_out,
                  CRef<Arr2D<CType>> args,
                  CRef<Arr2D<CType>> d_args) const override final {
    apply(out, args);
    if (args.cols() > 1) {
      d_out = d_args.col(0) - d_args(Eigen::placeholders::all, Eigen::seqN(1, d_args.cols() - 1)).rowwise().sum();
    } else {
      d_out = -d_args.col(0);
    }
  };

  std::string format(const std::vector<std::string>& args) const override final {
    std::ostringstream ss;
    if (args.size() == 1) {
      ss << "(-" << args[0] << ')';
    } else {
      ss << '(';
      for (usize i = 0; i < args.size(); i++) {
        if (i > 0) {
          ss << " - ";
        }
        ss << args[i];
      }
      ss << ')';
    }
    return ss.str();
  };
};

class OpMul : public OperatorBase {
 public:
  usize min_arity() const override final { return 2; };
  usize max_arity() const override final { return std::numeric_limits<usize>::max(); };

  bool is_commutative() const override final { return true; };

  void apply(Ref<Array<CType>> out, CRef<Arr2D<CType>> args) const override final { out = args.rowwise().prod(); };

  bool has_gradient() const override final { return true; };
  void apply_grad(Ref<Array<CType>> out,
                  Ref<Array<CType>> d_out,
                  CRef<Arr2D<CType>> args,
                  CRef<Arr2D<CType>> d_args) const override final {
    assert(d_args.cols() >= 2);
    apply(out, args);

    // sum(df_i * prod(f_{j!=i}))
    d_out = d_args.col(0) * args(Eigen::placeholders::all, Eigen::seq(1, args.cols() - 1)).rowwise().prod();

    for (usize i = 1; i < d_args.cols() - 1; i++) {
      d_out += args(Eigen::placeholders::all, Eigen::seq(0, i - 1)).rowwise().prod() * d_args.col(i) *
               args(Eigen::placeholders::all, Eigen::seq(i + 1, args.cols() - 1)).rowwise().prod();
    }
    d_out +=
        args(Eigen::placeholders::all, Eigen::seq(0, args.cols() - 2)).rowwise().prod() * d_args.col(d_args.cols() - 1);
  };

  std::string format(const std::vector<std::string>& args) const override final {
    std::ostringstream ss;
    ss << '(';
    for (usize i = 0; i < args.size(); i++) {
      if (i > 0) {
        ss << " * ";
      }
      ss << args[i];
    }
    ss << ')';
    return ss.str();
  };
};

class OpDiv : public OperatorBase {
 public:
  usize min_arity() const override final { return 2; };
  usize max_arity() const override final { return std::numeric_limits<usize>::max(); };

  bool is_commutative() const override final {
    // first argument is not commutative
    return false;
  };

  void apply(Ref<Array<CType>> out, CRef<Arr2D<CType>> args) const override final {
    out = args.col(0) / args(Eigen::placeholders::all, Eigen::seq(1, args.cols() - 1)).rowwise().prod();
  };

  bool has_gradient() const override final { return true; };
  void apply_grad(Ref<Array<CType>> out,
                  Ref<Array<CType>> d_out,
                  CRef<Arr2D<CType>> args,
                  CRef<Arr2D<CType>> d_args) const override final {
    assert(d_args.cols() >= 2);

    auto denom = args(Eigen::placeholders::all, Eigen::seq(1, args.cols() - 1)).rowwise().prod();
    out = args.col(0) / denom;

    d_out = d_args.col(0) / denom;
    for (usize i = 1; i < args.cols(); i++) {
      d_out -= d_args.col(i) * out / args.col(i);
    }
  };

  std::string format(const std::vector<std::string>& args) const override final {
    std::ostringstream ss;
    ss << '(' << args[0] << '/';
    if (args.size() > 2) {
      ss << '(';
      for (usize i = 1; i < args.size(); i++) {
        if (i > 1) {
          ss << " * ";
        }
        ss << args[i];
      }
      ss << ')';
    } else {
      ss << args[1];
    }
    ss << ')';
    return ss.str();
  };
};

class OpSin : public OperatorBase {
 public:
  usize min_arity() const override final { return 1; };
  usize max_arity() const override final { return 1; };

  bool is_commutative() const override final { return false; };

  void apply(Ref<Array<CType>> out, CRef<Arr2D<CType>> args) const override final { out = args.col(0).sin(); };

  bool has_gradient() const override final { return true; };
  void apply_grad(Ref<Array<CType>> out,
                  Ref<Array<CType>> d_out,
                  CRef<Arr2D<CType>> args,
                  CRef<Arr2D<CType>> d_args) const override final {
    apply(out, args);

    d_out = args.col(0).cos() * d_args.col(0);
  };

  std::string format(const std::vector<std::string>& args) const override final {
    return std::format("sin({})", args[0]);
  };
};

class OpCos : public OperatorBase {
 public:
  usize min_arity() const override final { return 1; };
  usize max_arity() const override final { return 1; };

  bool is_commutative() const override final { return false; };

  void apply(Ref<Array<CType>> out, CRef<Arr2D<CType>> args) const override final { out = args.col(0).cos(); };

  bool has_gradient() const override final { return true; };
  void apply_grad(Ref<Array<CType>> out,
                  Ref<Array<CType>> d_out,
                  CRef<Arr2D<CType>> args,
                  CRef<Arr2D<CType>> d_args) const override final {
    apply(out, args);

    d_out = -args.col(0).sin() * d_args.col(0);
  };

  std::string format(const std::vector<std::string>& args) const override final {
    return std::format("cos({})", args[0]);
  };
};

class OpExp : public OperatorBase {
 public:
  usize min_arity() const override final { return 1; };
  usize max_arity() const override final { return 1; };

  bool is_commutative() const override final { return false; };

  void apply(Ref<Array<CType>> out, CRef<Arr2D<CType>> args) const override final { out = args.col(0).exp(); };

  bool has_gradient() const override final { return true; };
  void apply_grad(Ref<Array<CType>> out,
                  Ref<Array<CType>> d_out,
                  CRef<Arr2D<CType>> args,
                  CRef<Arr2D<CType>> d_args) const override final {
    apply(out, args);

    d_out = out * d_args.col(0);
  };

  std::string format(const std::vector<std::string>& args) const override final {
    return std::format("exp({})", args[0]);
  };
};

class OpLog : public OperatorBase {
 public:
  usize min_arity() const override final { return 1; };
  usize max_arity() const override final { return 1; };

  bool is_commutative() const override final { return false; };

  void apply(Ref<Array<CType>> out, CRef<Arr2D<CType>> args) const override final { out = args.col(0).log(); };

  bool has_gradient() const override final { return true; };
  void apply_grad(Ref<Array<CType>> out,
                  Ref<Array<CType>> d_out,
                  CRef<Arr2D<CType>> args,
                  CRef<Arr2D<CType>> d_args) const override final {
    apply(out, args);

    d_out = d_args.col(0) / args.col(0);
  };

  std::string format(const std::vector<std::string>& args) const override final {
    return std::format("log({})", args[0]);
  };
};

class OpSquare : public OperatorBase {
 public:
  usize min_arity() const override final { return 1; };
  usize max_arity() const override final { return 1; };

  bool is_commutative() const override final { return false; };

  void apply(Ref<Array<CType>> out, CRef<Arr2D<CType>> args) const override final { out = args.col(0).square(); };

  bool has_gradient() const override final { return true; };
  void apply_grad(Ref<Array<CType>> out,
                  Ref<Array<CType>> d_out,
                  CRef<Arr2D<CType>> args,
                  CRef<Arr2D<CType>> d_args) const override final {
    apply(out, args);

    d_out = CType(2.0) * args.col(0) * d_args.col(0);
  };

  std::string format(const std::vector<std::string>& args) const override final {
    return std::format("pow({}, 2)", args[0]);
  };
};

class OpSqrt : public OperatorBase {
 public:
  usize min_arity() const override final { return 1; };
  usize max_arity() const override final { return 1; };

  bool is_commutative() const override final { return false; };

  void apply(Ref<Array<CType>> out, CRef<Arr2D<CType>> args) const override final { out = args.col(0).sqrt(); };

  bool has_gradient() const override final { return true; };
  void apply_grad(Ref<Array<CType>> out,
                  Ref<Array<CType>> d_out,
                  CRef<Arr2D<CType>> args,
                  CRef<Arr2D<CType>> d_args) const override final {
    apply(out, args);

    d_out = d_args.col(0) / (out + out);
  };

  std::string format(const std::vector<std::string>& args) const override final {
    return std::format("sqrt({})", args[0]);
  };
};

class OpPow : public OperatorBase {
 public:
  usize min_arity() const override final { return 2; };
  usize max_arity() const override final { return 2; };

  bool is_commutative() const override final { return false; };

  void apply(Ref<Array<CType>> out, CRef<Arr2D<CType>> args) const override final {
    out = args.col(0).pow(args.col(1));
  };

  bool has_gradient() const override final { return true; };
  void apply_grad(Ref<Array<CType>> out,
                  Ref<Array<CType>> d_out,
                  CRef<Arr2D<CType>> args,
                  CRef<Arr2D<CType>> d_args) const override final {
    apply(out, args);

    d_out = args.col(0).pow(args.col(1) - CType(1.0)) *
            (args.col(0) * d_args.col(1) * args.col(0).log() + args.col(1) * d_args.col(0));
  };

  std::string format(const std::vector<std::string>& args) const override final {
    return std::format("pow({}, {})", args[0], args[1]);
  };
};

class OpAbs : public OperatorBase {
 public:
  usize min_arity() const override final { return 1; };
  usize max_arity() const override final { return 1; };

  bool is_commutative() const override final { return false; };

  void apply(Ref<Array<CType>> out, CRef<Arr2D<CType>> args) const override final { out = args.col(0).abs(); };

  bool has_gradient() const override final { return true; };
  void apply_grad(Ref<Array<CType>> out,
                  Ref<Array<CType>> d_out,
                  CRef<Arr2D<CType>> args,
                  CRef<Arr2D<CType>> d_args) const override final {
    apply(out, args);

    d_out = args.col(0) * d_args.col(0) / out;
  };

  std::string format(const std::vector<std::string>& args) const override final {
    return std::format("abs({})", args[0]);
  };
};

class OpMin : public OperatorBase {
 public:
  usize min_arity() const override final { return 2; };
  usize max_arity() const override final { return std::numeric_limits<usize>::max(); };

  bool is_commutative() const override final { return true; };

  void apply(Ref<Array<CType>> out, CRef<Arr2D<CType>> args) const override final { out = args.rowwise().minCoeff(); };

  bool has_gradient() const override final { return true; };
  void apply_grad(Ref<Array<CType>> out,
                  Ref<Array<CType>> d_out,
                  CRef<Arr2D<CType>> args,
                  CRef<Arr2D<CType>> d_args) const override final {
    for (usize i = 0; i < args.rows(); i++) {
      usize arg_min;
      out(i) = args.row(i).minCoeff(&arg_min);
      d_out(i) = d_args(i, arg_min);
    }
  };

  std::string format(const std::vector<std::string>& args) const override final {
    std::ostringstream ss;
    ss << "min(";
    for (usize i = 0; i < args.size(); i++) {
      if (i > 0) {
        ss << ", ";
      }
      ss << args[i];
    }
    ss << ')';
    return ss.str();
  };
};

class OpMax : public OperatorBase {
 public:
  usize min_arity() const override final { return 2; };
  usize max_arity() const override final { return std::numeric_limits<usize>::max(); };

  bool is_commutative() const override final { return true; };

  void apply(Ref<Array<CType>> out, CRef<Arr2D<CType>> args) const override final { out = args.rowwise().maxCoeff(); };

  bool has_gradient() const override final { return true; };
  void apply_grad(Ref<Array<CType>> out,
                  Ref<Array<CType>> d_out,
                  CRef<Arr2D<CType>> args,
                  CRef<Arr2D<CType>> d_args) const override final {
    for (usize i = 0; i < args.rows(); i++) {
      usize arg_max;
      out(i) = args.row(i).maxCoeff(&arg_max);
      d_out(i) = d_args(i, arg_max);
    }
  };

  std::string format(const std::vector<std::string>& args) const override final {
    std::ostringstream ss;
    ss << "max(";
    for (usize i = 0; i < args.size(); i++) {
      if (i > 0) {
        ss << ", ";
      }
      ss << args[i];
    }
    ss << ')';
    return ss.str();
  };
};
};  // namespace goblin

#endif /* _GOBLIN_GP_OPERATOR_H */
