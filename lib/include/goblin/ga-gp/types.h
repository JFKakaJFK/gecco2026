#pragma once
#ifndef _GOBLIN_GA_GP_TYPES_H
#define _GOBLIN_GA_GP_TYPES_H

namespace goblin {

enum class NodeType : u_int8_t {
    Input,
    Constant,
    Operator,
};

enum class Operator : u_int8_t {
    Add,
    Sub,
    Mul,
    Div
};

constexpr NodeType C = NodeType::Constant;
constexpr NodeType I = NodeType::Input;
constexpr NodeType O = NodeType::Operator;

constexpr float Val(float x) { return x; }
constexpr float Val(int x) { return static_cast<float>(x); }
constexpr float Idx(int idx) { return static_cast<float>(idx); }
constexpr float Op(Operator op) { return static_cast<float>(op); }

constexpr float Add = Op(Operator::Add);
constexpr float Sub = Op(Operator::Sub);
constexpr float Mul = Op(Operator::Mul);
constexpr float Div = Op(Operator::Div);

}

#endif /* _GOBLIN_GA_GP_TYPES_H */