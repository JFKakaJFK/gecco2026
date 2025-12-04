#pragma once
#ifndef _GOBLIN_GA_GP_TYPES_H
#define _GOBLIN_GA_GP_TYPES_H

#include <cstdint>

namespace goblin {

enum class NodeType : u_int8_t {
    Input,
    Constant,
    Operator,
};

enum class Operator : uint8_t {
    Add,
    Sub,
    Mul,
    Div
};

constexpr float C = static_cast<float>(NodeType::Constant);
constexpr float I = static_cast<float>(NodeType::Input);
constexpr float O = static_cast<float>(NodeType::Operator);

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