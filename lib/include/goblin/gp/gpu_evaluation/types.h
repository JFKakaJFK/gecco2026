#pragma once
#ifndef _GOBLIN_GA_GP_TYPES_H
#define _GOBLIN_GA_GP_TYPES_H

#include <cstdint>
#include <string_view>

namespace goblin {

using u8 = std::uint8_t;

// SingleBlock: one block per solution; threads partition the datapoints within that block.
// DynamicBlock: multiple blocks per solution; improves SM occupancy for small populations.
enum class KernelVersion : u8 { SingleBlock, DynamicBlock };

constexpr std::string_view to_string(KernelVersion v) {
    switch (v) {
        case KernelVersion::SingleBlock:
            return "SingleBlock";
        case KernelVersion::DynamicBlock:
            return "DynamicBlock";
    }

    return "Unknown KernelVersion";
}

// Trees are encoded as parallel arrays (type[], value[]) in postfix order.
// Input: value holds the feature index.
// Constant: value holds the literal constant.
// Operator: value is the Operator enum cast to float.
enum class NodeType : u8 { Input, Constant, Operator, Parameter };

enum class Operator : u8 { Add, Sub, Mul, Div, Sin, Cos, Exp, Log, Square, Sqrt, Pow, Abs, Min, Max };

// Helpers for writing readable test trees using the same float encoding as the runtime.
namespace test {
constexpr u8 C = static_cast<u8>(NodeType::Constant);
constexpr u8 I = static_cast<u8>(NodeType::Input);
constexpr u8 O = static_cast<u8>(NodeType::Operator);

constexpr float Val(float x) {
    return x;
}
constexpr float Val(int x) {
    return static_cast<float>(x);
}
constexpr float Val(double x) {
    return static_cast<float>(x);
}
constexpr float Idx(int idx) {
    return static_cast<float>(idx);
}
constexpr float Op(Operator op) {
    return static_cast<float>(op);
}

constexpr float Add = Op(Operator::Add);
constexpr float Sub = Op(Operator::Sub);
constexpr float Mul = Op(Operator::Mul);
constexpr float Div = Op(Operator::Div);
constexpr float Sin = Op(Operator::Sin);
constexpr float Cos = Op(Operator::Cos);
constexpr float Exp = Op(Operator::Exp);
constexpr float Log = Op(Operator::Log);
constexpr float Square = Op(Operator::Square);
constexpr float Sqrt = Op(Operator::Sqrt);
constexpr float Pow = Op(Operator::Pow);
constexpr float Abs = Op(Operator::Abs);
constexpr float Min = Op(Operator::Min);
constexpr float Max = Op(Operator::Max);
}  // namespace test

}  // namespace goblin

#endif /* _GOBLIN_GA_GP_TYPES_H */
