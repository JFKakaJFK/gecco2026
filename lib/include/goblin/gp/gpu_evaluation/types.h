#pragma once
#ifndef _GOBLIN_GA_GP_TYPES_H
#define _GOBLIN_GA_GP_TYPES_H

#include <cstdint>
#include <string_view>

namespace goblin {

using u8 = std::uint8_t;

enum class KernelVersion : u8 {
    Baseline,
    Restrict,
    SharedMemory,
    BlockReduce,
    SingleKernel,
    SingleKernelFMAF,
    SingleKernelInplace,
    Hybrid
};

constexpr std::string_view to_string(KernelVersion v) {
    switch (v) {
        case KernelVersion::Baseline:            return "Baseline";
        case KernelVersion::Restrict:            return "Restrict";
        case KernelVersion::SharedMemory:        return "SharedMemory";
        case KernelVersion::BlockReduce:         return "BlockReduce";
        case KernelVersion::SingleKernel:        return "SingleKernel";
        case KernelVersion::SingleKernelFMAF:    return "SingleKernelFMAF";
        case KernelVersion::SingleKernelInplace: return "SingleKernelInPlace";
        case KernelVersion::Hybrid:              return "Hybrid";
    }

    return "Unknown KernelVersion";
}

enum class NodeType : u8 {
    Input,
    Constant,
    Operator,
    Parameter
};

enum class Operator : u8 {
    Add,
    Sub,
    Mul,
    Div,
    Sin,
    Cos,
    Exp,
    Log,
    Square,
    Sqrt,
    Pow,
    Abs,
    Min,
    Max
};

// The following declarations are used to create more readable test cases
namespace test {
    constexpr u8 C = static_cast<u8>(NodeType::Constant);
    constexpr u8 I = static_cast<u8>(NodeType::Input);
    constexpr u8 O = static_cast<u8>(NodeType::Operator);

    constexpr float Val(float x) { return x; }
    constexpr float Val(int x) { return static_cast<float>(x); }
    constexpr float Val(double x) { return static_cast<float>(x); }
    constexpr float Idx(int idx) { return static_cast<float>(idx); }
    constexpr float Op(Operator op) { return static_cast<float>(op); }

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
}

}

#endif /* _GOBLIN_GA_GP_TYPES_H */