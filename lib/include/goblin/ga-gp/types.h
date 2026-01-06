#pragma once
#ifndef _GOBLIN_GA_GP_TYPES_H
#define _GOBLIN_GA_GP_TYPES_H

#include <cstdint>

namespace goblin {

enum class KernelVersion : uint8_t {
    Baseline,
    Restrict,
    SharedMemory,
    BlockReduce,
    SingleKernel,
    SingleKernelFMAF,
    SingleKernelInplace
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
    }

    return "Unknown KernelVersion";
}

enum class NodeType : uint8_t {
    Input,
    Constant,
    Operator,
};

enum class Operator : uint8_t {
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
constexpr float C = static_cast<float>(NodeType::Constant);
constexpr float I = static_cast<float>(NodeType::Input);
constexpr float O = static_cast<float>(NodeType::Operator);

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

#endif /* _GOBLIN_GA_GP_TYPES_H */