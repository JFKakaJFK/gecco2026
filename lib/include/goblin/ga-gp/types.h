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

}

#endif /* _GOBLIN_GA_GP_TYPES_H */