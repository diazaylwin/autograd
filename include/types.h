#ifndef TYPES_H
#define TYPES_H

#include <cstdint>

using StorageID = uint32_t;   // where numbers live
using ValueID   = uint32_t;   // what is computed (symbolic variable)

enum class OpTag : uint8_t 
{
    Const,
    Add,
    Mul,
    Div,
    Neg,
    COUNT,
    ZeroLike,   // NEW
    // later: Reshape, Slice, Transpose, ReduceSum, Matmul, ...
};

#endif // TYPES_H