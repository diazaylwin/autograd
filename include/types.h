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
    Exp,
    Log,
    Sum,        // reduction: sum all elements
    Expand,     // broadcast scalar a to shape of b
    ZeroLike,
    Detach,
    Scan,
    ScanVJP,
    Call,       // library function call
    Custom,     // user-defined op with custom forward/backward
    CustomVJP,  // backward pass for custom op
    COUNT,
};

#endif // TYPES_H