#ifndef LIBRARY_H
#define LIBRARY_H

#include "symbols.h"

// Library functions: composed from primitives via the Call mechanism.
// Each function returns a Program with 1 input and 1 output.

// sigmoid(x) = 1 / (1 + exp(-x))
// Uses Expand to broadcast scalar 1.0 to shape of x for element-wise ops.
static inline Program make_sigmoid_body()
{
    Builder b;

    Var x         = Input(b);
    Var one_s     = Constant(b, 1.0);     // scalar {1}
    Var one       = Expand(b, one_s, x);  // broadcast to shape of x
    Var negx      = Neg(b, x);
    Var expn      = Exp(b, negx);
    Var denom     = one + expn;           // same shape now
    Var y         = one / denom;

    return Finalize(b, {y});
}

// Convenience: emit Sigmoid(x) = Call(sigmoid_body, x)
static inline Var Sigmoid(Builder& b, const Var& x)
{
    static Program body = make_sigmoid_body();
    return Call(b, body, x);
}

#endif // LIBRARY_H
