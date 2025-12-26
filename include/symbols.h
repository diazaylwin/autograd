#ifndef SYMBOLS_H
#define SYMBOLS_H

#include <vector>
#include <cstdint>

#include "types.h"
#include "utils.h"

// ============================================================
// IR
// ============================================================

struct Node
{
    OpTag   op;
    ValueID out;
    ValueID a;      // unary uses only a
    ValueID b;      // unused for unary
    uint32_t attr;  // side-table index (Const literal etc.)
};

struct Program
{
    std::vector<ValueID> inputs;
    std::vector<ValueID> outputs;
    std::vector<Node>    nodes;       // topological order
    uint32_t             num_values = 0;

    std::vector<double>  const_f64;   // Const nodes: attr indexes this
};

struct Builder; // fwd

// ============================================================
// Symbolic handle: (builder-context, value id)
// ============================================================

struct Var
{
    Builder* b  = nullptr;
    ValueID  id = ValueID(0);
};

// ============================================================
// Builder: owns the program-under-construction and id allocation
// ============================================================

struct Builder
{
    Program p;
    ValueID next = 0;
};

// ============================================================
// Internal helpers
// ============================================================

static inline void require_same_builder(const Var& x, const Var& y)
{
    require(x.b != nullptr);
    require(y.b != nullptr);
    require(x.b == y.b && "Vars belong to different Builders");
}

static inline Var emit_unary(Builder& b, const OpTag op, const Var& x, const uint32_t attr = 0u)
{
    require(x.b == &b && "Var belongs to a different Builder");

    const ValueID out = b.next++;

    Node n;
    n.op   = op;
    n.out  = out;
    n.a    = x.id;
    n.b    = ValueID(0);
    n.attr = attr;

    b.p.nodes.push_back(n);
    return Var{ &b, out };
}

static inline Var emit_binary(
    Builder& b,
    const OpTag op,
    const Var& x,
    const Var& y,
    const uint32_t attr = 0u
)
{
    require(x.b == &b && "Var belongs to a different Builder");
    require(y.b == &b && "Var belongs to a different Builder");

    const ValueID out = b.next++;

    Node n;
    n.op   = op;
    n.out  = out;
    n.a    = x.id;
    n.b    = y.id;
    n.attr = attr;

    b.p.nodes.push_back(n);
    return Var{ &b, out };
}

// ============================================================
// Construction (free functions)
// ============================================================

static inline Var Input(Builder& b)
{
    const ValueID id = b.next++;
    b.p.inputs.push_back(id);
    return Var{ &b, id };
}

// Scalar-tensor constant (rank-0 / numel=1 at runtime).
static inline Var Constant(Builder& b, const double c)
{
    const uint32_t idx = (uint32_t)b.p.const_f64.size();
    b.p.const_f64.push_back(c);

    const ValueID out = b.next++;

    Node n;
    n.op   = OpTag::Const;
    n.out  = out;
    n.a    = ValueID(0);
    n.b    = ValueID(0);
    n.attr = idx;

    b.p.nodes.push_back(n);
    return Var{ &b, out };
}

// ============================================================
// IR ops (free functions)
// ============================================================

static inline Var Add(Builder& b, const Var& x, const Var& y) { return emit_binary(b, OpTag::Add, x, y); }
static inline Var Mul(Builder& b, const Var& x, const Var& y) { return emit_binary(b, OpTag::Mul, x, y); }
static inline Var Div(Builder& b, const Var& x, const Var& y) { return emit_binary(b, OpTag::Div, x, y); }
static inline Var Neg(Builder& b, const Var& x)               { return emit_unary (b, OpTag::Neg, x); }

// ============================================================
// Finalize
// ============================================================

static inline Program Finalize(Builder& b, const std::vector<Var>& outs)
{
    b.p.outputs.clear();
    b.p.outputs.reserve(outs.size());

    for (size_t i = 0; i < outs.size(); ++i)
    {
        const Var& v = outs[i];
        require(v.b == &b && "Output Var belongs to a different Builder");
        b.p.outputs.push_back(v.id);
    }

    b.p.num_values = (uint32_t)b.next;

    // Minimal validation
    for (size_t i = 0; i < b.p.inputs.size(); ++i)
        require(b.p.inputs[i] < b.p.num_values);

    for (size_t i = 0; i < b.p.outputs.size(); ++i)
        require(b.p.outputs[i] < b.p.num_values);

    for (size_t i = 0; i < b.p.nodes.size(); ++i)
    {
        const Node& n = b.p.nodes[i];
        require(n.out < b.p.num_values);

        if (n.op == OpTag::Const)
        {
            require(n.attr < b.p.const_f64.size());
            continue;
        }

        require(n.a < b.p.num_values);

        const bool is_binary =
            (n.op == OpTag::Add) ||
            (n.op == OpTag::Mul) ||
            (n.op == OpTag::Div);

        if (is_binary)
            require(n.b < b.p.num_values);
    }

    return b.p;
}

// ============================================================
// Operator overloading on Var (Var ⊕ Var only)
// ============================================================

static inline Var operator+(const Var& x, const Var& y)
{
    require_same_builder(x, y);
    return Add(*x.b, x, y);
}

static inline Var operator*(const Var& x, const Var& y)
{
    require_same_builder(x, y);
    return Mul(*x.b, x, y);
}

static inline Var operator/(const Var& x, const Var& y)
{
    require_same_builder(x, y);
    return Div(*x.b, x, y);
}

static inline Var operator-(const Var& x) // unary minus
{
    require(x.b != nullptr);
    return Neg(*x.b, x);
}

static inline Var operator-(const Var& x, const Var& y) // subtraction sugar
{
    require_same_builder(x, y);
    return x + (-y);
}

// ============================================================
// Operator overloading on Var, double (scalar multiplication/division only)
// ============================================================

static inline Var operator*(const double c, const Var& x)
{
    require(x.b != nullptr);
    return Mul(*x.b, Constant(*x.b, c), x);
}

static inline Var operator*(const Var& x, const double c)
{
    require(x.b != nullptr);
    return Mul(*x.b, x, Constant(*x.b, c));
}

static inline Var operator/(const Var& x, const double c)
{
    require(x.b != nullptr);
    return Div(*x.b, x, Constant(*x.b, c));
}

#endif // SYMBOLS_H