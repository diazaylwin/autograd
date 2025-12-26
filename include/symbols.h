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

    // For most ops:
    //   unary: a is input, b unused
    //   binary: a,b are inputs
    //
    // For Scan:
    //   a = init state
    //   b unused
    ValueID a;
    ValueID b;

    // Side-table index:
    //   Const: attr = const_f64 index
    //   Scan:  attr = scans index
    uint32_t attr;

    // Var-arity extra args (Scan invariants live in Program.args)
    uint32_t nargs    = 0;
    uint32_t args_off = 0;

    // NEW: extra outputs (for multi-output ops like ScanVJP)
    uint32_t nouts    = 0;
    uint32_t outs_off = 0;
};

struct ScanInfo
{
    uint32_t body_id = 0; // index into Program::scan_bodies
    uint32_t steps   = 0; // repeat count

    // Convention for body program:
    // inputs[0] = carried state
    // inputs[1..] = invariants (same count as Node.nargs)
    // outputs[0] = next state
};

struct Program
{
    std::vector<ValueID> inputs;
    std::vector<ValueID> outputs;
    std::vector<Node>    nodes;       // topological order
    uint32_t             num_values = 0;

    std::vector<double>  const_f64;   // Const nodes: attr indexes this

    // Var-arity arg lists (flattened)
    std::vector<ValueID> args;

    // Scan side tables
    std::vector<Program>  scan_bodies; // owns one-step body programs
    std::vector<ScanInfo> scans;       // Scan nodes reference these by attr
    std::vector<ValueID> outs; // flattened extra outputs for multi-out nodes
};

struct Builder;

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
    n.op       = op;
    n.out      = out;
    n.a        = x.id;
    n.b        = ValueID(0);
    n.attr     = attr;
    n.nargs    = 0u;
    n.args_off = 0u;

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
    n.op       = op;
    n.out      = out;
    n.a        = x.id;
    n.b        = y.id;
    n.attr     = attr;
    n.nargs    = 0u;
    n.args_off = 0u;

    b.p.nodes.push_back(n);
    return Var{ &b, out };
}

static inline Var emit_scan(
    Builder& b,
    const Var& init_state,
    const Program& body,
    const uint32_t steps,
    const std::vector<Var>& invariants
)
{
    require(init_state.b == &b && "Var belongs to a different Builder");
    for (size_t i = 0; i < invariants.size(); ++i)
        require(invariants[i].b == &b && "Invariant Var belongs to a different Builder");

    require(steps >= 1u);

    // Store body program
    const uint32_t body_id = (uint32_t)b.p.scan_bodies.size();
    b.p.scan_bodies.push_back(body);

    // Store scan info
    ScanInfo info;
    info.body_id = body_id;
    info.steps   = steps;

    const uint32_t scan_id = (uint32_t)b.p.scans.size();
    b.p.scans.push_back(info);

    // Flatten invariant ids into Program.args
    const uint32_t off = (uint32_t)b.p.args.size();
    for (size_t i = 0; i < invariants.size(); ++i)
        b.p.args.push_back(invariants[i].id);

    const ValueID out = b.next++;

    Node n;
    n.op       = OpTag::Scan;
    n.out      = out;
    n.a        = init_state.id;
    n.b        = ValueID(0);
    n.attr     = scan_id;
    n.nargs    = (uint32_t)invariants.size();
    n.args_off = off;

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

// Scalar-tensor constant (rank-1 {1} at runtime).
static inline Var Constant(Builder& b, const double c)
{
    const uint32_t idx = (uint32_t)b.p.const_f64.size();
    b.p.const_f64.push_back(c);

    const ValueID out = b.next++;

    Node n;
    n.op       = OpTag::Const;
    n.out      = out;
    n.a        = ValueID(0);
    n.b        = ValueID(0);
    n.attr     = idx;
    n.nargs    = 0u;
    n.args_off = 0u;

    b.p.nodes.push_back(n);
    return Var{ &b, out };
}

static inline Var Scan(
    Builder& b,
    const Var& init_state,
    const Program& body,
    const uint32_t steps,
    const std::vector<Var>& invariants
){
    return emit_scan(b, init_state, body, steps, invariants);
}

// ============================================================
// IR ops (free functions)
// ============================================================

static inline Var Add(Builder& b, const Var& x, const Var& y) { return emit_binary(b, OpTag::Add, x, y); }
static inline Var Mul(Builder& b, const Var& x, const Var& y) { return emit_binary(b, OpTag::Mul, x, y); }
static inline Var Div(Builder& b, const Var& x, const Var& y) { return emit_binary(b, OpTag::Div, x, y); }
static inline Var Neg(Builder& b, const Var& x)               { return emit_unary (b, OpTag::Neg, x); }
static inline Var Detach(Builder& b, const Var& x)            { return emit_unary (b, OpTag::Detach, x); }

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

        if (n.op == OpTag::Scan)
        {
            require(n.attr < b.p.scans.size());
            const ScanInfo& s = b.p.scans[n.attr];
            require(s.steps >= 1u);
            require(s.body_id < b.p.scan_bodies.size());

            require(n.args_off + n.nargs <= b.p.args.size());
            for (uint32_t j = 0; j < n.nargs; ++j)
                require(b.p.args[n.args_off + j] < b.p.num_values);

            const Program& body = b.p.scan_bodies[s.body_id];
            require(body.inputs.size()  == 1u + n.nargs);
            require(body.outputs.size() == 1u);
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