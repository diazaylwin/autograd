#include "autodiff.h"
#include "runtime.h"
#include "utils.h"

#include <vector>

// ----------------------------------
// forward execution
// ----------------------------------

std::vector<Tensor> execute(
    const Program& prog,
    Runtime& rt,
    Tape& tape,
    const std::vector<Tensor>& input_tensors
)
{
    require(prog.inputs.size() == input_tensors.size());

    tape.primal.clear();
    tape.primal.resize(prog.num_values);

    // Bind inputs
    for (size_t i = 0; i < prog.inputs.size(); ++i)
    {
        ValueID vid = prog.inputs[i];
        require(vid < prog.num_values);
        tape.primal[vid] = input_tensors[i];
    }

    // Evaluate nodes
    for (size_t i = 0; i < prog.nodes.size(); ++i)
    {
        const Node& n = prog.nodes[i];
        require(n.out < prog.num_values);

        switch (n.op)
        {
            case OpTag::Const:
            {
                require(n.attr < prog.const_f64.size());
                const double c = prog.const_f64[n.attr];

                // scalar = rank-1 tensor {1}
                tape.primal[n.out] =
                    from_vector(rt, std::vector<uint64_t>{1},
                                   std::vector<double>{c});
                break;
            }

            case OpTag::Add:
                tape.primal[n.out] = add(rt, tape.primal[n.a], tape.primal[n.b]);
                break;

            case OpTag::Mul:
                tape.primal[n.out] = mul(rt, tape.primal[n.a], tape.primal[n.b]);
                break;

            case OpTag::Div:
                tape.primal[n.out] = div(rt, tape.primal[n.a], tape.primal[n.b]);
                break;

            case OpTag::Neg:
                tape.primal[n.out] = neg(rt, tape.primal[n.a]);
                break;

            case OpTag::ZeroLike:
                tape.primal[n.out] = zero_like(rt, tape.primal[n.a]);
                break;
            
            case OpTag::Detach:
                tape.primal[n.out] = detach(rt, tape.primal[n.a]);
                break;

            default:
                require(false && "unhandled op in execute()");
        }
    }

    // Collect outputs
    std::vector<Tensor> outs;
    outs.reserve(prog.outputs.size());
    for (size_t i = 0; i < prog.outputs.size(); ++i)
    {
        ValueID oid = prog.outputs[i];
        require(oid < prog.num_values);
        outs.push_back(tape.primal[oid]);
    }
    return outs;
}

// ----------------------------------
// backward program synthesis (VJP)
// ----------------------------------

static const ValueID INVALID_ID = (ValueID)(~(uint64_t)0);

// minimal builder emit helpers (ValueID-based)

static ValueID new_input(Builder& b)
{
    ValueID id = b.next++;
    b.p.inputs.push_back(id);
    return id;
}

static ValueID emit_unary(Builder& b, const OpTag op, const ValueID a, const uint32_t attr)
{
    ValueID out = b.next++;

    Node n;
    n.op   = op;
    n.out  = out;
    n.a    = a;
    n.b    = (ValueID)0;
    n.attr = attr;

    b.p.nodes.push_back(n);
    return out;
}

static ValueID emit_binary(Builder& b, const OpTag op, const ValueID a, const ValueID bb, const uint32_t attr)
{
    ValueID out = b.next++;

    Node n;
    n.op   = op;
    n.out  = out;
    n.a    = a;
    n.b    = bb;
    n.attr = attr;

    b.p.nodes.push_back(n);
    return out;
}

static ValueID emit_add(Builder& b, const ValueID x, const ValueID y) { return emit_binary(b, OpTag::Add, x, y, 0u); }
static ValueID emit_mul(Builder& b, const ValueID x, const ValueID y) { return emit_binary(b, OpTag::Mul, x, y, 0u); }
static ValueID emit_div(Builder& b, const ValueID x, const ValueID y) { return emit_binary(b, OpTag::Div, x, y, 0u); }
static ValueID emit_neg(Builder& b, const ValueID x)                  { return emit_unary (b, OpTag::Neg, x, 0u); }
static ValueID emit_zero_like(Builder& b, const ValueID x)            { return emit_unary (b, OpTag::ZeroLike, x, 0u); }

static void accum_adj(
    Builder& bwd,
    std::vector<ValueID>& g,
    const ValueID fwd_v,
    const ValueID delta
)
{
    if (g[(size_t)fwd_v] == INVALID_ID)
    {
        g[(size_t)fwd_v] = delta;
    }
    else
    {
        g[(size_t)fwd_v] = emit_add(bwd, g[(size_t)fwd_v], delta);
    }
}

Program build_vjp(const Program& fwd)
{
    Builder bwd;
    bwd.p.inputs.clear();
    bwd.p.outputs.clear();
    bwd.p.nodes.clear();
    bwd.p.const_f64.clear();
    bwd.p.num_values = 0u;
    bwd.next = 0;

    // backward inputs: primals for ALL forward values
    std::vector<ValueID> primal_in;
    primal_in.resize(fwd.num_values, INVALID_ID);

    for (uint32_t v = 0; v < fwd.num_values; ++v)
        primal_in[(size_t)v] = new_input(bwd);

    // backward inputs: output seeds
    std::vector<ValueID> seed_in;
    seed_in.resize(fwd.outputs.size(), INVALID_ID);

    for (size_t i = 0; i < fwd.outputs.size(); ++i)
        seed_in[i] = new_input(bwd);

    // g maps forward ValueID -> backward ValueID (accumulated adjoint)
    std::vector<ValueID> g;
    g.resize(fwd.num_values, INVALID_ID);

    // seed outputs
    for (size_t i = 0; i < fwd.outputs.size(); ++i)
        g[(size_t)fwd.outputs[i]] = seed_in[i];

    // reverse walk forward nodes, emit VJP nodes
    for (int i = (int)fwd.nodes.size() - 1; i >= 0; --i)
    {
        const Node& n = fwd.nodes[(size_t)i];

        if (g[(size_t)n.out] == INVALID_ID) continue;
        const ValueID ybar = g[(size_t)n.out];

        switch (n.op)
        {
            case OpTag::Const:
                break;

            case OpTag::Add:
                accum_adj(bwd, g, n.a, ybar);
                accum_adj(bwd, g, n.b, ybar);
                break;

            case OpTag::Mul:
            {
                const ValueID a = primal_in[(size_t)n.a];
                const ValueID b = primal_in[(size_t)n.b];
                accum_adj(bwd, g, n.a, emit_mul(bwd, ybar, b));
                accum_adj(bwd, g, n.b, emit_mul(bwd, ybar, a));
                break;
            }

            case OpTag::Div:
            {
                const ValueID a = primal_in[(size_t)n.a];
                const ValueID b = primal_in[(size_t)n.b];

                // ā += ȳ / b
                accum_adj(bwd, g, n.a, emit_div(bwd, ybar, b));

                // b̄ += ȳ * ( -a / b^2 )
                ValueID b2            = emit_mul(bwd, b, b);
                ValueID a_over_b2     = emit_div(bwd, a, b2);
                ValueID neg_a_over_b2 = emit_neg(bwd, a_over_b2);
                ValueID term          = emit_mul(bwd, ybar, neg_a_over_b2);
                accum_adj(bwd, g, n.b, term);
                break;
            }

            case OpTag::Neg:
                accum_adj(bwd, g, n.a, emit_neg(bwd, ybar));
                break;
            
            case OpTag::Detach:
                // y = detach(x). Gradient stops here: do nothing.
                break;

            default:
                require(false && "build_vjp: unhandled op");
        }
    }

    // outputs: grads aligned with forward inputs
    bwd.p.outputs.clear();
    bwd.p.outputs.reserve(fwd.inputs.size());

    for (size_t i = 0; i < fwd.inputs.size(); ++i)
    {
        ValueID in = fwd.inputs[i];

        if (g[(size_t)in] == INVALID_ID)
            bwd.p.outputs.push_back(emit_zero_like(bwd, primal_in[(size_t)in]));
        else
            bwd.p.outputs.push_back(g[(size_t)in]);
    }

    bwd.p.num_values = (uint32_t)bwd.next;
    return bwd.p;
}

// ----------------------------------
// backward pass = compile + execute
// ----------------------------------

std::vector<Tensor> backward(
    const Program& prog,
    Runtime& rt,
    Tape& tape,
    const std::vector<Tensor>& output_seeds
)
{
    require(prog.outputs.size() == output_seeds.size());
    require(tape.primal.size() == prog.num_values); // execute() must have run

    Program bwd = build_vjp(prog);

    // bwd inputs: [all forward primals..., output seeds...]
    std::vector<Tensor> bwd_inputs;
    bwd_inputs.reserve((size_t)prog.num_values + prog.outputs.size());

    for (uint32_t v = 0; v < prog.num_values; ++v)
        bwd_inputs.push_back(tape.primal[(size_t)v]);

    for (size_t i = 0; i < output_seeds.size(); ++i)
        bwd_inputs.push_back(output_seeds[i]);

    Tape tape_bwd;
    return execute(bwd, rt, tape_bwd, bwd_inputs);
}