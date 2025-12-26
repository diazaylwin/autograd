#include "autodiff.h"
#include "runtime.h"
#include "utils.h"

#include <vector>

// ----------------------------------
// helpers local to this translation unit
// ----------------------------------

static uint64_t numel_from_sizes(const std::vector<uint64_t>& sizes)
{
    uint64_t n = 1;
    for (size_t i = 0; i < sizes.size(); ++i) n *= sizes[i];
    return n;
}

static void accumulate_adj(Runtime& rt, Tape& tape, ValueID v, const Tensor& delta)
{
    if (!tape.has_adj[v])
    {
        // materialise: adjoints should own their storage
        tape.adj[v] = clone(rt, delta);
        tape.has_adj[v] = 1;
    }
    else
    {
        add_inplace(rt, tape.adj[v], delta);
    }
}

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
                tape.primal[n.out] =
                    add(rt, tape.primal[n.a], tape.primal[n.b]);
                break;

            case OpTag::Mul:
                tape.primal[n.out] =
                    mul(rt, tape.primal[n.a], tape.primal[n.b]);
                break;

            case OpTag::Div:
                tape.primal[n.out] =
                    div(rt, tape.primal[n.a], tape.primal[n.b]);
                break;

            case OpTag::Neg:
                tape.primal[n.out] =
                    neg(rt, tape.primal[n.a]);
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
// backward pass
// ----------------------------------

std::vector<Tensor> backward(
    const Program& prog,
    Runtime& rt,
    Tape& tape,
    const std::vector<Tensor>& output_seeds
)
{
    require(prog.outputs.size() == output_seeds.size());
    require(tape.primal.size() == prog.num_values);

    tape.adj.clear();
    tape.adj.resize(prog.num_values);
    tape.has_adj.assign(prog.num_values, 0);

    // Seed outputs
    for (size_t i = 0; i < prog.outputs.size(); ++i)
    {
        ValueID out = prog.outputs[i];
        require(out < prog.num_values);

        tape.adj[out] = clone(rt, output_seeds[i]);
        tape.has_adj[out] = 1;
    }

    // Reverse pass
    for (int i = (int)prog.nodes.size() - 1; i >= 0; --i)
    {
        const Node& n = prog.nodes[(size_t)i];
        if (!tape.has_adj[n.out]) continue;

        const Tensor ybar = tape.adj[n.out];

        switch (n.op)
        {
            case OpTag::Const:
                break;

            case OpTag::Add:
            {
                accumulate_adj(rt, tape, n.a, ybar);
                accumulate_adj(rt, tape, n.b, ybar);
                break;
            }

            case OpTag::Mul:
            {
                const Tensor a = tape.primal[n.a];
                const Tensor b = tape.primal[n.b];

                accumulate_adj(rt, tape, n.a, mul(rt, ybar, b));
                accumulate_adj(rt, tape, n.b, mul(rt, ybar, a));
                break;
            }

            case OpTag::Div:
            {
                const Tensor a = tape.primal[n.a];
                const Tensor b = tape.primal[n.b];

                // ā += ȳ / b
                accumulate_adj(rt, tape, n.a, div(rt, ybar, b));

                // b̄ += ȳ * ( -a / b^2 )
                const Tensor b2 = mul(rt, b, b);
                const Tensor a_over_b2 = div(rt, a, b2);
                const Tensor neg_a_over_b2 = neg(rt, a_over_b2);
                accumulate_adj(rt, tape, n.b,
                               mul(rt, ybar, neg_a_over_b2));
                break;
            }

            case OpTag::Neg:
            {
                accumulate_adj(rt, tape, n.a, neg(rt, ybar));
                break;
            }

            default:
                require(false && "unhandled op in backward()");
        }
    }

    // Return grads aligned with inputs
    std::vector<Tensor> grads;
    grads.reserve(prog.inputs.size());

    for (size_t i = 0; i < prog.inputs.size(); ++i)
    {
        ValueID in = prog.inputs[i];
        require(in < prog.num_values);

        if (tape.has_adj[in])
        {
            grads.push_back(tape.adj[in]);
        }
        else
        {
            // zero tensor via from_vector
            const uint64_t n = numel_from_sizes(tape.primal[in].sizes);
            std::vector<double> zeros(n, 0.0);
            grads.push_back(
                from_vector(rt, tape.primal[in].sizes, zeros)
            );
        }
    }

    return grads;
}
