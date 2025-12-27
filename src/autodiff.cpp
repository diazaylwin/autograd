#include "autodiff.h"
#include "runtime.h"
#include "utils.h"

#include <vector>

// ============================================================
// helpers
// ============================================================

static const ValueID INVALID_ID = (ValueID)(~(uint32_t)0);

// ============================================================
// forward execution
// ============================================================

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
        const ValueID vid = prog.inputs[i];
        require(vid < prog.num_values);
        tape.primal[(size_t)vid] = input_tensors[i];
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

                tape.primal[(size_t)n.out] =
                    from_vector(rt, std::vector<uint64_t>{1}, std::vector<double>{c});
                break;
            }

            case OpTag::Add:
                tape.primal[(size_t)n.out] = add(rt, tape.primal[(size_t)n.a], tape.primal[(size_t)n.b]);
                break;

            case OpTag::Mul:
                tape.primal[(size_t)n.out] = mul(rt, tape.primal[(size_t)n.a], tape.primal[(size_t)n.b]);
                break;

            case OpTag::Div:
                tape.primal[(size_t)n.out] = div(rt, tape.primal[(size_t)n.a], tape.primal[(size_t)n.b]);
                break;

            case OpTag::Neg:
                tape.primal[(size_t)n.out] = neg(rt, tape.primal[(size_t)n.a]);
                break;

            case OpTag::ZeroLike:
                tape.primal[(size_t)n.out] = zero_like(rt, tape.primal[(size_t)n.a]);
                break;

            case OpTag::Detach:
                tape.primal[(size_t)n.out] = detach(rt, tape.primal[(size_t)n.a]);
                break;

            case OpTag::Exp:
                tape.primal[(size_t)n.out] = exp(rt, tape.primal[(size_t)n.a]);
                break;

            case OpTag::Log:
                tape.primal[(size_t)n.out] = log(rt, tape.primal[(size_t)n.a]);
                break;

            case OpTag::Sum:
                tape.primal[(size_t)n.out] = sum(rt, tape.primal[(size_t)n.a]);
                break;

            case OpTag::Expand:
                tape.primal[(size_t)n.out] = expand(rt, tape.primal[(size_t)n.a], tape.primal[(size_t)n.b]);
                break;

            case OpTag::Call:
            {
                require(n.attr < prog.call_bodies.size());
                const Program& body = prog.call_bodies[n.attr];

                std::vector<Tensor> call_in = { tape.primal[(size_t)n.a] };
                Tape call_tape;
                std::vector<Tensor> call_out = execute(body, rt, call_tape, call_in);
                require(call_out.size() == 1);

                tape.primal[(size_t)n.out] = call_out[0];
                break;
            }

            case OpTag::Scan:
            {
                require(n.attr < prog.scans.size());
                const ScanInfo& si = prog.scans[n.attr];
                require(si.steps >= 1u);
                require(si.body_id < prog.scan_bodies.size());

                const Program& body = prog.scan_bodies[si.body_id];

                require(n.args_off + n.nargs <= prog.args.size());

                // invariants in outer program
                std::vector<Tensor> inv;
                inv.reserve(n.nargs);
                for (uint32_t j = 0; j < n.nargs; ++j)
                {
                    const ValueID v = prog.args[(size_t)n.args_off + (size_t)j];
                    require(v < prog.num_values);
                    inv.push_back(tape.primal[(size_t)v]);
                }

                Tensor s = tape.primal[(size_t)n.a];

                for (uint32_t t = 0; t < si.steps; ++t)
                {
                    std::vector<Tensor> body_in;
                    body_in.reserve(1u + inv.size());
                    body_in.push_back(s);
                    for (size_t k = 0; k < inv.size(); ++k) body_in.push_back(inv[k]);

                    Tape body_tape;
                    std::vector<Tensor> body_out = execute(body, rt, body_tape, body_in);
                    require(body_out.size() == 1u);

                    s = body_out[0];
                }

                tape.primal[(size_t)n.out] = s;
                break;
            }

            case OpTag::ScanVJP:
            {
                // Convention for ScanVJP node:
                //   n.a = primal init state s0
                //   invariants are referenced via prog.args[n.args_off .. n.args_off+n.nargs)
                //   n.b = seed ybar (adjoint of scan output)
                // Outputs:
                //   n.out = grad w.r.t. init state
                //   prog.outs[n.outs_off + j] = grad w.r.t. invariant j

                require(n.attr < prog.scans.size());
                const ScanInfo& si = prog.scans[n.attr];
                require(si.steps >= 1u);
                require(si.body_id < prog.scan_bodies.size());

                const Program& body = prog.scan_bodies[si.body_id];

                require(n.args_off + n.nargs <= prog.args.size());
                require(n.outs_off + n.nouts <= prog.outs.size());
                require(n.nouts == n.nargs);

                // invariants (outer program)
                std::vector<Tensor> inv;
                inv.reserve(n.nargs);
                for (uint32_t j = 0; j < n.nargs; ++j)
                {
                    const ValueID v = prog.args[(size_t)n.args_off + (size_t)j];
                    require(v < prog.num_values);
                    inv.push_back(tape.primal[(size_t)v]);
                }

                // rematerialise states s0..sT
                std::vector<Tensor> states;
                states.reserve((size_t)si.steps + 1u);

                Tensor s = tape.primal[(size_t)n.a];
                states.push_back(s);

                for (uint32_t t = 0; t < si.steps; ++t)
                {
                    std::vector<Tensor> body_in;
                    body_in.reserve(1u + inv.size());
                    body_in.push_back(s);
                    for (size_t k = 0; k < inv.size(); ++k) body_in.push_back(inv[k]);

                    Tape body_tape;
                    std::vector<Tensor> body_out = execute(body, rt, body_tape, body_in);
                    require(body_out.size() == 1u);

                    s = body_out[0];
                    states.push_back(s);
                }

                // compile body VJP once
                Program body_vjp = build_vjp(body);

                // running adjoint for carried state
                Tensor sbar = tape.primal[(size_t)n.b]; // ybar

                // invariant adjoints accumulator
                std::vector<Tensor> inv_bar;
                inv_bar.reserve(inv.size());
                for (size_t k = 0; k < inv.size(); ++k)
                    inv_bar.push_back(zero_like(rt, inv[k]));

                // reverse time
                for (int t = (int)si.steps - 1; t >= 0; --t)
                {
                    const Tensor s_t = states[(size_t)t];

                    // recompute body primals at step t
                    std::vector<Tensor> body_in;
                    body_in.reserve(1u + inv.size());
                    body_in.push_back(s_t);
                    for (size_t k = 0; k < inv.size(); ++k) body_in.push_back(inv[k]);

                    Tape body_tape;
                    std::vector<Tensor> body_out = execute(body, rt, body_tape, body_in);
                    require(body_out.size() == 1u);

                    // VJP inputs: [all body primals..., seed]
                    std::vector<Tensor> vjp_in;
                    vjp_in.reserve((size_t)body.num_values + 1u);
                    for (uint32_t v = 0; v < body.num_values; ++v)
                        vjp_in.push_back(body_tape.primal[(size_t)v]);
                    vjp_in.push_back(sbar);

                    Tape vjp_tape;
                    std::vector<Tensor> grads = execute(body_vjp, rt, vjp_tape, vjp_in);

                    // grads aligned with body.inputs: [grad_state, grad_inv0, ...]
                    require(grads.size() == body.inputs.size());
                    require(grads.size() == 1u + inv.size());

                    sbar = grads[0];
                    for (size_t k = 0; k < inv.size(); ++k)
                        add_inplace(rt, inv_bar[k], grads[1u + k]);
                }

                // write outputs
                tape.primal[(size_t)n.out] = sbar;

                for (uint32_t j = 0; j < n.nouts; ++j)
                {
                    const ValueID outj = prog.outs[(size_t)n.outs_off + (size_t)j];
                    require(outj < prog.num_values);
                    tape.primal[(size_t)outj] = inv_bar[(size_t)j];
                }

                break;
            }

            default:
                require(false && "unhandled op in execute()");
        }
    }

    // Collect outputs
    std::vector<Tensor> outs;
    outs.reserve(prog.outputs.size());
    for (size_t i = 0; i < prog.outputs.size(); ++i)
    {
        const ValueID oid = prog.outputs[i];
        require(oid < prog.num_values);
        outs.push_back(tape.primal[(size_t)oid]);
    }
    return outs;
}

// ============================================================
// backward program synthesis (VJP)
// ============================================================

static void builder_clear(Builder& b)
{
    b.p.inputs.clear();
    b.p.outputs.clear();
    b.p.nodes.clear();
    b.p.const_f64.clear();
    b.p.args.clear();
    b.p.scan_bodies.clear();
    b.p.scans.clear();
    b.p.outs.clear();
    b.p.num_values = 0u;
    b.next = 0;
}

static ValueID new_input(Builder& b)
{
    const ValueID id = b.next++;
    b.p.inputs.push_back(id);
    return id;
}

static ValueID emit_unary_v(Builder& b, const OpTag op, const ValueID a, const uint32_t attr = 0u)
{
    const ValueID out = b.next++;

    Node n;
    n.op       = op;
    n.out      = out;
    n.a        = a;
    n.b        = (ValueID)0;
    n.attr     = attr;
    n.nargs    = 0u;
    n.args_off = 0u;
    n.nouts    = 0u;
    n.outs_off = 0u;

    b.p.nodes.push_back(n);
    return out;
}

static ValueID emit_binary_v(Builder& b, const OpTag op, const ValueID a, const ValueID bb, const uint32_t attr = 0u)
{
    const ValueID out = b.next++;

    Node n;
    n.op       = op;
    n.out      = out;
    n.a        = a;
    n.b        = bb;
    n.attr     = attr;
    n.nargs    = 0u;
    n.args_off = 0u;
    n.nouts    = 0u;
    n.outs_off = 0u;

    b.p.nodes.push_back(n);
    return out;
}

static ValueID emit_add(Builder& b, const ValueID x, const ValueID y) { return emit_binary_v(b, OpTag::Add, x, y); }
static ValueID emit_mul(Builder& b, const ValueID x, const ValueID y) { return emit_binary_v(b, OpTag::Mul, x, y); }
static ValueID emit_div(Builder& b, const ValueID x, const ValueID y) { return emit_binary_v(b, OpTag::Div, x, y); }
static ValueID emit_neg(Builder& b, const ValueID x)                  { return emit_unary_v (b, OpTag::Neg, x); }
static ValueID emit_sum(Builder& b, const ValueID x)                  { return emit_unary_v (b, OpTag::Sum, x); }
static ValueID emit_expand(Builder& b, const ValueID x, const ValueID like) { return emit_binary_v(b, OpTag::Expand, x, like); }
static ValueID emit_zero_like(Builder& b, const ValueID x)            { return emit_unary_v (b, OpTag::ZeroLike, x); }

static void accum_adj_v(Builder& bwd, std::vector<ValueID>& g, const ValueID fwd_v, const ValueID delta)
{
    if (g[(size_t)fwd_v] == INVALID_ID) g[(size_t)fwd_v] = delta;
    else g[(size_t)fwd_v] = emit_add(bwd, g[(size_t)fwd_v], delta);
}

static ValueID emit_scanvjp(
    Builder& bwd,
    const uint32_t scan_id,
    const ValueID init_state_primal,
    const std::vector<ValueID>& inv_primals,
    const ValueID seed_ybar,
    std::vector<ValueID>& out_inv_grads
)
{
    require(scan_id < bwd.p.scans.size());

    const ValueID out0 = bwd.next++;

    // allocate extra outputs for invariant grads
    const uint32_t outs_off = (uint32_t)bwd.p.outs.size();
    out_inv_grads.clear();
    out_inv_grads.reserve(inv_primals.size());
    for (size_t k = 0; k < inv_primals.size(); ++k)
    {
        const ValueID gid = bwd.next++;
        out_inv_grads.push_back(gid);
        bwd.p.outs.push_back(gid);
    }

    // flatten invariant primals into args
    const uint32_t args_off = (uint32_t)bwd.p.args.size();
    for (size_t k = 0; k < inv_primals.size(); ++k)
        bwd.p.args.push_back(inv_primals[k]);

    Node n;
    n.op       = OpTag::ScanVJP;
    n.out      = out0;
    n.a        = init_state_primal;
    n.b        = seed_ybar;
    n.attr     = scan_id;
    n.nargs    = (uint32_t)inv_primals.size();
    n.args_off = args_off;
    n.nouts    = (uint32_t)inv_primals.size();
    n.outs_off = outs_off;

    bwd.p.nodes.push_back(n);
    return out0;
}

Program build_vjp(const Program& fwd)
{
    Builder bwd;
    builder_clear(bwd);

    // Copy scan side tables into the backward program (ScanVJP needs them)
    bwd.p.scan_bodies = fwd.scan_bodies;
    bwd.p.scans       = fwd.scans;

    // backward inputs: primals for ALL forward values
    std::vector<ValueID> primal_in(fwd.num_values, INVALID_ID);
    for (uint32_t v = 0; v < fwd.num_values; ++v)
        primal_in[(size_t)v] = new_input(bwd);

    // backward inputs: output seeds
    std::vector<ValueID> seed_in(fwd.outputs.size(), INVALID_ID);
    for (size_t i = 0; i < fwd.outputs.size(); ++i)
        seed_in[i] = new_input(bwd);

    // g: forward ValueID -> backward ValueID (accumulated adjoint)
    std::vector<ValueID> g(fwd.num_values, INVALID_ID);

    for (size_t i = 0; i < fwd.outputs.size(); ++i)
        g[(size_t)fwd.outputs[i]] = seed_in[i];

    // reverse walk
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
                accum_adj_v(bwd, g, n.a, ybar);
                accum_adj_v(bwd, g, n.b, ybar);
                break;

            case OpTag::Mul:
            {
                const ValueID a = primal_in[(size_t)n.a];
                const ValueID b = primal_in[(size_t)n.b];
                accum_adj_v(bwd, g, n.a, emit_mul(bwd, ybar, b));
                accum_adj_v(bwd, g, n.b, emit_mul(bwd, ybar, a));
                break;
            }

            case OpTag::Div:
            {
                const ValueID a = primal_in[(size_t)n.a];
                const ValueID b = primal_in[(size_t)n.b];

                accum_adj_v(bwd, g, n.a, emit_div(bwd, ybar, b));

                ValueID b2            = emit_mul(bwd, b, b);
                ValueID a_over_b2     = emit_div(bwd, a, b2);
                ValueID neg_a_over_b2 = emit_neg(bwd, a_over_b2);
                ValueID term          = emit_mul(bwd, ybar, neg_a_over_b2);
                accum_adj_v(bwd, g, n.b, term);
                break;
            }

            case OpTag::Neg:
                accum_adj_v(bwd, g, n.a, emit_neg(bwd, ybar));
                break;

            case OpTag::Detach:
                break;

            case OpTag::ZeroLike:
                break;

            case OpTag::Exp:
            {
                // d/dx exp(x) = exp(x) = out
                // grad = ybar * out
                const ValueID out_primal = primal_in[(size_t)n.out];
                accum_adj_v(bwd, g, n.a, emit_mul(bwd, ybar, out_primal));
                break;
            }

            case OpTag::Log:
            {
                // d/dx log(x) = 1/x
                // grad = ybar / x
                const ValueID a = primal_in[(size_t)n.a];
                accum_adj_v(bwd, g, n.a, emit_div(bwd, ybar, a));
                break;
            }

            case OpTag::Sum:
            {
                // d/dx sum(x) = 1 for each element
                // grad = expand(ybar, shape_of_input)
                const ValueID a = primal_in[(size_t)n.a];
                accum_adj_v(bwd, g, n.a, emit_expand(bwd, ybar, a));
                break;
            }

            case OpTag::Expand:
            {
                // expand broadcasts scalar to tensor shape
                // gradient is the reverse: sum
                accum_adj_v(bwd, g, n.a, emit_sum(bwd, ybar));
                // n.b is the "like" tensor used only for shape, no gradient
                break;
            }

            case OpTag::Call:
            {
                // Build VJP of the call body, then inline (body + body_vjp) into bwd.
                // This avoids needing a 2-input Call op.
                require(n.attr < fwd.call_bodies.size());
                const Program& body = fwd.call_bodies[n.attr];
                Program body_vjp = build_vjp(body);

                // Build combined program: input x, dy -> runs fwd body, then body_vjp
                {
                    Builder cb;
                    builder_clear(cb);

                    // input 0: x, input 1: dy (seed)
                    ValueID cb_x  = new_input(cb);
                    ValueID cb_dy = new_input(cb);

                    // embed fwd body nodes, remapping ids
                    std::vector<ValueID> body_remap(body.num_values, INVALID_ID);
                    require(body.inputs.size() == 1);
                    body_remap[(size_t)body.inputs[0]] = cb_x;

                    // copy const table
                    uint32_t const_base = (uint32_t)cb.p.const_f64.size();
                    for (size_t c = 0; c < body.const_f64.size(); ++c)
                        cb.p.const_f64.push_back(body.const_f64[c]);

                    for (size_t ni = 0; ni < body.nodes.size(); ++ni)
                    {
                        Node bn = body.nodes[ni];
                        bn.out = cb.next++;
                        body_remap[(size_t)body.nodes[ni].out] = bn.out;

                        if (bn.op == OpTag::Const)
                            bn.attr = bn.attr + const_base;
                        else
                        {
                            if (body_remap[(size_t)bn.a] != INVALID_ID)
                                bn.a = body_remap[(size_t)bn.a];
                            if (body_remap[(size_t)bn.b] != INVALID_ID)
                                bn.b = body_remap[(size_t)bn.b];
                        }
                        cb.p.nodes.push_back(bn);
                    }

                    // now embed body_vjp nodes
                    std::vector<ValueID> vjp_remap(body_vjp.num_values, INVALID_ID);

                    // body_vjp inputs: [primal_0..primal_{n-1}, seed]
                    require(body_vjp.inputs.size() == body.num_values + 1);
                    for (uint32_t v = 0; v < body.num_values; ++v)
                        vjp_remap[(size_t)body_vjp.inputs[(size_t)v]] = body_remap[(size_t)v];
                    vjp_remap[(size_t)body_vjp.inputs[body.num_values]] = cb_dy;

                    // copy vjp const table
                    uint32_t vjp_const_base = (uint32_t)cb.p.const_f64.size();
                    for (size_t c = 0; c < body_vjp.const_f64.size(); ++c)
                        cb.p.const_f64.push_back(body_vjp.const_f64[c]);

                    for (size_t ni = 0; ni < body_vjp.nodes.size(); ++ni)
                    {
                        Node vn = body_vjp.nodes[ni];
                        vn.out = cb.next++;
                        vjp_remap[(size_t)body_vjp.nodes[ni].out] = vn.out;

                        if (vn.op == OpTag::Const)
                            vn.attr = vn.attr + vjp_const_base;
                        else
                        {
                            if (vjp_remap[(size_t)vn.a] != INVALID_ID)
                                vn.a = vjp_remap[(size_t)vn.a];
                            if (vjp_remap[(size_t)vn.b] != INVALID_ID)
                                vn.b = vjp_remap[(size_t)vn.b];
                        }
                        cb.p.nodes.push_back(vn);
                    }

                    // output: grad w.r.t. input (first output of body_vjp)
                    require(body_vjp.outputs.size() == 1);
                    cb.p.outputs.push_back(vjp_remap[(size_t)body_vjp.outputs[0]]);
                    cb.p.num_values = cb.next;

                    // Now emit a 2-input Call? No, Call is single-input.
                    // We need to pack (primal_in[n.a], ybar) somehow.
                    // Alternative: build_vjp for Call emits inline ops rather than a Call.

                    // Since Call has 1 input and 1 output, the combined body takes 2 inputs.
                    // But our Call op only supports 1 input. So we can't use Call here.
                    // Instead, inline the combined body directly.

                    // Inline combined body into bwd program:
                    std::vector<ValueID> cb_remap(cb.p.num_values, INVALID_ID);
                    cb_remap[(size_t)cb_x]  = primal_in[(size_t)n.a];
                    cb_remap[(size_t)cb_dy] = ybar;

                    // copy cb const table
                    uint32_t cb_const_base = (uint32_t)bwd.p.const_f64.size();
                    for (size_t c = 0; c < cb.p.const_f64.size(); ++c)
                        bwd.p.const_f64.push_back(cb.p.const_f64[c]);

                    for (size_t ni = 0; ni < cb.p.nodes.size(); ++ni)
                    {
                        Node cn = cb.p.nodes[ni];
                        cn.out = bwd.next++;
                        cb_remap[(size_t)cb.p.nodes[ni].out] = cn.out;

                        if (cn.op == OpTag::Const)
                            cn.attr = cn.attr + cb_const_base;
                        else
                        {
                            if (cb_remap[(size_t)cn.a] != INVALID_ID)
                                cn.a = cb_remap[(size_t)cn.a];
                            if (cb_remap[(size_t)cn.b] != INVALID_ID)
                                cn.b = cb_remap[(size_t)cn.b];
                        }
                        bwd.p.nodes.push_back(cn);
                    }

                    // accumulate the gradient
                    require(cb.p.outputs.size() == 1);
                    ValueID grad_a = cb_remap[(size_t)cb.p.outputs[0]];
                    accum_adj_v(bwd, g, n.a, grad_a);
                }
                break;
            }

            case OpTag::Scan:
            {
                require(n.attr < fwd.scans.size());
                require(n.args_off + n.nargs <= fwd.args.size());

                const ValueID init_primal = primal_in[(size_t)n.a];

                std::vector<ValueID> inv_vids;
                inv_vids.reserve(n.nargs);

                std::vector<ValueID> inv_primals;
                inv_primals.reserve(n.nargs);

                for (uint32_t j = 0; j < n.nargs; ++j)
                {
                    const ValueID v = fwd.args[(size_t)n.args_off + (size_t)j];
                    require(v < fwd.num_values);
                    inv_vids.push_back(v);
                    inv_primals.push_back(primal_in[(size_t)v]);
                }

                std::vector<ValueID> inv_grad_vals;
                const ValueID init_grad = emit_scanvjp(
                    bwd,
                    n.attr,         // scan id (indices aligned by copy)
                    init_primal,
                    inv_primals,
                    ybar,
                    inv_grad_vals
                );

                accum_adj_v(bwd, g, n.a, init_grad);

                for (size_t k = 0; k < inv_vids.size(); ++k)
                    accum_adj_v(bwd, g, inv_vids[k], inv_grad_vals[k]);

                break;
            }

            case OpTag::ScanVJP:
                require(false && "build_vjp: unexpected ScanVJP in forward program");
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
        const ValueID in = fwd.inputs[i];

        if (g[(size_t)in] == INVALID_ID)
            bwd.p.outputs.push_back(emit_zero_like(bwd, primal_in[(size_t)in]));
        else
            bwd.p.outputs.push_back(g[(size_t)in]);
    }

    bwd.p.num_values = (uint32_t)bwd.next;
    return bwd.p;
}