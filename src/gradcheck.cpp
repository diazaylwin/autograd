#include "gradcheck.h"
#include <random>
#include <cmath>      // std::fabs
#include <algorithm>  // std::max

static double eval_objective(
    const Program& fwd,
    Runtime& rt,
    const std::vector<OwnedTensor>& inputs,
    const std::vector<OwnedTensor>& seeds
)
{
    require(inputs.size() == fwd.inputs.size());
    require(seeds.size()  == fwd.outputs.size());

    runtime_reset(&rt);

    // materialise inputs
    std::vector<Tensor> rt_inputs;
    rt_inputs.reserve(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i)
        rt_inputs.push_back(from_vector(rt, inputs[i].sizes, inputs[i].data));

    // forward
    Tape tape;
    std::vector<Tensor> outs = execute(fwd, rt, tape, rt_inputs);

    // dot(outputs, seeds)
    double F = 0.0;
    for (size_t j = 0; j < outs.size(); ++j)
    {
        OwnedTensor out_j = from_tensor(rt, outs[j]);
        require(out_j.sizes == seeds[j].sizes);
        F += dot_owned(out_j, seeds[j]);
    }

    return F;
}

static std::vector<OwnedTensor> grads_ad(
    const CompiledProgram& cp,
    Runtime& rt,
    const std::vector<OwnedTensor>& inputs,
    const std::vector<OwnedTensor>& seeds
)
{
    runtime_reset(&rt);

    // materialise inputs
    std::vector<Tensor> rt_inputs;
    rt_inputs.reserve(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i)
        rt_inputs.push_back(from_vector(rt, inputs[i].sizes, inputs[i].data));

    // forward
    Tape fwd_tape;
    execute(cp.fwd, rt, fwd_tape, rt_inputs);

    // build backward inputs: [all primals..., seeds...]
    std::vector<Tensor> bwd_inputs;
    bwd_inputs.reserve(cp.fwd.num_values + seeds.size());

    for (uint32_t v = 0; v < cp.fwd.num_values; ++v)
        bwd_inputs.push_back(fwd_tape.primal[(size_t)v]);

    for (size_t i = 0; i < seeds.size(); ++i)
        bwd_inputs.push_back(from_vector(rt, seeds[i].sizes, seeds[i].data));

    // backward
    Tape bwd_tape;
    std::vector<Tensor> grads = execute(cp.bwd, rt, bwd_tape, bwd_inputs);

    // convert to owned
    std::vector<OwnedTensor> out;
    out.reserve(grads.size());
    for (size_t i = 0; i < grads.size(); ++i)
        out.push_back(from_tensor(rt, grads[i]));

    return out;
}

GradcheckReport gradcheck(
    const CompiledProgram& cp,
    Runtime& rt,
    const std::vector<OwnedTensor>& inputs,
    const std::vector<OwnedTensor>& seeds,
    const GradcheckOptions& opt
)
{
    GradcheckReport rep;

    // Always report worst-case, even when OK.
    rep.ok = true;
    rep.node_index  = UINT32_MAX;
    rep.input_index = 0;
    rep.index       = 0;
    rep.ad          = 0.0;
    rep.fd          = 0.0;
    rep.abs_err     = 0.0;
    rep.rel_err     = 0.0;

    // AD gradients
    std::vector<OwnedTensor> gad = grads_ad(cp, rt, inputs, seeds);

    // ---------------------------
    // Coordinate mode
    // ---------------------------
    if (opt.mode == GRADCHECK_COORDINATE)
    {
        for (size_t k = 0; k < inputs.size(); ++k)
        {
            const OwnedTensor& x = inputs[k];
            const OwnedTensor& g = gad[k];
            uint64_t n = numel(x);

            uint64_t maxc = opt.u.coord.max_coords_per_input;
            if (maxc == 0 || maxc > n) maxc = n;

            for (uint64_t i = 0; i < maxc; ++i)
            {
                std::vector<OwnedTensor> xp = inputs;
                std::vector<OwnedTensor> xm = inputs;

                xp[k].data[i] += opt.eps;
                xm[k].data[i] -= opt.eps;

                double fp = eval_objective(cp.fwd, rt, xp, seeds);
                double fm = eval_objective(cp.fwd, rt, xm, seeds);

                double fd = (fp - fm) / (2.0 * opt.eps);
                double ad = g.data[i];

                double abs_err = std::fabs(ad - fd);
                double scale   = std::max(std::fabs(ad), std::fabs(fd));
                double tol     = opt.atol + opt.rtol * scale;

                // Diagnostic only (classic relative error)
                double rel_err = abs_err / std::max(1.0, scale);

                // Track worst discrepancy unconditionally
                if (abs_err > rep.abs_err)
                {
                    rep.input_index = (uint32_t)k;
                    rep.index   = i;
                    rep.ad      = ad;
                    rep.fd      = fd;
                    rep.abs_err = abs_err;
                    rep.rel_err = rel_err;
                }

                // Pass/fail decision
                if (abs_err > tol)
                    rep.ok = false;
            }
        }

        return rep;
    }

    // ---------------------------
    // Directional mode
    // ---------------------------
    else
    {
        std::mt19937 rng(opt.seed);
        std::normal_distribution<double> N(0.0, 1.0);

        for (uint32_t d = 0; d < opt.u.dir.num_directions; ++d)
        {
            std::vector<OwnedTensor> dir = inputs;

            // sample random directions
            for (size_t k = 0; k < dir.size(); ++k)
            {
                uint64_t n = numel(dir[k]);
                for (uint64_t i = 0; i < n; ++i)
                    dir[k].data[i] = N(rng);
            }

            // FD directional derivative
            std::vector<OwnedTensor> xp = inputs;
            std::vector<OwnedTensor> xm = inputs;

            for (size_t k = 0; k < inputs.size(); ++k)
            {
                uint64_t n = numel(inputs[k]);
                for (uint64_t i = 0; i < n; ++i)
                {
                    xp[k].data[i] += opt.eps * dir[k].data[i];
                    xm[k].data[i] -= opt.eps * dir[k].data[i];
                }
            }

            double fp = eval_objective(cp.fwd, rt, xp, seeds);
            double fm = eval_objective(cp.fwd, rt, xm, seeds);
            double fd = (fp - fm) / (2.0 * opt.eps);

            double ad = 0.0;
            for (size_t k = 0; k < inputs.size(); ++k)
            {
                uint64_t n = numel(inputs[k]);
                for (uint64_t i = 0; i < n; ++i)
                    ad += gad[k].data[i] * dir[k].data[i];
            }

            double abs_err = std::fabs(ad - fd);
            double scale   = std::max(std::fabs(ad), std::fabs(fd));
            double tol     = opt.atol + opt.rtol * scale;

            // Diagnostic only
            double rel_err = abs_err / std::max(1.0, scale);

            // Track worst discrepancy unconditionally
            if (abs_err > rep.abs_err)
            {
                rep.input_index = 0;     // directional mode: not per-input
                rep.index   = d;         // direction index
                rep.ad      = ad;
                rep.fd      = fd;
                rep.abs_err = abs_err;
                rep.rel_err = rel_err;
            }

            if (abs_err > tol)
                rep.ok = false;
        }

        return rep;
    }
}

// Build a prefix Program whose only output is `out` and whose nodes stop at the node
// that produces `out`. Assumes `out` is produced by some node in prog.nodes.
static Program prefix_to_value(const Program& prog, ValueID out)
{
    Program q = prog;

    // Find last node that writes `out`
    size_t last = (size_t)-1;
    for (size_t i = 0; i < prog.nodes.size(); ++i)
        if (prog.nodes[i].out == out)
            last = i;

    require(last != (size_t)-1 && "prefix_to_value: out not produced by any node");

    q.nodes.resize(last + 1);

    q.outputs.clear();
    q.outputs.push_back(out);

    return q;
}

// `seed` must match the shape of that output.
GradcheckReport gradcheck_prefix(
    const Program& prog,
    ValueID out,
    Runtime& rt,
    const std::vector<OwnedTensor>& inputs,
    const OwnedTensor& seed,
    const GradcheckOptions& opt
)
{
    Program q = prefix_to_value(prog, out);

    // Build CompiledProgram for the prefix
    CompiledProgram cp;
    cp.fwd = q;
    cp.bwd = build_vjp(cp.fwd);

    std::vector<OwnedTensor> seeds;
    seeds.reserve(1);
    seeds.push_back(seed);

    GradcheckReport rep = gradcheck(cp, rt, inputs, seeds, opt);
    return rep;
}

static OwnedTensor eval_prefix_output(
    const Program& prog,
    ValueID out,
    Runtime& rt,
    const std::vector<OwnedTensor>& inputs
)
{
    Program q = prefix_to_value(prog, out);

    runtime_reset(&rt);

    std::vector<Tensor> rt_inputs;
    rt_inputs.reserve(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i)
        rt_inputs.push_back(from_vector(rt, inputs[i].sizes, inputs[i].data));

    Tape tape;
    std::vector<Tensor> outs = execute(q, rt, tape, rt_inputs);
    require(outs.size() == 1);

    return from_tensor(rt, outs[0]);
}

// Scan prefixes in topological order; return first failing prefix (or ok=true).
GradcheckReport gradcheck_prefixes_until_fail(
    const Program& prog,
    Runtime& rt,
    const std::vector<OwnedTensor>& inputs,
    const GradcheckOptions& opt
)
{
    GradcheckReport rep_ok;
    rep_ok.ok = true;
    rep_ok.node_index = UINT32_MAX;

    for (size_t i = 0; i < prog.nodes.size(); ++i)
    {
        const Node& n = prog.nodes[i];

        OwnedTensor out = eval_prefix_output(prog, n.out, rt, inputs);
        OwnedTensor seed = ones_like(out);

        GradcheckReport rep = gradcheck_prefix(prog, n.out, rt, inputs, seed, opt);

        if (!rep.ok)
        {
            rep.node_index = (uint32_t)i;
            return rep;
        }
    }

    return rep_ok;
}