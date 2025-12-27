#include "optimization.h"
#include "utils.h"

#include <vector>
#include <cstdio>

// ============================================================
// DCE helpers
// ============================================================

static void mark_live(std::vector<bool>& live, const ValueID v)
{
    live[(size_t)v] = true;
}

static void mark_args_live(
    std::vector<bool>& live,
    const Program& p,
    const uint32_t args_off,
    const uint32_t nargs
)
{
    for (uint32_t j = 0; j < nargs; ++j)
    {
        const ValueID v = p.args[(size_t)args_off + (size_t)j];
        mark_live(live, v);
    }
}

// ============================================================
// DCE: dead code elimination
// ============================================================

// Forward declaration for recursion
static Program dce_impl(const Program& p);

static Program dce_impl(const Program& p)
{
    // Step 1: initialise liveness
    std::vector<bool> live(p.num_values, false);

    // Mark outputs as live
    for (size_t i = 0; i < p.outputs.size(); ++i)
        mark_live(live, p.outputs[i]);

    // Mark inputs as live (externally provided)
    for (size_t i = 0; i < p.inputs.size(); ++i)
        mark_live(live, p.inputs[i]);

    // Step 2: reverse walk — propagate liveness to inputs
    for (int i = (int)p.nodes.size() - 1; i >= 0; --i)
    {
        const Node& n = p.nodes[(size_t)i];

        if (!live[(size_t)n.out])
        {
            // For multi-output nodes (ScanVJP), check if any output is live
            if (n.op == OpTag::ScanVJP)
            {
                bool any_out_live = false;
                for (uint32_t j = 0; j < n.nouts; ++j)
                {
                    const ValueID extra = p.outs[(size_t)n.outs_off + (size_t)j];
                    if (live[(size_t)extra])
                    {
                        any_out_live = true;
                        break;
                    }
                }
                if (!any_out_live) continue;

                // If any output is live, mark primary output live too
                mark_live(live, n.out);
            }
            else
            {
                continue;
            }
        }

        switch (n.op)
        {
            case OpTag::Const:
                break;

            case OpTag::Neg:
            case OpTag::ZeroLike:
            case OpTag::Detach:
                mark_live(live, n.a);
                break;

            case OpTag::Add:
            case OpTag::Mul:
            case OpTag::Div:
                mark_live(live, n.a);
                mark_live(live, n.b);
                break;

            case OpTag::Scan:
                mark_live(live, n.a);
                mark_args_live(live, p, n.args_off, n.nargs);
                break;

            case OpTag::ScanVJP:
                mark_live(live, n.a);
                mark_live(live, n.b);
                mark_args_live(live, p, n.args_off, n.nargs);
                // Mark all extra outputs as live (can't partially compute)
                for (uint32_t j = 0; j < n.nouts; ++j)
                {
                    const ValueID extra = p.outs[(size_t)n.outs_off + (size_t)j];
                    mark_live(live, extra);
                }
                break;

            default:
                require(false && "dce: unhandled op");
        }
    }

    // Step 3: build output program with only live nodes
    Program out;
    out.inputs     = p.inputs;
    out.outputs    = p.outputs;
    out.num_values = p.num_values;
    out.const_f64  = p.const_f64;
    out.args       = p.args;
    out.outs       = p.outs;
    out.scans      = p.scans;

    // Recursively DCE scan bodies
    out.scan_bodies.reserve(p.scan_bodies.size());
    for (size_t i = 0; i < p.scan_bodies.size(); ++i)
        out.scan_bodies.push_back(dce_impl(p.scan_bodies[i]));

    // Keep only live nodes
    for (size_t i = 0; i < p.nodes.size(); ++i)
    {
        const Node& n = p.nodes[i];
        if (live[(size_t)n.out])
            out.nodes.push_back(n);
    }

    return out;
}

Program dce(const Program& p)
{
    return dce_impl(p);
}

// ============================================================
// Optimise
// ============================================================

static void print_dce_stats(const char* name, const size_t before, const size_t after)
{
    if (before == 0)
    {
        std::printf("%s: 0 nodes\n", name);
        return;
    }

    const int reduction = (int)(100.0 * (1.0 - (double)after / (double)before));
    std::printf("%s: %zu -> %zu nodes (%d%% reduction)\n", name, before, after, reduction);
}

static void compute_save_set(CompiledProgram& cp)
{
    const uint32_t num_fwd_values = cp.fwd.num_values;

    // Track which primal inputs (first num_fwd_values bwd inputs) are used
    std::vector<bool> used(num_fwd_values, false);

    // Walk bwd nodes and mark used primals
    for (size_t i = 0; i < cp.bwd.nodes.size(); ++i)
    {
        const Node& n = cp.bwd.nodes[i];

        // Check if node inputs reference primal values
        // (primal inputs are bwd ValueIDs 0..num_fwd_values-1)
        if (n.a < num_fwd_values) used[(size_t)n.a] = true;
        if (n.b < num_fwd_values) used[(size_t)n.b] = true;

        // Check args for Scan/ScanVJP
        for (uint32_t j = 0; j < n.nargs; ++j)
        {
            const ValueID arg = cp.bwd.args[(size_t)n.args_off + (size_t)j];
            if (arg < num_fwd_values) used[(size_t)arg] = true;
        }
    }

    // Build save_set: fwd ValueIDs that bwd needs
    cp.save_set.clear();
    for (uint32_t v = 0; v < num_fwd_values; ++v)
    {
        if (used[v])
            cp.save_set.push_back((ValueID)v);
    }
}

void Optimise(CompiledProgram& cp)
{
    const size_t fwd_before = cp.fwd.nodes.size();
    const size_t bwd_before = cp.bwd.nodes.size();

    cp.fwd = dce(cp.fwd);
    cp.bwd = dce(cp.bwd);

    print_dce_stats("fwd", fwd_before, cp.fwd.nodes.size());
    print_dce_stats("bwd", bwd_before, cp.bwd.nodes.size());

    compute_save_set(cp);
    std::printf("save_set: %zu values (of %u fwd values)\n",
                cp.save_set.size(), cp.fwd.num_values);
}
