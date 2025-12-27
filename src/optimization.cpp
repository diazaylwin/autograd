#include "optimization.h"
#include "utils.h"

#include <vector>
#include <map>
#include <tuple>
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
            case OpTag::Exp:
            case OpTag::Log:
            case OpTag::Sum:
                mark_live(live, n.a);
                break;

            case OpTag::Add:
            case OpTag::Mul:
            case OpTag::Div:
            case OpTag::Expand:
                mark_live(live, n.a);
                mark_live(live, n.b);
                break;

            case OpTag::Call:
                mark_live(live, n.a);
                break;

            case OpTag::Custom:
                mark_args_live(live, p, n.args_off, n.nargs);
                break;

            case OpTag::CustomVJP:
                mark_live(live, n.a);  // seed
                mark_args_live(live, p, n.args_off, n.nargs);  // primals
                // Mark all extra outputs as live
                for (uint32_t j = 0; j < n.nouts; ++j)
                {
                    const ValueID extra = p.outs[(size_t)n.outs_off + (size_t)j];
                    mark_live(live, extra);
                }
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

    // Recursively DCE call bodies
    out.call_bodies.reserve(p.call_bodies.size());
    for (size_t i = 0; i < p.call_bodies.size(); ++i)
        out.call_bodies.push_back(dce_impl(p.call_bodies[i]));

    // Copy custom ops (opaque, no recursion)
    out.custom_ops = p.custom_ops;

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
// CSE: common subexpression elimination
// ============================================================

// Key for CSE lookup: (op, input_a, input_b, attr)
// For unary ops, b is unused (set to 0).
// For Const, a and b are unused, attr indexes const_f64.
using CseKey = std::tuple<OpTag, ValueID, ValueID, uint32_t>;

static ValueID remap(const std::vector<ValueID>& remap_table, const ValueID v)
{
    return remap_table[(size_t)v];
}

static Program cse_impl(const Program& p);

static Program cse_impl(const Program& p)
{
    // remap_table[old_id] = new_id after CSE
    std::vector<ValueID> remap_table(p.num_values);
    for (uint32_t v = 0; v < p.num_values; ++v)
        remap_table[v] = (ValueID)v;

    // Map from CSE key to the ValueID that computes it
    std::map<CseKey, ValueID> seen;

    Program out;
    out.inputs     = p.inputs;
    out.num_values = p.num_values;
    out.const_f64  = p.const_f64;
    out.scans      = p.scans;
    out.args       = p.args;
    out.outs       = p.outs;

    // Recursively CSE scan bodies
    out.scan_bodies.reserve(p.scan_bodies.size());
    for (size_t i = 0; i < p.scan_bodies.size(); ++i)
        out.scan_bodies.push_back(cse_impl(p.scan_bodies[i]));

    // Recursively CSE call bodies
    out.call_bodies.reserve(p.call_bodies.size());
    for (size_t i = 0; i < p.call_bodies.size(); ++i)
        out.call_bodies.push_back(cse_impl(p.call_bodies[i]));

    // Copy custom ops (opaque, no recursion)
    out.custom_ops = p.custom_ops;

    // Process nodes
    for (size_t i = 0; i < p.nodes.size(); ++i)
    {
        Node n = p.nodes[i];

        // Remap inputs
        n.a = remap(remap_table, n.a);
        n.b = remap(remap_table, n.b);

        // Remap args for Scan/ScanVJP
        // (We don't modify p.args; we just use remapped values when building key)

        // Build CSE key based on op type
        CseKey key;

        switch (n.op)
        {
            case OpTag::Const:
                // Key by the constant value (via attr index)
                key = std::make_tuple(n.op, (ValueID)0, (ValueID)0, n.attr);
                break;

            case OpTag::Neg:
            case OpTag::ZeroLike:
            case OpTag::Detach:
            case OpTag::Exp:
            case OpTag::Log:
            case OpTag::Sum:
                key = std::make_tuple(n.op, n.a, (ValueID)0, 0u);
                break;

            case OpTag::Expand:
                key = std::make_tuple(n.op, n.a, n.b, 0u);
                break;

            case OpTag::Call:
            case OpTag::Custom:
            case OpTag::CustomVJP:
                // Don't CSE these (opaque, complex)
                out.nodes.push_back(n);
                continue;

            case OpTag::Add:
            case OpTag::Mul:
                // Commutative: normalize order
                if (n.a > n.b)
                {
                    ValueID tmp = n.a;
                    n.a = n.b;
                    n.b = tmp;
                }
                key = std::make_tuple(n.op, n.a, n.b, 0u);
                break;

            case OpTag::Div:
                // Not commutative
                key = std::make_tuple(n.op, n.a, n.b, 0u);
                break;

            case OpTag::Scan:
            case OpTag::ScanVJP:
                // Don't CSE Scan nodes (complex, rarely duplicated)
                out.nodes.push_back(n);
                continue;

            default:
                require(false && "cse: unhandled op");
        }

        // Check if we've seen this expression
        auto it = seen.find(key);
        if (it != seen.end())
        {
            // Already computed; remap this output to the existing one
            remap_table[(size_t)n.out] = it->second;
        }
        else
        {
            // First time seeing this expression; emit node
            seen[key] = n.out;
            out.nodes.push_back(n);
        }
    }

    // Remap outputs
    out.outputs.reserve(p.outputs.size());
    for (size_t i = 0; i < p.outputs.size(); ++i)
        out.outputs.push_back(remap(remap_table, p.outputs[i]));

    return out;
}

Program cse(const Program& p)
{
    return cse_impl(p);
}

// ============================================================
// Optimise
// ============================================================

static size_t count_nodes_recursive(const Program& p)
{
    size_t total = p.nodes.size();
    for (size_t i = 0; i < p.scan_bodies.size(); ++i)
        total += count_nodes_recursive(p.scan_bodies[i]);
    for (size_t i = 0; i < p.call_bodies.size(); ++i)
        total += count_nodes_recursive(p.call_bodies[i]);
    return total;
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
    const size_t fwd_before = count_nodes_recursive(cp.fwd);
    const size_t bwd_before = count_nodes_recursive(cp.bwd);

    // CSE first (merges duplicates), then DCE (removes dead code)
    cp.fwd = cse(cp.fwd);
    cp.fwd = dce(cp.fwd);

    cp.bwd = cse(cp.bwd);
    cp.bwd = dce(cp.bwd);

    const size_t fwd_after = count_nodes_recursive(cp.fwd);
    const size_t bwd_after = count_nodes_recursive(cp.bwd);

    std::printf("fwd: %zu -> %zu nodes (-%zu)\n", fwd_before, fwd_after, fwd_before - fwd_after);
    std::printf("bwd: %zu -> %zu nodes (-%zu)\n", bwd_before, bwd_after, bwd_before - bwd_after);

    compute_save_set(cp);
    std::printf("save_set: %zu values (of %u fwd values)\n",
                cp.save_set.size(), cp.fwd.num_values);
}
