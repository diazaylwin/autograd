#ifndef OPTIMISATION_H
#define OPTIMISATION_H

#include <vector>

#include "types.h"
#include "symbols.h"

// ============================================================
// CompiledProgram: forward + backward pair
// ============================================================
//
// Usage:
//   CompiledProgram cp;
//   cp.fwd = Finalize(...);
//   cp.bwd = build_vjp(cp.fwd);
//   Optimise(cp);
//

struct CompiledProgram
{
    Program fwd;
    Program bwd;

    // Minimal save-set: which fwd ValueIDs does bwd actually need?
    std::vector<ValueID> save_set;
};

// ============================================================
// Individual passes (Program -> Program)
// ============================================================

// Dead code elimination: removes nodes not needed to compute outputs.
Program dce(const Program& p);

// Common subexpression elimination: merges duplicate computations.
Program cse(const Program& p);

// ============================================================
// Main entry point
// ============================================================

// Optimises a CompiledProgram in place:
//   1. DCE on fwd
//   2. DCE on bwd
//   3. Computes minimal save_set
void Optimise(CompiledProgram& cp);

#endif // OPTIMISATION_H
