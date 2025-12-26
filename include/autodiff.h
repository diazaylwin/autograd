#ifndef AUTODIFF_H
#define AUTODIFF_H

#include <cstdint>
#include <vector>

#include "types.h"
#include "runtime.h"
#include "symbols.h"

struct Tape 
{
    std::vector<Tensor>  primal;   // size = prog.num_values

    // Used by backward() (your v0 reverse sweep and/or other paths)
    std::vector<Tensor>  adj;      // size = prog.num_values
    std::vector<uint8_t> has_adj;  // 0/1 flags

    // v0 Scan support:
    // For each ValueID that is a Scan output, store carried states s0..sT.
    // Empty for non-Scan values.
    std::vector<std::vector<Tensor>> scan_states;
};

std::vector<Tensor> execute
(
    const Program& prog,
    Runtime& rt,
    Tape& tape,
    const std::vector<Tensor>& input_tensors
);

// Build a VJP (reverse-mode) program for `fwd`.
//
// Convention (v0):
//   bwd.inputs  = [ primals for ALL fwd values v=0..num_values-1, then output seeds ]
//   bwd.outputs = [ grads aligned with fwd.inputs ]
//
Program build_vjp(const Program& fwd);

std::vector<Tensor> backward
(
    const Program& prog,
    Runtime& rt,
    Tape& tape,
    const std::vector<Tensor>& output_seeds
);

#endif // AUTODIFF_H