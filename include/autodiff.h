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
    std::vector<Tensor>  adj;      // size = prog.num_values (optional)
    std::vector<uint8_t> has_adj;  // 0/1 flags (or use optional<Tensor>)
    // later: residuals per-node for nontrivial VJP (argmax masks, etc.)
};

std::vector<Tensor> execute
(
    const Program& prog,
    Runtime& rt,
    Tape& tape,
    const std::vector<Tensor>& input_tensors
);

std::vector<Tensor> backward
(
    const Program& prog,
    Runtime& rt,
    Tape& tape,
    const std::vector<Tensor>& output_seeds
);

#endif // AUTODIFF_H