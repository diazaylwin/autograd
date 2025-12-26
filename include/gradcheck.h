#ifndef GRADCHECK_H
#define GRADCHECK_H

#include <cstdint>
#include <vector>

#include "symbols.h"        // Program, ValueID
#include "runtime.h"        // Runtime
#include "autodiff.h"       // Tape
#include "owned_tensor.h"   // OwnedTensor

enum GradcheckMode : uint8_t
{
    GRADCHECK_COORDINATE = 0,
    GRADCHECK_DIRECTIONAL = 1
};

struct GradcheckOptions
{
    double eps  = 1e-6;
    double rtol = 1e-5;
    double atol = 1e-8;

    uint32_t seed = 123u;
    GradcheckMode mode = GRADCHECK_COORDINATE;

    union
    {
        struct {
            uint32_t max_coords_per_input; // 0 = all
        } coord;

        struct {
            uint32_t num_directions;
        } dir;
    } u;
};

struct GradcheckReport
{
    bool ok = true;

    // For prefix scans: first failing node; otherwise UINT32_MAX.
    uint32_t node_index = UINT32_MAX;

    // Which input tensor was worst
    uint32_t input_index = 0;

    // In coordinate mode: flattened coordinate in that input.
    // In directional mode: direction index (0..num_directions-1).
    uint64_t index = 0;

    double ad      = 0.0;
    double fd      = 0.0;
    double abs_err = 0.0;
    double rel_err = 0.0;
};

// Whole-program gradcheck for the scalar objective:
//   F(inputs) = sum_j dot(outputs[j], seeds[j])
GradcheckReport gradcheck(
    const Program& prog,
    Runtime& rt,
    Tape& tape,
    const std::vector<OwnedTensor>& inputs,
    const std::vector<OwnedTensor>& seeds,
    const GradcheckOptions& opt
);

// Prefix gradcheck: treat internal ValueID `out` as the (only) output.
// `seed` must match the shape of that output.
GradcheckReport gradcheck_prefix(
    const Program& prog,
    ValueID out,
    Runtime& rt,
    Tape& tape,
    const std::vector<OwnedTensor>& inputs,
    const OwnedTensor& seed,
    const GradcheckOptions& opt
);

// Scan prefixes in topological order; return first failing prefix (or ok=true).
GradcheckReport gradcheck_prefixes_until_fail(
    const Program& prog,
    Runtime& rt,
    Tape& tape,
    const std::vector<OwnedTensor>& inputs,
    const GradcheckOptions& opt
);

#endif // GRADCHECK_H