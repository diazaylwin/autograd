#ifndef OWNED_TENSOR_H
#define OWNED_TENSOR_H

#include <cstdint>
#include <vector>
#include <cstring>  // std::memcpy

#include "runtime.h"
#include "utils.h"

// Owned, contiguous tensor snapshot (same shape/stride convention as Tensor).
// Scalars are rank-1 tensors with sizes={1}.
struct OwnedTensor
{
    uint64_t offset = 0;
    std::vector<uint64_t> sizes;
    std::vector<uint64_t> strides;
    std::vector<double>   data;

    inline bool is_scalar() const { return (sizes.size() == 1 && sizes[0] == 1); }
};

static inline uint64_t owned_numel(const std::vector<uint64_t>& sizes)
{
    require(sizes.size() >= 1);
    uint64_t n = 1;
    for (size_t i = 0; i < sizes.size(); ++i) n *= sizes[i];
    return n;
}

static inline uint64_t numel(const OwnedTensor& t)
{
    return owned_numel(t.sizes);
}

static inline double dot_owned(const OwnedTensor& a, const OwnedTensor& b)
{
    require(a.sizes == b.sizes);
    require(a.data.size() == b.data.size());
    uint64_t n = numel(a);
    double s = 0.0;
    for (uint64_t i = 0; i < n; ++i)
        s += a.data[i] * b.data[i];
    return s;
}

static inline void owned_contiguous_strides(std::vector<uint64_t>& strides, const std::vector<uint64_t>& sizes)
{
    require(sizes.size() >= 1);
    strides.resize(sizes.size());
    strides.back() = 1;
    for (int i = (int)sizes.size() - 2; i >= 0; --i)
        strides[(size_t)i] = strides[(size_t)i + 1] * sizes[(size_t)i + 1];
}

// Snapshot a runtime tensor into owned memory.
// v0: requires the runtime tensor to be contiguous with offset==0.
static inline OwnedTensor from_tensor(const Runtime& rt, const Tensor& t)
{
    require(t.sizes.size() >= 1);
    require(t.offset == 0);

    std::vector<uint64_t> expect;
    owned_contiguous_strides(expect, t.sizes);

    require(t.strides.size() == expect.size());
    for (size_t i = 0; i < expect.size(); ++i) require(t.strides[i] == expect[i]);

    OwnedTensor out;
    out.offset  = 0;
    out.sizes   = t.sizes;
    out.strides = t.strides;

    const uint64_t n = owned_numel(out.sizes);
    out.data.resize((size_t)n);

    const double* p = rt.base(t.storage);
    std::memcpy(out.data.data(), p, sizeof(double) * (size_t)n);

    return out;
}

static inline OwnedTensor ones_like(const OwnedTensor& x)
{
    OwnedTensor y;
    y.offset  = 0;
    y.sizes   = x.sizes;
    y.strides = x.strides;
    y.data.resize(x.data.size());

    for (size_t i = 0; i < y.data.size(); ++i)
        y.data[i] = 1.0;

    return y;
}

static inline OwnedTensor owned_from_vector(
    const std::vector<uint64_t>& sizes,
    const std::vector<double>& data
)
{
    OwnedTensor t;
    t.offset = 0;
    t.sizes  = sizes;
    owned_contiguous_strides(t.strides, t.sizes);

    require(owned_numel(t.sizes) == data.size());
    t.data = data;
    return t;
}

#endif // OWNED_TENSOR_H
