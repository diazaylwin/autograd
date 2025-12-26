// runtime.cpp (contiguity contract + SIMD-friendly hot loops)

#include "runtime.h"
#include "utils.h"

#include <cstdlib>  // std::malloc, std::free
#include <cstring>  // std::memcpy

void runtime_init(Runtime* rt, const uint64_t heap_cap) {
    require(rt != nullptr);
    rt->heap = (double*)std::malloc(sizeof(double) * heap_cap);
    require(rt->heap != nullptr);

    rt->cap = heap_cap;
    rt->top = 0;
    rt->storages.clear();
}

void runtime_reset(Runtime* rt) {
    require(rt != nullptr);
    rt->top = 0;
    rt->storages.clear();
}

void runtime_free(Runtime* rt) {
    if (!rt) return;
    std::free(rt->heap);
    rt->heap = nullptr;
    rt->cap = 0;
    rt->top = 0;
    rt->storages.clear();
}

StorageID Runtime::alloc(const uint64_t numel) {
    require(heap != nullptr);
    // overflow-safe
    require(numel <= cap - top);

    Storage s;
    s.offset = top;
    s.numel  = numel;

    storages.push_back(s);
    StorageID id = (StorageID)(storages.size() - 1);

    top += numel;
    return id;
}

Tensor empty(Runtime& rt, const std::vector<uint64_t>& sizes)
{
    require(sizes.size() >= 1); // forbid rank-0 tensors

    Tensor t;

    // 1) Record logical shape
    t.sizes = sizes;

    // 2) Compute contiguous (row-major) strides
    t.strides.resize(t.sizes.size());
    t.strides.back() = 1;
    for (int i = (int)t.sizes.size() - 2; i >= 0; --i)
    {
        t.strides[(size_t)i] = t.strides[(size_t)i + 1] * t.sizes[(size_t)i + 1];
    }

    // 3) Allocate storage
    uint64_t numel = 1;
    for (size_t i = 0; i < t.sizes.size(); ++i) numel *= t.sizes[i];

    t.storage = rt.alloc(numel);
    t.offset  = 0;
    return t;
}

Tensor from_vector(Runtime& rt,
                   const std::vector<uint64_t>& sizes,
                   const std::vector<double>& data)
{
    require(sizes.size() >= 1); // forbid rank-0

    uint64_t n = 1;
    for (size_t i = 0; i < sizes.size(); ++i) n *= sizes[i];
    require(data.size() == n);

    Tensor t = empty(rt, sizes);     // contiguous, offset=0
    std::memcpy(rt.base(t.storage), data.data(), sizeof(double) * n);
    return t;
}

// ----------------------------------------------------------------------------------------
// small internal helpers
// ----------------------------------------------------------------------------------------

static inline uint64_t numel_from_sizes(const std::vector<uint64_t>& sizes)
{
    uint64_t n = 1;
    for (size_t i = 0; i < sizes.size(); ++i) n *= sizes[i];
    return n;
}

static inline void require_well_formed(const Tensor& t)
{
    require(t.sizes.size() == t.strides.size());
    require(t.sizes.size() >= 1); // no rank-0 tensors
}

static inline void require_same_shape(const Tensor& a, const Tensor& b)
{
    require_well_formed(a);
    require_well_formed(b);
    require(a.sizes.size() == b.sizes.size());
    for (size_t i = 0; i < a.sizes.size(); ++i)
    {
        require(a.sizes[i] == b.sizes[i]);
    }
}

static inline bool is_contiguous(const Tensor& t)
{
    // basic sanity
    if (t.sizes.size() == 0) return false;
    if (t.sizes.size() != t.strides.size()) return false;

    // enforce offset == 0 for now (strict contract)
    if (t.offset != 0) return false;

    // row-major stride check
    uint64_t expected = 1;
    for (int k = (int)t.sizes.size() - 1; k >= 0; --k)
    {
        if (t.strides[(size_t)k] != expected) return false;
        expected *= t.sizes[(size_t)k];
    }
    return true;
}

static inline const double* ptr(const Runtime& rt, const Tensor& t)
{
    // If later you relax is_contiguous to allow offset!=0, this still works.
    return rt.base(t.storage) + t.offset;
}

static inline double* ptr(Runtime& rt, const Tensor& t)
{
    return rt.base(t.storage) + t.offset;
}

// Keep this around for later view support / materialisation if you want.
// Not used in hot loops under the contiguity contract.
static inline uint64_t storage_index_from_linear(const Tensor& t, uint64_t lin)
{
    uint64_t idx = t.offset;
    for (int k = (int)t.sizes.size() - 1; k >= 0; --k) {
        const uint64_t dim = t.sizes[(size_t)k];
        const uint64_t ik  = (dim == 0) ? 0 : (lin % dim);
        lin = (dim == 0) ? 0 : (lin / dim);
        idx += ik * t.strides[(size_t)k];
    }
    return idx;
}

Tensor contiguous(Runtime& rt, const Tensor& x)
{
    require_well_formed(x);

    if (is_contiguous(x)) return x;       // or clone(rt,x) if you want value semantics

    Tensor out = empty(rt, x.sizes);      // contiguous output
    const uint64_t n = numel_from_sizes(out.sizes);

    const double* xp = rt.base(x.storage);
    double*       op = rt.base(out.storage);

    for (uint64_t lin = 0; lin < n; ++lin) {
        op[lin] = xp[storage_index_from_linear(x, lin)];
    }
    return out;
}

// ----------------------------------------------------------------------------------------
// Elementwise ops (contiguous-only, SIMD-friendly)
// ----------------------------------------------------------------------------------------

Tensor neg(Runtime& rt, const Tensor& a)
{
    require_well_formed(a);
    require(is_contiguous(a));

    Tensor out = empty(rt, a.sizes);
    const uint64_t n = numel_from_sizes(out.sizes);

    const double* ap = ptr(rt, a);
    double*       op = ptr(rt, out);

    for (uint64_t i = 0; i < n; ++i) op[i] = -ap[i];
    return out;
}

Tensor add(Runtime& rt, const Tensor& a, const Tensor& b)
{
    require_same_shape(a, b);
    require(is_contiguous(a) && is_contiguous(b));

    Tensor out = empty(rt, a.sizes);
    const uint64_t n = numel_from_sizes(out.sizes);

    const double* ap = ptr(rt, a);
    const double* bp = ptr(rt, b);
    double*       op = ptr(rt, out);

    for (uint64_t i = 0; i < n; ++i) op[i] = ap[i] + bp[i];
    return out;
}

Tensor mul(Runtime& rt, const Tensor& a, const Tensor& b)
{
    require_same_shape(a, b);
    require(is_contiguous(a) && is_contiguous(b));

    Tensor out = empty(rt, a.sizes);
    const uint64_t n = numel_from_sizes(out.sizes);

    const double* ap = ptr(rt, a);
    const double* bp = ptr(rt, b);
    double*       op = ptr(rt, out);

    for (uint64_t i = 0; i < n; ++i) op[i] = ap[i] * bp[i];
    return out;
}

Tensor div(Runtime& rt, const Tensor& a, const Tensor& b)
{
    require_same_shape(a, b);
    require(is_contiguous(a) && is_contiguous(b));

    Tensor out = empty(rt, a.sizes);
    const uint64_t n = numel_from_sizes(out.sizes);

    const double* ap = ptr(rt, a);
    const double* bp = ptr(rt, b);
    double*       op = ptr(rt, out);

    for (uint64_t i = 0; i < n; ++i) op[i] = ap[i] / bp[i];
    return out;
}

// ----------------------------------------------------------------------------------------
// Accumulation + copy (contiguous-only, SIMD-friendly)
// ----------------------------------------------------------------------------------------

void add_inplace(Runtime& rt, Tensor& dst, const Tensor& src)
{
    require_same_shape(dst, src);
    require(is_contiguous(dst) && is_contiguous(src));

    const uint64_t n = numel_from_sizes(dst.sizes);

    double*       dp = ptr(rt, dst);
    const double* sp = ptr(rt, src);

    for (uint64_t i = 0; i < n; ++i) dp[i] += sp[i];
}

Tensor clone(Runtime& rt, const Tensor& x)
{
    require_well_formed(x);
    require(is_contiguous(x));

    Tensor out = empty(rt, x.sizes);
    const uint64_t n = numel_from_sizes(out.sizes);

    const double* xp = ptr(rt, x);
    double*       op = ptr(rt, out);

    std::memcpy(op, xp, sizeof(double) * n);
    return out;
}

Tensor zero_like(Runtime& rt, const Tensor& x)
{
    require_well_formed(x);
    require(is_contiguous(x));

    Tensor out = empty(rt, x.sizes);
    const uint64_t n = numel_from_sizes(out.sizes);

    double* op = ptr(rt, out);
    for (uint64_t i = 0; i < n; ++i) op[i] = 0.0;

    return out;
}