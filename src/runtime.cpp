#include "runtime.h"
#include "utils.h"

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
    require (top + numel <= cap);

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

// -------------------------
// small internal helpers
// -------------------------

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

// Map a logical linear index (row-major over sizes) -> storage index using strides/offset.
static inline uint64_t storage_index_from_linear(const Tensor& t, uint64_t lin)
{
    uint64_t idx = t.offset;

    // decode lin into multi-index using sizes (row-major), then apply strides
    for (int k = (int)t.sizes.size() - 1; k >= 0; --k) {
        const uint64_t dim = t.sizes[(size_t)k];
        // dim should be >0 in sane tensors; if you allow 0-sized dims, handle separately
        const uint64_t ik = (dim == 0) ? 0 : (lin % dim);
        lin = (dim == 0) ? 0 : (lin / dim);
        idx += ik * t.strides[(size_t)k];
    }
    return idx;
}

// -------------------------
// Elementwise ops
// -------------------------

Tensor neg(Runtime& rt, const Tensor& a)
{
    require_well_formed(a);

    Tensor out = empty(rt, a.sizes);                // contiguous output
    const uint64_t n = numel_from_sizes(out.sizes);

    const double* ap = rt.base(a.storage);
    double* op = rt.base(out.storage);

    for (uint64_t lin = 0; lin < n; ++lin) 
    {
        const uint64_t ai = storage_index_from_linear(a, lin);
        op[lin] = -ap[ai];
    }
    return out;
}

Tensor add(Runtime& rt, const Tensor& a, const Tensor& b)
{
    require_same_shape(a, b);

    Tensor out = empty(rt, a.sizes);
    const uint64_t n = numel_from_sizes(out.sizes);

    const double* ap = rt.base(a.storage);
    const double* bp = rt.base(b.storage);
    double*       op = rt.base(out.storage);

    for (uint64_t lin = 0; lin < n; ++lin) 
    {
        const uint64_t ai = storage_index_from_linear(a, lin);
        const uint64_t bi = storage_index_from_linear(b, lin);
        op[lin] = ap[ai] + bp[bi];
    }
    return out;
}

Tensor mul(Runtime& rt, const Tensor& a, const Tensor& b)
{
    require_same_shape(a, b);

    Tensor out = empty(rt, a.sizes);
    const uint64_t n = numel_from_sizes(out.sizes);

    const double* ap = rt.base(a.storage);
    const double* bp = rt.base(b.storage);
    double* op = rt.base(out.storage);

    for (uint64_t lin = 0; lin < n; ++lin) 
    {
        const uint64_t ai = storage_index_from_linear(a, lin);
        const uint64_t bi = storage_index_from_linear(b, lin);
        op[lin] = ap[ai] * bp[bi];
    }
    return out;
}

Tensor div(Runtime& rt, const Tensor& a, const Tensor& b)
{
    require_same_shape(a, b);

    Tensor out = empty(rt, a.sizes);
    const uint64_t n = numel_from_sizes(out.sizes);

    const double* ap = rt.base(a.storage);
    const double* bp = rt.base(b.storage);
    double* op = rt.base(out.storage);

    for (uint64_t lin = 0; lin < n; ++lin) 
    {
        const uint64_t ai = storage_index_from_linear(a, lin);
        const uint64_t bi = storage_index_from_linear(b, lin);
        op[lin] = ap[ai] / bp[bi];
    }
    return out;
}

// -------------------------
// Accumulation + copy
// -------------------------

void add_inplace(Runtime& rt, Tensor& dst, const Tensor& src)
{
    require_same_shape(dst, src);

    const uint64_t n = numel_from_sizes(dst.sizes);

    double* dp = rt.base(dst.storage);
    const double* sp = rt.base(src.storage);

    for (uint64_t lin = 0; lin < n; ++lin) 
    {
        const uint64_t di = storage_index_from_linear(dst, lin);
        const uint64_t si = storage_index_from_linear(src, lin);
        dp[di] += sp[si];
    }
}

Tensor clone(Runtime& rt, const Tensor& x)
{
    require_well_formed(x);

    Tensor out = empty(rt, x.sizes);
    const uint64_t n = numel_from_sizes(out.sizes);

    const double* xp = rt.base(x.storage);
    double* op = rt.base(out.storage);

    // Even if x is contiguous, this generic path is correct for all strides.
    for (uint64_t lin = 0; lin < n; ++lin) 
    {
        const uint64_t xi = storage_index_from_linear(x, lin);
        op[lin] = xp[xi];
    }
    return out;
}
