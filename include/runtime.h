#ifndef RUNTIME_H
#define RUNTIME_H

#include <cstdint>
#include <vector>

#include "utils.h"
#include "types.h"

// Want to make the simplest possible thing (i.e. operating on tensors == doubles) for the time being just to lock in the allocation/api
// All runtime kernels return fresh contiguous tensors.
// Scalars = rank-1 tensors

struct Tensor 
{
    StorageID storage = 0;
    uint64_t  offset  = 0;
    std::vector<uint64_t> sizes;
    std::vector<uint64_t> strides;

    // later: dtype, device, layout flags, etc.

    inline bool is_scalar() const {return (sizes.size() == 1 && sizes[0] == 1);}// lock in scalar = rank-1 tensors
};

struct Storage 
{
    uint64_t offset = 0;  // in doubles
    uint64_t numel  = 0;
};

struct Runtime 
{
    double* heap = nullptr;
    uint64_t cap = 0, top = 0;
    std::vector<Storage> storages;

    StorageID alloc(const uint64_t numel); // reserves numel consecutive doubles, returns its ID
    inline double* base(const StorageID id) // write 
    {
        require(heap != nullptr); 
        require(id < (StorageID)storages.size());
        return heap + storages[id].offset;
    }
    inline const double* base(const StorageID id) const // read
    {
        require(heap != nullptr); 
        require(id < (StorageID)storages.size());
        return heap + storages[id].offset;
    }
};

// Lifecycle Management 
void runtime_init(Runtime* rt, const uint64_t heap_cap);
void runtime_reset(Runtime* rt);
void runtime_free(Runtime* rt);

// Tensor creation (contiguous materialisation)
Tensor empty(Runtime& rt, const std::vector<uint64_t>& sizes);
Tensor from_vector(Runtime& rt, const std::vector<uint64_t>& sizes, const std::vector<double>& data);

// Elementwise ops (materialise outputs)
Tensor add(Runtime& rt, const Tensor& a, const Tensor& b);
Tensor mul(Runtime& rt, const Tensor& a, const Tensor& b);
Tensor div(Runtime& rt, const Tensor& a, const Tensor& b);
Tensor neg(Runtime& rt, const Tensor& a);
Tensor exp(Runtime& rt, const Tensor& a);
Tensor log(Runtime& rt, const Tensor& a);

// Reductions
Tensor sum(Runtime& rt, const Tensor& a);  // returns scalar {1}

// Broadcasting
Tensor expand(Runtime& rt, const Tensor& scalar, const Tensor& like);  // broadcast scalar to shape of like

// Accumulation + copy
void add_inplace(Runtime& rt, Tensor& dst, const Tensor& src);
Tensor clone(Runtime& rt, const Tensor& x);
Tensor contiguous(Runtime& rt, const Tensor& x);
Tensor zero_like(Runtime& rt, const Tensor& x);
Tensor detach(Runtime& rt, const Tensor& x);

#endif // RUNTIME_H