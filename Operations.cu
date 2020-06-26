#include "gpuErrchk.cu"
#include <cstdio>
#include <cuda_runtime.h>
#include <nvfunctional>

#define SEARCH_NOT_FOUND 0
#define ADDRESS_LANE 32
#define VALID_KEY_MASK 0x7fffffff
#define DELETED_KEY 0

const long long unsigned EMPTY = 0;
const long long unsigned EMPTY_PAIR = 0;
const long long unsigned EMPTY_POINTER = 0;
#define BASE_SLAB 1

struct Slab {
  unsigned long long *keyValue;
  unsigned long long *next;
};

volatile Slab **slabs = NULL;
__managed__ unsigned num_of_buckets = 0;

__host__ __device__ unsigned hash(unsigned src_key);

__forceinline__ __device__ unsigned long long
ReadSlab(const unsigned long long &next, const unsigned &src_bucket,
         const unsigned laneId, volatile Slab **slabs,
         unsigned num_of_buckets) {
  if (src_bucket >= num_of_buckets) {
    printf("Error\n");
  }
  if (laneId != 31) {
    return slabs[src_bucket][next - 1].keyValue[laneId];
  }
  return *slabs[src_bucket][next - 1].next;
}

__forceinline__ __device__ unsigned long long *
SlabAddress(const unsigned long long &next, const unsigned &src_bucket,
            const unsigned laneId, volatile Slab **slabs,
            unsigned num_of_buckets) {
  if (src_bucket >= num_of_buckets) {
    printf("Error\n");
  }
  if (laneId != 31) {
    // printf("Got addr %p src_bucket %d\n", slabs[src_bucket][next -
    // 1].keyValue + laneId, src_bucket);

    return (slabs[src_bucket][next - 1].keyValue + laneId);
  } else {
    // printf("Got addr for next\n");
    return slabs[src_bucket][next - 1].next;
  }
}

__forceinline__ __device__ unsigned long long warp_allocate() {
  printf("Didn't implement\n");
  return 0;
}
__forceinline__ __device__ unsigned long long deallocate(unsigned long long l) {
  printf("Didn't implement\n");
  return 0;
}

__forceinline__ __device__ void warp_operation(
    bool *__restrict__ is_active, const unsigned *__restrict__ myKey,
    unsigned *__restrict__ myValue,
    const nvstd::function<void(bool *__restrict__, const unsigned *__restrict__,
                               unsigned *__restrict__, unsigned &, unsigned &,
                               unsigned &, unsigned long long &,
                               unsigned long long &, volatile Slab **,
                               unsigned)> &operation,
    volatile Slab **slabs, unsigned num_of_buckets) {
  const int tid = threadIdx.x + blockDim.x * blockIdx.x;
  const unsigned laneId = threadIdx.x & 0x1F;
  unsigned long long next = BASE_SLAB;
  unsigned work_queue = __ballot_sync(~0, is_active[tid]);

  unsigned last_work_queue = work_queue;

  while (work_queue != 0) {
    next = (work_queue != last_work_queue) ? (BASE_SLAB) : next;
    unsigned src_lane = __ffs(work_queue);
    unsigned src_key = __shfl_sync(~0, myKey[tid], src_lane - 1);
    unsigned src_bucket = hash(src_key);
    // if (laneId == 0)
    //  printf("src_lane %d from %d: %d -> %d\n", src_lane, work_queue, src_key,
    //  src_bucket);
    unsigned long long read_data =
        ReadSlab(next, src_bucket, laneId, slabs, num_of_buckets);

    operation(is_active, myKey, myValue, src_lane, src_key, src_bucket,
              read_data, next, slabs, num_of_buckets);
    last_work_queue = work_queue;
    bool activity = is_active[tid];

    work_queue = __ballot_sync(~0, activity);
  }
}

__forceinline__ __device__ void
warp_search(bool *__restrict__ is_active, const unsigned *__restrict__ myKey,
            unsigned *__restrict__ myValue, unsigned &src_lane,
            unsigned &src_key, unsigned &src_bucket,
            unsigned long long &read_data, unsigned long long &next,
            volatile Slab **slabs, unsigned num_of_buckets) {
  const int tid = threadIdx.x + blockDim.x * blockIdx.x;
  const unsigned laneId = threadIdx.x % 32;
  unsigned key = (unsigned)((read_data >> 32) & 0xffffffff);
  unsigned found_lane =
      __ffs(__ballot_sync(~0, key == src_key) & VALID_KEY_MASK);

  if (found_lane != 0) {
    unsigned long long found_value = __shfl_sync(~0, read_data, found_lane - 1);
    if (laneId == src_lane - 1) {
      myValue[tid] = found_value & 0xffffffff;
      is_active[tid] = false;
    }
  } else {
    unsigned long long next_ptr = __shfl_sync(~0, read_data, ADDRESS_LANE - 1);
    if (next_ptr == 0) {
      if (laneId == src_lane - 1) {
        myValue[tid] = SEARCH_NOT_FOUND;
        is_active[tid] = false;
      }
    } else {
      next = next_ptr;
    }
  }
}

__forceinline__ __device__ void
warp_delete(bool *__restrict__ is_active, const unsigned *__restrict__ myKey,
            unsigned *__restrict__ myValue, unsigned &src_lane,
            unsigned &src_key, unsigned &src_bucket,
            unsigned long long &read_data, unsigned long long &next,
            volatile Slab **slabs, unsigned num_of_buckets) {
  const int tid = threadIdx.x + blockDim.x * blockIdx.x;
  const unsigned laneId = threadIdx.x % 32;
  unsigned key = (unsigned)((read_data >> 32) & 0xffffffff);
  unsigned dest_lane = __ffs(__ballot_sync(VALID_KEY_MASK, key == src_key));
  if (dest_lane != 0) {
    if (src_lane - 1 == laneId) {
      *(SlabAddress(next, src_bucket, dest_lane - 1, slabs, num_of_buckets)) =
          DELETED_KEY;
      is_active[tid] = false;
    }
  } else {
    unsigned long long next_ptr = __shfl_sync(~0, read_data, ADDRESS_LANE - 1);
    if (next_ptr == 0) {
      is_active[tid] = false;
    } else {
      next = next_ptr;
    }
  }
}

__forceinline__ __device__ void
warp_replace(bool *__restrict__ is_active, const unsigned *__restrict__ myKey,
             unsigned *__restrict__ myValue, unsigned &src_lane,
             unsigned &src_key, unsigned &src_bucket,
             unsigned long long &read_data, unsigned long long &next,
             volatile Slab **slabs, unsigned num_of_buckets) {
  const int tid = threadIdx.x + blockDim.x * blockIdx.x;
  const unsigned laneId = threadIdx.x % 32;
  unsigned key = (unsigned)((read_data >> 32) & 0xffffffff);
  bool to_share = (key == EMPTY || key == src_key);
  int masked_ballot = __ballot_sync(~0, to_share) & VALID_KEY_MASK;
  unsigned dest_lane = (unsigned)__ffs(masked_ballot);

  if (dest_lane != 0) {
    if (src_lane - 1 == laneId) {
      unsigned long long key = (unsigned long long)myKey[tid];
      unsigned long long value = (unsigned long long)myValue[tid];
      unsigned long long newPair = (key << 32) | value;
      unsigned long long *addr =
          SlabAddress(next, src_bucket, dest_lane - 1, slabs, num_of_buckets);
      unsigned long long old_pair = atomicCAS(addr, 0, newPair);
      if (old_pair == 0) {
        // printf("%d inserted\n", tid);
        is_active[tid] = false;
        __threadfence();
      } else if ((unsigned)((old_pair >> 32) & 0xffffffff) == key) {
        is_active[tid] = false;
        __threadfence();
        // printf("%d %d tried to insert but got %lld\n", tid, laneId,
        // ((old_pair >> 32) & 0xffffffff));
      }
    }
  } else {
    unsigned long long next_ptr = __shfl_sync(~0, read_data, ADDRESS_LANE - 1);
    if (next_ptr == 0) {
      unsigned long long new_slab_ptr = warp_allocate();
      if (laneId == ADDRESS_LANE) {
        unsigned long long temp = 0;
        temp = atomicCAS(
            SlabAddress(next, src_bucket, ADDRESS_LANE, slabs, num_of_buckets),
            EMPTY_POINTER, new_slab_ptr);
        if (temp != EMPTY_POINTER) {
          deallocate(new_slab_ptr);
        }
      }
    } else {
      next = next_ptr;
    }
  }
}

__forceinline__ __host__ __device__ unsigned hash(unsigned src_key) {
  return src_key % num_of_buckets;
}

void setUp(unsigned size, unsigned numberOfSlabsPerBucket) {
  num_of_buckets = size;
  gpuErrchk(cudaMallocManaged(&slabs, sizeof(Slab *) * size));
  for (int i = 0; i < size; i++) {

    gpuErrchk(
        cudaMallocManaged(&(slabs[i]), sizeof(Slab) * numberOfSlabsPerBucket));
    for (int k = 0; k < numberOfSlabsPerBucket; k++) {
      gpuErrchk(cudaMallocManaged(
          (unsigned long long **)&(slabs[i][k].keyValue), sizeof(long) * 31));
      gpuErrchk(cudaMallocManaged((unsigned long long **)&(slabs[i][k].next),
                                  sizeof(long)));

      for (int j = 0; j < 31; j++) {
        slabs[i][k].keyValue[j] = 0; // EMPTY_PAIR;
      }
      if (k < numberOfSlabsPerBucket - 1) {
        *slabs[i][k].next = (long)(k + 2);
      } else {
        *slabs[i][k].next = 0; // EMPTY_POINTER;
      }
    }
  }
}