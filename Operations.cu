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

__device__ unsigned long long ReadSlab(const volatile unsigned long long &next, const volatile unsigned &src_bucket, const unsigned laneId, volatile Slab **slabs, unsigned num_of_buckets) {
  if (src_bucket >= num_of_buckets) {
    printf("Error\n");
  }
  if (laneId != 31) {
    return slabs[src_bucket][next - 1].keyValue[laneId];
  }
  // printf("Reading next from %d next is %lld \n", src_bucket, *slabs[src_bucket][next - 1].next);
  return *slabs[src_bucket][next - 1].next;
}

__device__ unsigned long long *SlabAddress(const volatile unsigned long long &next, const volatile unsigned &src_bucket, const unsigned laneId, volatile Slab **slabs, unsigned num_of_buckets) {
  if (src_bucket >= num_of_buckets) {
    printf("Error\n");
  }
  if (laneId != 31) {
    // printf("Got addr %p src_bucket %d\n", slabs[src_bucket][next - 1].keyValue + laneId, src_bucket);

    return (slabs[src_bucket][next - 1].keyValue + laneId);
  } else {
    // printf("Got addr for next\n");
    return slabs[src_bucket][next - 1].next;
  }
}

__device__ unsigned long long warp_allocate() {
  printf("Didn't implement\n");
  return 0;
}
__device__ unsigned long long deallocate(unsigned long long l) {
  printf("Didn't implement\n");
  return 0;
}

__device__ void warp_operation(volatile bool *is_active, volatile unsigned *myKey, volatile unsigned *myValue,
                               const nvstd::function<void(volatile bool *, volatile unsigned *, volatile unsigned *, volatile unsigned &, volatile unsigned &, volatile unsigned &, volatile unsigned long long &, volatile unsigned long long &,
                                                          volatile Slab **, unsigned)> &operation,
                               volatile Slab **slabs, unsigned num_of_buckets) {
  const int tid = threadIdx.x + blockDim.x * blockIdx.x;
  const unsigned laneId = threadIdx.x % 32;
  volatile unsigned long long next = BASE_SLAB;
  volatile unsigned work_queue = __ballot_sync(~0, is_active[tid]);

  volatile unsigned last_work_queue = work_queue;

  while (work_queue != 0) {
    next = (work_queue != last_work_queue) ? (BASE_SLAB) : next;
    volatile unsigned src_lane = __ffs(work_queue);
    volatile unsigned src_key = __shfl_sync(~0, myKey[tid], src_lane - 1);
    volatile unsigned src_bucket = hash(src_key);
    // if (laneId == 0)
    //  printf("src_lane %d from %d: %d -> %d\n", src_lane, work_queue, src_key, src_bucket);
    volatile unsigned long long read_data = ReadSlab(next, src_bucket, laneId, slabs, num_of_buckets);

    operation(is_active, myKey, myValue, src_lane, src_key, src_bucket, read_data, next, slabs, num_of_buckets);
    last_work_queue = work_queue;
    bool activity = is_active[tid];

    work_queue = __ballot_sync(~0, activity);
  }
}

__device__ void warp_search(volatile bool *is_active, volatile unsigned *myKey, volatile unsigned *myValue, volatile unsigned &src_lane, volatile unsigned &src_key, volatile unsigned &src_bucket, volatile unsigned long long &read_data,
                            volatile unsigned long long &next, volatile Slab **slabs, unsigned num_of_buckets) {
  const int tid = threadIdx.x + blockDim.x * blockIdx.x;
  const unsigned laneId = threadIdx.x % 32;
  unsigned key = (unsigned)((read_data >> 32) & 0xffffffff);
  unsigned found_lane = __ffs(__ballot_sync(~0, key == src_key) & VALID_KEY_MASK);

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

__device__ void warp_delete(volatile bool *is_active, volatile unsigned *myKey, volatile unsigned *myValue, volatile unsigned &src_lane, volatile unsigned &src_key, volatile unsigned &src_bucket, volatile unsigned long long &read_data,
                            volatile unsigned long long &next, volatile Slab **slabs, unsigned num_of_buckets) {
  const int tid = threadIdx.x + blockDim.x * blockIdx.x;
  const unsigned laneId = threadIdx.x % 32;
  unsigned key = (unsigned)((read_data >> 32) & 0xffffffff);
  unsigned dest_lane = __ffs(__ballot_sync(VALID_KEY_MASK, key == src_key));
  if (dest_lane != 0) {
    if (src_lane - 1 == laneId) {
      *(SlabAddress(next, src_bucket, dest_lane - 1, slabs, num_of_buckets)) = DELETED_KEY;
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

__device__ void warp_replace(volatile bool *is_active, volatile unsigned *myKey, volatile unsigned *myValue, volatile unsigned &src_lane, volatile unsigned &src_key, volatile unsigned &src_bucket, volatile unsigned long long &read_data,
                             volatile unsigned long long &next, volatile Slab **slabs, unsigned num_of_buckets) {
  const int tid = threadIdx.x + blockDim.x * blockIdx.x;
  const unsigned laneId = threadIdx.x % 32;
  unsigned key = (unsigned)((read_data >> 32) & 0xffffffff);
  bool to_share = (key == EMPTY || key == myKey[tid]);
  int masked_ballot = __ballot_sync(~0, to_share) & VALID_KEY_MASK;
  unsigned dest_lane = (unsigned)__ffs(masked_ballot);

  if (dest_lane != 0) {
    if (src_lane - 1 == laneId) {
      unsigned long long key = (unsigned long long)myKey[tid];
      unsigned long long value = (unsigned long long)myValue[tid];
      unsigned long long newPair = (key << 32) | value;
      unsigned long long *addr = SlabAddress(next, src_bucket, dest_lane - 1, slabs, num_of_buckets);
      unsigned long long old_pair = atomicCAS(addr, 0, newPair);
      if (old_pair == 0 || (unsigned)((old_pair >> 32) & 0xffffffff) == key) {
        // printf("%d inserted\n", tid);
        is_active[tid] = false;
        __threadfence();
      } else {

        // printf("%d %d tried to insert but got %lld\n", tid, laneId, ((old_pair >> 32) & 0xffffffff));
      }
    }
  } else {
    unsigned long long next_ptr = __shfl_sync(~0, read_data, ADDRESS_LANE - 1);
    if (next_ptr == 0) {
      unsigned long long new_slab_ptr = warp_allocate();
      if (laneId == ADDRESS_LANE) {
        unsigned long long temp = 0;
        temp = atomicCAS(SlabAddress(next, src_bucket, ADDRESS_LANE, slabs, num_of_buckets), EMPTY_POINTER, new_slab_ptr);
        if (temp != EMPTY_POINTER) {
          deallocate(new_slab_ptr);
        }
      }
    } else {
      next = next_ptr;
    }
  }
}

__host__ __device__ unsigned hash(unsigned src_key) { return src_key % num_of_buckets; }

void setUp(unsigned size, unsigned numberOfSlabsPerBucket) {
  num_of_buckets = size;
  gpuErrchk(cudaMallocManaged(&slabs, sizeof(Slab *) * size));
  for (int i = 0; i < size; i++) {

    gpuErrchk(cudaMallocManaged(&(slabs[i]), sizeof(Slab) * numberOfSlabsPerBucket));
    for (int k = 0; k < numberOfSlabsPerBucket; k++) {
      gpuErrchk(cudaMallocManaged((unsigned long long **)&(slabs[i][k].keyValue), sizeof(long) * 31));
      gpuErrchk(cudaMallocManaged((unsigned long long **)&(slabs[i][k].next), sizeof(long)));

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