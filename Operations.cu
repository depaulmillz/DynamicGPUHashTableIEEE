#include "gpuErrchk.cu"
#include <cstdio>
#include <cuda_runtime.h>
#include <nvfunctional>

#define SEARCH_NOT_FOUND 0
#define ADDRESS_LANE 31
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

Slab **slabs = NULL;
__managed__ unsigned num_of_buckets = 0;

__host__ __device__ unsigned hash(unsigned src_key);

__device__ unsigned long long ReadSlab(const unsigned long long &next, const unsigned &src_bucket, const unsigned laneId, Slab **slabs, unsigned num_of_buckets) {
  if (src_bucket >= num_of_buckets) {
    printf("Error\n");
  }
  if (laneId != 31) {
    // printf("%d : %d\n", laneId, slabs[src_bucket][next - 1].keyValue[laneId]);
    return slabs[src_bucket][next - 1].keyValue[laneId];
  }
  // printf("%d : %d\n", laneId, slabs[src_bucket][next - 1].next);
  return *slabs[src_bucket][next - 1].next;
}

__device__ unsigned long long *SlabAddress(const unsigned long long &next, const unsigned &src_bucket, const unsigned laneId, Slab **slabs, unsigned num_of_buckets) {
  // printf("Reading slabs[%d][%d]\n", src_bucket, next - 1);
  if (src_bucket >= num_of_buckets) {
    printf("Error\n");
  }
  if (laneId != 31) {
    return (slabs[src_bucket][next - 1].keyValue + laneId);
  } else {
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
                               const nvstd::function<void(volatile bool *, volatile unsigned *, volatile unsigned *, unsigned &, unsigned &, unsigned &, unsigned long long &, unsigned long long &, Slab **, unsigned)> &operation, Slab **slabs,
                               unsigned num_of_buckets) {
  const int tid = threadIdx.x + blockDim.x * blockIdx.x;
  const unsigned laneId = threadIdx.x % 32;
  unsigned long long next = BASE_SLAB;
  volatile unsigned work_queue = __ballot_sync(~0, is_active[tid]);
  if (is_active[tid]) {
    printf("Work QUEUE %d\n", work_queue);
  }
  volatile unsigned last_work_queue = work_queue;

  while (work_queue != 0) {
    next = (work_queue != last_work_queue) ? (BASE_SLAB) : next;
    unsigned src_lane = __ffs(work_queue);
    unsigned src_key = __shfl_sync(~0, myKey[tid], src_lane);
    unsigned src_bucket = hash(src_key);
    unsigned long long read_data = ReadSlab(next, src_bucket, laneId, slabs, num_of_buckets);
    // printf("Read data %d\n", read_data);
    if (is_active[tid]) {
      // printf("next %d, src_lane %d, src_key %d, src_bucket %d, read_data %d\n", next, src_lane, src_key, src_bucket, read_data);
    }

    operation(is_active, myKey, myValue, src_lane, src_key, src_bucket, read_data, next, slabs, num_of_buckets);
    last_work_queue = work_queue;
    if (is_active[tid]) {
      // printf("Still active %d\n", tid);
    }
    bool activity = is_active[tid];
    work_queue = __ballot_sync(~0, activity);
    // printf("%d\n", work_queue);
  }
}

__device__ void warp_search(volatile bool *is_active, volatile unsigned *myKey, volatile unsigned *myValue, const unsigned &src_lane, const unsigned &src_key, const unsigned &src_bucket, const unsigned long long &read_data, unsigned long long &next,
                            Slab **slabs, unsigned num_of_buckets) {
  const int tid = threadIdx.x + blockDim.x * blockIdx.x;
  const unsigned laneId = threadIdx.x % 32;
  unsigned found_lane = __ffs(__ballot_sync(VALID_KEY_MASK, read_data == src_key));
  if (laneId == 0) {
    printf("Found lane %d\n");
  }
  if (found_lane != 0) {
    unsigned long long found_value = __shfl_sync(~0, read_data, found_lane + 1);
    if (laneId == src_lane - 1) {
      myValue[tid] = found_value & 0xffffffff;
      is_active[tid] = false;
    }
  } else {
    unsigned next_ptr = __shfl_sync(~0, read_data, ADDRESS_LANE);
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

__device__ void warp_delete(volatile bool *is_active, volatile unsigned *myKey, volatile unsigned *myValue, unsigned &src_lane, unsigned &src_key, unsigned &src_bucket, unsigned long long &read_data, unsigned long long &next, Slab **slabs,
                            unsigned num_of_buckets) {
  const int tid = threadIdx.x + blockDim.x * blockIdx.x;
  const unsigned laneId = threadIdx.x % 32;
  unsigned dest_lane = __ffs(__ballot_sync(VALID_KEY_MASK, read_data == src_key));
  if (dest_lane != 0) {
    if (src_lane - 1 == laneId) {
      *(SlabAddress(next, src_bucket, src_lane, slabs, num_of_buckets)) = DELETED_KEY;
      is_active[tid] = false;
    }
  } else {
    unsigned long long next_ptr = __shfl_sync(~0, read_data, ADDRESS_LANE);
    if (next_ptr == 0) {
      is_active[tid] = false;
    } else {
      next = next_ptr;
    }
  }
}

__device__ void warp_replace(volatile bool *is_active, volatile unsigned *myKey, volatile unsigned *myValue, unsigned &src_lane, unsigned &src_key, unsigned &src_bucket, unsigned long long &read_data, unsigned long long &next, Slab **slabs,
                             unsigned num_of_buckets) {
  const int tid = threadIdx.x + blockDim.x * blockIdx.x;
  const unsigned laneId = threadIdx.x % 32;
  // printf("Here %d\n", laneId);
  bool to_share = (read_data == EMPTY || ((read_data >> 32) & 0xffffffff) == myKey[tid]);
  // printf("to_share : %d\n", to_share);

  int masked_ballot = __ballot_sync(~0, to_share) & VALID_KEY_MASK;

  // printf("masked_ballot : %d\n", masked_ballot);

  unsigned dest_lane = (unsigned)__ffs(masked_ballot);

  if (dest_lane != 0) {
    if (src_lane - 1 == laneId) {
      unsigned long long key = (unsigned long long)myKey[tid];
      unsigned long long value = (unsigned long long)myValue[tid];
      unsigned long long newPair = (key << 32) | value;
      // printf("New pair %lld\n", newPair);
      unsigned long long *addr = SlabAddress(next, src_bucket, src_lane, slabs, num_of_buckets);
      unsigned long long old_pair = atomicCAS(addr, 0, newPair);
      if (old_pair == 0) {
        printf("Lane %d is operating\n", src_lane);
        printf("Success\n");
        is_active[tid] = false;
        printf("%d\n", is_active[tid]);
        __threadfence();
      }
      // printf("Old pair %lld\n", old_pair);
    }
  } else {
    unsigned long long next_ptr = __shfl_sync(~0, read_data, ADDRESS_LANE);
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
    printf("slabs[%d] has ptr %p\n", i, slabs[i]);
    for (int k = 0; k < numberOfSlabsPerBucket; k++) {
      gpuErrchk(cudaMallocManaged(&(slabs[i][k].keyValue), sizeof(long) * 31));
      gpuErrchk(cudaMallocManaged(&(slabs[i][k].next), sizeof(long)));

      for (int j = 0; j < 31; j++) {
        slabs[i][k].keyValue[j] = 0; // EMPTY_PAIR;
        printf("Slab[%d][%d] has keyvalue %lld\n", i, k, slabs[i][k].keyValue[j]);
      }
      if (k < numberOfSlabsPerBucket - 1) {
        *slabs[i][k].next = k + 1;
        printf("Slab[%d][%d] has next %lld which is k + 1\n", i, k, *slabs[i][k].next);
      } else {
        *slabs[i][k].next = 0; // EMPTY_POINTER;
        printf("Slab[%d][%d] has next %lld which is an empty ptr\n", i, k, *slabs[i][k].next);
      }
    }
  }
}