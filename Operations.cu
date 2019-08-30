#define SEARCH_NOT_FOUND 0
#define ADDRESS_LANE 31
#define VALID_KEY_MASK 0xefffffff
#define DELETED_KEY 0
#define EMPTY 0
#define EMPTY_PAIR 0
#define EMPTY_POINTER 0

struct Slab {
  long keyValue[31];
  long next;
};

__device__ void warp_operation(bool &is_active, unsigned &myKey,
                               unsigned &myValue, nvstd::function<> operation) {
  const unsigned laneId = threadIdx.x % warp_size;
  next = BASE_SLAB;
  unsigned work_queue = __ballot_sync(~0, is_active);
  while (work_queue != 0) {
    next = (if work queue is changed) ? (BASE SLAB) : next;
    unsigned src_lane = __ffs(work_queue);
    unsigned src_key = __shfl_sync(~0, myKey, src_lane);
    unsigned src_bucket = hash(src_key);
    unsigned read_data = ReadSlab(next, src_bucket, laneId);
    operation(is_active, myKey, myValue);
    work_queue = __ballot_sync(~0, is_active);
  }
}

__device__ void warp_search(bool &is_active, unsigned &myKey, unsigned &myValue,
                            unsigned &src_lane, unsigned &src_key,
                            unsigned &src_bucket, unsigned &read_data) {
  const unsigned laneId = threadIdx.x % warp_size;
  unsigned found_lane =
      __ffs(__ballot_sync(VALID_KEY_MASK, read_data == src_key));
  if (found_lane != 0) {
    unsigned found_value = __shfl_sync(~0, read_data, found_lane + 1);
    if (laneId == src_lane) {
      myValue = found_value;
      is_active = false;
    }
  } else {
    unsigned next_ptr = __shfl_sync(~0, read_data, ADDRESS_LANE);
    if (next_ptr == 0) {
      if (laneId == src_lane) {
        myValue = SEARCH_NOT_FOUND;
        is_active = false;
      }
    } else {
      next = next_ptr;
    }
  }
}

__device__ void warp_delete(bool &is_active, unsigned &myKey, unsigned &myValue,
                            unsigned &src_lane, unsigned &src_key,
                            unsigned &src_bucket, unsigned &read_data) {
  const unsigned laneId = threadIdx.x % warp_size;
  dest_lane = __ffs(__ballot_sync(VALID_KEY_MASK, read_data == src_key));
  if (dest_lane != 0) {
    if (src_lane == landId) {
      *(SlabAddress(next, src_bucket, src_lane)) = DELETED_KEY;
      is_active = false;
    }
  } else {
    next_ptr = __shfl_sync(~0, read_data, ADDRESS_LANE);
    if (next_ptr == 0) {
      is_active = false;
    } else {
      next = next_ptr;
    }
  }
}

__device__ void warp_replace(bool &is_active, unsigned &myKey,
                             unsigned &myValue, unsigned &src_lane,
                             unsigned &src_key, unsigned &src_bucket,
                             unsigned &read_data) {
  const unsigned laneId = threadIdx.x % warp_size;
  dest_lane = __ffs(
      __ballot_sync(VALID_KEY_MASK, read_data == EMPTY || read_data == myKey));
  if (dest_lane != 0) {
    if (src_lane == landId) {
      long newPair = myKey << 32 | myValue;
      old_pair = atomicCAS(SlabAddress(next, src_bucket, src_lane), EMPTY_PAIR,
                           newPair);
      if (old_pair == EMPTY_PAIR) {
        is_active = false;
      }
    }
  } else {
    next_ptr = __shfl_sync(~0, read_data, ADDRESS_LANE);
    if (next_ptr == 0) {
      new_slab_ptr = warp_allocate();
      if (laneId == ADDRESS_LANE) {
        temp = atomicCAS(SlabAddress(next, src_bucket, ADDRESS_LANE),
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

__device__ long warp_allocate() { printf("Didn't implement\n"); }
__device__ long deallocate(long l) {
  printf("Didn't implement\n");
  return 0;
}

__device__ long *SlabAddress(unsigned next, unsigned src_bucket,
                             unsigned lane) {
  if (lane != 31) {
    return &(slabs[src_bucket][next].keyValue[lane]);
  } else {
    return &(slabs[src_bucket][next].next);
  }
}