#include "../Operations.cu"
#include <iostream>
#include <stdexcept>

using namespace std;

template <typename T> //
void checkNotNull(T *ptr) {
  if (ptr == (T *)nullptr) {
    throw std::runtime_error("Setup does not work");
  }
}

__global__ void functionTester(volatile Slab **slabs, unsigned num_of_buckets, volatile bool *is_active, volatile unsigned *myKey, volatile unsigned *myValue, bool *results) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  is_active[tid] = false;
  myKey[tid] = 1;
  myValue[tid] = 2;
  if (threadIdx.x == 0) {
    is_active[tid] = true;
  } else {
    is_active[tid] = false;
  }

  warp_operation(is_active, myKey, myValue, warp_replace, slabs, num_of_buckets);
  if (threadIdx.x == 0) {
    printf("Insert: %d %d %d\n", is_active[tid], myKey[tid], myValue[tid]);
  }

  if (myKey[tid] == 1 && myValue[tid] == 2) {
    results[0] = true;
  }

  is_active[tid] = false;
  myKey[tid] = 1;
  myValue[tid] = 0;
  if (threadIdx.x == 0) {
    is_active[tid] = true;
  }

  warp_operation(is_active, myKey, myValue, warp_search, slabs, num_of_buckets);
  if (threadIdx.x == 0) {
    printf("Read : %d %d %d\n", is_active[tid], myKey[tid], myValue[tid]);
  }

  if (myKey[tid] == 1 && myValue[tid] == 2) {
    results[1] = true;
  }

  is_active[tid] = false;
  myKey[tid] = 1;
  myValue[tid] = 2;
  if (threadIdx.x == 0) {
    is_active[tid] = true;
  }

  warp_operation(is_active, myKey, myValue, warp_delete, slabs, num_of_buckets);
  if (threadIdx.x == 0) {
    printf("Removed : %d %d %d\n", is_active[tid], myKey[tid], myValue[tid]);
  }

  if (myKey[tid] == 1 && myValue[tid] == 2) {
    results[2] = true;
  }

  is_active[tid] = false;
  myKey[tid] = 1;
  myValue[tid] = 0;
  if (threadIdx.x == 0) {
    is_active[tid] = true;
  }

  warp_operation(is_active, myKey, myValue, warp_search, slabs, num_of_buckets);
  if (threadIdx.x == 0) {
    printf("Read : %d %d %d\n", is_active[tid], myKey[tid], myValue[tid]);
  }

  if (myKey[tid] == 1 && myValue[tid] == 0) {
    results[3] = true;
  }
}

int main() {

  unsigned size = 1;
  unsigned numberOfSlabsPerBucket = 1;

  if (slabs != nullptr) {
    throw std::runtime_error("Test for setup does not work");
  }

  setUp(size, numberOfSlabsPerBucket);

  if (num_of_buckets != size) {
    throw std::runtime_error("Setup does not work");
  }

  if (slabs[0][0].keyValue[0] != 0) {
    throw std::runtime_error("Setup does not work");
  }

  checkNotNull(slabs);
  for (int i = 0; i < size; i++) {
    checkNotNull(slabs[i]);
  }

  volatile bool *is_active;
  volatile unsigned *myKey;
  volatile unsigned *myValue;
  bool *results;

  gpuErrchk(cudaMallocManaged(&is_active, sizeof(bool) * 32));
  gpuErrchk(cudaMallocManaged(&myKey, sizeof(unsigned) * 32));
  gpuErrchk(cudaMallocManaged(&myValue, sizeof(unsigned) * 32));
  gpuErrchk(cudaMallocManaged(&results, sizeof(bool) * 4));

  for (int i = 0; i < 4; i++) {
    results[i] = false;
  }

  functionTester<<<1, 32>>>(slabs, num_of_buckets, is_active, myKey, myValue, results);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  for (int i = 0; i < 4; i++) {
    if (!results[i]) {
      throw std::runtime_error("Operations did not work\n");
    }
  }

  return 0;
}