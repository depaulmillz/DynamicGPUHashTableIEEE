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

__global__ void functionTester(Slab **slabs, unsigned num_of_buckets, volatile bool *is_active, volatile unsigned *myKey, volatile unsigned *myValue) {
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
    printf("%d %d %d\n", is_active, myKey, myValue);
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

  gpuErrchk(cudaMallocManaged(&is_active, sizeof(bool) * 32));
  gpuErrchk(cudaMallocManaged(&myKey, sizeof(unsigned) * 32));
  gpuErrchk(cudaMallocManaged(&myValue, sizeof(unsigned) * 32));

  functionTester<<<1, 32>>>(slabs, num_of_buckets, is_active, myKey, myValue);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  return 0;
}