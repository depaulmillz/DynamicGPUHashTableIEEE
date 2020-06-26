#include "Operations.cu"
#include "gpuErrchk.cu"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <unistd.h>

#define REQUEST_INSERT 1
#define REQUEST_GET 2
#define REQUEST_REMOVE 3
#define EMPTY 0

using namespace std;
using namespace chrono;

extern char *optarg;
extern int optopt;

double alpha = 100.0;
int zipf(double alpha, int n);
void printusage(char *exename);
set<unsigned> randomSet(int size);

__global__ void requestHandler(volatile Slab **slabs, unsigned num_of_buckets,
                               bool *is_active, unsigned *myKey,
                               unsigned *myValue, int *request) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (request[tid] == REQUEST_GET) {
    is_active[tid] = true;
  } else {
    is_active[tid] = false;
  }

  warp_operation(is_active, myKey, myValue, warp_search, slabs, num_of_buckets);

  if (request[tid] == REQUEST_INSERT) {
    is_active[tid] = true;
  } else {
    is_active[tid] = false;
  }
  warp_operation(is_active, myKey, myValue, warp_replace, slabs,
                 num_of_buckets);

  if (request[tid] == REQUEST_REMOVE) {
    is_active[tid] = true;
  } else {
    is_active[tid] = false;
  }
  warp_operation(is_active, myKey, myValue, warp_delete, slabs, num_of_buckets);
}

int main(int argc, char **argv) {
  unsigned mapSize = 100000;
  unsigned blocks = 6;
  unsigned threadsPerBlock = 512;
  const int ops = 100000;
  float percentageWrites = 0.0;
  char c;

  while ((c = getopt(argc, argv, "hw:m:")) != -1) {
    switch (c) {
    case 'm':
      mapSize = atoi(optarg);
      break;
    case 'w':
      percentageWrites = atof(optarg) / 100.0;
      break;
    case 'h':
      printusage(argv[0]);
      cerr << "\t-h : help" << endl;
      cerr << "\t-w percentageWrites : sets the percentage of writes" << endl;
      cerr << "\t-m mapsize : sets the map size" << endl;

      exit(0);
    case '?':
      cerr << "-" << static_cast<char>(optopt) << " is not an argument" << endl;
      cerr << "Try -h for help" << endl;
    default:
      printusage(argv[0]);
      exit(1);
    }
  }

  printf("Populating data structure\n");

  unsigned numberOfSlabsPerBucket = 10;

  setUp(mapSize, numberOfSlabsPerBucket);

  bool *is_active;
  unsigned *myKey;
  unsigned *myValue;
  int *request;

  unsigned allocationSize =
      ops > (unsigned)(ceil(mapSize / blocks / threadsPerBlock) * mapSize)
          ? ops
          : (unsigned)(ceil(mapSize / blocks / threadsPerBlock) * mapSize);

  gpuErrchk(cudaMallocManaged(&is_active, sizeof(bool) * allocationSize));
  gpuErrchk(cudaMallocManaged(&myKey, sizeof(unsigned) * allocationSize));
  gpuErrchk(cudaMallocManaged(&myValue, sizeof(unsigned) * allocationSize));
  gpuErrchk(cudaMallocManaged(&request, sizeof(int) * allocationSize));

  auto s = randomSet(mapSize);

  vector<unsigned> v(s.begin(), s.end());

  auto itr = s.begin();

  for (int i = 0; i < mapSize; i++) {
    request[i] = REQUEST_INSERT;
    myKey[i] = *itr;
    if (*itr == 0) {
      printf("Error iterator cannot be 0\n");
      exit(1);
    }
    myValue[i] = i + 1;
    itr++;
  }
  for (int i = mapSize; i < allocationSize; i++) {
    request[i] = EMPTY;
  }

  unsigned step = blocks * threadsPerBlock;
  for (int i = 0; i < allocationSize / mapSize; i++) {
    requestHandler<<<blocks, threadsPerBlock>>>(
        slabs, num_of_buckets, is_active + step * i, myKey + step * i,
        myValue + step * i, request + step * i);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
  }

  for (int k = 0; k < ops; k++) {
    if (k < percentageWrites * ops) {

      int toInsert = zipf(alpha, s.size()) - 1;
      toInsert = v[toInsert];

      int requestSwap = REQUEST_INSERT;
      int j = 0;
      for (; j < k && toInsert % mapSize > myKey[j] % mapSize; j++)
        ;

      for (; j <= k; j++) {
        int temp = myKey[j];
        int tempRequest = request[j];
        myKey[j] = toInsert;
        request[j] = requestSwap;
        toInsert = temp;
        requestSwap = tempRequest;
      }

    } else {

      int toInsert = zipf(alpha, s.size()) - 1;

      toInsert = v[toInsert];

      int requestSwap = REQUEST_GET;
      int j = 0;
      for (; j < k && toInsert % mapSize > myKey[j] % mapSize; j++)
        ;

      for (; j <= k; j++) {
        int temp = myKey[j];
        int tempRequest = request[j];
        myKey[j] = toInsert;
        request[j] = requestSwap;
        toInsert = temp;
        requestSwap = tempRequest;
      }
    }
  }
  for (int k = ops; k < allocationSize; k++) {
    request[k] = EMPTY;
  }

  cout << "Starting test with " << ops << " operations and a " << mapSize
       << " element map size" << endl;
  auto start = high_resolution_clock::now();
  for (int i = 0; i < ops / step; i++) {
    requestHandler<<<blocks, threadsPerBlock>>>(
        slabs, num_of_buckets, is_active + step * i, myKey + step * i,
        myValue + step * i, request + step * i);
    gpuErrchk(cudaDeviceSynchronize());
  }
  auto end = high_resolution_clock::now();
  duration<double> time_span = end - start;
  cout << "Throughput " << ops / time_span.count() << " ops/s" << endl;
}

void printusage(char *exename) {
  cerr << "Usage: " << exename
       << " [-v] [-h] [-d | -u] [-r repeats] [-l loadFactor] [-w "
          "percentageWrites]"
       << endl;
}

set<unsigned> randomSet(int size) {
  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution(1.0, 2.0 * size + 1.0);

  set<unsigned> s;
  while (s.size() < size) {
    s.insert((unsigned)(distribution(generator)));
  }
  return s;
}

/// Fast zipf found on stack overflow and modified
int zipf(double alpha, int n) {
  static bool first = true; // Static first time flag
  static double c = 0;      // Normalization constant
  double z;                 // Uniform random number (0 < z < 1)
  double sum_prob;          // Sum of probabilities
  double zipf_value;        // Computed exponential value to be returned
  int i;                    // Loop counter

  // Compute normalization constant on first call only
  if (first == true) {
    for (i = 1; i <= n; i++)
      c = c + (1.0 / pow((double)i, alpha));
    c = 1.0 / c;
    first = false;
  }

  // Pull a uniform random number (0 < z < 1)
  do {
    z = rand() / (double)RAND_MAX;
  } while ((z == 0) || (z == 1));

  // Map z to the value
  sum_prob = 0;
  for (i = 1; i <= n; i++) {
    sum_prob = sum_prob + c / pow((double)i, alpha);
    if (sum_prob >= z) {
      zipf_value = i;
      break;
    }
  }

  // Assert that zipf_value is between 1 and N
  assert((zipf_value >= 1) && (zipf_value <= n));

  return (zipf_value);
}
