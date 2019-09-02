#include "Operations.cu"
#include "gpuErrchk.cu"
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

void printusage(char *exename);
set<unsigned> randomSet(int size);

__global__ void requestHandler(volatile Slab **slabs, unsigned num_of_buckets, volatile bool *is_active, volatile unsigned *myKey, volatile unsigned *myValue, int *request) {
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
  warp_operation(is_active, myKey, myValue, warp_replace, slabs, num_of_buckets);

  if (request[tid] == REQUEST_REMOVE) {
    is_active[tid] = true;
  } else {
    is_active[tid] = false;
  }
  warp_operation(is_active, myKey, myValue, warp_delete, slabs, num_of_buckets);
}

int main(int argc, char **argv) {
  unsigned mapSize = 10000;
  unsigned blocks = 6;
  unsigned threadsPerBlock = 512;
  const int ops = blocks * threadsPerBlock;
  float percentageWrites = 0.0;
  int repeat = 3;
  double loadFactor = 0.7;
  char c;

  while ((c = getopt(argc, argv, "hr:l:w:m:")) != -1) {
    switch (c) {
    case 'm':
      mapSize = atoi(optarg);
      break;
    case 'r':
      repeat = atoi(optarg);
      if (repeat <= 0) {
        exit(1);
      }
      break;
    case 'l':
      loadFactor = atof(optarg);
      break;
    case 'w':
      percentageWrites = atof(optarg) / 100.0;
      break;
    case 'h':
      printusage(argv[0]);
      cerr << "\t-h : help" << endl;
      cerr << "\t-r repeats : changes the number of times the test is repeated" << endl;
      cerr << "\t-l loadFactor : sets the load factor of the test" << endl;
      cerr << "\t-w percentageWrites : sets the percentage of writes" << endl;

      exit(0);
    case '?':
      cerr << "-" << static_cast<char>(optopt) << " is not an argument" << endl;
      cerr << "Try -h for help" << endl;
    default:
      printusage(argv[0]);
      exit(1);
    }
  }

  unsigned numberOfSlabsPerBucket = 10;

  setUp(mapSize, numberOfSlabsPerBucket);

  volatile bool *is_active;
  volatile unsigned *myKey;
  volatile unsigned *myValue;
  int *request;

  unsigned allocationSize = ops > (unsigned)(ceil(mapSize / blocks / threadsPerBlock) * mapSize) ? ops : (unsigned)(ceil(mapSize / blocks / threadsPerBlock) * mapSize);

  gpuErrchk(cudaMallocManaged(&is_active, sizeof(bool) * allocationSize));
  gpuErrchk(cudaMallocManaged(&myKey, sizeof(unsigned) * allocationSize));
  gpuErrchk(cudaMallocManaged(&myValue, sizeof(unsigned) * allocationSize));
  gpuErrchk(cudaMallocManaged(&request, sizeof(int) * allocationSize));

  auto bigS = randomSet(mapSize + ops);
  set<unsigned> s;
  set<unsigned> canInsert;

  set<unsigned, greater<unsigned>>::iterator itr = bigS.begin();
  for (int i = 0; i < (int)(mapSize * loadFactor); i++) {
    s.emplace(*itr);
    itr++;
  }
  for (int i = 0; i < ops + (int)(mapSize * 1 - loadFactor); i++) {
    canInsert.emplace(*itr);
    itr++;
  }

  itr = s.begin();

  for (int i = 0; i < (int)(mapSize * loadFactor); i++) {
    request[i] = REQUEST_INSERT;
    myKey[i] = *itr;
    if (*itr == 0) {
      printf("Error iterator cannot be 0\n");
      exit(1);
    }
    myValue[i] = i + 1;
    itr++;
  }
  for (int i = (int)(mapSize * loadFactor); i < allocationSize; i++) {
    request[i] = EMPTY;
  }

  unsigned step = blocks * threadsPerBlock;
  for (int i = 0; i < allocationSize / mapSize; i++) {
    requestHandler<<<blocks, threadsPerBlock>>>(slabs, num_of_buckets, is_active + step * i, myKey + step * i, myValue + step * i, request + step * i);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
  }

  for (int i = 0; i < repeat; i++) {

    bool remove = true;
    for (int k = 0; k < ops; k++) {
      if (i < percentageWrites * ops) {
        if (remove) {
          int toInsert;
          if (s.size() != 0) {
            toInsert = (int)(rand() / (double)RAND_MAX * s.size());

            itr = s.begin();
            for (int w = 0; w < toInsert; w++) {
              itr++;
            }
            toInsert = *itr;
            canInsert.insert(*itr);
            s.erase(itr);
          } else {
            toInsert = 1;
          }

          int requestSwap = REQUEST_REMOVE;
          int j = 0;
          for (; j < i && toInsert % mapSize > myKey[j] % mapSize; j++)
            ;

          for (; j <= i; j++) {
            int temp = myKey[j];
            int tempRequest = request[j];
            myKey[j] = toInsert;
            request[j] = requestSwap;
            toInsert = temp;
            requestSwap = tempRequest;
          }
        } else {
          int toInsert;
          if (canInsert.size() != 0) {
            toInsert = (int)(rand() / (double)RAND_MAX * canInsert.size());

            itr = canInsert.begin();
            for (int w = 0; w < toInsert; w++) {
              itr++;
            }

            toInsert = *itr;

            s.insert(*itr);
            canInsert.erase(itr);
          } else {
            toInsert = 1;
          }

          int requestSwap = REQUEST_INSERT;
          int j = 0;
          for (; j < i && toInsert % mapSize > myKey[j] % mapSize; j++)
            ;

          for (; j <= i; j++) {
            int temp = myKey[j];
            int tempRequest = request[j];
            myKey[j] = toInsert;
            request[j] = requestSwap;
            toInsert = temp;
            requestSwap = tempRequest;
          }
        }
        remove = !remove;

      } else {

        int toInsert = (int)(rand() / (double)RAND_MAX * s.size());

        itr = s.begin();
        for (int w = 0; w < toInsert; w++) {
          itr++;
        }

        toInsert = *itr;

        int requestSwap = REQUEST_GET;
        int j = 0;
        for (; j < i && toInsert % mapSize > myKey[j] % mapSize; j++)
          ;

        for (; j <= i; j++) {
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

    cout << "Starting test" << endl;
    auto start = high_resolution_clock::now();
    for (int i = 0; i < ops / step; i++) {
      requestHandler<<<blocks, threadsPerBlock>>>(slabs, num_of_buckets, is_active + step * i, myKey + step * i, myValue + step * i, request + step * i);
      gpuErrchk(cudaPeekAtLastError());
      gpuErrchk(cudaDeviceSynchronize());
    }
    auto end = high_resolution_clock::now();
    duration<double> time_span = end - start;
    cout << "Throughput " << ops / time_span.count() << " ops/s" << endl;
  }
}

void printusage(char *exename) { cerr << "Usage: " << exename << " [-v] [-h] [-d | -u] [-r repeats] [-l loadFactor] [-w percentageWrites]" << endl; }

set<unsigned> randomSet(int size) {
  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution(1.0, 2.0 * size + 1.0);

  set<unsigned> s;
  while (s.size() < size) {
    s.insert((unsigned)(distribution(generator)));
  }
  return s;
}