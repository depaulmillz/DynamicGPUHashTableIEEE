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
  unsigned mapSize = 10000;
  unsigned blocks = 68;
  unsigned threadsPerBlock = 512;
  const int ops = blocks * threadsPerBlock;
  float percentageWrites = 0.0;
  int repeat = 3;
  double loadFactor = 0.5;
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
      cerr << "\t-r repeats : changes the number of times the test is repeated"
           << endl;
      cerr << "\t-l loadFactor : sets the load factor of the test" << endl;
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

  unsigned numberOfSlabsPerBucket = 10;

  setUp(mapSize, numberOfSlabsPerBucket);

  bool *is_active_k, *is_active_h;
  unsigned *myKey_k, *myKey_h;
  unsigned *myValue_k, *myValue_h;
  int *request_k, *request_h;

  unsigned allocationSize =
      ops > (unsigned)(ceil(mapSize / blocks / threadsPerBlock) * mapSize)
          ? ops
          : (unsigned)(ceil(mapSize / blocks / threadsPerBlock) * mapSize);

  gpuErrchk(cudaMalloc(&is_active_k, sizeof(bool) * allocationSize));
  is_active_h = new bool[allocationSize];
  gpuErrchk(cudaMalloc(&myKey_k, sizeof(unsigned) * allocationSize));
  myKey_h = new unsigned[allocationSize];
  gpuErrchk(cudaMalloc(&myValue_k, sizeof(unsigned) * allocationSize));
  myValue_h = new unsigned[allocationSize];
  gpuErrchk(cudaMalloc(&request_k, sizeof(int) * allocationSize));
  request_h = new int[allocationSize];

  int sizeNeededForMap = max(mapSize, (int)(mapSize * loadFactor));

  auto bigS = randomSet(sizeNeededForMap + ops);
  set<unsigned> s;
  set<unsigned> canInsert;

  set<unsigned, greater<unsigned>>::iterator itr = bigS.begin();
  for (int i = 0; i < (int)(mapSize * loadFactor); i++) {
    s.emplace(*itr);
    itr++;
  }
  for (int i = 0; i < sizeNeededForMap + ops - (int)(mapSize * loadFactor);
       i++) {
    canInsert.emplace(*itr);
    itr++;
  }

  itr = s.begin();

  for (int i = 0; i < (int)(mapSize * loadFactor); i++) {
    request_h[i] = REQUEST_INSERT;
    myKey_h[i] = *itr;
    if (*itr == 0) {
      printf("Error iterator cannot be 0\n");
      exit(1);
    }
    myValue_h[i] = i + 1;
    itr++;
  }
  for (int i = (int)(mapSize * loadFactor); i < allocationSize; i++) {
    request_h[i] = EMPTY;
  }

  gpuErrchk(cudaMemcpy(is_active_k, is_active_h, sizeof(bool) * allocationSize,
                       cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(myKey_k, myKey_h, sizeof(unsigned) * allocationSize,
                       cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(myValue_k, myValue_h, sizeof(unsigned) * allocationSize,
                       cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(request_k, request_h, sizeof(int) * allocationSize,
                       cudaMemcpyHostToDevice));

  unsigned step = blocks * threadsPerBlock;
  for (int i = 0; i < allocationSize / mapSize; i++) {
    requestHandler<<<blocks, threadsPerBlock>>>(
        slabs, num_of_buckets, is_active_k + step * i, myKey_k + step * i,
        myValue_k + step * i, request_k + step * i);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
  }

  gpuErrchk(cudaMemcpy(is_active_h, is_active_k, sizeof(bool) * allocationSize,
                       cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(myKey_h, myKey_k, sizeof(unsigned) * allocationSize,
                       cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(myValue_h, myValue_k, sizeof(unsigned) * allocationSize,
                       cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(request_h, request_k, sizeof(int) * allocationSize,
                       cudaMemcpyDeviceToHost));

  cout << "SLAB HASH TABLE\nLOAD FACTOR: " << loadFactor
       << "\nMAP SIZE: " << mapSize << endl
       << endl;

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
          for (; j < i && toInsert % mapSize > myKey_h[j] % mapSize; j++)
            ;

          for (; j <= i; j++) {
            int temp = myKey_h[j];
            int tempRequest = request_h[j];
            myKey_h[j] = toInsert;
            request_h[j] = requestSwap;
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
          for (; j < i && toInsert % mapSize > myKey_h[j] % mapSize; j++)
            ;

          for (; j <= i; j++) {
            int temp = myKey_h[j];
            int tempRequest = request_h[j];
            myKey_h[j] = toInsert;
            request_h[j] = requestSwap;
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
        for (; j < i && toInsert % mapSize > myKey_h[j] % mapSize; j++)
          ;

        for (; j <= i; j++) {
          int temp = myKey_h[j];
          int tempRequest = request_h[j];
          myKey_h[j] = toInsert;
          request_h[j] = requestSwap;
          toInsert = temp;
          requestSwap = tempRequest;
        }
      }
    }
    for (int k = ops; k < allocationSize; k++) {
      request_h[k] = EMPTY;
    }

    gpuErrchk(cudaMemcpy(is_active_k, is_active_h,
                         sizeof(bool) * allocationSize,
                         cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(myKey_k, myKey_h, sizeof(unsigned) * allocationSize,
                         cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(myValue_k, myValue_h,
                         sizeof(unsigned) * allocationSize,
                         cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(request_k, request_h, sizeof(int) * allocationSize,
                         cudaMemcpyHostToDevice));

    // cout << "Starting test" << endl;
    auto start = high_resolution_clock::now();
    for (int i = 0; i < ops / step; i++) {
      requestHandler<<<blocks, threadsPerBlock>>>(
          slabs, num_of_buckets, is_active_k + step * i, myKey_k + step * i,
          myValue_k + step * i, request_k + step * i);
      gpuErrchk(cudaDeviceSynchronize());
    }
    auto end = high_resolution_clock::now();
    duration<double> time_span = end - start;
    cout << "Throughput " << ops / time_span.count() / 1e6 << " Mops/s" << endl;
  }
}

void printusage(char *exename) {
  cerr << "Usage: " << exename
       << " [-v] [-h] [-d | -u] [-r repeats] [-l loadFactor] [-w "
          "percentageWrites] [-m mapsize]"
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