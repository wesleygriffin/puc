#include <cstdlib>

__global__ void sum(int n, float* x, float* y) {
  std::size_t const index = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t const stride = blockDim.x * gridDim.x;
  for (std::size_t i = index; i < n; i += stride) {
    y[i] = x[i] + y[i];
  }
}

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>

#include <mpi.h>
#include <unistd.h>

int main(int argc, char* argv[]) {
  int N = 0;

  char c;
  while ((c = getopt(argc, argv, "N:")) != -1) {
    switch(c) {
    case 'N':
      try {
        N = std::stoi(optarg);
      } catch (std::exception const& e) {
        std::fprintf(stderr, "invalid argument '%s': %s\n", optarg, e.what());
        std::exit(EXIT_FAILURE);
      }
      break;
    case '?':
    default:
      std::fprintf(stderr, "usage: %s -N <N>\n", argv[0]);
      std::exit(EXIT_FAILURE);
    }
  }

  if (N <= 0) {
    std::fprintf(stderr, "invalid N argument '%d': must be positive\n", N);
    std::exit(EXIT_FAILURE);
  }

  MPI_Init(nullptr, nullptr);

  int worldRank, worldSize;
  MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

  float* x = nullptr;
  float* y = nullptr;
  if (worldRank == 0) {
    try {
      x = new float[N * worldSize];
      y = new float[N * worldSize];
    } catch (std::exception const& e) {
      std::fprintf(stderr, "Unable to allocate 2 %zu float arrays\n", N *
                   worldSize);
      MPI_Finalize();
      std::exit(EXIT_FAILURE);
    }

    for (int i = 0; i < N * worldSize; ++i) x[i] = 1.f;
    for (int i = 0; i < N * worldSize; ++i) y[i] = 2.f;
  }

  cudaError cudaResult;
  int mpiResult;

  float* sub_x = nullptr;
  cudaResult = cudaMallocManaged(&sub_x, N * sizeof(float));
  if (cudaResult != cudaSuccess) {
    std::fprintf(stderr, "Unable to allocate CUDA managed memory: %s\n",
                 cudaGetErrorString(cudaResult));
    MPI_Finalize();
    std::exit(EXIT_FAILURE);
  }

  mpiResult = MPI_Scatter(x,             // sendbuf
                          N,             // sendcount
                          MPI_FLOAT,     // sendtype
                          sub_x,         // recvbuf
                          N,             // recvcount (to _each_ process)
                          MPI_FLOAT,     // recvtype
                          0,             // root
                          MPI_COMM_WORLD // comm
                         );
  if (mpiResult != MPI_SUCCESS) {
    int len = 2048;
    char str[len];
    MPI_Error_string(mpiResult, str, &len);
    std::fprintf(stderr, "Unable to scatter memory: %s\n", str);
    MPI_Finalize();
    std::exit(EXIT_FAILURE);
  }

  float* sub_y = nullptr;
  cudaResult = cudaMallocManaged(&sub_y, N * sizeof(float));
  if (cudaResult != cudaSuccess) {
    std::fprintf(stderr, "Unable to allocate CUDA managed memory: %s\n",
                 cudaGetErrorString(cudaResult));
    MPI_Finalize();
    std::exit(EXIT_FAILURE);
  }

  mpiResult = MPI_Scatter(y,             // sendbuf
                          N,             // sendcount
                          MPI_FLOAT,     // sendtype
                          sub_y,         // recvbuf
                          N,             // recvcount (to _each_ process)
                          MPI_FLOAT,     // recvtype
                          0,             // root
                          MPI_COMM_WORLD // comm
                         );
  if (mpiResult != MPI_SUCCESS) {
    int len = 2048;
    char str[len];
    MPI_Error_string(mpiResult, str, &len);
    std::fprintf(stderr, "Unable to scatter memory: %s\n", str);
    MPI_Finalize();
    std::exit(EXIT_FAILURE);
  }

  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;
  sum<<<numBlocks, blockSize>>>(N, sub_x, sub_y);

  cudaResult = cudaDeviceSynchronize();
  if (cudaResult != cudaSuccess) {
    std::fprintf(stderr, "Asynchronous CUDA error: %s\n",
                 cudaGetErrorString(cudaResult));
    MPI_Finalize();
    std::exit(EXIT_FAILURE);
  }

  float error = 0.f;
  for (int i = 0; i < N; ++i) {
    error = std::max(error, std::abs(sub_y[i] - 3.f));
  }
  std::printf("rank %d error: %g\n", worldRank, error);

  float* errors = nullptr;
  if (worldRank == 0) {
    errors = new float[worldSize];
  }

  mpiResult = MPI_Gather(&error,        // sendbuf
                         1,             // sendcount
                         MPI_FLOAT,     // sendtype
                         errors,        // recvbuf
                         1,             // recvcount (from _each_ process)
                         MPI_FLOAT,     // recvtype
                         0,             // root
                         MPI_COMM_WORLD // comm
                        );
  if (mpiResult != MPI_SUCCESS) {
    int len = 2048;
    char str[len];
    MPI_Error_string(mpiResult, str, &len);
    std::fprintf(stderr, "Unable to gather memory: %s\n", str);
    MPI_Finalize();
    std::exit(EXIT_FAILURE);
  }

  if (worldRank == 0) {
    float globalError = 0.f;
    for (int i = 0; i < worldSize; ++i) {
      globalError = std::max(globalError, errors[i]);
    }
    std::printf("globalError: %g\n", globalError);
  }

  cudaFree(sub_y);
  cudaFree(sub_x);

  if (worldRank == 0) {
    delete[] errors;
    delete[] y;
    delete[] x;
  }

  MPI_Finalize();
  return EXIT_SUCCESS;
}

