#include "../functions.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void divergent_func(int *ret) {
  int tx = threadIdx.x;
  int r = 3;
  int m = 128 * tx;
  int n = 128 * (tx + 1);

  int z = 4;
  int o = 128 * (tx + 2);
  int p = 128 * (tx + 3);

  bool p1 = tx < 16;

  if (p1) {
    FUNC(&m, &n, &r);
  } else {
    FUNC(&o, &p, &z);
  }
  // so DCE doesn't eliminate stuff
  ret[tx] = r + z;
  return;
}

int main() {
  printf("Base\n");
  int *out, *d_out;
  // Allocate host memory
  out = (int *)malloc(sizeof(int) * NUM_THREADS_PER_WARP);
  // Allocate device memory
  cudaMalloc((void **)&d_out, sizeof(int) * NUM_THREADS_PER_WARP);
  // Executing kernel
  divergent_func<<<NUM_WARPS, NUM_THREADS_PER_WARP>>>(d_out);
  cudaMemcpy(out, d_out, sizeof(int) * NUM_THREADS_PER_WARP,
             cudaMemcpyDeviceToHost);
  for (int i = 0; i < NUM_THREADS_PER_WARP; i++)
    printf("flat: out[%d] = %d\n", i, out[i]);
  printf("\n");

  printf("\n Error msg: %s\n", cudaGetErrorString(cudaGetLastError()));

  // Deallocate device memory
  cudaFree(d_out);
  // Deallocate hostmemory
  free(out);
}
