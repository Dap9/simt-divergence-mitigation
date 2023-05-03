#include "../functions.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

__global__ void divergent_func_opt(int *ret) {
  int tx = threadIdx.x;
  int r = 3;
  int z = 4;
  int m = 127 * tx;
  int o = 127 * (tx + 2);

  // Max 27, else errors out
  m %= MOD;
  o %= MOD;

  bool p1 = tx < 16;

  int j = o;
  if (p1)
    j = m;

  int *t = &z;
  if (p1)
    t = &r;

  recursive_fibonacci(j, t);
  ret[tx] = r + z;
  return;
}

int main() {
  printf("Opt\n");
  int *out, *d_out;
  // Allocate host memory
  out = (int *)malloc(sizeof(int) * NUM_THREADS_PER_WARP);
  // Allocate device memory
  cudaMalloc((void **)&d_out, sizeof(int) * NUM_THREADS_PER_WARP);
  // Executing kernel
  divergent_func_opt<<<NUM_WARPS, NUM_THREADS_PER_WARP>>>(d_out);
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
