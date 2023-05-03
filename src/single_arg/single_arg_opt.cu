#include "../functions.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void divergent_func_opt(int *ret) {
  int tx = threadIdx.x;
  int m = 128 * tx;
  int o = 128 * (tx + 2);

  bool p1 = tx < 16;

  // Predicate the arguments
  int j = o;
  if (p1)
    j = m;

  // Single function call, allowing the call to be done in lock-step
  FUNC(j);
  ret[tx] = m + o;
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
