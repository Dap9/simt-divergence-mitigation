#include "../functions.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

__global__ void divergent_func_opt(int *ret) {
  int tx = threadIdx.x;

  int a = 128 * tx;
  int b = 0;

  int a1 = 128 * (tx + 1);
  int b1 = 1;

  int a2 = 128 * (tx + 2);
  int b2 = 2;

  int a3 = 128 * (tx + 3);
  int b3 = 3;

  int a4 = 128 * (tx + 4);
  int b4 = 4;

  int a5 = 128 * (tx + 5);
  int b5 = 5;

  int a6 = 128 * (tx + 6);
  int b6 = 6;

  int a7 = 128 * (tx + 7);
  int b7 = 7;

  bool p1 = tx < 16;
  bool p2 = tx % 2 == 0;
  bool p3 = tx % 3 == 0;
  int *j = &a7;
  int *t = &b7;

  if (!p1 && !p2 && p3) j = &a6;
  if (!p1 && !p2 && p3) t = &b6;

  if (!p1 && p2 && !p3) j = &a5;
  if (!p1 && p2 && !p3) t = &b5;

  if (!p1 && p2 && p3) j = &a4;
  if (!p1 && p2 && p3) t = &b4;

  if (p1 && !p2 && !p3) j = &a3;
  if (p1 && !p2 && !p3) t = &b3;

  if (p1 && !p2 && p3) j = &a2;
  if (p1 && !p2 && p3) t = &b2;

  if (p1 && p2 && !p3) j = &a1;
  if (p1 && p2 && !p3) t = &b1;

  if (p1 && p2 && p3) j = &a;
  if (p1 && p2 && p3) t = &b;

  FUNC(j, t);
  ret[tx] = b + b1 + b2 + b3 + b4 + b5 + b6 + b7;
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
