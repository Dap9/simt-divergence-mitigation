#include "../functions.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

__global__ void divergent_func_ptr(int *ret) {
  int tx = threadIdx.x;
  int r = 3;
  int z = 4;
  int m = 128 * tx;
  int n = 128 * (tx + 1);
  int o = 128 * (tx + 2);
  int q = 128 * (tx + 3);
  bool p1 = tx < 16;
  if (p1) {
    FUNC(m, n, &r);

  } else {
    FUNC(o, q, &z);
  }
  // so DCE doesn't eliminate stuff
  ret[tx] = z + r;
  return;
}

__global__ void divergent_func_ptr_opt(int *ret) {
  int tx = threadIdx.x;
  int r = 3;
  int z = 4;
  bool p1 = tx < 16;
  int j = 128 * tx;
  if (!p1)
    j = 128 * (tx + 2);
  int k = 128 * (tx + 1);
  if (!p1)
    k = 128 * (tx + 3);
  int *t = &r;
  if (!p1)
    t = &z;
  FUNC(j, k, t);
  ret[tx] = r + z;
  return;
}

__global__ void divergent_func_nested_cond_exec(int *ret) {
  int tx = threadIdx.x;
  int r = 3;
  int z = 4;
  int m = 128 * tx;
  int n = 128 * (tx + 1);
  int o = 128 * (tx + 2);
  int q = 128 * (tx + 3);
  bool p1 = tx < 16;
  bool p2 = tx % 2 == 1;
  if (p1) {
    FUNC(m, n, &r);
  } else {
    if (p2) {
      FUNC(o, q, &z);
    } else {
      z = 420;
    }
  }
  // so DCE doesn't eliminate stuff
  ret[tx] = r + z;
  return;
}

__global__ void divergent_func_nested_cond_exec_opt(int *ret) {
  int tx = threadIdx.x;
  int r = 3;
  int z = 4;
  bool p1 = tx < 16;
  bool p2;
  if (!p1)
    p2 = tx % 2 == 1;
  int j = 128 * tx;
  if (!p1 && p2)
    j = 128 * (tx + 2);
  int k = 128 * (tx + 1);
  if (!p1 && p2)
    k = 128 * (tx + 3);
  int *t = &r;
  if (!p1 && p2)
    t = &z;
  if (p1 || p2)
    FUNC(j, k, t);
  if (!p1 && !p2)
    z = 420;
  ret[tx] = r + z;
  return;
}

void f(void) { std::cout << " hello" << std::endl; }

int main() {
  int *out, *d_out;
  // Allocate host memory
  out = (int *)malloc(sizeof(int) * 32);
  // Allocate device memory
  cudaMalloc((void **)&d_out, sizeof(int) * 32);
  // Executing kernel
  divergent_func_ptr<<<1, 32>>>(d_out);

  cudaMemcpy(out, d_out, sizeof(int) * 32, cudaMemcpyDeviceToHost);

  for (int i = 0; i < 32; i++)
    printf("flat: out[%d] = %d\n", i, out[i]);
  printf("\n");

  divergent_func_ptr_opt<<<1, 32>>>(d_out);
  cudaMemcpy(out, d_out, sizeof(int) * 32, cudaMemcpyDeviceToHost);
  for (int i = 0; i < 32; i++)
    printf("flat_opt: out[%d] = %d\n", i, out[i]);
  printf("\n");

  divergent_func_nested_cond_exec<<<1, 32>>>(d_out);
  cudaMemcpy(out, d_out, sizeof(int) * 32, cudaMemcpyDeviceToHost);
  for (int i = 0; i < 32; i++)
    printf("nested_cond_exec: out[%d] = %d\n", i, out[i]);
  printf("\n");

  divergent_func_nested_cond_exec_opt<<<1, 32>>>(d_out);
  cudaMemcpy(out, d_out, sizeof(int) * 32, cudaMemcpyDeviceToHost);
  for (int i = 0; i < 32; i++)
    printf("nested_cond_exec_opt: out[%d] = %d\n", i, out[i]);
  printf("\n");

  // Deallocate device memory
  cudaFree(d_out);
  // Deallocate hostmemory
  free(out);
}
