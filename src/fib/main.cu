#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <memory>

#ifndef MOD
#define MOD 27
#endif

// Recursive Fibonacci to run on the GPU Thread
__device__ __noinline__ void recursive_fibonacci(int fib, int *result){
    if (fib <= 1){
        *result += fib;
        return;
    }
    recursive_fibonacci(fib - 1, result);
    recursive_fibonacci(fib - 2, result);
}

__global__ void divergent_func_ptr(int *ret) {
  int tx = threadIdx.x;
  int r = 3;
  int z = 4;
  int m = 127 * tx;
  int o = 127 * (tx + 2);
  m %= MOD;
  o %= MOD;
  bool p1 = tx < 16;
  if (p1) {
    recursive_fibonacci(m, &r);
  } else {
    recursive_fibonacci(o, &z);
  }
  // so DCE doesn't eliminate stuff
  ret[tx] = r + z;
  return;
}

__global__ void divergent_func_ptr_opt(int *ret) {
  int tx = threadIdx.x;
  int r = 3;
  int z = 4;
  bool p1 = tx < 16;
  int j = 127 * tx;
  if (!p1)
    j = 127 * (tx + 2);
  int *t = &r;
  if (!p1)
    t = &z;
  j %= MOD;
  recursive_fibonacci(j, t);
  ret[tx] = r + z;
  return;
}

__global__ void divergent_func_nested(int *ret) {
  int tx = threadIdx.x;
  int r = 3;
  int z = 4;
  int m = (127 * tx) % MOD;
  int o = (127 * (tx + 2)) % MOD;
  bool p1 = tx < 16;
  bool p2 = tx % 2 == 1;
  if (p1) {
    recursive_fibonacci(m, &r);
  } else {
    if (p2) {
      recursive_fibonacci(o, &z);
    } else {
      z = 420;
    }
  }
  // so DCE doesn't eliminate stuff
  ret[tx] = r + z;
  return;
}

__global__ void divergent_func_nested_opt(int *ret) {
  int tx = threadIdx.x;
  int r = 3;
  int z = 4;
  bool p1 = tx < 16;
  bool p2;
  if (!p1)
    p2 = tx % 2 == 1;
  int j = 127 * tx;
  if (!p1 && p2)
    j = 127 * (tx + 2);
  j %= MOD;
  int *t = &r;
  if (!p1 && p2)
    t = &z;
  if (p1 || p2)
    recursive_fibonacci(j, t);
  if (!p1 && !p2)
    z = 420;
  ret[tx] = r + z;
  return;
}

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

  divergent_func_nested<<<1, 32>>>(d_out);
  cudaMemcpy(out, d_out, sizeof(int) * 32, cudaMemcpyDeviceToHost);
  for (int i = 0; i < 32; i++)
    printf("nested_cond_exec: out[%d] = %d\n", i, out[i]);
  printf("\n");

  divergent_func_nested_opt<<<1, 32>>>(d_out);
  cudaMemcpy(out, d_out, sizeof(int) * 32, cudaMemcpyDeviceToHost);
  for (int i = 0; i < 32; i++)
    printf("nested_cond_exec_opt: out[%d] = %d\n", i, out[i]);
  printf("\n");

  // Deallocate device memory
  cudaFree(d_out);
  // Deallocate hostmemory
  free(out);
}
