#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#ifdef CHEAP
#define FUNC f_cheap
#else
#define FUNC f
#endif

#ifndef MAX_ITER
#define MAX_ITER 1e6
#endif

// These functions are only to show example calls. They do not have any
// practicality to them as-is but the argument structure is useful for the sake
// of showing our optimization
__device__ __noinline__ void f(int *a) {
  for (int i = 0; i < MAX_ITER; i++) {
    // Do busy work, ensuring this statement has to be re-exucted to better
    // simulate real-world workloads
    if (*a == __funnelshift_r(*a, *a, *a)) {
      *a = 420;
    }
  }
}

// Example of a function that does not benefit on having our optimization being
// applied. It serves the purpose as only being an example, with no actual
// functionality
__device__ __noinline__ void f_cheap(int *a) {
  *a = (*a * *a) * *a + *a * *a + *a + *a;
}

__device__ __noinline__ void f(int a) {
  for (int i = 0; i < MAX_ITER; i++) {
    // Do busy work, ensuring this statement has to be re-exucted to better
    // simulate real-world scenarios
    if (a == __funnelshift_r(a, a, a)) {
      a = 420;
    }
  }
}

// Example of a function that does not benefit on having our optimization being
// applied. It serves the purpose as only being an example, with no actual
// functionality
__device__ __noinline__ void f_cheap(int a) { a = (a * a) * a + a * a + a + a; }

__global__ void divergent_func_opt(int *ret) {
  int tx = threadIdx.x;
  int m = 128 * tx;
  int o = 128 * (tx + 2);
  bool p1 = tx < 16;
  if (p1) {
    FUNC(m);
  } else {
    FUNC(o);
  }
  // so DCE doesn't eliminate stuff
  ret[tx] = m + o;
  return;
}

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

__global__ void divergent_func_ptr(int *ret) {
  int tx = threadIdx.x;
  int m = 128 * tx;
  int o = 128 * (tx + 2);
  bool p1 = tx < 16;
  if (p1) {
    FUNC(&m);
  } else {
    FUNC(&o);
  }
  // so DCE doesn't eliminate stuff
  ret[tx] = m + o;
  return;
}

__global__ void divergent_func_ptr_opt(int *ret) {
  int tx = threadIdx.x;
  int m = 128 * tx;
  int o = 128 * (tx + 2);

  bool p1 = tx < 16;

  // Predicate the arguments
  int *j = &o;
  if (p1)
    j = &m;

  // Single function call, allowing the call to be done in lock-step
  FUNC(j);
  ret[tx] = m + o;
  return;
}

int main() {
  int *out, *d_out;
  // Allocate host memory
  out = (int *)malloc(sizeof(int) * 32);
  // Allocate device memory
  cudaMalloc((void **)&d_out, sizeof(int) * 32);
  // Executing kernel
  divergent_func_opt<<<1, 32>>>(d_out);
  cudaMemcpy(out, d_out, sizeof(int) * 32, cudaMemcpyDeviceToHost);
  for (int i = 0; i < 32; i++)
    printf("flat: out[%d] = %d\n", i, out[i]);
  printf("\n");

  divergent_func_opt<<<1, 32>>>(d_out);
  cudaMemcpy(out, d_out, sizeof(int) * 32, cudaMemcpyDeviceToHost);
  for (int i = 0; i < 32; i++)
    printf("flat_opt: out[%d] = %d\n", i, out[i]);
  printf("\n");

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

  // Deallocate device memory
  cudaFree(d_out);
  // Deallocate hostmemory
  free(out);
}
