#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>

#ifdef CHEAP
#define FUNC f_cheap
#else
#define FUNC f
#endif

#ifndef MAX_ITER
#define MAX_ITER 1e6
#endif

__device__ __noinline__ void f(int a, int *result) {
  // Should be somewhat complex so that the runtime is significant and forces
  // unoptimized branches to be split/joined, not predicated (check PTX) must
  // write to result
  for (int i = 0; i < MAX_ITER; i++) {
    // Do busy work
    if (a == __funnelshift_r(a, a, a)) {
      a = 420;
    }
  }
  *result = (a * a) + a * a + a + a;
}

__device__ __noinline__ void f(int a, int b, int *result) {
  // Should be somewhat complex so that the runtime is significant and forces
  // unoptimized branches to be split/joined, not predicated (check PTX) must
  // write to result
  for (int i = 0; i < MAX_ITER; i++) {
    // Do busy work
    if (a == b && b == -69) {
      a = 420;
    }
  }
  *result = (a * b) * b + a * b + b + b;
}

__device__ __noinline__ void f(int a, int b, int c, int *result) {
  // Should be somewhat complex so that the runtime is significant and forces
  // unoptimized branches to be split/joined, not predicated (check PTX) must
  // write to result
  for (int i = 0; i < MAX_ITER; i++) {
    // Do busy work
    if (a == b && b == -69 && c == 1231253123) {
      a = 420;
    }
  }
  *result = (a * b) * b + a * b + b + b + c + c * c + c;
}

__device__ __noinline__ void f_cheap(int a, int *result) {
  int b = 69;
  *result = (a * b) * b + a * b + b + b;
}

__global__ void divergent_func_ptr(int *ret) {
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

  if (p1 && p2 && p3) {
    FUNC(a, &b);
  } else if (p1 && p2 && !p3) {
    FUNC(a1, &b1);
  } else if (p1 && !p2 && p3) {
    FUNC(a2, &b2);
  } else if (p1 && !p2 && !p3) {
    FUNC(a3, &b3);
  } else if (!p1 && p2 && p3) {
    FUNC(a4, &b4);
  } else if (!p1 && p2 && !p3) {
    FUNC(a5, &b5);
  } else if (!p1 && !p2 && p3) {
    FUNC(a6, &b6);
  }
  else {
    FUNC(a7, &b7);
  }
  // so DCE doesn't eliminate stuff
  ret[tx] = b + b1 + b2 + b3 + b4 + b5 + b6 + b7;
  return;
}

__global__ void divergent_func_ptr_opt(int *ret) {
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
  int j = a7;
  int *t = &b7;

  if (!p1 && !p2 && p3) j = a6;
  if (!p1 && !p2 && p3) t = &b6;

  if (!p1 && p2 && !p3) j = a5;
  if (!p1 && p2 && !p3) t = &b5;

  if (!p1 && p2 && p3) j = a4;
  if (!p1 && p2 && p3) t = &b4;

  if (p1 && !p2 && !p3) j = a3;
  if (p1 && !p2 && !p3) t = &b3;

  if (p1 && !p2 && p3) j = a2;
  if (p1 && !p2 && p3) t = &b2;

  if (p1 && p2 && !p3) j = a1;
  if (p1 && p2 && !p3) t = &b1;

  if (p1 && p2 && p3) j = a;
  if (p1 && p2 && p3) t = &b;

  FUNC(j, t);
  ret[tx] = b + b1 + b2 + b3 + b4 + b5 + b6 + b7;
  return;
}

__global__ void divergent_func_nested(int *ret) {
  int tx = threadIdx.x;

  int a = 128 * tx;
  int c = a % 1;
  int b = 0;

  int a1 = 128 * (tx + 1);
  int c1 = a1 % 2;
  int b1 = 1;

  int a2 = 128 * (tx + 2);
  int c2 = a2 % 3;
  int b2 = 2;

  int a3 = 128 * (tx + 3);
  int c3 = a3 % 4;
  int b3 = 3;

  int a4 = 128 * (tx + 4);
  int c4 = a4 % 5;
  int b4 = 4;

  int a5 = 128 * (tx + 5);
  int c5 = a5 % 6;
  int b5 = 5;

  int a6 = 128 * (tx + 6);
  int c6 = a6 % 7;
  int b6 = 6;

  int a7 = 128 * (tx + 7);
  int c7 = a7 % 8;
  int b7 = 7;

  bool p1 = tx < 16;
  bool p2 = tx % 2 == 0;
  bool p3 = tx % 3 == 0;

  if (p1 && p2 && p3) {
    FUNC(a, c, &b);
  } else if (p1 && p2 && !p3) {
    FUNC(a1, c1, &b1);
  } else if (p1 && !p2 && p3) {
    FUNC(a2, c2, &b2);
  } else if (p1 && !p2 && !p3) {
    FUNC(a3, c3, &b3);
  } else if (!p1 && p2 && p3) {
    FUNC(a4, c4, &b4);
  } else if (!p1 && p2 && !p3) {
    FUNC(a5, c5, &b5);
  } else if (!p1 && !p2 && p3) {
    FUNC(a6, c6, &b6);
  } else {
    FUNC(a7, c7, &b7);
  }
  // so DCE doesn't eliminate stuff
  ret[tx] = b + b1 + b2 + b3 + b4 + b5 + b6 + b7;
  return;
}

__global__ void divergent_func_nested_opt(int *ret) {
  int tx = threadIdx.x;

  int a = 128 * tx;
  int c = a % 1;
  int b = 0;

  int a1 = 128 * (tx + 1);
  int c1 = a1 % 2;
  int b1 = 1;

  int a2 = 128 * (tx + 2);
  int c2 = a2 % 3;
  int b2 = 2;

  int a3 = 128 * (tx + 3);
  int c3 = a3 % 4;
  int b3 = 3;

  int a4 = 128 * (tx + 4);
  int c4 = a4 % 5;
  int b4 = 4;

  int a5 = 128 * (tx + 5);
  int c5 = a5 % 6;
  int b5 = 5;

  int a6 = 128 * (tx + 6);
  int c6 = a6 % 7;
  int b6 = 6;

  int a7 = 128 * (tx + 7);
  int c7 = a7 % 8;
  int b7 = 7;
  bool p1 = tx < 16;
  bool p2 = tx % 2 == 0;
  bool p3 = tx % 3 == 0;

  int j = a7;
  int k = c7;
  int *t = &b7;

  if (!p1 && !p2 && p3) j = a6;
  if (!p1 && !p2 && p3) k = c6;
  if (!p1 && !p2 && p3) t = &b6;

  if (!p1 && p2 && !p3) j = a5;
  if (!p1 && p2 && !p3) k = c5;
  if (!p1 && p2 && !p3) t = &b5;

  if (!p1 && p2 && p3) j = a4;
  if (!p1 && p2 && p3) k = c4;
  if (!p1 && p2 && p3) t = &b4;

  if (p1 && !p2 && !p3) j = a3;
  if (p1 && !p2 && !p3) k = c3;
  if (p1 && !p2 && !p3) t = &b3;

  if (p1 && !p2 && p3) j = a2;
  if (p1 && !p2 && p3) k = c2;
  if (p1 && !p2 && p3) t = &b2;

  if (p1 && p2 && !p3) j = a1;
  if (p1 && p2 && !p3) k = c1;
  if (p1 && p2 && !p3) t = &b1;

  if (p1 && p2 && p3) j = a;
  if (p1 && p2 && p3) k = c;
  if (p1 && p2 && p3) t = &b;

  FUNC(j, k, t);
  ret[tx] = b + b1 + b2 + b3 + b4 + b5 + b6 + b7;
  return;
}

__global__ void divergent_func_three_args(int *ret) {
  int tx = threadIdx.x;

  int a = 128 * tx;
  int c = a % 1;
  int d = a % 10;
  int b = 0;

  int a1 = 128 * (tx + 1);
  int c1 = a1 % 2;
  int d1 = a1 % 20;
  int b1 = 1;

  int a2 = 128 * (tx + 2);
  int c2 = a2 % 3;
  int d2 = a2 % 30;
  int b2 = 2;

  int a3 = 128 * (tx + 3);
  int c3 = a3 % 4;
  int d3 = a3 % 40;
  int b3 = 3;

  int a4 = 128 * (tx + 4);
  int c4 = a4 % 5;
  int d4 = a4 % 50;
  int b4 = 4;

  int a5 = 128 * (tx + 5);
  int c5 = a5 % 6;
  int d5 = a5 % 60;
  int b5 = 5;

  int a6 = 128 * (tx + 6);
  int c6 = a6 % 7;
  int d6 = a6 % 70;
  int b6 = 6;

  int a7 = 128 * (tx + 7);
  int c7 = a7 % 8;
  int d7 = a7 % 80;
  int b7 = 7;

  bool p1 = tx < 16;
  bool p2 = tx % 2 == 0;
  bool p3 = tx % 3 == 0;

  if (p1 && p2 && p3) {
    FUNC(a, c, d, &b);
  } else if (p1 && p2 && !p3) {
    FUNC(a1, c1, d1, &b1);
  } else if (p1 && !p2 && p3) {
    FUNC(a2, c2, d2, &b2);
  } else if (p1 && !p2 && !p3) {
    FUNC(a3, c3, d3, &b3);
  } else if (!p1 && p2 && p3) {
    FUNC(a4, c4, d4, &b4);
  } else if (!p1 && p2 && !p3) {
    FUNC(a5, c5, d5, &b5);
  } else if (!p1 && !p2 && p3) {
    FUNC(a6, c6, d6, &b6);
  } else {
    FUNC(a7, c7, d7, &b7);
  }
  // so DCE doesn't eliminate stuff
  ret[tx] = b + b1 + b2 + b3 + b4 + b5 + b6 + b7;
  return;
}

__global__ void divergent_func_three_args_opt(int *ret) {
  int tx = threadIdx.x;

  int a = 128 * tx;
  int c = a % 1;
  int d = a % 10;
  int b = 0;

  int a1 = 128 * (tx + 1);
  int c1 = a1 % 2;
  int d1 = a1 % 20;
  int b1 = 1;

  int a2 = 128 * (tx + 2);
  int c2 = a2 % 3;
  int d2 = a2 % 30;
  int b2 = 2;

  int a3 = 128 * (tx + 3);
  int c3 = a3 % 4;
  int d3 = a3 % 40;
  int b3 = 3;

  int a4 = 128 * (tx + 4);
  int c4 = a4 % 5;
  int d4 = a4 % 50;
  int b4 = 4;

  int a5 = 128 * (tx + 5);
  int c5 = a5 % 6;
  int d5 = a5 % 60;
  int b5 = 5;

  int a6 = 128 * (tx + 6);
  int c6 = a6 % 7;
  int d6 = a6 % 70;
  int b6 = 6;

  int a7 = 128 * (tx + 7);
  int c7 = a7 % 8;
  int d7 = a7 % 80;
  int b7 = 7;

  bool p1 = tx < 16;
  bool p2 = tx % 2 == 0;
  bool p3 = tx % 3 == 0;

  int j = a7;
  int k = c7;
  int l = d7;
  int *t = &b7;

  if (!p1 && !p2 && p3) j = a6;
  if (!p1 && !p2 && p3) k = c6;
  if (!p1 && !p2 && p3) l = d6;
  if (!p1 && !p2 && p3) t = &b6;

  if (!p1 && p2 && !p3) j = a5;
  if (!p1 && p2 && !p3) k = c5;
  if (!p1 && p2 && !p3) l = d5;
  if (!p1 && p2 && !p3) t = &b5;

  if (!p1 && p2 && p3) j = a4;
  if (!p1 && p2 && p3) k = c4;
  if (!p1 && p2 && p3) l = d4;
  if (!p1 && p2 && p3) t = &b4;

  if (p1 && !p2 && !p3) j = a3;
  if (p1 && !p2 && !p3) k = c3;
  if (p1 && !p2 && !p3) l = d3;
  if (p1 && !p2 && !p3) t = &b3;

  if (p1 && !p2 && p3) j = a2;
  if (p1 && !p2 && p3) k = c2;
  if (p1 && !p2 && p3) l = d2;
  if (p1 && !p2 && p3) t = &b2;

  if (p1 && p2 && !p3) j = a1;
  if (p1 && p2 && !p3) k = c1;
  if (p1 && p2 && !p3) l = d1;
  if (p1 && p2 && !p3) t = &b1;

  if (p1 && p2 && p3) j = a;
  if (p1 && p2 && p3) k = c;
  if (p1 && p2 && p3) l = d;
  if (p1 && p2 && p3) t = &b;

  FUNC(j, k, l, t);
  ret[tx] = b + b1 + b2 + b3 + b4 + b5 + b6 + b7;
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

  divergent_func_three_args<<<1, 32>>>(d_out);
  cudaMemcpy(out, d_out, sizeof(int) * 32, cudaMemcpyDeviceToHost);
  for (int i = 0; i < 32; i++)
    printf("nested_cond_exec: out[%d] = %d\n", i, out[i]);
  printf("\n");

  divergent_func_three_args_opt<<<1, 32>>>(d_out);
  cudaMemcpy(out, d_out, sizeof(int) * 32, cudaMemcpyDeviceToHost);
  for (int i = 0; i < 32; i++)
    printf("nested_cond_exec_opt: out[%d] = %d\n", i, out[i]);
  printf("\n");
  // Deallocate device memory
  cudaFree(d_out);
  // Deallocate hostmemory
  free(out);
}

