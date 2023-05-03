#include "functions.h"

// This function is pure dead code as there is no effect, so this is completely
// eliminated in O3, however it gives the expected result when the optimiaztions
// are not applied. Thus for the sake of completeness this function is kept. An
// example of a function that takes in only 1 argument like this might be one
// that stores into a global variable. We do not speculate on the real
// functions, as they are specific to a developer's needs. The basic api of
// functions is the only thing relevent to our optimization and is thus what we
// test, regardless of what the function itself actually does
__device__ __noinline__ void f(int a) {
  for (int i = 0; i < MAX_ITER; i++) {
    // Do busy work, ensuring this statement has to be re-exucted to better
    // simulate real-world scenarios
    if (a + i == __funnelshift_r(a, a, a)) {
      a = 420;
    }
  }
  a = (a * a) + a;
}

// Example of a function that does not benefit on having our optimization being
// applied. It serves the purpose as only being an example, with no actual
// functionality
__device__ __noinline__ void f_cheap(int a) { a = (a * a) + a; }

// These functions are only to show example calls. They do not have any
// practicality to them as-is but the argument structure is useful for the sake
// of showing our optimization
__device__ __noinline__ void f(int *a) {
  for (int i = 0; i < MAX_ITER; i++) {
    // Do busy work, ensuring this statement has to be re-exucted to better
    // simulate real-world workloads
    if (*a + i == __funnelshift_r(*a, *a, *a)) {
      *a = 420;
    }
  }
  *a = (*a * *a) + *a;
}

// Example of a function that does not benefit on having our optimization being
// applied. It serves the purpose as only being an example, with no actual
// functionality
__device__ __noinline__ void f_cheap(int *a) { *a = (*a * *a) + *a; }

/*
 * Two arg functions
 */
__device__ __noinline__ void f(int a, int *result) {
  // Should be somewhat complex so that the runtime is significant and forces
  // unoptimized branches to be split/joined, not predicated (check PTX) must
  // write to result
  for (int i = 0; i < MAX_ITER; i++) {
    // Do busy work, ensuring this statement has to be re-exucted to better
    // simulate real-world scenarios
    if (a + i == __funnelshift_r(a, a, a) || *result == a + i) {
      a = 420;
    }
  }
  *result = (a * a) + a;
}

__device__ __noinline__ void f_cheap(int a, int *result) {
  *result = (a * a) + a;
}

__device__ __noinline__ void f(int *a, int *result) {
  // Should be somewhat complex so that the runtime is significant and forces
  // unoptimized branches to be split/joined, not predicated (check PTX) must
  // write to result
  for (int i = 0; i < MAX_ITER; i++) {
    // Do busy work, ensuring this statement has to be re-exucted to better
    // simulate real-world scenarios
    if (*a + i == __funnelshift_r(*a, *a, *a) || *result == *a + i) {
      *a = 420;
    }
  }
  *result = (*a * *a) + *a;
}

__device__ __noinline__ void f_cheap(int *a, int *result) {
  *result = (*a * *a) + *a;
}

/*
 * Three arg functions
 */
// 2 regular & 1 ptr
__device__ __noinline__ void f(int a, int b, int *result) {
  // Should be somewhat complex so that the runtime is significant and forces
  // unoptimized branches to be split/joined, not predicated (check PTX) must
  // write to result
  for (int i = 0; i < MAX_ITER; i++) {
    // Do busy work
    if (a + i == __funnelshift_r(a, a, a) || b + i == __funnelshift_r(b, b, b) || *result == a + i) {
      a = 420;
    }
  }
  *result = (a * b) + a * a + b * b + a + b;
}

__device__ __noinline__ void f_cheap(int a, int b, int *result) {
  *result = (a * b) + a * a + b * b + a + b;
}

__device__ __noinline__ void f(int *a, int *b, int *result) {
  // Should be somewhat complex so that the runtime is significant and forces
  // unoptimized branches to be split/joined, not predicated (check PTX) must
  // write to result
  for (int i = 0; i < MAX_ITER; i++) {
    // Do busy work
    if (*a + i == __funnelshift_r(*a, *a, *a) || *b + i == __funnelshift_r(*b, *b, *b) || *result == *a + i) {
      *a = 420;
    }
  }
  *result = (*a * *b) + *a * *a + *b * *b + *a + *b;
}

__device__ __noinline__ void f_cheap(int *a, int *b, int *result) {
  *result = (*a * *b) + *a * *a + *b * *b + *a + *b;
}

// Recursive Fibonacci to run on the GPU Thread
__device__ __noinline__ void recursive_fibonacci(int fib, int *result){
    if (fib <= 1){
        *result += fib;
        return;
    }
    recursive_fibonacci(fib - 1, result);
    recursive_fibonacci(fib - 2, result);
}
