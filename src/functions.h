#ifndef SINGLE_ARG_H_
#define SINGLE_ARG_H_

__device__ __noinline__ void f(int a);
__device__ __noinline__ void f_cheap(int a);
__device__ __noinline__ void f(int *a);
__device__ __noinline__ void f_cheap(int *a);

__device__ __noinline__ void f(int a, int *result);
__device__ __noinline__ void f_cheap(int a, int *result);
__device__ __noinline__ void f(int *a, int *result);
__device__ __noinline__ void f_cheap(int *a, int *result);

__device__ __noinline__ void f(int a, int b, int *result);
__device__ __noinline__ void f_cheap(int a, int b, int *result);
__device__ __noinline__ void f(int *a, int *b, int *result);
__device__ __noinline__ void f_cheap(int *a, int *b, int *result);

__device__ __noinline__ void recursive_fibonacci(int fib, int *result);

#ifndef MAX_ITER
#define MAX_ITER 1e6
#endif

#ifndef MOD
#define MOD 27
#endif

#ifdef CHEAP
#define FUNC f_cheap
#else
#define FUNC f
#endif

#ifndef NUM_WARPS
#define NUM_WARPS 1
#endif

#ifndef NUM_THREADS_PER_WARP
#define NUM_THREADS_PER_WARP 32
#endif

#endif // SINGLE_ARG_H_
