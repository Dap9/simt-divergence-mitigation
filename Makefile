.PHONY = all clean

MAX_ITER=1000000
MOD=27
NVCC=nvcc
CXXFLAGS +=-rdc=true -std=c++17 -DMAX_ITER=$(MAX_ITER) -DMOD=$(MOD)
LDLIBS =-lm

release: CXXFLAGS += -O3 
release: all

debug: CXXFLAGS += -G
debug: all

ptx: CXXFLAGS += -ptx
ptx: all

all: single_arg_all two_arg_all three_arg_all switch_cases_all fib_all

# Functions are split into multiple files due to issues with profiling on WSL2

# single argument functions
single_arg_all: single_arg single_arg_opt single_arg_ptr single_arg_ptr_opt

single_arg: src/single_arg/single_arg.cu src/functions.cu
	$(NVCC) -o $@ $(CXXFLAGS) $(LDFLAGS) $^ $(LDLIBS)

single_arg_opt: src/single_arg/single_arg_opt.cu src/functions.cu
	$(NVCC) -o $@ $(CXXFLAGS) $(LDFLAGS) $^ $(LDLIBS)

single_arg_ptr: src/single_arg/single_arg_ptr.cu src/functions.cu
	$(NVCC) -o $@ $(CXXFLAGS) $(LDFLAGS) $^ $(LDLIBS)

single_arg_ptr_opt: src/single_arg/single_arg_ptr_opt.cu src/functions.cu
	$(NVCC) -o $@ $(CXXFLAGS) $(LDFLAGS) $^ $(LDLIBS)

# two argument functions
two_arg_all: two_arg two_arg_opt two_arg_ptr two_arg_ptr_opt

two_arg: src/two_arg/two_arg.cu src/functions.cu
	$(NVCC) -o $@ $(CXXFLAGS) $(LDFLAGS) $^ $(LDLIBS)

two_arg_opt: src/two_arg/two_arg_opt.cu src/functions.cu
	$(NVCC) -o $@ $(CXXFLAGS) $(LDFLAGS) $^ $(LDLIBS)

two_arg_ptr: src/two_arg/two_arg_ptr.cu src/functions.cu
	$(NVCC) -o $@ $(CXXFLAGS) $(LDFLAGS) $^ $(LDLIBS)

two_arg_ptr_opt: src/two_arg/two_arg_ptr_opt.cu src/functions.cu
	$(NVCC) -o $@ $(CXXFLAGS) $(LDFLAGS) $^ $(LDLIBS)

# three argument functions, just to show that number of arguments' impact
three_arg_all: three_arg three_arg_opt three_arg_ptr three_arg_ptr_opt

three_arg: src/three_arg/three_arg.cu src/functions.cu
	$(NVCC) -o $@ $(CXXFLAGS) $(LDFLAGS) $^ $(LDLIBS)

three_arg_opt: src/three_arg/three_arg_opt.cu src/functions.cu
	$(NVCC) -o $@ $(CXXFLAGS) $(LDFLAGS) $^ $(LDLIBS)

three_arg_ptr: src/three_arg/three_arg_ptr.cu src/functions.cu
	$(NVCC) -o $@ $(CXXFLAGS) $(LDFLAGS) $^ $(LDLIBS)

three_arg_ptr_opt: src/three_arg/three_arg_ptr_opt.cu src/functions.cu
	$(NVCC) -o $@ $(CXXFLAGS) $(LDFLAGS) $^ $(LDLIBS)

switch_cases_all: switch_cases switch_cases_opt

switch_cases: src/switch_cases/switch_cases.cu src/functions.cu
	$(NVCC) -o $@ $(CXXFLAGS) $(LDFLAGS) $^ $(LDLIBS)

switch_cases_opt: src/switch_cases/switch_cases_opt.cu src/functions.cu
	$(NVCC) -o $@ $(CXXFLAGS) $(LDFLAGS) $^ $(LDLIBS)

fib_all: fib fib_opt

fib: src/fib/fib.cu src/functions.cu
	$(NVCC) -o $@ $(CXXFLAGS) $(LDFLAGS) $^ $(LDLIBS)

fib_opt: src/fib/fib_opt.cu src/functions.cu
	$(NVCC) -o $@ $(CXXFLAGS) $(LDFLAGS) $^ $(LDLIBS)

clean: 
	rm -rf *.o *.dYSM\
		single_arg single_arg_opt single_arg_ptr single_arg_ptr_opt\
		two_arg two_arg_opt two_arg_ptr two_arg_ptr_opt\
		three_arg three_arg_opt three_arg_ptr three_arg_ptr_opt\
		switch_cases switch_cases_opt\
		fib fib_opt
