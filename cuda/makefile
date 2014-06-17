##===----------------------------------------------------------------------===##
##
##     KernelGen -- A prototype of LLVM-based auto-parallelizing Fortran/C
##        compiler for NVIDIA GPUs, targeting numerical modeling code.
##
## This file is distributed under the University of Illinois Open Source
## License. See LICENSE.TXT for details.
##
##===----------------------------------------------------------------------===##

-include ../../makefile.in

# Verbose setting
VERBOSE ?= 1
SILENT ?= @
ifeq ($(VERBOSE), 1)
SILENT =
endif

# Floating-point type: float or double
real ?= float

# PGI/OpenACC compiler/linker options
GPUARCH ?= 30
GPUCUDA ?= 5.0


#flags for compiling with OpenMP
OMP ?= 1
NVCCFLAGS ?= -Xcompiler -lnvToolsExt
ifeq ($(OMP), 1)
NVCCFLAGS += -Xcompiler -fopenmp
LIB ?= -lgomp
endif

NVCC ?= nvcc -I. -I.. -O3 -arch=sm_$(GPUARCH) $(NVCCFLAGS) $(LIB)

# Default problem domain dimensions
NX ?= 512
NY ?= 256
NS ?= 256
NT ?= 10

# 0 - default stencil code
# 1 - UVM memory, function running on host
# 2 - UVM memory, function running on device
VERSION ?= 0

all: wave13pt

wave13pt: wave13pt.o timing.o cuda_profiling.o
	$(SILENT)$(NVCC) $^ -o $@ -I /opt/cuda/lib -lrt -lnvToolsExt -Xlinker --wrap=cudaLaunch -Xcompiler -rdynamic

wave13pt.o: ../wave13pt.cu ../timing.h cuda_profiling.h
	$(SILENT)$(NVCC) -Dreal=$(real) -c $< -o $@

timing.o: ../timing.c ../timing.h
	$(SILENT)$(NVCC) -c $< -o $@

cuda_profiling.o: cuda_profiling.cu cuda_profiling.h
	$(SILENT)$(NVCC) -c $< -o $@

clean:
	$(SILENT)rm -rf *.o wave13pt

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
test: wave13pt
	$(SILENT)./$< $(NX) $(NY) $(NS) $(NT) $(VERSION)
#-------------------------------------------------------------------------------
test_default_d: wave13pt
	$(SILENT)./$< $(NX) $(NY) $(NS) $(NT) 0

test_uvm_h_force: wave13pt
	$(SILENT) CUDA_MANAGED_FORCE_DEVICE_ALLOC=1 ./$< $(NX) $(NY) $(NS) $(NT) 1

test_uvm_d: wave13pt
	$(SILENT)./$< $(NX) $(NY) $(NS) $(NT) 2

test_uvm_d_force: wave13pt
	$(SILENT) CUDA_MANAGED_FORCE_DEVICE_ALLOC=1 ./$< $(NX) $(NY) $(NS) $(NT) 2
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

check: wave13pt
	$(SILENT)cuda-memcheck ./$< $(NX) $(NY) $(NS) $(NT) $(VERSION)

ifeq ($(real), float)
roofline: wave13pt
	$(SILENT)PROFILING_FNAME=`cat kernel` ./$< $(NX) $(NY) $(NS) $(NT) $(VERSION) && \
		nvprof --metrics flops_sp,dram_read_throughput,dram_write_throughput ./$< $(NX) $(NY) $(NS) $(NT)
endif
ifeq ($(real), double)
roofline: wave13pt
	$(SILENT)PROFILING_FNAME=`cat kernel` ./$< $(NX) $(NY) $(NS) $(NT) $(VERSION) && \
		nvprof --metrics flops_sp,dram_read_throughput,dram_write_throughput ./$< $(NX) $(NY) $(NS) $(NT)
endif
