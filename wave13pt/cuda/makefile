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
NVCCFLAGS ?= -g -Xcompiler -lnvToolsExt
ifeq ($(OMP), 1)
NVCCFLAGS += -Xcompiler -fopenmp
GCCFLAGS += -fopenmp
LIB ?= -lgomp
endif

NVCC ?= nvcc -I. -I.. -O3 -arch=sm_$(GPUARCH) $(NVCCFLAGS) $(LIB)
# GCC compiler/linker options
GCC ?= gcc -O3 -D_GNU_SOURCE -std=c99 $(GCCFLAGS)
LDFLAGS ?= -I/opt/cuda/include -L/opt/cuda/lib -lnvToolsExt

# Default problem domain dimensions
NX ?= 512
NY ?= 256
NS ?= 256
NT ?= 10

# 0 - default stencil code, regular pageable host explicit memory copies, stencil kernel running on device
# 1 - UVM memory(no explicit memcopies), stencil kernel running on host
# 2 - UVM memory(no explicit memcopies), stencil kernel running on device
# 3 - host uses pinned memory, explicit memory copies, stencil kernel running on device
# 4 - host uses pinned memory, device uses mapped host memory(no explicit memcopies), stencil kernel running on device
DEFAULT ?= 0
UVM_HOST ?= 1
UVM_DEVICE ?= 2
PINNED ?= 3
MAPPED ?= 4

# 1 - grid initialized on the host 
# 2 - grid initialized on the device
HOST_INIT ?= 1
DEVICE_INIT ?= 2

all: wave13pt wave13pt_cpu

#-------------------------------------------------------------------------------
#                                   GPU BUILD
#-------------------------------------------------------------------------------
wave13pt: wave13pt.o timing.o cuda_profiling.o
	$(SILENT)$(NVCC) $^ -o $@ -I /opt/cuda/lib -lrt -lnvToolsExt -Xlinker --wrap=cudaLaunch -Xcompiler -rdynamic

wave13pt.o: ../wave13pt.cu ../timing.h cuda_profiling.h
	$(SILENT)$(NVCC) -Dreal=$(real) -c $< -o $@

timing.o: ../timing.c ../timing.h
	$(SILENT)$(NVCC) -c $< -o $@

cuda_profiling.o: cuda_profiling.cu cuda_profiling.h
	$(SILENT)$(NVCC) -c $< -o $@

#-------------------------------------------------------------------------------
#                                   CPU BUILD
#-------------------------------------------------------------------------------
wave13pt_cpu: wave13pt_cpu.o timing_cpu.o
	$(SILENT)$(GCC) $^ -o $@ -lrt

wave13pt_cpu.o: ../wave13pt.c ../timing.h
	$(SILENT)$(GCC) -Dreal=$(real) -c $< -o $@

timing_cpu.o: ../timing.c ../timing.h
	$(SILENT)$(GCC) -c $< -o $@

#-------------------------------------------------------------------------------
#                                   TESTS SECTION
#-------------------------------------------------------------------------------
test_default_d: wave13pt
	$(SILENT)./$< $(NX) $(NY) $(NS) $(NT) $(DEFAULT) $(HOST_INIT)

test_default_h: wave13pt_cpu
	$(SILENT)./$< $(NX) $(NY) $(NS) $(NT) $(DEFAULT) $(HOST_INIT)

test_uvm_h_force: wave13pt
	$(SILENT) CUDA_MANAGED_FORCE_DEVICE_ALLOC=1 ./$< $(NX) $(NY) $(NS) $(NT) $(UVM_HOST) $(DEVICE_INIT)

test_uvm_h: wave13pt
	$(SILENT) ./$< $(NX) $(NY) $(NS) $(NT) $(UVM_HOST) $(DEVICE_INIT)

test_uvm_d_force: wave13pt
	$(SILENT) CUDA_MANAGED_FORCE_DEVICE_ALLOC=1 ./$< $(NX) $(NY) $(NS) $(NT) $(UVM_DEVICE) $(HOST_INIT)

test_uvm_d: wave13pt
	$(SILENT)./$< $(NX) $(NY) $(NS) $(NT) $(UVM_DEVICE) $(HOST_INIT)

test_pinned_d: wave13pt
	$(SILENT)./$< $(NX) $(NY) $(NS) $(NT) $(PINNED) $(HOST_INIT)

test_mapped_d: wave13pt
	$(SILENT)./$< $(NX) $(NY) $(NS) $(NT) $(MAPPED) $(HOST_INIT)
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
clean:
	$(SILENT)rm -rf *.o wave13pt wave13pt_cpu

check: wave13pt
	$(SILENT)cuda-memcheck ./$< $(NX) $(NY) $(NS) $(NT) $(DEFAULT) $(HOST_INIT)

ifeq ($(real), float)
roofline: wave13pt
	$(SILENT)PROFILING_FNAME=`cat kernel` ./$< $(NX) $(NY) $(NS) $(NT) $(DEFAULT) $(HOST_INIT) && \
		nvprof --metrics flops_sp,dram_read_throughput,dram_write_throughput ./$< $(NX) $(NY) $(NS) $(NT)
endif
ifeq ($(real), double)
roofline: wave13pt
	$(SILENT)PROFILING_FNAME=`cat kernel` ./$< $(NX) $(NY) $(NS) $(NT) $(DEFAULT) $(HOST_INIT) && \
		nvprof --metrics flops_sp,dram_read_throughput,dram_write_throughput ./$< $(NX) $(NY) $(NS) $(NT)
endif
