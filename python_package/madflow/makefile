CXX:=g++
NVCC:=#/opt/cuda/bin/nvcc

UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S), Darwin)
NPROCS = $(shell sysctl -n hw.ncpu)
CXX:=clang++
else
CXX:=g++
NVCC:=/usr/local/cuda-11.5/bin/nvcc
NPROCS = $(shell grep -c 'processor' /proc/cpuinfo)
endif

ifeq ($(NPROCS), "")
NPROCS=1
endif

MAKEFLAGS += -j$(NPROCS)

TF_CFLAGS=$(shell python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS=$(shell python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')
#PATH_TO_INCLUDE=$(shell python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
CUDA_LFLAGS= -x cu -Xcompiler -fPIC
#CUDA_PATH=/opt/cuda
CUDA_PATH=/usr/local/cuda-11.5

ifeq ($(UNAME_S), Darwin)
OMP_CFLAGS = -Xpreprocessor -fopenmp -lomp
else
OMP_CFLAGS = -fopenmp
endif

CSRCS = $(wildcard cpu/*.cc)
GSRCS = $(wildcard gpu/*.cc)

CUDASRC = $(wildcard gpu/*.cu.cc)
SOURCES = $(filter-out $(CUDASRC), $(GSRCS))

TARGET_LIB=matrix.so
#TARGET_LIB_CUDA=matrix_cu.so

#TARGETS=$(TARGET_LIB)

#TARGETS+=$(TARGET_LIB_CUDA)

#OBJECT_SRCS = $(SOURCES:.cc=.o)
#OBJECT_SRCS_CUDA = $(SRCS:.cc=.cudao)

OBJECT_SRCS = $(CSRCS:.cc=.o)
OBJECT_SRCS_CUDA = $(GSRCS:.cc=.cudao)

CFLAGS = ${TF_CFLAGS} ${OMP_CFLAGS} -fPIC -O2 -std=c++14
CFLAGS_CUDA = $(CFLAGS) -D GOOGLE_CUDA=1 -I$(CUDA_PATH)/include
CFLAGS_NVCC = ${TF_CFLAGS} -O2 -std=c++14 -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -DNDEBUG --expt-relaxed-constexpr

LDFLAGS = -shared ${TF_LFLAGS}
LDFLAGS_CUDA = $(LDFLAGS) -L$(CUDA_PATH)/lib64 -lcudart

#TARGET_LIB=$(shell ls cpu/ | grep ".h" | sed 's/\.h/.so/g')
TARGET_LIB_CUDA=$(shell ls gpu/ | grep ".h" | sed 's/\.h/_cu.so/g')

#TARGETS=$(TARGET_LIB)

TARGETS=$(TARGET_LIB_CUDA)

all: $(TARGET_LIB_CUDA)

cpu: $(TARGET_LIB)

gpu: $(TARGET_LIB_CUDA)

$(TARGET_LIB): $(OBJECT_SRCS)
	$(CXX) -o $@ $(CFLAGS) $^ $(LDFLAGS)

#$(TARGET_LIB_CUDA): $(OBJECT_SRCS_CUDA)
#	$(CXX) -o $@ $(CFLAGS_CUDA) $^ $(LDFLAGS_CUDA)

%_cu.so: gpu/%.cudao gpu/%.cu.cudao
	$(CXX) -o $@ $(CFLAGS_CUDA) $^ $(LDFLAGS_CUDA)

#matrix_gg_gttx_cu.so: gpu/matrix_gg_gttx.cudao gpu/matrix_gg_gttx.cu.cudao
#	$(CXX) -o matrix_gg_gttx_cu.so $(CFLAGS_CUDA) gpu/matrix_gg_gttx.cudao gpu/matrix_gg_gttx.cu.cudao $(LDFLAGS_CUDA)
#matrix_uux_gttx_cu.so: gpu/matrix_uux_gttx.cudao gpu/matrix_uux_gttx.cu.cudao
#	$(CXX) -o matrix_uux_gttx_cu.so $(CFLAGS_CUDA) gpu/matrix_uux_gttx.cudao gpu/matrix_uux_gttx.cu.cudao $(LDFLAGS_CUDA)

%.o: %.cc
	$(CXX) -c $(CFLAGS) $^ -o $@

%.cu.cudao: %.cu.cc
	$(NVCC) -c $(CFLAGS_NVCC) $^ -o $@

%.cudao: %.cc
	$(CXX) -c $(CFLAGS_CUDA) $^ -o $@

#cpu: zero_out.cc
#	$(CXX) -std=c++11 -I $(PATH_TO_INCLUDE) -shared zero_out.cc -o zero_out.so -fPIC $(TF_CFLAGS) $(TF_LFLAGS) -O2
	
#gpu: kernel_example.cc kernel_example.cu.cc kernel_example.h
#	$(NVCC) -std=c++11 -c kernel_example.cu.cc $(CUDA_LFLAGS) -o kernel_example.cu.o
#	$(CXX) -std=c++11 -I $(PATH_TO_INCLUDE) -shared kernel_example.cc kernel_example.cu.o kernel_example.h $(CUDA_LIB) -lcudart -o kernel_example.so -fPIC $(TF_CFLAGS) $(TF_LFLAGS) -O2

test: cpu test.py
	python3 test.py

clean:
	rm -f $(TARGETS) $(OBJECT_SRCS) $(OBJECT_SRCS_CUDA)

clean_all:
	rm -f $(TARGETS) $(OBJECT_SRCS) $(OBJECT_SRCS_CUDA)
	rm -rf prov/
	rm -f gpu/*
	rm -f matrix_1*

