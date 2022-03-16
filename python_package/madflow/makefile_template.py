makefile_name = "makefile"
cpp_compiler = "g++"
cpp_version = "c++14"
cuda_path = ""
# example: cuda_path = "/usr/local/cuda"


def write_compilers():
    """Adds C++ and CUDA compilers"""
    text = f"""CXX := {cpp_compiler}
NVCC := $(shell which nvcc)

"""
    return text


def write_shell_name():
    """Adds a line for the kernel name"""
    text = f"""UNAME_S := $(shell uname -s)

"""
    return text


def write_multithreading():
    """Find the number of processors and use as many threads as possible
    if the number of processors isn't found, default to 1"""
    text = f"""ifeq ($(UNAME_S), Darwin)
NPROCS = $(shell sysctl -n hw.ncpu)
else
NPROCS = $(shell grep -c 'processor' /proc/cpuinfo)
endif
ifeq ($(NPROCS),)
NPROCS = 1
endif
MAKEFLAGS += -j$(NPROCS)

"""
    return text


def write_tf_generic_flags():
    """Adds TensorFlow flags"""
    text = f"""TF_CFLAGS = $(shell python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS = $(shell python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

"""
    return text


def write_tf_cuda_flags():
    """Adds TansorFlow CUDA flags and the path to CUDA libraries.
    If the environment variable ${CUDA_PATH} isn't defined, use a default path"""
    #"""
    text = f"""CUDA_LFLAGS = -x cu -Xcompiler -fPIC
CUDA_PATH := $(shell echo ${{CUDA_PATH}})
ifeq ($(CUDA_PATH),)

"""
    """If the path for CUDA libraries isn't explicitly stated at the beginning of this file, find it from ${PATH}"""
    if cuda_path == "":
        text += 'CUDA_PATH = $(shell echo ${PATH} | sed -e "s&.*:\([^:]*cuda[^/]*\).*&\\1&g")\n'
    else:
        text += "CUDA_PATH = " + cuda_path + "\n"
    text += f"""endif

"""
    return text


def write_omp_flags():
    """Adds flags for OpenMP parallelization"""
    text = f"""ifeq ($(UNAME_S), Darwin)
OMP_CFLAGS = -Xpreprocessor -fopenmp -lomp
else
OMP_CFLAGS = -fopenmp
endif

"""
    return text


def write_cflags():
    """Adds C-Flags. C++ version is defined at the beginning of this file"""
    text = f"""CFLAGS = ${{TF_CFLAGS}} ${{OMP_CFLAGS}} -fPIC -O2 -std={cpp_version}
LDFLAGS = -shared ${{TF_LFLAGS}}

"""

    text += write_cflags_cuda()

    return text


def write_cflags_cuda():
    """Adds C-Flags for CUDA (only if nvcc is found)"""
    # fallback in case nvcc is not installed => use CXX
    text = f"""ifeq ($(NVCC),)
CFLAGS_CUDA = $(CFLAGS)
CFLAGS_NVCC = ${{TF_CFLAGS}}
LDFLAGS_CUDA = $(LDFLAGS)
NVCC = $(CXX)
else
CFLAGS_CUDA = $(CFLAGS) -D GOOGLE_CUDA=1 -I$(CUDA_PATH)/include
CFLAGS_NVCC = ${{TF_CFLAGS}} -O2 -std={cpp_version} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -DNDEBUG --expt-relaxed-constexpr
LDFLAGS_CUDA = $(LDFLAGS) -L$(CUDA_PATH)/lib64 -lcudart
endif

"""
    return text


def write_target():
    """Adds the name of the generated library (matrix_processName_cu.so)"""
    text = f"""TARGETS = $(shell ls gpu/ | grep ".h" | sed 's/\.h/_cu.so/g')

"""
    return text


def write_commands():
    """Adds commands for compiling of source code"""
    text = ""
    text += write_generic_commands()
    text += write_library_commands()
    text += write_source_commands()
    text += write_cleanup_commands()
    return text


def write_generic_commands():
    """all compiles all target libraries (one for each subprocess, i.e.: qq~ -> X + gg -> X"""
    text = f"""all: $(TARGETS)

"""
    return text


def write_library_commands():
    """Compile each library from object files"""
    text = f"""%_cu.so: gpu/%.cudao gpu/%.cu.cudao
\t$(CXX) -o $@ $(CFLAGS_CUDA) $^ $(LDFLAGS_CUDA)

"""
    return text


def write_source_commands():
    """Generate object files"""
    text = f"""%.o: %.cc
\t$(CXX) -c $(CFLAGS) $^ -o $@

%.cu.cudao: %.cu.cc
\t$(NVCC) -c $(CFLAGS_NVCC) $^ -o $@

%.cudao: %.cc
\t$(CXX) -c $(CFLAGS_CUDA) $^ -o $@

"""
    return text


def write_cleanup_commands():
    """Adds commmand for cleanup"""
    # remove generated libraries
    # remove generated libraries and source code
    text = f"""clean:
\trm -f $(TARGETS) $(OBJECT_SRCS_CUDA)

clean_all:
\trm -f $(TARGETS) $(OBJECT_SRCS_CUDA)
\trm -f gpu/*

"""
    return text


def write_makefile(destination):

    makefile_content = ""

    makefile_content += write_compilers()
    makefile_content += write_shell_name()
    makefile_content += write_multithreading()
    makefile_content += write_tf_generic_flags()
    makefile_content += write_tf_cuda_flags()
    # makefile_content += write_omp_flags()
    makefile_content += write_cflags()
    makefile_content += write_target()
    makefile_content += write_commands()

    # write the makefile
    with open(destination + makefile_name, "w") as fh:
        fh.write(makefile_content)


if __name__ == "__main__":
    write_makefile("")
