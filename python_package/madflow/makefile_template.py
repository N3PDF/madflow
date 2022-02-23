makefileName = "makefile"
cppCompiler = "g++"
cppVersion = "c++14"
cudaPath = ""
# example: cudaPath = "/usr/local/cuda"


def write_compilers():
    """Adds C++ and CUDA compilers"""
    text = "CXX := " + cppCompiler + "\n"
    text += "NVCC := $(shell which nvcc)\n"
    text += "\n"
    return text


def write_shell_name():
    """Adds a line for the kernel name"""
    text = "UNAME_S := $(shell uname -s)\n"
    text += "\n"
    return text


def write_multithreading():
    """Find the number of processors and use as many threads as possible"""
    text = "ifeq ($(UNAME_S), Darwin)\n"
    text += "NPROCS = $(shell sysctl -n hw.ncpu)\n"
    text += "else\n"
    text += "NPROCS = $(shell grep -c 'processor' /proc/cpuinfo)\n"
    text += "endif\n"
    # if the number of processors isn't found, default to 1
    text += "ifeq ($(NPROCS),)\n"
    text += "NPROCS = 1\n"
    text += "endif\n"
    text += "MAKEFLAGS += -j$(NPROCS)\n"
    text += "\n"
    return text


def write_tf_generic_flags():
    """Adds TensorFlow flags"""
    text = "TF_CFLAGS = $(shell python3 -c 'import tensorflow as tf; print(\" \".join(tf.sysconfig.get_compile_flags()))')\n"
    text += "TF_LFLAGS = $(shell python3 -c 'import tensorflow as tf; print(\" \".join(tf.sysconfig.get_link_flags()))')\n"
    text += "\n"
    return text


def write_tf_cuda_flags():
    """Adds TansorFlow CUDA flags and the path to CUDA libraries.
    If the environment variable ${CUDA_PATH} isn't defined, use a default path"""
    text = "CUDA_LFLAGS = -x cu -Xcompiler -fPIC\n"
    text += "CUDA_PATH := $(shell echo ${CUDA_PATH})\n"
    text += "ifeq ($(CUDA_PATH),)\n"
    """If the path for CUDA libraries isn't explicitly stated at the beginning of this file, find it from ${PATH}"""
    if cudaPath == "":
        text += 'CUDA_PATH = $(shell echo ${PATH} | sed -e "s&.*:\([^:]*cuda[^/]*\).*&\\1&g")\n'
    else:
        text += "CUDA_PATH = " + cudaPath + "\n"
    text += "endif\n"
    text += "\n"
    return text


def write_omp_flags():
    """Adds flags for OpenMP parallelization"""
    text = "ifeq ($(UNAME_S), Darwin)\n"
    text += "OMP_CFLAGS = -Xpreprocessor -fopenmp -lomp\n"
    text += "else\n"
    text += "OMP_CFLAGS = -fopenmp\n"
    text += "endif\n"
    text += "\n"
    return text


def write_cflags():
    """Adds C-Flags. C++ version is defined at the beginning of this file"""
    text = "CFLAGS = ${TF_CFLAGS} ${OMP_CFLAGS} -fPIC -O2 -std=" + cppVersion + "\n"
    text += "LDFLAGS = -shared ${TF_LFLAGS}\n"
    text += "\n"

    text += write_cflags_cuda()

    return text


def write_cflags_cuda():
    """Adds C-Flags for CUDA (only if nvcc is found)"""
    text = "ifeq ($(NVCC),)\n"
    text += "CFLAGS_CUDA = $(CFLAGS)\n"
    text += "CFLAGS_NVCC = ${TF_CFLAGS}\n"
    text += "LDFLAGS_CUDA = $(LDFLAGS)\n"
    text += "NVCC = $(CXX)\n"  # fallback in case nvcc is not installed
    text += "else\n"
    text += "CFLAGS_CUDA = $(CFLAGS) -D GOOGLE_CUDA=1 -I$(CUDA_PATH)/include\n"
    text += (
        "CFLAGS_NVCC = ${TF_CFLAGS} -O2 -std="
        + cppVersion
        + " -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -DNDEBUG --expt-relaxed-constexpr\n"
    )
    text += "LDFLAGS_CUDA = $(LDFLAGS) -L$(CUDA_PATH)/lib64 -lcudart\n"
    text += "endif\n"
    text += "\n"
    return text


def write_target():
    """Adds the name of the generated library (matrix_processName_cu.so)"""
    text = "TARGETS = $(shell ls gpu/ | grep \".h\" | sed 's/\.h/_cu.so/g')\n"
    text += "\n"
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
    text = "all: $(TARGETS)\n"
    text += "\n"
    return text


def write_library_commands():
    """Compile each library from object files"""
    text = "%_cu.so: gpu/%.cudao gpu/%.cu.cudao\n"
    text += "\t$(CXX) -o $@ $(CFLAGS_CUDA) $^ $(LDFLAGS_CUDA)\n"
    text += "\n"
    return text


def write_source_commands():
    """Generate object files"""
    text = "%.o: %.cc\n"
    text += "\t$(CXX) -c $(CFLAGS) $^ -o $@\n"
    text += "\n"
    text += "%.cu.cudao: %.cu.cc\n"
    text += "\t$(NVCC) -c $(CFLAGS_NVCC) $^ -o $@\n"
    text += "\n"
    text += "%.cudao: %.cc\n"
    text += "\t$(CXX) -c $(CFLAGS_CUDA) $^ -o $@\n"
    text += "\n"
    return text


def write_cleanup_commands():
    """Adds commmand for cleanup"""
    # remove generated libraries
    text = "clean:\n"
    text += "\trm -f $(TARGETS) $(OBJECT_SRCS_CUDA)\n"
    text += "\n"
    # remove generated libraries and source code
    text += "clean_all:\n"
    text += "\trm -f $(TARGETS) $(OBJECT_SRCS_CUDA)\n"
    text += "\trm -f gpu/*\n"
    text += "\n"
    return text


def write_makefile(destination):

    makefileContent = ""

    makefileContent += write_compilers()
    makefileContent += write_shell_name()
    makefileContent += write_multithreading()
    makefileContent += write_tf_generic_flags()
    makefileContent += write_tf_cuda_flags()
    # makefileContent += write_omp_flags()
    makefileContent += write_cflags()
    makefileContent += write_target()
    makefileContent += write_commands()

    # write the makefile
    with open(destination + makefileName, "w") as fh:
        fh.write(makefileContent)


if __name__ == "__main__":
    write_makefile("")
