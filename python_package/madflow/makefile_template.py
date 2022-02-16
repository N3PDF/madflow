makefileName = "makefile"
cppCompiler = "g++"
cppVersion = "c++14"
cudaPath = ""
# example: cudaPath = "/usr/local/cuda"


def write_compilers(text):
    text += "CXX := " + cppCompiler + "\n"
    text += "NVCC := $(shell which nvcc)\n"
    text = write_nl(text)
    return text


def write_shell_name(text):
    text += "UNAME_S := $(shell uname -s)\n"
    text = write_nl(text)
    return text


def write_multithreading(text):
    text += "ifeq ($(UNAME_S), Darwin)\n"
    text += "NPROCS = $(shell sysctl -n hw.ncpu)\n"
    text += "else\n"
    text += "NPROCS = $(shell grep -c 'processor' /proc/cpuinfo)\n"
    text += "endif\n"
    # if the number of processors isn't found, default to 1
    text += "ifeq ($(NPROCS),)\n"
    text += "NPROCS = 1\n"
    text += "endif\n"
    text += "MAKEFLAGS += -j$(NPROCS)\n"
    text = write_nl(text)
    return text


def write_tf_generic_flags(text):
    text += "TF_CFLAGS = $(shell python3 -c 'import tensorflow as tf; print(\" \".join(tf.sysconfig.get_compile_flags()))')\n"
    text += "TF_LFLAGS = $(shell python3 -c 'import tensorflow as tf; print(\" \".join(tf.sysconfig.get_link_flags()))')\n"
    text = write_nl(text)
    return text


def write_tf_cuda_flags(text):
    text += "CUDA_LFLAGS = -x cu -Xcompiler -fPIC\n"
    text += "CUDA_PATH := $(shell echo ${CUDA_PATH})\n"
    text += "ifeq ($(CUDA_PATH),)\n"
    if cudaPath == "":
        text += 'CUDA_PATH = $(shell echo ${PATH} | sed -e "s&.*:\([^:]*cuda[^/]*\).*&\\1&g")\n'
    else:
        text += "CUDA_PATH = " + cudaPath + "\n"
    text += "endif\n"
    text = write_nl(text)
    return text


def write_omp_flags(text):
    text += "ifeq ($(UNAME_S), Darwin)\n"
    text += "OMP_CFLAGS = -Xpreprocessor -fopenmp -lomp\n"
    text += "else\n"
    text += "OMP_CFLAGS = -fopenmp\n"
    text += "endif\n"
    text = write_nl(text)
    return text


def write_cflags(text):
    text += "CFLAGS = ${TF_CFLAGS} ${OMP_CFLAGS} -fPIC -O2 -std=" + cppVersion + "\n"
    text += "LDFLAGS = -shared ${TF_LFLAGS}\n"
    text = write_nl(text)

    text = write_cflags_cuda(text)

    return text


def write_cflags_cuda(text):
    text += "ifeq ($(NVCC),)\n"
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
    text = write_nl(text)
    return text


def write_target(text):
    text += "TARGETS = $(shell ls gpu/ | grep \".h\" | sed 's/\.h/_cu.so/g')\n"
    text = write_nl(text)
    return text


def write_commands(text):
    text = write_generic_commands(text)
    text = write_library_commands(text)
    text = write_source_commands(text)
    text = write_cleanup_commands(text)
    return text


def write_generic_commands(text):
    text += "all: $(TARGETS)\n"
    text = write_nl(text)
    return text


def write_library_commands(text):
    text += "%_cu.so: gpu/%.cudao gpu/%.cu.cudao\n"
    text += "\t$(CXX) -o $@ $(CFLAGS_CUDA) $^ $(LDFLAGS_CUDA)\n"
    text = write_nl(text)
    return text


def write_source_commands(text):
    text += "%.o: %.cc\n"
    text += "\t$(CXX) -c $(CFLAGS) $^ -o $@\n"
    text = write_nl(text)
    text += "%.cu.cudao: %.cu.cc\n"
    text += "\t$(NVCC) -c $(CFLAGS_NVCC) $^ -o $@\n"
    text = write_nl(text)
    text += "%.cudao: %.cc\n"
    text += "\t$(CXX) -c $(CFLAGS_CUDA) $^ -o $@\n"
    text = write_nl(text)
    return text


def write_cleanup_commands(text):
    text += "clean:\n"
    text += "\trm -f $(TARGETS) $(OBJECT_SRCS_CUDA)\n"
    text = write_nl(text)
    text += "clean_all:\n"
    text += "\trm -f $(TARGETS) $(OBJECT_SRCS_CUDA)\n"
    text += "\trm -f gpu/*\n"
    text = write_nl(text)
    return text


def write_nl(text):
    text += "\n"
    return text


def write_makefile(destination):

    makefileContent = ""

    makefileContent = write_compilers(makefileContent)
    makefileContent = write_shell_name(makefileContent)
    makefileContent = write_multithreading(makefileContent)
    makefileContent = write_tf_generic_flags(makefileContent)
    makefileContent = write_tf_cuda_flags(makefileContent)
    # makefileContent = write_omp_flags(makefileContent)
    makefileContent = write_cflags(makefileContent)
    makefileContent = write_target(makefileContent)
    makefileContent = write_commands(makefileContent)

    with open(destination + makefileName, "w") as fh:
        fh.write(makefileContent)


if __name__ == "__main__":
    write_makefile("")
