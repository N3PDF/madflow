"""Global constants and libraries used by the Custom Operator"""

# Types

INT64_TYPE = "int64_t"
DOUBLE_TYPE = "double"
COMPLEX_TYPE = "complex128"

# Parallelization

CPU_PARALLELIZATION = "ThreadPool"
GPU_PARALLELIZATION = "CUDA"

# Libraries (#include <libraryName>)
# Used in Matrix_xxxx.h

LIBRARIES = ["math.h", "unsupported/Eigen/CXX11/Tensor"]

# Header files (#include "Header.h")

HEADERS_ = [
    "tensorflow/core/framework/op.h",
    "tensorflow/core/framework/op_kernel.h",
    "tensorflow/core/util/work_sharder.h",
    "tensorflow/core/framework/shape_inference.h",
    "tensorflow/cc/ops/array_ops.h",
    "tensorflow/cc/ops/math_ops.h",
]

# Namespaces

NAMESPACE = "tensorflow"

# Constants

DEFINED = [
    "COMPLEX_CONJUGATE std::conj",
    "MAXIMUM std::max",
    "MINIMUM std::min",
    "CPUDevice Eigen::ThreadPoolDevice",
    "GPUDevice Eigen::GpuDevice",
    "DEFAULT_BLOCK_SIZE 32",
]
GLOBAL_CONSTANTS = [
    "const " + DOUBLE_TYPE + " SQH = 0.70710676908493",
    "const COMPLEX_TYPE CZERO = COMPLEX_TYPE(0.0, 0.0)",
]
CPU_CONSTANTS = ["using thread::ThreadPool"]  # Not used for the GPU Op
