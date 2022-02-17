# Types

INT64Type = "int64_t"
doubleType = "double"
complexType = "complex128"

# Libraries (#include <libraryName>)
# Used in Matrix_1_xxxx.h

libraries = ["iostream", "math.h", "unsupported/Eigen/CXX11/Tensor"]

# Header files (#include "Header.h")

headers_ = [
    "tensorflow/core/framework/op.h",
    "tensorflow/core/framework/op_kernel.h",
    "tensorflow/core/util/work_sharder.h",
    "tensorflow/core/framework/shape_inference.h",
    "tensorflow/cc/ops/array_ops.h",
    "tensorflow/cc/ops/math_ops.h",
]

# Namespaces

namespace = "tensorflow"

# Constants

defined = [
    "COMPLEX_CONJUGATE std::conj",
    "MAXIMUM std::max",
    "MINIMUM std::min",
    "CPUDevice Eigen::ThreadPoolDevice",
    "GPUDevice Eigen::GpuDevice",
    "DEFAULT_BLOCK_SIZE 32",
]
globalConstants = [
    "const " + doubleType + " SQH = 0.70710676908493",
    "const COMPLEX_TYPE CZERO = COMPLEX_TYPE(0.0, 0.0)",
]
cpuConstants = ["using thread::ThreadPool"]  # Not used for the GPU Op
