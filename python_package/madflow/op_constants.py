from madflow.op_global_constants import doubleType, complexType

# Templates ----------------------

libraryTemplate = (
    "{% for lib in variableName %}\
"
    "#include <{{ lib }}>\n\
"
    "{% endfor %}"
)

headerTemplate = (
    "{% for head in variableName %}\
"
    '#include "{{ head }}"\n\
'
    "{% endfor %}"
)

constantVariableTemplate = (
    "{% for var in constantVariable %}\
"
    "{{ dev }}{{ var}};\n\
"
    "{% endfor %}"
)

definedConstantTemplate = (
    "{% for var in variableName %}\
"
    "#define {{ var }}\n\
"
    "{% endfor %}"
)

functionDefinitionTemplate = "\
{{ func.template }}\n\
{{ dev }}{{ func.type }} {{ func.name }} (\
{% for i in range(func.argn - 1) %}\
{{ func.args[i].type }} {{ func.args[i].name }}, \
{% endfor %}\
{{ func.args[func.argn - 1].type }} {{ func.args[func.argn - 1].name }}\
);"

functionTemplate = "\
{{ func.template }}\n\
{{ dev }}{{ func.type }} {{ func.name }} (\
{% for i in range(func.argn - 1) %}\
{{ func.args[i].type }} {{ func.args[i].name }}, \
{% endfor %}\
{{ func.args[func.argn - 1].type }} {{ func.args[func.argn - 1].name }}\
) {\n\
{% for scope in func.scope %}\
    {{ scope }}\n\
{% endfor %}\
}"

headerFileTemplate = (
    '#ifndef MATRIX_H_\n\
#define MATRIX_H_\n\
\n\
#include <omp.h>\n\
#include <unsupported/Eigen/CXX11/Tensor>\n\
\n\
#include "tensorflow/core/framework/op.h"\n\
#include "tensorflow/core/framework/op_kernel.h"\n\
using namespace tensorflow;\n\
\n\
template <typename Device, typename T>\n\
struct MatrixFunctor {\n\
  void operator()(const Device& d, \
{% for i in range(func.argn - 1) %}\
{{ func.args[i].type }} {{ func.args[i].name }}, \
{% endfor %}\
{{ func.args[func.argn - 1].type }} {{ func.args[func.argn - 1].name }}\
);\n\
};\n\
\n\
#if GOOGLE_CUDA\n\
// Partially specialize functor for GpuDevice.\n\
template <typename T>\n\
struct MatrixFunctor<Eigen::GpuDevice, T> {\n\
  void operator()(const Eigen::GpuDevice& d, \
{% for i in range(func.argn - 1) %}\
{{ func.args[i].type }} {{ func.args[i].name }}, \
{% endfor %}\
{{ func.args[func.argn - 1].type }} {{ func.args[func.argn - 1].name }}\
);\n\
};\n\
#endif\n\
\n\
\n\
#define COMPLEX_TYPE '
    + complexType
    + "\n\
\n\
#endif"
)

cpuOpTemplate = '\
REGISTER_OP("Matrix{{ process }}")\n\
    .Attr("T: numbertype")\n\
{% for i in range(func.argn - 3) %}\
    .Input("{{ func.args[i].name|lower }}: {{ op_types[i] }}")\n\
{% endfor %}\
    .Output("{{ func.args[func.argn - 3].name|lower }}: {{ op_types[func.argn - 3] }}")\n\
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {\n\
      c->set_output(0, c->MakeShape({c->Dim(c->input(0), 0)}));\n\
      return Status::OK();\n\
    });\n\
\
\
\
template <typename T>\n\
struct {{ custom_op.functor_name }}<CPUDevice, T> {\n\
  void operator()(const CPUDevice& d,\
{% for i in range(func.argn - 1) %}\
{{ func.args[i].type }} {{ func.args[i].name }}, \
{% endfor %}\
{{ func.args[func.argn - 1].type }} {{ func.args[func.argn - 1].name }}\
) {\n\
      {{ func.name }}({% for i in range(func.argn - 1) %}\
{{ func.args[i].name }}, \
{% endfor %}\
{{ func.args[func.argn - 1].name }}\
);\n\
  }\n\
};\n\
\
\
\
template <typename Device, typename T>\n\
class {{ custom_op.name }} : public OpKernel {\n\
 public:\n\
  explicit {{ custom_op.name }}(OpKernelConstruction* context) : OpKernel(context) {}\n\
\n\
  void Compute(OpKernelContext* context) override {\n\
{% for scope in custom_op.scope %}\
    {{ scope }}\n\
{% endfor %}\
  }\n\
};\n\
#define REGISTER_CPU(T)\\\n\
  REGISTER_KERNEL_BUILDER(\\\n\
      Name("Matrix{{ process }}").Device(DEVICE_CPU).TypeConstraint<T>("T"),\\\n\
      MatrixOp<CPUDevice, T>);\n\
REGISTER_CPU(COMPLEX_TYPE);\n\
\n\
// Register the GPU kernels.\n\
#ifdef GOOGLE_CUDA\n\
#define REGISTER_GPU(T)\\\n\
  /* Declare explicit instantiations in kernel_example.cu.cc. */\\\n\
  extern template class MatrixFunctor<GPUDevice, T>;\\\n\
  REGISTER_KERNEL_BUILDER(\\\n\
      Name("Matrix{{ process }}").Device(DEVICE_GPU).TypeConstraint<T>("T"),\\\n\
      MatrixOp<GPUDevice, T>);\n\
REGISTER_GPU(COMPLEX_TYPE);\n\
#endif\n\
'

gpuOpTemplate = "\
template <typename T>\n\
void MatrixFunctor<GPUDevice, T>::operator()(\n\
    const GPUDevice& d,\
{% for i in range(func.argn - 1) %}\
{{ func.args[i].type }} {{ func.args[i].name }}, \
{% endfor %}\
{{ func.args[func.argn - 1].type }} {{ func.args[func.argn - 1].name }}\
) {\n\
    // Launch the cuda kernel.\n\
    //\n\
    // See core/util/gpu_kernel_helper.h for example of computing\n\
    // block count and thread_per_block count.\n\
    \n\
    int eventsPerBlock = 1;\n\
    \n\
    int blockSize = DEFAULT_BLOCK_SIZE;\n\
    int numBlocks = (nevents + blockSize - 1) / (eventsPerBlock * blockSize);\n\
    \n\
    if (nevents < blockSize) {\n\
      numBlocks = 1;\n\
      blockSize = nevents;\n\
    }\n\
    \n\
    \n\
    {{ func.name }}<T><<<numBlocks, blockSize, 0, d.stream()>>>({% for i in range(func.argn - 2) %}\
{{ func.args[i].name }}, \
{% endfor %}\
{{ func.args[func.argn - 2].name }}\
);\n\
    \n\
}\n\
\n\
// Explicitly instantiate functors for the types of OpKernels registered.\n\
template struct MatrixFunctor<GPUDevice, COMPLEX_TYPE>;"


# --------------------------------

gpuArithmeticOperators = (
    "__device__ COMPLEX_TYPE cconj(COMPLEX_TYPE a) {\n\
    return COMPLEX_TYPE(a.real(), -a.imag());\n\
}\n\
\n\
__device__ COMPLEX_TYPE operator+(const COMPLEX_TYPE& a, const COMPLEX_TYPE& b) {\n\
    return COMPLEX_TYPE(a.real() + b.real(), a.imag() + b.imag());\n\
}\n\
\n\
__device__ COMPLEX_TYPE operator-(const COMPLEX_TYPE& a, const COMPLEX_TYPE& b) {\n\
    return COMPLEX_TYPE(a.real() - b.real(), a.imag() - b.imag());\n\
}\n\
\n\
__device__ COMPLEX_TYPE operator*(const COMPLEX_TYPE& a, const COMPLEX_TYPE& b) {\n\
    return COMPLEX_TYPE(a.real() * b.real() - a.imag() * b.imag(), a.imag() * b.real() + a.real() * b.imag());\n\
}\n\
\n\
__device__ COMPLEX_TYPE operator/(const COMPLEX_TYPE& a, const COMPLEX_TYPE& b) {\n\
    "
    + doubleType
    + " norm = b.real() * b.real() + b.imag() * b.imag();\n\
    return COMPLEX_TYPE((a.real() * b.real() + a.imag() * b.imag())/norm, (a.imag() * b.real() - a.real() * b.imag())/norm);\n\
}\n\
\n\
__device__ COMPLEX_TYPE operator-(const COMPLEX_TYPE& a) {\n\
    return COMPLEX_TYPE(-a.real(), -a.imag());\n\
}\n\
\n\
__device__ COMPLEX_TYPE operator*(const COMPLEX_TYPE& a, const "
    + doubleType
    + "& b) {\n\
    return COMPLEX_TYPE(a.real() * b, a.imag() * b);\n\
}\n\
\n\
__device__ COMPLEX_TYPE operator*(const "
    + doubleType
    + "& a, const COMPLEX_TYPE& b) {\n\
    return b * a;\n\
}\n\
\n\
__device__ COMPLEX_TYPE operator/(const COMPLEX_TYPE& a, const "
    + doubleType
    + "& b) {\n\
    return COMPLEX_TYPE(a.real() / b, a.imag() / b);\n\
}\n"
)

# --------------------------------
