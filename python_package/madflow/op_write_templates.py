from jinja2 import Template
import re

from madflow.op_constants import *
from madflow.op_classes import *
from madflow.op_generation import *


def template_with_string(template_string, variable):

    template = Template(template_string)
    string = template.render(variableName=variable)

    return string


def write_libraries(temp, lib):
    """Writes libraries (#include <...>)"""
    temp += template_with_string(library_template, lib)
    return temp


def write_headers(head):
    """Writes external header files (#include "...")"""
    return template_with_string(header_template, head)


def write_namespaces(name):
    """Writes namespaces (using namespace ...;)"""
    return "using namespace " + name + ";\n"


def write_constants(const_var, device):
    """Writes constants ((__device__) const ...;)"""

    dev = ""
    if device == "gpu":
        dev = "__device__ "

    template_string = constant_variable_template

    template = Template(template_string)
    return template.render(constantVariable=const_var, dev=dev)


def write_defined(constants_, device):
    """Writes constants (#define ...)"""

    constants = []  # GPU constants are different from CPU constants
    for l in constants_:
        constants.append(l)
    if device == "gpu":
        for i in range(len(constants)):
            constants[i] = re.sub("std::", "", constants[i])
            constants[i] = re.sub("conj", "cconj", constants[i])

    return template_with_string(defined_constant_template, constants)


def write_function_definition(func, device):
    """Writes function definitions
    (void function(..., ..., ...);)"""

    dev = ""
    if device == "gpu":
        if func.name == "matrix":
            dev = "__global__ "
        else:
            dev = "__device__ "

    func.argn = len(func.args)
    if func.args[0].type == double_type + "*":
        func.args[0].type = "const " + func.args[0].type

    template_string = function_definition_template

    template = Template(template_string)
    return "\n" + template.render(func=func, dev=dev)


def write_function(fun, device):
    """Writes function implementations
    (void function(..., ..., ...) {
        ...
    })"""

    dev = ""

    if fun.name == "matrix":
        func = Function(fun.type, fun.name, fun.args, [], fun.scope_args, fun.template)
        for i in range(len(fun.scope)):
            func.scope.append(fun.scope[i])
        if device == "cpu":
            func = parallelize_function(func, cpu_parallelization)
        else:
            del func.args[-1]
            func = parallelize_function(func, gpu_parallelization)
            dev = "__global__ "
    else:
        func = fun
        if device == "gpu":
            dev = "__device__ "

    func.argn = len(func.args)
    template_string = function_template

    template = Template(template_string)
    return "\n" + template.render(func=func, dev=dev)


def write_header_file(custom_op, func):
    """Writes matrix_xxxxx.h"""
    # func2 = func
    op_types = []
    for i in range(len(func.args)):
        t = re.sub("const (.*)", "\g<1>", func.args[i].type)
        t = re.sub("([^&*]*)[&*]*", "\g<1>", t)
        op_types.append(t)
    func.argn = len(func.args)

    template_string = header_file_template
    template = Template(template_string)
    return "\n" + template.render(custom_op=custom_op, func=func, op_types=op_types)


def write_matrix_op(custom_op, func, device, process_name):
    # func2 = func
    op_types = []
    for i in range(len(func.args)):
        t = re.sub("const (.*)", "\g<1>", func.args[i].type)
        t = re.sub("([^&*]*)[&*]*", "\g<1>", t)
        op_types.append(t)
    func.argn = len(func.args)
    p = re.sub("_", "", process_name)

    if device == "cpu":
        template_string = cpu_op_template
    elif device == "gpu":
        template_string = gpu_op_template

    template = Template(template_string)
    return "\n" + template.render(custom_op=custom_op, func=func, op_types=op_types, process=p)


def write_custom_op(
    headers,
    namespace,
    defined,
    constants,
    cpuConstants,
    function_list,
    custom_op_list,
    destination,
    process_name,
    device,
):
    """Writes the Custom Operator:
    - heades and libraries
    - namespaces
    - global constants
    - function definitions
    - function implementations
    - "wrapper function" called by matrix_1_xxxxx.py"""

    extension = ""
    custom_op_code = ""
    if device == "cpu":
        extension = ".cc"
    elif device == "gpu":
        extension = ".cu.cc"

        custom_op_code += (
            "#ifdef GOOGLE_CUDA\n\
"
            "#define EIGEN_USE_GPU\n"
        )
    else:
        return

    custom_op_code += write_headers(headers)
    custom_op_code += write_namespaces(namespace)
    custom_op_code += write_defined(defined, device)
    custom_op_code += write_constants(constants, device)

    if device == "cpu":  # write 'using thread::ThreadPool' if using ThreadPool
        if cpu_parallelization == "ThreadPool":
            custom_op_code += write_constants(cpu_constants, device)

    for f in function_list:
        custom_op_code += write_function_definition(f, device)

    if device == "gpu":
        custom_op_code += "\n"
        custom_op_code += gpu_arithmetic_operators

    for f in function_list:
        custom_op_code += "\n"
        custom_op_code += write_function(f, device)

    if device == "gpu":
        function_list[-1].args.append(Argument("context", "const OpKernelContext*", 0, False, []))

    for c in custom_op_list:
        custom_op_code += write_matrix_op(c, function_list[-1], device, process_name)

    if device == "gpu":
        custom_op_code = re.sub("([ ,+\-*/]+)sign([ (;]+)", "\g<1>signn\g<2>", custom_op_code)
        custom_op_code = re.sub("([ ,+\-*/]+)signvec([ (;]+)", "\g<1>signvecc\g<2>", custom_op_code)

        custom_op_code += "\n#endif\n"

    # with open(destination / ("gpu/matrix_" + process_name + extension), "w") as fh:
    #    fh.write(custom_op_code)
    (destination / ("matrix_" + process_name + extension)).write_text(custom_op_code)
