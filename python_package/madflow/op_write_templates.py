from jinja2 import Template
import re

from madflow.op_constants import *
from madflow.op_classes import *
from madflow.op_generation import *


def template_with_string(templateString, variable):

    template = Template(templateString)
    string = template.render(variableName=variable)

    return string


def write_libraries(temp, lib):
    """Writes libraries (#include <...>)"""
    temp += template_with_string(libraryTemplate, lib)
    return temp


def write_headers(head):
    """Writes external header files (#include "...")"""
    return template_with_string(headerTemplate, head)


def write_namespaces(name):
    """Writes namespaces (using namespace ...;)"""
    return "using namespace " + name + ";\n"


def write_constants(constVar, device):
    """Writes constants ((__device__) const ...;)"""

    dev = ""
    if device == "gpu":
        dev = "__device__ "

    templateString = constantVariableTemplate

    template = Template(templateString)
    return template.render(constantVariable=constVar, dev=dev)


def write_defined(constants_, device):
    """Writes constants (#define ...)"""

    constants = []  # GPU constants are different from CPU constants
    for l in constants_:
        constants.append(l)
    if device == "gpu":
        for i in range(len(constants)):
            constants[i] = re.sub("std::", "", constants[i])
            constants[i] = re.sub("conj", "cconj", constants[i])

    return template_with_string(definedConstantTemplate, constants)


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
    if func.args[0].type == doubleType + "*":
        func.args[0].type = "const " + func.args[0].type

    templateString = functionDefinitionTemplate

    template = Template(templateString)
    return "\n" + template.render(func=func, dev=dev)


def write_function(fun, device):
    """Writes function implementations
    (void function(..., ..., ...) {
        ...
    })"""

    dev = ""

    if fun.name == "matrix":
        func = function(fun.type, fun.name, fun.args, [], fun.scope_args, fun.template)
        for i in range(len(fun.scope)):
            func.scope.append(fun.scope[i])
        if device == "cpu":
            func = parallelize_function(func, cpuParallelization)
        else:
            del func.args[-1]
            func = parallelize_function(func, gpuParallelization)
            dev = "__global__ "
    else:
        func = fun
        if device == "gpu":
            dev = "__device__ "

    func.argn = len(func.args)
    templateString = functionTemplate

    template = Template(templateString)
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

    templateString = headerFileTemplate
    template = Template(templateString)
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
        templateString = cpuOpTemplate
    elif device == "gpu":
        templateString = gpuOpTemplate

    template = Template(templateString)
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
    customOpCode = ""
    if device == "cpu":
        extension = ".cc"
    elif device == "gpu":
        extension = ".cu.cc"

        customOpCode += (
            "#ifdef GOOGLE_CUDA\n\
"
            "#define EIGEN_USE_GPU\n"
        )
    else:
        return

    customOpCode += write_headers(headers)
    customOpCode += write_namespaces(namespace)
    customOpCode += write_defined(defined, device)
    customOpCode += write_constants(constants, device)

    if device == "cpu":  # write 'using thread::ThreadPool' if using ThreadPool
        if cpuParallelization == "ThreadPool":
            customOpCode += write_constants(cpuConstants, device)

    for f in function_list:
        customOpCode += write_function_definition(f, device)

    if device == "gpu":
        customOpCode += "\n"
        customOpCode += gpuArithmeticOperators

    for f in function_list:
        customOpCode += "\n"
        customOpCode += write_function(f, device)

    if device == "gpu":
        function_list[-1].args.append(argument("context", "const OpKernelContext*", 0, False, []))

    for c in custom_op_list:
        customOpCode += write_matrix_op(c, function_list[-1], device, process_name)

    if device == "gpu":
        customOpCode = re.sub("([ ,+\-*/]+)sign([ (;]+)", "\g<1>signn\g<2>", customOpCode)
        customOpCode = re.sub("([ ,+\-*/]+)signvec([ (;]+)", "\g<1>signvecc\g<2>", customOpCode)

        customOpCode += "\n#endif\n"

    with open(destination + "gpu/matrix_" + process_name + extension, "w") as fh:
        fh.write(customOpCode)
