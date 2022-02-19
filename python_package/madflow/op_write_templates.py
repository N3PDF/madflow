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
    temp += template_with_string(libraryTemplate, lib)
    return temp


def write_headers(temp, head):
    temp += template_with_string(headerTemplate, head)
    return temp


def write_namespaces(temp, name):
    temp += "using namespace " + name + ";\n"
    return temp


def write_constants(temp, constVar, device):

    dev = ""
    if device == "gpu":
        dev = "__device__ "

    templateString = constantVariableTemplate

    template = Template(templateString)
    temp += template.render(constantVariable=constVar, dev=dev)
    return temp


def write_defined(temp, constants_, device):

    constants = []  # GPU constants are different from CPU constants
    for l in constants_:
        constants.append(l)
    if device == "gpu":
        for i in range(len(constants)):
            constants[i] = re.sub("std::", "", constants[i])
            constants[i] = re.sub("conj", "cconj", constants[i])

    temp += template_with_string(definedConstantTemplate, constants)
    return temp


def write_function_definition(temp, func, device):

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
    temp += "\n"
    temp += template.render(func=func, dev=dev)
    return temp


def write_function(temp, fun, device):

    dev = ""

    if fun.name == "matrix":
        func = function(fun.type, fun.name, fun.args, [], fun.scope_args, fun.template)
        for i in range(len(fun.scope)):
            func.scope.append(fun.scope[i])
        if device == "cpu":
            func = parallelize_function(func, "ThreadPool")
        else:
            del func.args[-1]
            func = parallelize_function(func, "CUDA")
            dev = "__global__ "
    else:
        func = fun
        if device == "gpu":
            dev = "__device__ "
    """
    if device == "gpu":
        if func.name == "matrix":
            dev = "__global__ "
        else:
            dev = "__device__ "
    """
    func.argn = len(func.args)
    templateString = functionTemplate

    template = Template(templateString)
    temp += "\n"
    temp += template.render(func=func, dev=dev)

    return temp


def write_header_file(temp, custom_op, func):
    func2 = func
    op_types = []
    for i in range(len(func.args)):
        t = re.sub("const (.*)", "\g<1>", func.args[i].type)
        t = re.sub("([^&*]*)[&*]*", "\g<1>", t)
        op_types.append(t)
    func.argn = len(func.args)

    templateString = headerFileTemplate
    template = Template(templateString)
    temp += "\n"
    temp += template.render(custom_op=custom_op, func=func, op_types=op_types)
    return temp


def write_matrix_op(temp, custom_op, func, device, process_name):
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
    temp += "\n"
    temp += template.render(custom_op=custom_op, func=func, op_types=op_types, process=p)
    return temp


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

    customOpCode = write_headers(customOpCode, headers)
    customOpCode = write_namespaces(customOpCode, namespace)
    customOpCode = write_defined(customOpCode, defined, device)

    customOpCode = write_constants(customOpCode, constants, device)

    if device == "cpu":
        customOpCode = write_constants(customOpCode, cpuConstants, device)

    """
    if device == "gpu":
        del function_list[-1].args[-1]

        i = 0
        while i < len(function_list[-1].scope):
            if function_list[-1].scope[i].startswith("auto thread_pool"):
                while (
                    i < len(function_list[-1].scope)
                    and function_list[-1].scope[i].startswith("for (auto it") == False
                ):
                    del function_list[-1].scope[i]
                function_list[-1].scope[i] = (
                    "for (int it = blockIdx.x * blockDim.x + threadIdx.x; it < "
                    + function_list[-1].args[-1].name
                    + "; it += blockDim.x * gridDim.x) {"
                )
            elif function_list[-1].scope[i] == "};":
                del function_list[-1].scope[i]
                del function_list[-1].scope[i]
                break
            i += 1
    """
    for f in function_list:
        customOpCode = write_function_definition(customOpCode, f, device)

    if device == "gpu":
        customOpCode += "\n"
        customOpCode += gpuArithmeticOperators

    for f in function_list:
        customOpCode += "\n"
        customOpCode = write_function(customOpCode, f, device)

    if device == "gpu":
        function_list[-1].args.append(argument("context", "const OpKernelContext*", 0, False, []))

    for c in custom_op_list:
        customOpCode = write_matrix_op(customOpCode, c, function_list[-1], device, process_name)

    if device == "gpu":
        customOpCode = re.sub("([ ,+\-*/]+)sign([ (;]+)", "\g<1>signn\g<2>", customOpCode)
        customOpCode = re.sub("([ ,+\-*/]+)signvec([ (;]+)", "\g<1>signvecc\g<2>", customOpCode)

        customOpCode += "\n#endif\n"

    with open(destination + "gpu/matrix_" + process_name + extension, "w") as fh:
        fh.write(customOpCode)
