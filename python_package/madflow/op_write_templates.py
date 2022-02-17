from jinja2 import Template
import re

from madflow.op_constants import *


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


def write_function(temp, func, device):

    dev = ""
    if device == "gpu":
        if func.name == "matrix":
            dev = "__global__ "
        else:
            dev = "__device__ "

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


def write_custom_op(temp, custom_op, func, device, process_name):
    func2 = func
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


def write_empty_line(temp):
    temp += "\n"
    return temp
