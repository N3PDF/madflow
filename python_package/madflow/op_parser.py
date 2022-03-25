"""Extraction of features from python code (function scope / signature)"""

import re

import madflow.op_aux_functions as op_af
import madflow.op_classes as op_cl
import madflow.op_transpiler as op_tp


# Function parsing


def parse_function_return(function_return, scope, scope_args, args):
    """Parse the return line
    function_return: a string containing the return line
    scope: list of strings containing the function scope
    scope_args: variables defined within function scope
    args: function arguments

    return: updated scope and scope variables"""

    function_return = re.sub("return", args[-1].name + " =", function_return)
    function_return = re.sub("#[^\n]*\n", "", function_return)
    inside_comment = False
    new_line, scope_args, scope, inside_comment = op_tp.parse_line(
        function_return, args, scope_args, scope, inside_comment
    )

    scope.append(new_line)

    return scope, scope_args


def parse_function_scope(function_scope, scope, scope_args, args):
    """Parse function scope
    function_scope: a string containing the return line
    scope: list of strings containing the function scope
    scope_args: variables defined within function scope
    args: function arguments

    return: updated scope and scope variables"""
    if len(function_scope) == 0:
        # empty scope
        return scope, scope_args
    # read the function scope

    i = 0
    inside_comment = False
    while i < len(function_scope):
        line = function_scope[i]  # read i-th line
        brackets_count = 0
        brackets_count = op_af.count_brackets(line, brackets_count)
        # if there are more '(' than ')', read more lines
        # (but don't go beyond len(function_scope)
        while brackets_count > 0 and i < len(function_scope) - 1:
            i += 1
            l = function_scope[i]
            brackets_count = op_af.count_brackets(l, brackets_count)
            line += l  # create a single line
        # parse (and transpile) the line
        # read also any variables defined in the scope
        # append those variables to scope_args
        new_line, scope_args, scope, inside_comment = op_tp.parse_line(
            line, args, scope_args, scope, inside_comment
        )
        scope.append(new_line)  # add the line to the function scope
        i += 1

    return scope, scope_args


def get_signature(line):
    """Read the signature from text
    line: line of text containing the signature

    return: a Signature object"""
    type_ = line.split("dtype=")[1]
    type_ = type_.split(")")[0]
    type_ = op_af.convert_type(type_)
    is_tensor = False

    shape = line.split("shape=[")[1]
    shape = shape.split("]")[0]
    slice_ = []

    if shape == "":
        shape = 0
    elif shape == "None":
        shape = 0
        is_tensor = True
    else:
        s = shape.split(",", 1)[-1]
        shape = op_af.clean_spaces(shape.split(",")[-1])
        s = s.split(",")
        prod = 1
        for a in s:
            slice_.append(a)
            if a != "None":
                prod *= int(a)
        slice_.append(str(prod))

    name = op_af.clean_spaces(line.split("=")[0])

    return op_cl.Signature(name, type_, shape, is_tensor, slice_)


def convert_signatures(signatures, signature_variables):
    """Read the signature from text and update signature_variables
    signatures: list of text Signature objects
    signature_variables: list of tf.function signatures"""
    for i in range(len(signature_variables)):
        for v in signature_variables[i].signature_name_list:
            for s in signatures:
                if s.name == v:
                    signature_variables[i].signature_list.append(s)
