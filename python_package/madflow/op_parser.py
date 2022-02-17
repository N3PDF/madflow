from madflow.op_transpiler import *


# Function parsing


def parse_function_return(function_return, scope, scope_args, args, f_type):

    function_return = re.sub("return", args[-1].name + " =", function_return)
    function_return = re.sub("#[^\n]*\n", "", function_return)
    inside_comment = False
    new_line, scope_args, scope, inside_comment = parse_line(
        function_return, args, scope_args, scope, inside_comment
    )

    scope.append(new_line)

    return scope, scope_args


def parse_function_scope(function_scope, scope, scope_args, args, f_type):
    if len(function_scope) == 0:
        return scope, scope_args
    i = 0
    line = function_scope[i]
    i += 1
    inside_comment = False
    while i <= len(function_scope):
        brackets_count = 0
        brackets_count = count_brackets(line, brackets_count)
        while brackets_count > 0 and i < len(function_scope):
            l = function_scope[i]
            i += 1
            brackets_count = count_brackets(l, brackets_count)
            line += l
        new_line, scope_args, scope, inside_comment = parse_line(
            line, args, scope_args, scope, inside_comment
        )
        scope.append(new_line)
        if i < len(function_scope):
            line = function_scope[i]
        i += 1
    return scope, scope_args


def get_signature(line):
    type_ = line.split("dtype=")[1]
    type_ = type_.split(")")[0]
    type_ = convert_type(type_)
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
        shape = clean_spaces(shape.split(",")[-1])
        s = s.split(",")
        prod = 1
        for a in s:
            slice_.append(a)
            if a != "None":
                prod *= int(a)
        slice_.append(str(prod))

    name = clean_spaces(line.split("=")[0])

    return signature(name, type_, shape, is_tensor, slice_)


def convert_signatures(signatures, signature_variables):
    for i in range(len(signature_variables)):
        for v in signature_variables[i].signature_name_list:
            for s in signatures:
                if s.name == v:
                    signature_variables[i].signature_list.append(s)
    return signature_variables
