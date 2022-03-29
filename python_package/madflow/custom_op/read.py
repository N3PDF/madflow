"""Reads functions and signatures line by line"""

import re

import madflow.custom_op.aux_functions as op_af
import madflow.custom_op.global_constants as op_gc
import madflow.custom_op.classes as op_cl
import madflow.custom_op.parser as op_pa


def grab_function_name(line):
    """Read function name
    line: a line of text that defines the function.
          i.e.: def function_name(args):

    return: function name"""
    return line.split("(", 1)[0]


def grab_function_arguments(line, f_name, signature_variables, signature_line):
    """Read function arguments
    line: a line of text that defines the function.
          i.e.: def function_name(args):
    f_name: function name
    signature_variables: list of Signature objects containing
                         any previously defined signature
    signature_line: a line of text defining function signature.
                    (@tf.function ...)

    return: a list of argument objects containing all function arguments"""
    line = line.split(")", 1)[0]
    line = line[len(f_name) + 1 :]
    # create a list of names of function arguments
    split_args = op_af.clean_spaces(line).split(",")

    # delete self if the function is a class method
    j = -1
    for i in range(len(split_args)):
        if split_args[i] == "self":
            j = i
            break
    if j != -1:
        del split_args[j]

    args = []
    split_types = []
    split_sizes = []
    split_tensors = []
    split_slices = []
    sig_list = []
    signature_line = signature_line.split("@tf.function(")[1]
    signature_line = signature_line.split(")")[0]
    signature_line = op_af.clean_spaces(signature_line).split("input_signature=")[1]
    if signature_line.startswith("["):
        s = op_pa.get_signature(signature_line)
        sig_list.append(s)
    else:
        for sv in signature_variables:
            if sv.name == signature_line:
                if len(sv.signature_list) == len(split_args):
                    sig_list = sv.signature_list

    for a in sig_list:
        t = a.type
        if a.size != 0:
            t += "*"
        split_types.append(t)
        split_sizes.append(a.size)
        split_tensors.append(a.tensor)
        split_slices.append(a.slice)

    for i in range(len(split_args)):
        split_args[i] = op_af.clean_spaces(split_args[i])
        args.append(
            op_cl.Argument(
                split_args[i], split_types[i], split_sizes[i], split_tensors[i], split_slices[i]
            )
        )

    return args


def grab_function_return(line, f_name, args):
    """Read function return value and type
    line: a line of text containing the return line
    f_name: function name
    args: list of argument objects containing
          any previously defined variable.

    return: an updated list of argument objects containing
            the return variable"""

    # Currently all functions are void
    f_type = "void"
    # The return value is passed by pointer
    args.append(op_cl.Argument("ret", op_gc.DOUBLE_TYPE, -1, False, []))
    return args, f_type


def grab_function_scope(f, args):
    """Read function scope
    f: file stream
    args: list of argument objects containing
          function arguments.

    return: an updated list of strings containing
            the function scope and scope variables"""

    line = f.readline()
    scope = []
    scope_args = []
    function_scope = []
    function_return = ""

    match = re.search("^ *return[ ]+", line)
    while op_af.clean_spaces(line).startswith("return") == False:
        function_scope.append(line)
        match = re.search("^ *return[ ]+", line)
        line = f.readline()
    while op_af.clean_spaces(line) != "":
        function_return += line
        line = f.readline()

    args[-1].name = grab_return_variable_name(function_return)
    scope, scope_args = op_pa.parse_function_scope(function_scope, scope, scope_args, args)
    scope, scope_args = op_pa.parse_function_return(function_return, scope, scope_args, args)

    return scope, scope_args


def grab_return_variable_name(function_return):
    """Read the name of the return variable, if defined
    otherwise, default to 'out_final'
    function_return: a line of text containing the return line

    return: string representing the name of the return variable"""
    ret_name = "out_final"
    function_return = op_af.clean_spaces(function_return)[len("return") :]
    st1 = "tf.stack("
    st2 = "tf.transpose("
    st3 = "tf.reshape(tf.stack("
    if function_return.startswith(st1):
        ret_name = function_return[len(st1) :].split(")")[0].split(",")[0]
    elif function_return.startswith(st2):
        ret_name = function_return[len(st2) :].split(")")[0].split(",")[0]
    elif function_return.startswith(st3):
        ret_name = function_return[len(st3) :].split(")")[0].split(",")[0]
    return ret_name


# Read from file


def read_file_from_source(function_list, file_source, signatures, signature_variables):
    """Read a file, looking for functions
    function_list: list of function objects containing
                   previously defined functions (updated)
    file_source: complete path to the file we need to read
    signatures: defined function signatures
    signature_variables: defined signature variables"""
    f = open(file_source, "r")
    line = f.readline()
    while line != "":
        if op_af.clean_spaces(line).startswith("@tf.function"):
            signature_line = line
            line = f.readline()
            l = line
            i = 0
            while l.endswith("):\n") == False:
                if i != 0:
                    line += l[:-1]
                l = f.readline()
                i += 1
            line = re.sub(" *def ", "", line)  # cut "def " from the line
            f_name = grab_function_name(line)

            already_defined = False
            for func in function_list:
                if f_name == func.name:
                    already_defined = True
                    break

            if already_defined == False:
                f_type = "void"
                args = []
                scope = []
                scope_args = []
                args = grab_function_arguments(line, f_name, signature_variables, signature_line)
                args, f_type = grab_function_return(line, f_name, args)
                scope, scope_args = grab_function_scope(f, args)
                new_function = op_cl.Function(
                    f_type, f_name, args, scope, scope_args, "template <typename T>"
                )
                function_list.append(new_function)

        line = f.readline()


def extract_matrix_from_file(function_list, file_source, signatures, signature_variables):
    """Read the matrix element file, looking for the polarized
       matrix element function
    function_list: list of function objects containing
                   previously defined functions (updated)
    file_source: complete path to the file we need to read
    signatures: defined function signatures
    signature_variables: defined signature variables"""
    f = open(file_source, "r")
    line = f.readline()
    while line != "":
        if op_af.clean_spaces(line).startswith("@tf.function"):
            signature_line = line
            line = f.readline()
            l = line
            i = 0
            while l.endswith("):\n") == False:
                if i != 0:
                    line += l[:-1]
                l = f.readline()
                i += 1
            line = re.sub(" *def ", "", line)  # cut "def " from the line
            f_name = grab_function_name(line)

            already_defined = False
            for func in function_list:
                if f_name == func.name:
                    already_defined = True
                    break

            if already_defined == False and f_name == "matrix":
                f_type = "void"
                args = []
                scope = []
                scope_args = []
                args = grab_function_arguments(line, f_name, signature_variables, signature_line)
                args, f_type = grab_function_return(line, f_name, args)
                scope, scope_args = grab_function_scope(f, args)
                new_function = op_cl.Function(
                    f_type, f_name, args, scope, scope_args, "template <typename T>"
                )
                function_list.append(new_function)

        line = f.readline()


def read_signatures(signatures, signature_variables, file_source):
    """Read signatures from file
    signatures: previously defined signatures (updated)
    signature_variables: previously defined signature variables (updated)
    file_source: complete path to the file we need to read"""
    f = open(file_source, "r")
    line = f.readline()
    while line != "":
        match = re.search("tf.TensorSpec", line)
        match2 = re.search("signature", line)
        if match != None and op_af.clean_spaces(line).startswith("@tf.function") == False:
            s = op_pa.get_signature(line)
            signatures.append(s)
        elif match2 != None and op_af.clean_spaces(line).startswith("@tf.function") == False:
            br_count = 0
            for letter in line:
                if letter == "[":
                    br_count += 1
                elif letter == "]":
                    br_count -= 1
            while br_count > 0:
                l = f.readline()
                for letter in l:
                    if letter == "[":
                        br_count += 1
                    elif letter == "]":
                        br_count -= 1
                line += l
            line = re.sub("(TensorSpec\([^)]*\) *),", "\g<1>?", line)
            name = op_af.clean_spaces(line.split("=")[0])
            line = line.split(" = ")[1]
            var_list = line.split("+")
            if len(var_list) == 1:
                var_list = line.split("?")
            sig_list = []
            s_list = []
            for var in var_list:
                match = re.search("tf.TensorSpec", var)
                if match != None:
                    var = re.sub(".*[\n]*.*(tf.TensorSpec\([^)]*\)).*", "\g<1>", var)
                    s_list.append(
                        op_cl.Signature(
                            var,
                            op_pa.get_signature(var).type,
                            op_pa.get_signature(var).size,
                            op_pa.get_signature(var).tensor,
                            op_pa.get_signature(var).slice,
                        )
                    )
                match = re.search("\[[a-zA-Z0-9_]+] *\*", var)
                sig_name = op_af.clean_spaces(re.sub("\[([a-zA-Z0-9_]+)].*", "\g<1>", var))
                times = 1
                if match != None:
                    times = int(
                        op_af.clean_spaces(re.sub("\[[a-zA-Z0-9_]+] *\* *(\d+)", "\g<1>", var))
                    )
                for i in range(times):
                    sig_list.append(sig_name)

            if len(s_list) > 0:
                s = op_cl.SignatureVariable(name, s_list, [])
                signature_variables.append(s)
            elif len(sig_list) > 0:
                s = op_cl.SignatureVariable(name, [], sig_list)
                signature_variables.append(s)
        line = f.readline()
    f.close()
