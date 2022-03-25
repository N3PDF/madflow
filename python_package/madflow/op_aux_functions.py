"""Sign functions used by the Custom Operator and functions used by the transpiler"""

import re

import madflow.op_global_constants as op_gc
import madflow.op_classes as op_cl


def generate_auxiliary_functions(function_list_):
    """generates sign functions

    function_list_: list of functions used by the Custom Operator (updated)"""
    aux_args = []
    aux_arg = op_cl.Argument("x", op_gc.DOUBLE_TYPE, 0, False, [])
    aux_args.append(aux_arg)
    aux_arg = op_cl.Argument("y", op_gc.DOUBLE_TYPE, 0, False, [])
    aux_args.append(aux_arg)
    aux_scope = ["int sign = 0;", "y >= 0 ? sign = 1 : sign = -1;", "return x * sign;"]
    aux_scope_args = [op_cl.Argument("sign", "int", 0, False, [])]
    aux_function = op_cl.Function(
        op_gc.DOUBLE_TYPE, "sign", aux_args, aux_scope, aux_scope_args, ""
    )
    function_list_.append(aux_function)

    aux_scope = ["return sign(x, y);"]
    aux_scope_args = []
    aux_function = op_cl.Function(
        op_gc.DOUBLE_TYPE, "signvec", aux_args, aux_scope, aux_scope_args, ""
    )
    function_list_.append(aux_function)


# Support functions


def clean_spaces(a):
    """remove spaces or endlines within a string"""
    return a.translate({ord(c): None for c in "\n "})


def clean_index(a):
    """return the index of an array element"""
    return a.split("[")[0]


def clean_pointer(var_type):
    """remove * from a variable type"""
    var_type = re.sub("[&*]*", "", var_type)
    return var_type


def count_brackets(line, brackets_count):
    """remove the count of brackets () in a string"""
    for letter in line:
        brackets_count = count_brackets_letter(letter, brackets_count)
    return brackets_count


def convert_grammar(old_value):
    """converts the grammar from Python to C++"""
    value = re.sub("tf.reshape", "", old_value)
    value = re.sub("\[:,[ :]*(\d+)\]", "[\g<1>]", value)
    value = re.sub("float_me\(([a-zA-Z0-9[\]+\-*/. ]*)\)", "\g<1>", value)
    value = re.sub("int_me", "(int)", value)
    value = re.sub("([a-zA-Z_0-9[\]]+) *\*\* *2", "\g<1> * \g<1>", value)
    value = re.sub("([a-zA-Z_0-9[\]]+) \*\* (\d+)", "pow(\g<1>, \g<2>)", value)
    value = re.sub("\( *([a-zA-Z_0-9[\]+\-*/ ]+)\) *// *2", "(int)(\g<1>) / 2", value)
    value = re.sub("tf.ones_like\([a-zA-Z_0-9[\]{}+\-*/=, \n]*\) *\**", "", value)
    value = re.sub("tfmath\.", "", value)
    value = re.sub("minimum", "MINIMUM", value)  # hhh
    value = re.sub("maximum", "MAXIMUM", value)  # hhh
    value = re.sub("tf.math.real\(([a-zA-Z0-9_()[\] +\-*/]*)\)", "\g<1>.real()", value)
    value = re.sub("tf.math.imag\(([a-zA-Z0-9_()[\] +\-*/]*)\)", "\g<1>.imag()", value)
    value = re.sub("tf.math.conj", "COMPLEX_CONJUGATE", value)  # hhh
    value = re.sub(
        "tf.stack\([ \n]*\[([a-zA-Z_0-9()[\]+\-*/,. ]*)] *, *axis=[0-9 \n]*\)", "{\g<1>}", value
    )
    value = re.sub("tf.stack\([ \n]*\[([a-zA-Z_0-9()[\]+\-*/,. ]*)][ \n]*\)", "{\g<1>}", value)
    value = re.sub("tf.stack\([ \n]*([a-zA-Z_0-9()[\]+\-*/,. ]*) *, *axis=[0-9 \n]*\)", "", value)
    value = re.sub("\(*tf.stack\([ \n]*([a-zA-Z_0-9()[\]+\-*/,. ]*) *, *\[[0-9, \n]*]\)", "", value)
    value = re.sub("complex_tf", "T", value)
    value = re.sub("complex_me", "T", value)
    value = re.sub("complex\(", "T(", value)
    value = re.sub("\( *\(([a-zA-Z_0-9()[\]{}+\-*/ \n]*)\) *\)", "(\g<1>)", value)
    return value


def convert_type(t):
    """converts TensorFlow types into C++ types"""
    t = clean_spaces(t)

    result = ""
    d = {
        "DTYPE": op_gc.DOUBLE_TYPE,
        "DTYPEINT": "int",
        "DTYPECOMPLEX": "T",
    }
    result = d.get(t, t)
    return result


def change_array_into_variable(line):
    """specific to denom
    chenges denom from denom[i] into denom"""
    match = re.search("denom", line)
    if match != None:
        line = re.sub("\[\]", "", line)
        line = re.sub("{([+\-0-9]+).*;", "\g<1>", line)
        return line
    else:
        line = re.sub(";", "", line)
        return line


def count_brackets_letter(letter, bracket_count):
    if letter == "(":
        bracket_count += 1
    elif letter == ")":
        bracket_count -= 1
    return bracket_count
