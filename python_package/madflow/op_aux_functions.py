from madflow.op_global_constants import *
from madflow.op_classes import *


def generate_auxiliary_functions(auxiliary_functions, function_list_):
    aux_args = []
    aux_arg = argument("x", doubleType, 0, False, [])
    aux_args.append(aux_arg)
    aux_arg = argument("y", doubleType, 0, False, [])
    aux_args.append(aux_arg)
    aux_scope = ["int sign = 0;", "y >= 0 ? sign = 1 : sign = -1;", "return x * sign;"]
    aux_scope_args = [argument("sign", "int", 0, False, [])]
    aux_function = function(doubleType, "sign", aux_args, aux_scope, aux_scope_args, "")
    function_list_.append(aux_function)
    auxiliary_functions.append(aux_function)

    aux_scope = ["return sign(x, y);"]
    aux_scope_args = []
    aux_function = function(doubleType, "signvec", aux_args, aux_scope, aux_scope_args, "")
    function_list_.append(aux_function)
    auxiliary_functions.append(aux_function)
    return auxiliary_functions, function_list_
