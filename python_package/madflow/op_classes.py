class function:
    """function
    type: void, int, double, ...
    name: function_name
    args: function arguments (list of argument objects)
    scope: function scope (list of strings, one for each line)
    scope_args: variables defined in the scope (list of argument objects)
    template: i.e. template <typename T>"""

    def __init__(self, f_type, f_name, f_args, f_scope, f_scope_args, f_template):
        self.type = f_type
        self.name = f_name
        self.args = f_args
        self.scope = f_scope
        self.scope_args = f_scope_args
        self.argn = len(f_args)
        self.template = f_template


class custom_operator:
    """Custom Operator
    name: op_name
    scope: op scope (list of strings, one for each line)
    functor_name: name of the functor (called by MatrixOp)"""

    def __init__(self, f_name, f_scope, f_functor_name):
        self.name = f_name
        self.scope = f_scope
        self.functor_name = f_functor_name


class argument:
    """variable
    type: void, int, double, ...
    name: variable_name
    size: size of the array
          0 => int x;
          6 => int x[6];
    tensor: true if the variable is a TensorFlow tensor
    slice: if tensor == true, length of a single tensor slice"""

    def __init__(self, a_name, a_type, a_size, a_tensor, a_slice):
        self.name = a_name
        self.type = a_type
        self.size = a_size
        self.tensor = a_tensor
        self.slice = a_slice


class signature:
    """signature
    name: signature_name
    type: complex, int, double, ...
    size: tensor shape
    tensor: true if the variable is a TensorFlow tensor
    slice: if tensor == true, length of a single tensor slice"""

    def __init__(self, a_name, a_type, a_size, a_tensor, a_slice):
        self.name = a_name
        self.type = a_type
        self.size = a_size
        self.tensor = a_tensor
        self.slice = a_slice


class signature_variable:
    """tf.function signature
    name: signature_name
    signature_name_list: list of strings containing signature.name
    signature_list: list of signature objects"""

    def __init__(self, a_name, signature_list, signature_name_list):
        self.name = a_name
        self.signature_name_list = signature_name_list
        self.signature_list = signature_list
