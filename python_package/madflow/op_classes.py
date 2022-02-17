class function:
    def __init__(self, f_type, f_name, f_args, f_scope, f_scope_args, f_template):
        self.type = f_type
        self.name = f_name
        self.args = f_args
        self.scope = f_scope
        self.scope_args = f_scope_args
        self.argn = len(f_args)
        self.template = f_template


class custom_operator:
    def __init__(self, f_name, f_scope, f_functor_name):
        self.name = f_name
        self.scope = f_scope
        self.functor_name = f_functor_name


class argument:
    def __init__(self, a_name, a_type, a_size, a_tensor, a_slice):
        self.name = a_name
        self.type = a_type
        self.size = a_size
        self.tensor = a_tensor
        self.slice = a_slice


class signature:
    def __init__(self, a_name, a_type, a_size, a_tensor, a_slice):
        self.name = a_name
        self.type = a_type
        self.size = a_size
        self.tensor = a_tensor
        self.slice = a_slice


class signature_variable:
    def __init__(self, a_name, signature_list, signature_name_list):
        self.name = a_name
        self.signature_name_list = signature_name_list
        self.signature_list = signature_list
