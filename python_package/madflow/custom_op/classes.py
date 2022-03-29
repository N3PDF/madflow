"""Classes used by the transpiler"""

from dataclasses import dataclass


@dataclass
class Argument:
    """variable
    type: void, int, double, ...
    name: variable_name
    size: size of the array
          0 => int x;
          6 => int x[6];
    tensor: true if the variable is a TensorFlow tensor
    slice: if tensor == true, length of a single tensor slice"""

    name: str
    type: str
    size: int
    tensor: bool
    slice: list  # [str]


@dataclass
class Function:
    """function
    type: void, int, double, ...
    name: function_name
    args: function arguments (list of argument objects)
    scope: function scope (list of strings, one for each line)
    scope_args: variables defined in the scope (list of argument objects)
    template: i.e. template <typename T>"""

    type: str
    name: str
    args: list  # [Argument]
    scope: list  # [str]
    scope_args: list  # [str]
    template: str
    argn: int = 0

    def __post_init__(self):
        self.argn = len(self.args)


@dataclass
class CustomOperator:
    """Custom Operator
    name: op_name
    scope: op scope (list of strings, one for each line)
    functor_name: name of the functor (called by MatrixOp)"""

    name: str
    scope: list  # [str]
    functor_name: str


@dataclass
class Signature:
    """signature
    name: signature_name
    type: complex, int, double, ...
    size: tensor shape
    tensor: true if the variable is a TensorFlow tensor
    slice: if tensor == true, length of a single tensor slice"""

    name: str
    type: str
    size: str
    tensor: bool
    slice: list  # [str]


@dataclass
class SignatureVariable:
    """tf.function signature
    name: signature_name
    signature_list: list of Signature objects
    signature_name_list: list of strings containing signature.name"""

    name: str
    signature_list: list  # [Signature]
    signature_name_list: list  # [str]
