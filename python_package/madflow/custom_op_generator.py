import subprocess

# import re

import madflow.wavefunctions_flow
from madflow.makefile_template import write_makefile

# from madflow.op_constants import * # to be removed
# from madflow.op_global_constants import *
# from madflow.op_aux_functions import *
# from madflow.op_classes import *
from madflow.op_write_templates import *
from madflow.op_syntax import *
from madflow.op_read import *
from madflow.op_generation import *


def clean_args(a):
    return a.translate({ord(c): None for c in "\n "})


folder_name = "prov/"

temp = ""


def translate(destination):

    if destination[-1] != "/":  # Avoid weird behaviours if destination does not end with '/'
        destination += "/"
    file_sources = [madflow.wavefunctions_flow.__file__]  # path to wavefunctions_flow.py

    # Create the directory for the Op source code and create the makefile

    subprocess.check_output(["/bin/sh", "-c", "mkdir -p " + destination + "gpu/"])
    write_makefile(destination)

    auxiliary_functions = []
    function_list_ = []
    auxiliary_functions, function_list_ = generate_auxiliary_functions(
        auxiliary_functions, function_list_
    )

    for file_source in file_sources:
        signatures_ = []
        signature_variables_ = []

        signatures_, signature_variables_ = read_signatures(
            signatures_, signature_variables_, file_source
        )

        signature_variables_ = convert_signatures(signatures_, signature_variables_)

        function_list_ = read_file_from_source(
            function_list_, file_source, signatures_, signature_variables_
        )

    files_list = (
        subprocess.check_output(["/bin/sh", "-c", "ls " + destination + " | grep matrix_1_"])
        .decode("utf-8")
        .split("\n")[:-1]
    )

    for _file_ in files_list:

        constants = []  # globalConstants

        for e in globalConstants:
            constants.append(e)

        process_name = re.sub("matrix_1_", "", _file_)
        process_name = re.sub("\.py", "", process_name)

        matrix_source = destination + "matrix_1_" + process_name + ".py"
        process_source = destination + "aloha_1_" + process_name + ".py"

        _file_ = process_source

        signatures = []
        for s in signatures_:
            signatures.append(s)
        signature_variables = []
        for s in signature_variables_:
            signature_variables.append(s)
        function_list = []
        for f in function_list_:
            function_list.append(f)
        headers = []
        for h in headers_:
            headers.append(h)
        headers.append("matrix_" + process_name + ".h")

        custom_op_list = []

        signatures, signature_variables = read_signatures(signatures, signature_variables, _file_)

        signature_variables = convert_signatures(signatures, signature_variables)

        function_list = read_file_from_source(
            function_list, _file_, signatures, signature_variables
        )

        matrix_name = "matrix_1_" + process_name + ".py"

        signatures, signature_variables = read_signatures(
            signatures, signature_variables, matrix_source
        )
        signature_variables = convert_signatures(signatures, signature_variables)

        function_list = extract_matrix_from_file(
            function_list, matrix_source, signatures, signature_variables
        )

        for i in range(len(function_list)):
            function_list = check_variables(i, function_list)

        for i in range(len(function_list)):
            function_list = check_lines(i, function_list)
        for i in range(len(function_list)):
            function_list = check_variables(i, function_list)

        function_list[-1] = parallelize_function(function_list[-1])

        custom_op_list = define_custom_op(custom_op_list, function_list[-1])

        function_list[-1], constants = extract_constants(function_list[-1], constants)

        temp = ""
        temp = write_headers(temp, headers)
        temp = write_namespaces(temp, namespace)
        temp = write_defined(temp, defined, "cpu")

        temp = write_constants(temp, constants, "cpu")
        temp = write_constants(temp, cpuConstants, "cpu")

        for i in range(len(function_list[-1].scope)):  # This loop can be reversed
            if clean_args(function_list[-1].scope[i]).startswith(function_list[-1].args[-3].name):
                function_list[-1].scope[i] = re.sub(".real\(\)", "", function_list[-1].scope[i])

        for f in function_list:
            temp = write_function_definition(temp, f, "cpu")

        for f in function_list:
            temp += "\n"
            temp = write_function(temp, f, "cpu")

        for c in custom_op_list:
            temp = write_custom_op(temp, c, function_list[-1], "cpu", process_name)

        with open(destination + "gpu/matrix_" + process_name + ".cc", "w") as fh:
            fh.write(temp)

        temp = ""
        temp += (
            "#ifdef GOOGLE_CUDA\n\
"
            "#define EIGEN_USE_GPU\n"
        )
        temp = write_libraries(temp, libraries)
        temp = write_headers(temp, headers)
        temp = write_namespaces(temp, namespace)
        temp = write_defined(temp, defined, "gpu")

        temp = write_constants(temp, constants, "gpu")

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

        for f in function_list:
            temp = write_function_definition(temp, f, "gpu")

        temp += "\n"
        temp += gpuArithmeticOperators

        for f in function_list:
            temp += "\n"
            temp = write_function(temp, f, "gpu")

        function_list[-1].args.append(argument("context", "const OpKernelContext*", 0, False, []))

        for c in custom_op_list:
            temp = write_custom_op(temp, c, function_list[-1], "gpu", process_name)

        temp = re.sub("([ ,+\-*/]+)sign([ (;]+)", "\g<1>signn\g<2>", temp)
        temp = re.sub("([ ,+\-*/]+)signvec([ (;]+)", "\g<1>signvecc\g<2>", temp)

        temp += "\n#endif\n"

        with open(destination + "gpu/matrix_" + process_name + ".cu.cc", "w") as fh:
            fh.write(temp)

        temp = ""
        temp = write_header_file(temp, c, function_list[-1])
        with open(destination + "gpu/matrix_" + process_name + ".h", "w") as fh:
            fh.write(temp)

        temp = ""
        temp = modify_matrix(matrix_source, temp, process_name, destination)
        with open(destination + matrix_name, "w") as fh:
            fh.write(temp)

        # --------------------------------------------------------------------------------------


def compile(destination):
    subprocess.check_output(["/bin/sh", "-c", "cd " + destination + "; make"])


if __name__ == "__main__":
    translate(folder_name)
