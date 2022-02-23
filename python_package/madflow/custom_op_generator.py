import subprocess


import madflow.wavefunctions_flow
from madflow.makefile_template import write_makefile

from madflow.op_write_templates import *
from madflow.op_syntax import *
from madflow.op_read import *
from madflow.op_generation import *


folder_name = "prov/"

temp = ""
devices = ["cpu", "gpu"]


def translate(destination):

    if destination[-1] != "/":  # Avoid weird behaviours if destination does not end with '/'
        destination += "/"
    file_sources = [madflow.wavefunctions_flow.__file__]  # path to wavefunctions_flow.py

    # Create the directory for the Op source code and create the makefile

    subprocess.check_output(["/bin/sh", "-c", "mkdir -p " + destination + "gpu/"])
    write_makefile(destination)

    # Generate sign functions
    auxiliary_functions = []
    function_list_ = []
    auxiliary_functions, function_list_ = generate_auxiliary_functions(
        auxiliary_functions, function_list_
    )

    # Read wavefunctions_flow.py
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

    # Find all generated matrix_1_xxxxx.py (one for each subprocess)
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

        function_list[-1] = serialize_function(function_list[-1])

        custom_op_list = define_custom_op(custom_op_list, function_list[-1])

        function_list[-1], constants = extract_constants(function_list[-1], constants)

        function_list[-1] = remove_real_ret(function_list[-1])

        # write the Op for both CPU and GPU
        for device in devices:
            write_custom_op(
                headers,
                namespace,
                defined,
                constants,
                cpuConstants,
                function_list,
                custom_op_list,
                destination,
                process_name,
                device,
            )

        # write matrix_xxxxx.h
        temp = ""
        for c in custom_op_list:
            temp += write_header_file(c, function_list[-1])
        with open(destination + "gpu/matrix_" + process_name + ".h", "w") as fh:
            fh.write(temp)

        # write matrix_1_xxxxx.py
        temp = ""
        temp = modify_matrix(matrix_source, process_name, destination)
        with open(destination + matrix_name, "w") as fh:
            fh.write(temp)

        # --------------------------------------------------------------------------------------


def compile(destination):
    subprocess.check_output(["/bin/sh", "-c", "cd " + destination + "; make"])


if __name__ == "__main__":
    translate(folder_name)
