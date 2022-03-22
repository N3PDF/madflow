"""Optimization of the Custom Operator (parallelization / memory optimizations)"""

import re

import madflow.op_aux_functions as op_af
import madflow.op_classes as op_cl
import madflow.op_global_constants as op_gc

n_events = op_cl.Argument("nevents", "const int", 0, False, [])
for_loop_string = "for (int it = 0; it < " + n_events.name + "; it += 1) {"


def serialize_function(f):
    """Create a loop over the total number of events
    f: function object

    return: updated function object"""
    for_loop = False
    spacing = "    "
    s = 0
    while s < len(f.scope):
        if for_loop == True:
            f.scope[s] = spacing + f.scope[s]
        elif op_af.clean_spaces(f.scope[s]).startswith("//Begin"):
            for_loop = True
            s += 1
            while op_af.clean_spaces(f.scope[s]).startswith("//") == True:
                s += 1

            f.scope.insert(s, for_loop_string)

        s += 1

    f.scope.insert(s, "}")
    s += 1

    f = prepare_custom_op(f, n_events)

    f.args.append(op_cl.Argument("context", "const OpKernelContext*", 0, False, []))

    return f


def parallelize_function(f, parallelization_type):
    """Parallelize the loop over the total number of events
    f: function object
    parallelization_type: OpenMP/ThreadPool/CUDA

    return: updated function object"""
    s = 0
    if parallelization_type == "OpenMP":
        while s < len(f.scope):
            if op_af.clean_spaces(f.scope[s]).startswith(clean_spaces(for_loop_string)):
                f.scope.insert(s, "#pragma omp parallel for")
                break
            s += 1
    elif parallelization_type == "ThreadPool":
        while s < len(f.scope):
            if op_af.clean_spaces(f.scope[s]).startswith(op_af.clean_spaces(for_loop_string)):
                f.scope.insert(
                    s,
                    "auto thread_pool = context->device()->tensorflow_cpu_worker_threads()->workers;",
                )
                s += 1
                f.scope.insert(s, "const int ncores = (int)thread_pool->NumThreads();")
                s += 1
                f.scope.insert(s, op_gc.INT64_TYPE + " nreps;")
                s += 1
                f.scope.insert(s, "if (ncores > 1) {")
                s += 1
                f.scope.insert(
                    s, "    nreps = (" + op_gc.INT64_TYPE + ")" + n_events.name + " / ncores;"
                )
                s += 1
                f.scope.insert(s, "} else {")
                s += 1
                f.scope.insert(s, "    nreps = 1;")
                s += 1
                f.scope.insert(s, "}")
                s += 1
                f.scope.insert(
                    s,
                    "const ThreadPool::SchedulingParams p(ThreadPool::SchedulingStrategy::kFixedBlockSize, absl::nullopt, nreps);",
                )
                s += 1
                f.scope.insert(
                    s, "auto DoWork = [&](" + op_gc.INT64_TYPE + " t, " + op_gc.INT64_TYPE + " w) {"
                )
                s += 1
                del f.scope[s]
                f.scope.insert(s, "for (auto it = t; it < w; it += 1) {")
                break
            s += 1

        s = len(f.scope)
        f.scope.insert(s, "};")
        s += 1
        f.scope.insert(s, "thread_pool->ParallelFor(" + n_events.name + ", p, DoWork);")
    elif parallelization_type == "CUDA":
        while s < len(f.scope):
            if op_af.clean_spaces(f.scope[s]).startswith(op_af.clean_spaces(for_loop_string)):
                f.scope[s] = (
                    "for (int it = blockIdx.x * blockDim.x + threadIdx.x; it < "
                    + f.args[-1].name
                    + "; it += blockDim.x * gridDim.x) {"
                )
                break
            s += 1

    return f


def prepare_custom_op(f, nevents):
    """Few changes to the structure of the Op
    f: function object
    nevents: number of MC events

    return: updated function object"""

    # momenta, masses, widths and coupling constants are const
    # pass them by pointer
    for i in range(len(f.args) - 1):
        f.args[i].type = "const " + f.args[i].type
        if f.args[i].type.endswith("*") == False:
            f.args[i].type += "*"
            if f.args[i].tensor == True:
                # Tensors are arrays
                for j in range(len(f.scope)):
                    f.scope[j] = re.sub(
                        "([()[\]{} ,+\-*/]*)" + f.args[i].name + "([()[\]{} ,+\-*/]*)",
                        "\g<1>" + f.args[i].name + "[it]" + "\g<2>",
                        f.scope[j],
                    )
            else:
                # Non-Tensors are arrays with only one component
                for j in range(len(f.scope)):
                    f.scope[j] = re.sub(
                        "([()[\]{} ,+\-*/]*)" + f.args[i].name + "([()[\]{} ,+\-*/]*)",
                        "\g<1>" + f.args[i].name + "[0]" + "\g<2>",
                        f.scope[j],
                    )

    # The polarized Matrix Element is an array of double
    f.args[-1].type = op_gc.DOUBLE_TYPE + "*"

    for j in range(len(f.scope)):
        f.scope[j] = re.sub(
            "([()[\]{} ,+\-*/]*)" + f.args[-1].name + "([()[\]{} ,+\-*/]*)",
            "\g<1>" + f.args[-1].name + "[it]" + "\g<2>",
            f.scope[j],
        )
        match = re.search("[()[\]{} ,+\-*/]*" + f.args[0].name + "\[", f.scope[j])
        if match != None:
            number = int(
                re.sub(
                    ".*[()[\]{} ,+\-*/]*" + f.args[0].name + "\[([0-9]*)\].*", "\g<1>", f.scope[j]
                )
            )
            f.scope[j] = re.sub(
                "([()[\]{} ,+\-*/]*)" + f.args[0].name + "\[([0-9]*)\]",
                "\g<1>"
                + f.args[0].name
                + "+("
                + str(f.args[0].slice[-1])
                + "*it + "
                + str(int(f.args[0].slice[-2]) * number)
                + ")",
                f.scope[j],
            )

    # Add the number events as a function argument
    f.args.append(nevents)

    return f


def define_custom_op(func):
    """Generates a custom_operator object
    func: Function object

    return: CustomOperator object"""
    s = []

    input_tensors_number = len(func.args) - 3

    for i in range(input_tensors_number):
        s.append("const Tensor& " + func.args[i].name + "_tensor = context->input(" + str(i) + ");")
        s.append(
            "auto "
            + func.args[i].name
            + " = "
            + func.args[i].name
            + "_tensor.flat<"
            + re.sub("const ([^&*]*)[&*]*", "\g<1>", func.args[i].type)
            + ">().data();"
        )
        s.append("")

    # func.args[-1] is context
    s.append(
        func.args[-2].type
        + " "
        + func.args[-2].name
        + " = "
        + func.args[0].name
        + "_tensor.shape().dim_size(0);"
    )

    s.append("Tensor* output_tensor = NULL;")
    s.append(
        "OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({"
        + func.args[-2].name
        + "}), &output_tensor));"
    )
    s.append(
        "auto "
        + func.args[-3].name
        + " = output_tensor->flat<"
        + re.sub("([^&*]*)[&*]*", "\g<1>", func.args[-3].type)
        + ">();"
    )

    functor_name = re.sub("Op", "Functor", "MatrixOp")

    s.append("")
    line = functor_name + "<Device, COMPLEX_TYPE>()(context->eigen_device<Device>()"
    for i in range(input_tensors_number):
        line += ", " + func.args[i].name
    line += ", " + func.args[-3].name + ".data()"
    line += ", " + func.args[-2].name
    line += ", " + func.args[-1].name + ");"
    s.append(line)

    return op_cl.CustomOperator("MatrixOp", s, functor_name)


def modify_matrix(infile, process_name, destination):
    """add the ability to execute the Op from MadFlow
    infile: complete path to matrix_1_xxxxx.py
    process_name: process name
                  (read from matrix_1_xxxxx.py)
    destination: directory of the Custom Operator

    return: CustomOperator object"""
    f = open(infile, "r")
    line = f.readline()
    previous_line = ""
    new_matrix = ""
    matrix_source_code = ""
    matrix_source_code_array = []
    skip_lines = False
    inside_matrix = False
    p = re.sub("_", "", process_name)
    while line != "":
        if skip_lines == True:
            if op_af.clean_spaces(line).startswith("return"):
                skip_lines = False
        else:
            # temp += line
            matrix_source_code_array.append(line)
        if op_af.clean_spaces(line).startswith(
            "defcusmatrix("
        ):  # I can re-run the script without creating duplicates of cusmatrix()
            skip_lines = True
            matrix_source_code_array.pop()
            matrix_source_code_array.pop()
        if op_af.clean_spaces(line).startswith("defsmatrix("):
            inside_matrix = True
            new_matrix += "\n"  # add empty line
            new_matrix += (
                previous_line  # add @tf.function() with the same input signature as smatrix
            )
        if inside_matrix == True:
            if op_af.clean_spaces(line).startswith("for"):
                space = line.split("for")[0]
                new_matrix += (
                    space
                    + "matrixOp = tf.load_op_library('"
                    + (destination / ("matrix_" + process_name + "_cu.so'")).as_posix()
                    + ")\n"
                )
            new_matrix += line
            if op_af.clean_spaces(line).startswith("return"):
                inside_matrix = False
                new_matrix = re.sub("smatrix\(", "cusmatrix(", new_matrix)
                new_matrix = re.sub("self\.matrix\(", "matrixOp.matrix" + p + "(", new_matrix)
                # temp += new_matrix
                matrix_source_code_array.append(new_matrix)
                # break
        if op_af.clean_spaces(line) != "":  # not checking if it is inside a comment !!!
            previous_line = line
        line = f.readline()

    for line in matrix_source_code_array:
        matrix_source_code += line

    return matrix_source_code


def extract_constants(func, constants):
    """cf and denom are constant (event-independent)
    this function moves them to global scope
    func: Function object
    constants: list of constants

    return: updated Function object and constants"""

    count = 0
    for i in range(len(func.scope)):
        if func.scope[i].startswith("const double "):
            constants.append(op_af.change_array_into_variable(func.scope[i]))
            del func.scope[i]
            i -= 1
            count += 1
        if count == 2:
            break

    for i in range(len(func.scope)):
        match = re.search("denom", func.scope[len(func.scope) - i - 1])
        if match != None:
            func.scope[len(func.scope) - i - 1] = re.sub(
                "denom\[[a-zA-Z0-9+\-*/_]\]", "denom", func.scope[len(func.scope) - i - 1]
            )
            break

    return func, constants


def remove_real_ret(func):
    """In the Op the return variable is already declared as double,
    therefore .real() must be removed
    func: Function object

    return: updated Function object"""

    for i in range(len(func.scope)):  # This loop can be reversed
        if op_af.clean_spaces(func.scope[len(func.scope) - i - 1]).startswith(func.args[-3].name):
            func.scope[len(func.scope) - i - 1] = re.sub(
                ".real\(\)", "", func.scope[len(func.scope) - i - 1]
            )
            break  # Only one occurrence

    return func
