from madflow.op_aux_functions import *

n_events = argument("nevents", "const int", 0, False, [])
forLoopString = "for (int it = 0; it < " + n_events.name + "; it += 1) {"


def serialize_function(f):
    """Create a loop over the total number of events"""
    forLoop = False
    spacing = "    "
    s = 0
    while s < len(f.scope):
        if forLoop == True:
            f.scope[s] = spacing + f.scope[s]
        elif clean_spaces(f.scope[s]).startswith("//Begin"):
            forLoop = True
            s += 1
            while clean_spaces(f.scope[s]).startswith("//") == True:
                s += 1

            f.scope.insert(s, forLoopString)

        s += 1

    f.scope.insert(s, "}")
    s += 1

    f = prepare_custom_op(f, n_events)

    f.args.append(argument("context", "const OpKernelContext*", 0, False, []))

    return f


def parallelize_function(f, parallelizationType):
    """Parallelize the loop over the total number of events"""
    s = 0
    if parallelizationType == "OpenMP":
        while s < len(f.scope):
            if clean_spaces(f.scope[s]).startswith(clean_spaces(forLoopString)):
                f.scope.insert(s, "#pragma omp parallel for")
                break
            s += 1
    elif parallelizationType == "ThreadPool":
        while s < len(f.scope):
            if clean_spaces(f.scope[s]).startswith(clean_spaces(forLoopString)):
                f.scope.insert(
                    s,
                    "auto thread_pool = context->device()->tensorflow_cpu_worker_threads()->workers;",
                )
                s += 1
                f.scope.insert(s, "const int ncores = (int)thread_pool->NumThreads();")
                s += 1
                f.scope.insert(s, INT64Type + " nreps;")
                s += 1
                f.scope.insert(s, "if (ncores > 1) {")
                s += 1
                f.scope.insert(s, "    nreps = (" + INT64Type + ")" + n_events.name + " / ncores;")
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
                f.scope.insert(s, "auto DoWork = [&](" + INT64Type + " t, " + INT64Type + " w) {")
                s += 1
                del f.scope[s]
                f.scope.insert(s, "for (auto it = t; it < w; it += 1) {")
                break
            s += 1

        s = len(f.scope)
        f.scope.insert(s, "};")
        s += 1
        f.scope.insert(s, "thread_pool->ParallelFor(" + n_events.name + ", p, DoWork);")
    elif parallelizationType == "CUDA":
        while s < len(f.scope):
            if clean_spaces(f.scope[s]).startswith(clean_spaces(forLoopString)):
                f.scope[s] = (
                    "for (int it = blockIdx.x * blockDim.x + threadIdx.x; it < "
                    + f.args[-1].name
                    + "; it += blockDim.x * gridDim.x) {"
                )
                break
            s += 1

    return f


def prepare_custom_op(f, nevents):
    """Few changes to the structure of the Op"""

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
    f.args[-1].type = doubleType + "*"

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


def define_custom_op(custom_op_list, func):
    """Generates a custom_operator object"""
    s = []

    inputTensorsNumber = len(func.args) - 3

    for i in range(inputTensorsNumber):
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
    for i in range(inputTensorsNumber):
        line += ", " + func.args[i].name
    line += ", " + func.args[-3].name + ".data()"
    line += ", " + func.args[-2].name
    line += ", " + func.args[-1].name + ");"
    s.append(line)

    c_op = custom_operator("MatrixOp", s, functor_name)
    custom_op_list.append(c_op)
    return custom_op_list


def modify_matrix(infile, process_name, destination):
    """add the ability to execute the Op from MadFlow"""
    f = open(infile, "r")
    line = f.readline()
    previousLine = ""
    new_matrix = ""
    matrixSourceCode = ""
    matrixSourceCodeArray = []
    skipLines = False
    inside_matrix = False
    p = re.sub("_", "", process_name)
    while line != "":
        if skipLines == True:
            if clean_spaces(line).startswith("return"):
                skipLines = False
        else:
            # temp += line
            matrixSourceCodeArray.append(line)
        if clean_spaces(line).startswith(
            "defcusmatrix("
        ):  # I can re-run the script without creating duplicates of cusmatrix()
            skipLines = True
            matrixSourceCodeArray.pop()
            matrixSourceCodeArray.pop()
        if clean_spaces(line).startswith("defsmatrix("):
            inside_matrix = True
            new_matrix += "\n"  # add empty line
            new_matrix += (
                previousLine  # add @tf.function() with the same input signature as smatrix
            )
        if inside_matrix == True:
            if clean_spaces(line).startswith("for"):
                space = line.split("for")[0]
                new_matrix += (
                    space
                    + "matrixOp = tf.load_op_library('"
                    + destination
                    + "./matrix_"
                    + process_name
                    + "_cu.so')\n"
                )
            new_matrix += line
            if clean_spaces(line).startswith("return"):
                inside_matrix = False
                new_matrix = re.sub("smatrix\(", "cusmatrix(", new_matrix)
                new_matrix = re.sub("self\.matrix\(", "matrixOp.matrix" + p + "(", new_matrix)
                # temp += new_matrix
                matrixSourceCodeArray.append(new_matrix)
                # break
        if clean_spaces(line) != "":  # not checking if it is inside a comment !!!
            previousLine = line
        line = f.readline()
    # new_matrix = re.sub("smatrix\(", "cusmatrix(", new_matrix)
    # new_matrix = re.sub("self\.matrix\(", "matrixOp.matrix" + p + "(", new_matrix)
    # temp += new_matrix
    # while line != "":
    #    temp += line
    #    line = f.readline()

    for line in matrixSourceCodeArray:
        matrixSourceCode += line

    return matrixSourceCode


def extract_constants(func, constants):
    """cf and denom are constant (event-independent)
    this function moves them to global scope"""

    count = 0
    for i in range(len(func.scope)):
        if func.scope[i].startswith("const double "):
            constants.append(change_array_into_variable(func.scope[i]))
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
    therefore .real() must be removed"""

    for i in range(len(func.scope)):  # This loop can be reversed
        if clean_spaces(func.scope[len(func.scope) - i - 1]).startswith(func.args[-3].name):
            func.scope[len(func.scope) - i - 1] = re.sub(
                ".real\(\)", "", func.scope[len(func.scope) - i - 1]
            )
            break  # Only one occurrence

    return func
