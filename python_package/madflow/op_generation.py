from madflow.op_aux_functions import *


def parallelize_function(f):
    parall = False
    n_events = argument("nevents", "const int", 0, False, [])
    spacing = "    "
    s = 0
    while s < len(f.scope):
        if parall == True:
            f.scope[s] = spacing + f.scope[s]
            # print(f.scope[s])
        elif clean_spaces(f.scope[s]).startswith("//Begin"):
            parall = True
            s += 1
            while clean_spaces(f.scope[s]).startswith("//") == True:
                # print(f.scope[s])
                s += 1

            f.scope.insert(
                s, "auto thread_pool = context->device()->tensorflow_cpu_worker_threads()->workers;"
            )
            s += 1
            f.scope.insert(s, "const int ncores = (int)thread_pool->NumThreads();")
            s += 1
            f.scope.insert(s, INT64Type + " nreps;")
            s += 1
            f.scope.insert(s, "if (ncores > 1) {")
            s += 1
            f.scope.insert(s, "    nreps = (" + INT64Type + ")nevents / ncores;")
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
            f.scope.insert(s, "for (auto it = t; it < w; it += 1) {")
            s += 1

        s += 1

    f.scope.insert(s, "}")
    s += 1
    f.scope.insert(s, "};")
    s += 1
    f.scope.insert(s, "thread_pool->ParallelFor(" + n_events.name + ", p, DoWork);")

    f = prepare_custom_op(f, n_events)

    f.args.append(argument("context", "const OpKernelContext*", 0, False, []))

    return f


def prepare_custom_op(f, nevents):

    for i in range(len(f.args) - 1):
        f.args[i].type = "const " + f.args[i].type
        if f.args[i].type.endswith("*") == False:
            f.args[i].type += "*"
            if f.args[i].tensor == True:
                for j in range(len(f.scope)):
                    f.scope[j] = re.sub(
                        "([()[\]{} ,+\-*/]*)" + f.args[i].name + "([()[\]{} ,+\-*/]*)",
                        "\g<1>" + f.args[i].name + "[it]" + "\g<2>",
                        f.scope[j],
                    )
            else:
                for j in range(len(f.scope)):
                    f.scope[j] = re.sub(
                        "([()[\]{} ,+\-*/]*)" + f.args[i].name + "([()[\]{} ,+\-*/]*)",
                        "\g<1>" + f.args[i].name + "[0]" + "\g<2>",
                        f.scope[j],
                    )

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

    f.args.append(nevents)

    return f


def define_custom_op(custom_op_list, func):
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


def modify_matrix(infile, temp, process_name, destination):
    f = open(infile, "r")
    line = f.readline()
    new_matrix = ""
    inside_matrix = False
    p = re.sub("_", "", process_name)
    while line != "":
        temp += line
        if clean_spaces(line).startswith("defsmatrix("):
            inside_matrix = True
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
                break
        line = f.readline()
    new_matrix = re.sub("smatrix\(", "cusmatrix(", new_matrix)
    new_matrix = re.sub("self\.matrix\(", "matrixOp.matrix" + p + "(", new_matrix)
    temp += new_matrix
    while line != "":
        temp += line
        line = f.readline()
    return temp


def extract_constants(func, constants):

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

    for i in range(len(func.scope)):  # This loop can be reversed
        if clean_spaces(func.scope[len(func.scope) - i - 1]).startswith(func.args[-3].name):
            func.scope[len(func.scope) - i - 1] = re.sub(".real\(\)", "", func.scope[len(func.scope) - i - 1])
            break # Only one occurrence

    return func
