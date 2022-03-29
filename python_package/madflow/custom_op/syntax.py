"""Control of C++/CUDA syntax robustness"""

import re

import madflow.custom_op.aux_functions as op_af
import madflow.custom_op.global_constants as op_gc
import madflow.custom_op.classes as op_cl


def check_variables(counter, function_list):
    """Check if all variables of i-th function are
    correctly defined
    counter: index of the i-th function (i)
    function_list: list of all function objects (updated)"""
    all_sizes_defined = True
    found = False
    i = 0

    # Check if all function arguments have a defined size
    for i in range(len(function_list[counter].args)):
        if (function_list[counter].args)[i].size == -1:
            all_sizes_defined = False
            break

    if all_sizes_defined == False:
        for j in range(len(function_list[counter].scope)):
            line = (function_list[counter].scope)[j]

            for k in range(len(function_list)):
                match = re.search(
                    function_list[k].name + "\(.*" + (function_list[counter].args)[i].name, line
                )
                if match != None:
                    check_variables(k, function_list)
                    if function_list[k].args[-1].size != -1:
                        (function_list[counter].args)[i].size = function_list[k].args[-1].size
                        (function_list[counter].args)[i].type = op_af.clean_pointer(
                            function_list[k].args[-1].type
                        )
                        if function_list[k].args[-1].size != 0:
                            (function_list[counter].args)[i].type += "*"
                        else:
                            (function_list[counter].args)[i].type += "&"
                        #found = False # to avoid counting multiple times

    i = 0
    all_sizes_defined = True
    for i in range(len(function_list[counter].scope_args)):
        if (function_list[counter].scope_args)[i].size == -1:
            all_sizes_defined = False
            break

    line_of_definition = -1
    variabe_type = ""
    new_size = -1

    if all_sizes_defined == False:
        for j in range(len(function_list[counter].scope)):
            line = (function_list[counter].scope)[j]
            match = re.search(
                (function_list[counter].scope_args)[i].type
                + "\** "
                + (function_list[counter].scope_args)[i].name
                + ";",
                line,
            )
            if match != None:
                found = True
                line_of_definition = j

            if found == True:
                for k in range(len(function_list)):
                    match = re.search(
                        function_list[k].name
                        + "\(.*"
                        + (function_list[counter].scope_args)[i].name,
                        line,
                    )
                    if match != None:
                        check_variables(k, function_list)
                        new_size = function_list[k].args[-1].size
                        (function_list[counter].scope_args)[i].size = new_size
                        variabe_type = re.sub("[&\*]*", "", function_list[k].args[-1].type)
                        (function_list[counter].scope_args)[i].type = variabe_type
                        found = False  # to avoid counting multiple times

    if variabe_type != "":
        (function_list[counter].scope)[line_of_definition] = re.sub(
            "^[a-zA-Z0-9_]* ",
            variabe_type + " ",
            (function_list[counter].scope)[line_of_definition],
        )
        if new_size != 0:
            (function_list[counter].scope)[line_of_definition] = re.sub(
                ";", "[" + str(new_size) + "];", (function_list[counter].scope)[line_of_definition]
            )


def check_lines(counter, function_list):
    """Check if all lines of the i-th function have
    correct grammar and syntax
    counter: index of the i-th function (i)
    function_list: list of all function objects (updated)

    return: updated function_list"""
    it = 0
    while it < len(function_list[counter].scope):
        line = function_list[counter].scope[it]

        if function_list[counter].args[-1].size == -1:
            match = re.search("^[ +\-*/,()[\]{}]*T\(", line)
            if match == None:
                l = line.split(" = ")
                if function_list[counter].args[-1].name == l[0]:
                    custom_type = "int"
                    type_value = 0
                    value = l[1]
                    for v in function_list[counter].args + function_list[counter].scope_args:
                        reassignment = re.search(
                            "[()[\]{}+\-*/, \n]" + v.name + "[()[\]{}+\-*/, \n;]", value
                        )
                        if reassignment == None:
                            reassignment = re.search("^" + v.name + "[()[\]{}+\-*/, \n;]", value)
                        if reassignment == None:
                            reassignment = re.search("[()[\]{}+\-*/, \n]" + v.name + "$", value)
                        if reassignment == None:
                            reassignment = re.search("^" + v.name + "$", value)
                        if reassignment != None:
                            if v.type.startswith("T"):
                                custom_type = "T"
                                break
                            elif v.type.startswith(op_gc.DOUBLE_TYPE):
                                type_value += 1

                    if custom_type != "T" and type_value > 0:
                        custom_type = op_gc.DOUBLE_TYPE

                    function_list[counter].args[-1].type = custom_type + "&"
                    function_list[counter].args[-1].size = 0

        ls = line.split(" = ")
        if len(ls) > 1:
            if ls[1].startswith("T("):
                value = ls[1]
                for v in function_list[counter].args + function_list[counter].scope_args:
                    reassignment = re.search(
                        "[()[\]{}+\-*/, \n]" + v.name + "[()[\]{}+\-*/, \n;]", value
                    )
                    if reassignment == None:
                        reassignment = re.search("^" + v.name + "[()[\]{}+\-*/, \n;]", value)
                    if reassignment == None:
                        reassignment = re.search("[()[\]{}+\-*/, \n]" + v.name + "$", value)
                    if reassignment == None:
                        reassignment = re.search("^" + v.name + "$", value)
                    if reassignment != None:
                        if v.type.startswith("T"):
                            match = re.search("T\( *" + v.name + "[0-9[\]]* *\)", value)
                            if match != None:
                                if ls[0].startswith(v.name):
                                    function_list[counter].scope[it] = ""
                                    break
                        elif v.type.startswith(op_gc.DOUBLE_TYPE):
                            match = re.search("T\( *" + v.name + "[0-9[\]]* *\)", value)
                            if match != None:
                                if ls[0].startswith(v.name):
                                    for it2 in range(it):
                                        function_list[counter].scope[it2] = re.sub(
                                            "([()[\]{}, +\-*/]*" + v.name + ")([()[\]{}, +\-*/;]*)",
                                            "\g<1>_\g<2>",
                                            function_list[counter].scope[it2],
                                        )
                                    function_list[counter].scope[it] = (
                                        "T "
                                        + ls[0]
                                        + " = "
                                        + re.sub(
                                            "([()[\]{}, +\-*/]*"
                                            + v.name
                                            + ")([()[\]{}, +\-*/;]*)\);",
                                            "\g<1>_\g<2>",
                                            ls[1],
                                        )
                                        + ", 0);"
                                    )
                                    function_list[counter].scope_args.append(
                                        op_cl.Argument(v.name, "T", 0, False, [])
                                    )
                                    for it2 in range(len(function_list[counter].args)):
                                        if v.name == function_list[counter].args[it2].name:
                                            function_list[counter].args[it2].name += "_"
                                    for it2 in range(len(function_list[counter].scope_args)):
                                        if v.name == function_list[counter].scope_args[it2].name:
                                            function_list[counter].scope_args[it2].name += "_"
                                    break
            else:
                for f in function_list:
                    match = re.search("^ *" + f.name + " *\(", ls[1])
                    if match != None:
                        if f.type == "void":
                            if (
                                ls[0].startswith("T")
                                or ls[0].startswith(op_gc.DOUBLE_TYPE)
                                or ls[0].startswith("int")
                            ):
                                for v in range(len(function_list[counter].scope_args)):
                                    if l[0].endswith(
                                        " " + function_list[counter].scope_args[v].name
                                    ):
                                        function_list[counter].scope_args[v].type = re.sub(
                                            "[&*]*", "", f.args[-1].type
                                        )
                                        function_list[counter].scope_args[v].size = f.args[-1].size
                                        break
                                if f.args[-1].size > 0:
                                    function_list[counter].scope.insert(
                                        it,
                                        function_list[counter].scope_args[v].type
                                        + " "
                                        + function_list[counter].scope_args[v].name
                                        + "["
                                        + str(f.args[-1].size)
                                        + "];",
                                    )
                                else:
                                    function_list[counter].scope.insert(
                                        it,
                                        function_list[counter].scope_args[v].type
                                        + " "
                                        + function_list[counter].scope_args[v].name
                                        + ";",
                                    )
                                it += 1
                                function_list[counter].scope[it] = re.sub(
                                    ".* +" + function_list[counter].scope_args[v].name + " *=",
                                    function_list[counter].scope_args[v].name + " =",
                                    function_list[counter].scope[it],
                                )
                            function_list[counter].scope[it] = re.sub(
                                "([a-zA-Z0-9_]*) *= *(.*)\) *;",
                                "\g<2>, \g<1>);",
                                function_list[counter].scope[it],
                            )

        match = re.search("tf.concat", line)
        if match != None:
            function_list[counter].scope.remove(line)
            line = re.sub("(.*)tf.concat\( *\[(.*) *] *, *axis.*", "\g<1>\g<2>", line)
            assigned = op_af.clean_spaces(line.split("=")[1])
            assigned_variable = op_af.clean_spaces(line.split("=")[0])
            var_list = assigned.split(",")
            var_length = []
            conc_size = 0
            unknown = False
            type_value = 0
            conc_type = "int"
            for var in var_list:
                for i in range(len(function_list[counter].args)):
                    if var == function_list[counter].args[i].name:
                        c_size = 0
                        if function_list[counter].args[i].size == 0:
                            c_size += 1
                        elif function_list[counter].args[i].size != -1:
                            c_size += args[i].size
                        else:
                            c_size = 0
                            unknown = True
                        if function_list[counter].args[i].type.startswith("T"):
                            conc_type = "T"
                        elif function_list[counter].args[i].type.startswith(op_gc.DOUBLE_TYPE):
                            type_value += 1
                        conc_size += c_size
                        var_length.append(c_size)
                        break
                for i in range(len(function_list[counter].scope_args)):
                    if var == function_list[counter].scope_args[i].name:
                        c_size = 0
                        if function_list[counter].scope_args[i].size == 0:
                            c_size += 1
                        elif function_list[counter].scope_args[i].size != -1:
                            c_size += int(function_list[counter].scope_args[i].size)
                        else:
                            c_size = 0
                            unknown = True
                        if function_list[counter].scope_args[i].type.startswith("T"):
                            conc_type = "T"
                        elif (
                            function_list[counter].scope_args[i].type.startswith(op_gc.DOUBLE_TYPE)
                        ):
                            type_value += 1
                        conc_size += c_size
                        var_length.append(c_size)
                        break

            if conc_type != "T":
                if type_value > 0:
                    conc_type = op_gc.DOUBLE_TYPE

            if unknown == False:

                for i in range(len(function_list[counter].args)):
                    if assigned_variable == function_list[counter].args[i].name:
                        function_list[counter].args[i].type = conc_type
                        if conc_size > 1:
                            function_list[counter].args[i].size = conc_size
                            function_list[counter].args[i].type += "*"
                        elif function_list[counter].args[i].size == 1:
                            function_list[counter].args[i].size = 0
                        break
                for i in range(len(function_list[counter].scope_args)):
                    if assigned_variable == function_list[counter].scope_args[i].name:
                        function_list[counter].scope_args[i].type = conc_type
                        if conc_size > 1:
                            function_list[counter].scope_args[i].size = conc_size
                        elif function_list[counter].scope_args[i].size == 1:
                            function_list[counter].scope_args[i].size = 0
                        break
                i = 0
                it2 = 0
                while i < conc_size:
                    for j in range(len(var_list)):
                        newline = ""
                        if var_length[j] == 1:
                            newline = assigned_variable + "[" + str(i) + "] = " + var_list[j] + ";"
                            function_list[counter].scope.insert(it + it2, newline)
                            it2 += 1
                        else:
                            function_list[counter].scope.insert(
                                it + it2,
                                "for (int it1 = 0; it1 < " + str(var_length[j]) + "; it1++) {",
                            )
                            it2 += 1
                            function_list[counter].scope.insert(
                                it + it2,
                                "    "
                                + assigned_variable
                                + "["
                                + str(i)
                                + " + it1] = "
                                + var_list[j]
                                + "[it1];",
                            )
                            it2 += 1
                            function_list[counter].scope.insert(it + it2, "}")
                            it2 += 1
                        i += int(var_length[j])
        it += 1
