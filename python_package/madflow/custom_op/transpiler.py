"""Line by line translation of python syntax into C++ syntax"""

import re

import madflow.custom_op.aux_functions as op_af
import madflow.custom_op.global_constants as op_gc
import madflow.custom_op.classes as op_cl

# Parse line


def parse_line(line, args, scope_variables, scope, inside_comment):
    """Translate a single line into C++ source code
    line: python source code (string)
    args: function arguments
    scope_variables: variables defined within the scope
    scope: list of strings with previous transpiled lines
    inside_comment: boolean -> true if the line is inside
                               a comment block

    return: transpiled line,
            updated scope_variables, scope, inside_comment"""
    return_line = ""

    if inside_comment == True:
        match = re.search('"""', line)
        if match != None:
            s = line.split('"""', 1)
            inside_comment = False
            scope.append(s[0] + "*/")
            line = s[1]
        else:
            scope.append(line)
            return return_line, scope_variables, scope, inside_comment

    s = line.split("#", 1)
    line = s[0]
    is_arrayy = ""
    is_array = False
    variable_type = op_gc.DOUBLE_TYPE  # not needed
    comment = ""
    defined_in_args = False
    defined_in_scope = False
    position = -1
    if len(s) > 1:
        comment = s[1]

    if op_af.clean_spaces(line).startswith('"""'):
        line = re.sub('"""', "/*", line)
        scope.append(line)
        inside_comment = True
        return return_line, scope_variables, scope, inside_comment

    if op_af.clean_spaces(line) != "":
        line = re.sub("([a-zA-Z0-9_()[\]{} ])=", "\g<1> =", line, 1)
        split_line = line.split(" = ")
        if op_af.clean_spaces(line).startswith("return"):
            return return_line, scope_variables, scope, inside_comment

        for i in range(len(split_line) - 1):
            split_line.insert(2 * i + 1, "=")
        split_line[0] = op_af.clean_spaces(split_line[0])
        assigned_variable = op_af.clean_index(split_line[0])
        already_defined = False
        for i in range(len(args)):
            if assigned_variable == args[i].name:
                already_defined = True
                position = i
                defined_in_args = True
                break
        if already_defined == False:
            for i in range(len(scope_variables)):
                if assigned_variable == scope_variables[i].name:
                    already_defined = True
                    position = i
                    defined_in_args = True
                    break

        if split_line[0].endswith("]"):
            is_array = True

        value = split_line[2]
        square_brackets = 0
        assigned = split_line[2]
        custom_type = op_gc.DOUBLE_TYPE
        custom_size = -1
        while op_af.clean_spaces(value).startswith("["):
            value = value[1:]
            square_brackets += 1
        if square_brackets > 0:
            is_arrayy = "*"
            is_array = True
            sz = op_af.clean_spaces(value.split("]")[-1])[1:]
            if sz != "":
                custom_size = int(sz)

            value = value.split("]", 1)[0]
            if sz != "":
                definition = ""
                if already_defined == False:
                    scope.append(
                        op_af.convert_grammar(f"{ custom_type } { split_line[0] }[{ sz }];")
                    )
                scope.append(f"for (int it1 = 0; it1 < { str(sz) }; it1++) " + "{")
                scope.append("    " + op_af.convert_grammar(f"{ split_line[0] }[it1] = { value };"))
                scope.append("}")
            else:
                scope.append(op_af.convert_grammar(f"{ split_line[0] } = { value };"))

            comparison_string = "complex_tf("
            if value.startswith(comparison_string):
                custom_type = "T"
                assigned = f"T({ (value.split('('))[1].split(')')[0] })"

            comment = f"array of size { str(custom_size) }"
            value = ""

            if already_defined == True and sz != "":
                for i in range(len(args)):
                    if assigned_variable == args[i].name:
                        args[i].size = custom_size
                        args[i].type = custom_type
                        if custom_size != 0:
                            args[i].type += "*"
                        else:
                            args[i].type += "&"
                        break
            if already_defined == True and sz != "":
                for i in range(len(scope_variables)):
                    if assigned_variable == scope_variables[i].name:
                        scope_variables[i].size = custom_size
                        break
            if already_defined == True:
                return return_line, scope_variables, scope, inside_comment

        if split_line[0] == op_af.clean_spaces(assigned):  # remove things like a = a;
            return_line = ""
            return return_line, scope_variables, scope, inside_comment

        if "tf.stack([" in op_af.clean_spaces(value):
            reassignment = re.search(
                "[()[\]{}+\-*/, ]" + assigned_variable + "[()[\]{}+\-*/, ]", value
            )
            if reassignment != None:
                counter = 0
                for s in scope:
                    scope[counter] = re.sub(
                        "([()[\]{}+\-*/, ]*" + assigned_variable + ")([()[\]{}+\-*/, ]*)",
                        "\g<1>_\g<2>",
                        s,
                    )
                    counter += 1

                value = re.sub(
                    "([()[\]{}+\-*/, ]*" + assigned_variable + ")([()[\]{}+\-*/, ]*)",
                    "\g<1>_\g<2>",
                    value,
                )
                already_defined = False

                counter = 0
                for v in scope_variables:
                    if v.name == assigned_variable:
                        scope_variables[counter].name += "_"
                    counter += 1

        type_value = 0
        if custom_type != "T":
            for v in scope_variables:
                reassignment = re.search(
                    "[()[\]{}+\-*/, \n]" + v.name + "[()[\]{}+\-*/, \n]", value
                )
                if reassignment == None:
                    reassignment = re.search("^" + v.name + "[()[\]{}+\-*/, \n]", value)
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
        if custom_type != "T":
            for v in args:
                reassignment = re.search(
                    "[()[\]{}+\-*/, \n^]" + v.name + "[()[\]{}+\-*/, \n$]", value
                )
                if reassignment == None:
                    reassignment = re.search("^" + v.name + "[()[\]{}+\-*/, \n]", value)
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
            reassignment = re.search("float_me\(", value)
            if reassignment != None:
                type_value += 1
        if custom_type != "T":
            if type_value == 0:
                custom_type = "int"

        st1 = "complex_tf("
        st2 = "tf.expand_dims("
        st3 = "int_me("
        st4 = "tf.stack("
        st5 = "tf.where("
        st6 = "tf.cond("
        st7 = "tf.concat("
        st8 = "tf.transpose("
        st9 = "tf.constant("
        st10 = "tf.einsum("

        if "!=" in value or "==" in value:
            if ("tf.where" in value) == False:
                custom_type = "const bool"
        if st1 in value:
            custom_type = "T"
            if already_defined == True and is_array == False:
                if defined_in_args == True:
                    args[position].size = 0
                    args[position].type = "T&"
                elif defined_in_scope == True:
                    scope_variables[position].size = 0
                    scope_variables[position].type = "T"

        if value.startswith(st1):
            custom_type = "T"

            match = re.search("tf.stack\(", value)

            if match != None:

                comp = ["", ""]
                real_part = True

                br_count = 0
                # bracketCount = 0

                for letter in value[len(st1) :]:

                    if letter == "(":
                        br_count += 1
                    elif letter == ")":
                        br_count -= 1
                    elif letter == "," and br_count == 0:
                        real_part = False

                    if real_part == True:
                        comp[0] += letter
                    else:
                        comp[1] += letter

                comp[1] = (comp[1])[1:-2]

                for e in range(len(comp)):
                    comp[e] = re.sub("tf.stack\(\[(.*)] *, *axis.*", "\g<1>", comp[e])
                    comp[e] = comp[e].split(",")

                size_of_stack = 0

                for el in comp:
                    if len(el) > size_of_stack:
                        size_of_stack = len(el)

                scope.append(
                    f"{ custom_type }{ is_arrayy } { split_line[0] }[{ str(size_of_stack) }];"
                )

                for e in range(len(comp)):
                    if len(comp[e]) != size_of_stack:
                        for i in range(1, size_of_stack):
                            comp[e].append(comp[e][0])

                for e in range(len(comp)):
                    for i in range(size_of_stack):
                        comp[e][i] = op_af.convert_grammar(comp[e][i])
                for e in range(size_of_stack):
                    scope.append(
                        f"{ split_line[0] }[{ str(e) }] = T({ comp[0][e] },{ comp[1][e] });"
                    )

                value = ""

        elif value.startswith(st2):
            value = value[len(st2) :]

            splitted = value.split("dtype=")
            if len(splitted) == 2:
                custom_type = op_af.convert_type(splitted[1].split(")")[0])

            if value.startswith(st1):
                custom_type = "T"

            br_count = 1
            has_comma = False

            vv = ""

            if custom_type == "T":
                vv = "T("

            if value.startswith("tf.zeros_like("):
                vv += "0,0"
            elif value.startswith(st1):
                value = value[len(st1) :]
                for letter in value:
                    if letter == "(":
                        br_count += 1
                    elif letter == ")":
                        br_count -= 1
                    elif letter == ",":
                        has_comma = True
                    if br_count == 0 and has_comma == True:
                        break
                    vv += letter
            elif value.startswith("float_me("):  # here
                value = value[len("float_me(") :]
                for letter in value:
                    if letter == "(":
                        br_count += 1
                    elif letter == ")":
                        br_count -= 1
                    elif letter == "[":
                        vv += "{"
                    elif letter == "]":
                        vv += "}"
                    elif letter == ",":
                        has_comma = True
                        vv += letter
                    else:
                        vv += letter
                    if br_count == 0 and has_comma == True:
                        break
            else:
                vv += value.split(",")[0]
            if custom_type == "T":
                vv += ")"
            value = vv
        elif value.startswith(st3):
            custom_type = "int"
        elif value.startswith(st4):
            value = re.sub("\[:, (\d+)\]", "[\g<1>]", value)
            if already_defined == False:
                split_line[0] += "[]"
                is_array = True
        elif value.startswith(st5) == True or value.startswith(st6) == True:
            if value.startswith(st6):
                value = re.sub("tf.cond", "tf.where", value)
                value = re.sub("lambda: ", "", value)
            value = value[len(st5) :]
            br_value = 1
            comma_num = 0
            vv = ""
            condition = []
            for letter in value:
                if letter == "(":
                    br_value += 1
                elif letter == ")":
                    br_value -= 1
                vv += letter
                if br_value == 1 and letter == ",":
                    condition.append(vv[:-1])
                    vv = ""
                    comma_num += 1
                if br_value == 0 and comma_num == 2:
                    break
            condition.append(vv[:-1])

            for i in range(3):
                condition[i] = op_af.convert_grammar(condition[i])

            if already_defined == False:
                if op_af.clean_spaces(condition[1]).startswith("T("):
                    custom_size = 0
                else:
                    custom_size = -1
                scope_variables.append(
                    op_cl.Argument(assigned_variable, custom_type, custom_size, False, [])
                )  # need to define size
                scope.append(custom_type + is_arrayy + " " + split_line[0] + ";")

            for i in range(1, 3):
                condition[i] = op_af.clean_spaces(condition[i])

                for var in args + scope_variables:
                    if var.size != 0 and var.size != -1:
                        match = re.search("[+\-*/ ]" + var.name + " *[,(){};]+", condition[i])
                        match = re.search("[(){}, +-]" + var.name + " *[+\-*/]+", condition[i])
                        if match != None:
                            for var2 in args + scope_variables:
                                match = re.search(
                                    "[(){}, +-]"
                                    + var.name
                                    + " *[+\-*/]+ *"
                                    + var2.name
                                    + " *[,(){};]+",
                                    condition[i],
                                )
                                if match != None:
                                    if var2.size != 0 and var2.size != -1:

                                        scope.append(
                                            re.sub("[&*]*", "", var.type)
                                            + " _"
                                            + var.name
                                            + "["
                                            + str(var.size)
                                            + "];"
                                        )
                                        scope.append(
                                            "for (int it1 = 0; it1 <" + str(var.size) + "; it1++) {"
                                        )
                                        scope.append(
                                            "    _"
                                            + var.name
                                            + "[it1] = "
                                            + var.name
                                            + "[it1] "
                                            + re.sub(
                                                ".*[(){}, +-]"
                                                + var.name
                                                + "( *[+\-*/]+ *)"
                                                + var2.name
                                                + " *[,(){};]+.*",
                                                "\g<1>",
                                                condition[i],
                                            )
                                            + var2.name
                                            + "[it1];"
                                        )
                                        scope.append("}")
                                        condition[i] = re.sub(
                                            "([(){}, +-])"
                                            + var.name
                                            + " *[+\-*/]+ *"
                                            + var2.name
                                            + "( *[,(){};]+)",
                                            "\g<1>_" + var.name + "\g<2>",
                                            condition[i],
                                        )
                                        found = True

                            if found == False:
                                scope.append(
                                    "for (int it1 = 0; it1 <" + str(var.size) + "; it1++) {"
                                )
                                scope.append(
                                    "    "
                                    + assigned_variable
                                    + "[it1] = "
                                    + op_af.clean_spaces(
                                        re.sub(
                                            "([(){}, +-])" + var.name + "( *[+\-*/]+)",
                                            "\g<1>" + var.name + "[it1]\g<2>",
                                            assigned,
                                        )
                                    )
                                    + ";"
                                )
                                scope.append("}")
                                assigned = ""
                                for v in range(len(args)):
                                    if assigned_variable == args[v].name:
                                        args[v].size = var.size
                                        args[v].type = var.type
                                        if args[v].size != 0:
                                            args[v].type += "*"
                                            print("u")
                                for v in range(len(scope_variables)):
                                    if assigned_variable == scope_variables[v].name:
                                        scope_variables[v].size = var.size
                                        scope_variables[v].type = var.type
                                        if scope_variables[v].size != 0:
                                            scope_variables[v].type += "*"
                                            print("v")

                if condition[i].startswith("T("):
                    condition[i] = split_line[0] + " = " + condition[i]
                else:
                    condition[i] = re.sub("\)$", ", " + split_line[0] + ")", condition[i])
            scope.append("if (" + op_af.clean_spaces(condition[0]) + ") {")
            scope.append("    " + op_af.clean_spaces(condition[1]) + ";")
            scope.append(op_af.clean_spaces("}"))
            scope.append(op_af.clean_spaces("else") + " {")
            scope.append("    " + op_af.clean_spaces(condition[2]) + ";")
            scope.append(op_af.clean_spaces("}"))

            return return_line, scope_variables, scope, inside_comment
        elif value.startswith(st7):
            value = re.sub("tf.concat\( *\[(.*)\] *, *axis.*", "\g<1>", value)
            value = op_af.clean_spaces(value)
            var_list = value.split(",")
            var_length = []
            conc_size = 0
            unknown = False
            type_value = 0
            conc_type = "int"
            for var in var_list:
                for arg in args:
                    if var == arg.name:
                        c_size = 0
                        if arg.size > 0:
                            c_size += arg.size
                        elif arg.size == 0:
                            c_size += 1
                        else:
                            c_size = 0
                            unknown = True
                        if arg.type.startswith("T"):
                            conc_type = "T"
                        elif arg.type.startswith(op_gc.DOUBLE_TYPE):
                            type_value += 1
                        conc_size += c_size
                        var_length.append(c_size)
                        break
                for scope_var in scope_variables:
                    if var == scope_var.name:
                        c_size = 0
                        if scope_var.size > 0:
                            c_size += scope_var.size
                        elif scope_var.size == 0:
                            c_size += 1
                        else:
                            c_size = 0
                            unknown = True
                        if scope_var.type.startswith("T"):
                            conc_type = "T"
                        elif scope_var.type.startswith(op_gc.DOUBLE_TYPE):
                            type_value += 1
                        conc_size += c_size
                        var_length.append(c_size)
                        break

            if conc_type != "T":
                if type_value > 0:
                    conc_type = op_gc.DOUBLE_TYPE

            if unknown == False:

                for arg in args:
                    if assigned_variable == arg.name:
                        arg.type = conc_type
                        if conc_size > 1:
                            arg.size = conc_size
                            arg.type += "*"
                        elif arg.size == 1:
                            arg.size = 0
                        break
                for scope_var in scope_variables:
                    if assigned_variable == scope_var.name:
                        scope_var.type = conc_type
                        if conc_size > 1:
                            scope_var.size = conc_size
                        elif scope_var.size == 1:
                            scope_var.size = 0
                        break
                i = 0
                while i < conc_size:
                    for j in range(len(var_list)):
                        newline = ""
                        if var_length[j] == 1:
                            newline = assigned_variable + "[" + str(i) + "] = " + var_list[j] + ";"
                        scope.append(newline)
                        i += 1
            else:
                scope.append(line)
            return return_line, scope_variables, scope, inside_comment
        elif value.startswith(st8):
            return return_line, scope_variables, scope, inside_comment
        elif value.startswith(st9):
            custom_type = op_af.convert_type(value.split("dtype=")[1][:-2])
            value = re.sub("[\[\]]*", "", value)
            value = re.sub(", *dtype.*", "", value)
            value = re.sub("tf.constant\(", "", value)
            sp = op_af.clean_spaces(value).split(",")
            custom_type = op_gc.DOUBLE_TYPE

            custom_size = len(sp)
            is_array = True
            newLine = "const " + custom_type + " " + assigned_variable + "[] = {"

            for v in range(len(sp) - 1):
                newLine += sp[v] + ","
            newLine += sp[-1] + "};"

            scope.append(newLine)
            value = ""
        elif value.startswith(st10):
            value = re.sub("tf.einsum\(", "", value)
            value = re.sub("tf.reshape\(([a-zA-Z0-9_]*) *,.*\)", "\g<1>", value)

            pattern = ""
            inside_quot = False

            for letter in value:
                if letter == '"':
                    inside_quot = not inside_quot
                elif letter == "," and inside_quot == False:
                    break
                else:
                    pattern += letter

            value = re.sub('"' + pattern + '" *,', "", value)

            final_indices = op_af.clean_spaces(pattern.split("->")[1])
            pattern = op_af.clean_spaces(pattern.split("->")[0])
            initial_indices = pattern.split(",")
            indices = []
            distinct_indices = []

            matrices = op_af.clean_spaces(value).split(",")

            for letter in pattern:
                already_there = False
                if letter != "," and letter != final_indices:
                    for x in distinct_indices:
                        if letter == x:
                            already_there = True
                    if already_there == False:
                        distinct_indices.append(letter)

            value = ""

            prov_len = "2"
            for x in scope_variables:
                if x.name == "denom":
                    prov_len = str(x.size)

            for m in range(len(matrices)):
                temp_arr = []
                for n in initial_indices[m]:
                    if n in final_indices:
                        temp_arr.append("0")
                    else:
                        temp_arr.append(n)
                indices.append(temp_arr)

                index = ""

                for idx in distinct_indices:
                    ind = len(temp_arr) - 1
                    prod = 1
                    while ind >= 0:
                        if temp_arr[ind] != "0":
                            if temp_arr[ind] == idx:
                                index += idx
                                if prod > 1:
                                    index += "*" + str(prod) + "+"
                            prod *= int(prov_len)
                        ind -= 1

                matrices[m] = re.sub(
                    "([^.]+[a-zA-Z0-9]+)([() +\-*/])", "\g<1>[" + index + "]\g<2>", matrices[m]
                )
                matrices[m] = re.sub("([^.]+[a-zA-Z0-9]+)$", "\g<1>[" + index + "]", matrices[m])

                value += matrices[m]
                value += " * "

            value = value[:-3]

            value = op_af.convert_grammar(value)

            custom_type = op_gc.DOUBLE_TYPE
            scope.append(custom_type + " " + assigned_variable + " = 0;")
            spacing = ""

            for distinct_index in distinct_indices:
                scope.append(
                    spacing
                    + "for (int "
                    + distinct_index
                    + " = 0; "
                    + distinct_index
                    + " < "
                    + prov_len
                    + "; "
                    + distinct_index
                    + "++) {"
                )
                spacing += "    "

            scope.append(spacing + assigned_variable + " += (" + value + ").real();")

            for x in range(len(distinct_indices)):
                spacing = spacing[:-4]
                scope.append(spacing + "}")

            value = ""

        assigned = op_af.convert_grammar(value)

        for var in args + scope_variables:
            found = False
            if var.size != 0 and var.size != -1:
                match = re.search("[+\-*/ ]" + var.name + " *[,(){};]+", assigned)
                if match != None:
                    print(var.name, var.size, match, assigned)
                match = re.search("[(){}, +-]" + var.name + " *[+\-*/]+", assigned)
                if match != None:
                    for var2 in args + scope_variables:
                        match = re.search(
                            "[(){}, +-]" + var.name + " *[+\-*/]+ *" + var2.name + " *[,(){};]+",
                            assigned,
                        )
                        if match != None:
                            if var2.size != 0 and var2.size != -1:
                                found = True

                    if found == False:
                        scope.append("for (int it1 = 0; it1 <" + str(var.size) + "; it1++) {")
                        scope.append(
                            "    "
                            + assigned_variable
                            + "[it1] = "
                            + op_af.clean_spaces(
                                re.sub(
                                    "([(){}, +-])" + var.name + "( *[+\-*/]+)",
                                    "\g<1>" + var.name + "[it1]\g<2>",
                                    assigned,
                                )
                            )
                            + ";"
                        )
                        scope.append("}")
                        for v in range(len(args)):
                            if assigned_variable == args[v].name:
                                args[v].size = var.size
                                if op_af.clean_spaces(assigned).startswith("T("):
                                    args[v].type = "T"
                                else:
                                    args[v].type = var.type
                                if args[v].size != 0:
                                    args[v].type += "*"
                                    # print('u', args[v].name, var.type)
                        for v in range(len(scope_variables)):
                            if assigned_variable == scope_variables[v].name:
                                scope_variables[v].size = var.size
                                if op_af.clean_spaces(
                                    re.sub(
                                        "([(){}, +-])" + var.name + "( *[+\-*/]+)",
                                        "\g<1>" + var.name + "[it1]\g<2>",
                                        assigned,
                                    )
                                ).startswith("T("):
                                    scope_variables[v].type = "T"
                                else:
                                    scope_variables[v].type = var.type
                                if scope_variables[v].size != 0:
                                    # scope_variables[v].type += '*'
                                    print("v")
                        assigned = ""

        if already_defined == False:
            if is_array == False:
                custom_size = 0
            if assigned.startswith("{"):
                if split_line[0].endswith("[]") == False:
                    split_line[0] += "[]"
                curly_br_count = 0
                br_count = 0
                custom_size = 1
                for letter in assigned:
                    if letter == "{":
                        curly_br_count += 1
                    elif letter == "}":
                        curly_br_count -= 1
                    elif letter == "(":
                        br_count += 1
                    elif letter == ")":
                        br_count -= 1
                    elif letter == "[":
                        br_count += 1
                    elif letter == "]":
                        br_count -= 1
                    elif letter == "," and br_count == 0 and curly_br_count == 1:
                        custom_size += 1
                comment = "array of size " + str(custom_size)
            scope_variables.append(
                op_cl.Argument(assigned_variable, custom_type, custom_size, False, [])
            )  # need to define size
            return_line += custom_type + is_arrayy + " "
        return_line += split_line[0] + " = " + op_af.clean_spaces(assigned)

        if op_af.clean_spaces(assigned) == "":
            return_line = ""

    if return_line != "":
        return_line += ";"

    if comment != "":
        return_line += "//" + comment

    return return_line, scope_variables, scope, inside_comment
