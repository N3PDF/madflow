from jinja2 import Template

import subprocess
import re

import madflow.wavefunctions_flow
from madflow.makefile_template import write_makefile
from madflow.op_constants import * # to be removed
from madflow.op_global_constants import *
from madflow.op_aux_functions import *
from madflow.op_classes import *
from madflow.op_write_templates import *


def read_file_from_source(function_list, file_source, signatures, signature_variables):
    f = open(file_source, "r")
    line = f.readline()
    while line != '':
        if clean_args(line).startswith('@tf.function'):
            signature_line = line
            line = f.readline()
            l = line
            i = 0
            while l.endswith('):\n') == False:
                if i != 0:
                    line += l[:-1]
                l = f.readline()
                i += 1
            line = re.sub(' *def ', '', line) # cut "def " from the line
            f_name = grab_function_name(line)
            
            already_defined = False
            for func in function_list:
                if f_name == func.name:
                    already_defined = True
                    break
            
            if already_defined == False:
                f_type = 'void'
                args = []
                scope = []
                scope_args = []
                args = grab_function_arguments(line, f, f_name, args, signatures, signature_variables, signature_line)
                args, f_type = grab_function_return(line, f, f_name, f_type, args)
                scope, scope_args = grab_function_scope(f, scope, scope_args, args, f_type)
                new_function = function(f_type, f_name, args, scope, scope_args, 'template <typename T>')
                function_list.append(new_function)
        
        line = f.readline()
    return function_list

def extract_matrix_from_file(function_list, file_source, signatures, signature_variables):
    f = open(file_source, "r")
    line = f.readline()
    while line != '':
        if clean_args(line).startswith('@tf.function'):
            signature_line = line
            line = f.readline()
            l = line
            i = 0
            while l.endswith('):\n') == False:
                if i != 0:
                    line += l[:-1]
                l = f.readline()
                i += 1
            line = re.sub(' *def ', '', line) # cut "def " from the line
            f_name = grab_function_name(line)
            
            already_defined = False
            for func in function_list:
                if f_name == func.name:
                    already_defined = True
                    break
            
            if already_defined == False and f_name == 'matrix':
                f_type = 'void'
                args = []
                scope = []
                scope_args = []
                args = grab_function_arguments(line, f, f_name, args, signatures, signature_variables, signature_line)
                args, f_type = grab_function_return(line, f, f_name, f_type, args)
                scope, scope_args = grab_function_scope(f, scope, scope_args, args, f_type)
                new_function = function(f_type, f_name, args, scope, scope_args, 'template <typename T>')
                function_list.append(new_function)
        
        line = f.readline()
    return function_list

def grab_function_name(line):
    return line.split("(", 1)[0]

def grab_function_arguments(line, f, f_name, args, signatures, signature_variables, signature_line):
    line = line.split(")", 1)[0]
    line = line[len(f_name)+1:]
    split_args = clean_args(line).split(",")
    
    j = -1
    for i in range(len(split_args)):
        if split_args[i] == 'self':
            j = i
            break
    if j != -1:
        del split_args[j]
    
    split_types = []
    split_sizes = []
    split_tensors = []
    split_slices = []
    sig_list = []
    signature_line = signature_line.split('@tf.function(')[1]
    signature_line = signature_line.split(')')[0]
    signature_line = clean_args(signature_line).split('input_signature=')[1]
    if signature_line.startswith('['):
        s = get_signature(signature_line)
        sig_list.append(s)
    else:
        for sv in signature_variables:
            if sv.name == signature_line:
                if len(sv.signature_list) == len(split_args):
                    sig_list = sv.signature_list
                    
    
    for a in sig_list:
        t = a.type
        if a.size != 0:
            t += '*'
        split_types.append(t)
        split_sizes.append(a.size)
        split_tensors.append(a.tensor)
        split_slices.append(a.slice)
    
    for i in range(len(split_args)):
        #print(split_args[i])
        split_args[i] = clean_args(split_args[i])
        args.append(argument(split_args[i], split_types[i], split_sizes[i], split_tensors[i], split_slices[i]))
    
    return args

def grab_function_return(line, f, f_name, f_type, args):
    
    args.append(argument('ret', doubleType, -1, False, []))
    return args, f_type
    
    split_types = []
    l = f.readline()
    comment_count = 0
    while l != '' and comment_count < 1:
        if clean_args(l) == '"""':
            comment_count += 1
        if l.startswith('    -------'):
            l = f.readline()
            if "shape=(" not in l:
                l = l[:-1]
                l += f.readline()
            splitted = l.split("shape=(")
            splitted[1] = clean_args(splitted[1])
            new_type = 'T'
            if splitted[1].startswith(')') == False and splitted[1].startswith('None)') == False:
                new_type += '*'
            else:
                new_type += '&'
            split_types.append(new_type)
            break
        l = f.readline()
    args.append(argument('ret', split_types[0], -1))
    return args, f_type

def grab_function_scope(f, scope, scope_args, args, f_type):
    
    line = f.readline()
    function_scope = []
    function_return = ''
    
    match = re.search('^ *return[ ]+', line)
    while clean_args(line).startswith('return') == False:
        function_scope.append(line)
        match = re.search('^ *return[ ]+', line)
        line = f.readline()
    while clean_args(line) != '':
        function_return += line
        line = f.readline()
    
    args[-1].name = grab_return_variable_name(function_return)
    scope, scope_args = parse_function_scope(function_scope, scope, scope_args, args, f_type)
    scope, scope_args = parse_function_return(function_return, scope, scope_args, args, f_type)
    
    return scope, scope_args

def grab_return_variable_name(function_return):
    ret_name = 'out_final'
    function_return = clean_args(function_return)[len('return'):]
    st1 = 'tf.stack('
    st2 = 'tf.transpose('
    st3 = 'tf.reshape(tf.stack('
    if function_return.startswith(st1):
        ret_name = function_return[len(st1):].split(')')[0].split(',')[0]
    elif function_return.startswith(st2):
        ret_name = function_return[len(st2):].split(')')[0].split(',')[0]
    elif function_return.startswith(st3):
        ret_name = function_return[len(st3):].split(')')[0].split(',')[0]
    return ret_name
    
def parse_function_return(function_return, scope, scope_args, args, f_type):
    
    function_return = re.sub('return', args[-1].name+' =', function_return)
    function_return = re.sub('#[^\n]*\n', '', function_return)
    inside_comment = False
    new_line, scope_args, scope, inside_comment = parse_line(function_return, args, scope_args, scope, inside_comment)
    
    scope.append(new_line)
    
    return scope, scope_args


def parse_function_scope(function_scope, scope, scope_args, args, f_type):
    if len(function_scope) == 0:
        return scope, scope_args
    i = 0
    line = function_scope[i]
    i += 1
    inside_comment = False
    while i <= len(function_scope):
        brackets_count = 0
        brackets_count = count_brackets(line, brackets_count)
        while brackets_count > 0 and i < len(function_scope):
            l = function_scope[i]
            i += 1
            brackets_count = count_brackets(l, brackets_count)
            line += l
        new_line, scope_args, scope, inside_comment = parse_line(line, args, scope_args, scope, inside_comment)
        scope.append(new_line)
        #print(i, len(function_scope))
        if i < len(function_scope):
            line = function_scope[i]
        i += 1
    return scope, scope_args


def clean_pointer(var_type):
    var_type = re.sub('[&*]*', '', var_type)
    return var_type

def parse_line(line, args, scope_variables, scope, inside_comment):
    return_line = ""
    
    if inside_comment == True:
        match = re.search('"""', line)
        if match != None:
            s = line.split('"""', 1)
            inside_comment = False
            scope.append(s[0] + '*/')
            line = s[1]
        else:
            scope.append(line)
            return return_line, scope_variables, scope, inside_comment
    
    s = line.split('#', 1)
    line = s[0]
    is_arrayy = ''
    is_array = False
    variable_type = doubleType # not needed
    comment = ''
    defined_in_args = False
    defined_in_scope = False
    position = -1
    if len(s) > 1:
        comment = s[1]
        
    if clean_args(line).startswith('"""'):
        line = re.sub('"""', '/*', line)
        scope.append(line)
        inside_comment = True
        return return_line, scope_variables, scope, inside_comment
    
    if clean_args(line) != '':
        line = re.sub('([a-zA-Z0-9_()[\]{} ])=', '\g<1> =', line, 1)
        split_line = line.split(' = ')
        if clean_args(line).startswith('return'):
            return return_line, scope_variables, scope, inside_comment
            
        for i in range(len(split_line) - 1):
            split_line.insert(2 * i + 1, '=')
        split_line[0] = clean_args(split_line[0])
        assigned_variable = clean_index(split_line[0])
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
        
        if split_line[0].endswith(']'):
            is_array = True
        
        value = split_line[2]
        square_brackets = 0
        assigned = split_line[2]
        custom_type = doubleType
        custom_size = -1
        while clean_args(value).startswith('['):
            value = value[1:]
            square_brackets += 1
        if square_brackets > 0:
            is_arrayy = '*'
            is_array = True
            sz = clean_args(value.split(']')[-1])[1:]
            if sz != '':
                custom_size = int(sz)
            
            value = value.split(']', 1)[0]
            if sz != '':
                definition = ''
                if already_defined == False:
                    scope.append(convert_grammar(custom_type + " " + split_line[0] + '[' + sz + '];'))
                scope.append('for (int it1 = 0; it1 < ' + str(sz) + '; it1++) {')
                scope.append('    ' + convert_grammar(split_line[0] + '[it1] = ' + value + ';'))
                scope.append('}')
            else:
                scope.append(convert_grammar(split_line[0] + ' = ' + value + ';'))
                
            comparison_string = 'complex_tf('
            if value.startswith(comparison_string):
                custom_type = 'T'
                assigned = 'T('+ (value.split('('))[1].split(')')[0] +')'
                
            comment = 'array of size ' + str(custom_size)
            value = ''
            
            
            if already_defined == True and sz != '':
                for i in range(len(args)):
                    if assigned_variable == args[i].name:
                        args[i].size = custom_size
                        args[i].type = custom_type
                        if custom_size != 0:
                            args[i].type += '*'
                        else:
                            args[i].type += '&'
                        break
            if already_defined == True and sz != '':
                for i in range(len(scope_variables)):
                    if assigned_variable == scope_variables[i].name:
                        scope_variables[i].size = custom_size
                        break
            if already_defined == True:
                return return_line, scope_variables, scope, inside_comment
        
        if split_line[0] == clean_args(assigned): # remove things like a = a;
            return_line = ''
            return return_line, scope_variables, scope, inside_comment
            
        if 'tf.stack([' in clean_args(value):
            reassignment = re.search('[()[\]{}+\-*/, ]'+assigned_variable+'[()[\]{}+\-*/, ]', value)
            if reassignment != None:
                counter = 0
                for s in scope:
                    scope[counter] = re.sub('([()[\]{}+\-*/, ]*'+assigned_variable+')([()[\]{}+\-*/, ]*)', '\g<1>_\g<2>', s)
                    counter += 1
                
                value = re.sub('([()[\]{}+\-*/, ]*'+assigned_variable+')([()[\]{}+\-*/, ]*)', '\g<1>_\g<2>', value)
                already_defined = False
                
                counter = 0
                for v in scope_variables:
                    if v.name == assigned_variable:
                        scope_variables[counter].name += '_'
                    counter += 1
        
        type_value = 0
        if custom_type != 'T':
            for v in scope_variables:
                reassignment = re.search('[()[\]{}+\-*/, \n]'+v.name+'[()[\]{}+\-*/, \n]', value)
                if reassignment == None:
                    reassignment = re.search('^'+v.name+'[()[\]{}+\-*/, \n]', value)
                if reassignment == None:
                    reassignment = re.search('[()[\]{}+\-*/, \n]'+v.name+'$', value)
                if reassignment == None:
                    reassignment = re.search('^'+v.name+'$', value)
                if reassignment != None:
                    if v.type.startswith('T'):
                        custom_type = 'T'
                        break
                    elif v.type.startswith(doubleType):
                        type_value += 1
        if custom_type != 'T':
            for v in args:
                reassignment = re.search('[()[\]{}+\-*/, \n^]'+v.name+'[()[\]{}+\-*/, \n$]', value)
                if reassignment == None:
                    reassignment = re.search('^'+v.name+'[()[\]{}+\-*/, \n]', value)
                if reassignment == None:
                    reassignment = re.search('[()[\]{}+\-*/, \n]'+v.name+'$', value)
                if reassignment == None:
                    reassignment = re.search('^'+v.name+'$', value)
                if reassignment != None:
                    if v.type.startswith('T'):
                        custom_type = 'T'
                        break
                    elif v.type.startswith(doubleType):
                        type_value += 1
            reassignment = re.search('float_me\(', value)
            if reassignment != None:
                type_value += 1
        if custom_type != 'T':
            if type_value == 0:
                custom_type = 'int'
        
        st1 = 'complex_tf('
        st2 = 'tf.expand_dims('
        st3 = 'int_me('
        st4 = 'tf.stack('
        st5 = 'tf.where('
        st6 = 'tf.cond('
        st7 = 'tf.concat('
        st8 = 'tf.transpose('
        st9 = 'tf.constant('
        st10 = 'tf.einsum('
        
        if '!=' in value or '==' in value:
            if ('tf.where' in value) == False:
                custom_type = 'const bool'
        if st1 in value:
            custom_type = 'T'
            if already_defined == True and is_array == False:
                if defined_in_args == True:
                    args[position].size = 0
                    args[position].type = 'T&'
                elif defined_in_scope == True:
                    scope_variables[position].size = 0
                    scope_variables[position].type = 'T'
        
        if value.startswith(st1):
            custom_type = 'T'
            
            match = re.search('tf.stack\(', value)
            
            if match != None:
                
                comp = ['', '']
                real_part = True
                
                br_count = 0
                
                for letter in value[len(st1):]:
                    if letter == '(':
                        br_count += 1
                    elif letter == ')':
                        br_count -= 1
                    elif letter == ',' and br_count == 0:
                        real_part = False
                    
                    if real_part == True:
                        comp[0] += letter
                    else:
                        comp[1] += letter
                
                
                comp[1] = (comp[1])[1:-2]
                
                for e in range(len(comp)):
                    comp[e] = re.sub('tf.stack\(\[(.*)] *, *axis.*', '\g<1>', comp[e])
                    comp[e] = comp[e].split(',')
                    
                
                size_of_stack = 0
                
                for el in comp:
                    if len(el) > size_of_stack:
                        size_of_stack = len(el)
                
                
                scope.append(custom_type + is_arrayy + " " + split_line[0] + '[' + str(size_of_stack) + '];')
                
                for e in range(len(comp)):
                    if len(comp[e]) != size_of_stack:
                        for i in range(1, size_of_stack):
                            comp[e].append(comp[e][0])
                
                for e in range(len(comp)):
                    for i in range(size_of_stack):
                        comp[e][i] = convert_grammar(comp[e][i])
                for e in range(size_of_stack):
                    scope.append(split_line[0] + '[' + str(e) + '] = T(' + comp[0][e] + ',' + comp[1][e] + ');')
                
                value = ''
            
        elif value.startswith(st2):
            value = value[len(st2):]
            
            splitted = value.split("dtype=")
            if len(splitted) == 2:
                custom_type = convert_type(splitted[1].split(')')[0])
            
            if value.startswith(st1):
                custom_type = 'T'
            
            br_count = 1
            has_comma = False
            
            vv = ''
            
            if custom_type == 'T':
                vv = 'T('
            
            if value.startswith('tf.zeros_like('):
                vv += '0,0'
            elif value.startswith(st1):
                value = value[len(st1):]
                for letter in value:
                    if letter == '(':
                        br_count += 1
                    elif letter == ')':
                        br_count -= 1
                    elif letter == ',':
                        has_comma = True
                    if br_count == 0 and has_comma == True:
                        break
                    vv += letter
            elif value.startswith('float_me('): # here
                value = value[len('float_me('):]
                for letter in value:
                    if letter == '(':
                        br_count += 1
                    elif letter == ')':
                        br_count -= 1
                    elif letter == '[':
                        vv += '{'
                    elif letter == ']':
                        vv += '}'
                    elif letter == ',':
                        has_comma = True
                        vv += letter
                    else:
                        vv += letter
                    if br_count == 0 and has_comma == True:
                        break
            else:
                vv += value.split(',')[0]
            if custom_type == 'T':
                vv += ')'
            value = vv
        elif value.startswith(st3):
            custom_type = 'int'
        elif value.startswith(st4):
            value = re.sub('\[:, (\d+)\]', '[\g<1>]', value)
            if already_defined == False:
                split_line[0] += '[]'
                is_array = True
        elif value.startswith(st5) == True or value.startswith(st6) == True:
            if value.startswith(st6):
                value = re.sub('tf.cond', 'tf.where', value)
                value = re.sub('lambda: ', '', value)
            value = value[len(st5):]
            br_value = 1
            comma_num = 0
            vv = ''
            condition = []
            for letter in value:
                if letter == '(':
                    br_value += 1
                elif letter == ')':
                    br_value -= 1
                vv += letter
                if br_value == 1 and letter == ',':
                    condition.append(vv[:-1])
                    vv = ''
                    comma_num += 1
                if br_value == 0 and comma_num == 2:
                    break
            condition.append(vv[:-1])
            
            
            for i in range(3):
                condition[i] = convert_grammar(condition[i])
            
            
            if already_defined == False:
                if clean_args(condition[1]).startswith('T('):
                    custom_size = 0
                else:
                    custom_size = -1
                scope_variables.append(argument(assigned_variable, custom_type, custom_size, False, [])) # need to define size
                scope.append(custom_type + is_arrayy + " " + split_line[0] + ';')
            
            for i in range(1,3):
                condition[i] = clean_args(condition[i])
                
                for var in args + scope_variables:
                    if var.size != 0 and var.size != -1:
                        match = re.search('[+\-*/ ]' + var.name + ' *[,(){};]+', condition[i])
                        match = re.search('[(){}, +-]' + var.name + ' *[+\-*/]+', condition[i])
                        if match != None:
                            for var2 in args + scope_variables:
                                match = re.search('[(){}, +-]' + var.name + ' *[+\-*/]+ *' + var2.name + ' *[,(){};]+', condition[i])
                                if match != None:
                                    if var2.size != 0 and var2.size != -1:
                                        
                                        scope.append(re.sub('[&*]*', '', var.type) + ' _' + var.name + '[' + str(var.size) + '];')
                                        scope.append('for (int it1 = 0; it1 <' + str(var.size) + '; it1++) {')
                                        scope.append('    _' + var.name + '[it1] = ' + var.name + '[it1] ' + re.sub('.*[(){}, +-]' + var.name + '( *[+\-*/]+ *)' + var2.name + ' *[,(){};]+.*', '\g<1>', condition[i]) + var2.name + '[it1];')
                                        scope.append('}')
                                        condition[i] = re.sub('([(){}, +-])' + var.name + ' *[+\-*/]+ *' + var2.name + '( *[,(){};]+)', '\g<1>_' + var.name + '\g<2>', condition[i])
                                        found = True
                     
                            if found == False:
                                scope.append('for (int it1 = 0; it1 <' + str(var.size) + '; it1++) {')
                                scope.append('    ' + assigned_variable + '[it1] = ' + clean_args(re.sub('([(){}, +-])' + var.name + '( *[+\-*/]+)', '\g<1>' + var.name + '[it1]\g<2>', assigned)) + ';')
                                scope.append('}')
                                assigned = ''
                                for v in range(len(args)):
                                    if assigned_variable == args[v].name:
                                        args[v].size = var.size
                                        args[v].type = var.type
                                        if args[v].size != 0:
                                            args[v].type += '*'
                                            print('u')
                                for v in range(len(scope_variables)):
                                    if assigned_variable == scope_variables[v].name:
                                        scope_variables[v].size = var.size
                                        scope_variables[v].type = var.type
                                        if scope_variables[v].size != 0:
                                            scope_variables[v].type += '*'
                                            print('v')
                                            
                
                if condition[i].startswith('T('):
                    condition[i] = split_line[0]+' = '+condition[i]
                else:
                    condition[i] = re.sub('\)$', ', '+split_line[0]+')', condition[i])
            scope.append('if ('+clean_args(condition[0])+') {')
            scope.append('    '+clean_args(condition[1])+';')
            scope.append(clean_args('}'))
            scope.append(clean_args('else')+' {')
            scope.append('    '+clean_args(condition[2])+';')
            scope.append(clean_args('}'))
            
            
            return return_line, scope_variables, scope, inside_comment
        elif value.startswith(st7):
            value = re.sub('tf.concat\( *\[(.*)\] *, *axis.*', '\g<1>', value)
            value = clean_args(value)
            var_list = value.split(',')
            var_length = []
            conc_size = 0
            unknown = False
            type_value = 0
            conc_type = 'int'
            for var in var_list:
                for i in range(len(args)):
                    if var == args[i].name:
                        c_size = 0
                        if args[i].size > 0:
                            c_size += args[i].size
                        elif args[i].size == 0:
                            c_size += 1
                        else:
                            c_size = 0
                            unknown = True
                        if args[i].type.startswith('T'):
                            conc_type = 'T'
                        elif args[i].type.startswith(doubleType):
                            type_value += 1
                        conc_size += c_size
                        var_length.append(c_size)
                        break
                for i in range(len(scope_variables)):
                    if var == scope_variables[i].name:
                        c_size = 0
                        if scope_variables[i].size > 0:
                            c_size += scope_variables[i].size
                        elif scope_variables[i].size == 0:
                            c_size += 1
                        else:
                            c_size = 0
                            unknown = True
                        if scope_variables[i].type.startswith('T'):
                            conc_type = 'T'
                        elif scope_variables[i].type.startswith(doubleType):
                            type_value += 1
                        conc_size += c_size
                        var_length.append(c_size)
                        break
            
            if conc_type != 'T':
                if type_value > 0:
                    conc_type = doubleType
            
            if unknown == False:
            
                for i in range(len(args)):
                    if assigned_variable == args[i].name:
                        args[i].type = conc_type
                        if conc_size > 1:
                            args[i].size = conc_size
                            args[i].type += '*'
                        elif args[i].size == 1:
                            args[i].size = 0
                        break
                for i in range(len(scope_variables)):
                    if assigned_variable == scope_variables[i].name:
                        scope_variables[i].type = conc_type
                        if conc_size > 1:
                            scope_variables[i].size = conc_size
                        elif scope_variables[i].size == 1:
                            scope_variables[i].size = 0
                        break
                i = 0
                while i < conc_size:
                    for j in range(len(var_list)):
                        newline = ''
                        if var_length[j] == 1:
                            newline = assigned_variable+'['+str(i)+'] = '+var_list[j]+';'
                        scope.append(newline)
                        i += 1
            else:
                scope.append(line)
            return return_line, scope_variables, scope, inside_comment
        elif value.startswith(st8):
            return return_line, scope_variables, scope, inside_comment
        elif value.startswith(st9):
            custom_type = convert_type(value.split('dtype=')[1][:-2])
            value = re.sub('[\[\]]*', '', value)
            value = re.sub(', *dtype.*', '', value)
            value = re.sub('tf.constant\(', '', value)
            sp = clean_args(value).split(',')
            custom_type = doubleType
            
            custom_size = len(sp)
            is_array = True
            newLine = 'const ' + custom_type + ' ' + assigned_variable + '[] = {'
            
            for v in range(len(sp) - 1):
                newLine += sp[v] + ','
            newLine += sp[-1] + '};'
            
            scope.append(newLine)
            value = ''
        elif value.startswith(st10):
            value = re.sub('tf.einsum\(', '', value)
            value = re.sub('tf.reshape\(([a-zA-Z0-9_]*) *,.*\)', '\g<1>', value)
            
            pattern = ''
            inside_quot = False
            
            for letter in value:
                if letter == '"':
                    inside_quot = not inside_quot
                elif letter == ',' and inside_quot == False:
                    break
                else:
                    pattern += letter
            
            value = re.sub('"' + pattern + '" *,' , '', value)
            
            final_indices = clean_args(pattern.split('->')[1])
            pattern = clean_args(pattern.split('->')[0])
            initial_indices = pattern.split(',')
            indices = []
            distinct_indices = []
            
            matrices = clean_args(value).split(',')
            
            for letter in pattern:
                already_there = False
                if letter != ',' and letter != final_indices:
                    for x in distinct_indices:
                        if letter == x:
                            already_there = True
                    if already_there == False:
                        distinct_indices.append(letter)
            
            value = ''
            
            prov_len = '2'
            for x in scope_variables:
                if x.name == 'denom':
                    prov_len = str(x.size)
            
            for m in range(len(matrices)):
                temp_arr = []
                for n in initial_indices[m]:
                    if n in final_indices:
                        temp_arr.append('0')
                    else:
                        temp_arr.append(n)
                indices.append(temp_arr)
                
                index = ''
                
                for idx in distinct_indices:
                    ind = len(temp_arr) - 1
                    prod = 1
                    while ind >= 0:
                        if temp_arr[ind] != '0':
                            if temp_arr[ind] == idx:
                                index += idx
                                if prod > 1:
                                    index += '*' + str(prod) + '+'
                            prod *= int(prov_len)
                        ind -= 1
                
                matrices[m] = re.sub('([^.]+[a-zA-Z0-9]+)([() +\-*/])', '\g<1>[' + index + ']\g<2>', matrices[m])
                matrices[m] = re.sub('([^.]+[a-zA-Z0-9]+)$', '\g<1>[' + index + ']', matrices[m])
                
                value += matrices[m]
                value += ' * '
            
            value = value[:-3]
            
            
            value = convert_grammar(value)
            
            custom_type = doubleType
            scope.append(custom_type + ' ' + assigned_variable + ' = 0;')
            spacing = ''
            
            
            for x in range(len(distinct_indices)):
                scope.append(spacing + 'for (int ' + distinct_indices[x] + ' = 0; ' + distinct_indices[x] + ' < ' + prov_len + '; ' + distinct_indices[x] + '++) {')
                spacing += '    '
            
            
            
            scope.append(spacing + assigned_variable + ' += (' + value + ').real();')
            
            for x in range(len(distinct_indices)):
                spacing = spacing[:-4]
                scope.append(spacing + '}')
            
            value = ''
        
        assigned = convert_grammar(value)
        
        for var in args + scope_variables:
            found = False
            if var.size != 0 and var.size != -1:
                match = re.search('[+\-*/ ]' + var.name + ' *[,(){};]+', assigned)
                if match != None:
                    print(var.name, var.size, match, assigned)
                match = re.search('[(){}, +-]' + var.name + ' *[+\-*/]+', assigned)
                if match != None:
                    for var2 in args + scope_variables:
                        match = re.search('[(){}, +-]' + var.name + ' *[+\-*/]+ *' + var2.name + ' *[,(){};]+', assigned)
                        if match != None:
                            if var2.size != 0 and var2.size != -1:
                                found = True
                            
                    if found == False:
                        scope.append('for (int it1 = 0; it1 <' + str(var.size) + '; it1++) {')
                        scope.append('    ' + assigned_variable + '[it1] = ' + clean_args(re.sub('([(){}, +-])' + var.name + '( *[+\-*/]+)', '\g<1>' + var.name + '[it1]\g<2>', assigned)) + ';')
                        scope.append('}')
                        for v in range(len(args)):
                            if assigned_variable == args[v].name:
                                args[v].size = var.size
                                if clean_args(assigned).startswith('T('):
                                    args[v].type = 'T'
                                else:
                                    args[v].type = var.type
                                if args[v].size != 0:
                                    args[v].type += '*'
                                    #print('u', args[v].name, var.type)
                        for v in range(len(scope_variables)):
                            if assigned_variable == scope_variables[v].name:
                                scope_variables[v].size = var.size
                                if clean_args(re.sub('([(){}, +-])' + var.name + '( *[+\-*/]+)', '\g<1>' + var.name + '[it1]\g<2>', assigned)).startswith('T('):
                                    scope_variables[v].type = 'T'
                                else:
                                    scope_variables[v].type = var.type
                                if scope_variables[v].size != 0:
                                    #scope_variables[v].type += '*'
                                    print('v')
                        assigned = ''
        
        if already_defined == False:
            if is_array == False:
                custom_size = 0
            if assigned.startswith('{'):
                if split_line[0].endswith('[]') == False:
                    split_line[0] += '[]'
                curly_br_count = 0
                br_count = 0
                custom_size = 1
                for letter in assigned:
                    if letter == '{':
                        curly_br_count += 1
                    elif letter == '}':
                        curly_br_count -= 1
                    elif letter == '(':
                        br_count += 1
                    elif letter == ')':
                        br_count -= 1
                    elif letter == '[':
                        br_count += 1
                    elif letter == ']':
                        br_count -= 1
                    elif letter == ',' and br_count == 0 and curly_br_count == 1:
                        custom_size += 1
                comment = 'array of size ' + str(custom_size)
            scope_variables.append(argument(assigned_variable, custom_type, custom_size, False, [])) # need to define size
            return_line += custom_type + is_arrayy + " "
        return_line += split_line[0] + ' = ' + clean_args(assigned)
    
        if clean_args(assigned) == '':
            return_line = ''
    
    if return_line != '':
        return_line += ';'
        
    if comment != '':
        return_line += '//' + comment
    
    return return_line, scope_variables, scope, inside_comment

def count_brackets(line, brackets_count):
    for letter in line:
        if letter == '(':
            brackets_count += 1
        elif letter == ')':
            brackets_count -= 1
    return brackets_count

def clean_args(a):
    return a.translate({ ord(c): None for c in "\n " })
    
def clean_index(a):
    return a.split('[')[0]
    
def convert_type(t):
    t = clean_args(t)
    
    result = ""
    d = {
        'DTYPE': doubleType,
        'DTYPEINT': 'int',
        'DTYPECOMPLEX': 'T',
    }
    result = d.get(t, t)
    return result

def convert_grammar(value):
    value = re.sub('tf.reshape', '', value)
    value = re.sub('\[:,[ :]*(\d+)\]', '[\g<1>]', value)
    value = re.sub('float_me\(([a-zA-Z0-9[\]+\-*/. ]*)\)', '\g<1>', value)
    value = re.sub('int_me', '(int)', value)
    value = re.sub('([a-zA-Z_0-9[\]]+) *\*\* *2', '\g<1> * \g<1>', value)
    value = re.sub('([a-zA-Z_0-9[\]]+) \*\* (\d+)', 'pow(\g<1>, \g<2>)', value)
    #value = re.sub('([a-zA-Z_0-9()[\]+\-*/ ]+)// *2', '\g<1> / 2', value)
    value = re.sub('\( *([a-zA-Z_0-9[\]+\-*/ ]+)\) *// *2', '(int)(\g<1>) / 2', value)
    value = re.sub('tf.ones_like\([a-zA-Z_0-9[\]{}+\-*/=, \n]*\) *\**', '', value)
    #value = re.sub('complex_tf\(([a-zA-Z_0-9[\]{}+\-*/=,. \n]*)\)', 'T(\g<1>)', value)
    value = re.sub('tfmath\.', '', value)
    value = re.sub('minimum', 'MINIMUM', value) #hhh
    value = re.sub('maximum', 'MAXIMUM', value) #hhh
    value = re.sub('tf.math.real\(([a-zA-Z0-9_()[\] +\-*/]*)\)', '\g<1>.real()', value)
    value = re.sub('tf.math.imag\(([a-zA-Z0-9_()[\] +\-*/]*)\)', '\g<1>.imag()', value)
    value = re.sub('tf.math.conj', 'COMPLEX_CONJUGATE', value) #hhh
    value = re.sub('tf.stack\([ \n]*\[([a-zA-Z_0-9()[\]+\-*/,. ]*)] *, *axis=[0-9 \n]*\)', '{\g<1>}', value)
    value = re.sub('tf.stack\([ \n]*\[([a-zA-Z_0-9()[\]+\-*/,. ]*)][ \n]*\)', '{\g<1>}', value)
    value = re.sub('tf.stack\([ \n]*([a-zA-Z_0-9()[\]+\-*/,. ]*) *, *axis=[0-9 \n]*\)', '', value)
    value = re.sub('\(*tf.stack\([ \n]*([a-zA-Z_0-9()[\]+\-*/,. ]*) *, *\[[0-9, \n]*]\)', '', value)
    value = re.sub('complex_tf', 'T', value)
    value = re.sub('complex_me', 'T', value)
    value = re.sub('complex\(', 'T(', value)
    value = re.sub('\( *\(([a-zA-Z_0-9()[\]{}+\-*/ \n]*)\) *\)', '(\g<1>)', value)
    return value

def check_variables(counter, function_list):
    all_sizes_defined = True
    found = False
    i = 0
    
    
    for i in range(len(function_list[counter].args)):
        if (function_list[counter].args)[i].size == -1:
            all_sizes_defined = False
            break
    
    if all_sizes_defined == False:
        for j in range(len(function_list[counter].scope)):
            line = (function_list[counter].scope)[j]
            
            for k in range(len(function_list)):
                match = re.search(function_list[k].name + '\(.*' + (function_list[counter].args)[i].name, line)
                if match != None:
                    function_list = check_variables(k, function_list)
                    #print(function_list[k].args[-1].size)
                    #print(line, match, k, function_list[k].name)
                    (function_list[counter].args)[i].size = function_list[k].args[-1].size
                    (function_list[counter].args)[i].type = clean_pointer(function_list[k].args[-1].type)
                    if function_list[k].args[-1].size != 0:
                        (function_list[counter].args)[i].type += '*'
                    else:
                        (function_list[counter].args)[i].type += '&'
                    #found = False # to avoid counting multiple times
    
    i = 0
    all_sizes_defined = True
    for i in range(len(function_list[counter].scope_args)):
        if (function_list[counter].scope_args)[i].size == -1:
            all_sizes_defined = False
            break
    
    
    line_of_definition = -1
    variabe_type = ''
    new_size = -1
    
    if all_sizes_defined == False:
        for j in range(len(function_list[counter].scope)):
            line = (function_list[counter].scope)[j]
            match = re.search((function_list[counter].scope_args)[i].type + '\** ' + (function_list[counter].scope_args)[i].name + ';', line)
            if match != None:
                found = True
                line_of_definition = j
            
            if found == True:
                for k in range(len(function_list)):
                    match = re.search(function_list[k].name + '\(.*' + (function_list[counter].scope_args)[i].name, line)
                    if match != None:
                        function_list = check_variables(k, function_list)
                        new_size = function_list[k].args[-1].size
                        (function_list[counter].scope_args)[i].size = new_size
                        variabe_type = re.sub('[&\*]*', '', function_list[k].args[-1].type)
                        (function_list[counter].scope_args)[i].type = variabe_type
                        found = False # to avoid counting multiple times
    
    if variabe_type != '':
        (function_list[counter].scope)[line_of_definition] = re.sub('^[a-zA-Z0-9_]* ', variabe_type + ' ', (function_list[counter].scope)[line_of_definition])
        if new_size != 0:
            (function_list[counter].scope)[line_of_definition] = re.sub(';', '[' + str(new_size) + '];', (function_list[counter].scope)[line_of_definition])
    
    return function_list



def check_lines(counter, function_list):
    it = 0
    while it < len(function_list[counter].scope):
        line = function_list[counter].scope[it]
        
        if function_list[counter].args[-1].size == -1:
            match = re.search('^[ +\-*/,()[\]{}]*T\(', line)
            if match == None:
                l = line.split(' = ')
                if function_list[counter].args[-1].name == l[0]:
                    custom_type = 'int'
                    type_value = 0
                    value = l[1]
                    for v in function_list[counter].args + function_list[counter].scope_args:
                        reassignment = re.search('[()[\]{}+\-*/, \n]'+v.name+'[()[\]{}+\-*/, \n;]', value)
                        if reassignment == None:
                            reassignment = re.search('^'+v.name+'[()[\]{}+\-*/, \n;]', value)
                        if reassignment == None:
                            reassignment = re.search('[()[\]{}+\-*/, \n]'+v.name+'$', value)
                        if reassignment == None:
                            reassignment = re.search('^'+v.name+'$', value)
                        if reassignment != None:
                            if v.type.startswith('T'):
                                custom_type = 'T'
                                break
                            elif v.type.startswith(doubleType):
                                type_value += 1
                    
                    if custom_type != 'T' and type_value > 0:
                        custom_type = doubleType
                    
                    function_list[counter].args[-1].type = custom_type + '&'
                    function_list[counter].args[-1].size = 0
        
        ls = line.split(' = ')
        if len(ls) > 1:
            if ls[1].startswith('T('):
                value = ls[1]
                for v in function_list[counter].args + function_list[counter].scope_args:
                    reassignment = re.search('[()[\]{}+\-*/, \n]'+v.name+'[()[\]{}+\-*/, \n;]', value)
                    if reassignment == None:
                        reassignment = re.search('^'+v.name+'[()[\]{}+\-*/, \n;]', value)
                    if reassignment == None:
                        reassignment = re.search('[()[\]{}+\-*/, \n]'+v.name+'$', value)
                    if reassignment == None:
                        reassignment = re.search('^'+v.name+'$', value)
                    if reassignment != None:
                        if v.type.startswith('T'):
                            match = re.search('T\( *' + v.name +'[0-9[\]]* *\)', value)
                            if match != None:
                                if ls[0].startswith(v.name):
                                    function_list[counter].scope[it] = ''
                                    break
                        elif v.type.startswith(doubleType):
                            match = re.search('T\( *' + v.name +'[0-9[\]]* *\)', value)
                            if match != None:
                                if ls[0].startswith(v.name):
                                    for it2 in range(it):
                                        function_list[counter].scope[it2] = re.sub('([()[\]{}, +\-*/]*' + v.name + ')([()[\]{}, +\-*/;]*)', '\g<1>_\g<2>', function_list[counter].scope[it2])
                                    function_list[counter].scope[it] = 'T ' + ls[0] + ' = ' + re.sub('([()[\]{}, +\-*/]*' + v.name + ')([()[\]{}, +\-*/;]*)\);', '\g<1>_\g<2>', ls[1]) + ', 0);'
                                    function_list[counter].scope_args.append(argument(v.name, 'T', 0, False, []))
                                    for it2 in range(len(function_list[counter].args)):
                                        if v.name == function_list[counter].args[it2].name:
                                            function_list[counter].args[it2].name += '_'
                                    for it2 in range(len(function_list[counter].scope_args)):
                                        if v.name == function_list[counter].scope_args[it2].name:
                                            function_list[counter].scope_args[it2].name += '_'
                                    break
            else:
                for f in function_list:
                    match = re.search('^ *' + f.name + ' *\(', ls[1])
                    if match != None:
                        if f.type == 'void':
                            if ls[0].startswith('T') or ls[0].startswith(doubleType) or ls[0].startswith('int'):
                                for v in range(len(function_list[counter].scope_args)):
                                    if l[0].endswith(' ' + function_list[counter].scope_args[v].name):
                                        function_list[counter].scope_args[v].type = re.sub('[&*]*', '', f.args[-1].type)
                                        function_list[counter].scope_args[v].size = f.args[-1].size
                                        break
                                if f.args[-1].size > 0:
                                    function_list[counter].scope.insert(it, function_list[counter].scope_args[v].type + ' ' + function_list[counter].scope_args[v].name + '[' + str(f.args[-1].size) + '];')
                                else:
                                    function_list[counter].scope.insert(it, function_list[counter].scope_args[v].type + ' ' + function_list[counter].scope_args[v].name + ';')
                                it += 1
                                function_list[counter].scope[it] = re.sub('.* +' + function_list[counter].scope_args[v].name + ' *=', function_list[counter].scope_args[v].name + ' =', function_list[counter].scope[it])
                            function_list[counter].scope[it] = re.sub('([a-zA-Z0-9_]*) *= *(.*)\) *;', '\g<2>, \g<1>);', function_list[counter].scope[it])
        
        match = re.search('tf.concat', line)
        if match != None:
            function_list[counter].scope.remove(line)
            line = re.sub('(.*)tf.concat\( *\[(.*) *] *, *axis.*', '\g<1>\g<2>', line)
            assigned = clean_args(line.split('=')[1])
            assigned_variable = clean_args(line.split('=')[0])
            var_list = assigned.split(',')
            var_length = []
            conc_size = 0
            unknown = False
            type_value = 0
            conc_type = 'int'
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
                        if function_list[counter].args[i].type.startswith('T'):
                            conc_type = 'T'
                        elif function_list[counter].args[i].type.startswith(doubleType):
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
                        if function_list[counter].scope_args[i].type.startswith('T'):
                            conc_type = 'T'
                        elif function_list[counter].scope_args[i].type.startswith(doubleType):
                            type_value += 1
                        conc_size += c_size
                        var_length.append(c_size)
                        break
            
            if conc_type != 'T':
                if type_value > 0:
                    conc_type = doubleType
            
            if unknown == False:
            
                for i in range(len(function_list[counter].args)):
                    if assigned_variable == function_list[counter].args[i].name:
                        function_list[counter].args[i].type = conc_type
                        if conc_size > 1:
                            function_list[counter].args[i].size = conc_size
                            function_list[counter].args[i].type += '*'
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
                        newline = ''
                        if var_length[j] == 1:
                            newline = assigned_variable+'['+str(i)+'] = '+var_list[j]+';'
                            function_list[counter].scope.insert(it + it2, newline)
                            it2 += 1
                        else:
                            function_list[counter].scope.insert(it + it2, 'for (int it1 = 0; it1 < ' + str(var_length[j]) + '; it1++) {')
                            it2 += 1
                            function_list[counter].scope.insert(it + it2, '    ' + assigned_variable+'['+str(i)+' + it1] = '+var_list[j]+'[it1];')
                            it2 += 1
                            function_list[counter].scope.insert(it + it2, '}')
                            it2 += 1
                        i += int(var_length[j])
        it += 1
    return function_list

def parallelize_function(f):
    parall = False
    n_events = argument('nevents', 'const int', 0, False, [])
    spacing = '    '
    s = 0
    while s < len(f.scope):
        if parall == True:
            f.scope[s] = spacing + f.scope[s]
            #print(f.scope[s])
        elif clean_args(f.scope[s]).startswith('//Begin'):
            parall = True
            s += 1
            while clean_args(f.scope[s]).startswith('//') == True:
                #print(f.scope[s])
                s += 1
            
            
            f.scope.insert(s, 'auto thread_pool = context->device()->tensorflow_cpu_worker_threads()->workers;')
            s += 1
            f.scope.insert(s, 'const int ncores = (int)thread_pool->NumThreads();')
            s += 1
            f.scope.insert(s, INT64Type + ' nreps;')
            s += 1
            f.scope.insert(s, 'if (ncores > 1) {')
            s += 1
            f.scope.insert(s, '    nreps = (' + INT64Type + ')nevents / ncores;')
            s += 1
            f.scope.insert(s, '} else {')
            s += 1
            f.scope.insert(s, '    nreps = 1;')
            s += 1
            f.scope.insert(s, '}')
            s += 1
            f.scope.insert(s, 'const ThreadPool::SchedulingParams p(ThreadPool::SchedulingStrategy::kFixedBlockSize, absl::nullopt, nreps);')
            s += 1
            f.scope.insert(s, 'auto DoWork = [&](' + INT64Type + ' t, ' + INT64Type + ' w) {')
            s += 1
            f.scope.insert(s, 'for (auto it = t; it < w; it += 1) {')
            s += 1
            
        s += 1
    
    f.scope.insert(s, '}')
    s += 1
    f.scope.insert(s, '};')
    s += 1
    f.scope.insert(s, 'thread_pool->ParallelFor(' + n_events.name + ', p, DoWork);')
    
    f = prepare_custom_op(f, n_events)
    
    f.args.append(argument('context', 'const OpKernelContext*', 0, False, []))
    
    return f

def prepare_custom_op(f, nevents):
    
    for i in range(len(f.args) - 1):
        f.args[i].type = 'const ' + f.args[i].type
        if f.args[i].type.endswith('*') == False:
            f.args[i].type += '*'
            if f.args[i].tensor == True:
                for j in range(len(f.scope)):
                    f.scope[j] = re.sub('([()[\]{} ,+\-*/]*)' + f.args[i].name + '([()[\]{} ,+\-*/]*)', '\g<1>' + f.args[i].name + '[it]' + '\g<2>', f.scope[j])
            else:
                for j in range(len(f.scope)):
                    f.scope[j] = re.sub('([()[\]{} ,+\-*/]*)' + f.args[i].name + '([()[\]{} ,+\-*/]*)', '\g<1>' + f.args[i].name + '[0]' + '\g<2>', f.scope[j])
    
    f.args[-1].type = doubleType+'*'
    
    for j in range(len(f.scope)):
        f.scope[j] = re.sub('([()[\]{} ,+\-*/]*)' + f.args[-1].name + '([()[\]{} ,+\-*/]*)', '\g<1>' + f.args[-1].name + '[it]' + '\g<2>', f.scope[j])
        match = re.search('[()[\]{} ,+\-*/]*'+ f.args[0].name +'\[', f.scope[j])
        if match != None:
            number = int(re.sub('.*[()[\]{} ,+\-*/]*'+ f.args[0].name +'\[([0-9]*)\].*', '\g<1>', f.scope[j]))
            f.scope[j] = re.sub('([()[\]{} ,+\-*/]*)'+ f.args[0].name +'\[([0-9]*)\]', '\g<1>'+ f.args[0].name +'+('+ str(f.args[0].slice[-1]) +'*it + ' + str(int(f.args[0].slice[-2]) * number) + ')', f.scope[j])
    
    
    f.args.append(nevents)
    
    return f

def get_signature(line):
    type_ = line.split('dtype=')[1]
    type_ = type_.split(')')[0]
    type_ = convert_type(type_)
    is_tensor = False
    
    shape = line.split('shape=[')[1]
    shape = shape.split(']')[0]
    slice_ = []
    
    if shape == '':
        shape = 0
    elif shape == 'None':
        shape = 0
        is_tensor = True
    else:
        s = shape.split(',', 1)[-1]
        shape = clean_args(shape.split(',')[-1])
        s = s.split(',')
        prod = 1
        for a in s:
            slice_.append(a)
            if a != 'None':
                prod *= int(a)
        slice_.append(str(prod))
        
    name = clean_args(line.split('=')[0])
    
    return signature(name, type_, shape, is_tensor, slice_)

def read_signatures(signatures, signature_variables, file_source):
    f = open(file_source, "r")
    line = f.readline()
    while line != '':
        match = re.search('tf.TensorSpec', line)
        match2 = re.search('signature', line)
        if match != None and clean_args(line).startswith('@tf.function') == False:
            s = get_signature(line)
            signatures.append(s)
        elif match2 != None and clean_args(line).startswith('@tf.function') == False:
            br_count = 0
            for letter in line:
                if letter == '[':
                    br_count += 1
                elif letter == ']':
                    br_count -= 1
            while br_count > 0:
                l = f.readline()
                for letter in l:
                    if letter == '[':
                        br_count += 1
                    elif letter == ']':
                        br_count -= 1
                line += l
            line = re.sub('(TensorSpec\([^)]*\) *),', '\g<1>?', line)
            name = clean_args(line.split('=')[0])
            line = line.split(' = ')[1]
            var_list = line.split('+')
            if len(var_list) == 1:
                var_list = line.split('?')
            sig_list = []
            s_list = []
            for var in var_list:
                match = re.search('tf.TensorSpec', var)
                if match != None:
                    var = re.sub('.*[\n]*.*(tf.TensorSpec\([^)]*\)).*', '\g<1>', var)
                    s_list.append(signature(var, get_signature(var).type, get_signature(var).size, get_signature(var).tensor, get_signature(var).slice))
                match = re.search('\[[a-zA-Z0-9_]+] *\*', var)
                sig_name = clean_args(re.sub('\[([a-zA-Z0-9_]+)].*', '\g<1>', var))
                times = 1
                if match != None:
                    times = int(clean_args(re.sub('\[[a-zA-Z0-9_]+] *\*(\d+)', '\g<1>', var)))
                for i in range(times):
                    #print('sig_name', sig_name, times)
                    sig_list.append(sig_name)
            
            if len(s_list) > 0:
                s = signature_variable(name, s_list, [])
                signature_variables.append(s)
            elif len(sig_list) > 0:
                s = signature_variable(name, [], sig_list)
                signature_variables.append(s)
                #print(s.name, s.signature_list)
        line = f.readline()
    f.close()
    return signatures, signature_variables

def convert_signatures(signatures, signature_variables):
    for i in range(len(signature_variables)):
        for v in signature_variables[i].signature_name_list:
            for s in signatures:
                if s.name == v:
                    signature_variables[i].signature_list.append(s)
    return signature_variables

def define_custom_op(custom_op_list, func):
    s = []
    
    inputTensorsNumber = len(func.args) - 3
    
    for i in range(inputTensorsNumber):
        s.append('const Tensor& ' + func.args[i].name + '_tensor = context->input(' +str(i) + ');')
        s.append('auto ' + func.args[i].name + ' = ' + func.args[i].name + '_tensor.flat<' + re.sub('const ([^&*]*)[&*]*', '\g<1>', func.args[i].type) + '>().data();')
        s.append('')
    
    # func.args[-1] is context
    s.append(func.args[-2].type + ' ' + func.args[-2].name + ' = ' + func.args[0].name + '_tensor.shape().dim_size(0);')
    
    s.append('Tensor* output_tensor = NULL;')
    s.append('OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({' + func.args[-2].name + '}), &output_tensor));')
    s.append('auto ' + func.args[-3].name + ' = output_tensor->flat<' + re.sub('([^&*]*)[&*]*', '\g<1>', func.args[-3].type) + '>();')
    
    functor_name = re.sub('Op', 'Functor', 'MatrixOp')
    
    s.append('')
    line = functor_name + '<Device, COMPLEX_TYPE>()(context->eigen_device<Device>()'
    for i in range(inputTensorsNumber):
        line += ', ' + func.args[i].name
    line += ', ' + func.args[-3].name + '.data()'
    line += ', ' + func.args[-2].name
    line += ', ' + func.args[-1].name + ');'
    s.append(line)
    
    
    c_op = custom_operator('MatrixOp', s, functor_name)
    custom_op_list.append(c_op)
    return custom_op_list

def modify_matrix(infile, temp, process_name, destination):
    f = open(infile, 'r')
    line = f.readline()
    new_matrix = ""
    inside_matrix = False
    p = re.sub('_', '', process_name)
    while line != '':
        temp += line
        if clean_args(line).startswith('defsmatrix('):
            inside_matrix = True
        if inside_matrix == True:
            #print(line)
            if clean_args(line).startswith('for'):
                space = line.split('for')[0]
                new_matrix += space + 'matrixOp = tf.load_op_library(\'' + destination + './matrix_' + process_name + '_cu.so\')\n'
            new_matrix += line
            if clean_args(line).startswith('return'):
                inside_matrix = False
                break
        line = f.readline()
    new_matrix = re.sub('smatrix\(', 'cusmatrix(', new_matrix)
    new_matrix = re.sub('self\.matrix\(', 'matrixOp.matrix' + p + '(', new_matrix)
    temp += new_matrix
    while line != '':
        temp += line
        line = f.readline()
    #print(new_matrix)
    return temp

def extract_constants(func, constants):
    
    count = 0
    for i in range(len(func.scope)):
        if func.scope[i].startswith('const double '):
            constants.append(change_array_into_variable(func.scope[i]))
            del func.scope[i]
            i -= 1
            count += 1
        if count == 2:
            break
    
    for i in range(len(func.scope)):
        match = re.search('denom', func.scope[len(func.scope) - i - 1])
        if match != None:
            func.scope[len(func.scope) - i - 1] = re.sub('denom\[[a-zA-Z0-9+\-*/_]\]', 'denom', func.scope[len(func.scope) - i - 1])
            break
    
    return func, constants
    
def change_array_into_variable(line):
    match = re.search('denom', line)
    if match != None:
        line = re.sub('\[\]', '', line)
        line = re.sub('{([+\-0-9]+).*;', '\g<1>', line)
        return line
    else:
        line = re.sub(';', '', line)
        return line



folder_name = 'prov/'

temp = ""

def translate(destination):
    
    if destination[-1] != '/': # Avoid weird behaviours if destination does not end with '/'
        destination += '/'
    file_sources = [madflow.wavefunctions_flow.__file__] # path to wavefunctions_flow.py
    
    # Create the directory for the Op source code and create the makefile
    
    subprocess.check_output(["/bin/sh", "-c", "mkdir " + destination + "gpu/"])
    write_makefile(destination)
    
    auxiliary_functions = []
    function_list_ = []
    auxiliary_functions, function_list_ = generate_auxiliary_functions(auxiliary_functions, function_list_)
    
    for file_source in file_sources:
        signatures_ = []
        signature_variables_ = []
    
        signatures_, signature_variables_ = read_signatures(signatures_, signature_variables_, file_source)
    
        signature_variables_ = convert_signatures(signatures_, signature_variables_)
    
        function_list_ = read_file_from_source(function_list_, file_source, signatures_, signature_variables_)
    
    files_list = subprocess.check_output(["/bin/sh", "-c", "ls " + destination + ' | grep matrix_1_']).decode('utf-8').split('\n')[:-1]
    
    for _file_ in files_list:
        
        
        constants = []#globalConstants
        
        for e in globalConstants:
            constants.append(e)
        
        process_name = re.sub('matrix_1_', '', _file_)
        process_name = re.sub('\.py', '', process_name)
        
        matrix_source = destination + 'matrix_1_' + process_name + '.py'
        process_source = destination + 'aloha_1_' + process_name + '.py'
        
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
    
        function_list = read_file_from_source(function_list, _file_, signatures, signature_variables)
        
        matrix_name = 'matrix_1_' + process_name + '.py'
        
        signatures, signature_variables = read_signatures(signatures, signature_variables, matrix_source)
        signature_variables = convert_signatures(signatures, signature_variables)
        
        function_list = extract_matrix_from_file(function_list, matrix_source, signatures, signature_variables)
        
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
        temp = write_defined(temp, defined, 'cpu')
        
        temp = write_constants(temp, constants, 'cpu')
        temp = write_constants(temp, cpuConstants, 'cpu')

        for i in range(len(function_list[-1].scope)):
            if clean_args(function_list[-1].scope[i]).startswith(function_list[-1].args[-3].name):
                function_list[-1].scope[i] = re.sub('.real\(\)', '', function_list[-1].scope[i])
        
        for f in function_list:
            temp = write_function_definition(temp, f, 'cpu')
        
        for f in function_list:
            temp = write_empty_line(temp)
            temp = write_function(temp, f, 'cpu')
        
        for c in custom_op_list:
            temp = write_custom_op(temp, c, function_list[-1], 'cpu', process_name)
            
        with open(destination + "gpu/matrix_" + process_name + ".cc", "w") as fh:
            fh.write(temp)
        
        
        temp = ""
        temp += "#ifdef GOOGLE_CUDA\n\
"               "#define EIGEN_USE_GPU\n"
        temp = write_libraries(temp, libraries)
        temp = write_headers(temp, headers)
        temp = write_namespaces(temp, namespace)
        temp = write_defined(temp, defined, 'gpu')
        
        
        temp = write_constants(temp, constants, 'gpu')
        
        
        del function_list[-1].args[-1]
        
        i = 0
        while i < len(function_list[-1].scope):
            if function_list[-1].scope[i].startswith('auto thread_pool'):
                while i < len(function_list[-1].scope) and function_list[-1].scope[i].startswith('for (auto it') == False:
                    del function_list[-1].scope[i]
                function_list[-1].scope[i] = 'for (int it = blockIdx.x * blockDim.x + threadIdx.x; it < ' + function_list[-1].args[-1].name + '; it += blockDim.x * gridDim.x) {'
            elif function_list[-1].scope[i] == '};':
                del function_list[-1].scope[i]
                del function_list[-1].scope[i]
                break
            i += 1
        
        for f in function_list:
            temp = write_function_definition(temp, f, 'gpu')
        
        temp = write_empty_line(temp)
        temp += gpuArithmeticOperators
        
        for f in function_list:
            temp = write_empty_line(temp)
            temp = write_function(temp, f, 'gpu')
        
        
        function_list[-1].args.append(argument('context', 'const OpKernelContext*', 0, False, []))
        
        for c in custom_op_list:
            temp = write_custom_op(temp, c, function_list[-1], 'gpu', process_name)
        
        temp = re.sub('([ ,+\-*/]+)sign([ (;]+)', '\g<1>signn\g<2>', temp) 
        temp = re.sub('([ ,+\-*/]+)signvec([ (;]+)', '\g<1>signvecc\g<2>', temp)    
        
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
            
        
        #--------------------------------------------------------------------------------------
      

def compile(destination):
    subprocess.check_output(["/bin/sh", "-c", "cd " + destination + "; make"])

if __name__ == "__main__":
    translate(folder_name)
