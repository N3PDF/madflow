#####################################################
#                                                   #
#  Source file of the Matrix Elements exports for   #
#  the PyOut MG5aMC plugin.                         #
#  Defines some classes which inherit from          #
#  those inside create aloha, but use the module    #
#  PyOut_PythonFileWriter.py for output 
#                                                   #
#####################################################

import madgraph.iolibs.file_writers as file_writers
import madgraph.various.misc as misc
import aloha
import aloha.create_aloha as create_aloha
import aloha.aloha_writers as aloha_writers
from . import PyOut_PythonFileWriter as PythonFileWriter

import cmath
import os
import re 
from numbers import Number
from collections import defaultdict
from fractions import Fraction
# fast way to deal with string
from six import StringIO
# Look at http://www.skymind.com/~ocrow/python_string/ 
# For knowing how to deal with long strings efficiently.
import itertools



class ALOHAWriterForTensorFlow(aloha_writers.ALOHAWriterForPython):
    """a new class, similar to the Python writer, but capable
    of generating TensorFlow-compatible functions
    """

    #extension = '.py'
    #writer = file_writers.PythonWriter

    ci_definition = 'cI = complex_tf(0,1)\n'
    realoperator = 'tf.math.real()'
    imagoperator = 'tf.math.imag()'

    # use complex_me everywhere
    type2def = {}    
    type2def['int'] = 'complex_me'
    type2def['double'] = 'complex_me'
    type2def['complex'] = 'complex_me'

    #@staticmethod
    def change_number_format(self, number):
        """Formating the number
        MZ: similar to the CPP function
        """

        if isinstance(number, complex):
            if number.imag:
                if number.real:
                    out = '(%s + %s*cI)' % (self.change_number_format(number.real), \
                                    self.change_number_format(number.imag))
                else:
                    if number.imag == 1:
                        out = 'cI'
                    elif number.imag == -1:
                        out = '-cI'
                    else: 
                        out = '%s * cI' % self.change_number_format(number.imag)
            else:
                out = '%s' % (self.change_number_format(number.real))
        else:
            tmp = Fraction(str(number))
            tmp = tmp.limit_denominator(100)
            if not abs(tmp - number) / abs(tmp + number) < 1e-8:
                out = '%.9f' % (number)
            else:
                out = '%s./%s.' % (tmp.numerator, tmp.denominator)
        return out


    def change_var_format(self, name): 
        """Formatting the variable name to Python format
        start to count at zero. 
        No neeed to define the variable in python -> no need to keep track of 
        the various variable
        """
        
        if '_' not in name:
            self.declaration.add((name.type, name))
        else:
            self.declaration.add(('', name.split('_',1)[0]))
        name = re.sub('(?P<var>\w*)_(?P<num>\d+)$', self.shift_indices , name)
        
        return name


    def get_fct_format(self, fct):
        """Put the function in the correct format"""
        if not hasattr(self, 'fct_format'):
            one = self.change_number_format(1)
            self.fct_format = {'csc' : '{0}/tf.math.cos(%s)'.format(one),
                   'sec': '{0}/tf.math.sin(%s)'.format(one),
                   'acsc': 'tf.math.asin({0}/(%s))'.format(one),
                   'asec': 'tf.math.acos({0}/(%s))'.format(one),
                   're': ' tf.math.real(%s)',
                   'im': 'tf.match.imac(%s)',
                   'cmath.sqrt': 'tf.math.sqrt(%s)',
                   'sqrt': 'tf.math.sqrt(%s)',
                   'pow': 'tf.math.pow(%s, %s)',
                   'complexconjugate': 'tf.math.conj(%s)',
                   '/' : '{0}/%s'.format(one),
                   'abs': 'tf.math.abs(%s)'
                   }
            
        if fct in self.fct_format:
            return self.fct_format[fct]
        elif hasattr(cmath, fct):
            self.declaration.add(('fct', fct))
            print ('MZ, this case should be changed for tensorflow', fct)
            return 'cmath.{0}(%s)'.format(fct)
        else:
            raise Exception("Unable to handle function name %s (no special rule defined and not in cmath)" % fct)


    def define_expression(self):
        """ Identical to the mother class function, but replace 1j with cI
        (strange errors were obtained when calling the mother class function
        """
        out = StringIO()

        if self.routine.contracted:
            keys = list( self.routine.contracted.keys())
            keys.sort()
            
            for name in keys:
                obj = self.routine.contracted[name]
                out.write('    %s = %s\n' % (name, self.write_obj(obj)))

        def sort_fct(a, b):
            if len(a) < len(b):
                return -1
            elif len(a) > len(b):
                return 1
            elif a < b:
                return -1
            else:
                return +1
            
        keys = list(self.routine.fct.keys())        
        keys.sort(key=misc.cmp_to_key(sort_fct))
        for name in keys:
            fct, objs = self.routine.fct[name]
            format = '    %s = %s\n' % (name, self.get_fct_format(fct))
            try:
                text = format % ','.join([self.write_obj(obj) for obj in objs])
            except TypeError:
                text = format % tuple([self.write_obj(obj) for obj in objs])
            finally:
                out.write(text)

        numerator = self.routine.expr
        if not 'Coup(1)' in self.routine.infostr:
            coup_name = 'COUP'
        else:
            coup_name = '%s' % self.change_number_format(1)

        if not self.offshell:
            if coup_name == 'COUP':
                out.write('    vertex = COUP*%s\n' % self.write_obj(numerator.get_rep([0])))
            else:
                out.write('    vertex = %s\n' % self.write_obj(numerator.get_rep([0])))
        else:
            OffShellParticle = '%s%d' % (self.particles[self.offshell-1],\
                                                                  self.offshell)

            if not 'L' in self.tag:
                coeff = 'denom'
                if not aloha.complex_mass:
                    if self.routine.denominator:
                        out.write('    denom = %(COUP)s/(%(denom)s)\n' % {'COUP': coup_name,\
                                'denom':self.write_obj(self.routine.denominator)}) 
                    else:
                        out.write('    denom = %(coup)s/(P%(i)s[0]**2-P%(i)s[1]**2-P%(i)s[2]**2-P%(i)s[3]**2 - M%(i)s * (M%(i)s -cI* W%(i)s))\n' % 
                          {'i': self.outgoing,'coup':coup_name})
                else:
                    if self.routine.denominator:
                        raise Exception('modify denominator are not compatible with complex mass scheme')                
                    
                    out.write('    denom = %(coup)s/(P%(i)s[0]**2-P%(i)s[1]**2-P%(i)s[2]**2-P%(i)s[3]**2 - M%(i)s**2)\n' % 
                          {'i': self.outgoing,'coup':coup_name})                    
            else:
                coeff = 'COUP'
                
            for ind in numerator.listindices():
                out.write('    %s[%d]= %s*%s\n' % (self.outname, 
                                        self.pass_to_HELAS(ind), coeff, 
                                        self.write_obj(numerator.get_rep(ind))))
        return out.getvalue()


    def get_foot_txt(self):
        if not self.offshell:
            return '    return vertex\n\n'
        else:
            return '    return tf.stack(%s, axis=0)\n\n' % (self.outname)



    def get_header_txt(self, name=None, couplings=None, **opt):
        if name is None:
            name = self.name
           
        out = StringIO()

        out.write('from alohaflow.config import DTYPE, DTYPEINT, complex_tf, complex_me, DTYPECOMPLEX\n')
        out.write('import tensorflow as tf\n\n')

        arguments = self.define_argument_list(couplings) 

        arguments_names = [arg[1] for arg in arguments]

        # the signature
        shape_dict = {'list_complex' : '[None,None]',
                      'complex' : '[None]',
                      'double' : '[]'}
        type_dict = {'list_complex' : 'DTYPECOMPLEX',
                      'complex' : 'DTYPECOMPLEX',
                      'double' : 'DTYPE'}

        out.write('%(name)s_signature = [\n') 

        for arg in arguments:
            fmt = arg[0]
            out.write('tf.TensorSpec(shape=%(shape)s, dtype=%(type)s),\n' %
                    {'shape': shape_dict[fmt], 'type': type_dict[fmt]})
        out.write(']\n\n')

        out.write('@tf.function(input_signature=%(name)s_signature)\n')
        out.write('def %(name)s(%(args)s):\n' )

        return out.getvalue() % {'name': name, 'args': ','.join(arguments_names)}


    def get_momenta_txt(self):
        """Define the Header of the fortran file. This include
            - momentum conservation
            - definition of the impulsion"""
             
        out = StringIO()
        
        # Define all the required momenta
        p = [] # a list for keeping track how to write the momentum
        
        signs = self.get_momentum_conservation_sign()
        
        for i,type in enumerate(self.particles):
            if self.declaration.is_used('OM%s' % (i+1)):
               out.write("    OM{0} = complex_tf(0, 0)\n    if (M{0}): OM{0}=complex_tf(1,0)/M{0}**2\n".format( (i+1) ))
            if i+1 == self.outgoing:
                out_type = type
                out_size = self.type_to_size[type] 
                continue
            elif self.offshell:
                p.append('{0}{1}{2}[%(i)s]'.format(signs[i],type,i+1))  
                
            if self.declaration.is_used('P%s' % (i+1)):
                self.get_one_momenta_def(i+1, out)             
             
        # define the resulting momenta
        if self.offshell:
            type = self.particles[self.outgoing-1]
            out.write('    %s%s = [complex_tf(0,0)] * %s\n' % (type, self.outgoing, out_size))
            if aloha.loop_mode:
                size_p = 4
            else:
                size_p = 2
            for i in range(size_p):
                dict_energy = {'i':i}
                rhs = ''.join(p) % dict_energy
                # remove trailing '+'
                if rhs.startswith('+'):
                    rhs = rhs[1:]
    
                out.write('    %s%s[%s] = %s\n' % (type,self.outgoing,i,rhs))
            
            self.get_one_momenta_def(self.outgoing, out)

               
        # Returning result
        return out.getvalue()


    def get_one_momenta_def(self, i, strfile):
        """return the string defining the momentum"""

        type = self.particles[i-1]
        
        main = '    P%d = complex_tf(tf.stack([' % i
        if aloha.loop_mode:
            template ='%(sign)s%(type)s%(i)d[%(nb)d]'
        else:
            template ='%(sign)stf.math%(operator)s(%(type)s%(i)d[%(nb2)d])'

        nb2 = 0
        strfile.write(main)
        data = []
        for j in range(4):
            if not aloha.loop_mode:
                nb = j
                if j == 0: 
                    assert not aloha.mp_precision 
                    operator = '.real' # not suppose to pass here in mp
                elif j == 1: 
                    nb2 += 1
                elif j == 2:
                    assert not aloha.mp_precision 
                    operator = '.imag' # not suppose to pass here in mp
                elif j ==3:
                    nb2 -= 1
            else:
                operator =''
                nb = j
                nb2 = j
            data.append(template % {'j':j,'type': type, 'i': i, 
                        'nb': nb, 'nb2': nb2, 'operator':operator,
                        'sign': self.get_P_sign(i)}) 
            
        strfile.write(', '.join(data))
        strfile.write('], axis=0), 0.)\n')


    def get_declaration_txt(self, add_i=True):
        """ Prototype for how to write the declaration of variable
            Include the symmetry line (entry FFV_2)
        """
        
        out = StringIO()
        argument_var = [name for type,name in self.call_arg]
        # define the complex number CI = 0+1j
        if add_i:
            out.write('    ' + self.ci_definition)
                    
        for type, name in self.declaration.tolist():
            # skip P, V, etc... only Coup, masses, CI,
            if type.startswith('list'): continue
            if type == '': continue
            if name.startswith('TMP'): continue
            out.write('    %s = %s(%s)\n' % (name, self.type2def[type], name))               

        return out.getvalue()



    def write_obj_Add(self, obj, prefactor=True):
        """Turns addvariable into a string. Avoids trailing '+'"""

        data = defaultdict(list)
        number = []
        [data[p.prefactor].append(p) if hasattr(p, 'prefactor') else number.append(p)
             for p in obj]

        file_str = StringIO()
        if prefactor and obj.prefactor != 1:
            formatted = self.change_number_format(obj.prefactor)
            if formatted.startswith(('+','-')):
                file_str.write('(%s)' % formatted)
            else:
                file_str.write(formatted)
            file_str.write('*(')
        else:
            file_str.write('(')
        first=True
        for value, obj_list in data.items():
            add= '+'
            if value not in  [-1,1]:
                nb_str = self.change_number_format(value)
                if nb_str[0] in ['+','-']:
                    file_str.write(nb_str)
                else:
                    # remove trailing '+'
                    if not first:
                        file_str.write('+')
                    file_str.write(nb_str)
                file_str.write('*(')
            elif value == -1:
                add = '-' 
                file_str.write('-')
            elif not first:
                file_str.write('+')
            else:
                file_str.write('')
            first = False
            file_str.write(add.join([self.write_obj(obj, prefactor=False) 
                                                          for obj in obj_list]))
            if value not in [1,-1]:
                file_str.write(')')
        if number:
            total = sum(number)
            file_str.write('+ %s' % self.change_number_format(total))

        file_str.write(')')
        return file_str.getvalue()



class PyOutAbstractRoutine(create_aloha.AbstractRoutine):
    """Same as AbstractRoutine, except for the write 
    function which forces the usage of a
    PyOut_PythonFileWriter.py 

    Also includes a copy constructor
    """

    def __init__(self, *args):
        """copy constructor if only a AbstractRoutine is passed. Otherwise calls
        the mother class
        """
        attrs_to_copy = [ 
            'spins',
            'expr',
            'denominator',
            'name',
            'outgoing',
            'infostr',
            'symmetries',
            'combined',
            'fct',
            'tag',
            'contracted']
        if len(args) == 1 and type(args[0])==create_aloha.AbstractRoutine:
            for attr in attrs_to_copy:
                setattr(self, attr, getattr(args[0], attr))
        else:
            super(PyOutAbstractRoutine, self).__init__(args)



    def write(self, output_dir, language='Fortran', mode='self', combine=True,**opt):
        """ write the content of the object. Same function as in aloha/create_aloha
        except for the first line
        """
        writer = ALOHAWriterForTensorFlow(self, output_dir)
        text = writer.write(mode=mode, **opt)
        if combine:
            for grouped in self.combined:
                if isinstance(text, tuple):
                    text = tuple([old.__add__(new)  for old, new in zip(text, 
                             writer.write_combined(grouped, mode=mode+'no_include', **opt))])
                else:
                    text += writer.write_combined(grouped, mode=mode+'no_include', **opt)        
        if aloha.mp_precision and 'MP' not in self.tag:
            self.tag.append('MP')
            text += self.write(output_dir, language, mode, **opt)
        return text
