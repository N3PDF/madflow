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

        out.write('import tensorflow as tf\n')
        out.write('from vegasflow.configflow import DTYPE, DTYPEINT\n')
        out.write('from config import complex_tf, complex_me, DTYPECOMPLEX\n')

        arguments = [arg for format, arg in self.define_argument_list(couplings)]       

        # the signature
        out.write('%(name)s_signature = [\n') 
        ##### ADD SIGNATURE
        out.write(']\n\n')

        out.write('@tf.function(input_signature=%(name)s_signature')
        out.write('def %(name)s(%(args)s):\n' % \
                                    {'name': name, 'args': ','.join(arguments)})

        return out.getvalue()



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

