#####################################################
#                                                   #
#  Source file of the Matrix Elements exports for   #
#  the PyOut MG5aMC plugin.                         #
#                                                   #
#####################################################

import os
import logging
import fractions
import shutil
import itertools
import copy
from math import fmod
import subprocess
import re
import string

plugin_path = os.path.dirname(os.path.realpath( __file__ ))

from madgraph import MadGraph5Error, InvalidCmd, MG5DIR
import madgraph.iolibs.export_python as export_python
import madgraph.iolibs.helas_call_writers as helas_call_writers
import madgraph.iolibs.files as files
import madgraph.core.color_algebra as color
import aloha
import aloha.create_aloha as create_aloha
import aloha.aloha_writers as aloha_writers
from . import PyOut_create_aloha as pyout_create_aloha
from . import PyOut_helas_call_writer as pyout_helas_call_writer

import models.check_param_card as check_param_card

logger = logging.getLogger('PyOut_plugin.MEExporter')

pjoin = os.path.join 

class PyOutExporterError(MadGraph5Error):
    """ Error from the Resummation MEs exporter. """ 


def coeff(ff_number, frac, is_imaginary, Nc_power, Nc_value=3, is_first = False):
    """Returns a nicely formatted string for the coefficients in JAMP lines"""

    total_coeff = ff_number * frac * fractions.Fraction(Nc_value) ** Nc_power

    if total_coeff == 1:
        plus = '+'
        if is_first:
            plus = ''
        if is_imaginary:
            return plus + 'complex_tf(0,1)*'
        else:
            return plus
    elif total_coeff == -1:
        if is_imaginary:
            return '-complex(0,1)*'
        else:
            return '-'

    if is_first:
        res_str = '%-i.' % total_coeff.numerator
    else:
        res_str = '%+i.' % total_coeff.numerator

    if total_coeff.denominator != 1:
        # Check if total_coeff is an integer
        res_str = res_str + '/%i.' % total_coeff.denominator

    if is_imaginary:
        res_str = res_str + '*complex(0,1)'

    return res_str + '*'


class PyOutExporter(export_python.ProcessExporterPython):
    """this exporter is built upon the Python exporter of MG5_aMC.
    If changes are requested wrt the latter, one can just define 
    here the corresponding function
    """
    
    # check status of the directory. Remove it if already exists
    check = True 
    # Language type: 'v4' for f77 'cpp' for C++ output
    exporter = 'v4'
    # Output type:
    #[Template/dir/None] copy the Template, just create dir  or do nothing 
    output = 'dir'
    # Decide which type of merging if used [madevent/madweight]
    grouped_mode = False
    # if no grouping on can decide to merge uu~ and u~u anyway:
    sa_symmetry = False

    def __init__(self, dir_path, *args, **opts): 
        os.mkdir(dir_path)
        self.dir_path = dir_path


    def pass_information_from_cmd(self, cmd):
        """Pass information for MA5"""
        
        self.proc_defs = cmd._curr_proc_defs
        self.model = cmd._curr_model



    def get_python_matrix_methods(self, gauge_check=False):
        """Write the matrix element calculation method for the processes"""

        replace_dict = {}

        # Extract version number and date from VERSION file
        info_lines = self.get_mg5_info_lines()
        replace_dict['info_lines'] = info_lines

        for ime, matrix_element in enumerate(self.matrix_elements):
            process_string = matrix_element.get('processes')[0].shell_string()
            if process_string in self.matrix_methods:
                continue

            replace_dict['process_string'] = process_string

            # Extract number of external particles
            (nexternal, ninitial) = matrix_element.get_nexternal_ninitial()
            replace_dict['nexternal'] = nexternal

            # Extract ncomb
            ncomb = matrix_element.get_helicity_combinations()
            replace_dict['ncomb'] = ncomb

            # Extract helicity lines
            helicity_lines = self.get_helicity_matrix(matrix_element)
            replace_dict['helicity_lines'] = helicity_lines

            # Extract overall denominator
            # Averaging initial state color, spin, and identical FS particles
            den_factor_line = self.get_den_factor_line(matrix_element)
            replace_dict['den_factor_line'] = den_factor_line

            # Extract process info lines for all processes
            process_lines = self.get_process_info_lines(matrix_element)
            replace_dict['process_lines'] = process_lines
        
            # Extract ngraphs
            ngraphs = matrix_element.get_number_of_amplitudes()
            replace_dict['ngraphs'] = ngraphs

            # Extract ndiags
            ndiags = len(matrix_element.get('diagrams'))
            replace_dict['ndiags'] = ndiags

            # Extract helas calls
            helas_calls = self.helas_call_writer.get_matrix_element_calls(\
                                                    matrix_element, gauge_check)
            replace_dict['helas_calls'] = "\n        ".join(helas_calls)

            # Extract nwavefuncs
            nwavefuncs = matrix_element.get_number_of_wavefunctions()
            replace_dict['nwavefuncs'] = nwavefuncs

            # Extract ncolor
            ncolor = max(1, len(matrix_element.get('color_basis')))
            replace_dict['ncolor'] = ncolor

            # Extract model parameter lines
            parameters, couplings = \
                                 self.get_model_parameters(matrix_element)

            model_parameter_lines =  "\n    ".join([\
                         "%(param)s = model.get(\'parameter_dict\')[\"%(param)s\"]"\
                         % {"param": param} for param in parameters]) + \
                                     "\n    " + "\n    ".join([\
                         "%(coup)s = model.get(\'coupling_dict\')[\"%(coup)s\"]"\
                              % {"coup": coup} for coup in couplings])

            if aloha.complex_mass:
                paramsignature = ",\n        ".join(['tf.TensorSpec(shape=[], dtype=DTYPECOMPLEX)'] * len(parameters+couplings))
                paramtuple = ",".join(["complex_me(%s)" % p for p in parameters+couplings]) 

            else:
                paramsignature = ",\n        ".join(['tf.TensorSpec(shape=[], dtype=DTYPE)'] * len(parameters) + 
                                                    ['tf.TensorSpec(shape=[], dtype=DTYPECOMPLEX)'] * len(couplings))
                paramtuple = ",".join(["float_me(%s)" % p for p in parameters] + ["complex_me(%s)" % p for p in couplings]) 

            params = ",".join([p for p in parameters + couplings])
            paramnames = ",".join(["\"%s\"" % p for p in parameters + couplings])

            replace_dict['model_parameters'] = model_parameter_lines
            replace_dict['paramsignature'] = paramsignature
            replace_dict['params'] = params
            replace_dict['paramnames'] = paramnames
            replace_dict['paramtuple'] = paramtuple

            # Extract color data lines
            color_matrix_lines = self.get_color_matrix_lines(matrix_element)
            replace_dict['color_matrix_lines'] = \
                                               "\n        ".join(color_matrix_lines)

            # Extract JAMP lines
            jamp_lines = self.get_jamp_lines(matrix_element)
            replace_dict['jamp_lines'] = jamp_lines

            # Extract amp2 lines
            amp2_lines = self.get_amp2_lines(matrix_element,
                                        self.config_maps.setdefault(ime, []))
            replace_dict['amp2_lines'] = '\n        #'.join(amp2_lines)

            replace_dict['model_path'] = self.model.path 
            replace_dict['root_path'] = MG5DIR

            replace_dict['aloha_imports'] = "\n".join(["from %s import *" % name for name in self.aloha_names]) 

            method_file = open(os.path.join(plugin_path, \
                       'template_files/matrix_method_python.inc')).read()
            method_file = method_file % replace_dict

            self.matrix_methods[process_string] = method_file

        return self.matrix_methods


    def get_helicity_matrix(self, matrix_element):
        """Return the Helicity matrix definition lines for this matrix element"""

        helicity_line = "helicities = float_me([ \\\n        "
        helicity_line_list = []

        for helicities in matrix_element.get_helicity_matrix():
           helicity_line_list.append("[" + ",".join(['%d'] * len(helicities)) % \
                                      tuple(helicities) + "]")
            
        return helicity_line + ",\n        ".join(helicity_line_list) + "])"


    def get_color_matrix_lines(self, matrix_element):
        """Return the color matrix definition lines for this matrix element. Split
        rows in chunks of size n."""

        if not matrix_element.get('color_matrix'):
            return ["denom = tf.constant([1.], dtype=DTYPECOMPLEX)", "cf = tf.constant([[1.]], dtype=DTYPECOMPLEX)"]
        else:
            color_denominators = matrix_element.get('color_matrix').\
                                                 get_line_denominators()
            denom_string = "denom = tf.constant([%s], dtype=DTYPECOMPLEX)" % \
                           ",".join(["%i" % denom for denom in color_denominators])

            matrix_strings = []
            my_cs = color.ColorString()
            for index, denominator in enumerate(color_denominators):
                # Then write the numerators for the matrix elements
                num_list = matrix_element.get('color_matrix').\
                                            get_line_numerators(index, denominator)

                matrix_strings.append("%s" % \
                                     ",".join(["%d" % i for i in num_list]))
            matrix_string = "cf = tf.constant([[" + \
                            "],\n                          [".join(matrix_strings) + "]], dtype=DTYPECOMPLEX)"
            return [denom_string, matrix_string]


    def get_den_factor_line(self, matrix_element):
        """Return the denominator factor line for this matrix element"""

        return "denominator = float_me(%d)" % \
               matrix_element.get_denominator_factor()


    def get_jamp_lines(self, matrix_element):
        """Return the jamp = sum(fermionfactor * amp[i]) lines"""

        res_list = []

        for i, coeff_list in enumerate(matrix_element.get_color_amplitudes()):

            is_first = i==0

            res = ""

            # Optimization: if all contributions to that color basis element have
            # the same coefficient (up to a sign), put it in front
            list_fracs = [abs(coefficient[0][1]) for coefficient in coeff_list]
            common_factor = False
            diff_fracs = list(set(list_fracs))
            if len(diff_fracs) == 1 and abs(diff_fracs[0]) != 1:
                common_factor = True
                global_factor = diff_fracs[0]
                res = res + '%s(' % coeff(1, global_factor, False, 0, is_first=is_first)

            for i2, (coefficient, amp_number) in enumerate(coeff_list):
                is_first2 = i2==0
                if common_factor:
                    res = res + "%samp%d" % (coeff(coefficient[0],
                                               coefficient[1] / abs(coefficient[1]),
                                               coefficient[2],
                                               coefficient[3], is_first=is_first2),
                                               amp_number - 1)
                else:
                    res = res + "%samp%d" % (coeff(coefficient[0],
                                               coefficient[1],
                                               coefficient[2],
                                               coefficient[3], is_first=is_first2),
                                               amp_number - 1)

            if common_factor:
                res = res + ')'

            res_list.append(res)

        return "jamp = tf.stack([" + ",".join([r for r in res_list]) + "], axis=0)"


    def get_model_parameters(self, matrix_element):
        """Return definitions for all model parameters used in this
        matrix element"""

        # Get all masses and widths used
        if aloha.complex_mass:
            parameters = [(wf.get('mass') == 'ZERO' or wf.get('width')=='ZERO') 
                          and wf.get('mass') or 'CMASS_%s' % wf.get('mass') 
                          for wf in \
                          matrix_element.get_all_wavefunctions()]
            parameters += [wf.get('mass') for wf in \
                      matrix_element.get_all_wavefunctions()]
        else:
            parameters = [wf.get('mass') for wf in \
                      matrix_element.get_all_wavefunctions()]
        parameters += [wf.get('width') for wf in \
                       matrix_element.get_all_wavefunctions()]
        parameters = list(set(parameters))
        if 'ZERO' in parameters:
            parameters.remove('ZERO')

        # Get all couplings used

        
        couplings = list(set([c.replace('-', '') for func \
                              in matrix_element.get_all_wavefunctions() + \
                              matrix_element.get_all_amplitudes() for c in func.get('coupling')
                              if func.get('mothers') ]))

        return sorted(parameters), sorted(couplings)
        


    def generate_subprocess_directory(self, subproc_group,
                                         fortran_model, me=None):

        self.helas_writer = pyout_helas_call_writer.PyOutUFOHelasCallWriter(self.model)

        super(PyOutExporter, self).__init__(subproc_group, self.helas_writer) 

        model_path = self.model.path

        # this has to be done before the methods, in order to know
        # the functions to be included

        self.aloha_names = self.write_alohas()

        python_matrix_elements = self.get_python_matrix_methods()

        for matrix_element in self.matrix_elements:
            proc = matrix_element.get('processes')[0].shell_string()
            me_text = python_matrix_elements[proc]

            fout = open(pjoin(self.dir_path, 'matrix_%s.py' % proc), 'w')
            fout.write(python_matrix_elements[proc])
            fout.close()




    #===========================================================================
    # convert_model
    #===========================================================================    
    def convert_model(self, model, wanted_lorentz = [], wanted_couplings = []):
         
        IGNORE_PATTERNS = ('*.pyc','*.dat','*.py~')
        try:
            shutil.rmtree(pjoin(self.dir_path,'bin','internal','ufomodel'))
        except OSError as error:
            pass
        model_path = model.get('modelpath')
        # This is not safe if there is a '##' or '-' in the path.
        shutil.copytree(model_path, 
                               pjoin(self.dir_path,'bin','internal','ufomodel'),
                               ignore=shutil.ignore_patterns(*IGNORE_PATTERNS))
        if hasattr(model, 'restrict_card'):
            out_path = pjoin(self.dir_path, 'bin', 'internal','ufomodel',
                                                         'restrict_default.dat')
            if isinstance(model.restrict_card, check_param_card.ParamCard):
                model.restrict_card.write(out_path)
            else:
                files.cp(model.restrict_card, out_path)


    def write_alohas(self):
        """ write the aloha functions, and returns a list of their names
        """
        aloha_model = create_aloha.AbstractALOHAModel(os.path.basename(self.model.get('modelpath')))
        aloha_model.add_Lorentz_object(self.model.get('lorentz'))
        
        wanted_lorentz = [me.get_used_lorentz() for me in self.matrix_elements]
        wanted_lorentz = set(sum(wanted_lorentz, []))
        # Compute the subroutines
        if wanted_lorentz:
            aloha_model.compute_subset(list(wanted_lorentz))
        else:
            aloha_model.compute_all(save=False)

        # convert them to PyOutAbstractRoutines
        routine_names = []
        for k,v in aloha_model.items():
            aloha_model[k] = pyout_create_aloha.PyOutAbstractRoutine(v)
            routine_names.append(aloha_writers.get_routine_name(abstract = aloha_model[k]))
        # Write them out
        write_dir=self.dir_path
        aloha_model.write(write_dir, 'Python')
        return routine_names



    def finalize(self, matrix_elements, history, mg5options, flaglist):
        """ do nothing at the moment"""
        pass


