#####################################################
#                                                   #
#  Source file of the Matrix Elements exports for   #
#  the PyOut MG5aMC plugin.                         #
#                                                   #
#####################################################

import os
import logging
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
import aloha.create_aloha as create_aloha
import aloha.aloha_writers as aloha_writers
from . import PyOut_create_aloha as pyout_create_aloha

import models.check_param_card as check_param_card

logger = logging.getLogger('PyOut_plugin.MEExporter')

pjoin = os.path.join 

class PyOutExporterError(MadGraph5Error):
    """ Error from the Resummation MEs exporter. """ 

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
            model_parameter_lines = \
                                 self.get_model_parameter_lines(matrix_element)
            replace_dict['model_parameters'] = model_parameter_lines

            # Extract color data lines
            color_matrix_lines = self.get_color_matrix_lines(matrix_element)
            replace_dict['color_matrix_lines'] = \
                                               "\n        ".join(color_matrix_lines)

            # Extract JAMP lines
            jamp_lines = self.get_jamp_lines(matrix_element)
            replace_dict['jamp_lines'] = "\n        ".join(jamp_lines)

            # Extract amp2 lines
            amp2_lines = self.get_amp2_lines(matrix_element,
                                        self.config_maps.setdefault(ime, []))
            replace_dict['amp2_lines'] = '\n        '.join(amp2_lines)

            replace_dict['model_path'] = self.model.path 
            replace_dict['root_path'] = MG5DIR

            replace_dict['aloha_imports'] = "\n".join(["from %s import *" % name for name in self.aloha_names]) 

            method_file = open(os.path.join(plugin_path, \
                       'template_files/matrix_method_python.inc')).read()
            method_file = method_file % replace_dict

            self.matrix_methods[process_string] = method_file

        return self.matrix_methods



    def generate_subprocess_directory(self, subproc_group,
                                         fortran_model, me=None):

        self.helas_writer = helas_call_writers.PythonUFOHelasCallWriter(self.model)

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
            aloha_model.compute_subset(wanted_lorentz)
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



