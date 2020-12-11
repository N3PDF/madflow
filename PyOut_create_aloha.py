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
import aloha
import aloha.create_aloha as create_aloha
import aloha.aloha_writers as aloha_writers
import PyOut_PythonFileWriter as PythonFileWriter


class ALOHAWriterForPyOut(aloha_writers.ALOHAWriterForPython):
    """identical to the mother class, just set the writer to the
    one in PyOut_PythonFileWriter
    """
    writer = PythonFileWriter.PyOutPythonWriter


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
        writer = ALOHAWriterForPyOut(self, output_dir)
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

