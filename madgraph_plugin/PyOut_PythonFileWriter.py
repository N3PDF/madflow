#####################################################
#                                                   #
#  Source file of the Matrix Elements exports for   #
#  the PyOut MG5aMC plugin.                         #
#                                                   #
#####################################################

import madgraph.iolibs.file_writers as file_writers


class PyOutPythonWriter(file_writers.FileWriter):
    
    def write_comments(self, text):
        text = '#%s\n' % text.replace('\n','\n#')
        self.write(text)

    def write_line(self, line):
        """Write a line with proper indent and splitting of long lines
        for the language in question."""
        return [line + '\n']
    
    ##def writelines(self, line):
    ##    self.writelines(line)
