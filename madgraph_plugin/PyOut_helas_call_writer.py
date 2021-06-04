#####################################################
#                                                   #
#  Source file of the Helas writer for              #
#  the PyOut MG5aMC plugin.                         #
#                                                   #
#####################################################

import madgraph.iolibs.helas_call_writers as helas_call_writers
import madgraph.core.helas_objects as helas_objects
import aloha
import aloha.aloha_writers as aloha_writers

class PyOutUFOHelasCallWriter(helas_call_writers.PythonUFOHelasCallWriter):
    """a modified version of the PythonUFOHelasCallWriter, which behaves properly with
    tensorflow
    """


    def generate_helas_call(self, argument, gauge_check=False):
        """Routine for automatic generation of Python Helas calls
        according to just the spin structure of the interaction.
        """

        if not isinstance(argument, helas_objects.HelasWavefunction) and \
           not isinstance(argument, helas_objects.HelasAmplitude):
            raise self.PhysicsObjectError("get_helas_call must be called with wavefunction or amplitude")
        
        call_function = None

        if isinstance(argument, helas_objects.HelasAmplitude) and \
           argument.get('interaction_id') == 0:
            call = "#"
            call_function = lambda amp: call
            self.add_amplitude(argument.get_call_key(), call_function)
            return

        if isinstance(argument, helas_objects.HelasWavefunction) and \
               not argument.get('mothers'):
            # String is just IXXXXX, OXXXXX, VXXXXX or SXXXXX
            call = "w%d = "

            call = call + helas_call_writers.HelasCallWriter.mother_dict[\
                argument.get_spin_state_number()].lower()
            # Fill out with X up to 6 positions
            call = call + 'x' * (12 - len(call))
            call = call + "(all_ps[:,%d],"
            if argument.get('spin') != 1:
                # For non-scalars, need mass and helicity
                if gauge_check and argument.get('spin') == 3 and \
                                                 argument.get('mass') == 'ZERO':
                    call = call + "%s, 4,"
                else:
                    call = call + "%s,hel[%d],"
            call = call + "float_me(%+d))"
            if argument.get('spin') == 1:
                call_function = lambda wf: call % \
                                (wf.get('me_id')-1,
                                 wf.get('number_external')-1,
                                 # For boson, need initial/final here
                                 (-1)**(wf.get('state') == 'initial'))
            elif argument.is_boson():
                if not gauge_check or argument.get('mass') != 'ZERO':
                    call_function = lambda wf: call % \
                                (wf.get('me_id')-1,
                                 wf.get('number_external')-1,
                                 wf.get('mass'),
                                 wf.get('number_external')-1,
                                 # For boson, need initial/final here
                                 (-1)**(wf.get('state') == 'initial'))
                else:
                    call_function = lambda wf: call % \
                                (wf.get('me_id')-1,
                                 wf.get('number_external')-1,
                                 'ZERO',
                                 # For boson, need initial/final here
                                 (-1)**(wf.get('state') == 'initial'))
            else:
                call_function = lambda wf: call % \
                                (wf.get('me_id')-1,
                                 wf.get('number_external')-1,
                                 wf.get('mass'),
                                 wf.get('number_external')-1,
                                 # For fermions, need particle/antiparticle
                                 -(-1)**wf.get_with_flow('is_part'))
        else:
            # String is LOR1_0, LOR1_2 etc.
            
            if isinstance(argument, helas_objects.HelasWavefunction):
                outgoing = argument.find_outgoing_number()
            else:
                outgoing = 0

            # Check if we need to append a charge conjugation flag
            l = [str(l) for l in argument.get('lorentz')]
            flag = []
            if argument.needs_hermitian_conjugate():
                flag = ['C%d' % i for i in argument.get_conjugate_index()]
                
                
            # Creating line formatting:
            call = '%(out)s= %(routine_name)s(%(wf)s%(coup)s%(mass)s)'
            # compute wf
            arg = {'routine_name': aloha_writers.combine_name(\
                                            '%s' % l[0], l[1:], outgoing, flag, True),
                   'wf': ("w%%(%d)d," * len(argument.get('mothers'))) % \
                                      tuple(range(len(argument.get('mothers')))),
                    'coup': ("%%(coup%d)s," * len(argument.get('coupling'))) % \
                                     tuple(range(len(argument.get('coupling'))))           
                   }

            if isinstance(argument, helas_objects.HelasWavefunction):
                arg['out'] = 'w%(out)d'
                if aloha.complex_mass:
                    arg['mass'] = "%(CM)s"
                else:
                    arg['mass'] = "%(M)s,%(W)s"
            else:
                arg['coup'] = arg['coup'][:-1] #removing the last coma
                arg['out'] = 'amp%(out)d'
                arg['mass'] = ''
                
            call = call % arg
            # Now we have a line correctly formatted
            call_function = lambda wf: call % wf.get_helas_call_dict(index=0)
                
            routine_name = aloha_writers.combine_name(
                                        '%s' % l[0], l[1:], outgoing, flag)
        
        # Add the constructed function to wavefunction or amplitude dictionary
        if isinstance(argument, helas_objects.HelasWavefunction):
            if not gauge_check:
                self.add_wavefunction(argument.get_call_key(), call_function)
        else:
            self.add_amplitude(argument.get_call_key(), call_function)

        return call_function
