import sys, os, six, gzip, copy
import numpy as np
from pathlib import Path
from multiprocessing.pool import ThreadPool as Pool

### go to the madgraph folder and load up anything that you need
original_path = copy.copy(sys.path)
mg5amcnlo_folder = '../../../mg5amcnlo'

__package__ = "madgraph.various"
sys.path.insert(0, mg5amcnlo_folder)
from madgraph.various import lhe_parser

sys.path = original_path
__package__ = None
################################################

class EventFlow(lhe_parser.Event):
    def __init__(self, info, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.nexternal = info.get('nexternal')
        self.ievent    = info.get('ievent')
        self.wgt       = info.get('wgt')
        self.aqcd      = info.get('aqcd')
        self.scale     = info.get('scale')
        self.aqed      = info.get('aqed')
        self.tag       = info.get('tag')
        self.comment   = info.get('comment')
    
    def add_particles(self, particles):
        self.extend(particles)


class ParticleFlow(lhe_parser.Particle):
    def __init__(self, info, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.pid      = info.get('pid')
        self.status   = info.get('status')
        self.mother1  = info.get('mother1')
        self.mother2  = info.get('mother2')
        self.color1   = info.get('color1')
        self.color2   = info.get('color2')
        self.px       = info.get('px')
        self.py       = info.get('py')
        self.pz       = info.get('pz')
        self.E        = info.get('E')
        self.mass     = info.get('mass')
        self.vtim     = info.get('vtim')
        self.helicity = info.get('helicity')


class LheWriter:
    def __init__(self, folder, run='run_1'):
        """
        Utility class to write Les Houches Event (LHE) file info: writes LHE
        events to <folder>/Events/<run>/weighted_events.lhe.gz

        Parameters
        ----------
            folder: str, the matrix element folder
            run: str, the run name
        
        """
        self.folder = folder
        self.run = run
        self.pool = Pool(processes=1)
        self.build_folder_and_check_streamer()

    def build_folder_and_check_streamer(self):
        # create LHE file directory tree
        lhe_folder = os.path.join(self.folder, f"Events/{self.run}")
        Path(lhe_folder).mkdir(parents=True, exist_ok=True)
        self.lhe_path = os.path.join(lhe_folder, 'weighted_events.lhe.gz')
        self.stream = gzip.open(self.lhe_path, 'wb')

    def dump_banner(self):
        self.stream.write('<LesHouchesEvent>\n'.encode('utf-8'))


    def dump_events(self, events_info, particles_info):
        """
        Get the vectorized information stored in a dict. Loop over events and
        particles to dump into LHE file.

        Parameters
        ----------
            events_info : list, list of events dict info
            particles_info : list, list particles dict info
            out : _io.TextIOWrapper, output file object
        """
        for info, p_info in zip(events_info, particles_info):
            evt = EventFlow(info)

            particles = [ParticleFlow(pinfo, event=evt) for pinfo in p_info]
            evt.add_particles(particles)
            self.stream.write(str(evt).encode('utf-8'))

    def async_dump(self, events_info, particles_info):
        """
        Dump info file in LHE format.

        Parameters
        ----------
            events_info: list, dictionaries for events info
            particles_info: list, dictionaries for particles info
        """
        self.dump_events(events_info, particles_info)
    
    def dump(self, *args):
        """
        Dumps info asynchronously.
        """
        self.pool.apply_async(self.async_dump, args)
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        """ Send closing signal to asynchronous dumping pool."""
        self.pool.close()
        self.pool.join()
        self.stream.write('</LesHouchesEvent>\n'.encode('utf-8'))
        self.stream.close()
    
    def __enter__(self):
        self.dump_banner()
        return self

class EventFileFlow(lhe_parser.EventFile):
    def __init__(self, path, mode='r', *args, **kwargs):
        super().__init__(path, mode, *args, **kwargs)

class FourMomentumFlow(lhe_parser.FourMomentum):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)