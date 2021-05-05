import sys, os, six, gzip, copy
from time import time as tm
import math
import numpy as np
from pathlib import Path
from multiprocessing.pool import ThreadPool as Pool
import logging

logger = logging.getLogger(__name__)

### go to the madgraph folder and load up anything that you need
original_path = copy.copy(sys.path)
mg5amcnlo_folder = '../../../mg5amcnlo'

__package__ = "madgraph.various"
sys.path.insert(0, mg5amcnlo_folder)
from madgraph.various import lhe_parser

sys.path = original_path
__package__ = None

################################################

def do_unweighting(wgt_path, unwgt_path=None, event_target=0):
    """
    From an LHE file of weighted events, do unweighting and produce a new LHE
    file of unweighted events.

    Parameters
    ----------
        wgt_path: str, input LHE file to load events from
        unwgt_path: str, output file, defaults to `unweighted_events.lhe.gz`

    Returns
    -------
        EventFileFlow object
    """
    lhe = EventFileFlow(wgt_path)
    if not unwgt_path:
        fname = "unweighted_events.lhe.gz"
        unwgt_path = Path(wgt_path).with_name(fname).as_posix()
    nb_keep = lhe.unweight(unwgt_path, event_target=event_target)
    return lhe, nb_keep # does unweight method modify the lhe object ?


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
    def __init__(self, folder, run='run_01', unweight=False, event_target=0):
        """
        Utility class to write Les Houches Event (LHE) file info: writes LHE
        events to <folder>/Events/<run>/weighted_events.lhe.gz

        Parameters
        ----------
            folder: str, the matrix element folder
            run: str, the run name
            unweight: bool, wether to unweight or not events before objects goes
                      out of scope
            event_target: int, number of requested unweighted events
        """
        self.folder = folder
        self.run = run
        self.unweight = unweight
        self.event_target = event_target
        self.pool = Pool(processes=1)
        self.build_folder_and_check_streamer()

    def build_folder_and_check_streamer(self):
        # create LHE file directory tree
        lhe_folder = os.path.join(self.folder, f"Events/{self.run}")
        Path(lhe_folder).mkdir(parents=True, exist_ok=True)
        self.lhe_path = os.path.join(lhe_folder, 'weighted_events.lhe.gz')
        self.stream = gzip.open(self.lhe_path, 'wb')
    
    def __enter__(self):
        self.dump_banner()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """ Send closing signal to asynchronous dumping pool."""
        self.pool.close()
        self.pool.join()
        self.stream.write('</LesHouchesEvent>\n'.encode('utf-8'))
        logger.debug(f"Saved LHE file at {self.lhe_path}")
        self.stream.close()
        if self.unweight:
            logger.debug("Unweighting ...")
            start = tm()
            lhe, nb_keep = do_unweighting(self.lhe_path, event_target=self.event_target)
            end = tm()-start
            log = "Unweighting stats: kept %d events out of %d (efficiency %.2g %%, time %.5f)" \
                        %(nb_keep, len(lhe), nb_keep/len(lhe)*100, end)
            logger.info(log)
            # TODO: drop weighted events?


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


class EventFileFlow(lhe_parser.EventFile):
    def __init__(self, path, mode='r', *args, **kwargs):
        super().__init__(path, mode, *args, **kwargs)


class FourMomentumFlow(lhe_parser.FourMomentum):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @property
    def phi(self):
        """ Return the azimuthal angle. """
        phi = 0.0 if (self.pt == 0.0) else math.atan2(self.py, self.px)
        if (phi < 0.0):
            phi += 2.0*math.pi
        if (phi > 2.0*math.pi):
            phi -= 2.0*math.pi
        return phi
