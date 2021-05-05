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
    def __init__(self, folder, run='run_01', no_unweight=False, event_target=0):
        """
        Utility class to write Les Houches Event (LHE) file info: writes LHE
        events to <folder>/Events/<run>/weighted_events.lhe.gz

        Parameters
        ----------
            folder: Path, the madflow output folder
            run: str, the run name
            no_unweight: bool, wether to unweight or not events before objects goes
                      out of scope
            event_target: int, number of requested unweighted events
        """
        self.folder = folder
        self.run = run
        self.no_unweight = no_unweight
        self.event_target = event_target
        self.pool = Pool(processes=1)
        
        # create LHE file directory tree
        lhe_folder = self.folder.joinpath(f"Events/{self.run}")
        lhe_folder.mkdir(parents=True, exist_ok=True)
        self.lhe_path = lhe_folder.joinpath('weighted_events.lhe.gz')

        # create I/O stream
        self.stream = gzip.open(self.lhe_path.as_posix(), 'wb')


    def __enter__(self):
        self.dump_banner()
        return self


    def __exit__(self, exc_type, exc_value, exc_traceback):
        """
        Send closing signal to asynchronous dumping pool. Triggers unweighting
        if self.no_unweight is False (default).
        """
        self.pool.close()
        self.pool.join()
        self.dump_exit()
        logger.debug(f"Saved LHE file at {self.lhe_path.as_posix()}")
        self.stream.close()
        if not self.no_unweight:
            logger.debug("Unweighting ...")
            start = tm()
            nb_keep, nb_wgt = self.do_unweighting(event_target=self.event_target)
            end = tm()-start
            log = "Unweighting stats: kept %d events out of %d (efficiency %.2g %%, time %.5f)" \
                        %(nb_keep, nb_wgt, nb_keep/nb_wgt*100, end)
            logger.info(log)


    def dump_banner(self, stream=None):
        """
        Parameters
        ----------
            stream: _io.TextIOWrapper, output file object, if None use default
                    self.stream
        """
        if stream:
            stream.write('<LesHouchesEvent>\n'.encode('utf-8'))
        else:
            self.stream.write('<LesHouchesEvent>\n'.encode('utf-8'))


    def dump_events(self, events_info, particles_info):
        """
        Get the vectorized information stored in a dict. Loop over events and
        particles to dump into LHE file.

        Parameters
        ----------
            events_info: list, list of events dict info
            particles_info: list, list particles dict info
        """
        for info, p_info in zip(events_info, particles_info):
            evt = EventFlow(info)

            particles = [ParticleFlow(pinfo, event=evt) for pinfo in p_info]
            evt.add_particles(particles)
            self.stream.write(str(evt).encode('utf-8'))


    def dump_exit(self, stream=None):
        """
        Parameters
        ----------
            stream: _io.TextIOWrapper, output file object, if None use default
                    self.stream
        """
        tag = '</LesHouchesEvent>\n'
        if stream:
            stream.write(tag.encode('utf-8'))
        else:
            self.stream.write(tag.encode('utf-8'))


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
    
    @property
    def cross(self):
        """ VegasFlow integrated cross section. """
        return self.__cross


    @cross.setter
    def cross(self, value):
        """ Cross section setter"""
        self.__cross = value


    @property
    def err(self):
        """ VegasFlow integrated cross section. """
        return self.__err


    @err.setter
    def err(self, value):
        """ Cross section setter"""
        self.__err = value

    
    def do_unweighting(self, event_target=0):
        """
        Does unweighting. Removes the weighted LHE file.
        
        Parameters
        ----------
            event_target: int, number of unweighted events requested
        """
        # load weighted LHE file
        lhe = EventFileFlow(self.lhe_path)
        nb_wgt = len(lhe)
        
        # open a tmp stream for unweighted LHE file
        tmp_path = self.lhe_path.with_name("tmp_unweighted_events.lhe.gz")
        # unweight
        nb_keep = lhe.unweight(tmp_path.as_posix(), event_target=event_target)

        # delete weighted LHE file
        self.lhe_path.unlink()
        
        # load tmp file
        tmp_lhe = EventFileFlow(tmp_path)

        # open a stream for final unweighted LHE file
        unwgt_path = tmp_path.with_name("unweighted_events.lhe.gz")
        with gzip.open(unwgt_path.as_posix(), 'wb') as stream:
            self.dump_banner(stream)
            for event in tmp_lhe:
                event.wgt = self.__cross
                stream.write(str(event).encode('utf-8'))
            self.dump_exit(stream)

        # delete tmp file
        tmp_path.unlink()
        return nb_keep, nb_wgt


class EventFileFlow(lhe_parser.EventFile):
    def __init__(self, path, mode='r', *args, **kwargs):
        if isinstance(path, Path):
            path = path.as_posix()
        super().__init__(path, mode, *args, **kwargs)


class FourMomentumFlow(lhe_parser.FourMomentum):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @property
    def phi(self):
        """ Returns the azimuthal angle. """
        phi = 0.0 if (self.pt == 0.0) else math.atan2(self.py, self.px)
        if (phi < 0.0):
            phi += 2.0*math.pi
        if (phi > 2.0*math.pi):
            phi -= 2.0*math.pi
        return phi
