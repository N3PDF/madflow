from madflow.config import get_madgraph_path
import sys, os, six, gzip, copy
from time import time as tm
import math
import numpy as np
import tensorflow as tf
from pathlib import Path
from multiprocessing.pool import ThreadPool as Pool

from pdfflow.configflow import fzero

import logging

logger = logging.getLogger(__name__)

### go to the madgraph folder and load up anything that you need
original_path = copy.copy(sys.path)

mg5amcnlo_folder = get_madgraph_path()
sys.path.insert(0, mg5amcnlo_folder.as_posix())

original_package = __package__
__package__ = "madgraph.various"
from madgraph.various import lhe_parser

sys.path = original_path
__package__ = original_package

from collections import namedtuple

EventInfoTuple = namedtuple("EventInfo", ["x1", "x2", "scale", "aqcd"])

################################################


class EventFlow(lhe_parser.Event):
    """
    Wrapper class for madgraph lhe_parser.Event class. EventFlow deals with
    holding the LHE info for the event.
    Subclass of list class: looping over self yields ParticleFlow objects
    contained in the event.
    """

    def __init__(self, info, *args, **kwargs):
        """
        Parameters
        ----------
            info: dict | lhe_parser.Event | list, event information to be stored
        """
        super().__init__(*args, **kwargs)

        self.nexternal = info.get("nexternal")
        self.ievent = info.get("ievent")
        self.wgt = info.get("wgt")
        self.scale = info.get("scale")
        self.aqed = info.get("aqed")
        self.aqcd = info.get("aqcd")
        self.tag = info.get("tag")
        self.comment = info.get("comment")

    def add_particles(self, particles):
        """
        Parameters
        ----------
            particles: list, ParticleFlow objects list to extend the event with
        """
        self.extend(particles)

    def as_bytes(self):
        """ Returns byte string event representation. """
        return self.__str__().encode("utf-8")
    
    @property
    def err(self):
        """ Cross section and statistical error. """
        return self.__err

    @err.setter
    def err(self, value):
        """ Cross section and statistical error setter. """
        self.__err = value


class ParticleFlow(lhe_parser.Particle):
    """
    Wrapper class for madgraph lhe_parser.Particle class. Holds particle info.
    """

    def __init__(self, info, *args, **kwargs):
        """
        Parameters
        ----------
            info: dict, particle information to be stored
        """
        super().__init__(*args, **kwargs)

        self.pid = info.get("pid")
        self.status = info.get("status")
        self.mother1 = info.get("mother1")
        self.mother2 = info.get("mother2")
        self.color1 = info.get("color1")
        self.color2 = info.get("color2")
        self.px = info.get("px")
        self.py = info.get("py")
        self.pz = info.get("pz")
        self.E = info.get("E")
        self.mass = info.get("mass")
        self.vtim = info.get("vtim")
        self.helicity = info.get("helicity")


class LheWriter:
    # must set the initial states from the matrix elements
    # the number of events: because each matrix element has its own initial/final state?
    def __init__(
        self,
        folder,
        run="run_01",
        no_unweight=False,
        event_target=0,
        initial_states=None,
    ):
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
            initial_states: list, of lists of initial states for each matrix
                            element in the process
        """
        self.folder = folder
        self.run = run
        self.no_unweight = no_unweight
        self.event_target = event_target
        self.initial_states = initial_states
        self.final_states = [6, -6]  # t, t~
        self.pool = Pool(processes=1)

        # create LHE file directory tree
        lhe_folder = self.folder.joinpath(f"Events/{self.run}")
        lhe_folder.mkdir(parents=True, exist_ok=True)
        self.lhe_path = lhe_folder.joinpath("weighted_events.lhe.gz")

        # create I/O stream
        self.stream = gzip.open(self.lhe_path, "wb")

    def __enter__(self):
        self.dump_banner()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """
        Send closing signal to asynchronous dumping pool. Triggers unweighting
        if self.no_unweight is False (default).

        Note: this function should be called after having stored the cross
        section and statistical error values
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
            end = tm() - start
            log = (
                "Unweighting stats: kept %d events out of %d (efficiency %.2g %%, time %.5f)"
                % (nb_keep, nb_wgt, nb_keep / nb_wgt * 100, end)
            )
            logger.info(log)

    def lumis_to_pids(self, probs):
        """
        Draw a random number to select event initial state. Probabilities are
        weighted according to matrix elements luminosity relative importance.

        Parameters
        ----------
            - probs: np.array, matrix elements relative luminosities of
                     shape=(num matrices,)

        Returns
        -------
            - pids: list, of PDGs
            - status_vec: list, of particle status (-1|+1 for final|initial state)
        """
        ME = np.random.choice(len(probs), p=probs)
        initials = self.initial_states[ME]
        idx = int(np.random.rand() * len(initials))
        initial = initials[idx]
        status_vec = [-1] * len(initial) + [1] * len(self.final_states)
        pids = initial + self.final_states
        return pids, status_vec

    def lhe_parser(self, all_ps, res, all_lumis, evt_info):
        """
        Takes care of storing and dumping LHE info from the integrator.
        To be passed as argument to generate the Vegasflow custom integrand.

        Parameters
        ----------
            all_ps: tf.Tensor, phase space points of shape=(nevents,nexternal,ndims)
            res: tf.Tensor, weights of shape=(nevents,)
            all_lumis: tf.Tensor, luminosity values of shape=(nevents, len(matrices))
            evt_info: tf.Tensor, event info of shape=(nevents, num info),
                      each info contains x1, x2, scale, aqcd for the event
        """
        all_lumis = all_lumis / tf.reduce_sum(all_lumis, axis=1, keepdims=True)

        events_info = []
        particles_info = []
        for wgt, ps_external, lumis, info in zip(
            res.numpy(), all_ps.numpy(), all_lumis.numpy(), evt_info.numpy()
        ):
            pids, status_vec = self.lumis_to_pids(lumis)
            nexternal = len(status_vec)
            info = EventInfoTuple(*info)
            events_info.append(
                {
                    "nexternal": nexternal,
                    "ievent": 1,
                    "wgt": wgt,
                    "scale": info.scale,
                    "aqed": 0.0,  # alpha EW value , get this from vegasflow?
                    "aqcd": info.aqcd,
                    "tag": "",
                    "comment": "",
                }
            )

            # we are missing the virtual particles
            particles_info.append(
                [
                    {
                        "pid": pid,
                        "status": status,
                        "mother1": 0,
                        "mother2": 0,
                        "color1": 0,
                        "color2": 0,
                        "E": ps[0],
                        "px": ps[1],
                        "py": ps[2],
                        "pz": ps[3],
                        "mass": np.sqrt(
                            ps[0] ** 2 - ps[1] ** 2 - ps[2] ** 2 - ps[3] ** 2
                        ),  # vectorize this?
                        "vtim": 0,
                        "helicity": 0,
                    }
                    for pid, status, ps in zip(pids, status_vec, ps_external)
                ]
            )

        self.dump(events_info, particles_info)

        return fzero

    def dump_banner(self, stream=None):
        """
        Parameters
        ----------
            stream: _io.TextIOWrapper, output file object, if None use default
                    self.stream
        """
        if stream:
            stream.write("<LesHouchesEvent>\n".encode("utf-8"))
        else:
            self.stream.write("<LesHouchesEvent>\n".encode("utf-8"))

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
            self.stream.write(evt.as_bytes())

    def dump_exit(self, stream=None):
        """
        Parameters
        ----------
            stream: _io.TextIOWrapper, output file object, if None use default
                    self.stream
        """
        tag = "</LesHouchesEvent>\n"
        if stream:
            stream.write(tag.encode("utf-8"))
        else:
            self.stream.write(tag.encode("utf-8"))

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
        """ Dumps info asynchronously. """
        self.pool.apply_async(self.async_dump, args)

    @property
    def cross(self):
        """ Cross section. """
        return self.__cross

    @cross.setter
    def cross(self, value):
        """ Cross section setter. """
        self.__cross = value

    @property
    def err(self):
        """ Cross section's statistical error. """
        return self.__err

    @err.setter
    def err(self, value):
        """ Error section setter"""
        self.__err = value

    def store_result(self, result):
        """
        Stores integration result in numpy format.

        Parameters
        ----------
            result: list, cross section and statistical error
        """
        self.__cross = float(result[0])
        self.__err = float(result[1])

    def do_unweighting(self, event_target=0):
        """
        Does unweighting. Removes the weighted LHE file.

        Parameters
        ----------
            event_target: int, number of unweighted events requested

        Note: this function should be called after having stored the cross
        section and statistical error values
        """
        # load weighted LHE file
        lhe = EventFileFlow(self.lhe_path)
        nb_wgt = len(lhe)

        # open a tmp stream for unweighted LHE file
        tmp_path = self.lhe_path.with_name("tmp_unweighted_events.lhe.gz")
        # unweight
        nb_keep = lhe.unweight(tmp_path.as_posix(), event_target=event_target)

        # delete weighted LHE file
        # self.lhe_path.unlink()

        # load tmp file
        tmp_lhe = EventFileFlow(tmp_path)

        # open a stream for final unweighted LHE file
        unwgt_path = tmp_path.with_name("unweighted_events.lhe.gz")
        with gzip.open(unwgt_path, "wb") as stream:
            self.dump_banner(stream)
            for event in tmp_lhe:
                event.wgt = self.__cross
                stream.write(event.as_bytes())
            self.dump_exit(stream)

        # delete tmp file
        tmp_path.unlink()
        return nb_keep, nb_wgt


class EventFileFlow(lhe_parser.EventFile):
    """
    Wrapper class for madgraph lhe_parser.EventFile class. Loads, modifies and
    dumps the events contained in a LHE file.
    """

    def __init__(self, path, mode="r", *args, **kwargs):
        """
        Parameters
        ----------
            path: Path or str, path pointing to a valid LHE file (both with
                  .lhe or .lhe.gz extension)
            mode: str, file opening mode
        """
        if isinstance(path, Path):
            path = path.as_posix()
        super().__init__(path, mode, *args, **kwargs)

    def __next__(self):
        """
        Replacing the mother class method returnin an EventFileFlow, not an
        lhe_parser.EventFile.

        Note: This won't work with <eventgroup> (if self.eventgroup is True).
        """
        event = super().__next__()
        if isinstance(event, lhe_parser.Event):
            event.__class__ = EventFlow
        # EventFile.__len__ method loops over self and returns a list
        # instead of an Event, but the returned object is not used then. In
        # this case it's fine to return a non EventFileFlow object.
        return event


class FourMomentumFlow(lhe_parser.FourMomentum):
    """
    Wrapper class for madgraph lhe_parser.FourMomentum class. Stores (E,px,py,pz)
    of a particle and allows access to its kinematical quantities.
    """

    def __init__(self, obj=0, px=0, py=0, pz=0, E=0):
        """
        Parameters
        ----------
            obj: FourMomentumFlow|ParticleFlow|list|tuple|str|six.text_type|float
                 object to copy momentum components from.
                 - If is FourMomentumFlow or ParticleFlow this function acts like
                 a copy constructor.
                 - If is list or tuple, momentum components should be
                 (E,px,py,pz) ordered.
                 - If is str or six.text_type, a space separated string with
                 (E,px,py,pz) ordered components.
                 - If is float, superseeds the E argument
            px: float, x momentum component
            py: float, y momentum component
            pz: float, z momentum component
            E: float, particle energy
        """
        super().__init__(obj, px, py, pz, E)

    @property
    def phi(self):
        """ Returns the azimuthal angle. """
        phi = 0.0 if (self.pt == 0.0) else math.atan2(self.py, self.px)
        return phi % (2.0 * np.pi)
