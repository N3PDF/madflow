import sys
import os
import gzip
import copy
import tensorflow as tf
import numpy as np
from time import time as tm

import collections
momentum = collections.namedtuple("Momentum", ['E', 'px', 'py', 'pz'])

### go to the madgraph folder and load up anything that you need
original_path = copy.copy(sys.path)
mg5amcnlo_folder = '../../../mg5amcnlo'

__package__ = "madgraph.various"
sys.path.insert(0, mg5amcnlo_folder)
from madgraph.various import lhe_parser

sys.path = original_path
__package__ = None
################################################


def set_attributes_from_dict(obj, d):
    """
    Parameters
    ----------
        - obj : obj, class instance
        - d   : dict, key-value dictionary to update obj attributes
    """
    for key, value in d.items():
        setattr(obj, key, value)


def dump_lhe_banner(out):
    """
    Dump the LHE banner. This includes custom and mandatory (<init>) XML tokens.

    Parameters
    ----------
        - out       : _io.TextIOWrapper, output file object
    """
    pass


def dump_lhe_events(evt_info, part_info, out):
    """
    Get the vectorized information stored in a dict. Loop over events and
    particles to dump into LHE file.

    Parameters
    ----------
        - evt_info  : list, list of events dict info
        - part_info : list, list particles dict info
        - out       : _io.TextIOWrapper, output file object
    """
    for info, p_info in zip(evt_info, part_info):
        evt = lhe_parser.Event()
        set_attributes_from_dict(evt, info)

        particles = []
        for info in p_info:
            p = lhe_parser.Particle(event=evt)
            set_attributes_from_dict(p, info)
            particles.append(p)
        evt.extend(particles)
        out.write(str(evt).encode('utf-8'))


def dump_lhe(mc_info, output_fname):
    """
    Dump info file in LHE format.

    Parameters
    ----------
        - mc_info : list: list of dict information [evt_info, part_info]
        - fname   : str, output filename
    """
    with gzip.open(output_fname, 'wb') as out:
    # we must generate the banner which contains the mandatory <init> tag
        dump_lhe_banner(out)
        dump_lhe_events(*mc_info, out)


def main():
    # generate some random events info
    nb_events = 10
    output_fname = '../example.lhe.gz'

    event_info = [{
        'nexternal': np.random.randint(3,7),
        'ievent': 1,
        'wgt': np.random.normal()*505.946,
        'aqcd': 0.1,
        'scale': 800,
        'aqed': 0.00729,
        'tag': '',
        'comment': ''
    } for i in range(nb_events)]

    particle_info = [[{
        'pid': np.random.choice([21,1,2,-1,-2]),
        'status': np.random.choice([-1,1,2]),
        'mother1': 0,
        'mother2': 0,
        'color1': 0,
        'color2': 0,
        'px': np.random.rand(),
        'py': np.random.rand(),
        'pz': np.random.rand(),
        'E': np.random.rand(),
        'mass': np.random.rand(),
        'vtim': 0,
        'helicity': np.random.choice([+1,-1]),
    } for i in range(info['nexternal'])] for info in event_info]

    info = [event_info, particle_info]
    dump_lhe(info, output_fname)

if __name__ == '__main__':
    start = tm()
    main()
    print(f"Program done in {tm()-start} s")