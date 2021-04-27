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
        evt = EventFlow(info)

        particles = [ParticleFlow(pinfo, event=evt) for pinfo in p_info]
        evt.add_particles(particles)
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