
import os, argparse
from time import time as tm
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from alohaflow.lhe_writer import EventFileFlow, FourMomentumFlow

def top_hists(lhe, nbins=50, printings=False):
    """
    
    Computes pT, Pseudorapidity, histograms from LHE event file.

    Parameters
    ----------
        lhe: EventFileFlow, LHE events file
        nbins: int, number of histogram bins

    Returns
    -------
        np.array, containing histogram
        np.array, containing bin edges
    """
    pt_bins = np.linspace(0,300,nbins+1)
    eta_bins = np.linspace(-4,4,nbins+1)
    pts  = []
    etas = []
    wgts = []
    nb_kept = 0
    for event in lhe:
        etaabs = 0 
        etafinal = 0
        for particle in event:
            if particle.status==1 and particle.pid==6:
                p = FourMomentumFlow(particle)
                eta = p.pseudorapidity
                pt  = p.pt
                if abs(eta) > etaabs:
                    etafinal = eta
                    etaabs = abs(eta)
        if etaabs < 4:
            nb_kept += 1
            etas.append(etafinal)
            wgts.append(event.wgt)
            pts.append(pt)
    wgts = np.array(wgts)
    pt_hist = np.histogram(pts, bins=pt_bins, weights=wgts/nb_kept)
    eta_hist = np.histogram(etas, bins=eta_bins, weights=wgts/nb_kept)

    return pt_hist, eta_hist

def plot_hist(hist_flow, hist_mg5, xlabel, fname):
    """
    Plots madflow vs mg5 histograms.

    Parameters
    ----------
        hist_flow: list, madflow histogram weights and bin edges
        hist_mg5: list, mg5 histogram weights and bin edges
        xlabel: str, label of x axis
        fname: Path, plot file name
    """
    fig = plt.figure()
    gs = fig.add_gridspec(nrows=5, ncols=1, wspace=0.05)
    ax = fig.add_subplot(gs[:-1])
    ax.title.set_text('g g > t t~')

    h_flow, bins_flow = hist_flow
    ax.step(bins_flow[:-1], h_flow, where='post', label='madflow', lw=0.75)
    h_mg5, bins_mg5 = hist_mg5
    ax.step(bins_mg5[:-1], h_mg5, where='post', label='mg5_aMC', lw=0.75)
    ax.tick_params(axis='x', which='both', direction='in',
                   bottom=True, labelbottom=False,
                   top=True, labeltop=False)
    ax.tick_params(axis='y', which='both', direction='in',
                   left=True, labelleft=True,
                   right=True, labelright=False)
    ax.legend()

    ax = fig.add_subplot(gs[-1])
    ax.set_ylabel("Ratio")
    ax.step(bins_flow[:-1], (h_flow-h_mg5)/h_mg5, where='post', lw=0.75)
    ax.plot(
        [bins_flow[0], bins_flow[-2]], [0,0],
        lw=0.8, color='black', linestyle='dashed'
        )
    ax.set_xlabel(xlabel, loc='right')
    ax.set_ylim([-1,1])
    ax.tick_params(axis='x', which='both', direction='in',
                   bottom=True, labelbottom=True,
                   top=True, labeltop=False)
    ax.tick_params(axis='y', which='both', direction='in',
                   left=True, labelleft=True,
                   right=True, labelright=False)

    print(f"Saved histogram at {fname}")
    plt.savefig(fname.as_posix(), bbox_inches='tight', dpi=300)
    plt.close()


def main():
    arger = argparse.ArgumentParser(
        """
    Example script to compare madflow-mg5 histograms on some observables.
    """
    )
    arger.add_argument(
        "--madflow", help="Path to the madflow output folder", type=Path
    )
    arger.add_argument(
        "--mg5", help="Path to the mg5_aMC output folder", type=Path
    )
    arger.add_argument(
        "--nbins", help="Path to the mg5_aMC output folder", type=int, default=30
    )
    arger.add_argument(
        "--run", help="Madflow run folder name", type=str, default="run_01"
    )
    args = arger.parse_args()
    path_flow = args.madflow.joinpath(f'Events/{args.run}/unweighted_events.lhe.gz')
    path_mg5 = args.mg5.joinpath('Events/run_01/unweighted_events.lhe.gz')
    if not (path_flow.exists() and path_mg5.exists()):
        raise FileNotFoundError(f"LHE files with unweighted events do not exist")
    
    lhe_flow = EventFileFlow(path_flow)
    lhe_mg5 = EventFileFlow(path_mg5)

    print(f"Filling MadFlow histograms with {len(lhe_flow)} events")
    pt_flow, eta_flow = top_hists(lhe_flow, args.nbins, printings=True)

    print(f"Filling mg5_aMC histograms with {len(lhe_mg5)} events")
    pt_mg5, eta_mg5 = top_hists(lhe_mg5, args.nbins)

    lhe_folder = path_flow.parent
    plot_hist(pt_flow, pt_mg5, 'top pT [MeV]', lhe_folder.joinpath('top_pt.png'))
    plot_hist(eta_flow, eta_mg5, 'top \N{GREEK SMALL LETTER ETA}', lhe_folder.joinpath('top_eta.png'))


if __name__ == '__main__':
    start = tm()
    main()
    print(f"Program done in {tm()-start} s")
