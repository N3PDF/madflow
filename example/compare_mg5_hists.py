
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
    pts  = []
    etas = []
    wgts = []
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
            etas.append(etafinal)
            wgts.append(event.wgt)
            pts.append(pt)
    
    pt_hist = np.histogram(pts, nbins, ) #weights=wgts)
    eta_hist = np.histogram(etas, nbins, ) #weights=wgts)

    return pt_hist, eta_hist

def plot_hist(hist_flow, hist_mg5, xlabel, fname):
    """
    Plots madflow vs mg5 histograms.

    Parameters
    ----------
        hist_flow: list, madflow histogram weights and bin edges
        hist_mg5: list, mg5 histogram weights and bin edges
        xlabel: str, label of x axis
        fname: str, plot file name
    """
    gs1 = gridspec.GridSpec(2, 1, height_ratios=[5,1])
    gs1.update(wspace=0, hspace=0) # set the spacing between axes. 
    ax = plt.subplot(gs1[0])

    h_flow, bins_flow = hist_flow
    flow_n, flow_bins, flow_patches = ax.hist(
                bins_flow[:-1], bins_flow, weights=h_flow,
                histtype='step', label='madflow'
                                             )
    h_mg5, bins_mg5 = hist_mg5
    flow_n, flow_bins, flow_patches = ax.hist(
                bins_mg5[:-1], bins_mg5, weights=h_mg5,
                histtype='step', label='mg5_aMC'
                                             )
    ax_c = ax.twinx()
    ax_c.set_ylabel('MadFlow')
    ax_c.yaxis.set_label_coords(1.01, 0.25)
    ax_c.set_yticks(ax.get_yticks())
    ax_c.set_yticklabels([])
    ax.set_xlabel(xlabel, loc='right')
    ax.legend()

    ax.title.set_text('g g > t t~')
    plt.axis('on')
    plt.xlabel('weight ratio')
    print(f"Saved histogram at {fname}")
    plt.savefig(fname, bbox_inches='tight', dpi=200)
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
        "--nbins", help="Path to the mg5_aMC output folder", type=int, default=50
    )
    args = arger.parse_args()
    path_flow = args.madflow.joinpath('Events/run_01/unweighted_events.lhe.gz')
    path_mg5 = args.mg5.joinpath('Events/run_01/unweighted_events.lhe.gz')
    if path_flow.exists() and path_mg5.exists():
        path_lhe_flow = path_flow.as_posix()
        path_lhe_mg5 = path_mg5.as_posix()
    else:
        raise FileNotFoundError(f"LHE files with unweighted events do not exist")
    
    lhe_flow = EventFileFlow(path_lhe_flow)
    lhe_mg5 = EventFileFlow(path_lhe_mg5)

    print(f"Filling MadFlow histograms with {len(lhe_flow)} events")
    print(f"Filling mg5_aMC histograms with {len(lhe_mg5)} events")

    pt_flow, eta_flow = top_hists(lhe_flow, args.nbins, printings=True)
    pt_mg5, eta_mg5 = top_hists(lhe_mg5, args.nbins)

    lhe_folder = os.path.dirname(path_lhe_flow)
    plot_hist(pt_flow, pt_mg5, 'top pT [MeV]', os.path.join(lhe_folder, 'top_pt.png'))
    plot_hist(eta_flow, eta_mg5, 'top \N{GREEK SMALL LETTER ETA}', os.path.join(lhe_folder, 'top_eta.png'))


if __name__ == '__main__':
    start = tm()
    main()
    print(f"Program done in {tm()-start} s")
