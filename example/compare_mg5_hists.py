#!/usr/bin/env python3
"""
    Script for the generation of plots based on Madflow/Madgraph
    LHE output
"""
import os, argparse
from time import time as tm
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from numpy.lib import histograms

from madflow.lhe_writer import EventFileFlow, FourMomentumFlow


def top_hists(lhe, nbins=50, printings=False):
    """

    Computes pT, Pseudorapidity, histograms from LHE event file.

    Parameters
    ----------
        lhe: EventFileFlow, LHE events file
        nbins: int, number of histogram bins

    Returns
    -------
        pt_hist: np.array, containing histogram
        eta_hist: np.array, containing bin edges
        weight: float, weight normalization factor for each event

    """
    pt_bins = np.linspace(0, 300, nbins + 1)
    eta_bins = np.linspace(-4, 4, nbins + 1)
    pts = []
    etas = []
    wgts = []
    nb_kept = 0
    for event in lhe:
        etaabs = 0
        etafinal = 0
        for particle in event:
            if particle.status == 1 and particle.pid == 6:
                p = FourMomentumFlow(particle)
                eta = p.pseudorapidity
                pt = p.pt
                if abs(eta) > etaabs:
                    etafinal = eta
                    etaabs = abs(eta)
        if etaabs < 4:
            nb_kept += 1
            etas.append(etafinal)
            wgts.append(event.wgt)
            pts.append(pt)

    wgts = np.array(wgts)
    np.testing.assert_allclose(
        np.abs(wgts), np.full_like(wgts, abs(wgts[0])), np.finfo(wgts.dtype).eps
    )
    weights = wgts / nb_kept

    pt_hist = np.histogram(pts, bins=pt_bins, weights=weights)
    eta_hist = np.histogram(etas, bins=eta_bins, weights=weights)

    return pt_hist, eta_hist, weights[0]


def plot_hist(hists, wgts, errs, xlabel, fname):
    """
    Plots madflow vs mg5 histograms.

    Parameters
    ----------
        hists: tuple, of np.arrays, madflow and mg5 histogram weights and bin edges
        wgts: tuple, of floats, madflow and mg5 event weight normalization factor
        errs: tuple, of floats, madflow and mg5 cross section errors
        xlabel: str, label of x axis
        fname: Path, plot file name
    """

    hist_flow, hist_mg5 = hists
    wgt_flow, wgt_mg5 = wgts
    err_flow, err_mg5 = errs

    fig = plt.figure()
    gs = fig.add_gridspec(nrows=5, ncols=1, wspace=0.05)

    # histogram
    ax = fig.add_subplot(gs[:-1])
    ax.title.set_text("g g > t t~")

    h_flow, bins_flow = hist_flow
    err = np.sqrt(h_flow / wgt_flow) * err_flow  # propagate error
    ax.step(
        bins_flow[:-1], h_flow, where="post", label="madflow", lw=0.75, color="blue"
    )
    ax.fill_between(
        bins_flow[:-1],
        h_flow - err,
        h_flow + err,
        step="post",
        color="blue",
        alpha=0.5,
    )
    h_mg5, bins_mg5 = hist_mg5
    err = np.sqrt(h_mg5 / wgt_mg5) * err_mg5  # propagate error
    ax.step(
        bins_mg5[:-1], h_mg5, where="post", label="mg5_aMC", lw=0.75, color="orange"
    )
    ax.fill_between(
        bins_mg5[:-1], h_mg5 - err, h_mg5 + err, step="post", color="orange", alpha=0.5
    )
    ax.tick_params(
        axis="x",
        which="both",
        direction="in",
        bottom=True,
        labelbottom=False,
        top=True,
        labeltop=False,
    )
    ax.tick_params(
        axis="y",
        which="both",
        direction="in",
        left=True,
        labelleft=True,
        right=True,
        labelright=False,
    )
    ax.legend()

    # madflow to mg5 percentage difference
    ax = fig.add_subplot(gs[-1])
    ax.set_ylabel("Ratio")
    ratio = (h_flow - h_mg5) / h_mg5
    err = np.sqrt((err_flow / h_mg5) ** 2 + (err_mg5 * h_flow / h_mg5 ** 2) ** 2)
    ax.step(bins_flow[:-1], ratio, where="post", lw=0.75)
    ax.fill_between(
        bins_flow[:-1], ratio - err, ratio + err, step="post", color="blue", alpha=0.2
    )
    ax.plot(
        [bins_flow[0], bins_flow[-2]], [0, 0], lw=0.8, color="black", linestyle="dashed"
    )
    ax.set_xlabel(xlabel, loc="right")
    ax.set_ylim([-1, 1])
    ax.tick_params(
        axis="x",
        which="both",
        direction="in",
        bottom=True,
        labelbottom=True,
        top=True,
        labeltop=False,
    )
    ax.tick_params(
        axis="y",
        which="both",
        direction="in",
        left=True,
        labelleft=True,
        right=True,
        labelright=False,
    )

    print(f"Saved histogram at {fname}")
    plt.savefig(fname.as_posix(), bbox_inches="tight", dpi=300)
    plt.close()


def main():
    """
    Example script to compare madflow-mg5 histograms on some observables.
    """
    arger = argparse.ArgumentParser(main.__doc__)
    arger.add_argument(
        "--madflow",
        help="Path to folder where madflow unweighted events are",
        type=Path,
    )
    arger.add_argument("--mg5", help="Path to the mg5_aMC output folder", type=Path)
    arger.add_argument(
        "--nbins", help="Number of bins in the histogram", type=int, default=30
    )
    args = arger.parse_args()
    unw_filename = "unweighted_events.lhe.gz"
    path_flow = args.madflow / unw_filename
    path_mg5 = args.mg5 / "Events/run_01" / unw_filename
    if not path_flow.exists():
        raise FileNotFoundError(f"LHE file for madflow not found at: {path_flow}")
    if not path_mg5.exists():
        raise FileNotFoundError(f"LHE file for madgraph not found at: {path_mg5}")

    lhe_flow = EventFileFlow(path_flow)
    _, err_flow = (0, 0.1)  # TODO: implement lhe banner

    lhe_mg5 = EventFileFlow(path_mg5)
    _, err_mg5 = lhe_mg5.get_banner().get_cross(witherror=True)

    print(f"Filling MadFlow histograms with {len(lhe_flow)} events")
    pt_flow, eta_flow, wgts_flow = top_hists(lhe_flow, args.nbins, printings=True)

    print(f"Filling mg5_aMC histograms with {len(lhe_mg5)} events")
    pt_mg5, eta_mg5, wgts_mg5 = top_hists(lhe_mg5, args.nbins)

    lhe_folder = path_flow.parent
    plot_hist(
        (pt_flow, pt_mg5),
        (wgts_flow, wgts_mg5),
        (err_flow, err_mg5),
        "top pT [MeV]",
        lhe_folder.joinpath("top_pt.png"),
    )
    plot_hist(
        (eta_flow, eta_mg5),
        (wgts_flow, wgts_mg5),
        (err_flow, err_mg5),
        "top \N{GREEK SMALL LETTER ETA}",
        lhe_folder.joinpath("top_eta.png"),
    )


if __name__ == "__main__":
    start = tm()
    main()
    print(f"Program done in {tm()-start} s")
