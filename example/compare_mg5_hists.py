#!/usr/bin/env python3
"""
    Script for the generation of plots based on Madflow/Madgraph
    LHE output
"""
import argparse
from time import time as tm
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from madflow.lhe_writer import EventFileFlow, FourMomentumFlow


def top_hists(lhe, nbins=50):
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
    weights = wgts / nb_kept

    pt_hist = np.histogram(pts, bins=pt_bins, weights=weights)
    sqrtN = np.sqrt(np.histogram(pts, bins=pt_bins)[0])
    err_pt = pt_hist[0] / sqrtN

    eta_hist = np.histogram(etas, bins=eta_bins, weights=weights)
    sqrtN = np.sqrt(np.histogram(etas, bins=eta_bins)[0])
    err_eta = eta_hist[0] / sqrtN

    return (pt_hist, err_pt), (eta_hist, err_eta)


def plot_hist(hist_flow, hist_mg5, xlabel, fname):
    """
    Plots madflow vs mg5 histograms.

    Parameters
    ----------
        hist_flow: tuple, madflow histogram weights, bin edges and error bars
        hist_mg5: tuple, mg5 histogram weights, bin edges and error bars
        xlabel: str, label of x axis
        fname: Path, plot file name
    """
    plt.rcParams.update({"savefig.format": "pdf"})
    fig = plt.figure()
    gs = fig.add_gridspec(nrows=5, ncols=1, wspace=0.05)
    ax = fig.add_subplot(gs[:-1])
    ax.title.set_text("g g > t t~")

    (h_flow, bins_flow), h_err_flow = hist_flow
    ax.step(
        bins_flow[:-1], h_flow, where="post", label="madflow", lw=0.75, color="blue"
    )
    ax.fill_between(
        bins_flow[:-1],
        h_flow - h_err_flow,
        h_flow + h_err_flow,
        step="post",
        color="blue",
        alpha=0.3,
    )

    (h_mg5, bins_mg5), h_err_mg5 = hist_mg5
    ax.step(
        bins_mg5[:-1], h_mg5, where="post", label="mg5_aMC", lw=0.75, color="orange"
    )
    ax.fill_between(
        bins_mg5[:-1],
        h_mg5 - h_err_mg5,
        h_mg5 + h_err_mg5,
        step="post",
        color="orange",
        alpha=0.3,
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

    ax = fig.add_subplot(gs[-1])
    h_ratio = h_flow / h_mg5
    h_ratio_err = h_ratio * np.sqrt(
        (h_err_flow / h_flow) ** 2 + (h_err_mg5 / h_mg5) ** 2
    )

    ax.set_ylabel("Ratio")
    ax.step(bins_flow[:-1], h_ratio, where="post", lw=0.75, color="blue")
    ax.fill_between(
        bins_flow[:-1],
        h_ratio - h_ratio_err,
        h_ratio + h_ratio_err,
        step="post",
        color="blue",
        alpha=0.3,
    )
    ax.plot(
        [bins_flow[0], bins_flow[-2]], [1, 1], lw=0.8, color="black", linestyle="dashed"
    )
    ax.set_xlabel(xlabel, loc="right")
    ax.set_ylim([0.5, 1.5])
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
    print(f"Filling MadFlow histograms with {len(lhe_flow)} events")
    pt_flow, eta_flow = top_hists(lhe_flow, args.nbins)

    lhe_mg5 = EventFileFlow(path_mg5)
    print(f"Filling mg5_aMC histograms with {len(lhe_mg5)} events")
    pt_mg5, eta_mg5 = top_hists(lhe_mg5, args.nbins)

    lhe_folder = path_flow.parent
    plot_hist(
        pt_flow,
        pt_mg5,
        "top pT [MeV]",
        lhe_folder.joinpath("ggttbarpt.pdf"),
    )
    plot_hist(
        eta_flow,
        eta_mg5,
        "top \N{GREEK SMALL LETTER ETA}",
        lhe_folder.joinpath("ggttbareta.pdf"),
    )


if __name__ == "__main__":
    start = tm()
    main()
    print(f"Program done in {tm()-start} s")
