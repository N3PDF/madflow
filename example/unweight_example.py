import os, argparse
from vegasflow import run_eager
from alohaflow.lhe_writer import EventFileFlow, FourMomentumFlow
from pathlib import Path

def do_unweighting(wgt_path, unwgt_path=None):
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
    lhe.unweight(unwgt_path)
    return lhe

def collect_rapidity(filename):
    print(f"Collecting rapidity from events at {filename}")
    lhe = EventFileFlow(filename)
    nbins = 100
        
    nb_pass = 0
    data = []
    wgts = []
    for event in lhe:
        etaabs = 0 
        etafinal = 0
        for particle in event:
            if particle.status==1:
                p = FourMomentumFlow(particle)
                eta = p.pseudorapidity
                if abs(eta) > etaabs:
                    etafinal = eta
                    etaabs = abs(eta)
        if etaabs < 4:
            data.append(etafinal)
            wgts.append(event.wgt)
            nb_pass +=1
    print(f"Events passing rapidity cuts: {nb_pass}")
    return  nb_pass, data, nbins, wgts

def main():
    arger = argparse.ArgumentParser(
        """
    Example script to integrate Madgraph tensorflow compatible generated matrix element.

    In order to generate comparable results it is necessary to set the seed (-s) and not compile the integrand
        ~$ ./integrate_example.py -s 4 -r
    results are expected to be equal.

    It is also possible to run both at the same time and get equal results by setting eager mode
    so that both runs are truly independent.
        ~$ ./integrate_example.py -s 4 -e

    """
    )
    arger.add_argument("-p", "--path", help="Path with the madgraph matrix element", type=Path)
    arger.add_argument("-u", "--unweight", help="Run eager", action="store_true")
    arger.add_argument("-e", "--eager", help="Run eager", action="store_true")
    args = arger.parse_args()
    
    if args.eager:
        run_eager(True)
    
    if args.path:
        if not args.path.exists():
            raise ValueError(f"Cannot find {args.path}")
        matrix_elm_folder = args.path.as_posix()
    else:
        matrix_elm_folder = "../../mg5amcnlo/vegasflow_example"
    
    lhe_folder = os.path.join(matrix_elm_folder, 'Events/run_1')
    weighted_path = os.path.join(lhe_folder, 'weighted_events.lhe.gz')
    unweighted_path = os.path.join(lhe_folder, 'unweighted_events.lhe.gz')

    if args.unweight:
        lhe = do_unweighting(weighted_path)

    if True:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        wgt_nb_pass, wgt_data, wgt_nbins, wgt_wgts = collect_rapidity(weighted_path)
        unwgt_nb_pass, unwgt_data, unwgt_nbins, unwgt_wgts = collect_rapidity(unweighted_path)
                        
        gs1 = gridspec.GridSpec(2, 1, height_ratios=[5,1])
        gs1.update(wspace=0, hspace=0) # set the spacing between axes. 
        ax = plt.subplot(gs1[0])
        
        label = f"wgt, {wgt_nb_pass} entries"
        wgt_n, wgt_bins, wgt_patches = ax.hist(wgt_data, wgt_nbins, weights=wgt_wgts, histtype='step', label=label)
        label = f"unwgt, {unwgt_nb_pass} entries"
        unwgt_n, unwgt_bins, unwgt_patches = ax.hist(unwgt_data, unwgt_nbins, weights=unwgt_wgts, histtype='step', label=label)
        ax_c = ax.twinx()
        ax_c.set_ylabel('MadGraph5_aMC@NLO')
        ax_c.yaxis.set_label_coords(1.01, 0.25)
        ax_c.set_yticks(ax.get_yticks())
        ax_c.set_yticklabels([])
        ax.set_xlim([-4,4])
        ax.legend(loc='upper left')

        # print("bin value:", wgt_n)
        # print("start/end point of bins", wgt_bins)
        # print("bin value:", unwgt_n)
        # print("start/end point of bins", unwgt_bins)

        plt.axis('on')
        plt.xlabel('weight ratio')
        fname = os.path.join(lhe_folder, 'rapidity.png')
        plt.savefig(fname, bbox_inches='tight', dpi=200)

if __name__=="__main__":
    main()