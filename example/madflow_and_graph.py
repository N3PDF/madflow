#!/usr/bin/env python3
"""
    Run madgraph and madflow side to side

    First it runs madflow with whatever arguments and then
    uses said arguments to generate the appropiate madgraph call

    To ensure you get all available output, do
    export MADFLOW_LOG_LEVEL=3
"""
from time import time
from pathlib import Path
import tempfile
import subprocess as sp
from madflow.scripts.madflow_exec import madflow_main, logger
from madflow.config import get_madgraph_exe

_flav_dict = {"g": 21, "d": 1, "u": 2, "s": 3, "c": 4, "b": 5, "t": 6}
DEFAULT_PDF = "NNPDF31_nnlo_as_0118"

out_path = Path(tempfile.mkdtemp(prefix="mad"))
script_path = Path(tempfile.mktemp(prefix="mad_script"))

if __name__ == "__main__":
    logger.info("Running madflow")
    start_madflow = time()
    args, final_res, events_folder = madflow_main()
    end_madflow = time()

    logger.info("Running madgraph")

    # Prepare the madgraph script
    if args.fixed_scale is None:
        scale = "set run_card dynamical_scale_choice 3"
    else:
        qsqrt = args.fixed_scale
        scale = f"""set run_card fixed_ren_scale true
set run_card fixed_fac_scale true
set run_card scale {qsqrt}
set run_card dsqrt_q2fact1 {qsqrt}
set run_card dsqrt_q2fact2 {qsqrt}
"""

    if args.pt_cut is None:
        cuts = ""
    else:
        nparticles = len(args.madgraph_process.strip().split()) - 1
        outgoing_particles = args.madgraph_process.rsplit(" ", nparticles - 2)[1:]
        dict_cuts = {_flav_dict[i[0]]: args.pt_cut for i in outgoing_particles}
        # All pt must be above PT_CUT
        cuts = f"set run_card pt_min_pdg {dict_cuts}"
    if args.pdf != DEFAULT_PDF:
        logger.warning("Madgraph will run with pdf=%s instead of %s", DEFAULT_PDF, args.pdf)

    madgraph_script = f"""generate {args.madgraph_process}
output {out_path}
launch
set run_card nevents 10000
set run_card systematics none
set run_card pdlabel lhapdf
set run_card lhaid 303600
{scale}
{cuts}
"""
    script_path.write_text(madgraph_script)
    madgraph_run = f"{get_madgraph_exe()} -f {script_path}"

    logger.info("Running madgraph script %s", script_path)
    process = sp.run(madgraph_run, shell=True)
    logger.info("Madgraph output can be found at %s", out_path)
    logger.info("Madflow result %.4f +- %.4f fb", *final_res)

    logger.info("Madgraph took: %.4fs", time() - end_madflow)
    logger.info("Madflow took: %.4fs", end_madflow - start_madflow)


    if args.histograms:
        generate_histograms = [
            "./compare_mg5_hists.py",
            "--madflow",
            events_folder.as_posix(),
            "--mg5",
            out_path.as_posix(),
        ]
        sp.run(generate_histograms, check=True)
