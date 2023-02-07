#!/usr/bin/env python3

import argparse
import subprocess
import re
from math import ceil

models = {
    "2080Ti": "NVIDIAGeForceRTX2080Ti",
    "V100": "TeslaV100_SXM2_32GB",
    "A100": "A100_PCIE_40GB",
}

def main(args):
    cmd = f"bsub -n {args.cpus} -W {args.time} {args.warn} -G ls_polle_s "
    if args.name is not None:
        cmd += f"-J {args.name} "
    cmd += f"-R 'rusage[mem={ceil(args.mem/args.cpus)},scratch={ceil(args.scratch/args.cpus)}]' "
    cmd += f"-R 'rusage[ngpus_excl_p={args.gpus}]' "
    cond = []
    if args.gmod is not None:
        cond.append(f"gpu_model0=={models[args.gmod]}")
    if args.gmem is not None:
        cond.append(f"gpu_mtotal0>={args.gmem}")
    if args.gpus > 0 and len(cond) > 0:
        cmd += f"-R 'select[{'&&'.join(cond)}]' "
    if args.interactive:
        cmd += "-I "
    cmd += ' '.join(args.command)
    print(cmd)
    ret = subprocess.run(
        cmd, shell=True, check=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out = ret.stdout
    print(out)
    job_id, = re.findall(r'Job <(\d+)>', out)
    print(job_id)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', '-n', type=str)
    parser.add_argument('--cpus', '-c', type=int, default=32)
    parser.add_argument('--mem', '-m', type=int, default=10_000,
                        help="Total memory (NOT per core)")
    parser.add_argument('--scratch', type=int, default=50_000)
    parser.add_argument('--time', type=str, default="24:00")
    parser.add_argument('--warn', type=str, default="-wt 5 -wa INT",
                        help="default: send an interrupt 5 minutes before the end")
    parser.add_argument('--gpus', '-g', type=int, default=1)
    parser.add_argument('--gmod', type=str, choices=list(models), help="GPU model")
    parser.add_argument('--gmem', type=int, default=10_240, help="GPU memory")
    parser.add_argument('--interactive', '-I', action="store_true")
    parser.add_argument('command', nargs=argparse.REMAINDER)
    main(parser.parse_args())
