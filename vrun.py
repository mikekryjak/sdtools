#!/usr/bin/env python3

from distutils.command.build_clib import build_clib
import subprocess, shlex
import os
import argparse
import numpy as np
from subprocess import PIPE, run
import time


# https://realpython.com/python-subprocess/

parser = argparse.ArgumentParser(description = "Run case")
parser.add_argument("casepath", type=str, help = "Case to run")
parser.add_argument("--restart", action="store_true", help = "Restart?")
parser.add_argument("--append", action="store_true", help = "Append?")
parser.add_argument("--build", type = str, nargs = 1, help = "Build folder to use")

args = parser.parse_args()

abscasepath = os.path.join(os.getcwd(), args.casepath)
casename = os.path.basename(os.path.normpath(args.casepath))

sd1dpath = "/ssd_scratch/SD1D-mk/"
runscriptpath = os.path.join(abscasepath, f"run.sh")

if args.build != None:
    build_folder = args.build[0]
else:
    build_folder = "build"

# print(f"Running case {args.case} in {build_folder}")

if args.restart:
    restartappend = "restart"
if args.append:
    restartappend = "append"
if args.restart and args.append:
    restartappend = "restart append"
if args.restart == False and args.append == False:
    restartappend = ""

jobname = casename
nodes = 1
cores = 24
partition = "nodes"
time = "48:00:00"

slurmcommand = \
f"""#!/bin/bash 
#SBATCH -J {jobname}
#SBATCH -N {nodes}
#SBATCH --tasks-per-node={cores}
#SBATCH -p {partition}
#SBATCH --time={time}
#SBATCH -o /mnt/lustre/users/mjk557/cases/slurmlogs/{jobname}.out
#SBATCH -e /mnt/lustre/users/mjk557/cases/slurmlogs/{jobname}.err

mpirun -n {nodes*cores} /mnt/lustre/users/mjk557/hermes-3/build/hermes-3 -d {abscasepath} {restartappend}

"""

with open(runscriptpath, "w") as f:
    f.write(slurmcommand)

print("Made slurm script at ", runscriptpath)
# run(f"sbatch", "{runscriptpath}")

# print(f"Exectuted running of {jobname}")
