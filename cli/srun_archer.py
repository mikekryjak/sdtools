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
parser.add_argument("-b", type=str, help = "Branch (build folder name)")
parser.add_argument("-c", type=str, help = "Number of cores")
parser.add_argument("-t", type=str, help = "Time in hh:mm:ss")
parser.add_argument("-N", type=str, help = "Number of nodes")
parser.add_argument("-restart", action="store_true", help = "Restart?")
parser.add_argument("-append", action="store_true", help = "Append?")


args = parser.parse_args()

if args.b == None:
    print("Please specify branch with --b <branch_name>")
    quit()
    
if args.c == None:
    print("Please specify number of cores with --c <core_count>")
    quit()

abscasepath = os.path.join(os.getcwd(), args.casepath)
casename = os.path.basename(os.path.normpath(args.casepath))
runscriptpath = os.path.join(abscasepath, f"run.sh")

if args.restart:
    restartappend = "restart"
if args.append:
    restartappend = "append"
if args.restart and args.append:
    restartappend = "restart append"
if args.restart == False and args.append == False:
    restartappend = ""

jobname = casename
nodes = args.N
cores_per_node = int(int(args.c) / int(args.N))
partition = "standard"


slurmcommand = \
f"""#!/bin/bash 
#SBATCH --job-name={jobname}
#SBATCH --nodes={nodes}
#SBATCH --tasks-per-node={cores_per_node}
#SBATCH --account=e281-bout
#SBATCH --partition={partition}
#SBATCH --qos=standard
#SBATCH --time={args.t}
#SBATCH -o /work/e281/e281/mkryjak/slurmlogs/{jobname}.out
#SBATCH -e /work/e281/e281/mkryjak/slurmlogs/{jobname}.err
#SBATCH --mail-type=BEGIN,END,FAIL               # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=mike.kryjak@york.ac.uk        # Where to send mail

# Set the number of threads to 1
#   This prevents any threaded system libraries from automatically 
#   using threading.
export OMP_NUM_THREADS=1

source /work/e281/e281/mkryjak/bout.env
srun --distribution=block:block --hint=nomultithread /work/e281/e281/mkryjak/hermes-3/{args.b}/hermes-3 -d {abscasepath} {restartappend}
"""

with open(runscriptpath, "w") as f:
    f.write(slurmcommand)

print(f"\n** Using branch {args.b}.")
print(f"\n** Running on {args.c} cores.")
if args.restart:
    print("\n-> Restarting")
if args.append:
    print("\n-> Appending")

print(f"\n--> Made slurm script at ", runscriptpath, "\n")
# run(f"sbatch", "{runscriptpath}")

# print(f"Exectuted running of {jobname}")
