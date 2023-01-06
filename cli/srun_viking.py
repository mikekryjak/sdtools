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
parser.add_argument("--b", type=str, help = "Branch (build folder name)")
parser.add_argument("--c", type=str, help = "Number of cores")
parser.add_argument("--restart", action="store_true", help = "Restart?")
parser.add_argument("--append", action="store_true", help = "Append?")


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
nodes = 2
cores = args.c
partition = "nodes"
time = "48:00:00"

slurmcommand = \
f"""#!/bin/bash 
#SBATCH -J {jobname}
#SBATCH -N {nodes}
#SBATCH --tasks-per-node={cores/2}
#SBATCH -p {partition}
#SBATCH --time={time}
#SBATCH -o /mnt/lustre/users/mjk557/cases/slurmlogs/{jobname}.out
#SBATCH -e /mnt/lustre/users/mjk557/cases/slurmlogs/{jobname}.err

mpirun -n {nodes*cores} /mnt/lustre/users/mjk557/hermes-3/{args.b}/hermes-3 -d {abscasepath} {restartappend}

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
