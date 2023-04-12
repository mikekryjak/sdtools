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
parser.add_argument("-N", type=str, help = "Number of nodes")
parser.add_argument("-t", type=str, help = "Time in hh:mm:ss")
parser.add_argument("-restart", action="store_true", help = "Restart?")
parser.add_argument("-append", action="store_true", help = "Append?")


args = parser.parse_args()

if args.b == None:
    print("Please specify branch with --b <branch_name>")
    quit()
    
if args.c == None:
    print("Please specify number of cores with --c <core_count>")
    quit()
    
if args.N == None:
    print("Please specify number of nodes with --n <core_count>")
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
nodes = int(args.N)
cores_per_node = int(int(args.c)/nodes)
partition = "nodes"
time = args.t

slurmcommand = \
f"""#!/bin/bash 
#SBATCH -J {jobname}
#SBATCH -N {nodes}
#SBATCH --tasks-per-node={cores_per_node}
#SBATCH -p {partition}
#SBATCH --time={time}
#SBATCH -o /mnt/lustre/users/mjk557/cases/slurmlogs/{jobname}.out
#SBATCH -e /mnt/lustre/users/mjk557/cases/slurmlogs/{jobname}.err
#SBATCH --account=phys-bout-2019         # Project account
#SBATCH --mail-type=BEGIN,END,FAIL               # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=mike.kryjak@york.ac.uk        # Where to send mail

source /mnt/lustre/users/mjk557/bout.env
mpirun -n {nodes*cores_per_node} /mnt/lustre/users/mjk557/hermes-3/{args.b}/hermes-3 -d {abscasepath} {restartappend}

"""

with open(runscriptpath, "w") as f:
    f.write(slurmcommand)

print(f"\n** Using branch {args.b}.")
print(f"\n** Running on {args.c} cores on {args.N} nodes.")
if args.restart:
    print("\n-> Restarting")
if args.append:
    print("\n-> Appending")

print(f"\n--> Made slurm script at ", runscriptpath, "\n")
# run(f"sbatch", "{runscriptpath}")

# print(f"Exectuted running of {jobname}")


# Here is how you can get dir names, but you can't pass them to the commented out lines very easily...
"""
#!/bin/bash

# From https://stackoverflow.com/questions/27708656/pass-command-line-arguments-via-sbatch
sbatch << EOT
#!/bin/bash
# From https://stackoverflow.com/questions/59895/how-do-i-get-the-directory-where-a-bash-script-is-located-from-within-the-script
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# From https://stackoverflow.com/questions/23162299/how-to-get-the-last-part-of-dirname-in-bash
JOB_NAME $(basename $SCRIPT_DIR)

#SBATCH -J "$JOB_NAME"
#SBATCH -N 2
#SBATCH --tasks-per-node=20
#SBATCH -p nodes
#SBATCH --time=00:05:00
#SBATCH -o /mnt/lustre/users/mjk557/cases/slurmlogs/"$JOB_NAME".out
#SBATCH -e /mnt/lustre/users/mjk557/cases/slurmlogs/"$JOB_NAME".err
#SBATCH --account=phys-bout-2019         # Project account
#SBATCH --mail-type=BEGIN,END,FAIL               # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=mike.kryjak@york.ac.uk        # Where to send mail

mpirun -n 40 /mnt/lustre/users/mjk557/hermes-3/anomalous-improvements/hermes-3 -d "$SCRIPT_DIR" restart
EOT


"""