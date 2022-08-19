#!/usr/bin/env python3

import subprocess, shlex
import os
import argparse
import numpy as np
from subprocess import PIPE, run
from show_log import *
import time


# https://realpython.com/python-subprocess/

parser = argparse.ArgumentParser(description = "Run case")
parser.add_argument("case", type=str, help = "Case to run")
parser.add_argument("--restart", action="store_true", help = "Restart?")
parser.add_argument("--append", action="store_true", help = "Append?")
parser.add_argument("--b2", action="store_true", help = "Use 2nd build?")
parser.add_argument("--np", type = int, nargs = "?", help = "Number of cores to use")

args = parser.parse_args()

cwd = os.getcwd()
sep = os.path.sep

casepath = cwd + sep + args.case
sd1dpath = "/ssd_scratch/SD1D-mk/"
runscriptpath = os.path.join(casepath,"run.sh")


if args.b2 == True:
    build_folder = "build2"
else:
    build_folder = "build"

print(casepath)

if args.restart:
    restartappend = "restart"
if args.append:
    restartappend = "append"
if args.restart and args.append:
    restartappend = "restart append"
if args.restart == False and args.append == False:
    restartappend = ""

if args.np != np.nan:
    mpicommand = f"mpirun -np {args.np}"
    parallel = True
else:
    mpicommand = ""
    parallel = False

runcommand = f'{mpicommand} {sd1dpath}{build_folder}/sd1d -d {casepath} {restartappend}'
with open(runscriptpath, "w") as f:
    f.write(runcommand)

command = shlex.split(f"screen -dmS {args.case} sh {runscriptpath}")

run(command)

if args.restart:
    print("---> RESTARTING")
if args.append:
    print("---> APPENDING")
if parallel:
    print(f"---> PARALLEL ON {args.np} CORES")

print("-> Case {} deployed. Here is what it's doing:".format(args.case))

time.sleep(10)
show_log(args.case, 10)

