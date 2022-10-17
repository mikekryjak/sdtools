#!/usr/bin/env python3

from distutils.command.build_clib import build_clib
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
parser.add_argument("--build", type = str, nargs = 1, help = "Build folder to use")
parser.add_argument("--np", type = int, nargs = "?", help = "Number of cores to use")
parser.add_argument("--check", action = "store_true", help = "Show log after deploying?")

args = parser.parse_args()

cwd = os.getcwd()
sep = os.path.sep

casepath = cwd + sep + args.case
sd1dpath = "/ssd_scratch/SD1D-mk/"
runscriptpath = os.path.join(casepath,"run.sh")

if args.build != None:
    build_folder = args.build[0]
else:
    build_folder = "build"

print(f"Running case {args.case} in {build_folder}")

if args.restart:
    restartappend = "restart"
if args.append:
    restartappend = "append"
if args.restart and args.append:
    restartappend = "restart append"
if args.restart == False and args.append == False:
    restartappend = ""

if args.np != None:
    mpicommand = f"mpirun -np {args.np}"
    parallel = True
else:
    mpicommand = ""
    parallel = False

runcommand = f'{mpicommand} {sd1dpath}{build_folder}/sd1d -d {casepath} {restartappend}'
with open(runscriptpath, "w") as f:
    f.write(runcommand)

command = shlex.split(f"screen -dmS {args.case} sh {runscriptpath}")

print(f"Running command: {runcommand}")
run(command)

if args.restart:
    print("---> RESTARTING")
if args.append:
    print("---> APPENDING")
if parallel:
    print(f"---> PARALLEL ON {args.np} CORES")

if args.check:
    print("---> Case {} deployed. Here is what it's doing:".format(args.case))
    time.sleep(5)
    show_log(args.case, 10)

