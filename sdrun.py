#!/usr/bin/env python3

import subprocess
import os
import argparse


parser = argparse.ArgumentParser(description = "Run case")
parser.add_argument("case", type=str, help = "Case to run")
args = parser.parse_args()

cwd = os.getcwd()
sep = os.path.sep

casepath = cwd + sep + args.case
sd1dpath = "/ssd_scratch/SD1D-mk/build"

os.chdir(sd1dpath)
print(casepath)

subprocess.call(["screen", "-dmS", args.case, "./sd1d", "-d", casepath])

print("-> Case {} running".format(args.case))
