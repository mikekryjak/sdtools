#!/usr/bin/env python3

import subprocess
import os
import argparse


parser = argparse.ArgumentParser(description = "Run case")
parser.add_argument("case", type=str, help = "Case to run")
parser.add_argument("--l", action="store_true", help = "open log?")

args = parser.parse_args()

cwd = os.getcwd()
sep = os.path.sep

if args.l:
    path = os.path.join(args.case, "BOUT.log.0")
else:
    
    path = os.path.join(args.case, "BOUT.inp")


subprocess.call(["nano", path])


