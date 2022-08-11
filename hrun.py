#!/usr/bin/env python3

import subprocess
import os
import argparse


parser = argparse.ArgumentParser(description = "Run case")
parser.add_argument("case", type=str, help = "Case to run")
parser.add_argument("--restart", action="store_true", help = "Restart?")
parser.add_argument("--append", action="store_true", help = "Append?")

args = parser.parse_args()

cwd = os.getcwd()
sep = os.path.sep

casepath = cwd + sep + args.case
hermespath = "/ssd_scratch/hermes3/build/"


os.chdir(hermespath)
print(casepath)

if args.restart and args.append:
    subprocess.call(["screen", "-dmS", args.case, "./hermes-3", "-d", casepath, "restart", "append"])
elif args.restart and not args.append:
    subprocess.call(["screen", "-dmS", args.case, "./hermes-3", "-d", casepath, "restart"])
elif args.append and not args.restart:
    subprocess.call(["screen", "-dmS", args.case, "./hermes-3", "-d", casepath, "append"])
else:
    subprocess.call(["screen", "-dmS", args.case, "./hermes-3", "-d", casepath])

print(f"Restart: {args.restart}, Append: {args.append}")
print("-> Case {} running".format(args.case))
