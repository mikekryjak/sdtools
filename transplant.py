#!/usr/bin/env python3

import argparse
import os
import shutil
from clean import *

def transplant(donor, recipient):
    """
    Transplants dump and restart files from one case to another
    """
    
    path_donor = os.getcwd() + os.path.sep + donor
    path_recipient = os.getcwd() + os.path.sep + recipient
    
    # Clean recipient
    clean(recipient)
    
    # Find cases to copy
    to_copy = []
    copied = False
    
    for item in os.listdir(path_donor):
        if any([x in item for x in ["dmp", "restart"]]):
            to_copy.append(item)
            shutil.copy2(os.path.join(path_donor, item), os.path.join(path_recipient, item))
            print(f"-> Copied {item}")
            copied = True
            
    if copied == False:
        print(f"Couldn't find any dump or restart files in {donor}")
    
    else:
        print(f"Transplant completed from {donor} to {recipient}")

    
#------------------------------------------------------------
# PARSER
#------------------------------------------------------------
if __name__ == "__main__":
    
    # Define arguments
    parser = argparse.ArgumentParser(description = "SD1D options reader")
    parser.add_argument("donor", type=str, help = "Transplant from this case")
    parser.add_argument("recipient", type=str, help = "Transplant to this case")

    # Extract arguments and call function
    args = parser.parse_args()
    transplant(args.donor, args.recipient)

