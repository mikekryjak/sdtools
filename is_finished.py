from boutdata.data import BoutData
import os
import sys
import getopt
from optparse import OptionParser
import argparse


def is_finished(path, key, quiet = False):
    
    
    """
    -> is_finished(path, key, quiet = False)
    Will print status of all cases with names matching key in path
    Statuses:
    Finished - Current timestep matches intended
    Error - can't read boutdata
    Not finished - Current timestep doesn't match intended
    Not started - Input file exists but no dump file
    Missing input file - No input file
    """
    
    path = sys.argv[0]
    key = sys.argv[1]
    
    folders = os.listdir(path)
    
    for folder in folders:
        if key in folder and "." not in folder:
            
            path_folder = path + os.path.sep + folder
            
            files = os.listdir(path_folder)
            
            found_dmp = False
            found_inp = False
            boutdata_ok = False
            
            for file in files:
                if "dmp" in file:
                    found_dmp = True
                if ".inp" in file:
                    found_inp = True
            
            if found_dmp == True:
            
                try:
                    data = BoutData(path + os.path.sep + folder)
                    boutdata_ok = True
                except:
                    print("Tried to read a file that's not a case: {}".format(folder))
                    
            if boutdata_ok == True:

                if data["options"]["timestep"] * data["options"]["nout"] == data["outputs"]["tt"]:
                    status = "Finished"
                else:
                    status = "Not finished"
                    
            if boutdata_ok == False:
                status = "Error"
                
            if found_dmp == False:
                status = "Not started"
                
            if found_inp == False:
                status = "Missing input file"

            if quiet == False:
                print(f"{folder} || {status}")
            else:
                return status
            
parser = argparse.ArgumentParser()
parser.add_argument("-quiet", action="store_false", help = "suppress print")
args = parser.parse_args()

is_finished(args.i[0], args.i[1], quiet = quiet)