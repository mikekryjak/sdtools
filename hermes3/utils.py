import pickle as pkl

def read_file(filename, quiet = False):
# Reads a pickle file and returns it.

    with open(filename, "rb") as f:
    # Open file in read binary mode, dump file to result.
        data = pkl.load(f)
        if not quiet:
            print("{} loaded".format(filename))
        
    return data

def write_file(data, filename, quiet = False):
# Writes an object to a pickle file.
    
    with open(filename, "wb") as file:
    # Open file in write binary mode, dump result to file
        pkl.dump(data, file)
        if not quiet:
            print("{} written".format(filename))

def constants(name):
    
    d = dict()
    d["mass_p"] = 1.6726219e-27 # Proton mass [kg]
    d["mass_e"] = 9.1093837e-31 # Electron mass [kg]
    d["a0"] = 5.29177e-11 # Bohr radius [m]
    d["q_e"] = 1.60217662E-19 # electron charge [C] or [J ev^-1]
    d["k_b"] = 1.3806488e-23 # Boltzmann self.ant [JK^-1]
    
    return d[name]


def mike_cmap():
    return ["navy", "deeppink", "teal", "darkorange", "limegreen", "crimson"]