

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