import pandas as pd


def parse_petsc_logview(path):
    
    """
    Extracts the first big table (CPU cost) from BOUT++ logs when 
    using PETSc with the log_view flag. Provide path to the file with the
    console dump showing the PETSc logs. Returns a dataframe.
    """
    
    lines = []
    with open(path, "r") as f:
        for line in f.readlines():
            lines.append(line)

    
    read = False

    perf_lines = []

    for line in lines:
        
        if "Event Stage" in line:
            read = True
        if "Memory usage is given in bytes:" in line:
            break
        
        if read and line != "\n":
            perf_lines.append(line[:-1])
            
    perf_lines = perf_lines[1:-1]
    perf_lines[3] = perf_lines[3].replace("Total BOUT++", "Total_BOUT++")   # Remove space which we use for delimiter
    # perf_lines

    df = pd.DataFrame()
    for line in perf_lines:

        split = line.split()
        name = split[0]
        
        data_delimiter = line.split(split[2])[0]  # data after second column has hardcoded widths
        
        data = line.split(data_delimiter)[1]

        
        df.loc[name, "count_max"] = split[1]  # count_max has no hardcoded width
        df.loc[name, "count_ratio"] = data[:3]
        df.loc[name, "time_max"] = data[3:14]       # s
        df.loc[name, "time_ratio"] = data[14:18]    # s
        df.loc[name, "flop_max"] = data[18:27]
        df.loc[name, "flop_ratio"] = data[27:32]
        df.loc[name, "mess"] = data[32:40]
        df.loc[name, "avglen"] = data[40:48]
        df.loc[name, "reduct"] = data[48:55]
        df.loc[name, "global_t"] = data[55:58]      # %
        df.loc[name, "global_f"] = data[58:61]      # %
        df.loc[name, "global_m"] = data[61:64]      # %
        df.loc[name, "global_l"] = data[64:67]      # %
        df.loc[name, "global_r"] = data[67:70]      # %
        df.loc[name, "stage_t"] = data[70:74]       # %
        df.loc[name, "stage_f"] = data[74:77]       # %
        df.loc[name, "stage_m"] = data[77:80]       # %
        df.loc[name, "stage_l"] = data[80:83]       # %
        df.loc[name, "stage_r"] = data[83:86]       # %
        df.loc[name, "total"] = split[-1]           # Mflop/s

        
        
    df = df.astype(float)
    
    return df