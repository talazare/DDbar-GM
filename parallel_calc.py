import numpy as np
import pandas as pd
import multiprocessing as mp
from root_numpy import fill_hist
from machine_learning_hep.utilities import create_folder_struc, seldf_singlevar, openfile
from multiprocessing import Pool, cpu_count

num_cores = int(cpu_count()*0.5)
num_part  = num_cores*4

def split_df(df, num_part):
    split_indices = (df.shape[0] // num_part) * np.arange(1, num_part, dtype=np.int)
    for i in range (0, num_part-1):
        while ( df.iloc[split_indices[i]][["run_number", "ev_id"]] ==
                df.iloc[split_indices[i]-1][["run_number", "ev_id"]]).all():
            split_indices[i] += 1
    df_split = np.split(df, split_indices)
    return df_split

def parallelize_df(df, func, num_cores=num_cores, num_part=num_part):
    #NB: work with presorted dataframe!!!
    df_split = split_df(df, num_part)
    pool = Pool(num_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df
