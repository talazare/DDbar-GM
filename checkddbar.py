import numpy as np
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt

from machine_learning_hep.utilities import create_folder_struc, seldf_singlevar, openfile
from multiprocessing import Pool, cpu_count
from parallel_calc import split_df, parallelize_df, num_cores, num_part

from load_data import main
import time

main(False, False, True)

#parallelized functions over the dataframe

print("parallelizing will be done with", num_cores, "cores")

def filter_phi(df):
    delta_phi_all = []
    grouped = df.groupby(["run_number", "ev_id"], sort = False)
    df["is_d"] = 0
    for name, group in grouped:
        pt_max = group["pt_cand"].idxmax()
        phi_max = df.loc[pt_max, "phi_cand"]
        df.loc[pt_max, "is_d"] = 1
        delta_phi = np.abs(phi_max - group["phi_cand"])
        delta_phi_all.extend(delta_phi)
    df["delta_phi"] = delta_phi_all
    def max_el(group):
        df.loc[group.index, "pt_cand_max"]  = df.loc[group["pt_cand"].idxmax(), "pt_cand"]
        df.loc[group.index, "inv_cand_max"] = df.loc[group["pt_cand"].idxmax(), "inv_mass"]
        df.loc[group.index, "phi_cand_max"] = df.loc[group["pt_cand"].idxmax(), "phi_cand"]
        df.loc[group.index, "eta_cand_max"] = df.loc[group["pt_cand"].idxmax(), "eta_cand"]
        return df
    grouped = df.groupby(["run_number", "ev_id"]).apply(max_el)
    return df

# create smaller dataframes to work with
start = time.time()
if (real_data):
    df_work = dfreco[["run_number", "ev_id", "pt_cand", "inv_mass", "phi_cand",
            "eta_cand"]]
else:
    df_work = dfreco[["run_number", "ev_id", "pt_cand", "inv_mass", "phi_cand",
            "eta_cand", "ismcsignal"]]

end = time.time()
print("creating workin df", end - start)
split_const = int(df_work.shape[0]/500000)
df_work.sort_values(["run_number", "ev_id"], inplace=True)
working_df = split_df(df_work, split_const)
start = time.time()
dataframe = []
timing = 0
est = 0
for i in range (0,  split_const):
    start = time.time()
    print("progress:", i, "out of", split_const, "| process time: ", timing,
    "| estimated full time:", est)
    df = parallelize_df(working_df[i], filter_phi)
    end = time.time()
    est = (end - start)*split_const
    timing += end - start
    dataframe.append(df)
filtrated_phi_0 = pd.concat(dataframe)
filtrated_phi_0.to_pickle("./filtrated_df_mc.pkl")
end2 = time.time()
print(filtrated_phi_0)
print("paralellized calculations are done in", end2 - start, "sec")
