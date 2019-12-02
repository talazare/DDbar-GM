import numpy as np
import pandas as pd
import pickle
import multiprocessing as mp
import matplotlib.pyplot as plt

from ROOT import TH1F, TH2F, TH3F, TF1, TCanvas, TFile
from ROOT import kBlack, kBlue, kRed, kGreen, kMagenta, TLegend
from root_numpy import fill_hist
from machine_learning_hep.utilities import create_folder_struc, seldf_singlevar, openfile
from multiprocessing import Pool, cpu_count

import lz4.frame
import time

#debug = True
debug = False

#real_data = True
real_data = False

#plots = True
plots = False

start= time.time()

if (debug):
    dframe = pickle.load(openfile("/data/Derived/D0kINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2018_data/260_20191004-0008/skpkldecmerged/AnalysisResultsReco4_6_0.65.pkl.lz4", "rb"))

else:
   if (real_data):
        dfreco0 = pickle.load(openfile("/data/Derived/D0kINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2018_data/260_20191004-0008/skpkldecmerged/AnalysisResultsReco1_2_0.75.pkl.lz4", "rb"))
        dfreco1 = pickle.load(openfile("/data/Derived/D0kINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2018_data/260_20191004-0008/skpkldecmerged/AnalysisResultsReco2_4_0.75.pkl.lz4", "rb"))
        dfreco2 = pickle.load(openfile("/data/Derived/D0kINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2018_data/260_20191004-0008/skpkldecmerged/AnalysisResultsReco4_6_0.65.pkl.lz4", "rb"))
        dfreco3 = pickle.load(openfile("/data/Derived/D0kINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2018_data/260_20191004-0008/skpkldecmerged/AnalysisResultsReco6_8_0.65.pkl.lz4", "rb"))
        dfreco4 = pickle.load(openfile("/data/Derived/D0kINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2018_data/260_20191004-0008/skpkldecmerged/AnalysisResultsReco8_24_0.45.pkl.lz4", "rb"))
        frames = [dfreco0, dfreco1, dfreco2, dfreco3, dfreco4]
        dframe = pd.concat(frames)

   else:
        dfreco0 = pickle.load(openfile("/data/Derived/D0kINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2018_mc_prodD2H/261_20191004-0007/skpkldecmerged/AnalysisResultsReco1_2_0.75.pkl.lz4", "rb"))
        dfreco1 = pickle.load(openfile("/data/Derived/D0kINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2018_mc_prodD2H/261_20191004-0007/skpkldecmerged/AnalysisResultsReco2_4_0.75.pkl.lz4", "rb"))
        dfreco2 = pickle.load(openfile("/data/Derived/D0kINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2018_mc_prodD2H/261_20191004-0007/skpkldecmerged/AnalysisResultsReco4_6_0.65.pkl.lz4", "rb"))
        dfreco3 = pickle.load(openfile("/data/Derived/D0kINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2018_mc_prodD2H/261_20191004-0007/skpkldecmerged/AnalysisResultsReco6_8_0.65.pkl.lz4", "rb"))
        dfreco4 = pickle.load(openfile("/data/Derived/D0kINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2018_mc_prodD2H/261_20191004-0007/skpkldecmerged/AnalysisResultsReco8_24_0.45.pkl.lz4", "rb"))
        frames = [dfreco0, dfreco1, dfreco2, dfreco3, dfreco4]
        dframe = pd.concat(frames)

#dframe = dframe.query("y_test_probxgboost>0.5")
#dframe = dframe.query("pt_cand > 10")
#dframe = dframe.query("pt_cand < 10")
dfreco = dframe.reset_index(drop = True)

end = time.time()
print("Data loaded in", end - start, "sec")

if(debug):
    print("Debug mode: reduced data")
    dfreco = dfreco[:1000000]
print("Size of data", dfreco.shape)

print(dfreco.columns)

binning = 200

hfile = TFile('pre_selection_histos.root', 'RECREATE', 'ROOT file with histograms' )
cYields = TCanvas('cYields', 'The Fit Canvas')
fit_fun1 = TF1("fit_fun1", "expo" ,1.64, 1.82)
fit_fun2 = TF1("fit_fun2", "gaus", 1.82, 1.92)
fit_total = TF1("fit_total", "expo(0) + gaus(2) + expo(5)", 1.64, 2.1)
h_invmass = TH1F("invariant mass" , "", binning, dfreco.inv_mass.min(),
        dfreco.inv_mass.max())
fill_hist(h_invmass, dfreco.inv_mass)
h_invmass.Fit(fit_fun1, "R")
par1 = fit_fun1.GetParameters()
h_invmass.Fit(fit_fun2, "R+")
par2 = fit_fun2.GetParameters()
fit_total.SetParameters(par1[0], par1[1], par2[0], par2[1], par2[2],
        par1[0],par1[1])
h_invmass.Fit(fit_total,"R+")
par = fit_total.GetParameters()
h_invmass.Draw()
cYields.SaveAs("h_invmass.png")

if (plots):
    cYields.SetLogy(True)
    h_d_len = TH1F("d_len" , "", 200, dfreco.d_len.min(), dfreco.d_len.max())
    fill_hist(h_d_len, dfreco.d_len)
    h_d_len.Draw()
    cYields.SaveAs("h_d_len.png")

    h_norm_dl = TH1F("norm dl" , "", 200, dfreco.norm_dl.min(), dfreco.norm_dl.max())
    fill_hist(h_norm_dl, dfreco.norm_dl)
    h_norm_dl.Draw()
    cYields.SaveAs("h_norm_dl.png")

    cYields.SetLogy(False)
    h_cos_p = TH1F("cos_p" , "", 200, dfreco.cos_p.min(), dfreco.cos_p.max())
    fill_hist(h_cos_p, dfreco.cos_p)
    h_cos_p.Draw()
    cYields.SaveAs("h_cos_p.png")

    cYields.SetLogy(True)
    h_nsigTPC_K_0 = TH1F("nsigma TPC K_0" , "", 200, dfreco.nsigTPC_K_0.min(),
            dfreco.nsigTPC_K_0.max())
    fill_hist(h_nsigTPC_K_0, dfreco.nsigTPC_K_0)
    h_nsigTPC_K_0.Draw()
    cYields.SaveAs("nsigTPC_K_0.png")

    h_nsigTPC_K_1 = TH1F("nsigTPC_K_1 " , "", 200, dfreco.nsigTPC_K_1.min(),
            dfreco.nsigTPC_K_1.max())
    fill_hist(h_nsigTPC_K_1, dfreco.nsigTPC_K_1)
    h_nsigTPC_K_1.Draw()
    cYields.SaveAs("h_nsigTPC_K_1.png")

    h_nsigTOF_K_0 = TH1F("nsigma TOF K_0" , "", 200, dfreco.nsigTOF_K_0.min() ,
            dfreco.nsigTOF_K_0.max() )
    fill_hist(h_nsigTOF_K_0 , dfreco.nsigTOF_K_0 )
    h_nsigTOF_K_0.Draw()
    cYields.SaveAs("nsigTOF_K_0.png")

    h_nsigTOF_K_1 = TH1F("nsigTOF_K_1 " , "", 200, dfreco.nsigTOF_K_1.min(),
            dfreco.nsigTOF_K_1.max())
    fill_hist(h_nsigTOF_K_1, dfreco.nsigTOF_K_1)
    h_nsigTOF_K_1.Draw()
    cYields.SaveAs("h_nsigTOF_K_1.png")

    cYields.SetLogy(False)
    h_pt_prong0 = TH1F("pt prong_0" , "", 200,  dfreco.pt_prong0.min(),
            dfreco.pt_prong0.max())
    fill_hist(h_pt_prong0, dfreco.pt_prong0)
    h_pt_prong0.Draw()
    cYields.SaveAs("h_pt_prong0.png")

    h_pt_prong1 = TH1F("pt prong_1" , "", 200,  dfreco.pt_prong1.min(),
            dfreco.pt_prong1.max())
    fill_hist(h_pt_prong1, dfreco.pt_prong1)
    h_pt_prong1.Draw()
    cYields.SaveAs("h_pt_prong1.png")

    h_eta_prong0 = TH1F("eta prong_0" , "", 200, dfreco.eta_prong0.min(),
            dfreco.eta_prong0.max())
    fill_hist(h_eta_prong0, dfreco.eta_prong0)
    h_eta_prong0.Draw()
    cYields.SaveAs("h_eta_prong0.png")

    h_eta_prong1 = TH1F("eta prong_1" , "", 200, dfreco.eta_prong1.max(),
            dfreco.eta_prong1.max())
    fill_hist(h_eta_prong1, dfreco.eta_prong1)
    h_eta_prong1.Draw()
    cYields.SaveAs("h_eta_prong1.png")

    h_eta_cand = TH1F("eta cand" , "", 200, dfreco.eta_cand.min(),
            dfreco.eta_cand.max())
    fill_hist(h_eta_cand, dfreco.eta_cand)
    h_eta_cand.Draw()
    cYields.SaveAs("h_eta_cand.png")

    h_phi_cand = TH1F("phi cand" , "", 200, dfreco.eta_cand.min(),
            dfreco.eta_cand.max())
    fill_hist(h_phi_cand, dfreco.phi_cand)
    h_phi_cand.Draw()
    cYields.SaveAs("h_phi_cand.png")

    h_pt_cand = TH1F("pt cand" , "", 200, dfreco.pt_cand.min(),
            dfreco.pt_cand.max())
    fill_hist(h_pt_cand, dfreco.pt_cand)
    h_pt_cand.Draw()
    cYields.SaveAs("h_pt_cand.png")

grouped = dfreco.groupby(["run_number","ev_id"])
grouplen = pd.array(grouped.size())
gmin = grouplen.min()
gmax = grouplen.max()
g_bins = gmax - gmin
print("creating grouplen array", end - start, "sec")
h_grouplen = TH1F("group_length" , "", int(g_bins), gmin, gmax)
fill_hist(h_grouplen, grouplen)
cYields.SetLogy(True)
h_grouplen.Draw()
cYields.SaveAs("h_grouplen.png")

hfile.Write()

#parallelized functions over the dataframe

num_cores = int(cpu_count()*0.5)
num_part  = num_cores*4

print("parallelizing will be done with", num_cores, "cores")

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

def filter_phi(df):
    delta_phi_all = []
    grouped = df.groupby(["run_number", "ev_id"], sort = False)
    for name, group in grouped:
        pt_max = group["pt_cand"].idxmax()
        phi_max = df.loc[pt_max, "phi_cand"]
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
df_work = dfreco[["run_number", "ev_id", "pt_cand", "inv_mass", "phi_cand",
        "eta_cand"]]
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
