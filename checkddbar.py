import numpy as np
import pandas as pd
import pickle
import multiprocessing as mp
import matplotlib.pyplot as plt

from ROOT import TH1F, TH2F, TF1, TCanvas
from ROOT import kBlack, kBlue, kRed, kGreen, kMagenta, TLegend
from root_numpy import fill_hist
from machine_learning_hep.utilities import create_folder_struc, seldf_singlevar, openfile
from multiprocessing import Pool, cpu_count

import lz4.frame
import time

#debug = True
debug = False

real_data = True
#real_data = False

#plots = True
plots = False

#make_phi_compare = True
make_phi_compare = False

d_phi_cut = 0.

b_cut_lower = np.pi/2
a_cut_lower = 3*np.pi/4
a_cut_upper = 5*np.pi/4
b_cut_upper = 3*np.pi/2

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
        dfreco0 = pickle.load(openfile("/data/Derived/D0kINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2018_mc_prodD2H/261_20191004-0007/skpkldecmerged/AnalysisResultsGen_pt_cand1_2.pkl.lz4", "rb"))
        dfreco1 = pickle.load(openfile("/data/Derived/D0kINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2018_mc_prodD2H/261_20191004-0007/skpkldecmerged/AnalysisResultsGen_pt_cand2_4.pkl.lz4", "rb"))
        dfreco2 = pickle.load(openfile("/data/Derived/D0kINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2018_mc_prodD2H/261_20191004-0007/skpkldecmerged/AnalysisResultsGen_pt_cand4_6.pkl.lz4", "rb"))
        dfreco3 = pickle.load(openfile("/data/Derived/D0kINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2018_mc_prodD2H/261_20191004-0007/skpkldecmerged/AnalysisResultsGen_pt_cand6_8.pkl.lz4", "rb"))
        dfreco4 = pickle.load(openfile("/data/Derived/D0kINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2018_mc_prodD2H/261_20191004-0007/skpkldecmerged/AnalysisResultsGen_pt_cand8_24.pkl.lz4", "rb"))
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
    dfreco = dfreco[:200000]
print("Size of data", dfreco.shape)

print(dfreco.columns)

binning = 200

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

#parallelized functions over the dataframe

num_cores = int(cpu_count()/2)
num_part  = num_cores*2

print("parallelizing will be done with", num_cores, "cores")
def parallelize_df(df, func, num_cores=num_cores, num_part=num_part):
    #sort dataframe
    df.sort_values(["run_number", "ev_id"], inplace=True)
    split_indices = (df.shape[0] // num_part) * np.arange(1, num_part, dtype=np.int)
    for i in range (0, num_part-1):
        while ( df.iloc[split_indices[i]][["run_number", "ev_id"]] ==
                df.iloc[split_indices[i]-1][["run_number", "ev_id"]]).all():
            split_indices[i] += 1
    df_split = np.split(df, split_indices)
    #df_split = np.array_split(df, num_part)
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
    return df

start = time.time()

filtrated_phi_0 = parallelize_df(dfreco, filter_phi)

end2 = time.time()

print("paralellized calculations of delta phi is done in", end2 - start, "sec")

# so far everything is perfect

filtrated_phi = filtrated_phi_0[filtrated_phi_0["delta_phi"] > 0]

h_d_phi_cand = TH1F("delta phi cand" , "", 200, filtrated_phi.delta_phi.min(),
        filtrated_phi.delta_phi.max())
fill_hist(h_d_phi_cand, filtrated_phi["delta_phi"])
cYields.SetLogy(True)
h_d_phi_cand.Draw()
cYields.SaveAs("h_d_phi_cand.png")

if (make_phi_compare):

    cYields_2 = TCanvas('cYields_2', 'The Fit Canvas 2')

    filtrated_phi_1 = filtrated_phi.query("pt_cand < 4")

    d_phi_dist_1 =  filtrated_phi_1["delta_phi"]

    filtrated_phi_2 = filtrated_phi.query("pt_cand > 4")
    filtrated_phi_2 = filtrated_phi_2.query("pt_cand < 6")

    d_phi_dist_2 =  filtrated_phi_2["delta_phi"]

    filtrated_phi_3 = filtrated_phi.query("pt_cand > 6")
    filtrated_phi_3 = filtrated_phi_3.query("pt_cand < 8")

    d_phi_dist_3 =  filtrated_phi_3["delta_phi"]

    filtrated_phi_4 = filtrated_phi.query("pt_cand > 8")
    filtrated_phi_4 = filtrated_phi_4.query("pt_cand < 10")

    d_phi_dist_4 =  filtrated_phi_4["delta_phi"]

    filtrated_phi_5 = filtrated_phi.query("pt_cand > 10")
    filtrated_phi_5 = filtrated_phi_5.query("pt_cand < 24")

    d_phi_dist_5 =  filtrated_phi_5["delta_phi"]

    h_d_phi_cand_1 = TH1F("delta phi cand, pt range:[<4]" , "Normalized plot", 200,
            d_phi_dist_1.min(), d_phi_dist_1.max())
    fill_hist(h_d_phi_cand_1, d_phi_dist_1)
    h_d_phi_cand_1.Scale(1/ h_d_phi_cand_1.Integral())
    h_d_phi_cand_2 = TH1F("delta phi cand, pt range:[4-6]" , "", 200,
            d_phi_dist_1.min(), d_phi_dist_1.max())
    fill_hist(h_d_phi_cand_2, d_phi_dist_2)
    h_d_phi_cand_2.Scale(1/ h_d_phi_cand_2.Integral())
    h_d_phi_cand_3 = TH1F("delta phi cand, pt range:[6-8]" , "", 200,
            d_phi_dist_1.min(), d_phi_dist_1.max())
    fill_hist(h_d_phi_cand_3, d_phi_dist_3)
    h_d_phi_cand_3.Scale(1/ h_d_phi_cand_3.Integral())
    h_d_phi_cand_4 = TH1F("delta phi cand, pt range:[8-10]" , "", 200,
            d_phi_dist_1.min(), d_phi_dist_1.max())
    fill_hist(h_d_phi_cand_4, d_phi_dist_4)
    h_d_phi_cand_4.Scale(1/ h_d_phi_cand_4.Integral())
    h_d_phi_cand_5 = TH1F("delta phi cand, pt range:[10-24]" , "", 200,
            d_phi_dist_1.min(), d_phi_dist_1.max())
    fill_hist(h_d_phi_cand_5, d_phi_dist_5)
    h_d_phi_cand_5.Scale(1/ h_d_phi_cand_5.Integral())
    cYields_2.SetLogy(True)
    h_d_phi_cand_1.SetStats(0)
    h_d_phi_cand_1.SetLineColor(kBlack)
    h_d_phi_cand_1.Draw()
    h_d_phi_cand_2.SetStats(0)
    h_d_phi_cand_2.SetLineColor(kRed)
    h_d_phi_cand_2.Draw("same")
    h_d_phi_cand_3.SetStats(0)
    h_d_phi_cand_3.SetLineColor(kBlue)
    h_d_phi_cand_3.Draw("same")
    h_d_phi_cand_4.SetStats(0)
    h_d_phi_cand_4.SetLineColor(kGreen)
    h_d_phi_cand_4.Draw("same")
    h_d_phi_cand_5.SetStats(0)
    h_d_phi_cand_5.SetLineColor(kMagenta)
    h_d_phi_cand_5.Draw("same")
    leg = TLegend(0.45, 0.7, 0.95, 0.87)
    leg.SetBorderSize(0)
    leg.SetFillColor(0)
    leg.SetFillStyle(0)
    leg.SetTextFont(42)
    leg.SetTextSize(0.035)
    leg.AddEntry(h_d_phi_cand_1, h_d_phi_cand_1.GetName(),"L")
    leg.AddEntry(h_d_phi_cand_2, h_d_phi_cand_2.GetName(),"L")
    leg.AddEntry(h_d_phi_cand_3, h_d_phi_cand_3.GetName(),"L")
    leg.AddEntry(h_d_phi_cand_4, h_d_phi_cand_4.GetName(),"L")
    leg.AddEntry(h_d_phi_cand_5, h_d_phi_cand_5.GetName(),"L")
    leg.Draw("same")
    cYields_2.SaveAs("h_d_phi_cand_compare.png")

print("Delta phi cuts region A: [", a_cut_lower, a_cut_upper,"]")
print("Delta phi cuts region B: [", b_cut_lower, a_cut_lower,"]&[",
        a_cut_upper, b_cut_upper,"]")

filtrated_phi_1 = filtrated_phi[filtrated_phi["delta_phi"] > b_cut_lower]
filtrated_phi_1 = filtrated_phi_1[filtrated_phi_1["delta_phi"] < a_cut_lower]

filtrated_phi_2 = filtrated_phi[filtrated_phi["delta_phi"] > a_cut_upper]
filtrated_phi_2 = filtrated_phi_2[filtrated_phi_2["delta_phi"] < b_cut_upper]

frames = [filtrated_phi_1, filtrated_phi_2]
filtrated_phi_b = pd.concat(frames)

filtrated_phi_a = filtrated_phi[filtrated_phi["delta_phi"] > a_cut_lower]
filtrated_phi_a = filtrated_phi_a[filtrated_phi_a["delta_phi"] < a_cut_upper]

#so far it's fine

def filter_max(df):
    grouped = df.groupby(["run_number", "ev_id"])
    pt_max = grouped["pt_cand"].idxmax()
    df = df.loc[pt_max, ]
    return df

start = time.time()
df_max = parallelize_df(filtrated_phi_0, filter_max)
end = time.time()
print("creating dataframe contains only pt_max lines:", end - start, "sec")

frames_a = [filtrated_phi_a, df_max]
new_df_total_a = pd.concat(frames_a)
new_df_total_a = new_df_total_a.reset_index(drop = True)
frames_b = [filtrated_phi_b, df_max]
new_df_total_b = pd.concat(frames_b)
new_df_total_b = new_df_total_b.reset_index(drop = True)

def filter_max_2(df):
    df = df.groupby(["run_number", "ev_id"], sort=True).filter(lambda x:
            len(x) > 1)
#    df["pt_cand_max"] = 0.
#    df["inv_cand_max"] = 0.
#    df["phi_cand_max"] = 0.
#    df["eta_cand_max"] = 0.
    def max_el(group):
        df.loc[group.index, "pt_cand_max"]  = df.loc[group["pt_cand"].idxmax(), "pt_cand"]
        df.loc[group.index, "inv_cand_max"] = df.loc[group["pt_cand"].idxmax(), "inv_mass"]
        df.loc[group.index, "phi_cand_max"] = df.loc[group["pt_cand"].idxmax(), "phi_cand"]
        df.loc[group.index, "eta_cand_max"] = df.loc[group["pt_cand"].idxmax(), "eta_cand"]
        return df
    grouped = df.groupby(["run_number", "ev_id"]).apply(max_el)
    df = df[df["delta_phi"] > 0]
    return df

#def filter_max_2(df):
#    df = df.groupby(["run_number", "ev_id"], sort=True).filter(lambda x:
#            len(x) > 1)
##    df["pt_cand_max"] = 0.
##    df["inv_cand_max"] = 0.
##    df["phi_cand_max"] = 0.
##    df["eta_cand_max"] = 0.
#    grouped = df.groupby(["run_number", "ev_id"])
#    for name, group in grouped:
#        df.loc[group.index, "pt_cand_max"]  = df.loc[group["pt_cand"].idxmax(), "pt_cand"]
#        df.loc[group.index, "inv_cand_max"] = df.loc[group["pt_cand"].idxmax(), "inv_mass"]
#        df.loc[group.index, "phi_cand_max"] = df.loc[group["pt_cand"].idxmax(), "phi_cand"]
#        df.loc[group.index, "eta_cand_max"] = df.loc[group["pt_cand"].idxmax(), "eta_cand"]
#    pt_max = grouped["pt_cand"].idxmax()
#    df = df[df["delta_phi"] > 0]
#    return df


start = time.time()
new_df_max_a = parallelize_df(new_df_total_a, filter_max_2)
print(new_df_max_a)
new_df_max_b = parallelize_df(new_df_total_b, filter_max_2)
#new_df_max = filter_max_2(new_df_total)
end = time.time()
print("creating primary particles vectors", end - start)

pt_vec_max_a = new_df_max_a["pt_cand_max"]
pt_vec_max_b = new_df_max_b["pt_cand_max"]
pt_vec_rest_a = filtrated_phi_a["pt_cand"]
pt_vec_rest_b = filtrated_phi_b["pt_cand"]

#print(len(pt_vec_max_a), len(pt_vec_rest_a), len(pt_vec_max_b),
#        len(pt_vec_rest_b))

phi_max_vec_a = new_df_max_a["phi_cand_max"]
phi_max_vec_b = new_df_max_b["phi_cand_max"]
phi_vec_a = filtrated_phi_a["phi_cand"]
phi_vec_b = filtrated_phi_b["phi_cand"]

eta_max_vec_a = new_df_max_a["eta_cand_max"]
eta_max_vec_b = new_df_max_b["eta_cand_max"]
eta_vec_a = filtrated_phi_a["eta_cand"]
eta_vec_b = filtrated_phi_b["eta_cand"]

inv_mass_max_vec_a = new_df_max_a["inv_cand_max"]
inv_mass_max_vec_b = new_df_max_b["inv_cand_max"]
inv_mass_vec_a = filtrated_phi_a["inv_mass"]
inv_mass_vec_b = filtrated_phi_b["inv_mass"]

cYields = TCanvas('cYields', 'The Fit Canvas')
h_DDbar_mass_a = TH2F("Dbar-D plot region A" , "", 200,
        inv_mass_vec_a.min(), inv_mass_vec_a.max(), 200,
        inv_mass_max_vec_a.min(), inv_mass_max_vec_a.max())
DDbar_a = np.column_stack((inv_mass_vec_a, inv_mass_max_vec_a))
fill_hist(h_DDbar_mass_a, DDbar_a)
h_DDbar_mass_a.Draw("colz")
cYields.SaveAs("h_DDbar_A.png")

cYields = TCanvas('cYields', 'The Fit Canvas')
h_DDbar_mass_b = TH2F("Dbar-D plot region B" , "", 200,
        inv_mass_vec_b.min(), inv_mass_vec_b.max(), 200,
        inv_mass_max_vec_b.min(), inv_mass_max_vec_b.max())
DDbar_b = np.column_stack((inv_mass_vec_b, inv_mass_max_vec_b))
fill_hist(h_DDbar_mass_b, DDbar_b)
h_DDbar_mass_b.Draw("colz")
cYields.SaveAs("h_DDbar_B.png")

cYields.SetLogy(False)
h_first_cand_mass = TH1F("inv_mass of the first cand" , "", 200,
        inv_mass_max_vec_a.min(), inv_mass_max_vec_a.max())
fill_hist(h_first_cand_mass, inv_mass_max_vec_a)
h_second_cand_mass_a = TH1F("inv_mass in range A" , "", 200,
        inv_mass_max_vec_a.min(), inv_mass_max_vec_a.max())
fill_hist(h_second_cand_mass_a, inv_mass_vec_a)
h_second_cand_mass_b = TH1F("inv_mass in range B" , "", 200,
        inv_mass_max_vec_a.min(), inv_mass_max_vec_a.max())
fill_hist(h_second_cand_mass_b, inv_mass_vec_b)
h_first_cand_mass.SetLineColor(kBlack)
h_second_cand_mass_a.SetLineColor(kRed)
h_second_cand_mass_b.SetLineColor(kBlue)
h_first_cand_mass.Draw()
h_first_cand_mass.SetStats(0)
h_second_cand_mass_a.Draw("same")
h_second_cand_mass_b.Draw("same")
leg = TLegend(0.6, 0.7, 0.95, 0.87)
leg.SetBorderSize(0)
leg.SetFillColor(0)
leg.SetFillStyle(0)
leg.SetTextFont(42)
leg.SetTextSize(0.035)
leg.AddEntry(h_first_cand_mass, h_first_cand_mass.GetName(),"L")
leg.AddEntry(h_second_cand_mass_a, h_second_cand_mass_a.GetName(),"L")
leg.AddEntry(h_second_cand_mass_b, h_second_cand_mass_b.GetName(),"L")
leg.Draw("same")
cYields.SaveAs("h_inv_mass_cand.png")

h_first_cand_pt = TH1F("pt of the first cand" , "", 200,
        pt_vec_rest_a.min(), pt_vec_rest_a.max())
fill_hist(h_first_cand_pt, pt_vec_max_a)
h_second_cand_pt_a = TH1F("pt in range A" , "", 200,
        pt_vec_rest_a.min(), pt_vec_rest_a.max())
fill_hist(h_second_cand_pt_a, pt_vec_rest_a)
h_second_cand_pt_b = TH1F("pt in range B" , "", 200,
        pt_vec_rest_a.min(),pt_vec_rest_a.max())
fill_hist(h_second_cand_pt_b, pt_vec_rest_b)
h_first_cand_pt.SetLineColor(kBlack)
h_second_cand_pt_a.SetLineColor(kRed)
h_second_cand_pt_b.SetLineColor(kBlue)
h_second_cand_pt_a.SetStats(0)
#h_first_cand_pt.Draw()
h_second_cand_pt_a.Draw("")
h_second_cand_pt_b.Draw("same")
leg = TLegend(0.6, 0.7, 0.95, 0.87)
leg.SetBorderSize(0)
leg.SetFillColor(0)
leg.SetFillStyle(0)
leg.SetTextFont(42)
leg.SetTextSize(0.035)
#leg.AddEntry(h_d_phi_cand_1, h_d_phi_cand_1.GetName(),"L")
leg.AddEntry(h_second_cand_pt_a, h_second_cand_pt_a.GetName(),"L")
leg.AddEntry(h_second_cand_pt_b, h_second_cand_pt_b.GetName(),"L")
leg.Draw("same")
cYields.SaveAs("h_pt_cand_max_min.png")

h_first_cand_eta = TH1F("eta of the first cand" , "", 200,
        eta_max_vec_a.min(), eta_max_vec_a.max())
fill_hist(h_first_cand_eta, eta_max_vec_a)
h_second_cand_eta_a = TH1F("eta in range A" , "", 200,
        eta_max_vec_a.min(), eta_max_vec_a.max())
fill_hist(h_second_cand_eta_a, eta_vec_a)
h_second_cand_eta_b = TH1F("eta in range B" , "", 200,
        eta_max_vec_a.min(), eta_max_vec_a.max())
fill_hist(h_second_cand_eta_b, eta_vec_b)
h_first_cand_eta.SetLineColor(kBlack)
h_second_cand_eta_a.SetLineColor(kRed)
h_second_cand_eta_b.SetLineColor(kBlue)
h_second_cand_eta_a.SetStats(0)
#h_first_cand_eta.Draw()
h_second_cand_eta_a.Draw("")
h_second_cand_eta_b.Draw("same")
leg = TLegend(0.6, 0.7, 0.95, 0.87)
leg.SetBorderSize(0)
leg.SetFillColor(0)
leg.SetFillStyle(0)
leg.SetTextFont(42)
leg.SetTextSize(0.035)
# leg.AddEntry(h_d_phi_cand_1, h_d_phi_cand_1.GetName(),"L")
leg.AddEntry(h_second_cand_eta_a, h_second_cand_eta_a.GetName(),"L")
leg.AddEntry(h_second_cand_eta_b, h_second_cand_eta_b.GetName(),"L")
leg.Draw("same")
cYields.SaveAs("h_eta_cand_max_min.png")



