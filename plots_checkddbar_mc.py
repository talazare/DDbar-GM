import numpy as np
import pandas as pd
import pickle
import multiprocessing as mp
import matplotlib.pyplot as plt
import os, sys

from ROOT import TH1F, TH2F, TH3F, TF1, TF2, TCanvas, TFile
from ROOT import kBlack, kBlue, kRed, kGreen, kMagenta, TLegend
from root_numpy import fill_hist
from machine_learning_hep.utilities import create_folder_struc, seldf_singlevar, openfile
from multiprocessing import Pool, cpu_count
from parallel_calc import split_df, parallelize_df, num_cores, num_part
from phi_compare import make_phi_compare

import lz4.frame
import time

binning = 50

#debug = True
debug = False

#sig_sig = True
sig_sig = False

#sig_fake = True
sig_fake = False

#fake_sig = True
fake_sig = False

#fake_fake = True
fake_fake = False

full_data = True
#full_data = False

compare_phi_before = False
compare_phi_after = False


d_phi_cut = 0.

b_cut_lower = np.pi/2
a_cut_lower = 3*np.pi/4
a_cut_upper = 5*np.pi/4
b_cut_upper = 3*np.pi/2

start= time.time()

dfreco = pickle.load(openfile("./filtrated_df_mc.pkl", "rb"))

dfreco = dfreco.reset_index(drop = True)

end = time.time()
print("Data loaded in", end - start, "sec")

if(debug):
    print("Debug mode: reduced data")
    dfreco = dfreco[:10000]
print("Size of data", dfreco.shape)

print(dfreco.columns)

foldname = "./results_mc"
os.makedirs(foldname, exist_ok=True);

os.chdir(foldname)
if compare_phi_before:
    make_phi_compare(dfreco)

df_d = dfreco[dfreco["is_d"] == 1]
df_dbar = dfreco[dfreco["is_d"] == 0]
df_d_sig = df_d[df_d["ismcsignal"] == 1]
df_d_fake = df_d[df_d["ismcsignal"] == 0]
df_dbar_sig = df_dbar[df_dbar["ismcsignal"] == 1]
df_dbar_fake = df_dbar[df_dbar["ismcsignal"] == 0]

def total_fit():
    Nsig = 83420
    Nbkg = 5499
    Nsigbkg = 16616
    Nbkgsig = 13914
    cYields = TCanvas('cYields', 'The Fit Canvas')
    fit_fun1 = TF1("fit_fun_1", "gaus", 1.64, 2.1)
    h_invmass_dsig = TH1F("invariant mass" , "", binning, df_d_sig.inv_mass.min(),
            df_d_sig.inv_mass.max())
    fill_hist(h_invmass_dsig, df_d_sig.inv_mass)
    h_invmass_dsig.Fit(fit_fun1)
    par1 = fit_fun1.GetParameters()
    h_invmass_dsig.Draw()
    cYields.SaveAs("h_invmass_dsig.png")

    fit_fun2 = TF1("fit_fun2", "gaus", 1.82, 1.92)
    h_invmass_dbkg = TH1F("invariant mass" , "", binning, df_d_fake.inv_mass.min(),
            df_d_fake.inv_mass.max())
    fill_hist(h_invmass_dbkg, df_d_fake.inv_mass)
    h_invmass_dbkg.Fit(fit_fun2)
    par2 = fit_fun2.GetParameters()
    h_invmass_dbkg.Draw()
    cYields.SaveAs("h_invmass_dbkg.png")

    fit_fun3 = TF1("fit_fun_3", "gaus", 1.64, 2.1)
    h_invmass_dbarsig = TH1F("invariant mass" , "", binning, df_dbar_sig.inv_mass.min(),
            df_dbar_sig.inv_mass.max())
    fill_hist(h_invmass_dbarsig, df_dbar_sig.inv_mass)
    h_invmass_dbarsig.Fit(fit_fun3)
    par3 = fit_fun3.GetParameters()
    h_invmass_dbarsig.Draw()
    cYields.SaveAs("h_invmass_dbarsig.png")

    fit_fun4 = TF1("fit_fun_4", "gaus", 1.64, 2.1)
    h_invmass_dbarbkg = TH1F("invariant mass" , "", binning, df_dbar_fake.inv_mass.min(),
            df_dbar_fake.inv_mass.max())
    fill_hist(h_invmass_dbarbkg, df_dbar_fake.inv_mass)
    h_invmass_dbarbkg.Fit(fit_fun4)
    par4 = fit_fun4.GetParameters()
    h_invmass_dbarbkg.Draw()
    cYields.SaveAs("h_invmass_dbarbkg.png")
    #fit_func = str(Nsig) + "*[0]*np.exp(-((x-[1])**2)/(2*[2]))*[3]*np.exp(-((y-[4])**2)/(2*[5]))+Nsigbkg*[0]*np.exp(-((x-[1])**2)/(2*[2]))*[6]*np.exp(-((y-[7])**2)/(2*[8]))+Nbkgsig*[9]*np.exp(-((x-[10])**2)/(2*[11]))*[3]*np.exp(-((y-[4])**2)/(2*[5]))+Nbkg*[9]*np.exp(-((x-[10])**2)/(2*[11]))*[6]*np.exp(-((y-[7])**2)/(2*[8]))"
    fit_func = str(Nsig) + "*[0]*exp(-pow((x-[1]),2)/(2*[2]))*[3]*exp(-pow((y-[4]),2)/(2*[5]))+"+str(Nsigbkg)+"*[0]*exp(-pow((x-[1]),2)/(2*[2]))*[6]*exp(-pow((y-[7]),2)/(2*[8]))+"+str(Nbkgsig)+"*[9]*exp(-pow((x-[10]),2)/(2*[11]))*[3]*exp(-pow((y-[4]),2)/(2*[5]))+"+str(Nbkg)+"*[9]*exp(-pow((x-[10]),2)/(2*[11]))*[6]*exp(-pow((y-[7]),2)/(2*[8]))"
    total_fit = TF2("total_fit", fit_func, 1.64, 2.1, 1.64, 2.1)
    parameters = np.array([par1[0], par1[1], par1[2], par2[0], par2[1],
            par2[2], par3[0], par3[1], par3[2], par4[0], par4[1], par4[2]])
    total_fit.SetParameters(parameters)
    return total_fit

if (sig_sig):
    frames = [df_d_sig, df_dbar_sig]
    filtrated_phi_0 = pd.concat(frames)
    foldname = "./results_mc_sig_sig"
    os.makedirs(foldname, exist_ok=True);

if (sig_fake):
    frames = [df_d_sig, df_dbar_fake]
    filtrated_phi_0 = pd.concat(frames)
    foldname = "./results_mc_sig_sig"
    os.makedirs(foldname, exist_ok=True);

if (fake_sig):
    frames = [df_d_fake, df_dbar_sig]
    filtrated_phi_0 = pd.concat(frames)
    foldname = "./results_mc_sig_sig"
    os.makedirs(foldname, exist_ok=True);

if (fake_fake):
    frames = [df_d_fake, df_dbar_fake]
    filtrated_phi_0 = pd.concat(frames)
    foldname = "./results_mc_sig_sig"
    os.makedirs(foldname, exist_ok=True);

if (full_data):
    filtrated_phi = dfreco[dfreco["delta_phi"] > 0]
else:
    def filtrate_df(df):
        grouped = df.groupby(["run_number", "ev_id"]).filter(lambda
              x: len(x) > 1)
        new_df = grouped[grouped["delta_phi"] > 0]
        return new_df

    start = time.time()
    print("sorting values...")
    filtrated_phi_0.sort_values(["run_number", "ev_id"], inplace=True)
    print("start parallelise...")
    filtrated_phi = parallelize_df(filtrated_phi_0, filtrate_df)
    end = time.time()
    print("paralellized calculations done in", end-start, "sec")
    os.chdir(foldname)

hfile = TFile('post_selection_histos_mc.root', 'RECREATE', 'ROOT file with histograms' )

cYields = TCanvas('cYields', 'The Fit Canvas')

h_d_phi_cand = TH1F("delta phi cand" , "", 200, 0.1, 6)
fill_hist(h_d_phi_cand, filtrated_phi["delta_phi"])
cYields.SetLogy(True)
h_d_phi_cand.Draw()
cYields.SaveAs("h_d_phi_cand.png")

if compare_phi_after:
    make_phi_compare(filtrated_phi)

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

print(filtrated_phi_a)

pt_vec_max_a = filtrated_phi_a["pt_cand_max"]
pt_vec_max_b = filtrated_phi_b["pt_cand_max"]
pt_vec_rest_a = filtrated_phi_a["pt_cand"]
pt_vec_rest_b = filtrated_phi_b["pt_cand"]
pt_min_a = pt_vec_max_a.min()
pt_max_a = pt_vec_max_a.max()
pt_min_b = pt_vec_max_b.min()
pt_max_b = pt_vec_max_b.max()

phi_max_vec_a = filtrated_phi_a["phi_cand_max"]
phi_max_vec_b = filtrated_phi_b["phi_cand_max"]
phi_vec_a = filtrated_phi_a["phi_cand"]
phi_vec_b = filtrated_phi_b["phi_cand"]
phi_min_a = phi_vec_a.min()
phi_max_a = phi_vec_a.max()
phi_min_b = phi_vec_b.min()
phi_max_b = phi_vec_b.max()

eta_max_vec_a = filtrated_phi_a["eta_cand_max"]
eta_max_vec_b = filtrated_phi_b["eta_cand_max"]
eta_vec_a = filtrated_phi_a["eta_cand"]
eta_vec_b = filtrated_phi_b["eta_cand"]
eta_min_a = eta_vec_a.min()
eta_max_a = eta_vec_a.max()
eta_min_b = eta_vec_b.min()
eta_max_b = eta_vec_b.max()

inv_mass_max_vec_a = filtrated_phi_a["inv_cand_max"].to_list()
inv_mass_max_vec_b = filtrated_phi_b["inv_cand_max"].to_list()
inv_mass_vec_a = filtrated_phi_a["inv_mass"].to_list()
inv_mass_vec_b = filtrated_phi_b["inv_mass"].to_list()
inv_mass_tot = filtrated_phi["inv_mass"].tolist()
inv_mass_tot_max = filtrated_phi["inv_cand_max"].tolist()
mass_min_a = filtrated_phi_a["inv_mass"].min()
mass_max_a = filtrated_phi_a["inv_mass"].max()
mass_min_b = filtrated_phi_b["inv_mass"].min()
mass_max_b = filtrated_phi_b["inv_mass"].max()
mass_tot_max_min = filtrated_phi["inv_mass"].min()
mass_tot_max_max = filtrated_phi["inv_mass"].max()
mass_tot_min = filtrated_phi["inv_mass"].min()
mass_tot_max = filtrated_phi["inv_mass"].max()

cYields.cd()

plots(sig_sig)
plots(sig_bkg)
plots(bkg_sig)
plots(bkg_bkg)

h_DDbar_mass_tot = TH2F("Dbar-D plot" , "", 50, mass_tot_min, mass_tot_max,
        50, mass_tot_max_min, mass_tot_max_max)
#DDbar_a = np.column_stack((inv_mass_vec_a, inv_mass_max_vec_a))
t = 0
est = 0
for i in range (0, len(inv_mass_tot)-1):
    start = time.time()
    if i%10000 == 0:
        print("count is", i, "out of", len(inv_mass_tot), "time passed:", t,
        "total time", est)
    h_DDbar_mass_tot.Fill(inv_mass_tot[i], inv_mass_tot_max[i])
    end = time.time()
    t += end-start
    est = (end - start)*len(inv_mass_tot)
if (full_data):
    fit_fun = total_fit()
    h_DDbar_mass_tot.Fit(fit_fun)
    par = fit_fun.GetParameters()
h_DDbar_mass_tot.GetXaxis().SetTitleOffset(1.8)
h_DDbar_mass_tot.GetXaxis().SetTitle("inv_mass of Dbar, GeV")
h_DDbar_mass_tot.GetYaxis().SetTitleOffset(1.8)
h_DDbar_mass_tot.GetYaxis().SetTitle("inv_mass of D, GeV")
h_DDbar_mass_tot.SetOption("lego2z")
h_DDbar_mass_tot.Draw("same")
hfile.Write()
cYields.SaveAs("h_DDbar_tot.png")


h_DDbar_mass_a = TH2F("Dbar-D plot region A" , "", 50, mass_min_a, mass_max_a,
        50, mass_tot_max_min, mass_tot_max_max)
#DDbar_a = np.column_stack((inv_mass_vec_a, inv_mass_max_vec_a))
t = 0
est = 0
for i in range (0, len(inv_mass_vec_a)-1):
    start = time.time()
    if i%10000 == 0:
        print("count is", i, "out of", len(inv_mass_vec_a), "time passed:", t,
        "total time", est)
    h_DDbar_mass_a.Fill(inv_mass_vec_a[i],
                inv_mass_max_vec_a[i])
    end = time.time()
    t += end-start
    est = (end - start)*len(inv_mass_vec_a)
h_DDbar_mass_a.GetXaxis().SetTitleOffset(1.8)
h_DDbar_mass_a.GetXaxis().SetTitle("inv_mass of Dbar, GeV")
h_DDbar_mass_a.GetYaxis().SetTitleOffset(1.8)
h_DDbar_mass_a.GetYaxis().SetTitle("inv_mass of D, GeV")
h_DDbar_mass_a.SetOption("lego2z")
h_DDbar_mass_a.Draw("")
cYields.SaveAs("h_DDbar_A.png")

t = 0
est = 0
h_DDbar_mass_b = TH2F("Dbar-D plot region B" , "", 50, mass_min_b, mass_max_b,
        50, mass_tot_max_min, mass_tot_max_max)
#DDbar_b = np.column_stack((inv_mass_vec_b, inv_mass_max_vec_b))
for i in range (0, len(inv_mass_vec_b)-1):
    start = time.time()
    if i%10000 == 0:
        print("count is", i, "out of", len(inv_mass_vec_b), "time passed:", t,
        "total time", est)
    h_DDbar_mass_b.Fill(inv_mass_vec_b[i], inv_mass_max_vec_b[i])
    end = time.time()
    t += end-start
    est = (end - start)*len(inv_mass_vec_b)
h_DDbar_mass_b.GetXaxis().SetTitleOffset(1.8)
h_DDbar_mass_b.GetXaxis().SetTitle("inv_mass of Dbar, GeV")
h_DDbar_mass_b.GetYaxis().SetTitleOffset(1.8)
h_DDbar_mass_b.GetYaxis().SetTitle("inv_mass of D, GeV")
h_DDbar_mass_b.SetOption("lego2z")
h_DDbar_mass_b.Draw("")
cYields.SaveAs("h_DDbar_B.png")

cYields.SetLogy(False)
h_first_cand_mass = TH1F("inv_mass of the first cand" , "", 200,
        mass_min_a, mass_max_a)
fill_hist(h_first_cand_mass, inv_mass_max_vec_a)
h_second_cand_mass_a = TH1F("inv_mass in range A" , "", 200,
        mass_min_a, mass_max_a)
fill_hist(h_second_cand_mass_a, inv_mass_vec_a)
h_second_cand_mass_b = TH1F("inv_mass in range B" , "", 200,
        mass_min_a, mass_max_a)
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



