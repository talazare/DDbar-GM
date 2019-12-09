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
from final_plots import main

import lz4.frame
import time

binning = 50

#debug = True
debug = False

sig_sig = True
#sig_sig = False

sig_fake = True
#sig_fake = False

fake_sig = True
#fake_sig = False

fake_fake = True
#fake_fake = False

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

main(True, False, False, False, False, df_d_sig, df_d_fake, df_dbar_sig,
        df_dbar_fake, dfreco)
Nsig = tot_entries
main(False, True, False, False, False, df_d_sig, df_d_fake, df_dbar_sig,
        df_dbar_fake, dfreco)
Nsigbkg = tot_entries
main(False, False, True, False, False, df_d_sig, df_d_fake, df_dbar_sig,
        df_dbar_fake, dfreco)
Nbkgsig = tot_entries
main(False, False, False, True, False, df_d_sig, df_d_fake, df_dbar_sig,
        df_dbar_fake, dfreco)
Nbkg = tot_entries
main(False, False, False, False, True, df_d_sig, df_d_fake, df_dbar_sig,
        df_dbar_fake, dfreco)
N_full = tot_entries

def total_fit():
#    fit_func = str(Nsig) + "*[0]*exp(-pow((x-[1]),2)/(2*[2]))*[3]*exp(-pow((y-[4]),2)/(2*[5]))+"+str(Nsigbkg)+"*[0]*exp(-pow((x-[1]),2)/(2*[2]))*[6]*exp(-pow((y-[7]),2)/(2*[8]))+"+str(Nbkgsig)+"*[9]*exp(-pow((x-[10]),2)/(2*[11]))*[3]*exp(-pow((y-[4]),2)/(2*[5]))+"+str(Nbkg)+"*[9]*exp(-pow((x-[10]),2)/(2*[11]))*[6]*exp(-pow((y-[7]),2)/(2*[8]))"
    fit_func = "[12]*[0]*exp(-pow((x-[1]),2)/(2*[2]))*[3]*exp(-pow((y-[4]),2)/(2*[5]))+[13]*[0]*exp(-pow((x-[1]),2)/(2*[2]))*[6]*exp(-pow((y-[7]),2)/(2*[8]))+[14]*[9]*exp(-pow((x-[10]),2)/(2*[11]))*[3]*exp(-pow((y-[4]),2)/(2*[5]))+[15]*[9]*exp(-pow((x-[10]),2)/(2*[11]))*[6]*exp(-pow((y-[7]),2)/(2*[8]))"
    total_fit = TF2("total_fit", fit_func, 1.64, 2.1, 1.64, 2.1)
    parameters = np.array([par1[0], par1[1], par1[2], par2[0], par2[1],
            par2[2], par3[0], par3[1], par3[2], par4[0], par4[1], par4[2],
            Nsig, Nsigbkg, Nbkgsig, Nbkg])
    total_fit.SetParameters(parameters)
    return total_fit


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
cYields.SaveAs("h_DDbar_tot_fit.png")

