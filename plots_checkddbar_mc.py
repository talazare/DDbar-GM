import numpy as np
import pandas as pd
import pickle
import multiprocessing as mp
import matplotlib.pyplot as plt
import os, sys

from ROOT import TH1F, TH2F, TH3F, TF1, TF2, TF3, TCanvas, TFile, TMath, TPad
from ROOT import kBlack, kBlue, kRed, kGreen, kMagenta, TLegend
from root_numpy import fill_hist
from machine_learning_hep.utilities import create_folder_struc, seldf_singlevar, openfile
from multiprocessing import Pool, cpu_count
from parallel_calc import split_df, parallelize_df, num_cores, num_part
from phi_compare import make_phi_compare
from 2d2d_plots import main
from fitters import total_fit_py, total_fit
from scipy.optimize import curve_fit
from array import array
import lz4.frame
import time

binning = 50

#debug = True
debug = False

compare_phi_before = False
compare_phi_after = False

d_phi_cut = 0.

b_cut_lower = np.pi/2
a_cut_lower = 3*np.pi/4
a_cut_upper = 5*np.pi/4
b_cut_upper = 3*np.pi/2

start= time.time()

#loading montecarlo dataframe modified in checkddbar.py
dfreco = pickle.load(openfile("./filtrated_df_mc.pkl", "rb"))

dfreco = dfreco.reset_index(drop = True)
end = time.time()
print("Data loaded in", end - start, "sec")

if(debug):
    print("Debug mode: reduced data")
    dfreco = dfreco[:10000]
print("Size of data", dfreco.shape)

print(dfreco.columns)

foldname = "/home/talazare/DDbar-GM/results/results_mc"
os.makedirs(foldname, exist_ok=True);

os.chdir(foldname)

if compare_phi_before:
    make_phi_compare(dfreco)

df_d = dfreco[dfreco["is_d"] == 1] # only with max_pt in the event (consider as d)
df_dbar = dfreco[dfreco["is_d"] == 0] # everything else (not d)
df_d_sig = df_d[df_d["ismcsignal"] == 1] # only d signal
df_d_fake = df_d[df_d["ismcsignal"] == 0] # only d background
df_dbar_sig = df_dbar[df_dbar["ismcsignal"] == 1] # only not d signal
df_dbar_fake = df_dbar[df_dbar["ismcsignal"] == 0] # only not d background

cYields = TCanvas('cYields', 'The Fit Canvas')
fit_fun1 = TF1("fit_fun_1", "gaus", 1.64, 2.1)
h_invmass_dsig = TH1F("invariant mass" , "", binning, df_d_sig.inv_mass.min(),
        df_d_sig.inv_mass.max())
fill_hist(h_invmass_dsig, df_d_sig.inv_mass)
h_invmass_dsig.Fit(fit_fun1)
par1 = fit_fun1.GetParameters()
h_invmass_dsig.Draw()
cYields.SaveAs("h_invmass_dsig.png")

fit_fun2 = TF1("fit_fun2", "pol2", 1.82, 1.92)
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

fit_fun4 = TF1("fit_fun_4", "pol2", 1.64, 2.1)
h_invmass_dbarbkg = TH1F("invariant mass" , "", binning, df_dbar_fake.inv_mass.min(),
        df_dbar_fake.inv_mass.max())
fill_hist(h_invmass_dbarbkg, df_dbar_fake.inv_mass)
h_invmass_dbarbkg.Fit(fit_fun4)
par4 = fit_fun4.GetParameters()
h_invmass_dbarbkg.Draw()
cYields.SaveAs("h_invmass_dbarbkg.png")

# make pairs for each event to extract Nsig, Nsigbkg, Nbkgsig, Nbkg, plot distributions
# main(sig_sig, sig_fake, fake_sig, fake_fake, full data, df_d_sig, df_d_fake, df_dbar_sig, df_dbar_fake, dfreco, )

Nsig = main(True, False, False, False, False, df_d_sig, df_d_fake, df_dbar_sig,
        df_dbar_fake, dfreco, False)
print("Nsig", Nsig)

Nsigbkg = main(False, True, False, False, False, df_d_sig, df_d_fake, df_dbar_sig,
        df_dbar_fake, dfreco, False)
print("Nsigbkg", Nsigbkg)

Nbkgsig = main(False, False, True, False, False, df_d_sig, df_d_fake, df_dbar_sig,
        df_dbar_fake, dfreco, False)
print("Nbkgsig", Nbkgsig)

Nbkg = main(False, False, False, True, False, df_d_sig, df_d_fake, df_dbar_sig,
        df_dbar_fake, dfreco, False)
print("Nbkg", Nbkg)

#main(False, False, False, False, True, df_d_sig, df_d_fake, df_dbar_sig,
#        df_dbar_fake, dfreco, False)


os.chdir(foldname)

#make fit for the monte carlo data after getting all parameters

filtrated_phi = dfreco[dfreco["delta_phi"]>0]
inv_mass_tot = filtrated_phi["inv_mass"].tolist()
inv_mass_tot_max = filtrated_phi["inv_cand_max"].tolist()
mass_tot_max_min = filtrated_phi["inv_mass"].min()
mass_tot_max_max = filtrated_phi["inv_mass"].max()
mass_tot_min = filtrated_phi["inv_mass"].min()
mass_tot_max = filtrated_phi["inv_mass"].max()

# Make fit with python instruments
params, params_covariance = curve_fit(total_fit_py,
        np.array(inv_mass_tot), np.array(inv_mass_tot_max), p0=[par1[0],
        par1[1], par1[2],  par2[0], par2[1], par2[2],  par3[0], par3[1],
        par3[2], par4[0], par4[1], par4[2], Nsig, Nsigbkg, Nbkgsig, Nbkg])

fit_fun_py = total_fit()
par_py = array('d', 16*[0.])
par_py[0], par_py[1], par_py[2] =  params[0], params[1], params[2] #d-sig (gaus)
par_py[3], par_py[4], par_py[5] =  params[3], params[4], params[5] #d-bkg (pol2)
par_py[6], par_py[7], par_py[8] =  params[6], params[7], params[8] #dbar-sig (gaus)
par_py[9], par_py[10],par_py[11] = params[9], params[10], params[11] #dbar-bkg (pol2)
par_py[12], par_py[13], par_py[14], par_py[15] = params[12], params[13], params[14],params[15]

print(par_py)

fit_fun_py.SetParameters(par_py)
print(params)
input()

# Make plot and fit with root instruments
par = array('d', 16*[0.])
par[0], par[1], par[2] =  par1[0], par1[1], par1[2] #d-sig (gaus)
par[3], par[4], par[5] =  par2[3], par2[4], par2[5] #d-bkg (pol2)
par[6], par[7], par[8] =  par3[6], par3[7], par3[8] #dbar-sig (gaus)
par[9], par[10],par[11] = par4[9], par4[10], par4[11] #dbar-bkg (pol2)
par[12], par[13], par[14], par[15] = Nsig, Nsigbkg, Nbkgsig, Nbkg


hfile = TFile('post_selection_histos_mc.root', 'RECREATE', 'ROOT file with histograms' )
cYields_fin = TCanvas('cYields', 'The Fit Canvas')

#pad1 = TPad( 'pad1', 'Data', 0.03, 0.50, 0.98, 0.95, 21 )
#pad2 = TPad( 'pad2', 'Fit',      0.03, 0.02, 0.98, 0.48, 21 )
#pad1.Draw()
#pad2.Draw()

h_DDbar_mass_tot = TH2F("Dbar-D plot" , "", 50, mass_tot_min, mass_tot_max,
        50, mass_tot_max_min, mass_tot_max_max)
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

fit_fun = total_fit()
print(par)
fit_fun.SetParameters(par)
h_DDbar_mass_tot.Fit(fit_fun, "Q")
h_DDbar_mass_tot.GetXaxis().SetTitleOffset(1.8)
h_DDbar_mass_tot.GetXaxis().SetTitle("inv_mass of Dbar, GeV")
h_DDbar_mass_tot.GetYaxis().SetTitleOffset(1.8)
h_DDbar_mass_tot.GetYaxis().SetTitle("inv_mass of D, GeV")
#pad1.cd()
h_DDbar_mass_tot.SetOption("lego2 z")
h_DDbar_mass_tot.Draw()
#pad2.cd()
#fit_fun_py.Draw("surf 4")
cYields_fin.SaveAs("h_DDbar_tot_fit.png")
cYields_fin.Update()

# do the same with real data
dfreco = pickle.load(openfile("../../data/filtrated_df.pkl", "rb"))

filtrated_phi = dfreco[dfreco["delta_phi"]>0]
inv_mass_tot = filtrated_phi["inv_mass"].tolist()
inv_mass_tot_max = filtrated_phi["inv_cand_max"].tolist()
mass_tot_max_min = filtrated_phi["inv_mass"].min()
mass_tot_max_max = filtrated_phi["inv_mass"].max()
mass_tot_min = filtrated_phi["inv_mass"].min()
mass_tot_max = filtrated_phi["inv_mass"].max()

# Make fit with python instruments
new_params, params_covariance = curve_fit(total_fit_py,
        np.array(inv_mass_tot), np.array(inv_mass_tot_max), p0=[par1[0],
        par1[1], par1[2],  par2[0], par2[1], par2[2],  par3[0], par3[1],
        par3[2],  par4[0], par4[1], par4[2], Nsig, Nsigbkg, Nbkgsig, Nbkg])

fit_fun_py = total_fit()
par_py = array('d', 16*[0.])
par_py[0], par_py[1], par_py[2] =  new_params[0], new_params[1], new_params[2] #d-sig (gaus)
par_py[3], par_py[4], par_py[5] =  new_params[3], new_params[4], new_params[5] #d-bkg (pol2)
par_py[6], par_py[7], par_py[8] =  new_params[6], new_params[7], new_params[8] #dbar-sig (gaus)
par_py[9], par_py[10], par_py[11] = new_params[9], new_params[10], new_params[11] #dbar-bkg (pol2)
par_py[12], par_py[13] = new_params[12], new_params[13]
par_py[14], par_py[15] = new_params[14], new_params[15]

print(par_py)

fit_fun_py.SetParameters(par_py)
print(new_params)
input()


# Make plot and fit with root instruments
hfile = TFile('post_selection_histos_mc.root', 'RECREATE', 'ROOT file with histograms' )
cYields_fin = TCanvas('cYields', 'The Fit Canvas')
#pad1 = TPad( 'pad1', 'Data', 0.03, 0.50, 0.98, 0.95, 21 )
#pad2 = TPad( 'pad2', 'Fit',      0.03, 0.02, 0.98, 0.48, 21 )
#pad1.Draw()
#pad2.Draw()

h_DDbar_mass_rd = TH2F("Dbar-D plot real data" , "", 50, mass_tot_min, mass_tot_max,
        50, mass_tot_max_min, mass_tot_max_max)
t = 0
est = 0
for i in range (0, len(inv_mass_tot)-1):
    start = time.time()
    if i%10000 == 0:
        print("count is", i, "out of", len(inv_mass_tot), "time passed:", t,
        "total time", est)
    h_DDbar_mass_rd.Fill(inv_mass_tot[i], inv_mass_tot_max[i])
    end = time.time()
    t += end-start
    est = (end - start)*len(inv_mass_tot)

fit_fun = total_fit()
print(par)
fit_fun.SetParameters(par)
h_DDbar_mass_rd.Fit(fit_fun, "Q")
h_DDbar_mass_rd.GetXaxis().SetTitleOffset(1.8)
h_DDbar_mass_rd.GetXaxis().SetTitle("inv_mass of Dbar, GeV")
h_DDbar_mass_rd.GetYaxis().SetTitleOffset(1.8)
h_DDbar_mass_rd.GetYaxis().SetTitle("inv_mass of D, GeV")
#pad1.cd()
h_DDbar_mass_rd.SetOption("lego2 z")
h_DDbar_mass_rd.Draw()
#pad2.cd()
#fit_fun_py.Draw("surf 4")
cYields_fin.SaveAs("h_DDbar_rd_fit.png")
cYields_fin.Update()


