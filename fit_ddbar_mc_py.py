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
from plots_2d import main
from fitters import total_fit_py, total_fit, PDF_Fit2d
from scipy.optimize import curve_fit
from array import array
import lz4.frame
import time

binning = 50

#debug = True
debug = False

reduced_data = True
#reduced_data=False

compare_phi_before = False
compare_phi_after = False

d_phi_cut = 0.

b_cut_lower = np.pi/2
a_cut_lower = 3*np.pi/4
a_cut_upper = 5*np.pi/4
b_cut_upper = 3*np.pi/2

start= time.time()

#loading montecarlo dataframe modified in checkddbar.py
dfreco = pickle.load(openfile("./data/filtrated_df_mc.pkl", "rb"))

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

if (reduced_data):
    rd_d_sig = int(df_d_sig.shape[0]/4)
    rd_d_fake = int(df_d_fake.shape[0]/4)
    rd_nd_sig = int(df_dbar_sig.shape[0]/4)
    rd_nd_fake = int(df_dbar_fake.shape[0]/4)
    print(df_d_sig.shape, rd_d_sig, rd_d_fake, rd_nd_sig, rd_nd_fake)
    df_d_sig = df_d_sig[:rd_d_sig] # only d signal
#    df_d_fake = df_d_fake[:rd_d_fake] # only d background
    df_dbar_sig = df_dbar_sig[:rd_nd_sig] # only not d signal
#    df_dbar_fake = df_dbar_fake[:rd_nd_fake] # only not d background

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

## make pairs for each event to extract Nsig, Nsigbkg, Nbkgsig, Nbkg, plot distributions
## main(sig_sig, sig_fake, fake_sig, fake_fake, full data, df_d_sig, df_d_fake, df_dbar_sig, df_dbar_fake, dfreco, )

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

##main(False, False, False, False, True, df_d_sig, df_d_fake, df_dbar_sig,
##        df_dbar_fake, dfreco, False)


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

params = [par1[1], par1[2], par2[1], par2[2], par3[1], par3[2], par4[1],
        par4[2], Nsig, Nsigbkg, Nbkgsig, Nbkg]

data = np.stack((np.array(inv_mass_tot), np.array(inv_mass_tot_max)))
print(data.shape)
py_fit = PDF_Fit2d(data, total_fit_py(par1[0], par3[0], par2[0], par4[0]),
        params=params)
print(py_fit.fit())

fit_fun_py = total_fit(0.,0.,0.,0.)
par_py = array('d', py_fit.fit())

print(par_py)

fit_fun_py.SetParameters(par_py)

print(params)
input()

# Make plot and fit with root instruments
par = array('d', par_py)

hfile = TFile('post_selection_histos_rd.root', 'RECREATE', 'ROOT file with histograms' )
cYields_fin = TCanvas('cYields', 'The Fit Canvas')

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

fit_fun = total_fit(par1[0], par3[0], par2[0], par4[0])

print(par_py)
fit_fun.SetParameters(par)
h_DDbar_mass_tot.Fit(fit_fun, "Q")
new_par = fit_fun.GetParameters()
for i in range(12):
    print(new_par[i])
h_DDbar_mass_tot.GetXaxis().SetTitleOffset(1.8)
h_DDbar_mass_tot.GetXaxis().SetTitle("inv_mass of Dbar, GeV")
h_DDbar_mass_tot.GetYaxis().SetTitleOffset(1.8)
h_DDbar_mass_tot.GetYaxis().SetTitle("inv_mass of D, GeV")
h_DDbar_mass_tot.SetOption("lego2 z")
h_DDbar_mass_tot.Draw()
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
data = np.stack((np.array(inv_mass_tot), np.array(inv_mass_tot_max)))
print(data.shape)
py_fit = PDF_Fit2d(data, total_fit_py(par1[0], par3[0], par2[0], par4[0]),
        params=params)
print(py_fit.fit())

fit_fun_py = total_fit(0.,0.,0.,0.)
par_py = array('d', py_fit.fit())


print(par_py)

fit_fun_py.SetParameters(par_py)

# Make plot and fit with root instruments
hfile = TFile('post_selection_histos_mc.root', 'RECREATE', 'ROOT file with histograms' )
cYields_fin = TCanvas('cYields', 'The Fit Canvas')
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

fit_fun = total_fit(par1[0], par3[0], par2[0], par4[0])
print(par)
fit_fun.SetParameters(par)
h_DDbar_mass_rd.Fit(fit_fun, "Q")
h_DDbar_mass_rd.GetXaxis().SetTitleOffset(1.8)
h_DDbar_mass_rd.GetXaxis().SetTitle("inv_mass of Dbar, GeV")
h_DDbar_mass_rd.GetYaxis().SetTitleOffset(1.8)
h_DDbar_mass_rd.GetYaxis().SetTitle("inv_mass of D, GeV")
h_DDbar_mass_rd.SetOption("lego2 z")
h_DDbar_mass_rd.Draw()
cYields_fin.SaveAs("h_DDbar_rd_fit.png")
hfile.Write()
