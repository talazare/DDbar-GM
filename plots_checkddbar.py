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

make_phi_compare = True
#make_phi_compare = False

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
    dfreco = dfreco[:1000000]
print("Size of data", dfreco.shape)

print(dfreco.columns)

filtrated_phi = dfreco[dfreco["delta_phi"] > 0]

hfile = TFile( 'post_selection_histos_mc.root', 'RECREATE', 'ROOT file with histograms' )

cYields = TCanvas('cYields', 'The Fit Canvas')

h_d_phi_cand = TH1F("delta phi cand" , "", 200, 0.1, 6)
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
h_DDbar_mass_tot.SetOption("lego2z")
h_DDbar_mass_tot.Draw("LEGO2Z")
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
h_DDbar_mass_a.SetOption("lego2z")
h_DDbar_mass_a.Draw("")
cYields.SaveAs("h_DDbar_A.png")

t = 0
est = 0
h_DDbar_mass_b = TH2F("Dbar-D plot region B" , "", 50, mass_min_b, mass_max_b,
        50, mass_tot_max_min, mass_tot_max_max)
#DDbar_b = np.column_stack((inv_mass_vec_b, inv_mass_max_vec_b))
for i in range (0, len(inv_mass_vec_b)-1):
     if i%10000 == 0:
        print("count is", i, "out of", len(inv_mass_vec_b), "time passed:", t,
        "total time", est)
     h_DDbar_mass_b.Fill(inv_mass_vec_b[i], inv_mass_max_vec_b[i])
     end = time.time()
     t += end-start
     est = (end - start)*len(inv_mass_vec_b)
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

hfile.Write()


