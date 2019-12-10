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
d_phi_cut = 0.

b_cut_lower = np.pi/2
a_cut_lower = 3*np.pi/4
a_cut_upper = 5*np.pi/4
b_cut_upper = 3*np.pi/2

def filtrate_df(df):
    grouped = df.groupby(["run_number", "ev_id"]).filter(lambda
          x: len(x) > 1)
    new_df = grouped[grouped["delta_phi"] > 0]
    return new_df


def main(sig_sig, sig_fake, fake_sig, fake_fake, full_data, df_d_sig, df_d_fake, df_dbar_sig,
        df_dbar_fake, dfreco, compare_phi_after, tot_entries):

    if (sig_sig):
        frames = [df_d_sig, df_dbar_sig]
        filtrated_phi_0 = pd.concat(frames)
        foldname = "/home/talazare/DDbar-GM/results/results_mc/results_mc_sig_sig"
        os.makedirs(foldname, exist_ok=True);

    if (sig_fake):
        frames = [df_d_sig, df_dbar_fake]
        filtrated_phi_0 = pd.concat(frames)
        foldname = "/home/talazare/DDbar-GM/results/results_mc/results_mc_sig_fake"
        os.makedirs(foldname, exist_ok=True);

    if (fake_sig):
        frames = [df_d_fake, df_dbar_sig]
        filtrated_phi_0 = pd.concat(frames)
        foldname = "/home/talazare/DDbar-GM/results/results_mc/results_mc_fake_sig"
        os.makedirs(foldname, exist_ok=True);

    if (fake_fake):
        frames = [df_d_fake, df_dbar_fake]
        filtrated_phi_0 = pd.concat(frames)
        foldname = "/home/talazare/DDbar-GM/results/results_mc/results_mc_fake_fake"
        os.makedirs(foldname, exist_ok=True);

    if (full_data):
        filtrated_phi = dfreco[dfreco["delta_phi"] > 0]

    if (full_data == False):
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


    cYields_tot = TCanvas('cYields', 'The Fit Canvas')
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
    tot_entries = h_DDbar_mass_tot.GetEntries()
    print("entries", tot_entries)
    h_DDbar_mass_tot.GetXaxis().SetTitleOffset(1.8)
    h_DDbar_mass_tot.GetXaxis().SetTitle("inv_mass of Dbar, GeV")
    h_DDbar_mass_tot.GetYaxis().SetTitleOffset(1.8)
    h_DDbar_mass_tot.GetYaxis().SetTitle("inv_mass of D, GeV")
    h_DDbar_mass_tot.SetOption("lego2z")
    h_DDbar_mass_tot.Draw("same")
    cYields_tot.SaveAs("h_DDbar_tot.png")

    cYields_a = TCanvas('cYields', 'The Fit Canvas')
    h_DDbar_mass_a = TH2F("Dbar-D plot region A" , "", 50, mass_min_a, mass_max_a,
            50, mass_tot_max_min, mass_tot_max_max)
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
    cYields_a.SaveAs("h_DDbar_A.png")

    cYields_b = TCanvas('cYields', 'The Fit Canvas')
    t = 0
    est = 0
    h_DDbar_mass_b = TH2F("Dbar-D plot region B" , "", 50, mass_min_b, mass_max_b,
            50, mass_tot_max_min, mass_tot_max_max)
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
    cYields_b.SaveAs("h_DDbar_B.png")

    cYields_all= TCanvas('cYields', 'The Fit Canvas')
    cYields_all.SetLogy(False)
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
    cYields_all.SaveAs("h_inv_mass_cand.png")

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
    cYields_all.SaveAs("h_pt_cand_max_min.png")

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
    cYields_all.SaveAs("h_eta_cand_max_min.png")

    hfile.Write()
