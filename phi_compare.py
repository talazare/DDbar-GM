import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import os, sys

from ROOT import TH1F, TH2F, TH3F, TF1, TCanvas, TFile
from ROOT import kBlack, kBlue, kRed, kGreen, kMagenta, TLegend
from root_numpy import fill_hist
from machine_learning_hep.utilities import create_folder_struc, seldf_singlevar, openfile

def make_phi_compare(filtrated_phi):

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
