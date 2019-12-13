import numpy as np
import pandas as pd
import pickle
import multiprocessing as mp
import matplotlib.pyplot as plt
import os, sys

from ROOT import TH1F, TH2F, TH3F, TF1, TF2, TF3, TCanvas, TFile, TMath, TPad
from root_numpy import fill_hist
from machine_learning_hep.utilities import create_folder_struc, seldf_singlevar, openfile

#PDF = Nsig*gaus(d_signal)*gaus(not_d_signal) +
#      Nsigbkg*gaus(d_signal)*pol2(not_d_bkg) +
#      Nbkgsig*pol2(d_bkg)*gaus(not_d_signal) +
#      Nbkg*pol2(d_bkg)*pol2(not_d_bkg)

def total_fit_py(x, *par):
    fit_func_py = par[12]*par[0]*np.exp((-(x[0]-par[1])**2)/(2*par[2]))*par[6]*np.exp((-(x[1]-par[7])**2)/(2*par[8]))+par[13]*par[0]*np.exp((-(x[0]-par[1])**2)/(2*par[2]))*(par[9]+par[10]*x[1]+par[11]*x[1]**2)+par[14]*(par[3]+par[4]*x[0]+par[5]*x[0]**2)*par[6]*np.exp((-(x[1]-par[7])**2)/(2*par[8]))+par[15]*(par[3]+par[4]*x[0]+par[5]*x[0]**2)*(par[9]+par[10]*x[1]+par[11]*x[1]**2)
#   fit_func_py = par[12]*par[0]*np.exp((-(x[0]-par[1])**2)/(2*par[2]))*par[6]*np.exp((-(x[1]*par[7])**2)/(2*par[8]))+par[13]*par[0]*np.exp((-(x[0]-par[1])**2)/(2*par[2]))*par[9]*np.exp((-(x[1]-par[10])**2)/(2*par[11]))+par[14]*par[3]*np.exp((-(x[0]-par[4])**2)/(2*par[5]))*par[6]*np.exp((-(x[1]-par[7])**2)/(2*par[8]))+par[15]*par[3]*np.exp((-(x[0]-par[4])**2)/(2*par[5]))*par[9]*np.exp((-(x[1]-par[10])**2)/(2*par[11]))
    return fit_func_py

def total_fit():
    fit_func = "[12]*[0]*exp((-(x-[1])**2)/(2*[2]))*[6]*exp((-(y-[7])**2)/(2*[8]))+[13]*[0]*exp((-(x-[1])**2)/(2*[2]))*([9]+[10]*y+[11]*y**2)+[14]*([3]+[4]*x+[5]*x**2)*[6]*exp((-(y-[7])**2)/(2*[8]))+[15]*([3]+[4]*x+[5]*x**2)*([9]+[10]*y+[11]*y**2)"
#    fit_func = "[12]*[0]*exp((-(x-[1])**2)/(2*[2]))*[6]*exp((-(y-[7])**2)/(2*[8]))+[13]*[0]*exp((-(x-[1])**2)/(2*[2]))*[9]*exp((-(y-[10])**2)/(2*[11]))+[14]*[3]*exp((-(x-[4])**2)/(2*[5]))*[6]*exp((-(y-[7])**2)/(2*[8]))+[15]*[3]*exp((-(x-[4])**2)/(2*[5]))*[9]*exp((-(y-[10])**2)/(2*[11]))"
    total_fit = TF2("total_fit", fit_func, 1.64, 2.1, 1.64, 2.1)
    return total_fit



