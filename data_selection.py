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
from scipy.optimize import curve_fit
from array import array
import lz4.frame
import time

def create_bkg_df(size):
    dfreco = pickle.load(openfile("./results_test/filtrated_df.pkl", "rb"))
    print("datasamlpe loaded", dfreco.shape)
    d_bkg = dfreco[dfreco["is_d"]==1]
    nd_bkg = dfreco[dfreco["is_d"]==0]

    frames = [d_bkg, nd_bkg]
    bkg = pd.concat(frames)
    bkg["ismcsignal"] = 0
    bkg = bkg.sample(n = size)
    print(bkg)
    return bkg
