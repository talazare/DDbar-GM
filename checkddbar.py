import pandas as pd
import pickle
import matplotlib.pyplot as plt
from ROOT import TH1F, TCanvas
from root_numpy import fill_hist
from machine_learning_hep.utilities import create_folder_struc, seldf_singlevar, openfile
import lz4.frame

dfreco = pickle.load(openfile("/data/Derived/D0kINT7HighMultwithJets/vAN-20191003_ROOT6-1/pp_2018_data/260_20191004-0008/skpkldecmerged/AnalysisResultsReco4_6_0.65.pkl.lz4", "rb"))
dfreco = dfreco.query("y_test_probxgboost>0.8")
h_invmass = TH1F("hmass" , "", 200, 1.64, 2.1)
fill_hist(h_invmass, dfreco.inv_mass)

cYields = TCanvas('cYields', 'The Fit Canvas')
h_invmass.Draw()
cYields.SaveAs("h_invmass.pdf")
