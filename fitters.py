import numpy as np
import pandas as pd
import pickle
import multiprocessing as mp
import matplotlib.pyplot as plt
import os, sys
from ROOT import RooRealVar, RooArgList, RooArgSet, RooLinkedList, RooGenericPdf, RooDataSet
from ROOT import TH1F, TH2F, TH3F, TF1, TF2, TF3, TCanvas, TFile, TMath, TPad
from root_numpy import fill_hist
from machine_learning_hep.utilities import create_folder_struc, seldf_singlevar, openfile
import scipy.optimize

#PDF = Nsig*gaus(d_signal)*gaus(not_d_signal) +
#      Nsigbkg*gaus(d_signal)*pol2(not_d_bkg) +
#      Nbkgsig*pol2(d_bkg)*gaus(not_d_signal) +
#      Nbkg*pol2(d_bkg)*pol2(not_d_bkg)


class PDF_Fit2d:
    def __init__(self, data, f, params):
        self.data                   = data
        self.f                      = f
        self.hist, self.x, self.y   = np.histogram2d(data[0,...], data[1,...],
                50, range=[[1.80, 1.92], [1.8, 1.92]])
        self.x = .5 * (self.x[:-1] + self.x[1:])
        self.y = .5 * (self.y[:-1] + self.y[1:])
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.XY = np.stack((self.X.ravel(), self.Y.ravel()))
        self.p0                     =   params

    def fit(self):
        self.params, self.cov = scipy.optimize.curve_fit(self.f, self.XY,
                self.hist.ravel(), p0=self.p0)
        return self.params

def gaussian(x, mu, sig):
    return np.exp(-.5 * ((x - mu) / sig) ** 2.)


def total_fit_py(N1, N2, N3, N4):
    def fit_func(x, *par):
        gauss       = N1 * gaussian(x[0,...], par[0], par[1])
        pol         = N3 + par[2] * x[0,...] + par[3] * x[0,...]**2
        gauss_not   = N2 * gaussian(x[1,...], par[4], par[5])
        pol_not     = N4 + par[6] * x[1,...] + par[7] * x[1,...]**2
        fit_func_py = ( par[8]  * gauss * gauss_not +
                        par[9]  * gauss * pol_not +
                        par[10] * pol   * gauss_not +
                        par[11] * pol   * pol_not)
        return fit_func_py
#            par[12] *par[0]*np.exp((-(x[0]-par[1])**2)/(2*par[2]))  * par[6]*np.exp((-(x[1]-par[7])**2)/(2*par[8]))
#            +par[13]*par[0]*np.exp((-(x[0]-par[1])**2)/(2*par[2]))  * (par[9]+par[10]*x[1]+par[11]*x[1]**2)
#            +par[14]*(par[3]+par[4]*x[0]+par[5]*x[0]**02)            * par[6]*np.exp((-(x[1]-par[7])**2)/(2*par[8]))
#            +par[15]*(par[3]+par[4]*x[0]+par[5]*x[0]**2)            * (par[9]+par[10]*x[1]+par[11]*x[1]**2))
#   fit_func_py = par[12]*par[0]*np.exp((-(x[0]-par[1])**2)/(2*par[2]))*par[6]*np.exp((-(x[1]*par[7])**2)/(2*par[8]))+par[13]*par[0]*np.exp((-(x[0]-par[1])**2)/(2*par[2]))*par[9]*np.exp((-(x[1]-par[10])**2)/(2*par[11]))+par[14]*par[3]*np.exp((-(x[0]-par[4])**2)/(2*par[5]))*par[6]*np.exp((-(x[1]-par[7])**2)/(2*par[8]))+par[15]*par[3]*np.exp((-(x[0]-par[4])**2)/(2*par[5]))*par[9]*np.exp((-(x[1]-par[10])**2)/(2*par[11]))
    return fit_func

def total_fit(N1, N2, N3, N4):
#    fit_func = "[12]*[0]*exp((-(x-[1])**2)/(2*[2]))*[6]*exp((-(y-[7])**2)/(2*[8]))+[13]*[0]*exp((-(x-[1])**2)/(2*[2]))*([9]+[10]*y+[11]*y**2)+[14]*([3]+[4]*x+[5]*x**2)*[6]*exp((-(y-[7])**2)/(2*[8]))+[15]*([3]+[4]*x+[5]*x**2)*([9]+[10]*y+[11]*y**2)"
#    fit_func = "[12]*[0]*exp((-(x-[1])**2)/(2*[2]))*[6]*exp((-(y-[7])**2)/(2*[8]))+[13]*[0]*exp((-(x-[1])**2)/(2*[2]))*[9]*exp((-(y-[10])**2)/(2*[11]))+[14]*[3]*exp((-(x-[4])**2)/(2*[5]))*[6]*exp((-(y-[7])**2)/(2*[8]))+[15]*[3]*exp((-(x-[4])**2)/(2*[5]))*[9]*exp((-(y-[10])**2)/(2*[11]))"
    gauss       = "(%f*exp(-0.5*((x - [0])/[1])**2))" % N1
    pol         = "(%f + [2] * x + [3] * x**2)" % N3
    gauss_not   = "(%f*exp(-0.5*((y - [4])/[5])**2))" % N2
    pol_not     = "(%f + [6] * y + [7] * y**2)" % N4
    fit_func = ( "+[8] *" + gauss + "*" + gauss_not +
                    "+[9] *" + gauss + "*" + pol_not +
                    "+[10]*" + pol   + "*" + gauss_not +
                    "+[11]*" + pol   + "*" + pol_not)
    total_fit = TF2("total_fit", fit_func, 1.8, 1.92, 1.8, 1.92)
    return total_fit

def min_par_fit(p1_0, p1_1, p1_2, p2_0, p2_1, p2_2, p3_0, p3_1, p3_2, p4_0,
        p4_1, p4_2):
#    fit_func = "[12]*[0]*exp((-(x-[1])**2)/(2*[2]))*[6]*exp((-(y-[7])**2)/(2*[8]))+[13]*[0]*exp((-(x-[1])**2)/(2*[2]))*([9]+[10]*y+[11]*y**2)+[14]*([3]+[4]*x+[5]*x**2)*[6]*exp((-(y-[7])**2)/(2*[8]))+[15]*([3]+[4]*x+[5]*x**2)*([9]+[10]*y+[11]*y**2)"
#    fit_func = "[12]*[0]*exp((-(x-[1])**2)/(2*[2]))*[6]*exp((-(y-[7])**2)/(2*[8]))+[13]*[0]*exp((-(x-[1])**2)/(2*[2]))*[9]*exp((-(y-[10])**2)/(2*[11]))+[14]*[3]*exp((-(x-[4])**2)/(2*[5]))*[6]*exp((-(y-[7])**2)/(2*[8]))+[15]*[3]*exp((-(x-[4])**2)/(2*[5]))*[9]*exp((-(y-[10])**2)/(2*[11]))"
    gauss       = "(%f*exp(-0.5*((x - %f)/%f)**2))" % (p1_0,  p1_1,  p1_2)
    pol         = "(%f + %f * x + %f * x**2)" %  (p2_0,  p2_1,  p2_2)
    gauss_not   = "(%f*exp(-0.5*((y - %f)/%f)**2))" %  (p3_0,  p3_1,  p3_2)
    pol_not     = "(%f + %f * y + %f * y**2)" %  (p4_0,  p4_1,  p4_2)
    fit_func = ( "+[0] *" + gauss + "*" + gauss_not +
                    "+[1] *" + gauss + "*" + pol_not +
                    "+[2]*" + pol   + "*" + gauss_not +
                    "+[3]*" + pol   + "*" + pol_not)
    total_fit = TF2("total_fit", fit_func, 1.64, 2.1, 1.64, 2.1)
    return total_fit

def roo_pdf(p1_0, p1_1, p1_2, p2_0, p2_1, p2_2, p3_0, p3_1, p3_2, p4_0,
        p4_1, p4_2):
#    fit_func = "[12]*[0]*exp((-(x-[1])**2)/(2*[2]))*[6]*exp((-(y-[7])**2)/(2*[8]))+[13]*[0]*exp((-(x-[1])**2)/(2*[2]))*([9]+[10]*y+[11]*y**2)+[14]*([3]+[4]*x+[5]*x**2)*[6]*exp((-(y-[7])**2)/(2*[8]))+[15]*([3]+[4]*x+[5]*x**2)*([9]+[10]*y+[11]*y**2)"
#    fit_func = "[12]*[0]*exp((-(x-[1])**2)/(2*[2]))*[6]*exp((-(y-[7])**2)/(2*[8]))+[13]*[0]*exp((-(x-[1])**2)/(2*[2]))*[9]*exp((-(y-[10])**2)/(2*[11]))+[14]*[3]*exp((-(x-[4])**2)/(2*[5]))*[6]*exp((-(y-[7])**2)/(2*[8]))+[15]*[3]*exp((-(x-[4])**2)/(2*[5]))*[9]*exp((-(y-[10])**2)/(2*[11]))"
    gauss       = "(%f*exp(-0.5*((@0 - %f)/%f)**2))" % (p1_0,  p1_1,  p1_2)
    pol         = "(%f + %f * @0 + %f * @0**2)" %  (p2_0,  p2_1,  p2_2)
    gauss_not   = "(%f*exp(-0.5*((@1 - %f)/%f)**2))" %  (p3_0,  p3_1,  p3_2)
    pol_not     = "(%f + %f * @1 + %f * @1**2)" %  (p4_0,  p4_1,  p4_2)
    fit_func = ( "@2 *" + gauss + "*" + gauss_not +
                    "+@3 *" + gauss + "*" + pol_not +
                    "+@4*" + pol   + "*" + gauss_not +
                    "+@5*" + pol   + "*" + pol_not)
    return fit_func



