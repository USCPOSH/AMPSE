#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 18:57:55 2020

@author: mutian
"""

# Mohsen Hassanpourghadi

# Design VCO-TH-DRV with TF.

# ==================================================================
# *****************  Loading the libraries  ************************
# ==================================================================

import sys

# sys.path.insert(0,'/shares/MLLibs/GlobalLibrary')
sys.path.insert(0, '/home/mutian/python_codes/AMPSE2/GlobalLibrary')
# sys.path.insert(0,'D:/PYTHON_PHD/GlobalLibrary')


import os


from Netlist_Database import Compp_spice2, DACTH2_spice, Seqpart1_spice, Seqpart2_spice

from tensorflow_circuit import TF_DEFAULT, make_var, np_elu, np_sigmoid, np_sigmoid_inv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import math
import tensorflow as tf

from scipy.io import savemat
from pickle import dump

tf.config.optimizer.set_jit(True)

# ==================================================================
# *******************  Initialization  *****************************
# ==================================================================

KT = 4.14e-21  # Boltzman Constant * 300
# fs=1.0e8                    # Sampling Frequency
# nbit=10                     # Number of Bits

print(sys.argv)

nbit = int(sys.argv[1])
fs = float(sys.argv[2])
home_address = sys.argv[3]
opt = sys.argv[4]

tedad = 1
epsilon = 1e-4  # Epsilon in GD
# tedad=1                     # number of parameter candidates

maxiter = 10000  # Maximum iteration
n_stairs = 10  # Maximum number of quantized inverters
# ==================================================================
# ****************  Loading the Regressors  ************************
# ==================================================================
weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 100.0, 1 / 40])


# MR. Comparator
class COMPP2(TF_DEFAULT):
    # parameters: 'fck1','fck2','fcompinv','finnn','fninv','fpinv','fpr1','fpr2','frdynand','fttt','fnor3','frdy','mrdy','fload','mload'
    # metrics :'power','readyp','delayr','delayf','kickn','cin','scin','irn'
    def __init__(self, tech=65):
        self.tech = tech
        self.default_loading()

    def default_loading(self):
        if self.tech == 65:
            drive = home_address + '/Reg_files/PY_COMPPin6502_TT'
            sx_f = drive + '/scX_compp65.pkl'
            sy_f = drive + '/scY_compp65.pkl'
            w_f = drive + '/w8_compp65.p'
            self.w_json = drive + '/model_compp65.json'
            self.w_h5 = drive + '/reg_compp65.h5'

            self.minx = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 4, 1, 4]])
            self.maxx = np.array([[10, 20, 40, 20, 40, 40, 8, 8, 8, 80, 8, 32, 12, 16, 12]])
            self.step = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1]])

        self.loading(sx_f, sy_f, w_f)


# class COMPP(TF_DEFAULT):
#    # parameters: wn,  fn,  wp,   fp
#    # metrics :'Power', 'VCM','Vd','Noise','freq1','freq2','freq3','freq4','freq5','freq6','freq7','freq8'
#    def __init__(self,tech=65):
#
#        self.tech=tech
#        self.default_loading()
#
#    def default_loading(self):
#        if self.tech==65:
#            drive       = home_address+'/Reg_files/PY_COMPPin6502_TT'
#            sx_f        = drive + '/scX_compp65.pkl'
#            sy_f        = drive + '/scY_compp65.pkl'
#            w_f         = drive + '/w8_compp65.p'
#            self.w_json = drive + '/model_compp65.json'
#            self.w_h5   = drive + '/reg_compp65.h5'
#
#            self.minx = np.array([[1    , 1    , 1        , 1     , 1     , 1     , 1    , 1    , 1        , 2    ,   1   ,   1  ,  4   ,  1    ,  4    ]])
#            self.maxx = np.array([[10   , 20   , 40       , 20    , 40    , 40    , 8    , 8    , 8        , 80   ,   8   ,   10 ,  12  ,  10   ,  12   ]])
#            self.step = np.array([[1    , 1    , 1        , 1     , 1     , 1     , 1    , 1    , 1        , 2    ,   1   ,   1  ,  1   ,  1    ,  1    ]])
#        self.loading(sx_f,sy_f,w_f)

# MS. TH
class THDAC2(TF_DEFAULT):
    # parameters ['caps','fppp','wppp','swfn','swnn', 'swpp']

    # metrics : [cinn, cinp, clkfeed, ctot, kicknoise, trackbwMIN]

    def __init__(self, tech=65):
        self.tech = tech
        self.default_loading()

    def default_loading(self):
        if self.tech == 65:
            drive = home_address + '/Reg_files/PY_THDAC6502_TT'
            sx_f = drive + '/scX_th65.pkl'
            sy_f = drive + '/scY_th65.pkl'
            w_f = drive + '/w8_th65.p'
            self.w_json = drive + '/model_th65.json'
            self.w_h5 = drive + '/reg_th65.h5'
            self.minx = np.array([[2, 4, 0.5e-15, 2, 2, 2e-15]])
            self.maxx = np.array([[16, 12, 5.0e-15, 40, 60, 3e-14]])
            self.step = np.array([[2, 1, 0.5e-15, 2, 2, 0.5e-15]])
        self.loading(sx_f, sy_f, w_f)


class THDAC(TF_DEFAULT):
    # parameters ['caps','fppp','wppp','swfn','swnn', 'swpp']

    # metrics : [cinn, cinp, clkfeed, ctot, kicknoise, trackbwMIN]

    def __init__(self, tech=65):
        self.tech = tech
        self.default_loading()

    def default_loading(self):
        if self.tech == 65:
            drive = home_address + '/Reg_files/PY_THDAC6501_TT'
            sx_f = drive + '/scX_th65.pkl'
            sy_f = drive + '/scY_th65.pkl'
            w_f = drive + '/w8_th65.p'
            self.w_json = drive + '/model_th65.json'
            self.w_h5 = drive + '/reg_th65.h5'
            self.minx = np.array([[3, 0.5e-15, 2, 2, 1, 1, 2e-15]])
            self.maxx = np.array([[11, 5.0e-15, 40, 40, 10, 10, 3e-14]])
            self.step = np.array([[1, 0.5e-15, 2, 2, 1, 1, 2e-15]])
        self.loading(sx_f, sy_f, w_f)


# Baby DRV
class SEQ1(TF_DEFAULT):
    def __init__(self, tech=65):
        self.tech = tech
        self.default_loading()

    def default_loading(self):
        if self.tech == 65:
            drive = home_address + '/Reg_files/PY_SEQ16501_TT'
            sx_f = drive + '/scX_seqp165.pkl'
            sy_f = drive + '/scY_seqp165.pkl'
            w_f = drive + '/w8_seqp165.p'
            self.w_json = drive + '/model_seqp165.json'
            self.w_h5 = drive + '/reg_seqp165.h5'
            #            self.inv_spectre = INVSpice()
            #                                 [ 'drvbuf11','drvbuf21','drvbuf31','drvdffck1','drvinv1','drvnand1','fdffck','finv','fnand','frefnn','frefpp','div','mdacbig']
            self.minx = np.array([[1, 2, 2, 1, 2, 2, 1, 1, 1, 1, 1, 2, 3]])
            self.maxx = np.array([[12, 24, 48, 10, 16, 24, 10, 10, 10, 10, 10, 16, 11]])
            self.step = np.array([[1, 2, 2, 1, 2, 2, 1, 1, 1, 1, 1, 2, 1]])

        self.loading(sx_f, sy_f, w_f)


class SEQ2(TF_DEFAULT):
    def __init__(self, tech=65):
        self.tech = tech
        self.default_loading()

    def default_loading(self):
        if self.tech == 65:
            drive = home_address + '/Reg_files/PY_SEQ26501_TT'
            sx_f = drive + '/scX_seqp265.pkl'
            sy_f = drive + '/scY_seqp265.pkl'
            w_f = drive + '/w8_seqp265.p'
            self.w_json = drive + '/model_seqp265.json'
            self.w_h5 = drive + '/reg_seqp265.h5'
            #            self.inv_spectre = INVSpice()

            self.minx = np.array([[1, 1]])
            self.maxx = np.array([[10, 12]])
            self.step = np.array([[1, 1]])
        self.loading(sx_f, sy_f, w_f)


def tf_quant_with_sigmoid(sxin, num=10):
    v = np.linspace(0.5 / num, 1 - 0.5 / num, num)
    out = []
    for vv in v:
        out.append(tf.nn.sigmoid(100.0 * (sxin - vv)))

    return tf.reduce_sum(out) + 1.0


def param_to_sxin(param, seqp1, seqp2, compp, thdac):
    x_seqp11 = param[0][0]
    x_seqp21 = param[1][0]
    x_compp1 = param[2][0]
    x_th1 = param[3][0]
    x_ndly = [(param[4] - 1) / n_stairs]
    x_dtr = [2 * param[5] - 1]

    sx_seqp11 = seqp1.np_scalex(x_seqp11)
    sx_seqp21 = seqp2.np_scalex(x_seqp21)
    sx_compp1 = compp.np_scalex(x_compp1)
    sx_th1 = thdac.np_scalex(x_th1)

    cx_seqp11 = list(sx_seqp11[[0, 1, 2, 3, 4, 5, 6, 7, 8, 11]])
    cx_seqp21 = list(sx_seqp21)
    cx_compp1 = list(sx_compp1[1:10])
    cx_th1 = list(sx_th1[2:5])

    sx_out = np.array([cx_seqp11 + cx_seqp21 + cx_compp1 + cx_th1 + x_ndly + x_dtr])

    return sx_out


def step2sxin(seqp1, seqp2, compp, thdac):
    n_stairs = 10
    ss_seqp1 = seqp1.step * seqp1.scXscale
    ss_seqp2 = seqp2.step * seqp2.scXscale
    ss_compp = compp.step * compp.scXscale
    ss_thdac = thdac.step * thdac.scXscale

    sx_out = np.array([list(ss_seqp1[0][[0, 1, 2, 3, 4, 5, 6, 7, 8, 11]]) + list(ss_seqp2[0]) + list(
        ss_compp[0][1:10]) + list(ss_thdac[0][1:4]) + [1 / n_stairs, 0.01]])
    return sx_out


# ==================================================================
# *****************  Building the graph  ***************************
# ==================================================================
def graph_tf3(sxin, seqp11, seqp21, compp1, thdac1):
    # Constants:
    c_nbit = tf.constant(nbit, dtype=tf.float32)
    c_lvls = tf.constant(2 ** (nbit - 1), dtype=tf.float32)
    # ----------SEQ1's graph----------
    chosen = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0])
    vars_seqp11 = sxin[:, 0:10]
    cnst_seqp11 = seqp11.tf_scalex2(tf.stack([1.0, 1.0, c_nbit - 1], axis=0), 1 - chosen)
    sx_seqp11 = tf.reshape(
        tf.concat((vars_seqp11[0][0:9], cnst_seqp11[0:2], vars_seqp11[0][9:10], cnst_seqp11[2:3]), axis=0), [1, 13])
    x_seqp11 = seqp11.tf_rescalex(sx_seqp11)
    sy_seqp11 = seqp11.tf_reg_sigmoid(sx_seqp11)  # sigmoid usage
    y_seqp11 = seqp11.tf_rescaley(sy_seqp11)

    # ----------SEQ2's graph----------
    vars_seqp21 = sxin[:, 10:12]
    sx_seqp21 = vars_seqp21
    x_seqp21 = seqp21.tf_rescalex(sx_seqp21)
    sy_seqp21 = seqp21.tf_reg_sigmoid(sx_seqp21)  # sigmoid usage
    y_seqp21 = seqp21.tf_rescaley(sy_seqp21)

    # ----------COMP's graph----------
    chosen = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    vars_compp1 = sxin[:, 12:21]
    cnst_commp1 = compp1.tf_scalex2(
        tf.stack([x_seqp21[0, 1], x_seqp21[0, 0], 2 * x_seqp11[0, 7] + x_seqp11[0, 6], c_nbit, x_seqp11[0, 4], c_nbit],
                 axis=0), 1 - chosen)
    sx_compp1 = tf.reshape(tf.concat((cnst_commp1[0:1], vars_compp1[0], cnst_commp1[1:]), axis=0), [1, 15])
    x_compp1 = compp1.tf_rescalex(sx_compp1)
    sy_compp1 = compp1.tf_reg_sigmoid(sx_compp1)  # changed to sigmoid
    y_compp1 = compp1.tf_rescaley(sy_compp1)

    # ----------THDAC's graph----------
    #    'div','mdac',   'cs', 'fthn', 'fthp',   'cp'
    chosen = np.array([0, 0, 1, 1, 1, 0])
    vars_th1 = sxin[:, 21:24]
    cnst_th1 = thdac1.tf_scalex2(tf.stack((x_seqp11[0, 11], c_nbit, y_compp1[0, 5]), axis=0), 1 - chosen)
    sx_th1 = tf.reshape(tf.concat((cnst_th1[0:2], vars_th1[0][0:], cnst_th1[2:]), axis=0), [1, 6])
    x_th1 = thdac1.tf_rescalex(sx_th1)
    sy_th1 = thdac1.tf_reg_sigmoid(sx_th1)
    y_th1 = thdac1.tf_rescaley(sy_th1)

    # --------Other variables--------
    #    n_dly = tf_quant_with_sigmoid(tf.math.abs(sxin[0,25]),10)
    #    d_tr  = sxin[0,26]
    n_dly = tf_quant_with_sigmoid(tf.math.abs(sxin[0, 24]), n_stairs)
    d_tr = (sxin[0, 25] + 1) / 2.0
    # ==================================================================
    # ***************  Define constraints and Cost(P)  *****************
    # ==================================================================
    specs = []

    dacdelay = y_th1[0, 3] + y_seqp11[0, 2] + y_seqp11[0, 3]
    digdelay = y_seqp21[0, 3] + y_seqp21[0, 5] + n_dly * y_seqp21[0, 4]
    ctot = x_th1[0, 1] * 2 * c_lvls
    bw1 = tf.nn.elu(y_th1[0, 0])
    bw2 = tf.nn.elu(y_th1[0, 1])
    #    d_ts  = tf.nn.sigmoid( d_tr)*(1.0-fs*100e-12)+fs*100e-12
    d_ts = d_tr * (1.0 - fs * 200e-12) + fs * 200e-12

    ts = d_ts * 1 / fs
    #   0       1          2        3       4       5   6       7
    # 'power','readyp','delayr','delayf','kickn','cin','scin','irn'
    specs.append(digdelay - dacdelay - 20e-12)  # 0- Delay of DAC vs delay of the loop +
    specs.append(1 / fs - ts - c_nbit * (
                y_compp1[0, 1] + y_compp1[0, 2] + 2 * digdelay) - 100e-12)  # 1- loop delay less than fs  +
    specs.append(10 * tf.math.log((4 * y_th1[0, 2]) ** 2 / (4 * KT / ctot + y_compp1[0, 7] ** 2)) / tf.math.log(
        10.0))  # 2- SNR more than 6*nbit + 11.76
    specs.append(x_th1[0, 2] - 6 * y_compp1[0, 6])  # 3- Comparator non-linear Caps +
    specs.append(ts * bw1 * 6.28 - tf.math.log(2.0) * c_nbit)  # 4- Track and Hold Law +
    specs.append(ts * bw2 * 6.28 - tf.math.log(2.0) * c_nbit)  # 5- Track and Hold Law +
    specs.append(4 - ts * bw1 * 6.28 + tf.math.log(2.0) * c_nbit)  # 6- Track and Hold Law +
    specs.append(4 - ts * bw2 * 6.28 + tf.math.log(2.0) * c_nbit)  # 7- Track and Hold Law +
    specs.append(tf.nn.sigmoid(10.0 * (y_compp1[0, 3] - 0.5)))  # 8- delayf must be 0 -
    specs.append(2 * y_seqp11[0, 0] + c_nbit * y_seqp11[0, 1] + (
                y_seqp21[0, 0] + n_dly * y_seqp21[0, 1] + y_seqp21[0, 2] + y_compp1[
            0, 0]) * c_nbit)  # 9- Power consumption -

    constraints = []
    constraints.append(tf.nn.elu(-specs[0] / digdelay * weights[0]))
    constraints.append(tf.nn.elu(-specs[1] / thdac1.scYscale[3] * weights[1]))
    constraints.append(tf.nn.elu(nbit + 10 / 6 - specs[2] / 6 * weights[2]))
    constraints.append(tf.nn.elu(-specs[3] / compp1.scYscale[5] * weights[3]))
    constraints.append(tf.nn.elu(-specs[4] * weights[4]))
    constraints.append(tf.nn.elu(-specs[5] * weights[5]))
    constraints.append(tf.nn.elu(-specs[6] * weights[6]))
    constraints.append(tf.nn.elu(-specs[7] * weights[7]))
    constraints.append(specs[8] * weights[8])
    constraints.append(specs[-1] / compp1.scYscale[0] * weights[9])

    softcost = tf.reduce_sum(constraints[:-1])
    hardcost = tf.reduce_sum(constraints)

    return hardcost, softcost, specs, [x_seqp11, x_seqp21, x_compp1, x_th1, n_dly, d_tr], [y_seqp11, y_seqp21, y_compp1,
                                                                                           y_th1, ts], [dacdelay,
                                                                                                        digdelay, ctot,
                                                                                                        d_ts, bw1,
                                                                                                        bw2], constraints


# ==================================================================
# *****************  Building the graph  ***************************
# ==================================================================
def graph_spice3(sxin, seqp11, seqp21, compp1, thdac1, seq1spice, seq2spice, comppspice, thdacspice, put_on_csv=False):
    # Constants:
    c_nbit = nbit
    c_lvls = 2 ** (nbit - 1)
    # ----------SEQ1's graph----------
    chosen = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0])
    vars_seqp11 = sxin[:, 0:10]
    cnst_seqp11 = seqp11.np_scalex2(np.stack([1.0, 1.0, c_nbit - 1], axis=0), 1 - chosen)

    sx_seqp11 = np.reshape(
        np.concatenate((vars_seqp11[0][0:9], cnst_seqp11[0:2], vars_seqp11[0][9:10], cnst_seqp11[2:3]), axis=0),
        [1, 13])

    x_seqp11 = seqp11.np_rescalex(sx_seqp11)
    x_seqp11, d_seqp11 = seq1spice.wholerun_std(x_seqp11[0], put_on_csv)
    y_seqp11 = np.array([d_seqp11])

    # ----------SEQ2's graph----------
    vars_seqp21 = sxin[:, 10:12]
    sx_seqp21 = vars_seqp21
    x_seqp21 = seqp21.np_rescalex(sx_seqp21)
    x_seqp21, d_seqp21 = seq2spice.wholerun_std(x_seqp21[0], put_on_csv)
    y_seqp21 = np.array([d_seqp21])

    # ----------COMP's graph----------
    chosen = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    vars_compp1 = sxin[:, 12:21]
    cnst_commp1 = compp1.np_scalex2(
        np.stack([x_seqp21[1], x_seqp21[0], 2 * x_seqp11[7] + x_seqp11[6], c_nbit, x_seqp11[4], c_nbit], axis=0),
        1 - chosen)
    sx_compp1 = np.reshape(np.concatenate((cnst_commp1[0:1], vars_compp1[0], cnst_commp1[1:]), axis=0), [1, 15])
    x_compp1 = compp1.np_rescalex(sx_compp1)
    x_compp1, d_compp1 = comppspice.wholerun_std(np.array(list(x_compp1[0]) + [1, 1, 100e-6, 0.4, 1.0]), put_on_csv)
    y_compp1 = np.array([np.array(d_compp1)])

    # ----------THDAC's graph----------
    chosen = np.array([0, 0, 1, 1, 1, 0])
    vars_th1 = sxin[:, 21:24]
    cnst_th1 = thdac1.np_scalex2(np.stack((x_seqp11[11], c_nbit, y_compp1[0, 5]), axis=0), 1 - chosen)
    sx_th1 = np.reshape(np.concatenate((cnst_th1[0:2], vars_th1[0][0:], cnst_th1[2:]), axis=0), [1, 6])
    x_th1 = thdac1.np_rescalex(sx_th1)
    x_th1, d_th1 = thdacspice.wholerun_std(np.array(list(x_th1[0][:-1]) + [1, 1, x_th1[0][-1]]), put_on_csv)
    y_th1 = np.array([d_th1])

    # --------Other variables--------

    n_dly = np.round(np.abs(sxin[0, 24]) * n_stairs + 1.0)
    d_tr = (sxin[0, 25] + 1) / 2.0
    # ==================================================================
    # ***************  Define constraints and Cost(P)  *****************
    # ==================================================================

    dacdelay = y_th1[0, 3] + y_seqp11[0, 2] + y_seqp11[0, 3]
    digdelay = y_seqp21[0, 3] + y_seqp21[0, 5] + n_dly * y_seqp21[0, 4]
    ctot = x_th1[1] * 2 * c_lvls
    bw1 = y_th1[0, 0]
    bw2 = y_th1[0, 1]
    d_ts = d_tr * (1.0 - fs * 200e-12) + fs * 200e-12
    ts = d_ts * 1 / fs

    specs = []
    specs.append(digdelay - dacdelay - 20e-12)  # 0- Delay of DAC vs delay of the loop +
    specs.append(1 / fs - ts - c_nbit * (
                y_compp1[0, 1] + y_compp1[0, 2] + 2 * digdelay) - 100e-12)  # 1- loop delay less than fs  +
    specs.append(10 * np.log((4 * y_th1[0, 2]) ** 2 / (4 * KT / ctot + y_compp1[0, 7] ** 2)) / np.log(
        10.0))  # 2- SNR more than 6*nbit + 11.76
    specs.append(x_th1[2] - 6 * y_compp1[0, 6])  # 3- Comparator non-linear Caps +
    specs.append(ts * bw1 * 6.28 - np.log(2.0) * c_nbit)  # 4- Track and Hold Law +
    specs.append(ts * bw2 * 6.28 - np.log(2.0) * c_nbit)  # 5- Track and Hold Law +
    specs.append(4 - ts * bw1 * 6.28 + np.log(2.0) * c_nbit)  # 4- Track and Hold Law +
    specs.append(4 - ts * bw2 * 6.28 + np.log(2.0) * c_nbit)  # 5- Track and Hold Law +
    specs.append(y_compp1[0, 3])  # 6- zero or one for comparator
    specs.append((y_seqp11[0, 0] + y_seqp11[0, 1] + y_seqp21[0, 0] + n_dly * y_seqp21[0, 1] + y_seqp21[0, 2]) * c_nbit +
                 y_compp1[0, 0])  # 7- Power consumption -

    constraints = []
    constraints.append(np_elu(-specs[0] / thdac1.scYscale[3] * weights[0]))
    constraints.append(np_elu(-specs[1] / thdac1.scYscale[3] * weights[1]))
    constraints.append(np_elu(nbit + 10 / 6 - specs[2] / 6 * weights[2]))
    constraints.append(np_elu(-specs[3] / compp1.scYscale[5] * weights[3]))
    constraints.append(np_elu(-specs[4] * weights[4]))
    constraints.append(np_elu(-specs[5] * weights[5]))
    constraints.append(np_elu(-specs[6] * weights[6]))
    constraints.append(np_elu(-specs[7] * weights[7]))
    constraints.append(specs[8] * weights[8])
    constraints.append(specs[-1] / compp1.scYscale[0] * weights[9])

    softcost = sum(constraints[:-1])
    hardcost = sum(constraints)

    return hardcost, softcost, specs, [x_seqp11, x_seqp21, x_compp1, x_th1, n_dly, d_tr], [y_seqp11, y_seqp21, y_compp1,
                                                                                           y_th1, ts], [dacdelay,
                                                                                                        digdelay, ctot,
                                                                                                        d_ts, bw1,
                                                                                                        bw2], constraints


# ==================================================================
# *********************  Main code  ********************************
# ==================================================================

if __name__ == '__main__':

    y = np.loadtxt('rnddata.csv', delimiter=',')
    # ==================================================================
    # *****************  Building the graph  ***************************
    # ==================================================================

    tf.compat.v1.disable_eager_execution()
    # ----------Initialize----------
    tf.compat.v1.reset_default_graph()
    
    if opt == 'Adams':
        #    optimizer1 = tf.compat.v1.train.GradientDescentOptimizer(0.01)
        #    optimizer1 = tf.compat.v1.train.RMSPropOptimizer(0.01)
        #    optimizer1 = tf.compat.v1.train.AdadeltaOptimizer(0.01)
        #    optimizer1 = tf.compat.v1.train.AdagradOptimizer(0.01)
        optimizer1 = tf.compat.v1.train.AdamOptimizer(0.01)
    
        #    optimizer2 = tf.compat.v1.train.GradientDescentOptimizer(0.001)
        #    optimizer2 = tf.compat.v1.train.RMSPropOptimizer(0.001)
        #    optimizer2 = tf.compat.v1.train.AdadeltaOptimizer(0.001)
        #    optimizer2 = tf.compat.v1.train.AdagradOptimizer(0.001)
        optimizer2 = tf.compat.v1.train.AdamOptimizer(0.001)
        
    elif opt == 'GD':
         optimizer1 = tf.compat.v1.train.GradientDescentOptimizer(0.01)
         optimizer2 = tf.compat.v1.train.GradientDescentOptimizer(0.001)

    # ----------load regressors----------
    seqp11 = SEQ1()
    seqp21 = SEQ2()
    compp1 = COMPP2()
    thdac1 = THDAC2()

    comppspice1 = Compp_spice2()
    dacthspice1 = DACTH2_spice()
    seqp1spice1 = Seqpart1_spice()
    seqp2spice1 = Seqpart2_spice()

    var_in = make_var("SAR_ADC", "SEQ_COMPP_THDAC", (1, 26),
                      tf.random_uniform_initializer(-np.ones((1, 26)), np.ones((1, 26))))

    xload = tf.compat.v1.placeholder(tf.float32, shape=(1, 26))
    initvar = var_in.assign(xload)

    sxin = 2 * tf.math.sigmoid(var_in) - 1.0

    hardcost, softcost, tf_specs, tf_params, tf_metrics, tf_mids, tf_const = graph_tf3(sxin, seqp11, seqp21, compp1,
                                                                                       thdac1)

    #    tf.float32.real_dtype
    # ==================================================================
    # ********************  Tensorflow Initiation  *********************
    # ==================================================================

    grd1 = optimizer1.compute_gradients(softcost)
    opt1 = optimizer1.apply_gradients(grd1)
    grd2 = optimizer2.compute_gradients(hardcost)
    opt2 = optimizer2.apply_gradients(grd2)
    init = tf.compat.v1.global_variables_initializer()

    calc = 1

    lastvalue = -1000000
    lst_params = []
    lst_metrics = []
    lst_specs = []
    lst_value = []
    lst_midvalues = []
    tstart = time.time()

    print('Algorithm Starts Here:')
    k = 0
    for j in range(tedad):
        reg_specs = []
        const = []

        lst_amps = []
        lst_vcms = []
        lst_coefs = []
        lst_mets = []
        with tf.compat.v1.Session() as sess:
            sess.run(init)
            sess.run(initvar, feed_dict={xload: y[j:(j + 1), :]})
            # ==================================================================
            # *****************  Tensorflow Gradient Descent  ******************
            # ==================================================================

            i = 0
            while i < maxiter:

                try:
                    if i < maxiter / 2:
                        _, value, smallspecs = sess.run([opt1, softcost, tf_specs])
                    else:
                        _, value, smallspecs = sess.run([opt2, hardcost, tf_specs])

                    k += 1
                except:
                    print('Terminated due to error!')
                    break
                print('%1.0f:, %1.0f : %1.3f \n' % (j, i, value))
                reg_specs.append(smallspecs)
                if math.isnan(value) or math.isinf(value):
                    break
                else:
                    smallspecs = sess.run(tf_specs)
                    parameters = sess.run(tf_params)
                    metrics = sess.run(tf_metrics)
                    midvalues = sess.run(tf_mids)
                    const.append(sess.run(tf_const))
                    np_sxin = sess.run(sxin)
                    dacdelay, digdelay, ctot, d_ts, bw1, bw2 = midvalues

                    #                    lst_coefs.append(coefs_vco)
                    lst_mets.append(metrics[0])
                if i < maxiter / 2 and np.abs(lastvalue - value) < 0.001:
                    i = int(maxiter / 2) + 1
                    print(i)
                elif i > maxiter / 2 and np.abs(lastvalue - value) < epsilon:
                    break
                else:
                    lastvalue = value
                i += 1

            #            if not math.isnan(value):

            # ==================================================================
            # **********************  Saving the values  ***********************
            # ==================================================================
            tend = time.time()

            #            print('user1: %1.2f, user2: %1.2f, user3: %1.2f, user4: %1.2f, user5: %1.2f\n' %(sess.run(user1),sess.run(user2),sess.run(user3),sess.run(user4),sess.run(user5)))
            print('the elapsed time %1.2f S\n' % (tend - tstart))

        var_specs = np.array(reg_specs)
        lst_params.append(parameters)
        lst_metrics.append(metrics)
        lst_specs.append(reg_specs[-1])
        lst_value.append(value)
        lst_midvalues.append(midvalues)

        mydict = {'lst_params': lst_params, 'lst_metrics': lst_metrics, 'lst_specs': lst_specs, 'lst_value': lst_value}
        dump(mydict, open('regsearch_results1_' + str(nbit) + str(fs) + '.p', "wb"))
        #        6bit:
        #        k=218387
        #        elapsed=4536
        #        8bit:
        #        k=163990
        #        elapsed=3540
        #        10bit:
        #        k=145073
        #        elapsed=3205.83

        savemat('regsearch_constraints.mat', {'const_np': var_specs})

#        sp_value,_,sp_specs,sp_params,sp_metrics,sp_mids,sp_const= graph_spice3(np_sxin,seqp11,seqp21,compp1,thdac1,seqp1spice1,seqp2spice1,comppspice1,dacthspice1,False)


#        sp_sxin = param_to_sxin(parameters,seqp11,seqp21,compp1,thdac1)
#        sp_value,_,sp_specs,sp_params,sp_metrics,sp_mids,sp_const= graph_spice3(sp_sxin,seqp11,seqp21,compp1,thdac1,seqp1spice1,seqp2spice1,comppspice1,dacthspice1,False)


# Mohsen Hassanpourghadi

# Design VCO-TH-DRV with TF.

# ==================================================================
# *****************  Loading the libraries  ************************
# ==================================================================

import sys

# sys.path.insert(0,'/shares/MLLibs/GlobalLibrary')
import numpy as np
# from VCOSpectre import obj as vco_simul
# from THSpectre import obj as th_simul
#from AMPSE_Graphs import COMPP2, THDAC2, SEQ1, SEQ2
from tensorflow_circuit import action2sxin, rw2, vector_constraints
from pickle import load
from Netlist_Database import Compp_spice2, DACTH2_spice, Seqpart1_spice, Seqpart2_spice
#from AMPSE_Graphs import KT, graph_tf3, graph_spice3, param_to_sxin, step2sxin
import matplotlib.pyplot as plt

#nbit = int(sys.argv[1])
#fs = float(sys.argv[2])

#graph_spice3.nbit = nbit

file_ampse = 'regsearch_results1_' + str(nbit) + str(fs) + '.p'
# 6bit  1GS/s:
# 146296 - 2856.75S


tedad = 50


def choose_best(dict_loads, tedad):
    lst_params = dict_loads['lst_params']
    lst_metrics = dict_loads['lst_metrics']
    lst_specs = dict_loads['lst_specs']
    lst_values = dict_loads['lst_value']
    chosen_np = np.argsort(lst_values)[:tedad][::+1]
    lst_params_chosen = [lst_params[i] for i in chosen_np]
    lst_metrics_chosen = [lst_metrics[i] for i in chosen_np]
    lst_specs_chosen = [lst_specs[i] for i in chosen_np]
    lst_values_chosen = [lst_values[i] for i in chosen_np]

    return lst_params_chosen, lst_metrics_chosen, lst_specs_chosen, lst_values_chosen, chosen_np


def choose_one(dict_loads, i):
    lst_params = dict_loads['lst_params']
    lst_metrics = dict_loads['lst_metrics']
    lst_specs = dict_loads['lst_specs']
    lst_values = dict_loads['lst_value']
    lst_params_chosen = lst_params[i]
    lst_metrics_chosen = lst_metrics[i]
    lst_specs_chosen = lst_specs[i]
    lst_values_chosen = lst_values[i]

    return lst_params_chosen, lst_metrics_chosen, lst_specs_chosen, lst_values_chosen, 1


def test_spice(lst_params_chosen, seqp11, seqp21, compp1, thdac1, seqp1spice1, seqp2spice1, comppspice1, dacthspice1):
    lst_metrics_spice = []
    lst_specs_spice = []
    lst_value_spice = []
    lst_mids_spice = []
    lst_const_spice = []

    for i in range(len(lst_params_chosen)):
        sp_sxin = param_to_sxin(lst_params_chosen[i], seqp11, seqp21, compp1, thdac1)
        sp_value, _, sp_specs, sp_params, sp_metrics, sp_mids, sp_const = graph_spice3(sp_sxin, seqp11, seqp21, compp1,
                                                                                       thdac1, seqp1spice1, seqp2spice1,
                                                                                       comppspice1, dacthspice1, True)

        lst_metrics_spice.append(sp_metrics)
        lst_specs_spice.append(sp_specs)
        lst_value_spice.append(sp_value)
        lst_mids_spice.append(sp_mids)
        lst_const_spice.append(sp_const)

    return lst_metrics_spice, lst_specs_spice, lst_value_spice, lst_mids_spice, lst_const_spice


if __name__ == '__main__':

    # ==================================================================
    # *******************  Initialization  *****************************
    # ==================================================================
    comppspice1 = Compp_spice2()
    dacthspice1 = DACTH2_spice()
    seqp1spice1 = Seqpart1_spice()
    seqp2spice1 = Seqpart2_spice()

    compp1 = COMPP2()
    thdac1 = THDAC2()
    seqp11 = SEQ1()
    seqp21 = SEQ2()

    what_to_do = 0

    # ==================================================================
    # *******************  Initialization  *****************************
    # ==================================================================
    dict_loads = load(open(file_ampse, "rb"))
    lst_params_chosen, lst_metrics_chosen, lst_specs_chosen, lst_values_chosen, chosen_np = choose_best(dict_loads,
                                                                                                        tedad)

    if what_to_do == 0:
        sp_sxin = param_to_sxin(lst_params_chosen[0], seqp11, seqp21, compp1, thdac1)
        sp_value, _, sp_specs, sp_params, sp_metrics, sp_mids, sp_const = graph_spice3(sp_sxin, seqp11, seqp21, compp1,
                                                                                       thdac1, seqp1spice1, seqp2spice1,
                                                                                       comppspice1, dacthspice1)

    elif what_to_do == 1:
        lst_metrics_spice, lst_specs_spice, lst_value_spice, lst_mids_spice, lst_const_spice = test_spice(
            lst_params_chosen, seqp11, seqp21, compp1, thdac1, seqp1spice1, seqp2spice1, comppspice1, dacthspice1)
        np_specs_spice = np.array(lst_specs_spice)
        np_specs_chosen = np.array(lst_specs_chosen)
        np.savetxt('np_specs_spice10.csv', np_specs_spice, delimiter=',')
        np.savetxt('np_specs_chosen10.csv', np_specs_chosen, delimiter=',')

    elif what_to_do == 2:
        lst_rw_specs = []
        lst_rw_action = []
        lst_rw_value = []
        dd = np.array([1, 1, 1, 1, 1, 1, 1])
        prev_sxin = param_to_sxin(lst_params_chosen[43], seqp11, seqp21, compp1, thdac1)
        new_value, _, sp_specs, sp_params, sp_metrics, sp_mids, prev_const = graph_spice3(prev_sxin, seqp11, seqp21,
                                                                                          compp1, thdac1, seqp1spice1,
                                                                                          seqp2spice1, comppspice1,
                                                                                          dacthspice1)
        n_action = len(prev_sxin[0])
        action = 2 * n_action
        bad_action = 2 * n_action
        goodjob = 0
        ssin = step2sxin(seqp11, seqp21, compp1, thdac1)
        lst_rw_specs.append(sp_specs)
        lst_rw_action.append(action)
        lst_rw_value.append(new_value)
        for i in range(tedad):
            action = rw2(n_action, bad_action)
            new_sxin, _ = action2sxin(action, prev_sxin, ssin)

            sp_value, _, sp_specs, sp_params, sp_metrics, sp_mids, new_const = graph_spice3(new_sxin, seqp11, seqp21,
                                                                                            compp1, thdac1, seqp1spice1,
                                                                                            seqp2spice1, comppspice1,
                                                                                            dacthspice1)

            reward = vector_constraints(prev_const, new_const, dd)

            if reward > 0:
                prev_sxin = new_sxin
                prev_const = new_const
                bad_action = 2 * n_action
                new_value = sp_value
                goodjob += 1
            else:
                bad_action = action

            lst_rw_specs.append(sp_specs)
            lst_rw_action.append(action)
            lst_rw_value.append(new_value)
            print(reward)

        np_rw_specs = np.array(lst_rw_specs)
        np_rw_action = np.array(lst_rw_action)
        np_rw_value = np.array(lst_rw_value)

        from scipy.io import savemat

        savemat('regsearch_rw1.mat', {'rw_spec': np_rw_specs, 'rw_action': np_rw_action, 'rw_value': np_rw_value})

    elif what_to_do == 3:
        import time

        lst_gr_specs = []
        lst_gr_value = []
        prev_sxin = param_to_sxin(lst_params_chosen[0], seqp11, seqp21, compp1, thdac1)
        sp_value, _, sp_specs, sp_params, sp_metrics, sp_mids, prev_const = graph_spice3(prev_sxin, seqp11, seqp21,
                                                                                         compp1, thdac1, seqp1spice1,
                                                                                         seqp2spice1, comppspice1,
                                                                                         dacthspice1)
        prev_value = sp_value
        ssin = step2sxin(seqp11, seqp21, compp1, thdac1)
        num_step = len(ssin[0])

        dy = np.zeros_like(ssin)
        lst_gr_specs.append(sp_specs)
        lst_gr_value.append(prev_value)
        tstart = time.time()
        for j in range(tedad):
            for i in range(num_step):
                dx = ssin[0, i]
                new_sxin, _ = action2sxin(i, prev_sxin, ssin)
                new_value, _, sp_specs, sp_params, sp_metrics, sp_mids, prev_const = graph_spice3(new_sxin, seqp11,
                                                                                                  seqp21, compp1,
                                                                                                  thdac1, seqp1spice1,
                                                                                                  seqp2spice1,
                                                                                                  comppspice1,
                                                                                                  dacthspice1)

                dy[0, i] = (new_value - prev_value) / dx
            lr = 0.000002
            pro_sxin = prev_sxin - lr * dy
            new_value, _, sp_specs, sp_params, sp_metrics, sp_mids, prev_const = graph_spice3(pro_sxin, seqp11, seqp21,
                                                                                              compp1, thdac1,
                                                                                              seqp1spice1, seqp2spice1,
                                                                                              comppspice1, dacthspice1)
            print(new_value, j, time.time() - tstart)
            prev_value = new_value
            prev_sxin = pro_sxin
            lst_gr_specs.append(sp_specs)
            lst_gr_value.append(prev_value)

        np_gr_specs = np.array(lst_gr_specs)
        np_gr_value = np.array(lst_gr_value)

        from scipy.io import savemat

        savemat('regsearch_gradients.mat', {'gr_spec': np_gr_specs, 'rw_value': np_gr_value})
print('The specifications are \n\
0- Delay of DAC vs delay of the loop \n\
1- loop delay less than fs  \n\
2- SNR more than 6*nbit + 11.76 \n\
3- Comparator non-linear Caps  \n\
4- Track and Hold Law \n\
5- Track and Hold Law \n\
4- Track and Hold Law \n\
5- Track and Hold Law \n\
6- zero or one for comparator\n\
7- Power consumption - \n')

print('The specs predicted by the regressor are')
print(lst_specs_chosen[0])
print('The specs simulated by afs are \n')
print(sp_specs)




