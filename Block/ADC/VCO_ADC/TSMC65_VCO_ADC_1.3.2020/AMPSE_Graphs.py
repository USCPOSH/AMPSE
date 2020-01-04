# Designed by Mohsen Hassanpourghadi
# Sanitized by Mohsen Hassanpourghadi



#==================================================================
#*****************  Loading the libraries  ************************
#==================================================================

import sys
sys.path.insert(0,'/GlobalLibrary')

import os
home_address  = os.getcwd()
from tensorflow_circuit import TF_DEFAULT, make_var, np_elu, np_sigmoid, np_sigmoid_inv
from Netlist_Database import VCOSpice,INBUF2Spice

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import math
import tensorflow as tf

from scipy.io import savemat
from pickle import dump


#==================================================================
#**********************  User Intent  *****************************
#==================================================================

bw=0.04e9                   # Bandwidth
nbit=10                     # Number of Bits

#==================================================================
#*******************  Initialization  *****************************
#==================================================================

KT=4.14e-21                 # Boltzman Constant * 300
epsilon=1e-5                # Epsilon in GD
tedad=50                    # number of parameter candidates 
fing_clk=4                  # number of fingers for the initial driver
maxiter=10000
minosr=8
maxosr=1000
minamp=0.05
maxamp=0.3
weights = np.array([1.,1.,1.,1.,0.,0.,1000.,1000.,1000.,1.])
#==================================================================
#****************  Loading the Regressors  ************************
#==================================================================


# MR. INBUF2
class INBUF2(TF_DEFAULT):
    def __init__(self,tech=65):

        self.tech=tech
        self.default_loading()
        
    def default_loading(self):
        if self.tech==65:
            drive       = home_address+'/reg_files/INBUF2'
            sx_f        = drive + '/scX_inbuf265.pkl'
            sy_f        = drive + '/scY_inbuf265.pkl'    
            w_f         = drive + '/w8_inbuf265.p'
            self.w_json = drive + '/model_inbuf265.json'
            self.w_h5   = drive + '/reg_inbuf265.h5'
            self.minx  = np.array([[1      ,1        , 60e-9 ,1         ,0.55 ,0.2e-6 , 2     ]])
            self.maxx  = np.array([[20     ,50       ,400e-9 ,50        ,0.9  ,1.2e-6 , 20    ]])
            self.step  = np.array([[1      ,1        , 10e-9 ,1         ,0.01 ,10.0e-9, 1     ]])
        self.loading(sx_f,sy_f,w_f)

# MR. VCO
class VCO(TF_DEFAULT):
    def __init__(self,tech=65):

        self.tech=tech
        self.default_loading()
        
    def default_loading(self):
        if self.tech==65:
            drive       = home_address+'/reg_files/VCO'
            sx_f        = drive + '/scX_vco65.pkl'
            sy_f        = drive + '/scY_vco65.pkl'    
            w_f         = drive + '/w8_vco65.p'
            self.w_json = drive + '/model_vco65.json'
            self.w_h5   = drive + '/reg_vco65.h5'
            self.minx = np.array([[0.2e-6 ,2  ,0.2e-6,2  ]])
            self.maxx = np.array([[1.2e-6 ,20 ,1.2e-6,20 ]])
            self.step = np.array([[1.0e-8 ,1  ,1.0e-8,1  ]])
        self.loading(sx_f,sy_f,w_f)        


#==================================================================
#*******************  Functions  *****************************
#==================================================================

def multi_fitting(coef1,coef2):
    a0=coef1[0]
    a1=coef1[1]
    a2=coef1[2]
    a3=coef1[3]
    b0=coef2[0]
    b1=coef2[1]
    b2=coef2[2]
    b3=coef2[3]
    b4=coef2[4]
    b5=coef2[5]
    b6=coef2[6]
    b7=coef2[7]
    
    c21= a3**7*b7
    c20= 7*a2*a3**6*b7
    c19= 7*a3**5*b7*(3*a2**2 + a1*a3)
    c18= a3**4*(a3**2*b6 + 35*a2**3*b7 + 7*a0*a3**2*b7 + 42*a1*a2*a3*b7)
    c17= a3**3*(35*a2**4*b7 + 21*a1**2*a3**2*b7 + 6*a2*a3**2*b6 + 42*a0*a2*a3**2*b7 + 105*a1*a2**2*a3*b7)
    c16= a3**2*(21*a2**5*b7 + 15*a2**2*a3**2*b6 + 6*a1*a3**3*b6 + 42*a0*a1*a3**3*b7 + 140*a1*a2**3*a3*b7 + 105*a0*a2**2*a3**2*b7 + 105*a1**2*a2*a3**2*b7)
    c15= a3*(21*b7*a0**2*a3**4 + 210*b7*a0*a1*a2*a3**3 + 140*b7*a0*a2**3*a3**2 + 6*b6*a0*a3**4 + 35*b7*a1**3*a3**3 + 210*b7*a1**2*a2**2*a3**2 + 105*b7*a1*a2**4*a3 + 30*b6*a1*a2*a3**3 + 7*b7*a2**6 + 20*b6*a2**3*a3**2 + b5*a3**4)
    c14= 105*b7*a0**2*a2*a3**4 + 105*b7*a0*a1**2*a3**4 + 420*b7*a0*a1*a2**2*a3**3 + 105*b7*a0*a2**4*a3**2 + 30*b6*a0*a2*a3**4 + 140*b7*a1**3*a2*a3**3 + 210*b7*a1**2*a2**3*a3**2 + 15*b6*a1**2*a3**4 + 42*b7*a1*a2**5*a3 + 60*b6*a1*a2**2*a3**3 + b7*a2**7 + 15*b6*a2**4*a3**2 + 5*b5*a2*a3**4
    c13= 105*b7*a0**2*a1*a3**4 + 210*b7*a0**2*a2**2*a3**3 + 420*b7*a0*a1**2*a2*a3**3 + 420*b7*a0*a1*a2**3*a3**2 + 30*b6*a0*a1*a3**4 + 42*b7*a0*a2**5*a3 + 60*b6*a0*a2**2*a3**3 + 35*b7*a1**4*a3**3 + 210*b7*a1**3*a2**2*a3**2 + 105*b7*a1**2*a2**4*a3 + 60*b6*a1**2*a2*a3**3 + 7*b7*a1*a2**6 + 60*b6*a1*a2**3*a3**2 + 5*b5*a1*a3**4 + 6*b6*a2**5*a3 + 10*b5*a2**2*a3**3
    c12= 35*b7*a0**3*a3**4 + 420*b7*a0**2*a1*a2*a3**3 + 210*b7*a0**2*a2**3*a3**2 + 15*b6*a0**2*a3**4 + 140*b7*a0*a1**3*a3**3 + 630*b7*a0*a1**2*a2**2*a3**2 + 210*b7*a0*a1*a2**4*a3 + 120*b6*a0*a1*a2*a3**3 + 7*b7*a0*a2**6 + 60*b6*a0*a2**3*a3**2 + 5*b5*a0*a3**4 + 105*b7*a1**4*a2*a3**2 + 140*b7*a1**3*a2**3*a3 + 20*b6*a1**3*a3**3 + 21*b7*a1**2*a2**5 + 90*b6*a1**2*a2**2*a3**2 + 30*b6*a1*a2**4*a3 + 20*b5*a1*a2*a3**3 + b6*a2**6 + 10*b5*a2**3*a3**2 + b4*a3**4
    c11= 140*b7*a0**3*a2*a3**3 + 210*b7*a0**2*a1**2*a3**3 + 630*b7*a0**2*a1*a2**2*a3**2 + 105*b7*a0**2*a2**4*a3 + 60*b6*a0**2*a2*a3**3 + 420*b7*a0*a1**3*a2*a3**2 + 420*b7*a0*a1**2*a2**3*a3 + 60*b6*a0*a1**2*a3**3 + 42*b7*a0*a1*a2**5 + 180*b6*a0*a1*a2**2*a3**2 + 30*b6*a0*a2**4*a3 + 20*b5*a0*a2*a3**3 + 21*b7*a1**5*a3**2 + 105*b7*a1**4*a2**2*a3 + 35*b7*a1**3*a2**4 + 60*b6*a1**3*a2*a3**2 + 60*b6*a1**2*a2**3*a3 + 10*b5*a1**2*a3**3 + 6*b6*a1*a2**5 + 30*b5*a1*a2**2*a3**2 + 5*b5*a2**4*a3 + 4*b4*a2*a3**3
    c10= 140*b7*a0**3*a1*a3**3 + 210*b7*a0**3*a2**2*a3**2 + 630*b7*a0**2*a1**2*a2*a3**2 + 420*b7*a0**2*a1*a2**3*a3 + 60*b6*a0**2*a1*a3**3 + 21*b7*a0**2*a2**5 + 90*b6*a0**2*a2**2*a3**2 + 105*b7*a0*a1**4*a3**2 + 420*b7*a0*a1**3*a2**2*a3 + 105*b7*a0*a1**2*a2**4 + 180*b6*a0*a1**2*a2*a3**2 + 120*b6*a0*a1*a2**3*a3 + 20*b5*a0*a1*a3**3 + 6*b6*a0*a2**5 + 30*b5*a0*a2**2*a3**2 + 42*b7*a1**5*a2*a3 + 35*b7*a1**4*a2**3 + 15*b6*a1**4*a3**2 + 60*b6*a1**3*a2**2*a3 + 15*b6*a1**2*a2**4 + 30*b5*a1**2*a2*a3**2 + 20*b5*a1*a2**3*a3 + 4*b4*a1*a3**3 + b5*a2**5 + 6*b4*a2**2*a3**2
    c9 = 35*b7*a0**4*a3**3 + 420*b7*a0**3*a1*a2*a3**2 + 140*b7*a0**3*a2**3*a3 + 20*b6*a0**3*a3**3 + 210*b7*a0**2*a1**3*a3**2 + 630*b7*a0**2*a1**2*a2**2*a3 + 105*b7*a0**2*a1*a2**4 + 180*b6*a0**2*a1*a2*a3**2 + 60*b6*a0**2*a2**3*a3 + 10*b5*a0**2*a3**3 + 210*b7*a0*a1**4*a2*a3 + 140*b7*a0*a1**3*a2**3 + 60*b6*a0*a1**3*a3**2 + 180*b6*a0*a1**2*a2**2*a3 + 30*b6*a0*a1*a2**4 + 60*b5*a0*a1*a2*a3**2 + 20*b5*a0*a2**3*a3 + 4*b4*a0*a3**3 + 7*b7*a1**6*a3 + 21*b7*a1**5*a2**2 + 30*b6*a1**4*a2*a3 + 20*b6*a1**3*a2**3 + 10*b5*a1**3*a3**2 + 30*b5*a1**2*a2**2*a3 + 5*b5*a1*a2**4 + 12*b4*a1*a2*a3**2 + 4*b4*a2**3*a3 + b3*a3**3
    c8 = 105*b7*a0**4*a2*a3**2 + 210*b7*a0**3*a1**2*a3**2 + 420*b7*a0**3*a1*a2**2*a3 + 35*b7*a0**3*a2**4 + 60*b6*a0**3*a2*a3**2 + 420*b7*a0**2*a1**3*a2*a3 + 210*b7*a0**2*a1**2*a2**3 + 90*b6*a0**2*a1**2*a3**2 + 180*b6*a0**2*a1*a2**2*a3 + 15*b6*a0**2*a2**4 + 30*b5*a0**2*a2*a3**2 + 42*b7*a0*a1**5*a3 + 105*b7*a0*a1**4*a2**2 + 120*b6*a0*a1**3*a2*a3 + 60*b6*a0*a1**2*a2**3 + 30*b5*a0*a1**2*a3**2 + 60*b5*a0*a1*a2**2*a3 + 5*b5*a0*a2**4 + 12*b4*a0*a2*a3**2 + 7*b7*a1**6*a2 + 6*b6*a1**5*a3 + 15*b6*a1**4*a2**2 + 20*b5*a1**3*a2*a3 + 10*b5*a1**2*a2**3 + 6*b4*a1**2*a3**2 + 12*b4*a1*a2**2*a3 + b4*a2**4 + 3*b3*a2*a3**2
    c7 = 105*b7*a0**4*a1*a3**2 + 105*b7*a0**4*a2**2*a3 + 420*b7*a0**3*a1**2*a2*a3 + 140*b7*a0**3*a1*a2**3 + 60*b6*a0**3*a1*a3**2 + 60*b6*a0**3*a2**2*a3 + 105*b7*a0**2*a1**4*a3 + 210*b7*a0**2*a1**3*a2**2 + 180*b6*a0**2*a1**2*a2*a3 + 60*b6*a0**2*a1*a2**3 + 30*b5*a0**2*a1*a3**2 + 30*b5*a0**2*a2**2*a3 + 42*b7*a0*a1**5*a2 + 30*b6*a0*a1**4*a3 + 60*b6*a0*a1**3*a2**2 + 60*b5*a0*a1**2*a2*a3 + 20*b5*a0*a1*a2**3 + 12*b4*a0*a1*a3**2 + 12*b4*a0*a2**2*a3 + b7*a1**7 + 6*b6*a1**5*a2 + 5*b5*a1**4*a3 + 10*b5*a1**3*a2**2 + 12*b4*a1**2*a2*a3 + 4*b4*a1*a2**3 + 3*b3*a1*a3**2 + 3*b3*a2**2*a3
    c6 = 21*b7*a0**5*a3**2 + 210*b7*a0**4*a1*a2*a3 + 35*b7*a0**4*a2**3 + 15*b6*a0**4*a3**2 + 140*b7*a0**3*a1**3*a3 + 210*b7*a0**3*a1**2*a2**2 + 120*b6*a0**3*a1*a2*a3 + 20*b6*a0**3*a2**3 + 10*b5*a0**3*a3**2 + 105*b7*a0**2*a1**4*a2 + 60*b6*a0**2*a1**3*a3 + 90*b6*a0**2*a1**2*a2**2 + 60*b5*a0**2*a1*a2*a3 + 10*b5*a0**2*a2**3 + 6*b4*a0**2*a3**2 + 7*b7*a0*a1**6 + 30*b6*a0*a1**4*a2 + 20*b5*a0*a1**3*a3 + 30*b5*a0*a1**2*a2**2 + 24*b4*a0*a1*a2*a3 + 4*b4*a0*a2**3 + 3*b3*a0*a3**2 + b6*a1**6 + 5*b5*a1**4*a2 + 4*b4*a1**3*a3 + 6*b4*a1**2*a2**2 + 6*b3*a1*a2*a3 + b3*a2**3 + b2*a3**2
    c5 = 42*a3*b7*a0**5*a2 + 105*a3*b7*a0**4*a1**2 + 105*b7*a0**4*a1*a2**2 + 30*a3*b6*a0**4*a2 + 140*b7*a0**3*a1**3*a2 + 60*a3*b6*a0**3*a1**2 + 60*b6*a0**3*a1*a2**2 + 20*a3*b5*a0**3*a2 + 21*b7*a0**2*a1**5 + 60*b6*a0**2*a1**3*a2 + 30*a3*b5*a0**2*a1**2 + 30*b5*a0**2*a1*a2**2 + 12*a3*b4*a0**2*a2 + 6*b6*a0*a1**5 + 20*b5*a0*a1**3*a2 + 12*a3*b4*a0*a1**2 + 12*b4*a0*a1*a2**2 + 6*a3*b3*a0*a2 + b5*a1**5 + 4*b4*a1**3*a2 + 3*a3*b3*a1**2 + 3*b3*a1*a2**2 + 2*a3*b2*a2
    c4 = 42*a3*b7*a0**5*a1 + 21*b7*a0**5*a2**2 + 105*b7*a0**4*a1**2*a2 + 30*a3*b6*a0**4*a1 + 15*b6*a0**4*a2**2 + 35*b7*a0**3*a1**4 + 60*b6*a0**3*a1**2*a2 + 20*a3*b5*a0**3*a1 + 10*b5*a0**3*a2**2 + 15*b6*a0**2*a1**4 + 30*b5*a0**2*a1**2*a2 + 12*a3*b4*a0**2*a1 + 6*b4*a0**2*a2**2 + 5*b5*a0*a1**4 + 12*b4*a0*a1**2*a2 + 6*a3*b3*a0*a1 + 3*b3*a0*a2**2 + b4*a1**4 + 3*b3*a1**2*a2 + 2*a3*b2*a1 + b2*a2**2
    c3 = 7*a3*b7*a0**6 + 42*a2*b7*a0**5*a1 + 6*a3*b6*a0**5 + 35*b7*a0**4*a1**3 + 30*a2*b6*a0**4*a1 + 5*a3*b5*a0**4 + 20*b6*a0**3*a1**3 + 20*a2*b5*a0**3*a1 + 4*a3*b4*a0**3 + 10*b5*a0**2*a1**3 + 12*a2*b4*a0**2*a1 + 3*a3*b3*a0**2 + 4*b4*a0*a1**3 + 6*a2*b3*a0*a1 + 2*a3*b2*a0 + b3*a1**3 + 2*a2*b2*a1 + a3*b1
    c2 = 7*a2*b7*a0**6 + 21*b7*a0**5*a1**2 + 6*a2*b6*a0**5 + 15*b6*a0**4*a1**2 + 5*a2*b5*a0**4 + 10*b5*a0**3*a1**2 + 4*a2*b4*a0**3 + 6*b4*a0**2*a1**2 + 3*a2*b3*a0**2 + 3*b3*a0*a1**2 + 2*a2*b2*a0 + b2*a1**2 + a2*b1
    c1 = a1*(7*b7*a0**6 + 6*b6*a0**5 + 5*b5*a0**4 + 4*b4*a0**3 + 3*b3*a0**2 + 2*b2*a0 + b1)
    c0 = b7*a0**7 + b6*a0**6 + b5*a0**5 + b4*a0**4 + b3*a0**3 + b2*a0**2 + b1*a0 + b0
    
    coefss=[c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c16,c17,c18,c19,c20,c21]
    
    return coefss


def curve_fitting(points,vcm_chosen,vcm,vid,polylength=8):
    
    vs = tf.linspace(vcm-vid, vcm+vid, polylength)-vcm_chosen
    A  = tf.concat(([tf.ones_like( vs)],[vs],[vs**2],[vs**3],[vs**4],[vs**5],[vs**6],[vs**7]),axis=0)
    A  = A[0:polylength]
    points2 = tf.reshape(points,(polylength,1))
    coefs2 = tf.transpose(tf.linalg.solve(tf.transpose(A),points2))    
    return coefs2[0]


def curve_fitting_np(points,vcm_chosen,vcm,vid,polylength=8):
    
    vs = np.linspace(vcm-vid, vcm+vid, polylength)-vcm_chosen
    A  = np.concatenate(([np.ones_like( vs)],[vs],[vs**2],[vs**3],[vs**4],[vs**5],[vs**6],[vs**7]),axis=0)
    A  = A[0:polylength]
    points2 = np.reshape(points,(polylength,1))
    coefs2 = np.transpose(np.linalg.solve(np.transpose(A),points2))
    return coefs2[0]


def linearity_coef(coef,vin):
    # vin here is th4 amplitude of the sinusoidal signal:
    
    frange = tf.math.maximum(+2.0*(coef[1]*vin+ coef[3]*vin**3+coef[5]*vin**5+coef[7]*vin**7+coef[9]*vin**9+coef[11]*vin**11+coef[13]*vin**13+coef[15]*vin**15+coef[17]*vin**17),1.0)
    A1  = coef[1]     + 3/4*coef[3]*vin**2+ 10/16*coef[5]*vin**4+ 35/64*coef[7]*vin**6+ 126/256*coef[9]*vin**8+  462/1024*coef[11]*vin**10
    A3  =             + 1/4*coef[3]*vin**2+  5/16*coef[5]*vin**4+ 21/64*coef[7]*vin**6+  84/256*coef[9]*vin**8+  330/1024*coef[11]*vin**10
    A5  =                                 +  1/16*coef[5]*vin**4+  7/64*coef[7]*vin**6+  36/256*coef[9]*vin**8+  165/1024*coef[11]*vin**10
    A7  =                                                       +  1/64*coef[7]*vin**6+   9/256*coef[9]*vin**8+   55/1024*coef[11]*vin**10
    sfdr3  =20*tf.math.log(tf.abs(A1/A3))/2.3
    sfdr5  =20*tf.math.log(tf.abs(A1/A5))/2.3
    sfdr7  =20*tf.math.log(tf.abs(A1/A7))/2.3
    return frange, sfdr3, sfdr5, sfdr7    

def linearity_coef_np(coef,vin):
    frange = max(+2.0*(coef[1]*vin+ coef[3]*vin**3+coef[5]*vin**5+coef[7]*vin**7+coef[9]*vin**9+coef[11]*vin**11+coef[13]*vin**13+coef[15]*vin**15+coef[17]*vin**17),1.0)
    A1  = coef[1]     + 3/4*coef[3]*vin**2+ 10/16*coef[5]*vin**4+ 35/64*coef[7]*vin**6+ 126/256*coef[9]*vin**8+  462/1024*coef[11]*vin**10
    A3  =             + 1/4*coef[3]*vin**2+  5/16*coef[5]*vin**4+ 21/64*coef[7]*vin**6+  84/256*coef[9]*vin**8+  330/1024*coef[11]*vin**10
    A5  =                                 +  1/16*coef[5]*vin**4+  7/64*coef[7]*vin**6+  36/256*coef[9]*vin**8+  165/1024*coef[11]*vin**10
    A7  =                                                       +  1/64*coef[7]*vin**6+   9/256*coef[9]*vin**8+   55/1024*coef[11]*vin**10
    sfdr3  =20*np.log(np.abs(A1/A3))/2.3
    sfdr5  =20*np.log(np.abs(A1/A5))/2.3
    sfdr7  =20*np.log(np.abs(A1/A7))/2.3
    return frange, sfdr3, sfdr5, sfdr7 


def param_to_sxin(param,vco,inbuf):
    x_vco1   = param[0][0]
    x_inbuf1 = param[1][0]
    osr   =    param[2] 
    amp   =    param[3]

    sx_more1 = np_sigmoid_inv((osr-minosr)/(maxosr-minosr))
    sx_more2 = np_sigmoid_inv((amp-minamp)/(maxamp-minamp))
    
    sx_vco1 = vco.np_scalex(x_vco1)
    sx_inbuf1  = inbuf.np_scalex(x_inbuf1) 
    sxout =  np.array([list(sx_vco1[0:4])+list(sx_inbuf1[0:6])+[sx_more1]+[sx_more2]])
    
    return sxout
    

def step2sxin(vco,inbuf):
    ss_vco1 = vco.step*vco.scXscale
    ss_inbuf1 = inbuf.step*inbuf.scXscale
    sx_list = np.array([list(ss_vco1[0][0:4])+list(ss_inbuf1[0][0:6])+[0.01,0.001]])
    return sx_list





    
#==================================================================
#*****************  Building the graph  ***************************
#==================================================================
def graph_tf2(sxin,vco1,inbuf2):

    #----------vco's graph----------
    #['lastt','wnnn','fnnn','wpppn','fppp', 'rres','VBIAS','VDD']
    #['power','vcm','vfs','fnoise','f1','f2','f3','f4','f5','f6','f7','f8']
    sx_vco1 = sxin[:,0:4]
    x_vco1=vco1.tf_rescalex(sx_vco1)
    sy_vco1=vco1.tf_reg_elu(sx_vco1)
    y_vco1=vco1.tf_rescaley(sy_vco1)
    
    #----------inbuf's graph----------    
    #['multi','fing_in','l_ttt','fing_ttt','VCM','dvv','wpppp','fpppp']
    #['power','gain','bw','outvcm', 'avcm','kickn','noise','outn3','outn1','outp1','outp3']
    #     0       1   2      3       4        5     6        7       8       9       10 
    chosen=np.array([1,1,1,1,1,0,0])
    vars_inbuf2 = sxin[:,4:9] 
    cnst_inbuf2=inbuf2.tf_scalex2(x_vco1[0,2:4], 1-chosen)
    sx_inbuf2=tf.reshape(tf.concat([vars_inbuf2,[cnst_inbuf2]],axis=1),[1,7])
    x_inbuf2=inbuf2.tf_rescalex(sx_inbuf2)
    sy_inbuf2=inbuf2.tf_reg_sigmoid(sx_inbuf2)    
    y_inbuf2=inbuf2.tf_rescaley(sy_inbuf2)
    
    #--------Other variables--------
    morep= sxin[:,9:11]
    osr = tf.nn.sigmoid(morep[0][0])*(maxosr-minosr)+minosr
    ampin = minamp + tf.nn.sigmoid(morep[0][1])*(maxamp-minamp)
    
    # 
    fs = 2*bw*osr                                      # Sampling frequency
    vcm   = y_inbuf2[0,3]
    vcmin = x_inbuf2[0,4]
    gain = 10.0**(y_inbuf2[0,1]/20)
    coefs_inbuf2 = curve_fitting(y_inbuf2[0][7:11]-vcm,0,       0,         0.3,polylength=4)
    
    
    
    coefs_vco    = curve_fitting(  y_vco1[0][4:12],vcm  ,y_vco1[0][1],y_vco1[0][2],polylength=8)
    
    
    total_coefs = multi_fitting(coefs_inbuf2,coefs_vco)
    frange, sfdr3, sfdr5, sfdr7  = linearity_coef(total_coefs,ampin)

    amp  = gain*ampin
    
    drvnoise = (y_inbuf2[0,6]*frange/2/amp)**2.0
    vconoise = (y_vco1[0][3])**2.0
    mmmbit = tf.math.log(frange*16/fs)/tf.math.log(2.0)
    totalsnr = (frange**2.0)*osr/(8*(vconoise+ drvnoise))

#    init=tf.compat.v1.global_variables_initializer()
#    with tf.compat.v1.Session() as sess:
#        sess.run(init)
#        print(sess.run(frange))
#        print(sess.run(coefs_vco))    
#        print(sess.run(coefs_inbuf2))  
#        print(sess.run(totalsnr))  
        
    #==================================================================
    #***************  Define constraints and Cost(P)  *****************
    #==================================================================
    
    specs = []    
    specs.append(6.0*mmmbit - 3.41 + 30.0*tf.math.log(osr)/tf.math.log(10.0))                                       # 0- SQNR LAW
    specs.append( tf.minimum(sfdr3,tf.minimum(sfdr5,sfdr7)))                                                        # 1- SFDR 
    specs.append(10.0*tf.math.log(totalsnr)/2.3)                                                                    # 2- SNR
    specs.append(y_inbuf2[0,2] - bw)                                                                                # 3- Bandwidth
    specs.append(y_inbuf2[0,5])                                                                                     # 4- Kickback
    specs.append(y_inbuf2[0,4])                                                                                     # 5- VCM gain
    specs.append(-(vcm+amp)  +y_vco1[0][1]+y_vco1[0][2])                                                            # 6- vcm + amp < vco(vcm) + vco(amp)  +
    specs.append(  vcm-amp   -y_vco1[0][1]+y_vco1[0][2])                                                            # 7- vcm - amp > vco(vcm) - vco(amp)  +
    specs.append(y_vco1[0][2]-amp)                                                                                  # 8- vco(amp) - amp > 0
    specs.append(2*y_vco1[0,0]+y_inbuf2[0,0]+fs*2e-12)                                                              #-1- Power consumption
    
    constraints = []    
    constraints.append(tf.nn.elu((nbit     -specs[0]/6)*weights[0]))
    constraints.append(tf.nn.elu((nbit+10/6-specs[1]/6)*weights[1]))
    constraints.append(tf.nn.elu((nbit+10/6-specs[2]/6)*weights[2]))
    constraints.append(tf.nn.elu((         -specs[3]/inbuf2.scYscale[2])*weights[3]))
    constraints.append(tf.nn.elu((nbit/2   -specs[4]/6/inbuf2.scYscale[6])*weights[4]))
    constraints.append(tf.nn.elu((nbit/2   -specs[5]/6/inbuf2.scYscale[5])*weights[5]))
    constraints.append(tf.nn.elu((         -specs[6]   )*weights[6]))    
    constraints.append(tf.nn.elu((         -specs[7]   )*weights[7]))   
    constraints.append(tf.nn.elu((         -specs[8]   )*weights[8]))    
    constraints.append(tf.nn.elu((specs[-1]/vco1.scYscale[0]*1)*weights[-1]))    
 
    
    
    hardcost=tf.reduce_sum(constraints)
    usercost=tf.reduce_sum(constraints[6:-1])
    
    return hardcost,usercost,specs,[x_vco1,x_inbuf2,osr,ampin],[y_vco1,y_inbuf2,amp,fs],[frange,amp,vcm,vcmin,coefs_inbuf2,coefs_vco,total_coefs],constraints

#==================================================================
#*****************  Building the graph  ***************************
#==================================================================
def graph_spice2(sxin,vco1,inbuf2,vcospice,inbufspice):
    
    #----------vco's graph---------------
    sx_vco1 = sxin[:,0:4]
    x_vco1=vco1.np_rescalex(sx_vco1)
    x_vco1,d_vco1 = vcospice.wholerun_std(np.array([60e-9] + list(x_vco1[0]) + [2000 ,0.9  ,1.0 ]))
    y_vco1=np.array([d_vco1])
    
    #---------In Buffer's graph----------
    chosen=np.array([1,1,1,1,1,0,0])
    vars_inbuf2 = sxin[:,4:9] 
    cnst_inbuf2=inbuf2.np_scalex2(x_vco1[3:5], 1-chosen)
    sx_inbuf2=np.reshape(np.concatenate([vars_inbuf2,[cnst_inbuf2]],axis=1),[1,7])
    x_inbuf2=inbuf2.np_rescalex(sx_inbuf2)
    
    x_inbuf2, d_inbuf2=inbufspice.wholerun_std(np.array(list(x_inbuf2[0,0:5])+[0]+list(x_inbuf2[0,5:7])))    
    y_inbuf2 = np.array([d_inbuf2])

    
    #--------Other variables--------
    morep= sxin[:,9:11]
    osr = np.round(np_sigmoid(morep[0][0])*(maxosr-minosr)+minosr)
    ampin = minamp + np_sigmoid(morep[0][1])*(maxamp-minamp)
    
    # 
    fs = 2*bw*osr                                      # Sampling frequency
    vcm   = y_inbuf2[0,3]
    vcmin = x_inbuf2[4]
    gain = 10.0**(y_inbuf2[0,1]/20)
    coefs_inbuf2 = curve_fitting_np(y_inbuf2[0][7:11]-vcm,0,       0,         0.3,polylength=4)
    
    coefs_vco    = curve_fitting_np(  y_vco1[0][4:12],vcm  ,y_vco1[0][1],y_vco1[0][2],polylength=8)
    
    
    total_coefs = multi_fitting(coefs_inbuf2,coefs_vco)
    frange, sfdr3, sfdr5, sfdr7  = linearity_coef_np(total_coefs,ampin)

    amp  = gain*ampin
    
    drvnoise = (y_inbuf2[0,6]*frange/2/amp)**2.0
    vconoise = (y_vco1[0][3])**2.0
    mmmbit = np.log(frange*16/fs)/np.log(2.0)
    totalsnr = (frange**2.0)*osr/(8*(vconoise+ drvnoise))
    
    #==================================================================
    #***************  Define constraints and Cost(P)  *****************
    #==================================================================
    
    specs = []    
    specs.append(6.0*mmmbit - 3.41 + 30.0*np.log(osr)/np.log(10.0))                                                 # 0- SQNR LAW
    specs.append( min(sfdr3,sfdr5,sfdr7))                                                                           # 1- SFDR 
    specs.append(10.0*np.log(totalsnr)/2.3)                                                                         # 2- SNR
    specs.append(y_inbuf2[0,2] - bw)                                                                                # 3- Bandwidth
    specs.append(y_inbuf2[0,5])                                                                                     # 4- Kickback
    specs.append(y_inbuf2[0,4])                                                                                     # 5- VCM gain
    specs.append(-(vcm+amp)  +y_vco1[0][1]+y_vco1[0][2])                                                            # 6- vcm + amp < vco(vcm) + vco(amp)  +
    specs.append(  vcm-amp   -y_vco1[0][1]+y_vco1[0][2])                                                            # 7- vcm - amp > vco(vcm) - vco(amp)  +
    specs.append(y_vco1[0][2]-amp)                                                                                  # 8- vco(amp) - amp > 0
    specs.append(2*y_vco1[0,0]+y_inbuf2[0,0]+fs*2e-12)                                                              #-1- Power consumption



    constraints = []    
    constraints.append(np_elu((nbit     -specs[0]/6)*weights[0]))
    constraints.append(np_elu((nbit+10/6-specs[1]/6)*weights[1]))
    constraints.append(np_elu((nbit+10/6-specs[2]/6)*weights[2]))
    constraints.append(np_elu((         -specs[3]/inbuf2.scYscale[2])*weights[3]))
    constraints.append(np_elu((nbit/2   -specs[4]/6/inbuf2.scYscale[6])*weights[4]))
    constraints.append(np_elu((nbit/2   -specs[5]/6/inbuf2.scYscale[5])*weights[5]))
    constraints.append(np_elu((         -specs[6]   )*weights[6]))    
    constraints.append(np_elu((         -specs[7]   )*weights[7]))   
    constraints.append(np_elu((         -specs[8]   )*weights[8]))    
    constraints.append(np_elu((specs[-1]/vco1.scYscale[0]*1)*weights[-1])) 

    hardcost = sum(constraints)
    usercost = sum(constraints[6:-1])
    
    
    return hardcost,usercost,specs,[x_vco1,x_inbuf2,osr,ampin],[y_vco1,y_inbuf2,amp,fs],[frange,amp,vcm,vcmin,coefs_inbuf2,coefs_vco,total_coefs],constraints
    
#==================================================================
#*********************  Main code  ********************************
#==================================================================

if __name__ == '__main__':
    

    #==================================================================
    #*****************  Building the graph  ***************************
    #==================================================================
    
    tf.compat.v1.disable_eager_execution()
    #----------Initialize----------
    tf.compat.v1.reset_default_graph()
    optimizer1 = tf.compat.v1.train.AdamOptimizer(0.01)
    optimizer2 = tf.compat.v1.train.AdamOptimizer(0.001)
    
    

    #----------load regressors----------
    vco1 = VCO(tech=65)
    inbuf2 = INBUF2(tech=65)    
    #--------load spice netlists--------
    vcospice1   = VCOSpice()
    inbufspice2 = INBUF2Spice()
    
    
    var_in = make_var("VCO_ADC", "BUF_VCO", (1,12), tf.random_uniform_initializer(-np.ones((1,12)),np.ones((1,12))))
    
    hardcost,usercost,tf_specs,tf_params,tf_metrics,tf_mids,tf_const = graph_tf2(var_in,vco1,inbuf2)
    
    
    

    #==================================================================
    #********************  Tensorflow Initiation  *********************
    #==================================================================    
        
    opt1=optimizer1.minimize(usercost)
    opt2=optimizer2.minimize(hardcost)
    init=tf.compat.v1.global_variables_initializer()
    
    calc=1

    
    lastvalue=-1000000
    lst_params=[]
    lst_metrics=[]
    lst_specs=[]
    lst_value=[]
    lst_midvalues=[]
    tstart=time.time()
    
    k=0
    doogh=0
    for j in range(tedad):
        reg_specs=[]
        const=[]
        
        lst_amps=[]
        lst_vcms=[]
        lst_coefs=[]
        lst_mets=[]
        with tf.compat.v1.Session() as sess:
            sess.run(init)
            sess.run(initvar,feed_dict={xload:y[j:(j+1),:]})
            doogh=0        
            #==================================================================
            #*****************  Tensorflow Gradient Descent  ******************
            #==================================================================
            
            i=0
            while i <maxiter:

                
                try:
                    if i<maxiter/2:
                        _,value,smallspecs = sess.run([opt1,usercost,tf_specs])                
                    else:
                        _,value,smallspecs = sess.run([opt2,hardcost,tf_specs])   
                    
                    doogh+=1
                    k+=1
                except:
                    print('Terminated due to error!')
                    break
                print('%1.0f:, %1.0f : %1.3f \n'%(j, i, value))
                smallspecs.append(doogh)
                reg_specs.append(smallspecs)
                if math.isnan(value) or math.isinf(value):
                    break
                else:
                    smallspecs = sess.run(tf_specs)
                    parameters = sess.run(tf_params)
                    metrics    = sess.run(tf_metrics)
                    midvalues = sess.run(tf_mids)
                    const.append(sess.run(tf_const))
                    np_sxin = sess.run(var_in)
                    frange,amp,vcm,vcmin,coefs_inbuf2,coefs_vco,total_coefs= midvalues
                    lst_amps.append(amp)
                    lst_vcms.append(vcm)
#                    lst_coefs.append(coefs_vco)
                    lst_mets.append(metrics[0])
                if i<maxiter/2 and np.abs(lastvalue-value)<0.001:
                    i=int(maxiter/2)+1
                    print(i)
                elif i>maxiter/2 and np.abs(lastvalue-value)<epsilon:
                    break
                else:
                    lastvalue=value
                i+=1
                
                
#            if not math.isnan(value):
                

            #==================================================================
            #**********************  Saving the values  ***********************
            #==================================================================
            tend=time.time()
            

            print('the elapsed time %1.2f S\n' %(tend-tstart))
            
        var_specs=np.array(reg_specs)
        lst_params.append(parameters)
        lst_metrics.append(metrics)
        lst_specs.append(reg_specs[-1])
        lst_value.append(value)
        lst_midvalues.append(midvalues)
        np_specs = np.array(reg_specs)
        
        np_specs_gd = np.array(lst_specs)
        mydict= {'lst_params':lst_params,'lst_metrics':lst_metrics,'lst_specs':lst_specs,'lst_value':lst_value,'lst_amps':lst_amps,'lst_vcms':lst_vcms}
        dump( mydict, open( 'regsearch_results1_'+str(nbit)+str(bw/1e6)+'.p', "wb" ) )
        savemat('regsearch_constraints.mat',{'np_specs':np_specs})        




