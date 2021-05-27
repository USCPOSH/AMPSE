

from tensorflow_circuit import TF_DEFAULT
import os
home_address  = os.getcwd()
import numpy as np
import tensorflow as tf

#==================================================================
#****************  Loading the Regressors  ************************
#==================================================================


# Comparator initialization:
class COMPP2(TF_DEFAULT):
    # parameters: 'fck1','fck2','fcompinv','finnn','fninv','fpinv','fpr1','fpr2','frdynand','fttt','fnor3','frdy','mrdy','fload','mload'
    # metrics :'power','readyp','delayr','delayf','kickn','cin','scin','irn'
    def __init__(self,tech=65):

        self.tech=tech
        self.default_loading()
        
    def default_loading(self):
        if self.tech==65:
            drive       = home_address+'/Reg_files/PY_COMPPin6503_TT'
            sx_f        = drive + '/scX_compp65.pkl'
            sy_f        = drive + '/scY_compp65.pkl'    
            w_f         = drive + '/w8_compp65.p'
            self.w_json = drive + '/model_compp65.json'
            self.w_h5   = drive + '/reg_compp65.h5'

            self.minx = np.array([[1    , 1    , 1        , 1     , 1     , 1     , 1    , 1    , 1        , 2    ,   1   ,   1  ,  4   ,  1    ,  4    ]])
            self.maxx = np.array([[10   , 20   , 40       , 20    , 40    , 40    , 8    , 8    , 8        , 80   ,   10  ,   48 ,  12  ,  16   ,  12   ]])
            self.step = np.array([[1    , 1    , 1        , 1     , 1     , 1     , 1    , 1    , 1        , 2    ,   1   ,   1  ,  1   ,  1    ,  1    ]])
            
            
        self.loading(sx_f,sy_f,w_f)
        

# MS. TH        
class THDAC2(TF_DEFAULT):
    # parameters ['caps','fppp','wppp','swfn','swnn', 'swpp']
    
    # metrics : [cinn, cinp, clkfeed, ctot, kicknoise, trackbwMIN]
    
    def __init__(self,tech=65):

        self.tech=tech
        self.default_loading()
        
        
    def default_loading(self):
        if self.tech==65:
            drive       = home_address+'/Reg_files/PY_THDAC6502_TT'
            sx_f        = drive + '/scX_th65.pkl'
            sy_f        = drive + '/scY_th65.pkl'    
            w_f         = drive + '/w8_th65.p'
            self.w_json = drive + '/model_th65.json'
            self.w_h5   = drive + '/reg_th65.h5'
            self.minx = np.array([[2    ,4      ,0.5e-15, 2     , 2     ,  2e-15 ]])
            self.maxx = np.array([[16   ,12     ,5.0e-15, 40    , 60    ,  3e-14 ]]) 
            self.step = np.array([[2    ,1      ,0.5e-15, 2     , 2     ,0.5e-15 ]])    
        self.loading(sx_f,sy_f,w_f)
        
        
class THDAC(TF_DEFAULT):
    # parameters ['caps','fppp','wppp','swfn','swnn', 'swpp']
    
    # metrics : [cinn, cinp, clkfeed, ctot, kicknoise, trackbwMIN]
    
    def __init__(self,tech=65):

        self.tech=tech
        self.default_loading()
        
        
    def default_loading(self):
        if self.tech==65:
            drive       = home_address+'/Reg_files/PY_THDAC6501_TT'
            sx_f        = drive + '/scX_th65.pkl'
            sy_f        = drive + '/scY_th65.pkl'    
            w_f         = drive + '/w8_th65.p'
            self.w_json = drive + '/model_th65.json'
            self.w_h5   = drive + '/reg_th65.h5'
            self.minx = np.array([[3      ,0.5e-15, 2     , 2     , 1     , 1     , 2e-15 ]])
            self.maxx = np.array([[11     ,5.0e-15, 40    , 40    , 10    , 10    , 3e-14 ]]) 
            self.step = np.array([[1      ,0.5e-15, 2     , 2     , 1     , 1     , 2e-15 ]])    
        self.loading(sx_f,sy_f,w_f)
        
# Baby DRV
class SEQ1(TF_DEFAULT):
    def __init__(self,tech=65):
        self.tech=tech
        self.default_loading()
   
    def default_loading(self):
        if self.tech==65:
            drive       = home_address+'/Reg_files/PY_SEQ16501_TT'
            sx_f        = drive + '/scX_seqp165.pkl'
            sy_f        = drive + '/scY_seqp165.pkl'    
            w_f         = drive + '/w8_seqp165.p'
            self.w_json = drive + '/model_seqp165.json'
            self.w_h5   = drive + '/reg_seqp165.h5'
#            self.inv_spectre = INVSpice()
#                                 [ 'drvbuf11','drvbuf21','drvbuf31','drvdffck1','drvinv1','drvnand1','fdffck','finv','fnand','frefnn','frefpp','div','mdacbig']
            self.minx = np.array([[          1,         2,         2,          1,        2,         2,       1,     1,      1,       1,       1,    2,       3 ]])
            self.maxx = np.array([[         12,        24,        96,         10,       16,        24,      16,    16,     16,       2,       2,   16,      11 ]])    
            self.step = np.array([[          1,         2,         2,          1,        2,         2,       1,     1,      1,       1,       1,    2,       1 ]])
            
        self.loading(sx_f,sy_f,w_f)

class SEQ2(TF_DEFAULT):
    def __init__(self,tech=65):
        self.tech=tech
        self.default_loading()
        

    
    def default_loading(self):
        if self.tech==65:
            drive       = home_address+'/Reg_files/PY_SEQ26501_TT'
            sx_f        = drive + '/scX_seqp265.pkl'
            sy_f        = drive + '/scY_seqp265.pkl'    
            w_f         = drive + '/w8_seqp265.p'
            self.w_json = drive + '/model_seqp265.json'
            self.w_h5   = drive + '/reg_seqp265.h5'
#            self.inv_spectre = INVSpice()
            
            self.minx = np.array([[      1,     1]])
            self.maxx = np.array([[     10,    12]])    
            self.step = np.array([[      1,     1]])    
        self.loading(sx_f,sy_f,w_f)

def tf_quant_with_sigmoid(sxin,num=10):
    v = np.linspace(0.5/num,1-0.5/num,num)
    out=[]
    for vv in v:
        out.append(tf.nn.sigmoid(100.0*(sxin-vv)))
    
    return tf.reduce_sum(out,axis=0)+1.0

def param_to_sxin(param,seqp1,seqp2,compp,thdac,n_stairs):
    
    x_seqp11 = np.squeeze(param[0])
    x_seqp21 = np.squeeze(param[1])
    x_compp1 = np.squeeze(param[2])
    x_th1    = np.squeeze(param[3])

    x_ndly   = [(param[4]-1)/n_stairs]
    x_dtr    = [2*param[5]-1]
    
    sx_seqp11 = seqp1.np_scalex(x_seqp11)
    sx_seqp21 = seqp2.np_scalex(x_seqp21)
    sx_compp1 = compp.np_scalex(x_compp1) 
    sx_th1    = thdac.np_scalex(x_th1   )
    
    cx_seqp11 = list(sx_seqp11[[0,1,2,3,4,5,6,7,8,11]])
    cx_seqp21 = list(sx_seqp21)
    cx_compp1 = list(sx_compp1[1:10])
    cx_th1    = list(sx_th1[2:5])
    
    sx_out =  np.array([cx_seqp11+cx_seqp21+cx_compp1+cx_th1+ x_ndly+x_dtr])
    
    return sx_out

def step2sxin(seqp1,seqp2,compp,thdac,n_stairs):
    ss_seqp1 = seqp1.step*seqp1.scXscale
    ss_seqp2 = seqp2.step*seqp2.scXscale
    ss_compp = compp.step*compp.scXscale
    ss_thdac = thdac.step*thdac.scXscale

    sx_out =  np.array([list(ss_seqp1[0][[0,1,2,3,4,5,6,7,8,11]])+list(ss_seqp2[0])+list(ss_compp[0][1:10])+list(ss_thdac[0][1:4])+[1/n_stairs,0.01]])
    return sx_out


