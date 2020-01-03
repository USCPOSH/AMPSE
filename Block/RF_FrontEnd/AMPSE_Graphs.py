# Mostafa Ayesh
# Design RF Front-End with TF.

#==================================================================
#*****************  Loading the libraries  ************************
#==================================================================

import sys
#sys.path.insert(0,'/shares/MLLibs/GlobalLibrary')
sys.path.insert(0,'/home/mohsen/PYTHON_PHD/GlobalLibrary')

import os
home_address  = os.getcwd()
#from tensorflow_circuit import TF_DEFAULT, np_elu, np_sigmoid, np_sigmoid_inv, make_var
from tensorflow_circuit import TF_DEFAULT, make_var

import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
import time
import math
import tensorflow as tf

from scipy.io import savemat
#from pickle import dump

#==================================================================
#*******************  Initialization  *****************************
#==================================================================

tedad=10                    # number of parameter candidates 
epsilon=1e-5                # Epsilon in GD
maxiter=5000

#==================================================================
#****************  Loading the Regressors  ************************
#==================================================================

# MR. LNA
class LNA(TF_DEFAULT):
    def __init__(self,tech=28):

        self.tech=tech
        self.default_loading()
        
    def default_loading(self):
        if self.tech==28:
            drive       = home_address+'/reg_files/LNA'
            sx_f        = drive + '/scX_lna28.pkl'
            sy_f        = drive + '/scY_lna28.pkl'    
            w_f         = drive + '/w8_lna28.p'
            self.w_json = drive + '/model_lna28.json'
            self.w_h5   = drive + '/reg_lna28.h5'
            self.minx  = np.array([[25   ,10e-6  , 5e-15  ,10 ]])
            self.maxx  = np.array([[35   ,40e-6  , 50e-15 ,20 ]])
            self.step  = np.array([[5    ,5e-6   , 5e-15  ,2  ]])
            self.loading(sx_f,sy_f,w_f)

# MR. SH
class SH(TF_DEFAULT):
    def __init__(self,tech=28):

        self.tech=tech
        self.default_loading()
        
    def default_loading(self):
        if self.tech==28:
            drive       = home_address+'/reg_files/SH'
            sx_f        = drive + '/scX_sh28.pkl'
            sy_f        = drive + '/scY_sh28.pkl'    
            w_f         = drive + '/w8_sh28.p'
            self.w_json = drive + '/model_sh28.json'
            self.w_h5   = drive + '/reg_sh28.h5'
            self.minx = np.array([[2  ,  2e-15   , 25.3e9 , 50  , 30 ]])
            self.maxx = np.array([[8  ,  10e-15  , 26e9   , 100 , 80 ]])
            self.step = np.array([[2  ,  2e-15   , 2e6    , 10  , 5 ]])
            self.loading(sx_f,sy_f,w_f)        


#==================================================================
#*****************  Building the graph  ***************************
#==================================================================
def graph_tf(sxin,lna1,sh1):

    #----------LNA's graph----------
    #['Wpmos_in','L_Radius','Cin_OD','Wnmos_follower'] Parameters
    #     0          1         2             3                    
    #['center_freq','Rout','Power','Gain','sfdr','snr','enob'] Metrics
  
    sx_lna1 = sxin[:,0:4]
    x_lna1=lna1.tf_rescalex(sx_lna1)
    sy_lna1=lna1.tf_reg_elu(sx_lna1)
    y_lna1=lna1.tf_rescaley(sy_lna1)
    
    # x_lna1[2] should go into SH
    # y_lna1[0] should go into SH
    # y_lna1[1] should go into SH
    
    #----------SH's graph---------- 
    #['fin_main_sw','fin_sampling_Cap','frf','fin_battery_cap','Rout_LNA'] Parameters
    #     4                5             6         7               8
    #['Current Consumption','Output PP Amplitude','Tracking Bandwidth','ENOB','SNR','SFDR'] Metrics 

    chosen=np.array([1,0,0,1,0])
    vars_sh1 = sxin[:,4:9] # from the SH itself 
    
#    print(x_lna1[0,2])
#    print(y_lna1[0,0])
#    print(y_lna1[0,1])
    cnst_sh1=sh1.tf_scalex2(tf.stack([x_lna1[0,2],y_lna1[0,0],y_lna1[0,1]],axis=0), 1-chosen) # from the LNA
    
#    print(cnst_sh1)
#    print(vars_sh1)
    
    sx_sh1=tf.reshape(tf.concat((vars_sh1[0,0:1] , cnst_sh1[0:1] , cnst_sh1[1:2] , vars_sh1[0,3:4] , cnst_sh1[2:3]),axis=0),[1,5])
    x_sh1=sh1.tf_rescalex(sx_sh1)
    sy_sh1=sh1.tf_reg_sigmoid(sx_sh1)    
    y_sh1=sh1.tf_rescaley(sy_sh1)
    
    #==================================================================
    #***************  Define constraints and Cost(P)  *****************
    #==================================================================
    #----------LNA----------
    #['Wpmos_in','L_Radius','Cin_OD','Wnmos_follower'] Parameters              
    #['center_freq','Rout','Power','Gain','sfdr','snr','enob'] Metrics 
    
    #----------SH---------- 
    #['fin_main_sw','fin_sampling_Cap','frf','fin_battery_cap','Rout_LNA'] Parameters
    #['Current Consumption','Output PP Amplitude','Tracking Bandwidth','ENOB','SNR','SFDR'] Metrics   

    center_freq    = y_lna1[0,0]; 
    LNA_Gain       = y_lna1[0,3]; 
    LNA_SFDR       = y_lna1[0,4]; 
    SH_Tracking_BW = y_sh1[0,2]; 
    SH_ENOB        = y_sh1[0,3]; 
    SH_SFDR        = y_sh1[0,5]; 
    Total_Power    = y_lna1[0,2] * 1e-3 + y_sh1[0,0];    
    
    specs = []                                  
    specs.append(center_freq)                     #   0-Center Frequency
    specs.append(LNA_Gain)                        #   1-LNA Gain
    specs.append(LNA_SFDR)                        #   2-LNA SFDR    
    specs.append(SH_Tracking_BW)                  #   3-SH Tracking BW
    specs.append(SH_ENOB)                         #   4-SH ENOB
    specs.append(SH_SFDR)                         #   5-SH SFDR
    specs.append(Total_Power)                     #   6-Total Power

    constraints = []    
    constraints.append(tf.nn.elu(  -1* (specs[0] -  23e9   )*10. ))     # Center Frequency constaint
    constraints.append(tf.nn.elu(  -1* (26e9    - specs[0] )*10. ))     # Center Frequency constaint
    constraints.append(tf.nn.elu(  -1* (specs[3] - 25)))                # Maximize the SH Tracking BW  
    constraints.append(tf.nn.elu(  -1* (specs[4]) ))                    # Maximize the SH ENOB
    constraints.append(tf.nn.elu(  -1* (specs[5]) ))                    # Maximize the SH SFDR
    constraints.append(tf.nn.elu(      (specs[6]) ))                    # Minimize the Total power consumption    
  
    
    hardcost=tf.reduce_sum(constraints)
    usercost=tf.reduce_sum(constraints[2:-1]) 
    
#                            specs   parameters     metrics
    return hardcost,usercost,specs,[x_lna1,x_sh1],[y_lna1,y_sh1],constraints


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
    lna1 = LNA(tech=28)
    sh1  = SH(tech=28)
 
    sxin = make_var("RF_FE", "LNA", (1,9), tf.random_uniform_initializer(-np.ones((1,9)),np.ones((1,9))))
    hardcost,usercost,tf_specs,tf_params,tf_metrics,tf_const = graph_tf(sxin,lna1,sh1)
    
    #==================================================================
    #********************  Tensorflow Initiation  *********************
    #==================================================================    
        
    opt1 = optimizer1.minimize(usercost)
    opt2 = optimizer2.minimize(hardcost)
    init = tf.compat.v1.global_variables_initializer()
    
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
                    const.append(sess.run(tf_const))
                    np_sxin = sess.run(sxin)
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

            #==================================================================
            #**********************  Saving the values  ***********************
            #==================================================================
            tend=time.time()
            

#            print('user1: %1.2f, user2: %1.2f, user3: %1.2f, user4: %1.2f, user5: %1.2f\n' %(sess.run(user1),sess.run(user2),sess.run(user3),sess.run(user4),sess.run(user5)))
            print('the elapsed time %1.2f S\n' %(tend-tstart))
            
        var_specs=np.array(reg_specs)
        lst_params.append(parameters)
        lst_metrics.append(metrics)
        lst_specs.append(reg_specs[-1])
        lst_value.append(value)
        np_specs = np.array(reg_specs)
        
        np_specs_gd = np.array(lst_specs)
#        mydict= {'lst_params':lst_params,'lst_metrics':lst_metrics,'lst_specs':lst_specs,'lst_value':lst_value}
#        dump( mydict, open( 'regsearch_results1_'+str(nbit)+str(bw/1e6)+'.p', "wb" ) )
#        savemat('regsearch_constraints.mat',{'np_specs':np_specs})        
#        sp_value,_,sp_specs,sp_params,sp_metrics,sp_mids,sp_const = graph_spice2(np_sxin,lna1,sh1,vcospice1,inbufspice2)
        savemat('Spec_opt_adam.mat',{'gd':np_specs_gd})

    """
    frange,amp,vcm,vcmin,coefs_sh1,coefs_vco,total_coefs= sp_mids
    SFDRin=sp_params[3]
    y2 = sp_metrics[1][0,7:]-vcm
    yy = np.linspace(sp_metrics[0][0,1]-sp_metrics[0][0,2],sp_metrics[0][0,1]+sp_metrics[0][0,2],8)-vcm
    zz = sp_metrics[0][0,4:]
    yy2 = np.linspace(sp_metrics[0][0,1]-sp_metrics[0][0,2],sp_metrics[0][0,1]+sp_metrics[0][0,2],1000)-vcm
    
    frange,amp,vcm,vcmin,coefs_sh1,coefs_vco,total_coefs= midvalues
    SFDRin=parameters[3]
    y2=metrics[1][0,7:]-vcm
    yy = np.linspace(metrics[0][0,1]-metrics[0][0,2],metrics[0][0,1]+metrics[0][0,2],8)-vcm
    zz = metrics[0][0,4:]
    yy2 = np.linspace(metrics[0][0,1]-metrics[0][0,2],metrics[0][0,1]+metrics[0][0,2],1000)-vcm
    
    x=np.linspace(-0.3,0.3,100);
    y=coefs_sh1[0]+coefs_sh1[1]*x+coefs_sh1[2]*x**2+coefs_sh1[3]*x**3
    x2=np.linspace(-0.3,0.3,4);
    z = coefs_vco[0]+coefs_vco[1]*y+coefs_vco[2]*y**2+coefs_vco[3]*y**3+coefs_vco[4]*y**4+coefs_vco[5]*y**5+coefs_vco[6]*y**6+coefs_vco[7]*y**7
    w=total_coefs[0]+total_coefs[1]*x+total_coefs[2]*x**2+total_coefs[3]*x**3+total_coefs[4]*x**4+total_coefs[5]*x**5+total_coefs[6]*x**6+total_coefs[7]*x**7+total_coefs[8]*x**8+total_coefs[9]*x**9+total_coefs[10]*x**10+total_coefs[11]*x**11
    xx=np.linspace(-SFDRin,SFDRin,100);
    ww=total_coefs[0]+total_coefs[1]*xx+total_coefs[2]*xx**2+total_coefs[3]*xx**3+total_coefs[4]*xx**4+total_coefs[5]*xx**5+total_coefs[6]*xx**6+total_coefs[7]*xx**7+total_coefs[8]*xx**8+total_coefs[9]*xx**9+total_coefs[10]*x**10+total_coefs[11]*x**11 
    zz2 = coefs_vco[0]+coefs_vco[1]*yy2+coefs_vco[2]*yy2**2+coefs_vco[3]*yy2**3+coefs_vco[4]*yy2**4+coefs_vco[5]*yy2**5+coefs_vco[6]*yy2**6+coefs_vco[7]*yy2**7
    yyy = np.linspace(-SFDRin,+SFDRin,8)
    zzz = coefs_vco[0]+coefs_vco[1]*yyy+coefs_vco[2]*yyy**2+coefs_vco[3]*yyy**3+coefs_vco[4]*yyy**4+coefs_vco[5]*yyy**5+coefs_vco[6]*yyy**6+coefs_vco[7]*yyy**7

    plt.plot(x,w)
    plt.plot(xx,ww)
    
    plt.plot(x,y)
    plt.plot(x2,y2)
    
    plt.plot(y,z,linewidth=2)
    plt.plot(yy,zz,linewidth=2)
    plt.plot(yy2,zz2,linewidth=2)
    plt.plot(yyy,zzz,linewidth=4)

    """
#['center_freq','Rout','Power','Gain','LNA_sfdr','LNA_snr','LNA_enob'],...
#...['Current Consumption','Output PP Amplitude','Tracking Bandwidth','SH_ENOB','SH_SNR','SH_SFDR'] Metrics
print(metrics)    # metrics from both LNA and SH

#['Wpmos_in','L_Radius','Cin_OD','Wnmos_follower'],['fin_main_sw','fin_sampling_Cap','frf','fin_battery_cap','Rout_LNA'] Parameters
#print (parameters) # parameters from both LNA and SH

#['center_freq','Gain','LNA_SFDR','Tracking Bandwidth','SH_ENOB','SH_SFDR ','Power']
#     Hz         dB       dB              GHZ             bit       dB         W
smallspecs
