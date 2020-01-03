# Rezwan A Rasul

# Design 2-stage amplifier.

#==================================================================
#*****************  Loading the libraries  ************************
#==================================================================

import sys
#sys.path.insert(0,'/shares/MLLibs/GlobalLibrary')
import os
home_address  = os.getcwd()
sys.path.insert(0, home_address+'/GlobalLibrary')
#sys.path.insert(0,'/home/mohsen/PYTHON_PHD/GlobalLibrary')
#from Netlist_Database import Folded_Cascode_spice, ClassAB_spice

from tensorflow_circuit import TF_DEFAULT, make_var, np_elu
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import math
import tensorflow as tf

from scipy.io import savemat
from pickle import dump



#==================================================================
#*******************  Initialization  *****************************
#==================================================================
'''
gain = 90                # Desired value in dB
ugb = 1e8                 # Desired value in GHz
phase_margin = 1.047        # Desired value in phase margin
cload = 1e-12               # capacitor after each stage
'''

epsilon = 1e-3              # Epsilon in GD
tedad = 20                 # number of parameter candidates 


#==================================================================
#****************  Loading the Regressors  ************************
#==================================================================

# DTC 1st stage
class vcdl(TF_DEFAULT):

    def __init__(self,tech=28):

        self.tech=tech
        self.default_loading()
        
    def default_loading(self):
        if self.tech==28:
            drive       = home_address+'/reg_files/vcdl_28'
            sx_f        = drive + '/scX_vcdl_28.pkl'
            sy_f        = drive + '/scY_vcdl_28.pkl'    
            w_f         = drive + '/w8_vcdl_28.p'
            self.w_json = drive + '/model_vcdl_28.json'
            self.w_h5   = drive + '/reg_vcdl_28.h5'

            #self.parname = [ 'vcdl_n_width','vcdl_p_width','tc_n_width','tc_p_width','vdd']				
            #self.metricname = ['fmin', 'fmax', 'pmin', 'pmax'] 
          
            self.minx  = np.array([450e-9 , 720e-9 , 1170e-9 , 1710e-9 , 0.8])
            self.maxx  = np.array([550e-9 , 880e-9 , 1430e-9 , 2090e-9 , 1.0])
            self.step  = np.array([20e-9  , 40e-9 , 40e-9 , 40e-9 , 0.02])
        
        self.loading(sx_f,sy_f,w_f)
        

# DTC 1st stage      
class dll(TF_DEFAULT):
    
    def __init__(self,tech=28):

        self.tech=tech
        self.default_loading()
        
        
    def default_loading(self):
        if self.tech==28:
            drive       = home_address+'/reg_files/dll_28'
            sx_f        = drive + '/scX_dll_28.pkl'
            sy_f        = drive + '/scY_dll_28.pkl'    
            w_f         = drive + '/w8_dll_28.p'
            self.w_json = drive + '/model_dll_28.json'
            self.w_h5   = drive + '/reg_dll_28.h5'

            #self.parname = [ 'CP_out_width','nor_n_width','nor_p_width','tspc_n_width','tspc_p_width', 'vdd']
            #self.metricname = ['deadzone','power'] 
                      
            self.minx  = np.array([720e-9 , 225e-9 , 360e-9 , 225e-9 , 360e-9, 0.8])
            self.maxx  = np.array([880e-9 , 225e-9 , 360e-9 , 225e-9 , 360e-9, 1.0])
            self.step  = np.array([40e-9 , 20e-9 , 30e-9 , 20e-9 , 30e-9, 0.04])

        self.loading(sx_f,sy_f,w_f)

# for spice graph
#def param_to_sxin(param,cs_driver_cml,cs_array_8b):
#    x_cs_driver_cml = param[0][0]
#    x_cs_array_8b    = param[1][0]
#    sx_cs_driver_cml = cs_driver_cml.np_scalex(x_cs_driver_cml) 
#    sx_cs_array_8b	= cs_array_8b.np_scalex(x_cs_array_8b)
#    sx_out =  np.array([list(sx_cs_driver_cml)+ list(sx_cs_array_8b)])
#    
#    return sx_out

#==================================================================
#*****************  Building the graph  ***************************
#==================================================================
def graph_tf(sxin,vcdl,dll):

    #----------vcdl----------
    # Parameters: ['vcdl_n_width','vcdl_p_width','tc_n_width','tc_p_width','vdd']
    #                   0               1            2            3           4
    # Metrics:    [ 'fmin', 'fmax', 'pmin', 'pmax'] 
    #                 0        1       2        3
    sx_vcdl = sxin[:,0:5] #Giao#
#    print(sx_cs_driver_cml)
    x_vcdl  = vcdl.tf_rescalex(sx_vcdl)
    sy_vcdl = vcdl.tf_reg_relu(sx_vcdl)
    y_vcdl  = vcdl.tf_rescaley(sy_vcdl)
#    print(y_cs_driver_cml)
    
    #----------dll----------
    # Parameters: ['CP_out_width','nor_n_width','nor_p_width','tspc_n_width','tspc_p_width', 'vdd']
    #                  5                6             7             8              9
    # Metrics:    [ 'deadzone','power'] 
#    cnst_cs_array_8b = cs_array_8b.tf_scalex(tf.concat([y_cs_driver_cml[:,2],x_cs_driver_cml[:,1],x_cs_driver_cml[:,2],x_cs_driver_cml[:,3]],axis=0)) #Giao#
    chosen = np.array([1,1,1,1,1,0])
    vars_dll = sxin[:,5:10]; # parameters for dll itself  
    cnst_dll = dll.tf_scalex2(tf.stack([x_vcdl[0,4]],axis=0), 1-chosen) # from vcdl
    sx_dll = tf.concat((vars_dll[0,0:5], cnst_dll[0:1]),axis=0)
    sx_dll = tf.reshape(sx_dll,(1,6)) #Giao#
    x_dll = dll.tf_rescalex(sx_dll)
    sy_dll = dll.tf_reg_relu(sx_dll)    
    y_dll = dll.tf_rescaley(sy_dll)
    
    

    #==================================================================
    #***************  Define constraints and Cost(P)  *****************
    #==================================================================
    
    # vcdl metrics
    fmin = y_vcdl[0,0];
    fmax = y_vcdl[0,1];
    pmin = y_vcdl[0,2];
    pmax = y_vcdl[0,3];
    # dll metrics
    deadzone = y_dll[0,0];
    

    
    specs = [] 
    specs.append(fmin)                   # fmin >= 6.5GHz
    specs.append(fmax)           # fmax < 7GHz  
    specs.append(pmin)   # pmin > 3.5mW   
    specs.append(pmax)   # pmax < 4mW      
    specs.append(deadzone)   # deadzone <= 1ps  
    
    constraints = []    #Giao#
    constraints.append(tf.nn.elu((specs[0]-6.5e9)*(-1)/vcdl.scYscale[0])) # Specify dll running frequency range
    constraints.append(tf.nn.elu((specs[1]-7e9)/vcdl.scYscale[1])) # Specify dll running frequency range
    constraints.append(tf.nn.elu((specs[2]-1e-3)*(-1)/vcdl.scYscale[2])) # 
    constraints.append(tf.nn.elu((specs[3]-4e-3)/vcdl.scYscale[3])) # Minimize the power
    constraints.append(tf.nn.elu((specs[4]-1e-12)/dll.scYscale[0])) # Minimize the deadzone
    
    hardcost=tf.reduce_sum(constraints)
    usercost=tf.reduce_sum(constraints[0:2]+constraints[3:5])
        
    return hardcost,usercost,specs,[x_vcdl, x_dll],[y_vcdl, y_dll],[fmin,fmax,pmin,pmax,deadzone],constraints
   
#==================================================================
#*****************  Building the graph  ***************************
#==================================================================
#def graph_spice(sxin,folded_cascode,classab,folded_cascode_spice,classab_spice):
#    #----------Folded Cascode's graph----------
#    sx_folded_cascode = sxin[:,0:18]
#    x_folded_cascode  = folded_cascode.np_rescalex(sx_folded_cascode)
#    x_folded_cascode , d_folded_cascode = folded_cascode_spice.wholerun_std(np.array(list(x_folded_cascode[0]) + [cload])) 
#    y_folded_cascode = np.array([d_folded_cascode])
#    
#    #----------Class AB's graph----------
#    sx_classab = sxin[:,18:26]
#    x_classab  = classab.np_rescalex(sx_classab)
#    x_classab , d_classab = classab_spice.wholerun_std(np.array(list(x_classab[0]) + [cload]))
#    y_classab = np.array([d_classab])
#    
#    #--------Other variables--------
#    r_c = np.absolute(sxin[0,26]) * 1e6
#    c_c = np.absolute(sxin[0,27]) * 1e-12
#
#    #==================================================================
#    #***************  Define constraints and Cost(P)  *****************
#    #==================================================================
#
#    gain_value = y_folded_cascode[0,3] * y_folded_cascode[0,6] *  y_classab[0,3] * y_classab[0,5]
#    pole_1 = 1 / (y_folded_cascode[0,6] * ((1 + y_classab[0,3] * y_classab[0,5]) * c_c + y_folded_cascode[0,1] + y_classab[0,0]) + y_classab[0,5] * (c_c + cload + y_classab[0,1]))
#    ugb_approx = gain_value * pole_1 / 6.28
#    pole_2_denom = y_folded_cascode[0,6] * y_classab[0,5] * (c_c * (y_folded_cascode[0,1] + y_classab[0,0]) + c_c * cload + (y_folded_cascode[0,1] + y_classab[0,0]) * cload)
#    pole_2 = 1 / (pole_1 * pole_2_denom)
#    zero_1 = 1/(c_c * (1 / y_classab[0,3] - r_c))
#
#    specs = []
#    specs.append(y_folded_cascode[0,2] + y_classab[0,2])                                                               # 0- gain of the amplifier +
#    specs.append(ugb_approx)                                                                           # 1- UGB of the amplifier  +
#    specs.append(np.arctan(ugb_approx/zero_1) - np.arctan(ugb_approx/pole_2) - np.arctan(ugb_approx/pole_1) + 3.1416)
##    specs.append(zero_1)                                                                                               # 2- pole-zero cancellation + 
#    specs.append(y_folded_cascode[0,9])                                                                                         
#    specs.append(y_folded_cascode[0,10])
#    specs.append(y_folded_cascode[0,11])
#    specs.append(y_folded_cascode[0,12])
#    specs.append(y_folded_cascode[0,13])
#    specs.append(y_folded_cascode[0,14])
#    specs.append(y_classab[0,9])
#    specs.append(y_classab[0,10])
#    specs.append(y_folded_cascode[0,8] + y_classab[0,8])                                                                         # 5- Power consumption - 
#    
#    
#    constraints = []    
#    constraints.append(max(gain - specs[0]  ,0)/folded_cascode.scYscale[2])
#    constraints.append(max(ugb - specs[1]  ,0)/folded_cascode.scYscale[4])
#    constraints.append(max(phase_margin - specs[2]  ,0)/phase_margin)
#    constraints.append(max(-1 * specs[3]  ,0)/folded_cascode.scYscale[9])
#    constraints.append(max(-1 * specs[4]  ,0)/folded_cascode.scYscale[10])
#    constraints.append(max(-1 * specs[5]  ,0)/folded_cascode.scYscale[11])
#    constraints.append(max(-1 * specs[6]  ,0)/folded_cascode.scYscale[12])
#    constraints.append(max(-1 * specs[7]  ,0)/folded_cascode.scYscale[13])
#    constraints.append(max(-1 * specs[8]  ,0)/folded_cascode.scYscale[14])
#    constraints.append(max(-1 * specs[9]  ,0)/classab.scYscale[9])
#    constraints.append(max(-1 * specs[10]  ,0)/classab.scYscale[10])    
#    constraints.append(-1 * specs[-1]/folded_cascode.scYscale[8])    
#
#    hardcost= max(constraints)
#    usercost= max (constraints[:-1])
#        
#    return hardcost,usercost,specs,[x_folded_cascode, x_classab, r_c, c_c],[y_folded_cascode, y_classab],[pole_1, pole_2, zero_1],constraints
   
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
#    optimizer1 = tf.train.GradientDescentOptimizer(0.001)
    optimizer1 = tf.compat.v1.train.AdamOptimizer(0.001)
#    optimizer2 = tf.train.GradientDescentOptimizer(0.0005)
    optimizer2 = tf.compat.v1.train.AdamOptimizer(0.001)
    
    #----------load regressors----------
    vcdl = vcdl()
    dll = dll()


    sxin = make_var("vcdl", "dll", (1,10), tf.random_uniform_initializer(-np.ones((1,10)),np.ones((1,10)))) #Giao#
    #==================================================================
    #********************  Tensorflow Initiation  *********************
    #==================================================================    
    hardcost,usercost,tf_specs,tf_param,tf_metric,tf_mid, tf_const = graph_tf(sxin,vcdl,dll)
    
    
    opt1=optimizer1.minimize(hardcost)
    opt2=optimizer2.minimize(hardcost)
    init=tf.compat.v1.global_variables_initializer()
    
    calc=1

    
    lastvalue=-1000000
    lst_params=[]
    lst_metrics=[]
    lst_specs=[]
    lst_value=[]
    lst_midvalues=[]

    for j in range(tedad):
        const=[]
        with tf.compat.v1.Session() as sess:
            sess.run(init)
            #==================================================================
            #*****************  Tensorflow Gradient Descent  ******************
            #==================================================================
            tstart=time.time()
            for i in range(5000):
                try:
                    _,value,smallspecs,last_const = sess.run([opt1,hardcost,tf_specs,tf_const])                
                except:
                    print('Terminated due to error!')
                    break
                
                print('Optimizer1 = %1.0f:, %1.0f : %1.3f \n'%(j, i, value))
                const.append(smallspecs)
                if math.isnan(value):
                    break
                if np.abs(lastvalue-value)<epsilon:
                    break
                else:
                    lastvalue=value
                
            for i in range(5000):
                try:
                    _,value,smallspecs,last_const = sess.run([opt2,hardcost,tf_specs,tf_const])                
                except:
                    print('Terminated due to error!')
                    break                  
                print('Optimizer2 = %1.0f:, %1.0f : %1.3f \n'%(j, i, value))
                const.append(smallspecs)
                
                
                if math.isnan(value):
                    break
                if np.abs(lastvalue-value)<epsilon:
                    break
                else:
                    lastvalue=value
                    
            #==================================================================
            #**********************  Saving the values  ***********************
            #==================================================================    
            print('%1.0f : %1.3f \n'%(i, value))
            tend=time.time()
            np_sxin = sess.run(sxin)
            parameters = sess.run(tf_param)
            metrics    = sess.run(tf_metric)
            midvalues  = sess.run(tf_mid)
          

#            print('user1: %1.2f, user2: %1.2f, user3: %1.2f, user4: %1.2f, user5: %1.2f\n' %(sess.run(user1),sess.run(user2),sess.run(user3),sess.run(user4),sess.run(user5)))
            print('the elapsed time %1.2f S\n' %(tend-tstart))
        
        const_np=np.array(const)
        lst_params.append(parameters)
        lst_metrics.append(metrics)
        lst_specs.append(const[-1])
        lst_value.append(value)
        lst_midvalues.append(midvalues)
        
        mydict= {'lst_params':lst_params,'lst_metrics':lst_metrics,'lst_specs':lst_specs,'lst_value':lst_value}
#        dump( mydict, open( 'regsearch_results1_'+str(cload)+'.p', "wb" ) )
#        savemat('regsearch_constraints.mat',{'const_np':const_np})
    
        
#        sp_value,_,sp_specs,sp_params,sp_metrics,sp_mids,sp_const= graph_spice(np_sxin,folded_cascode,classab,folded_cascode_spice,classab_spice)

        
