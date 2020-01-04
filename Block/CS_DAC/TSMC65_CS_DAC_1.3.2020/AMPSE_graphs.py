# Rezwan A Rasul

# Design 2-stage amplifier.

#==================================================================
#*****************  Loading the libraries  ************************
#==================================================================

import sys
#sys.path.insert(0,'/shares/MLLibs/GlobalLibrary')
import os
home_address  = os.getcwd()
#sys.path.insert(0, home_address+'/MLLibs/GlobalLibrary')
sys.path.insert(0,'/home/mohsen/PYTHON_PHD/GlobalLibrary')
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
class cs_driver_cml(TF_DEFAULT):

    def __init__(self,tech=65):

        self.tech=tech
        self.default_loading()
        
    def default_loading(self):
        if self.tech==65:
            drive       = home_address+'/reg_files/cs_driver_cml'
            sx_f        = drive + '/scX_cs_driver_cml.pkl'
            sy_f        = drive + '/scY_cs_driver_cml.pkl'    
            w_f         = drive + '/w8_cs_driver_cml.p'
            self.w_json = drive + '/model_cs_driver_cml.json'
            self.w_h5   = drive + '/reg_cs_driver_cml.h5'

            #self.parname = [ 'R_driver','l_cs_driver','v_swing','vbias','w_cs_driver','w_sw']				
            #self.metricname = ['pwr', 'output_swing', 'rise_time', 'fall_time'] 
          
            self.minx  = np.array([40 , 300e-9 , 0.3 , 0.3 , 6e-6  , 400e-9])
            self.maxx  = np.array([70 , 500e-9 , 0.6 , 0.7 , 10e-6 , 700e-9])
            self.step  = np.array([5  , 100e-9 , 0.1 , 0.1 , 1e-6  , 100e-9])
        
        self.loading(sx_f,sy_f,w_f)
        

# DTC 1st stage      
class cs_array_8b(TF_DEFAULT):
    
    def __init__(self,tech=65):

        self.tech=tech
        self.default_loading()
        
        
    def default_loading(self):
        if self.tech==65:
            drive       = home_address+'/reg_files/cs_array_8b'
            sx_f        = drive + '/scX_cs_array_8b.pkl'
            sy_f        = drive + '/scY_cs_array_8b.pkl'    
            w_f         = drive + '/w8_cs_array_8b.p'
            self.w_json = drive + '/model_cs_array_8b.json'
            self.w_h5   = drive + '/reg_cs_array_8b.h5'

            #self.parname = [ 'l_cs','v_swing','vbias_SW_cas','w_cs','w_sw']
            #self.metricname = ['SFDR','pwr','i_fullscale'] 
                      
            self.minx  = np.array([400e-9 , 0.1 , 1.2 , 0.25e-6 , 400e-9])
            self.maxx  = np.array([800e-9 , 0.5 , 2   , 2e-6    , 700e-9])
            self.step  = np.array([100e-9 , 0.1 , 0.2 , 0.25e-6 , 100e-9])

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
def graph_tf(sxin,cs_driver_cml,cs_array_8b):

    #----------cs_driver_cml----------
    # Parameters: [ 'R_driver','l_cs_driver','v_swing_driver','vbias_driver','w_cs_driver','w_sw']
    #                   0               1            2               3             4          5
    # Metrics:    [ 'pwr', 'output_swing', 'rise_time', 'fall_time'] 
    #                 0           1             2            3
    sx_cs_driver_cml = sxin[:,0:6] #Giao#
#    print(sx_cs_driver_cml)
    x_cs_driver_cml  = cs_driver_cml.tf_rescalex(sx_cs_driver_cml)
    sy_cs_driver_cml = cs_driver_cml.tf_reg_relu(sx_cs_driver_cml)
    y_cs_driver_cml  = cs_driver_cml.tf_rescaley(sy_cs_driver_cml)
#    print(y_cs_driver_cml)
    
    #----------cs_array_8b----------
    # Parameters: [ 'l_cs','output_swing','vbias_SW_cas','w_cs','w_sw']
    #                  6                         7         8
    # Metrics:    [ 'SFDR','pwr','i_fullscale'] 
#    cnst_cs_array_8b = cs_array_8b.tf_scalex(tf.concat([y_cs_driver_cml[:,2],x_cs_driver_cml[:,1],x_cs_driver_cml[:,2],x_cs_driver_cml[:,3]],axis=0)) #Giao#
    chosen = np.array([1,0,1,1,0])
    vars_cs_array_8b = sxin[:,6:9]; # parameters for cs_array_8b itself  
    cnst_cs_array_8b = cs_array_8b.tf_scalex2(tf.stack([y_cs_driver_cml[0,1],x_cs_driver_cml[0,5]],axis=0), 1-chosen) # from cs_driver_cml
    sx_cs_array_8b = tf.concat((vars_cs_array_8b[0,0:1], cnst_cs_array_8b[0:1], vars_cs_array_8b[0,1:3], cnst_cs_array_8b[1:2]),axis=0)
    sx_cs_array_8b = tf.reshape(sx_cs_array_8b,(1,5)) #Giao#
    x_cs_array_8b = cs_array_8b.tf_rescalex(sx_cs_array_8b)
    sy_cs_array_8b = cs_array_8b.tf_reg_relu(sx_cs_array_8b)    
    y_cs_array_8b = cs_array_8b.tf_rescaley(sy_cs_array_8b)
    
    

    #==================================================================
    #***************  Define constraints and Cost(P)  *****************
    #==================================================================
    
    # driver metrics
    power_driver = y_cs_driver_cml[0,0];
    output_swing_driver = y_cs_driver_cml[0,1];
    rise_time = y_cs_driver_cml[0,2];
    fall_time = y_cs_driver_cml[0,3];
    # cs_array metrics
    SFDR = y_cs_array_8b[0,0];
    power_cs = y_cs_array_8b[0,1];
    fullscale_current = y_cs_array_8b[0,2];
    power_total = power_driver * 19 + power_cs;

    #dtc_gain = y_cs_driver_cml[0,0] - y_cs_driver_cml[0,1] #Giao#
    #dtc_offset = y_cs_driver_cml[0,1] + y_cs_array_8b[0,0] #Giao#
    #tr = y_cs_array_8b[0,1] #Giao#
    
    specs = [] 
    specs.append(SFDR)                   # SFDR >= 70dB
    specs.append(power_total)           # pwr_cs <= 100mW  
    specs.append(fullscale_current)   # i_fullscale >= 16mA   
    specs.append(rise_time)   # t_rise <= 12ps      
    specs.append(fall_time)   # f_rise <= 12ps  
    
    constraints = []    #Giao#
    constraints.append(tf.nn.elu((specs[0]-70)*(-1)/cs_array_8b.scYscale[0])) # Maximize the SFDR
    constraints.append(tf.nn.elu((specs[1]-200e-3)/cs_array_8b.scYscale[1])) # Minimize the power consumption
    constraints.append(tf.nn.elu((specs[2]-16e-3)*(-1)/cs_array_8b.scYscale[2])) # Maximize the full-scale output current
    constraints.append(tf.nn.elu((specs[3]-12e-12)/cs_driver_cml.scYscale[2])) # Minimize the rise time
    constraints.append(tf.nn.elu((specs[4]-12e-12)/cs_driver_cml.scYscale[3])) # Minimize the fall time
    
    hardcost=tf.reduce_sum(constraints)
    usercost=tf.reduce_sum(constraints[:-1])
        
    return hardcost,usercost,specs,[x_cs_driver_cml, x_cs_array_8b],[y_cs_driver_cml, y_cs_array_8b],[SFDR,power_total,fullscale_current,rise_time,fall_time],constraints
   
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
    cs_driver_cml = cs_driver_cml()
    cs_array_8b = cs_array_8b()


    sxin = make_var("cs_driver_cml", "cs_array_8b", (1,9), tf.random_uniform_initializer(-np.ones((1,9)),np.ones((1,9)))) #Giao#
    #==================================================================
    #********************  Tensorflow Initiation  *********************
    #==================================================================    
    hardcost,usercost,tf_specs,tf_param,tf_metric,tf_mid, tf_const = graph_tf(sxin,cs_driver_cml,cs_array_8b)
    
    
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

        

