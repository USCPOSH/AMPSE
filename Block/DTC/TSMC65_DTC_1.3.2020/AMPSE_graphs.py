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

gain = 90                # Desired value in dB
ugb = 1e8                 # Desired value in GHz
phase_margin = 1.047        # Desired value in phase margin
cload = 1e-12               # capacitor after each stage
epsilon = 1e-3              # Epsilon in GD
tedad = 5                 # number of parameter candidates 

#==================================================================
#****************  Loading the Regressors  ************************
#==================================================================


# DTC 1st stage
class DTC1(TF_DEFAULT):

    def __init__(self,tech=65):

        self.tech=tech
        self.default_loading()
        
    def default_loading(self):
        if self.tech==65:
            drive       = home_address+'/reg_files/tb_DTC_PNinj'
            sx_f        = drive + '/scX_tb_DTC_PNinj.pkl'
            sy_f        = drive + '/scY_tb_DTC_PNinj.pkl'    
            w_f         = drive + '/w8_tb_DTC_PNinj.p'
            self.w_json = drive + '/model_tb_DTC_PNinj.json'
            self.w_h5   = drive + '/reg_tb_DTC_PNinj.h5'

            #self.parname =          [ 'lbias','lbp','lbn','lin1','lin2','ltn','ltp','vcmo','mamp','fbias','fbp','fbn','fin1','fin2','ftn1','ftn2','ftp1','ftp2']				
            #self.metricname = ['cin', 'cout', 'gain', 'gm', 'pole1', 'pole2', 'rout', 'cmo', 'pwr', 'swing14', 'swing7', 'swingn', 'swingn1', 'swingn4', 'swingp', 'irn'] 
          
            self.minx  = np.array([10 ,10 ,2000 ,10 ])
            self.maxx  = np.array([21 ,21 ,4000 ,21 ])
            self.step  = np.array([1  ,1  ,10 ,1])
        
        self.loading(sx_f,sy_f,w_f)
        

# DTC 1st stage      
class DTC2(TF_DEFAULT):
    
    def __init__(self,tech=65):

        self.tech=tech
        self.default_loading()
        
        
    def default_loading(self):
        if self.tech==65:
            drive       = home_address+'/reg_files/tb_DTC_2nd_stage'
            sx_f        = drive + '/scX_tb_DTC_2nd_stage.pkl'
            sy_f        = drive + '/scY_tb_DTC_2nd_stage.pkl'    
            w_f         = drive + '/w8_tb_DTC_2nd_stage.p'
            self.w_json = drive + '/model_tb_DTC_2nd_stage.json'
            self.w_h5   = drive + '/reg_tb_DTC_2nd_stage.h5'

            #self.parname =          [ 'fbias','lbias','fin','fp','lin','lp','cload','vcmo','mamp']
            #self.metricname = ['cin','cout','gain', 'gm','pole1','rout','zero','cmo','pwr','swingn','swingp'] 
                      
            self.minx  = np.array([100e-12 ,10 ,2000 ,10 ])
            self.maxx  = np.array([300e-12 ,21 ,4000 ,21 ])
            self.step  = np.array([10e-12  ,1  ,100 ,1])

        self.loading(sx_f,sy_f,w_f)

# for spice graph
#def param_to_sxin(param,DTC1,DTC2):
#    x_DTC1 = param[0][0]
#    x_DTC2    = param[1][0]
#    sx_DTC1 = DTC1.np_scalex(x_DTC1) 
#    sx_DTC2	= DTC2.np_scalex(x_DTC2)
#    sx_out =  np.array([list(sx_DTC1)+ list(sx_DTC2)])
#    
#    return sx_out

#==================================================================
#*****************  Building the graph  ***************************
#==================================================================
def graph_tf(sxin,DTC1,DTC2):

    #----------DTC1----------
    sx_DTC1 = sxin[:,0:4]
    print(sx_DTC1)
    x_DTC1  = DTC1.tf_rescalex(sx_DTC1)
    sy_DTC1 = DTC1.tf_reg_relu(sx_DTC1)
    y_DTC1  = DTC1.tf_rescaley(sy_DTC1)
    print(y_DTC1)
    
    #----------DTC2----------
    cnst_DTC2 = DTC2.tf_scalex(tf.concat([y_DTC1[:,2],x_DTC1[:,1],x_DTC1[:,2],x_DTC1[:,3]],axis=0))
    sx_DTC2 = cnst_DTC2
    sx_DTC2 = tf.reshape(sx_DTC2,(1,4))
    x_DTC2 = DTC2.tf_rescalex(sx_DTC2)
    sy_DTC2 = DTC2.tf_reg_relu(sx_DTC2)    
    y_DTC2 = DTC2.tf_rescaley(sy_DTC2)
    
    

    #==================================================================
    #***************  Define constraints and Cost(P)  *****************
    #==================================================================

    dtc_gain = y_DTC1[0,0] - y_DTC1[0,1]
    dtc_offset = y_DTC1[0,1] + y_DTC2[0,0]
    tr = y_DTC2[0,1]
    
    specs = []
    specs.append((dtc_gain-120e-12))                                                               # 0- DTC gain +
    specs.append((dtc_offset-130e-12))                                                             # 1- DTC offset  +
    specs.append((tr-6e-11))                                                                     # 2- rise time + 

    
    constraints = []    
    constraints.append(tf.nn.elu(specs[0]/DTC1.scYscale[0]))
    constraints.append(tf.nn.elu(specs[0]*-1/DTC1.scYscale[0]))
    constraints.append(tf.nn.elu(specs[1]/DTC1.scYscale[1]))
    constraints.append(tf.nn.elu(specs[1]*-1/DTC1.scYscale[1]))
    constraints.append(tf.nn.elu(specs[2]/DTC2.scYscale[1]))
    constraints.append(tf.nn.elu(specs[2]*-1/DTC2.scYscale[1]))


    hardcost=tf.reduce_sum(constraints)
    usercost=tf.reduce_sum(constraints[:-1])
        
    return hardcost,usercost,specs,[x_DTC1, x_DTC2],[y_DTC1, y_DTC2],[dtc_gain,dtc_offset,tr],constraints
   
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
    dtc1 = DTC1()
    dtc2 = DTC2()


    sxin = make_var("DTC1", "DTC2", (1,4), tf.random_uniform_initializer(-np.ones((1,4)),np.ones((1,4))))
    #==================================================================
    #********************  Tensorflow Initiation  *********************
    #==================================================================    
    hardcost,usercost,tf_specs,tf_param,tf_metric,tf_mid, tf_const = graph_tf(sxin,dtc1,dtc2)
    
    
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
        dump( mydict, open( 'regsearch_results1_'+str(cload)+'.p', "wb" ) )
#        savemat('regsearch_constraints.mat',{'const_np':const_np})
    
        
#        sp_value,_,sp_specs,sp_params,sp_metrics,sp_mids,sp_const= graph_spice(np_sxin,folded_cascode,classab,folded_cascode_spice,classab_spice)

        

