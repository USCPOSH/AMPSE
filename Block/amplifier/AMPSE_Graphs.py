# Rezwan A Rasul

# Design 2-stage amplifier.

#==================================================================
#*****************  Loading the libraries  ************************
#==================================================================

import sys
#sys.path.insert(0,'/shares/MLLibs/GlobalLibrary')
import os
home_address  = os.getcwd()
sys.path.insert(0, home_address+'/MLLibs/GlobalLibrary')
from Netlist_Database import Folded_Cascode_spice, ClassAB_spice

from tensorflow_circuit import TF_DEFAULT, make_var
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


# Folded Cascode
class Folded_Cascode(TF_DEFAULT):

    def __init__(self,tech=65):

        self.tech=tech
        self.default_loading()
        
    def default_loading(self):
        if self.tech==65:
            drive       = home_address+'/Reg_files/folded_cascode_45'
            sx_f        = drive + '/scX_folded_cascode.pkl'
            sy_f        = drive + '/scY_folded_cascode.pkl'    
            w_f         = drive + '/w8_folded_cascode.p'
            self.w_json = drive + '/model_folded_cascode.json'
            self.w_h5   = drive + '/reg_folded_cascode.h5'

            #self.parname =          [ 'lbias','lbp','lbn','lin1','lin2','ltn','ltp','vcmo','mamp','fbias','fbp','fbn','fin1','fin2','ftn1','ftn2','ftp1','ftp2']				
            #self.metricname = ['cin', 'cout', 'gain', 'gm', 'pole1', 'pole2', 'rout', 'cmo', 'pwr', 'swing14', 'swing7', 'swingn', 'swingn1', 'swingn4', 'swingp', 'irn'] 

            self.minx  = np.array([  45e-9,  45e-9,   45e-9,  45e-9, 45e-9,   45e-9,  45e-9, 400e-3,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1])
            self.maxx  = np.array([  600e-9, 600e-9, 600e-9, 600e-9, 600e-9, 600e-9, 600e-9, 600e-3, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100])
            self.step  = np.array([  10e-9,  10e-9,   10e-9,  10e-9, 10e-9,   10e-9,  10e-9,  20e-3,  1,   1,  1,  1,  1,  1,  1,  1,  1, 1])

        self.loading(sx_f,sy_f,w_f)
        

# ClassAB      
class ClassAB(TF_DEFAULT):
    
    def __init__(self,tech=65):

        self.tech=tech
        self.default_loading()
        
        
    def default_loading(self):
        if self.tech==65:
            drive       = home_address+'/Reg_files/class_ab_45'
            sx_f        = drive + '/scX_classAB.pkl'
            sy_f        = drive + '/scY_classAB.pkl'    
            w_f         = drive + '/w8_classAB.p'
            self.w_json = drive + '/model_classAB.json'
            self.w_h5   = drive + '/reg_classAB.h5'

            #self.parname =          [ 'fbias','lbias','fin','fp','lin','lp','cload','vcmo','mamp']
            #self.metricname = ['cin','cout','gain', 'gm','pole1','rout','zero','cmo','pwr','swingn','swingp'] 
            
            self.minx  = np.array([  1,  45e-9,   1,   1,  45e-9,  45e-9, 400e-3,   1])
            self.maxx  = np.array([100, 900e-9, 100, 100, 600e-9, 600e-9, 600e-3, 100])
            self.step  = np.array([  1,  20e-9,   1,   1,  10e-9,  10e-9,  20e-3,   1])

        self.loading(sx_f,sy_f,w_f)


def param_to_sxin(param,folded_cascode,classab):
    x_folded_cascode = param[0][0]
    x_classab    = param[1][0]
    r_c = param[2]
    c_c = param[3]
    sx_rc = r_c*1e-6
    sx_cc = c_c*1e12
    sx_folded_cascode = folded_cascode.np_scalex(x_folded_cascode) 
    sx_classab	= classab.np_scalex(x_classab)
    sx_out =  np.array([list(sx_folded_cascode)+ list(sx_classab)+[sx_rc,sx_cc]])
    
    return sx_out

#==================================================================
#*****************  Building the graph  ***************************
#==================================================================
def graph_tf(sxin,folded_cascode,classab):
    # Constants:
    c_cload = tf.constant(cload,dtype = tf.float32)
    #----------Folded Cascode's graph----------
    sx_folded_cascode = sxin[:,0:18]
    x_folded_cascode  = folded_cascode.tf_rescalex(sx_folded_cascode)
    sy_folded_cascode = folded_cascode.tf_reg_elu(sx_folded_cascode)
    y_folded_cascode  = folded_cascode.tf_rescaley(sy_folded_cascode)
    
    #----------Class AB's graph----------
    sx_classab = sxin[:,18:26]
    x_classab = classab.tf_rescalex(sx_classab)
    sy_classab = classab.tf_reg_elu(sx_classab)    
    y_classab = classab.tf_rescaley(sy_classab)
    
    #--------Other variables--------
    r_c = tf.math.abs(sxin[0,26]) * 1e6
    c_c = tf.math.abs(sxin[0,27]) * 1e-12

    #==================================================================
    #***************  Define constraints and Cost(P)  *****************
    #==================================================================

    gain_value = y_folded_cascode[0,3] * y_folded_cascode[0,6] *  y_classab[0,3] * y_classab[0,5]
    pole_1 = 1 / (y_folded_cascode[0,6] * ((1 + y_classab[0,3] * y_classab[0,5]) * c_c + y_folded_cascode[0,1] + y_classab[0,0]) + y_classab[0,5] * (c_c + c_cload + y_classab[0,1]))
    ugb_approx = gain_value * pole_1 / 6.28
    pole_2_denom = y_folded_cascode[0,6] * y_classab[0,5] * (c_c * (y_folded_cascode[0,1] + y_classab[0,0]) + c_c * c_cload + (y_folded_cascode[0,1] + y_classab[0,0]) * c_cload)
    pole_2 = 1 / (pole_1 * pole_2_denom)
    zero_1 = 1/(c_c * (1 / y_classab[0,3] - r_c))

    specs = []
    specs.append(y_folded_cascode[0,2] + y_classab[0,2])                                                               # 0- gain of the amplifier +
    specs.append(ugb_approx)                                                                           # 1- UGB of the amplifier  +
    specs.append(tf.math.atan(ugb_approx/zero_1) - tf.math.atan(ugb_approx/pole_2) - tf.math.atan(ugb_approx/pole_1) + 3.1416)
#    specs.append(zero_1)                                                                                               # 2- pole-zero cancellation + 
    specs.append(y_folded_cascode[0,9])                                                                                         
    specs.append(y_folded_cascode[0,10])
    specs.append(y_folded_cascode[0,11])
    specs.append(y_folded_cascode[0,12])
    specs.append(y_folded_cascode[0,13])
    specs.append(y_folded_cascode[0,14])
    specs.append(y_classab[0,9])
    specs.append(y_classab[0,10])
    specs.append(y_folded_cascode[0,8] + y_classab[0,8])                                                                         # 5- Power consumption - 
    
    
    constraints = []    
    constraints.append(tf.math.maximum(gain - specs[0]  ,0)/folded_cascode.scYscale[2])
    constraints.append(tf.math.maximum(ugb - specs[1]  ,0)/folded_cascode.scYscale[4])
    constraints.append(tf.math.maximum(phase_margin - specs[2]  ,0)/phase_margin)
    constraints.append(tf.math.maximum(-1 * specs[3]  ,0)/folded_cascode.scYscale[9])
    constraints.append(tf.math.maximum(-1 * specs[4]  ,0)/folded_cascode.scYscale[10])
    constraints.append(tf.math.maximum(-1 * specs[5]  ,0)/folded_cascode.scYscale[11])
    constraints.append(tf.math.maximum(-1 * specs[6]  ,0)/folded_cascode.scYscale[12])
    constraints.append(tf.math.maximum(-1 * specs[7]  ,0)/folded_cascode.scYscale[13])
    constraints.append(tf.math.maximum(-1 * specs[8]  ,0)/folded_cascode.scYscale[14])
    constraints.append(tf.math.maximum(-1 * specs[9]  ,0)/classab.scYscale[9])
    constraints.append(tf.math.maximum(-1 * specs[10]  ,0)/classab.scYscale[10])
    constraints.append(-1*specs[-1]/folded_cascode.scYscale[8])    

    hardcost=tf.reduce_sum(constraints)
    usercost=tf.reduce_sum(constraints[:-1])
        
    return hardcost,usercost,specs,[x_folded_cascode, x_classab, r_c, c_c],[y_folded_cascode, y_classab],[pole_1, pole_2, zero_1],constraints
   
#==================================================================
#*****************  Building the graph  ***************************
#==================================================================
def graph_spice(sxin,folded_cascode,classab,folded_cascode_spice,classab_spice):
    #----------Folded Cascode's graph----------
    sx_folded_cascode = sxin[:,0:18]
    x_folded_cascode  = folded_cascode.np_rescalex(sx_folded_cascode)
    x_folded_cascode , d_folded_cascode = folded_cascode_spice.wholerun_std(np.array(list(x_folded_cascode[0]) + [cload])) 
    y_folded_cascode = np.array([d_folded_cascode])
    
    #----------Class AB's graph----------
    sx_classab = sxin[:,18:26]
    x_classab  = classab.np_rescalex(sx_classab)
    x_classab , d_classab = classab_spice.wholerun_std(np.array(list(x_classab[0]) + [cload]))
    y_classab = np.array([d_classab])
    
    #--------Other variables--------
    r_c = np.absolute(sxin[0,26]) * 1e6
    c_c = np.absolute(sxin[0,27]) * 1e-12

    #==================================================================
    #***************  Define constraints and Cost(P)  *****************
    #==================================================================

    gain_value = y_folded_cascode[0,3] * y_folded_cascode[0,6] *  y_classab[0,3] * y_classab[0,5]
    pole_1 = 1 / (y_folded_cascode[0,6] * ((1 + y_classab[0,3] * y_classab[0,5]) * c_c + y_folded_cascode[0,1] + y_classab[0,0]) + y_classab[0,5] * (c_c + cload + y_classab[0,1]))
    ugb_approx = gain_value * pole_1 / 6.28
    pole_2_denom = y_folded_cascode[0,6] * y_classab[0,5] * (c_c * (y_folded_cascode[0,1] + y_classab[0,0]) + c_c * cload + (y_folded_cascode[0,1] + y_classab[0,0]) * cload)
    pole_2 = 1 / (pole_1 * pole_2_denom)
    zero_1 = 1/(c_c * (1 / y_classab[0,3] - r_c))

    specs = []
    specs.append(y_folded_cascode[0,2] + y_classab[0,2])                                                               # 0- gain of the amplifier +
    specs.append(ugb_approx)                                                                           # 1- UGB of the amplifier  +
    specs.append(np.arctan(ugb_approx/zero_1) - np.arctan(ugb_approx/pole_2) - np.arctan(ugb_approx/pole_1) + 3.1416)
#    specs.append(zero_1)                                                                                               # 2- pole-zero cancellation + 
    specs.append(y_folded_cascode[0,9])                                                                                         
    specs.append(y_folded_cascode[0,10])
    specs.append(y_folded_cascode[0,11])
    specs.append(y_folded_cascode[0,12])
    specs.append(y_folded_cascode[0,13])
    specs.append(y_folded_cascode[0,14])
    specs.append(y_classab[0,9])
    specs.append(y_classab[0,10])
    specs.append(y_folded_cascode[0,8] + y_classab[0,8])                                                                         # 5- Power consumption - 
    
    
    constraints = []    
    constraints.append(max(gain - specs[0]  ,0)/folded_cascode.scYscale[2])
    constraints.append(max(ugb - specs[1]  ,0)/folded_cascode.scYscale[4])
    constraints.append(max(phase_margin - specs[2]  ,0)/phase_margin)
    constraints.append(max(-1 * specs[3]  ,0)/folded_cascode.scYscale[9])
    constraints.append(max(-1 * specs[4]  ,0)/folded_cascode.scYscale[10])
    constraints.append(max(-1 * specs[5]  ,0)/folded_cascode.scYscale[11])
    constraints.append(max(-1 * specs[6]  ,0)/folded_cascode.scYscale[12])
    constraints.append(max(-1 * specs[7]  ,0)/folded_cascode.scYscale[13])
    constraints.append(max(-1 * specs[8]  ,0)/folded_cascode.scYscale[14])
    constraints.append(max(-1 * specs[9]  ,0)/classab.scYscale[9])
    constraints.append(max(-1 * specs[10]  ,0)/classab.scYscale[10])    
    constraints.append(-1 * specs[-1]/folded_cascode.scYscale[8])    

    hardcost= max(constraints)
    usercost= max (constraints[:-1])
        
    return hardcost,usercost,specs,[x_folded_cascode, x_classab, r_c, c_c],[y_folded_cascode, y_classab],[pole_1, pole_2, zero_1],constraints
   
#==================================================================
#*********************  Main code  ********************************
#==================================================================

if __name__ == '__main__':
    

    
  
    #==================================================================
    #*****************  Building the graph  ***************************
    #==================================================================
    
    #----------Initialize----------
    tf.reset_default_graph()
#    optimizer1 = tf.train.GradientDescentOptimizer(0.001)
    optimizer1 = tf.train.AdamOptimizer(0.01)
#    optimizer2 = tf.train.GradientDescentOptimizer(0.0005)
    optimizer2 = tf.train.AdamOptimizer(0.001)
    
    #----------load regressors----------
    folded_cascode = Folded_Cascode()
    classab = ClassAB()

    folded_cascode_spice = Folded_Cascode_spice()
    classab_spice = ClassAB_spice()

    sxin = make_var("amplifier", "fc_classab", (1,28), tf.random_uniform_initializer(-np.ones((1,28)),np.ones((1,28))))
    #==================================================================
    #********************  Tensorflow Initiation  *********************
    #==================================================================    
    hardcost,usercost,tf_specs,tf_param,tf_metric,tf_mid, tf_const = graph_tf(sxin,folded_cascode,classab)
    
    
    opt1=optimizer1.minimize(hardcost)
    opt2=optimizer2.minimize(hardcost)
    init=tf.global_variables_initializer()
    
    calc=1

    
    lastvalue=-1000000
    lst_params=[]
    lst_metrics=[]
    lst_specs=[]
    lst_value=[]
    lst_midvalues=[]

    for j in range(tedad):
        const=[]
        with tf.Session() as sess:
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

        

        sp_sxin = param_to_sxin(parameters,folded_cascode,classab)
