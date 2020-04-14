# This script shows the netlist database for the Inverter Database Using the new version of spectreIOlib
# Example for EE536B tutorial
# Qiaochu Zhang, from Mike Chen's Mixed-Signal Group, Ming Hsieh Dept. of ECE, USC
# 03/12/2020

#==================================================================
#*****************  Loading the libraries  ************************
#==================================================================

import sys
#sys.path.insert(0,'/shares/MLLibs/GlobalLibrary')
import os
home_address  = os.getcwd()
#sys.path.insert(0, home_address+'/MLLibs/GlobalLibrary')
sys.path.insert(0,'/home/mohsen/PYTHON_PHD/GlobalLibrary')
#download GlobalLibrary from GitHub, and change the path accordingly
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

tedad = 5 
# number of parameter candidates you will see in the end. This number will control the number of 
# gradient descent which will be performed.

#==================================================================
#****************  Loading the Regressors  ************************
#==================================================================


# Define the class of an Inverter
class INV(TF_DEFAULT): #change the name of the class to the name of your module

    def __init__(self,tech=65):

        self.tech=tech
        self.default_loading()
        
    def default_loading(self):
        if self.tech==65: # change the path of the regression files
            drive       = home_address+'/reg_files/tb_inv'
            sx_f        = drive + '/scX_tb_inv.pkl'
            sy_f        = drive + '/scY_tb_inv.pkl'    
            w_f         = drive + '/w8_tb_inv.p'
            self.w_json = drive + '/model_tb_inv.json'
            self.w_h5   = drive + '/reg_tb_inv.h5'

            self.minx  = np.array([100e-9,100e-9 ])# parameter lower bound
            self.maxx  = np.array([1000e-9,1000e-9 ])# parameter upper bound
            self.stppar  = np.array([100e-9,100e-9])# sampling step
        
        self.loading(sx_f,sy_f,w_f)
    

#==================================================================
#*****************  Building the graph  ***************************
#==================================================================
def graph_tf(sxin,INV): 
# sxin: scaled input parameters, INV: name of the class; and you should include all
# the names of the modules inside the parenthesis

    sx_INV = sxin[:,0:2] # define input parameters for INV
    print(sx_INV)
    x_INV  = INV.tf_rescalex(sx_INV) #rescale input parameters back to real value
    sy_INV = INV.tf_reg_relu(sx_INV) # calculate y = f(x)
    y_INV  = INV.tf_rescaley(sy_INV) #rescale output metrics back to real value
    print(y_INV)
    

    #==================================================================
    #***************  Define constraints and Cost(P)  *****************
    #==================================================================

    delay = y_INV #define metrics of your neural network

# use the metrics of all the modules to calculate top level specs, and store them in an array    
    specs = []
    specs.append((delay-3e-12))
    specs.append(tf.reshape(tf.reduce_sum(x_INV),[1,1]))

# use the specs to construct constraints
    constraints = []    
    constraints.append((tf.nn.relu(specs[0])/INV.scYscale[0])*100)
    constraints.append(tf.abs(specs[1]))
    
# You can define different loss functions based on different combinations of constraints
    hardcost=constraints[0]
    usercost=tf.reduce_sum(constraints)
        
    return hardcost,usercost,specs,[x_INV],[y_INV],[delay],constraints

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
    
    # Define to optimizers. You can change the values in the parenthesis, which is the learning rate
    optimizer1 = tf.compat.v1.train.AdamOptimizer(0.001)
    optimizer2 = tf.compat.v1.train.AdamOptimizer(0.001)
    
    #----------load regressors----------
    
    # Define an object in class INV
    inv = INV()
    # Initialize all design parameters. We use random initialization in this case. We have in total 2 design parameters, 1 module here
    sxin = make_var("INV","INV", (1,2), tf.random_uniform_initializer(-np.ones((1,2)),np.ones((1,2))))
    #==================================================================
    #********************  Tensorflow Initiation  *********************
    #==================================================================    
    hardcost,usercost,tf_specs,tf_param,tf_metric,tf_mid, tf_const = graph_tf(sxin,inv)
    
    # Optimizer1 will minimize hardcost, and optimizer 2 will minimize usercost. Both are defined previously.
    opt1=optimizer1.minimize(hardcost)
    opt2=optimizer2.minimize(usercost)
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
            for i in range(100): # the value in the parenthesis is the total steps of gradient descent that optimizer 1 will perform. You can change to another value.
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
                
            for i in range(100):# the value in the parenthesis is the total steps of gradient descent that optimizer 2 will perform. You can change to another value.
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

        

