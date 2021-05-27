# Designed by Mohsen Hassanpourghadi & Rezwan A Rasul
# Date 5/26/2020

# Design SAR ADC with TF.

#==================================================================
#*****************  Loading the libraries  ************************
#==================================================================
fs=200                      # Sampling Frequency
nbit=11                     # Number of Bits

len_mc=10
weights = [10.0,2.0,1.0,0.0,1.0,1.0,1.0,1.0,100.0,1.0/5]
epsilon=1e-5                # Epsilon in GD
tedad=1                     # number of parameter candidates 
maxiter=10000               # Maximum iteration
n_stairs=10                 # Maximum number of quantized inverters
lr_hard = 0.2
lr_soft = 0.02
globallibrary = 'E:\PYTHON_PHD\GlobalLibrary'
try:
        
    exec(open("Inputs.txt",'rb').read())

except Exception:
    print('warning: could not access Inputs.txt')

import sys
sys.path.insert(0,globallibrary)


import os
home_address  = os.getcwd()
from Netlist_Database import Compp_spice3,DACTH2_spice,Seqpart1_spice,Seqpart2_spice

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import math

from scipy.io import savemat
from pickle import dump, load

from tensorflow_circuit import TF_DEFAULT, make_var, np_elu, np_sigmoid, np_sigmoid_inv
import tensorflow as tf
from reg_database import COMPP2, THDAC2, SEQ1, SEQ2, tf_quant_with_sigmoid, param_to_sxin, step2sxin
from ampse_graph import graph_tf3
from termcolor import colored, cprint

tf.config.optimizer.set_jit(True)

# limit verbosity
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_WARNINGS'] = 'off'
import tensorflow as tf

from tensorflow.python.util import module_wrapper as deprecation
deprecation._PER_MODULE_WARNING_LIMIT = 0
deprecation._PRINT_DEPRECATION_WARNINGS = False







#==================================================================
#*********************  Main code  ********************************
#==================================================================

if __name__ == '__main__':
    
    
    
    
    
    u=np.array([nbit,fs])
    s=[n_stairs,weights]
    #==================================================================
    #*****************  Building the graph  ***************************
    #==================================================================
    
    tf.compat.v1.disable_eager_execution()
    #----------Initialize----------
    tf.compat.v1.reset_default_graph()
    optimizer1 = tf.compat.v1.train.AdamOptimizer(lr_hard)
    optimizer2 = tf.compat.v1.train.AdamOptimizer(lr_soft)
    
    
    #----------load regressors----------
    seqp11 = SEQ1()
    seqp21 = SEQ2()
    compp1 = COMPP2()
    thdac1 = THDAC2()
    

    comppspice1 = Compp_spice3()
    dacthspice1 = DACTH2_spice()
    seqp1spice1 = Seqpart1_spice()
    seqp2spice1 = Seqpart2_spice()
    

    var_in = make_var("SAR_ADC", "SEQ_COMPP_THDAC", (len_mc,26), tf.random_uniform_initializer(-np.ones((len_mc,26))*3,np.ones((len_mc,26))*3))
    
    xload = tf.compat.v1.placeholder(tf.float32, shape=(len_mc,26))
    initvar = var_in.assign(xload)
#    
    tf_u = tf.compat.v1.placeholder(tf.float32,shape=(2,))
    sxin = 2*tf.math.sigmoid(var_in)-1.0

    softcost,hardcost,tf_specs,tf_params,tf_metrics,tf_mids,tf_const,tf_softs,tf_hards = graph_tf3(sxin,seqp11,seqp21,compp1,thdac1,tf_u,s)
    
    #==================================================================
    #********************  Tensorflow Initiation  *********************
    #==================================================================    
    
    
    grd1=optimizer1.compute_gradients(hardcost)
    opt1=optimizer1.apply_gradients(grd1)
    grd2 = optimizer2.compute_gradients(softcost,[var_in])
    opt2 = optimizer2.apply_gradients(grd2)
    
    
        

    
    init=tf.compat.v1.global_variables_initializer()
    
    calc=1

    
    lastvalue=-1000000

    
    
    
    print('Algorithm Starts Here:')
    k=0
    doogh=0
    lst_possibility=[]
    lst_super_possibility=[]
    lst_fom=[]
    lst_vvvv=[]
    
    tstart=time.time()
    for nbit in range(nbit,nbit+1):
        for fs_MHz in range(int(fs),int(fs+1),1):
            
            fs = fs_MHz*1e6
            
    
            u=np.array([nbit,fs])
            reg_specs=[]
            const=[]
            
            
            
            keep_list_x =[]
            keep_list_y =[]
            keep_list_v =[]
            lst_add=[]
            with tf.compat.v1.Session() as sess:
                sess.run(init)
                
                
                # sess.run(initvar,feed_dict={xload:y[:len_mc,:]})
                #==================================================================
                #*****************  Tensorflow Gradient Descent  ******************
                #==================================================================
                
                i=0
                doogh=0
                tstart_iter=time.time()
                while i <maxiter:
    
                    
                    try:
                        if i<maxiter/2:
                            _,value,smallspecs = sess.run([opt1,hardcost,tf_specs],feed_dict={tf_u:u})                
                        else:
                            _,value,smallspecs = sess.run([opt2,softcost,tf_specs],feed_dict={tf_u:u})   
                            
                        k+=1
                        doogh+=1
                    except:
                        print('Terminated due to error!')
                        break
                    print('%1.0f-bit,%1.0f-MS/s, iter: %1.0f, cost: %1.3f \n'%(nbit,fs_MHz, i, value))
                    
                    
    #                smallspecs.append(doogh)
    #                reg_specs.append(smallspecs)
                    if math.isnan(value) or math.isinf(value):
                        break
                    else:
                        soft_costs = sess.run(tf_softs,feed_dict={tf_u:u})
                        lst_add.append([min(soft_costs)[0],time.time()-tstart_iter])
                        
                        smallspecs = sess.run(tf_specs,feed_dict={tf_u:u})
                        parameters = sess.run(tf_params,feed_dict={tf_u:u})
                        metrics    = sess.run(tf_metrics,feed_dict={tf_u:u})
                        midvalues = sess.run(tf_mids,feed_dict={tf_u:u})
                        const = sess.run(tf_const,feed_dict={tf_u:u})
                        np_sxin = sess.run(sxin,feed_dict={tf_u:u})
                        dacdelay, digdelay, ctot, d_ts,bw1,bw2= midvalues
                        
                    if len_mc<2:    
                        keep_list_x.append(np_sxin)
                        keep_list_y.append(smallspecs)
                        keep_list_v.append(value)
                        
                    if i<maxiter/2 and np.abs(lastvalue-value)<0.001*len_mc:
                        i=int(maxiter/2)+1
                        print(i)
                    elif i>maxiter/2 and np.abs(lastvalue-value)<epsilon*len_mc:
                        break
                    else:
                        lastvalue=value
                    i+=1
                    
    #            if not math.isnan(value):
                hard_costs = sess.run(tf_hards,feed_dict={tf_u:u})
                soft_costs = sess.run(tf_softs,feed_dict={tf_u:u})
                gradients  = sess.run(grd2,feed_dict={tf_u:u})
                possibility = (1*(const[0]<=0)*( const[4]<=0)*( const[5]<=0)*(const[6]<=0)*( const[7]<=0)*(const[8]<=0))
                super_possibility = (1*(const[0]<=0)*(const[1]<=0)*(const[2]<=0)*( const[4]<=0)*( const[5]<=0)*(const[6]<=0)*( const[7]<=0)*(const[8]<=0))
                lst_possibility.append([nbit,fs,sum(possibility.T[0])/len_mc])
                lst_super_possibility.append([nbit,fs,sum(super_possibility.T[0])/len_mc])
                

            
            #==================================================================
            #**********************  Saving the values  ***********************
            #==================================================================
            

            tend=time.time()
            print('%1.0f:' %(len_mc))
            print(tend-tstart)
            print(k*len_mc)
            print(min(soft_costs)[0]) 
            print(np.mean(soft_costs))
            new_fs = 1/(1/fs-smallspecs[1])
            enob   = (10*np.log10(1/(10**(-(6*nbit+1.76)/10)+10**(-smallspecs[2]/10)))-1.76)/6
            fom    = smallspecs[-1]/(2**enob)/new_fs
            vvvv = np.concatenate([u*np.ones((len_mc,2)),new_fs*possibility,enob*possibility,fom*possibility],axis=1)
            lst_vvvv = lst_vvvv+list(vvvv)
            print('the elapsed time %1.2f S\n' %(tend-tstart))
            if sum(super_possibility)==0:
                break
    np_sth = np.array(lst_possibility)
    np_vvvv = np.array(lst_vvvv)      
    

    
    lst_metrics=[]
    lst_params=[]
    lst_specs=[]
    lst_mids=[]
    lst_value=[]
    for i in range(len_mc):
        lst_metrics.append([metrics[0][i,:],metrics[1][i,:],metrics[2][i,:],metrics[3][i,:],metrics[4][i,:]])
        lst_params.append([parameters[0][i,:],parameters[1][i,:],parameters[2][i,:],parameters[3][i,:],parameters[4][i,:][0],parameters[5][i,:][0]])
        lst_specs.append([smallspecs[j][i,:] for j in range(len(smallspecs))])
        lst_mids.append([midvalues[j][i,:] for j in range(len(midvalues))])
    lst_value = list(soft_costs.T[0])    
    


    mydict= {'lst_params':lst_params,'lst_metrics':lst_metrics,'lst_specs':lst_specs,'lst_value':lst_value,'midvalues':lst_mids,'grads':gradients[0][1],'time_elapsed':tend-tstart}
    
    
    
    
    np_specs = np.array(smallspecs).squeeze().T
    np_mids  = np.array(midvalues).squeeze().T
    
    np_finalresult = np.concatenate((np.arange(1,len_mc+1).reshape(len_mc,1),np_mids[:,[2,4]],np_specs[:,[2,9]],np_vvvv[:,[2,3,4]]),axis=1)
    columns = ['Index',"Total input cap","Input's bandwidth","SNR","Power consumption","Sampling frequency","ENOB","FOM"]
    
    f= open('saved_pickle/out_'+str(nbit)+str(int(fs/1e6))+'.p','wb')
    dump(mydict,f)
    f.close()
    
    np_params = np.zeros((len_mc,39))
    for j,i in enumerate(lst_params):
        np_params[j,:] = np.concatenate([[j+1],i[0],i[1],i[2],i[3],[i[4],i[5]]])
   
    p_columns = ['index']+['SEQ1']*13 + ['SEQ2']*2+['Comparator']*15+['TH and CDAC']*6 +['# of buffers']+['Duty Cycle of sampling']
    
    df_specs = pd.DataFrame(np_finalresult)
    
    df_params = pd.DataFrame(np_params)
    df_specs.to_csv('results/out_'+str(nbit)+str(int(fs/1e6))+'.csv',index=False,header=columns)
    df_params.to_csv('results/par_'+str(nbit)+str(int(fs/1e6))+'.csv',index=False,header=p_columns)

    
    
    
    
    