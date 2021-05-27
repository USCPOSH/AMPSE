# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from reg_database import  tf_quant_with_sigmoid
from tensorflow_circuit import np_elu

#==================================================================
#*****************  Building the graph  ***************************
#==================================================================

KT=4.14e-21                 # Boltzman Constant * 300



def graph_tf3(sxin,seqp11,seqp21,compp1,thdac1,u,s):
    
    nbit = u[0]
    fs   = u[1]
    n_stairs =s[0]
    weights = s[1]
    # Constants:
    c_nbit = tf.dtypes.cast(nbit,dtype = tf.float32)
    c_lvls = tf.dtypes.cast(2**(nbit-1),dtype = tf.float32)
    len_mc = sxin.get_shape().as_list()[0]
    #----------SEQ1's graph----------    
    # [ 'drvbuf11','drvbuf21','drvbuf31','drvdffck1','drvinv1','drvnand1','fdffck','finv','fnand','frefnn','frefpp','div','mdacbig']
    chosen         = np.array([1,1,1,1,1,1,1,1,1,0,0,1,0])
    vars_seqp11    = sxin[:,0:10]                 # variable for SEQ1
    cnst_seqp11    = seqp11.tf_scalex2(tf.concat([tf.ones([len_mc, 2], tf.float32),tf.ones([len_mc, 1], tf.float32)*(c_nbit-1)],axis=1), 1-chosen) 
    sx_seqp11      = tf.concat((vars_seqp11[:,0:9],cnst_seqp11[:,0:2],vars_seqp11[:,9:10],cnst_seqp11[:,2:3]),axis=1)
    x_seqp11       = seqp11.tf_rescalex(sx_seqp11)
    sy_seqp11      = seqp11.tf_reg_sigmoid(sx_seqp11) # sigmoid usage
    y_seqp11       = seqp11.tf_rescaley(sy_seqp11)
    
    
    #----------SEQ2's graph----------
    vars_seqp21  = sxin[:,10:12]
    sx_seqp21    = vars_seqp21
    x_seqp21     = seqp21.tf_rescalex(sx_seqp21)
    sy_seqp21    = seqp21.tf_reg_sigmoid(sx_seqp21) # sigmoid usage
    y_seqp21     = seqp21.tf_rescaley(sy_seqp21)
    
    
    #----------COMP's graph----------
    # 'fck1','fck2','fcompinv','finnn','fninv','fpinv','fpr1','fpr2','frdynand','fttt','fnor3','frdy','mrdy','fload','mload'
    chosen       = np.array([0,1,1,1,1,1,1,1,1,1,0,0,0,0,0])
    vars_compp1  = sxin[:,12:21]
    cnst_commp1  = compp1.tf_scalex2(tf.concat([x_seqp21[:,1:2],x_seqp21[:,0:1],2*x_seqp11[:,7:8]+x_seqp11[:,6:7],tf.ones([len_mc, 1], tf.float32)*c_nbit,
                                                x_seqp11[:,4:5],tf.ones([len_mc, 1], tf.float32)*c_nbit],axis=1), 1-chosen)
    sx_compp1    = tf.concat((cnst_commp1[:,0:1],vars_compp1,cnst_commp1[:,1:]),axis=1)
    x_compp1     = compp1.tf_rescalex(sx_compp1)
    sy_compp1    = compp1.tf_reg_sigmoid(sx_compp1) # changed to sigmoid
    y_compp1     = compp1.tf_rescaley(sy_compp1)
    
    
    #----------THDAC's graph----------
    #    'div','mdac',   'cs', 'fthn', 'fthp',   'cp'
    chosen    = np.array([0,0,1,1,1,0])
    vars_th1  = sxin[:,21:24]
    cnst_th1  = thdac1.tf_scalex2(tf.concat((x_seqp11[:,11:12],tf.ones([len_mc, 1], tf.float32)*c_nbit,y_compp1[:,5:6]),axis=1), 1-chosen)
    sx_th1    = tf.concat((cnst_th1[:,0:2],vars_th1,cnst_th1[:,2:]),axis=1)
    x_th1     = thdac1.tf_rescalex(sx_th1)
    sy_th1    = thdac1.tf_reg_sigmoid(sx_th1)    
    y_th1     = thdac1.tf_rescaley(sy_th1)
    
    
    #--------Other variables--------
#    n_dly = tf_quant_with_sigmoid(tf.math.abs(sxin[0,25]),10)
#    d_tr  = sxin[0,26]
    n_dly = tf_quant_with_sigmoid(tf.math.abs(sxin[:,24:25]),n_stairs)
    d_tr  = (sxin[:,25:26]+1)/2.0
    #==================================================================
    #***************  Define constraints and Cost(P)  *****************
    #==================================================================
    specs = []
    
    dacdelay = y_th1[:,3:4]+y_seqp11[:,2:3]+y_seqp11[:,3:4]
    digdelay =  y_compp1[:,2:3]+(y_seqp21[:,3:4]+y_seqp21[:,5:6]+n_dly*y_seqp21[:,4:5])*2.0
    ctot = x_th1[:,2:3]*2*c_lvls+y_compp1[:,5:6]
    bw1 = tf.nn.elu(y_th1[:,0:1])
    bw2 = tf.nn.elu(y_th1[:,1:2])
#    d_ts  = tf.nn.sigmoid( d_tr)*(1.0-fs*100e-12)+fs*100e-12
    d_ts  = d_tr*(1.0-fs*200e-12)+fs*200e-12

    ts = d_ts * 1/fs 
    
    power_seq1 =          2*(y_seqp11[:,0:1]+y_seqp11[:,1:2])*fs/62.5e6
    power_seq2 =     c_nbit*(y_seqp21[:,0:1]+n_dly*y_seqp21[:,1:2]+y_seqp21[:,2:3])*fs/200e6
    power_comp =     c_nbit*y_compp1[:,0:1]*fs/500e6
       
    #   0       1          2        3       4       5   6       7 
    #'power','readyp','delayr','delayf','kickn','cin','scin','irn'
    specs.append(digdelay-dacdelay-20e-12)                                                                                     # 0- Delay of DAC vs delay of the loop +
    specs.append(1/fs - ts - c_nbit*(y_compp1[:,1:2]+ 1*digdelay)-000e-12)                                                      # 1- loop delay less than fs  +
    specs.append(10*tf.math.log((4*y_th1[:,2:3])**2/(4*KT/ctot+y_compp1[:,7:8]**2))/tf.math.log(10.0))                          # 2- SNR more than 6*nbit + 11.76 
    specs.append(x_th1[:,2:3] - 6*y_compp1[:,6:7])                                                                              # 3- Comparator non-linear Caps + 
    specs.append(ts*bw1*6.28-tf.math.log(2.0)*c_nbit)                                                                           # 4- Track and Hold Law +
    specs.append(ts*bw2*6.28-tf.math.log(2.0)*c_nbit)                                                                           # 5- Track and Hold Law +
    specs.append(4-ts*bw1*6.28+tf.math.log(2.0)*c_nbit)                                                                         # 6- Track and Hold Law +
    specs.append(4-ts*bw2*6.28+tf.math.log(2.0)*c_nbit)                                                                         # 7- Track and Hold Law +
    specs.append(y_compp1[:,3:4])                                                                                                 # 8- vomin to be less
    specs.append(power_seq1+power_seq2+power_comp)                                                                              # 9- Power consumption - 
    
    
    constraints = []    
    constraints.append(tf.nn.elu(          -specs[0]/digdelay*weights[0]))
    constraints.append(tf.nn.elu(-specs[1]/thdac1.scYscale[3]*weights[1]))
    constraints.append(tf.nn.elu((nbit+10/6      -specs[2]/6)*weights[2]))
    constraints.append(tf.nn.elu(-specs[3]/compp1.scYscale[5]*weights[3]))
    constraints.append(tf.nn.elu(                   -specs[4]*weights[4]))
    constraints.append(tf.nn.elu(                   -specs[5]*weights[5]))
    constraints.append(tf.nn.elu(                   -specs[6]*weights[6]))
    constraints.append(tf.nn.elu(                   -specs[7]*weights[7]))
    constraints.append(tf.nn.elu(             -(specs[8]-0.6)*weights[8]))
    constraints.append(          specs[-1]/compp1.scYscale[0]*weights[9] )    
    
    hardcost_0=tf.reduce_sum(constraints[0:3],axis=0)+tf.reduce_sum(constraints[4:-1],axis=0)
    softcost_0=tf.reduce_sum(constraints,axis=0)
    
    hardcost = tf.reduce_sum(hardcost_0)
    softcost = tf.reduce_sum(softcost_0)
    return softcost,hardcost,specs,[x_seqp11,x_seqp21,x_compp1,x_th1,n_dly,d_tr],[y_seqp11,y_seqp21,y_compp1,y_th1,ts],[dacdelay, digdelay, ctot, d_ts,bw1,bw2],constraints,softcost_0,hardcost_0
   


#==================================================================
#*****************  Building the graph  ***************************
#==================================================================
def graph_spice3(sxin,seqp11,seqp21,compp1,thdac1,seq1spice,seq2spice,comppspice,thdacspice,u,s,put_on_csv=False):
    
    nbit = u[0]
    fs   = u[1]
    n_stairs =s[0]
    weights = s[1]
    # Constants:
    c_nbit = nbit
    c_lvls = 2**(nbit-1)
    #----------SEQ1's graph----------    
    chosen=np.array([1,1,1,1,1,1,1,1,1,0,0,1,0])
    vars_seqp11  = sxin[:,0:10]
    cnst_seqp11 = seqp11.np_scalex2(np.stack([1.0,1.0,c_nbit-1],axis=0), 1-chosen)
    
    
    sx_seqp11 = np.reshape(np.concatenate((vars_seqp11[0][0:9],cnst_seqp11[0:2],vars_seqp11[0][9:10],cnst_seqp11[2:3]),axis=0),[1,13])
    
    x_seqp11  = seqp11.np_rescalex(sx_seqp11)
    x_seqp11, d_seqp11 = seq1spice.wholerun_std(x_seqp11[0],put_on_csv)
    y_seqp11 = np.array([d_seqp11])
    
    #----------SEQ2's graph----------
    vars_seqp21  = sxin[:,10:12]
    sx_seqp21 = vars_seqp21
    x_seqp21  = seqp21.np_rescalex(sx_seqp21)
    x_seqp21 , d_seqp21 = seq2spice.wholerun_std(x_seqp21[0],put_on_csv)
    y_seqp21 = np.array([d_seqp21])
    
    #----------COMP's graph----------
    chosen=np.array([0,1,1,1,1,1,1,1,1,1,0,0,0,0,0])
    vars_compp1  = sxin[:,12:21]
    cnst_commp1 = compp1.np_scalex2(np.stack([x_seqp21[1],x_seqp21[0],2*x_seqp11[7]+x_seqp11[6],c_nbit,x_seqp11[4],c_nbit],axis=0), 1-chosen)
    sx_compp1   = np.reshape(np.concatenate((cnst_commp1[0:1],vars_compp1[0],cnst_commp1[1:]),axis=0),[1,15])
    x_compp1    = compp1.np_rescalex(sx_compp1)
    # print(x_compp1)
    # x_compp1, d_compp1 = comppspice.wholerun_std(np.array(list(x_compp1[0])+[1     ,    1      ,100e-6, 0.4 , 1.0]),put_on_csv)
    x_compp1, d_compp1 = comppspice.wholerun_std(np.array(list(x_compp1[0][0:14])),put_on_csv)
    # ['fck1','fck2','fcompinv','finnn','fninv','fpinv','fpr1','fpr2','frdynand','fttt','fnor3','frdy','mrdy','fload']
    y_compp1    = np.array([np.array(d_compp1)    ])
    
    #----------THDAC's graph----------
    chosen=np.array([0,0,1,1,1,0])
    vars_th1=sxin[:,21:24]
    cnst_th1=thdac1.np_scalex2(np.stack((x_seqp11[11],c_nbit,y_compp1[0,5]),axis=0), 1-chosen)
    sx_th1=np.reshape(np.concatenate((cnst_th1[0:2],vars_th1[0][0:],cnst_th1[2:]),axis=0),[1,6])
    x_th1=thdac1.np_rescalex(sx_th1)
    x_th1, d_th1 = thdacspice.wholerun_std(np.array(list(x_th1[0][:-1])+[1,1,x_th1[0][-1]]),put_on_csv)
    y_th1 = np.array([d_th1])
    
    #--------Other variables--------

    
    n_dly = np.round(np.abs(sxin[0,24])*n_stairs+1.0)
    d_tr  = (sxin[0,25]+1)/2.0    
    #==================================================================
    #***************  Define constraints and Cost(P)  *****************
    #==================================================================
   
    
    dacdelay = y_th1[0,3]+y_seqp11[0,2]+y_seqp11[0,3]
    digdelay =  y_compp1[0,2]+(y_seqp21[0,3]+y_seqp21[0,5]+n_dly*y_seqp21[0,4])*2.0
    ctot = x_th1[2]*2*c_lvls+y_compp1[0,5]
    bw1 = y_th1[0,0]
    bw2 = y_th1[0,1]
    d_ts  = d_tr*(1.0-fs*200e-12)+fs*200e-12
    ts = d_ts * 1/fs 

    power_seq1 =     c_nbit*(y_seqp11[0,0]+y_seqp11[0,1])*fs/62.5e6
    power_seq2 =     c_nbit*(y_seqp21[0,0]+n_dly*y_seqp21[0,1]+y_seqp21[0,2])*fs/200e6
    power_comp =     c_nbit*y_compp1[0,0]
    
    specs = []
    specs.append(digdelay-dacdelay-20e-12)                                                                                     # 0- Delay of DAC vs delay of the loop +
    specs.append(1/fs - ts - c_nbit*(y_compp1[0,1]+ 1*digdelay)-000e-12)                                                        # 1- loop delay less than fs  +
    specs.append(10*np.log((4*y_th1[0,2])**2/(4*KT/ctot+y_compp1[0,7]**2))/np.log(10.0))                                        # 2- SNR more than 6*nbit + 11.76 
    specs.append(x_th1[2] - 6*y_compp1[0,6])                                                                                    # 3- Comparator non-linear Caps + 
    specs.append(ts*bw1*6.28-np.log(2.0)*c_nbit)                                                                                # 4- Track and Hold Law +
    specs.append(ts*bw2*6.28-np.log(2.0)*c_nbit)                                                                                # 5- Track and Hold Law +
    specs.append(4-ts*bw1*6.28+np.log(2.0)*c_nbit)                                                                              # 6- Track and Hold Law +
    specs.append(4-ts*bw2*6.28+np.log(2.0)*c_nbit)                                                                              # 7- Track and Hold Law +
    specs.append(y_compp1[0,3])                                                                                                 # 8- zero or one for comparator
    specs.append(power_seq1+power_seq2+power_comp)                                                                              # 9- Power consumption - 
     
    
    
    constraints = []    
    constraints.append(np_elu(          -specs[0]/digdelay*weights[0]))
    constraints.append(np_elu(-specs[1]/thdac1.scYscale[3]*weights[1]))
    constraints.append(np_elu((nbit+10/6      -specs[2]/6)*weights[2]))
    constraints.append(np_elu(-specs[3]/compp1.scYscale[5]*weights[3]))
    constraints.append(np_elu(                   -specs[4]*weights[4]))
    constraints.append(np_elu(                   -specs[5]*weights[5]))
    constraints.append(np_elu(                   -specs[6]*weights[6]))
    constraints.append(np_elu(                   -specs[7]*weights[7]))
    constraints.append(np_elu(             -(specs[8]-0.6)*weights[8]))
    constraints.append(       specs[-1]/compp1.scYscale[0]*weights[9] )
    
    
    hardcost=sum(constraints[0:3])+sum(constraints[4:-1])
    softcost=sum(constraints)
    
        
    return softcost,hardcost,specs,[x_seqp11,x_seqp21,x_compp1,x_th1,n_dly,d_tr],[y_seqp11,y_seqp21,y_compp1,y_th1,ts],[dacdelay, digdelay, ctot, d_ts,bw1,bw2],constraints

def sxin2param(sxin,seqp11,seqp21,compp1,thdac1,seq1spice,seq2spice,comppspice,thdacspice,u,s,put_on_csv=False):
    
    nbit = u[0]
    fs   = u[1]
    n_stairs =s[0]
    weights = s[1]
    # Constants:
    c_nbit = nbit
    #----------SEQ1's graph----------    
    chosen=np.array([1,1,1,1,1,1,1,1,1,0,0,1,0])
    vars_seqp11  = sxin[:,0:10]
    cnst_seqp11 = seqp11.np_scalex2(np.stack([1.0,1.0,c_nbit-1],axis=0), 1-chosen)
    
    
    sx_seqp11 = np.reshape(np.concatenate((vars_seqp11[0][0:9],cnst_seqp11[0:2],vars_seqp11[0][9:10],cnst_seqp11[2:3]),axis=0),[1,13])
    
    x_seqp11  = seqp11.np_rescalex(sx_seqp11)
    x_seqp11 = seq1spice.param_std(x_seqp11[0])
    
    
    #----------SEQ2's graph----------
    vars_seqp21  = sxin[:,10:12]
    sx_seqp21 = vars_seqp21
    x_seqp21  = seqp21.np_rescalex(sx_seqp21)
    x_seqp21  = seq2spice.param_std(x_seqp21[0])

    
    #----------COMP's graph----------
    chosen=np.array([0,1,1,1,1,1,1,1,1,1,0,0,0,0,0])
    vars_compp1  = sxin[:,12:21]
    cnst_commp1 = compp1.np_scalex2(np.stack([x_seqp21[1],x_seqp21[0],2*x_seqp11[7]+x_seqp11[6],c_nbit,x_seqp11[4],c_nbit],axis=0), 1-chosen)
    sx_compp1   = np.reshape(np.concatenate((cnst_commp1[0:1],vars_compp1[0],cnst_commp1[1:]),axis=0),[1,15])
    x_compp1    = compp1.np_rescalex(sx_compp1)
    x_compp1 = comppspice.param_std(np.array(list(x_compp1[0][0:14])))

    
    #----------THDAC's graph----------
    chosen=np.array([0,0,1,1,1,0])
    vars_th1=sxin[:,21:24]
    cnst_th1=thdac1.np_scalex2(np.stack((x_seqp11[11],c_nbit,0),axis=0), 1-chosen)
    sx_th1=np.reshape(np.concatenate((cnst_th1[0:2],vars_th1[0][0:],cnst_th1[2:]),axis=0),[1,6])
    x_th1=thdac1.np_rescalex(sx_th1)
    x_th1 = thdacspice.param_std(np.array(list(x_th1[0][:-1])+[1,1,x_th1[0][-1]]))
    
    
    #--------Other variables--------
    n_dly = np.round(np.abs(sxin[0,24])*n_stairs+1.0)
    d_tr  = (sxin[0,25]+1)/2.0    

    
        
    return [x_seqp11,x_seqp21,x_compp1,x_th1,n_dly,d_tr]