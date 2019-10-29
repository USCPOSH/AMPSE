# Mohsen Hassanpourghadi

# Design VCO-TH-DRV with TF.

#==================================================================
#*****************  Loading the libraries  ************************
#==================================================================

import sys
#sys.path.insert(0,'/shares/MLLibs/GlobalLibrary')
import os
home_address  = os.getcwd()
sys.path.insert(0, home_address+'/MLLibs/GlobalLibrary')

import numpy as np
from AMPSE_Graphs import Folded_Cascode,ClassAB
from pickle import load
from Netlist_Database import Folded_Cascode_spice, ClassAB_spice
from AMPSE_Graphs import cload,graph_tf,graph_spice,param_to_sxin
import matplotlib.pyplot as plt



file_ampse = 'regsearch_results1_'+str(cload)+'.p'

tedad = 5


def choose_best(dict_loads,tedad):
    
    lst_params = dict_loads['lst_params']
    lst_metrics= dict_loads['lst_metrics']
    lst_specs  = dict_loads['lst_specs']
    lst_values = dict_loads['lst_value']
    chosen_np = np.argsort(lst_values)[:tedad][::+1]
    lst_params_chosen = [lst_params[i] for i in chosen_np]
    lst_metrics_chosen= [lst_metrics[i] for i in chosen_np]
    lst_specs_chosen= [lst_specs[i] for i in chosen_np]
    lst_values_chosen= [lst_values[i] for i in chosen_np]
    
    return lst_params_chosen, lst_metrics_chosen, lst_specs_chosen, lst_values_chosen,chosen_np

def choose_one(dict_loads,i):
    
    lst_params = dict_loads['lst_params']
    lst_metrics= dict_loads['lst_metrics']
    lst_specs  = dict_loads['lst_specs']
    lst_values = dict_loads['lst_value']
    lst_params_chosen = lst_params[i]
    lst_metrics_chosen= lst_metrics[i]
    lst_specs_chosen= lst_specs[i]
    lst_values_chosen= lst_values[i]
    
    return lst_params_chosen, lst_metrics_chosen, lst_specs_chosen, lst_values_chosen, 1

def test_spice(lst_params_chosen,folded_cascode,classab,folded_cascode_spice,classab_spice):
    lst_metrics_spice=[]
    lst_specs_spice=[]
    lst_value_spice=[]
    lst_mids_spice=[]
    lst_const_spice=[]
    
    for i in range(len(lst_params_chosen)):
        
        sp_sxin = param_to_sxin(lst_params_chosen[i],folded_cascode,classab)
        sp_value,_,sp_specs,sp_params,sp_metrics,sp_mids,sp_const= graph_spice(sp_sxin,folded_cascode,classab,folded_cascode_spice,classab_spice)
        
        lst_metrics_spice.append(sp_metrics)
        lst_specs_spice.append(sp_specs)
        lst_value_spice.append(sp_value)        
        lst_mids_spice.append(sp_mids)
        lst_const_spice.append(sp_const)
        
    return lst_metrics_spice,lst_specs_spice,lst_value_spice,lst_mids_spice,lst_const_spice


if __name__ == '__main__':
    
    #==================================================================
    #*******************  Initialization  *****************************
    #==================================================================
    folded_cascode = Folded_Cascode()
    classab = ClassAB()

    folded_cascode_spice = Folded_Cascode_spice()
    classab_spice = ClassAB_spice()
    
    
    
    #==================================================================
    #*******************  Initialization  *****************************
    #==================================================================
    dict_loads = load(open( file_ampse, "rb" ) )
    lst_params_chosen, lst_metrics_chosen, lst_specs_chosen, lst_values_chosen, chosen_np = choose_best(dict_loads,tedad)

    
    lst_metrics_spice,lst_specs_spice,lst_value_spice,lst_mids_spice,lst_const_spice = test_spice(lst_params_chosen,folded_cascode,classab,folded_cascode_spice,classab_spice)
    np_specs_spice = np.array(lst_specs_spice) 
    np_specs_chosen = np.array(lst_specs_chosen) 
    np.savetxt('np_specs_spice.csv', np_specs_spice, delimiter=',')
    np.savetxt('np_specs_chosen.csv', np_specs_chosen, delimiter=',')
    
    """
    lst_rw_specs=[]
    lst_rw_action=[]
    lst_rw_value=[]
    dd = np.array([1,1,1,1,1,1,1])
    prev_sxin = param_to_sxin(lst_params_chosen[5],seqp11,seqp21,compp1,thdac1)
    new_value,_,sp_specs,sp_params,sp_metrics,sp_mids,prev_const= graph_spice2(prev_sxin,seqp11,seqp21,compp1,thdac1,seqp1spice1,seqp2spice1,comppspice1,dacthspice1)
    n_action = len(prev_sxin[0])
    action=2*n_action
    bad_action=2*n_action
    goodjob=0
    ssin = step2sxin(seqp11,seqp21,compp1,thdac1)
    lst_rw_specs.append(sp_specs)
    lst_rw_action.append(action)
    lst_rw_value.append(new_value)
    for i in range(tedad):
        action = rw2(n_action,bad_action)
        new_sxin,_ = action2sxin(action,prev_sxin,ssin)
        
        
        sp_value,_,sp_specs,sp_params,sp_metrics,sp_mids,new_const= graph_spice2(new_sxin,seqp11,seqp21,compp1,thdac1,seqp1spice1,seqp2spice1,comppspice1,dacthspice1)
        
        reward = vector_constraints(prev_const,new_const,dd)
        
        if reward>0 :
            prev_sxin = new_sxin
            prev_const = new_const
            bad_action=2*n_action
            new_value = sp_value
            goodjob+=1
        else:
            bad_action=action
        
        lst_rw_specs.append(sp_specs)
        lst_rw_action.append(action)
        lst_rw_value.append(new_value)
        print(reward)
    
    np_rw_specs = np.array(lst_rw_specs)
    np_rw_action= np.array(lst_rw_action)
    np_rw_value = np.array(lst_rw_value)
        
        
    from scipy.io import savemat
    savemat('regsearch_rw1.mat',{'rw_spec':np_rw_specs,'rw_action':np_rw_action,'rw_value':np_rw_value})
    """
    """
    import time
    lst_gr_specs=[]
    lst_gr_value=[]
    prev_sxin = param_to_sxin(lst_params_chosen[0],folded_cascode,classab)
    sp_value,_,sp_specs,sp_params,sp_metrics,sp_mids,prev_const= graph_spice(prev_sxin,folded_cascode,classab,folded_cascode_spice,classab_spice)
    prev_value = sp_value
    ssin = step2sxin(folded_cascode,classab)
    num_step = len(ssin[0])
    lr = 0.002
    dy = np.zeros_like(ssin)
    lst_gr_specs.append(sp_specs)
    lst_gr_value.append(prev_value)
    tstart = time.time()
    for j in range(tedad):
        for i in range(num_step):
            dx = ssin[0,i]
            new_sxin,_ = action2sxin(i,prev_sxin,ssin)
            new_value,_,sp_specs,sp_params,sp_metrics,sp_mids,prev_const= graph_spice(new_sxin,folded_cascode,classab,folded_cascode_spice,classab_spice)

            dy[0,i] = (new_value - prev_value)/dx
            
        pro_sxin = prev_sxin - lr*dy
        new_value,_,sp_specs,sp_params,sp_metrics,sp_mids,prev_const= graph_spice(pro_sxin,folded_cascode,classab,folded_cascode_spice,classab_spice)
        print(new_value,j,time.time()-tstart)
        prev_value = new_value
        prev_sxin  = pro_sxin
        lst_gr_specs.append(sp_specs)
        lst_gr_value.append(prev_value)
        
        
    np_gr_specs  = np.array(lst_gr_specs)
    np_gr_value  = np.array(lst_gr_value)
    
#    from scipy.io import savemat
#    savemat('regsearch_gradients.mat',{'gr_spec':np_gr_specs,'rw_value':np_gr_value})
   """     