# The test on circuit after AMPSE:

import sys

sys.path.insert(0,'/GlobalLibrary')

import numpy as np
from pickle import load
from spectreIOlib import TestSpice, randomchoice
from Netlist_Database import VCOSpice,INBUF2Spice
from tensorflow_circuit import action2sxin,rw3,vector_constraints
from AMPSE_Graphs2 import *


file_ampse = 'regsearch_results1_'+str(nbit)+str(bw/1e6)+'.p'


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



def test_spice(lst_params_chosen,vco1,inbuf1,vcospice1,inbufspice1):
    lst_metrics_spice=[]
    lst_specs_spice=[]
    lst_value_spice=[]
    lst_mids_spice=[]
    lst_const_spice=[]
    
    for i in range(len(lst_params_chosen)):
        
        sp_sxin = param_to_sxin(lst_params_chosen[i],vco1,inbuf1)
        sp_value,_,sp_specs,sp_params,sp_metrics,sp_mids,sp_const= graph_spice2(sp_sxin,vco1,inbuf1,vcospice1,inbufspice1)
        
        lst_metrics_spice.append(sp_metrics)
        lst_specs_spice.append(sp_specs)
        lst_value_spice.append(sp_value)        
        lst_mids_spice.append(sp_mids)
        lst_const_spice.append(sp_const)
        
    return lst_metrics_spice,lst_specs_spice,lst_value_spice,lst_mids_spice,lst_const_spice



if __name__ == '__main__':
    
    
    vcospice1 = VCOSpice()
    inbufspice1 = INBUF2Spice()
    vco1 = VCO(tech=65)
    inbuf1 = INBUF2(tech=65)
    tedad = 50
    what_to_do = 1
    
    
    #==================================================================
    #*******************  Initialization  *****************************
    #==================================================================
    
    if what_to_do == 0:
        tstart = time.time()
        dict_loads = load(open( file_ampse, "rb" ) )
        lst_params_chosen, lst_metrics_chosen, lst_specs_chosen, lst_values_chosen, lst_chosen = choose_best(dict_loads,tedad)
        sp_sxin = param_to_sxin(lst_params_chosen[0],vco1,inbuf1)
        sp_value,_,sp_specs,sp_params,sp_metrics,sp_mids,sp_const= graph_spice2(sp_sxin,vco1,inbuf1,vcospice1,inbufspice1)
        tend = time.time()
        print(tend-tstart)
    elif what_to_do ==1:
        tstart = time.time()
        dict_loads = load(open( file_ampse, "rb" ) )
        lst_params_chosen, lst_metrics_chosen, lst_specs_chosen, lst_values_chosen, lst_chosen = choose_best(dict_loads,tedad)
        lst_metrics_spice,lst_specs_spice,lst_value_spice,lst_mids_spice,lst_const_spice = test_spice(lst_params_chosen,vco1,inbuf1,vcospice1,inbufspice1)    
        np_specs_spice = np.array(lst_specs_spice) 
        np_specs_chosen = np.array(lst_specs_chosen) 
#        np.savetxt('np_specs_spice.csv', np_specs_spice, delimiter=',')
#        np.savetxt('np_specs_chosen.csv', np_specs_chosen, delimiter=',')
        tend = time.time()
        print(tend-tstart)
    elif what_to_do ==2:
        lst_rw_specs=[]
        lst_rw_action=[]
        lst_rw_value=[]
        dd = np.array([1,1,1,1,1,1,1,1,1,1,1,1])
        prev_sxin = param_to_sxin(lst_params_chosen[0],vco1,inbuf1)
        sp_value,_,sp_specs,sp_params,sp_metrics,sp_mids,prev_const= graph_spice2(prev_sxin,vco1,inbuf1,vcospice1,inbufspice1)
        n_action = len(prev_sxin[0])
        action=12
        bad_action=24
        goodjob=0
        new_value = sp_value
        ssin = step2sxin(vco1,inbuf1)
        lst_rw_specs.append(sp_specs)
        lst_rw_action.append(action)
        lst_rw_value.append(new_value)
        for i in range(tedad):
            action = rw2(n_action,bad_action)
            new_sxin,_ = action2sxin(action,prev_sxin,ssin)
            
            
            sp_value,_,sp_specs,sp_params,sp_metrics,sp_mids,new_const= graph_spice2(new_sxin,vco1,inbuf1,vcospice1,inbufspice1)
            
            reward = vector_constraints(prev_const,new_const,dd)
            
            if reward>0 :
                prev_sxin = new_sxin
                prev_const = new_const
                bad_action=24
                goodjob+=1
                new_value  = sp_value
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
            savemat('regsearch_constraints.mat',{'rw_spec':np_rw_specs,'rw_action':np_rw_action,'rw_value':np_rw_value})
    elif what_to_do ==3:
        lst_gr_specs=[]
        lst_gr_value=[]
        prev_sxin = param_to_sxin(lst_params_chosen[3],vco1,inbuf1)
        sp_value,_,sp_specs,sp_params,sp_metrics,sp_mids,prev_const= graph_spice2(prev_sxin,vco1,inbuf1,vcospice1,inbufspice1)
        prev_value = sp_value
        ssin = step2sxin(vco1,inbuf1)
        num_step = len(ssin[0])
        lr = 0.0001
        dy = np.zeros_like(ssin)
        lst_gr_specs.append(sp_specs)
        lst_gr_value.append(prev_value)
        tstart = time.time()
        for j in range(1):
            for i in range(num_step):
                dx = ssin[0,i]
                new_sxin,_ = action2sxin(i,prev_sxin,ssin)
                new_value,_,sp_specs,sp_params,sp_metrics,sp_mids,new_const= graph_spice2(new_sxin,vco1,inbuf1,vcospice1,inbufspice1)
                
                dy[0,i] = (new_value - prev_value)/dx
                
            pro_sxin = prev_sxin - lr*dy
            new_value,_,sp_specs,sp_params,sp_metrics,sp_mids,new_const= graph_spice2(pro_sxin,vco1,inbuf1,vcospice1,inbufspice1)
            print(new_value,j,time.time()-tstart)
            prev_value = new_value
            lst_gr_specs.append(sp_specs)
            lst_gr_value.append(prev_value)
            
            
        np_gr_specs  = np.array(lst_gr_specs)
        np_gr_value  = np.array(lst_gr_value)
        
        from scipy.io import savemat
        savemat('regsearch_gradients.mat',{'gr_spec':np_gr_specs,'rw_value':np_gr_value})
        
        
        
        
    
    
            